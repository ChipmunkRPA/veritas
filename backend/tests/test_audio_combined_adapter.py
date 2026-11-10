"""Engine/report contracts for an optional combined speech check; no model calls."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.audio import combined
from app.detectors.audio import combined_adapter as adapter


def context(pcm):
    mapping = {
        "sample_rate": 16000,
        "samples": len(pcm),
        "timeline_time_base": "1/16000",
        "limit_samples": 9600000,
        "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 320}],
    }
    return SimpleNamespace(
        outputs=SimpleNamespace(audio=SimpleNamespace(status="analysed", pcm=pcm, source_mapping=mapping))
    )


@pytest.mark.parametrize("samples", [0, 40000, 63999, 64001, 128000])
def test_adapter_does_not_broaden_the_combination_scope(samples):
    assert not adapter.CombinedSpeechDetector().applicable(context(np.zeros(samples, np.float32))).ok


def test_adapter_declares_every_real_component_for_engine_budgeting():
    detector = adapter.CombinedSpeechDetector()
    declared = set(detector.spec.requirements.model_ids)
    assert declared == {adapter.BASELINE_ID, adapter.ACTIVITY_ID, adapter.ENCODER_ID, adapter.CALIBRATION_ID}
    assert {use.model_id for use in detector.activation_plan(None)} == declared
    assert detector.spec.requirements.devices == ("cpu",)


def test_report_reference_has_no_learned_parameters_or_provider_identity():
    descriptor = json.loads(adapter.calibration_descriptor())
    assert descriptor["sha256"] == adapter.BUNDLE_SHA256
    assert descriptor["reference_prior_ai"] == 0.5
    text = json.dumps(descriptor).lower()
    assert all(
        name not in text for name in ("wavlm", "pella", "microsoft", "coefficients", "center", "source_ids")
    )


def test_actual_checked_window_is_linked_to_result_exhibit_and_source_pcm(monkeypatch):
    pcm = np.linspace(-0.1, 0.1, 64000, dtype=np.float32)
    ctx = context(pcm)
    calls = []
    ctx.lease = SimpleNamespace(get=lambda name: SimpleNamespace(obj=name, revision="a" * 40))
    ctx.deadline = object()

    def series(data, **kwargs):
        calls.append((data, kwargs))
        return "exhibit-speech"

    ctx.exhibits = SimpleNamespace(series=series)
    screen = {
        "status": "ok",
        "probability_ai": 0.7,
        "decision": "ai",
        "reason": "Leaning towards AI-generated speech.",
        "limitation": combined.NOTE,
        "calibration_sha256": adapter.BUNDLE_SHA256,
        "checked_windows": [{"start_sample": 0, "end_sample": 64000, "start_s": 0.0, "end_s": 4.0}],
    }
    seen = []

    class Runtime:
        def __init__(self, calibration, encoder):
            assert calibration == adapter.CALIBRATION_ID and encoder == adapter.ENCODER_ID

        def analyze(self, source, baseline, activity, deadline):
            assert baseline == adapter.BASELINE_ID and activity == adapter.ACTIVITY_ID
            seen.append(source is pcm and deadline is ctx.deadline)
            return screen

    monkeypatch.setattr(adapter.combined, "CombinedSpeechRuntime", Runtime)
    result = adapter.CombinedSpeechDetector().run(ctx)
    assert seen == [True]
    assert result.aggregate.raw == 0.7 and result.unit_scores[0].raw == 0.7
    assert result.unit_scores[0].quality == {
        "start_s": 0.0,
        "end_s": 4.0,
        "start_sample": 0.0,
        "end_sample": 64000.0,
    }
    assert result.exhibit_ids == ["exhibit-speech"]
    assert calls[0][0][0]["points"] == [(2.0, 0.7)] and calls[0][1]["media_ref"] == "audio"
    assert result.details["audio_screening"]["probability_ai"] == screen["probability_ai"]
    assert result.details["audio_screening"]["checked_source_ranges"] == [
        {
            "start_sample": 0,
            "end_sample": 64000,
            "source_start_pts": 320,
            "source_end_pts": 64320,
            "time_base": "1/16000",
        }
    ]
    assert (
        result.details["audio_input"]["pcm_sha256"] == hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
    )


def test_skipped_runtime_result_creates_no_probability_or_exhibit(monkeypatch):
    ctx = context(np.zeros(64000, np.float32))
    ctx.lease = SimpleNamespace(get=lambda name: SimpleNamespace(obj=None))
    ctx.deadline = object()
    monkeypatch.setattr(
        adapter.combined,
        "CombinedSpeechRuntime",
        lambda *args: SimpleNamespace(analyze=lambda *args: combined.unassessed("No detected speech.")),
    )
    result = adapter.CombinedSpeechDetector().run(ctx)
    assert result.status == "not_applicable"
    assert result.aggregate is None and result.unit_scores == [] and result.exhibit_ids == []


@pytest.mark.parametrize("change", ["missing", "gap", "hash", "truncated"])
def test_missing_discontinuous_or_truncated_source_mapping_is_refused(change):
    ctx = context(np.zeros(64000, np.float32))
    mapping = ctx.outputs.audio.source_mapping
    if change == "missing":
        ctx.outputs.audio.source_mapping = None
    elif change == "gap":
        mapping["timeline"] = [
            {"start_sample": 0, "end_sample": 32000, "source_pts": 0},
            {"start_sample": 32000, "end_sample": 64000, "source_pts": 33600},
        ]
    elif change == "hash":
        mapping["pcm_sha256"] = "a" * 64
    else:
        mapping["limit_samples"] = 64000
    assert not adapter.CombinedSpeechDetector().applicable(ctx).ok
    result = adapter.CombinedSpeechDetector().run(ctx)
    assert result.status == "not_applicable" and result.aggregate is None
