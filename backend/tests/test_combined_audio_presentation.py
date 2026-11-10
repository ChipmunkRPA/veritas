"""A new speech reference must keep its own score, scope and rollback semantics."""

import copy
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.detectors.audio import combined_adapter
from app.detectors.audio.reference import COMBINED_SHA256, combined_tendency
from app.engine.customer_assessment import customer_assessment
from app.engine.finding_assessment import finding_assessment
from app.engine.fusion.registry import fusion_status
from app.engine.registry import get_registry


def example(p=0.55):
    direction = "ai" if p > 0.5 else "original" if p < 0.5 else "inconclusive"
    screen = {
        "status": "ok",
        "calibration_sha256": COMBINED_SHA256,
        "scope_supported": True,
        "probability_ai": p,
        "decision": direction,
        "checked_windows": [{"start_sample": 0, "end_sample": 64000, "start_s": 0.0, "end_s": 4.0}],
        "checked_source_ranges": [
            {
                "start_sample": 0,
                "end_sample": 64000,
                "source_start_pts": 320,
                "source_end_pts": 64320,
                "time_base": "1/16000",
            }
        ],
    }
    signal = {
        "id": "speech:global",
        "detector_id": combined_adapter.DETECTOR_ID,
        "status": "ok",
        "scope": "global",
        "modality": "audio",
        "family": "voice",
        "category": "model",
        "calibration_ref": COMBINED_SHA256,
        "calibrated_probability": p,
        "effective_weight": 1.0,
        "details": {"audio_screening": screen},
    }
    result = {
        "kind": "audio",
        "status": "done",
        "signals": [signal],
        "decision": {
            "fuser_sha256": COMBINED_SHA256,
            "probability_ai_calibrated": p,
            "code": {"ai": "ai_supported", "original": "no_ai_indicators"}.get(direction, "inconclusive"),
            "fusion_mode": "artefact",
        },
    }
    return result, screen


@pytest.mark.parametrize(
    "p,direction,confidence",
    [(0.55, "ai", 55.0), (0.45, "original", 55.0), (0.85, "ai", 85.0), (0.5, "inconclusive", None)],
)
def test_combination_uses_its_own_frozen_tendency_and_reference_estimate(p, direction, confidence):
    result, _ = example(p)
    assert COMBINED_SHA256 == combined_adapter.BUNDLE_SHA256
    assessment = customer_assessment(result)
    assert assessment["direction"] == direction and assessment["confidence_pct"] == confidence
    finding = {
        "id": "F1",
        "family": "A1_voice",
        "scope": None,
        "direction": "for" if p > 0.5 else "against",
        "strength": "weak",
        "evidence_ids": ["speech:global"],
    }
    observed = finding_assessment(finding, result)
    assert observed["direction"] == direction and observed["confidence_pct"] == confidence
    assert assessment["interval_pct"] is None


@pytest.mark.parametrize(
    "problem", ["scope", "missing_source", "gap", "clock", "status", "reference", "decision"]
)
def test_invalid_saved_scope_or_skipped_check_never_becomes_original(problem):
    result, screen = example(0.1)
    if problem == "scope":
        screen["scope_supported"] = False
    elif problem == "missing_source":
        screen.pop("checked_source_ranges")
    elif problem == "gap":
        screen["checked_source_ranges"][0]["source_end_pts"] += 1600
    elif problem == "clock":
        screen["checked_windows"][0]["end_sample"] = 63999
    elif problem == "status":
        screen["status"] = "not_applicable"
    elif problem == "reference":
        screen["calibration_sha256"] = "a" * 64
    else:
        screen["decision"] = "inconclusive"
    assert combined_tendency(screen)[1] == "inconclusive"
    assessment = customer_assessment(result)
    assert assessment["direction"] == "inconclusive" and assessment["confidence_pct"] is None


def test_video_speech_estimate_remains_separate_from_visual_verdict():
    result, _ = example(0.55)
    before = copy.deepcopy(result)
    result["kind"] = "video"
    result.pop("decision")
    assessment = customer_assessment(result)
    assert assessment["direction"] == "unavailable" and assessment["confidence_pct"] is None
    assert (
        assessment["soundtrack"]["direction"] == "ai" and assessment["soundtrack"]["confidence_pct"] == 55.0
    )
    assert result["signals"] == before["signals"]


def test_combined_option_is_explicit_and_legacy_rollback_disables_it(monkeypatch):
    registry = get_registry()
    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(settings, "audio_combined_enabled", False)
    assert not registry.decision(combined_adapter.DETECTOR_ID).enabled
    monkeypatch.setattr(settings, "audio_combined_enabled", True)
    assert registry.decision(combined_adapter.DETECTOR_ID).enabled
    assert registry.decision("aud.pella_v2").enabled  # explicit reference/rollback, never double-counted
    assert registry.instance(combined_adapter.DETECTOR_ID).spec.id == combined_adapter.DETECTOR_ID
    assert fusion_status(registry, "full")["audio"] == "artefact"
    monkeypatch.setattr(settings, "audio_detector_mode", "legacy")
    assert not registry.decision(combined_adapter.DETECTOR_ID).enabled
    assert registry.decision("aud.gustking_xlsr").enabled


@pytest.mark.parametrize("case", ["valid", "unsupported", "missing"])
def test_fuser_selects_one_reference_and_never_exports_private_parameters(monkeypatch, case):
    from app.engine.stages import audio_screening

    result, screen = example(0.55)
    primary = result["signals"][0]
    old = {"id": "rollback:global", "detector_id": "aud.pella_v2", "family": "voice", "details": {}}
    signals = [old] if case == "missing" else [old, primary]
    if case == "unsupported":
        screen.update(scope_supported=False, decision="original", probability_ai=0.01)
    monkeypatch.setattr(audio_screening.video, "synthesis_signals", lambda *args: signals)
    monkeypatch.setattr(audio_screening.video, "attach_temporal", lambda *args: None)
    cfg = combined_adapter.CalibrationReference()
    monkeypatch.setattr(cfg, "load", lambda: None)
    stored = []

    def register(db, store, payload, **metadata):
        stored.append(payload)
        return SimpleNamespace(
            id="descriptor", sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), **metadata
        )

    monkeypatch.setattr(audio_screening, "register_artefact", register)
    db = SimpleNamespace(get=lambda *args: None, add=lambda row: None, commit=lambda: None)
    ctx = SimpleNamespace(
        outputs={"probe": {"facts": {"duration_s": 4.0}}},
        sessions=lambda: nullcontext(db),
        store=object(),
        analysis_id="authored-test",
    )
    fused = audio_screening.fuse(ctx, configuration=cfg)
    assert all(s["detector_id"] != "aud.pella_v2" for s in fused["signals"])
    assert fused["decision"]["fuser_sha256"] == COMBINED_SHA256
    if case == "valid":
        assert (
            fused["decision"]["probability_ai_calibrated"] == 0.55
            and fused["decision"]["code"] == "ai_supported"
        )
    else:
        assert fused["decision"]["probability_ai_calibrated"] is None
        assert fused["decision"]["code"] in {"inconclusive", "not_assessable"}
        assert all(
            s.get("calibrated_probability") is None and s.get("effective_weight", 0.0) == 0.0
            for s in fused["signals"]
        )
    descriptor = json.loads(stored[0])
    assert descriptor["sha256"] == COMBINED_SHA256
    assert all(
        name not in json.dumps(descriptor).lower()
        for name in ("coefficients", "center", "scale", "wavlm", "pella")
    )


@pytest.mark.parametrize(
    "samples,enabled,pin,selected",
    [
        (64000, True, None, "combined"),
        (48000, True, None, "current"),
        (128000, True, None, "current"),
        (64000, False, None, "current"),
        (64000, True, "current", "current"),
        (64000, False, "combined", "combined"),
    ],
)
def test_fusion_route_preserves_scope_and_saved_reference_pin(monkeypatch, samples, enabled, pin, selected):
    from app.detectors.audio import calibration
    from app.engine import provenance_gate
    from app.engine.stages import audio, audio_screening

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(settings, "audio_combined_enabled", enabled)
    monkeypatch.setattr(provenance_gate, "stop", lambda ctx: False)
    monkeypatch.setattr(audio_screening, "fuse", lambda ctx, configuration: configuration.SHA256)
    pins = {"combined": COMBINED_SHA256, "current": calibration.SHA256, None: None}
    ctx = SimpleNamespace(state={"fuser_pin": pins[pin]}, outputs={"decode": {"samples": samples}})
    assert audio.fuse(ctx) == pins[selected]
