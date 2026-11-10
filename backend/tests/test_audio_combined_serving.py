"""Scope, failure and inference-only calibration contracts, without neural models."""

import copy
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.audio import combined


def bundle():
    return {
        "schema": combined.SCHEMA,
        "source_bindings": dict.fromkeys(
            (
                "study_freeze_sha256",
                "selected_global_sha256",
                "trained_fusion_sha256",
                "complement_head_sha256",
            ),
            "a" * 64,
        ),
        "complement": {
            "center": [0.0] * combined.WIDTH,
            "scale": [1.0] * combined.WIDTH,
            "variance_factor": float(np.sqrt(16 / combined.WIDTH)),
            "coefficients": [1.0] + [0.0] * combined.WIDTH + [0.0],
        },
        "global": {"coefficients": [0.25, 0.75, 0.0], "calibration": [1.0, 0.0]},
    }


def runtime():
    return combined.CombinedSpeechRuntime(combined.CombinedCalibration(bundle(), "a" * 64), object())


def deadline():
    return SimpleNamespace(check=lambda stage: None)


class Forbidden:
    def __getattr__(self, name):
        raise AssertionError("An unsupported check must not call a neural component")


@pytest.mark.parametrize("p", [0.0, 0.01, 0.5, 0.8, 1.0])
def test_authored_identity_map_preserves_direction_and_bounded_endpoints(p):
    expected_logit = float(np.clip(math.log(max(p, 1e-6)) - math.log(max(1 - p, 1e-6)), -8, 8))
    expected = 1 / (1 + math.exp(-expected_logit))
    result = runtime().calibration.predict(p, np.zeros(combined.WIDTH, np.float32))
    assert result == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    "pcm",
    [
        np.zeros(63999, np.float32),
        np.zeros(64001, np.float32),
        np.zeros(64000, np.float64),
        np.zeros((64000, 1), np.float32),
        np.full(64000, np.nan, np.float32),
    ],
)
def test_unsupported_source_is_missing_not_padded_truncated_or_original(pcm):
    result = runtime().analyze(pcm, Forbidden(), Forbidden(), deadline())
    assert result["probability_ai"] is None and result["decision"] == "inconclusive"
    assert result["checked_windows"] == [] and not result["scope_supported"]


def test_insufficient_speech_never_calls_models_or_clears_original():
    result = runtime().analyze(
        np.zeros(64000, np.float32), Forbidden(), SimpleNamespace(spans=lambda p: [(0.0, 1.9)]), deadline()
    )
    assert result["status"] == "not_applicable" and result["probability_ai"] is None
    assert result["decision"] == "inconclusive"


def test_exact_source_samples_and_evidence_window_survive_neural_boundary(monkeypatch):
    pcm = np.linspace(-0.1, 0.1, 64000, dtype=np.float32)
    original = pcm.copy()
    seen = []

    def predict(chunks):
        assert len(chunks) == 1 and np.array_equal(chunks[0], original)
        chunks[0][:] = 0  # A component cannot mutate another component's input.
        return [0.8]

    def extract(source, encoder):
        seen.append(np.array_equal(source, original))
        return np.zeros(combined.WIDTH, np.float32)

    monkeypatch.setattr(combined, "extract", extract)
    result = runtime().analyze(
        pcm, SimpleNamespace(predict=predict), SimpleNamespace(spans=lambda p: [(0.0, 4.0)]), deadline()
    )
    assert seen == [True] and np.array_equal(pcm, original)
    assert result["decision"] == "ai" and result["probability_ai"] == pytest.approx(0.8)
    assert result["checked_windows"] == [
        {"start_sample": 0, "end_sample": 64000, "start_s": 0.0, "end_s": 4.0}
    ]
    assert not any(name in json.dumps(result).lower() for name in ("wavlm", "pella", "microsoft", "sadanie"))


@pytest.mark.parametrize("spans", [[(0, 3), (2, 4)], [(0, 5)], [(float("nan"), 4)]])
def test_corrupt_or_overlapping_activity_spans_cannot_inflate_coverage(spans):
    with pytest.raises(ValueError):
        runtime().analyze(
            np.zeros(64000, np.float32), Forbidden(), SimpleNamespace(spans=lambda p: spans), deadline()
        )


@pytest.mark.parametrize("scores", [[], [0.2, 0.3], [float("nan")], [True]])
def test_failed_or_invalid_component_cannot_become_original(scores):
    with pytest.raises(ValueError):
        runtime().analyze(
            np.zeros(64000, np.float32),
            SimpleNamespace(predict=lambda p: scores),
            SimpleNamespace(spans=lambda p: [(0, 4)]),
            deadline(),
        )


def test_expired_job_does_not_continue_to_a_voice_component():
    def expired(stage):
        raise TimeoutError(stage)

    with pytest.raises(TimeoutError):
        runtime().analyze(
            np.zeros(64000, np.float32), Forbidden(), Forbidden(), SimpleNamespace(check=expired)
        )


def test_bundle_integrity_and_parameter_corruption_are_refused(tmp_path):
    path = tmp_path / "head.json"
    path.write_text(json.dumps(bundle()))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert combined.load_calibration(path, expected).sha256 == expected
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="integrity"):
        combined.load_calibration(path, expected)
    for key, value in [("scale", [0.0] * combined.WIDTH), ("coefficients", [True] * (combined.WIDTH + 2))]:
        changed = copy.deepcopy(bundle())
        changed["complement"][key] = value
        with pytest.raises(ValueError):
            combined.CombinedCalibration(changed, "a" * 64)
