"""Lazy probing must preserve the frozen final decision and original score."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audio_calibration_v3_methods as method
from audio_calibration_v3_scoring import score_candidate


def calibration(ambiguous=False):
    guards = {str(label): [{"speaker": f"{label}:{i}", "maximum": .1} for i in range(12)] for label in (0, 1)}
    if ambiguous:
        for row in guards["1"]:
            row["maximum"] = .99
    return {"parameters": {"new": {"slope": 1., "intercept": 0.}}, "guards": {"new": guards}}


class Activity:
    def __init__(self, speech=True):
        self.calls, self.speech = 0, speech

    def spans(self, pcm):
        self.calls += 1
        return [(0., len(pcm)/16000)] if self.speech else []


class Model:
    def __init__(self, raw, band, fail_probe=False):
        self.raw, self.band, self.fail_probe = raw, band, fail_probe
        self.calls = []

    def predict(self, chunks):
        self.calls.append([chunk.copy() for chunk in chunks])
        if len(self.calls) == 2 and self.fail_probe:
            raise RuntimeError("probe unavailable")
        score = self.raw if len(self.calls) == 1 else self.band
        return [score] * len(chunks)


@pytest.mark.parametrize("raw,band,ambiguous,expected,probe", [
    (.01, .011, False, "original", True),
    (.01, .2, False, "inconclusive", True),
    (.95, .01, False, "ai", False),
    (.5, .99, False, "inconclusive", False),
    (.01, .02, True, "inconclusive", False),
])
def test_lazy_and_full_probe_have_identical_probabilities_and_final_decisions(raw, band, ambiguous, expected, probe):
    pcm = (np.sin(np.arange(128000) / 20) * .1).astype(np.float32)
    cal = calibration(ambiguous)
    full_model, lazy_model = Model(raw, band), Model(raw, band)
    full = method.score_views(pcm, full_model, Activity())
    full_p = method.probability(full["raw_unrounded"], cal["parameters"]["new"])
    full_decision = method.decide(full_p, full["raw"], full["band"], cal["guards"]["new"])
    lazy = score_candidate(pcm, lazy_model, Activity(), cal)
    assert lazy["decision_detail"]["decision"] == full_decision["decision"] == expected
    assert lazy["decision_detail"]["probability_ai"] == full_decision["probability_ai"]
    assert lazy["raw"] == full["raw"]
    assert lazy["raw_unrounded"] == full["raw_unrounded"]
    assert lazy["source_pcm_sha256"] == full["source_pcm_sha256"]
    assert [(w["start_sample"], w["end_sample"], w["raw"]) for w in lazy["windows"]] == [
        (w["start_sample"], w["end_sample"], w["raw"]) for w in full["windows"]]
    assert len(full_model.calls) == 2
    assert len(lazy_model.calls) == (2 if probe else 1)
    if probe:
        assert lazy["band"] == full["band"] and lazy["band_pcm_sha256"] == full["band_pcm_sha256"]
        assert lazy["probe_status"] == "computed"
    else:
        assert lazy["band"] is None and lazy["band_pcm_sha256"] is None
        assert lazy["probe_status"] == "not_required_for_decision"
        assert lazy["probe_error"] is None


def test_fit_calibration_scores_skip_probe_entirely_without_becoming_a_failure(monkeypatch):
    def forbidden(_):
        raise AssertionError("The unused filter must not execute")

    monkeypatch.setattr(method, "band_view", forbidden)
    model = Model(.01, .99)
    result = score_candidate(np.ones(64000, np.float32) * .01, model, Activity())
    assert len(model.calls) == 1
    assert result["raw"] == .01 and result["band"] is None
    assert result["status"] == "ok" and result["probe_error"] is None
    assert result["probe_status"] == "not_required_for_decision"
    assert "decision_detail" not in result


def test_missing_speech_has_no_classifier_calls_and_remains_inconclusive():
    model, activity = Model(.01, .99), Activity(speech=False)
    result = score_candidate(np.ones(64000, np.float32) * .01, model, activity, calibration())
    assert not model.calls and activity.calls == 1
    assert result["status"] == "not_applicable"
    assert result["decision_detail"]["decision"] == "inconclusive"
    assert result["probe_status"] == "not_required_for_decision"


def test_required_probe_failure_still_vetoes_an_original():
    model = Model(.01, .99, fail_probe=True)
    result = score_candidate(np.ones(64000, np.float32) * .01, model, Activity(), calibration())
    assert len(model.calls) == 2
    assert result["probe_status"] == "failed"
    assert result["decision_detail"]["decision"] == "inconclusive"
    assert result["decision_detail"]["identity_guard_decision"] == "original"


@pytest.mark.parametrize("source", [np.zeros(15999), np.zeros((16000, 2)), np.full(16000, np.nan)])
def test_skipping_filter_does_not_skip_pcm_validation(source):
    with pytest.raises(ValueError):
        score_candidate(source, Model(.95, .01), Activity(), calibration())
