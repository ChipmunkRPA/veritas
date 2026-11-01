"""Behavioral checks for the offline, original-only noise sensitivity veto."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audio_calibration_v3_methods as method


def guard(n=12):
    return {str(label): [{"speaker": f"{label}:{i}", "maximum": .1} for i in range(n)] for label in (0, 1)}


def test_sensitive_original_abstains_without_changing_probability_or_creating_ai():
    result = method.decide(.05, .003136, .007343, guard())
    assert result["probability_ai"] == .05
    assert result["identity_guard_decision"] == "original"
    assert result["decision"] == "inconclusive"
    assert result["original_stability_veto"]


def test_sensitivity_only_vetoes_original_and_preserves_baseline_abstention():
    assert method.decide(.95, .9, .01, guard())["decision"] == "ai"
    assert method.decide(.7, .7, .001, guard())["decision"] == "inconclusive"
    assert method.decide(.05, .01, .011, guard())["decision"] == "original"


@pytest.mark.parametrize("probe", [None, float("nan"), float("inf"), -.01, 1.01])
def test_missing_or_invalid_probe_cannot_clear_original(probe):
    result = method.decide(.05, .001, probe, guard())
    assert result["decision"] == "inconclusive"
    assert not result["sensitivity"]["available"]


def test_sensitivity_threshold_is_fixed_on_absolute_log_odds_not_probability_difference():
    assert method.sensitivity(.01, .019)["original_clearance_stable"]
    assert not method.sensitivity(.01, .021)["original_clearance_stable"]
    assert method.sensitivity(.019, .01)["original_clearance_stable"]
    assert not method.sensitivity(.021, .01)["original_clearance_stable"]


def test_original_requires_probability_threshold_and_unique_class_set():
    assert method.decide(.21, .01, .01, guard())["decision"] == "inconclusive"
    ambiguous = guard()
    for row in ambiguous["1"]:
        row["maximum"] = .99
    assert method.decide(.05, .01, .01, ambiguous)["decision"] == "inconclusive"
    assert method.decide(.05, .01, .01, guard(8))["decision"] == "inconclusive"


def test_filter_preserves_length_and_source_and_reduces_out_of_band_energy():
    t = np.arange(32000) / 16000
    source = (.1*np.sin(2*np.pi*1000*t) + .1*np.sin(2*np.pi*6000*t)).astype(np.float32)
    before = source.copy()
    filtered = method.band_view(source)
    assert filtered.shape == source.shape
    assert filtered.dtype == np.float32
    assert np.array_equal(source, before)
    spec = np.abs(np.fft.rfft(filtered[8000:24000]))
    assert spec[1000] > 50 * spec[6000]


@pytest.mark.parametrize("source", [np.zeros(15999), np.zeros((16000, 2)), np.full(16000, np.nan)])
def test_filter_rejects_invalid_or_too_short_pcm(source):
    with pytest.raises(ValueError):
        method.band_view(source)


def test_scorer_uses_original_gate_once_and_identical_source_windows_for_both_views():
    class Activity:
        calls = 0

        def spans(self, pcm):
            self.calls += 1
            return [(2., 6.)]

    class Model:
        calls = []

        def predict(self, chunks):
            self.calls.append([c.copy() for c in chunks])
            return [.01 * (i+1) for i in range(len(chunks))]

    activity, model = Activity(), Model()
    source = np.sin(np.arange(8 * 16000) / 2).astype(np.float32) * .1
    result = method.score_views(source, model, activity)
    assert activity.calls == 1
    assert len(model.calls) == 2
    assert [(w["start_sample"], w["end_sample"]) for w in result["windows"]] == [
        (0, 64000), (32000, 96000), (64000, 128000)]
    assert np.array_equal(model.calls[0][1], source[32000:96000])
    assert np.array_equal(model.calls[1][1], method.band_view(source)[32000:96000])
    assert result["raw"] == result["band"] == .02


def test_probe_failure_keeps_original_score_but_cannot_clear_it():
    class Activity:
        def spans(self, pcm):
            return [(0., 4.)]

    class Model:
        calls = 0

        def predict(self, chunks):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("probe unavailable")
            return [.01]

    result = method.score_views(np.ones(64000, np.float32) * .01, Model(), Activity())
    assert result["raw"] == .01 and result["band"] is None
    assert "probe unavailable" in result["probe_error"]
    assert not result["sensitivity"]["original_clearance_stable"]


def test_fit_rejects_cal_and_test_rows_and_is_source_balanced():
    rows = [{"split": "fit", "base_id": f"{label}:{i}", "label": label, "raw": value}
            for label, values in [(0, [.01, .2, .3]), (1, [.7, .8, .99])]
            for i, value in enumerate(values)]
    fitted = method.fit_platt(rows)
    repeated = method.fit_platt(rows + [rows[0]] * 20)
    assert fitted["slope"] >= 0
    assert repeated["slope"] == pytest.approx(fitted["slope"], abs=1e-8)
    assert repeated["intercept"] == pytest.approx(fitted["intercept"], abs=1e-8)
    assert method.probability(.1, fitted) < method.probability(.9, fitted)
    for split in ("cal", "test"):
        with pytest.raises(ValueError, match="fit rows only"):
            method.fit_platt(rows + [dict(rows[0], split=split)])


def test_class_calibration_uses_identity_max_and_retains_missing():
    rows = [{"split": "cal", "speaker": "voice", "label": 1, "p": p} for p in (.95, .7, None)]
    result = method.calibrate_class_set(rows)
    assert result["1"] == [{"speaker": "voice", "maximum": None, "rows": 3}]
    with pytest.raises(ValueError, match="cal rows only"):
        method.calibrate_class_set([dict(rows[0], split="test")])
