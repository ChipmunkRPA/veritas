import copy

import numpy as np
import pytest

from scripts.audio_multisource_v8 import balanced_weights, calibrate, final, fit, predict


def row(identifier, label, family, group=None, role="fit", raw=0.1):
    return {
        "base_id": identifier,
        "label": label,
        "generator": family,
        "group": group or identifier,
        "role": role,
        "raw": raw,
    }


def test_more_voices_files_or_variants_do_not_dominate_a_generator_family():
    rows = [
        row("real-a", 0, "human"),
        row("real-b", 0, "human"),
        row("a1", 1, "engine-a", "voice-a"),
        row("a1", 1, "engine-a", "voice-a"),
        row("a2", 1, "engine-a", "voice-a"),
        row("b", 1, "engine-b"),
    ]
    weights = balanced_weights(rows)
    assert weights.tolist() == pytest.approx([0.25, 0.25, 0.0625, 0.0625, 0.125, 0.25])
    assert sum(w for r, w in zip(rows, weights, strict=True) if r["label"] == 0) == pytest.approx(0.5)
    assert sum(
        w for r, w in zip(rows, weights, strict=True) if r["generator"] == "engine-a"
    ) == pytest.approx(0.25)


def test_a_paired_genuine_and_cloned_voice_can_calibrate_both_truth_classes():
    rows = [
        row(f"{c}-{i}", c, "human" if c == 0 else "engine", f"voice-{i}", "cal", 0.1 if c == 0 else 0.9)
        for c in (0, 1)
        for i in range(10)
    ]
    fitted = {"transform": None, "coefficients": [1.0, 0.0]}
    before = copy.deepcopy(fitted)
    guards = calibrate(rows, fitted)
    assert len(guards["0"]) == len(guards["1"]) == 10
    assert [r["group"] for r in guards["0"]] == [r["group"] for r in guards["1"]]
    assert all(r["maximum"] == pytest.approx(0.1) for r in guards["0"] + guards["1"])
    assert fitted == before


def test_missing_calibration_window_is_retained_as_infinite_nonconformity():
    rows = [
        row(f"{c}-{i}", c, "human" if c == 0 else "engine", f"voice-{i}", "cal", 0.1 if c == 0 else 0.9)
        for c in (0, 1)
        for i in range(10)
    ]
    rows.append({**rows[0], "raw": None})
    guards = calibrate(rows, {"transform": None, "coefficients": [1.0, 0.0]})
    assert guards["0"][0]["maximum"] is None
    assert guards["0"][0]["rows"] == 2
    assert len(guards["0"]) == 10


def test_calibration_and_development_labels_cannot_fit_parameters():
    for role in ("cal", "development"):
        with pytest.raises(ValueError, match="Only fitting roles"):
            fit([row("a", 0, "human", role=role)], [1.0] * 20)
    with pytest.raises(ValueError, match="Only independent calibration"):
        calibrate([row("a", 0, "human", role="fit")], {})


def test_fit_uses_only_original_audio_features_for_predictions():
    rng = np.random.default_rng(84)
    rows = []
    for i in range(40):
        r = row(str(i), i // 20, "human" if i < 20 else "a" if i % 2 else "b", raw=0.2 if i < 20 else 0.8)
        r["embedding"] = rng.normal(size=20).tolist()
        rows.append(r)
    fitted = fit(rows, np.arange(1, 21, dtype=float))
    assert fitted["fit_groups_by_class"] == [20, 20]
    assert fitted["coefficients"][0] >= 0
    p = predict(rows[0], fitted)
    changed = {
        **rows[0],
        "label": 1,
        "generator": "unknown-generator",
        "group": "different-voice",
        "role": "development",
        "condition": "different-background",
        "snr_db": -100,
    }
    assert predict(changed, fitted) == p


@pytest.fixture
def guards():
    return {str(c): [{"maximum": 0.2} for _ in range(10)] for c in (0, 1)}


def test_flat_calibration_alone_cannot_clear_raw_unstable_original(guards):
    outcome = final(0.05, 0.06, 0.01, 0.001, guards)
    assert outcome["proposed"] == "original"
    assert outcome["decision"] == "inconclusive"
    assert outcome["raw_stability_log_odds_change"] > np.log(2)


def test_original_requires_both_probes_and_missing_views_never_clear(guards):
    assert final(0.05, 0.05, 0.01, 0.01, guards)["decision"] == "original"
    assert final(0.05, None, 0.01, 0.01, guards)["decision"] == "inconclusive"
    assert final(0.05, 0.05, 0.01, None, guards)["decision"] == "inconclusive"
    assert final(None, None, None, None, guards)["decision"] == "inconclusive"
    assert final(0.99, None, 0.99, None, guards)["decision"] == "ai"
