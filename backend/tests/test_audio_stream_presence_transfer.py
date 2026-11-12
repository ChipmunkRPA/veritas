import numpy as np
import pytest

from scripts import audio_stream_presence_transfer as trial
from scripts import audio_two_three_separator_transfer as paired
from scripts import prepare_natural_overlap_transfer as natural


def three_signals():
    samples = np.arange(24000) / 8000
    return np.stack([np.sin(2 * np.pi * frequency * samples) for frequency in (101, 197, 307)])


def test_stability_is_permutation_sign_and_gain_invariant():
    base = three_signals()
    shifted = base[[2, 0, 1]] * np.array([[-2], [.3], [1.1]])
    assert trial.aligned_correlations(base, shifted) == pytest.approx([1, 1, 1])


def test_duplicate_output_keeps_energy_but_lacks_independence():
    base = three_signals()
    duplicated = np.stack([base[0], base[1], .5 * base[0]])
    active = [np.ones(trial.VAD_LENGTH, dtype=bool)] * 3
    measured = trial.assess(base[0] + base[1], [duplicated] * 3, active)
    assert measured["raw_energy"] and measured["qualified"]
    assert measured["scores"]["aligned_stability"] == pytest.approx(1)
    assert measured["scores"]["stable_independent"] == pytest.approx(0)
    assert trial.known_source_audit(base[0] + base[1], duplicated, base[:2]) == {
        "extra_output_category": "largely_one_known_voice"}


def test_stable_component_without_simultaneous_activity_is_unassessed():
    base = three_signals()
    masks = [np.arange(trial.VAD_LENGTH) < 24000, np.arange(trial.VAD_LENGTH) >= 24000, np.ones(trial.VAD_LENGTH, dtype=bool)]
    result = trial.assess(np.sum(base, axis=0), [base] * 3, masks)
    assert not result["qualified"]
    assert all(value is None for value in result["scores"].values())


def test_two_stream_projection_suppresses_duplicate_but_keeps_distinct_signal():
    a, b, c = three_signals()
    extra, novel, match = paired.extra_measurement(a + b, np.stack([a, b]), np.stack([b, .2 * a, -a]))
    assert novel == pytest.approx(0, abs=1e-10)
    assert match == pytest.approx(1)
    _, novel, match = paired.extra_measurement(a + b + c, np.stack([a, b]), np.stack([b, c, a]))
    assert novel == pytest.approx(1 / np.sqrt(3), abs=1e-10)
    assert match == pytest.approx(1)


def test_points_use_one_maximum_per_development_group():
    rows = []
    for group in range(8):
        rows.extend({"group": group, "expected_voices": 2,
                     "observation": {"scores": dict.fromkeys(trial.ARMS, score)}}
                    for score in (group / 10, group / 10 + .02))
    assert trial.fit_points(rows) == pytest.approx(dict.fromkeys(trial.ARMS, .62))


def test_natural_label_uses_true_coactivity_and_excludes_fourth_voice():
    assert natural.candidate_windows([[(310, 390), (605, 670)], [(320, 400), (600, 680)], [(620, 670)], [(690, 700), (950, 955)]]) == {
        2: [300], 3: []}
    assert natural.candidate_windows([[(310, 390), (605, 670)], [(320, 400), (600, 680)], [(620, 670)], [(690, 695), (950, 955)]]) == {
        2: [300], 3: [600]}
