import numpy as np
import pytest

from scripts import audio_independent_overlap_count as trial
from scripts import audio_overlap_count_level_replay as level_replay


def responses(estimate):
    return [float(i == estimate) for i in range(11)]


def measured(route):
    return {name: {"route": route} for name in trial.prior.POINTS}


def test_padding_preserves_all_source_samples_and_never_repeats_speech():
    source = np.arange(48000, dtype="float64")
    for left in trial.POSITIONS.values():
        view = trial.padded(source, left)
        assert view.shape == (80000,)
        np.testing.assert_array_equal(view[left:left + len(source)], source)
        assert not view[:left].any() and not view[left + len(source):].any()
    with pytest.raises(ValueError):
        trial.padded(source[:-1], 16000)


def test_exact_features_have_reference_shape_and_normalization():
    samples = np.arange(80000, dtype="float64")
    audio = np.sin(2 * np.pi * 400 * samples / 16000)
    measured_features = trial.features(audio, np.zeros(201), np.ones(201))
    assert measured_features.shape == (500, 201)
    assert measured_features[250].argmax() == 10
    assert np.linalg.norm(measured_features, axis=1).mean() == pytest.approx(1.0)


def test_count_and_veto_do_not_promote_unassessed_or_overlap():
    assert trial.count(responses(3)) == 3
    with pytest.raises(ValueError):
        trial.count([float("nan")] + responses(3)[1:])
    counts = {"left": 2, "center": 3, "right": 2}
    assert trial.route_arms(measured("single_interferer"), counts) == {
        "diarization": "single_interferer", "center_veto": "overlap_withheld", "repeat_veto": "single_interferer",
    }
    counts["left"] = 3
    assert trial.route_arms(measured("single_interferer"), counts)["repeat_veto"] == "overlap_withheld"
    for route in ("unassessed", "overlap_detected"):
        assert set(trial.route_arms(measured(route), counts).values()) == {route}


def test_aggregation_counts_only_previously_inconclusive_recovered_groups():
    common = {"group": "source", "panel": trial.CONFIRM, "condition": "mostly_single_interferer",
              "label": 1, "crossed_room_point": {"raw_point": True}, "pcm_sha256": "digest"}
    rows = [{**common, "id": index, "decision": decision} for index, decision in enumerate(("inconclusive", "ai"))]
    model = [{"id": index, "measurements": measured("single_interferer")} for index in range(2)]
    data = [{"id": index, "pcm_sha256": "digest", "probabilities": {name: responses(3) for name in trial.POSITIONS}}
            for index in range(2)]
    result = trial.evaluate(rows, model, data)
    assert result["synthetic_groups_added"] == {f"{trial.CONFIRM}:diarization:raw_point": 1}
    assert result["routes"][trial.CONFIRM]["mostly_single_interferer"]["repeat_veto:overlap_withheld"] == 2
    data[0]["pcm_sha256"] = "wrong digest"
    with pytest.raises(ValueError):
        trial.evaluate(rows, model, data)


def test_level_normalization_uses_one_gain_without_clipping_or_creating_audio():
    signal = np.tile([.01, -.01], 24000)
    scaled, gain, limited = level_replay.level(signal, -24)
    np.testing.assert_array_equal(scaled, signal * gain)
    assert np.linalg.norm(scaled) / np.sqrt(len(signal)) == pytest.approx(10 ** (-24 / 20))
    assert not limited
    transient = np.zeros_like(signal)
    transient[24000] = .9
    scaled, _, limited = level_replay.level(transient, -18)
    assert limited and scaled[24000] == pytest.approx(.95)
    assert np.count_nonzero(scaled) == 1
    assert level_replay.level(np.zeros_like(signal), -18)[1:] == (1., False)
