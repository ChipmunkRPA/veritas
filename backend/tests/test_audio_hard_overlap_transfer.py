import numpy as np
import pytest

from scripts import audio_hard_overlap_transfer as trial
from scripts import review_audio_overlap_transfer as audit


def sine(hz):
    return np.sin(2 * np.pi * np.arange(48000) * hz / 16000)


def test_pitch_qualification_and_full_sample_mixture():
    assert trial.median_pitch(sine(110)) == pytest.approx(45, abs=.1)
    assert trial.median_pitch(sine(220)) == pytest.approx(57, abs=.15)
    assert trial.median_pitch(np.zeros(48000)) is None
    variants = trial.mix([sine(110), sine(122), sine(145)])
    assert len(variants) == 24 and all(len(row) == 96000 for row in variants.values())
    assert len(set(variants.values())) == 24
    data = [np.frombuffer(x, dtype="<i2").astype(float) for x in variants.values()]
    assert max(max(np.abs(x)) for x in data) <= 30148


def test_triplets_discard_wide_pitch_and_bind_distinct_source_groups():
    active = np.ones(48000, dtype=bool)
    pool = {str(i): {"identity": str(i), "gender": "female", "clip": (None, active, pitch)}
            for i, pitch in enumerate((48, 58, 59, 61, 62, 63, 64))}
    used = {"6"}
    groups = list(trial.triplets(pool, 2, used))
    assert [[x["identity"] for x in group] for group, _ in groups] == [["1", "2", "3"]]
    assert used == {"1", "2", "3", "6"}


def test_frozen_veto_withholds_zero_and_three_only_from_admitted_routes():
    def dist(index):
        return [float(i == index) for i in range(11)]

    rows = [{"id": i, "group": i, "language": "et", "delivery": "dry", "condition": "two" if expected == 2 else "three_quiet",
             "expected_voices": expected, "pcm_sha256": "hash"} for i, expected in enumerate((2, 3, 3))]
    measured = [{"id": i, "measurements": {k: {"route": "single_interferer"} for k in trial.diarization.POINTS}}
                for i in range(3)]
    counts = [{"id": i, "pcm_sha256": "hash", "probabilities": {"original": dist(2), "minus24": dist(expected)}}
              for i, expected in enumerate((2, 3, 0))]
    result = trial.summarize(rows, measured, counts)
    assert result["counts"]["diarization/all"]["incorrect_three_routes"] == 2
    assert result["counts"]["normalized_overlap_veto/all"]["incorrect_three_routes"] == 1
    assert result["counts"]["normalized_zero_or_overlap_veto/all"]["incorrect_three_routes"] == 0
    assert result["counts"]["normalized_zero_or_overlap_veto/all"]["correct_two_routes"] == 1
    assert not result["advance"]


def test_independent_interval_replay_deduplicates_tracks_and_checks_clock():
    two = [{"start": 0, "end": .40, "speaker": 1}, {"start": .01, "end": .39, "speaker": 1},
           {"start": .02, "end": .40, "speaker": 2}]
    assert audit.grid(two)["route"] == "single_interferer"
    assert audit.grid(two)["two_samples"] == 38 * 160
    overlap = [{"start": .02, "end": .19, "speaker": 3}]
    assert audit.grid(two + overlap)["route"] == "overlap_detected"
    assert audit.grid([{"start": 2.98, "end": 3.01, "speaker": 1}])["route"] == "unassessed"
    with pytest.raises(ValueError, match="sample grid"):
        audit.grid([{"start": .022, "end": .31, "speaker": 1}])
