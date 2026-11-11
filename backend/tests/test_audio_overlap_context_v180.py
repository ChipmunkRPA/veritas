import numpy as np
import pytest

from scripts import audio_overlap_context_v180 as probe


def response():
    probabilities = np.full((1, probe.FRAMES, 7), 0.01)
    probabilities[:, :, 1] = 0.94
    return probabilities


def test_intervals_use_only_source_supported_frames_and_sample_cells():
    probabilities = response()
    probabilities[:, 2:17, 1] = 0.01
    probabilities[:, 2:17, 4] = 0.94
    probabilities[:, 175:, 1] = 0.01
    probabilities[:, 175:, 6] = 0.94
    found = probe.overlap_record(np.log(probabilities), 48_000)
    assert found["eligible_frames"] == 175
    assert found["overlap_frames"] == 15
    assert found["longest_overlap_samples"] == 4050
    assert found["private_intervals_samples"] == [[900, 4950]]
    assert found["diagnostic_context"] is True


def test_separate_short_runs_do_not_add_up_to_contiguous_overlap():
    probabilities = response()
    for indices in (slice(2, 12), slice(18, 28)):
        probabilities[:, indices, 1] = 0.01
        probabilities[:, indices, 5] = 0.94
    found = probe.overlap_record(np.log(probabilities), 48_000)
    assert found["overlap_frames"] == 20
    assert found["longest_overlap_samples"] == 2700
    assert found["diagnostic_context"] is False


def test_rejects_unscaled_scores_and_malformed_response():
    with pytest.raises(ValueError, match="normalized"):
        probe.overlap_record(response(), 48_000)
    with pytest.raises(ValueError, match="shape"):
        probe.overlap_record(np.zeros((1, 175, 7)), 48_000)
