import hashlib

import cv2
import numpy as np
import pytest

from scripts.prepare_mouth_training_v6 import POSITIONS, save_rgb, selected_indices


@pytest.mark.parametrize("frames", [50, 62, 100])
def test_short_training_sources_preserve_real_frames_and_prior_driver_context(frames):
    start, indices = selected_indices(frames)
    assert len(set(indices)) == 8
    assert tuple(i - start for i in indices) == POSITIONS
    assert min(indices) - 8 >= 0 and max(indices) < frames
    assert max(indices) - start == 19


@pytest.mark.parametrize("frames", [0, 49, 101])
def test_training_frame_plan_never_pads_short_sources(frames):
    with pytest.raises(ValueError):
        selected_indices(frames)


def test_png_training_input_is_exact_native_rgb(tmp_path):
    rgb = np.random.default_rng(91).integers(0, 256, (67, 93, 3), dtype=np.uint8)
    path = tmp_path / "frame.png"
    receipt = save_rgb(path, rgb)
    assert receipt["native_rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
    assert np.array_equal(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), rgb)
