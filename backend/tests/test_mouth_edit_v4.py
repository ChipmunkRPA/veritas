from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.audio.decode import wav_bytes
from scripts.prepare_mouth_edit_v4 import composite, encode, interval, mouth_alpha


@pytest.mark.parametrize("frames", [63, 75, 100])
def test_known_edit_range_retains_before_and_after_context(frames):
    start, end = interval(frames)
    assert end - start == 20 and start >= 21 and frames - end >= 21


@pytest.mark.parametrize("frames", [62, 101])
def test_unsupported_clip_is_refused_instead_of_looped(frames):
    with pytest.raises(ValueError):
        interval(frames)


def test_native_pixels_outside_the_recorded_mouth_support_are_unchanged():
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    landmarks[78], landmarks[308] = SimpleNamespace(x=0.35, y=0.55), SimpleNamespace(x=0.65, y=0.55)
    landmarks[13], landmarks[14] = SimpleNamespace(x=0.5, y=0.52), SimpleNamespace(x=0.5, y=0.58)
    original = np.random.default_rng(3).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    alpha = mouth_alpha(original.shape, landmarks)
    saved = original.copy()
    result = composite(original, np.zeros_like(original), alpha, 0.5)
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_array_equal(result[alpha == 0], original[alpha == 0])
    assert alpha[10, 10] == 0 and alpha[55, 60] == 1 and np.any(result != original)


def test_lossless_video_roundtrip_preserves_every_planned_rgb_frame_and_pcm(tmp_path):
    frames = [np.random.default_rng(i).integers(0, 256, (16, 16, 3), dtype=np.uint8) for i in range(4)]
    audio = tmp_path / "source.wav"
    pcm = np.linspace(-0.2, 0.2, 4 * 640, dtype=np.float32)
    audio.write_bytes(wav_bytes(pcm))
    receipt = encode(frames, audio, tmp_path / "roundtrip.mkv")
    assert receipt["frames"] == 4 and receipt["audio_samples"] == 2560
    assert receipt["pts"] == [0, 40, 80, 120]
