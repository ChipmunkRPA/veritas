import numpy as np
import pytest

from scripts import audio_rueg_activity_crop_v252 as study


def test_continuous_activity_can_use_exact_fallback():
    pcm = np.full(80_000, 0.08, dtype=np.float32)
    with pytest.raises(ValueError, match="No pause-bounded"):
        study.source.crop.select_crop(pcm)
    bounds, receipt = study.select_crop(pcm)
    assert bounds == (0, 56_000)
    assert receipt["selection"] == "activity_ranked_crop"


def test_silent_signal_still_fails_and_existing_short_crop_stays_exact():
    with pytest.raises(ValueError, match="No pause-bounded"):
        study.select_crop(np.zeros(80_000, dtype=np.float32))
    pcm = np.full(48_000, 0.08, dtype=np.float32)
    assert study.select_crop(pcm) == study.source.crop.select_crop(pcm)
