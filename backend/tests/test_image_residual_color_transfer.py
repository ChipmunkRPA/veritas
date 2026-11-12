"""Checks for native residual-color response and unsupported texture."""

import numpy as np

from scripts import image_residual_color_transfer as trial


def test_color_relationship_change_has_localized_multiband_support():
    rng = np.random.default_rng(223)
    noise = rng.normal(0, 17, (384, 384))
    native = np.uint8(np.clip(127 + np.repeat(noise[:, :, None], 3, axis=2), 0, 255))
    changed = native.copy()
    changed[128:256, 128:256, 1] = np.uint8(np.clip(127 - noise[128:256, 128:256], 0, 255))
    original = trial.response(native)
    edited = trial.response(changed)
    assert all(row["score"] < trial.MIN_POINT for row in original.values())
    assert all(row["box"] == [128, 128, 256, 256] and row["score"] > 1 for row in edited.values())
    bands = edited["mid_agreement"]["bands"]
    assert len(bands) == 2 and edited["mid_agreement"]["coherence"] > .95
    assert all(row["core_distance"] > 1 and row["side_distance"] < .01 for row in bands)


def test_clipped_or_featureless_native_pixels_are_unassessed():
    for tone in (0, 130, 255):
        native = np.full((384, 384, 3), tone, dtype=np.uint8)
        assert trial.response(native) == dict.fromkeys(trial.ARMS)
