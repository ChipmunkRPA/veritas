from __future__ import annotations

import cv2
import numpy as np

from scripts import image_chromatic_field_pilot as pilot


def test_authored_radial_scale_and_feathered_shift_replay():
    cv2.setNumThreads(1)
    result = pilot.authored()
    before = result["baseline"]["channels"]
    after = result["feathered_control"]["channels"]
    for channel, scale in (("red", .0024), ("blue", -.0020)):
        assert before[channel]["status"] == "measurable"
        assert after[channel]["status"] == "measurable"
        assert before[channel]["fold_tiles"] == [18, 18]
        assert np.allclose(before[channel]["fold_slope_px"], scale * 576, atol=.05)
    assert result["target_before_px"] < .03
    assert result["target_after_px"] > 1.2
    assert result["target_rank"] == 1
    assert result["exact_changed_pixels"] > result["target_tile_pixels"]**2


def test_textureless_and_constant_shift_stay_unassessed():
    flat = np.full((512, 768, 3), 128, np.uint8)
    result, residuals = pilot.analyze(flat)
    assert result["status"] == "field_not_supported"
    assert not residuals["red"]
    rows = []
    for y in range(6):
        for x in range(6):
            rows.append({"tile": (y, x), "xy": ((x - 2.5) / 6, (y - 2.5) / 6), "fold": (x+y) % 2,
                         "quadrant": (y >= 3, x >= 3), "shift": (.55, -.2), "corr": .98})
    null, _ = pilot.summarize_channel(rows)
    assert null["status"] == "field_not_supported"
    assert null["field_extent_px"] < .01


def test_frozen_native_profile_identifies_geometry_and_pinned_decoder():
    lock = pilot.freeze_document()
    assert lock["profile"]["tile_short_side_fraction"] == .125
    assert lock["rawpy_wheel_sha256"] == pilot.prior.WHEEL_SHA
    assert len(lock["implementation_sha256"]) == len(lock["source_receipt_sha256"]) == 64
