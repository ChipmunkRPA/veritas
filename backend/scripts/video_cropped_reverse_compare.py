# SPDX-License-Identifier: MIT
"""Run both research arms of the reversed crop comparison."""
from app.detectors.video import picture_crop, picture_crop_reverse


def find(frames, *, center_only=False):
    return picture_crop_reverse.find(
        frames, percentages=picture_crop.CROP_PERCENTAGES, landmarks=not center_only,
    )
