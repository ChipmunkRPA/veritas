# SPDX-License-Identifier: Apache-2.0
"""Bounded picture correspondence for paired, independently encoded H.264 clips."""

from fractions import Fraction
from subprocess import SubprocessError

import cv2
import numpy as np

from app.detectors.video.decode import DecodeError
from app.evidence.video_reference_local import _carrier

BASIS = "paired_h264_soundtrack_picture_grid/1"
FRAME_COUNT = 63
HOP = 640
PICTURE_SIZE = (480, 360)
POINTS = {"whole_q65": 2.5, "whole_q90": 4.0, "local_five_frame_median": 6.27}
SCOPE = (
    "This compares two supplied clips on the same verified picture and sound clocks. The picture comparison allows "
    "small H.264 re-encoding differences. A reported shift measures where matching sound moved between the copies. "
    "Ordinary edits can produce the same result. This check carries zero weight in the AI estimate."
)


def eligible(paths, tables):
    if any((table.width, table.height) != PICTURE_SIZE or len(table) != FRAME_COUNT
           or table.rotation or table.interpolated or table.truncated
           or any((tick - table.pts[0]) * table.tb != Fraction(index, 25) for index, tick in enumerate(table.pts))
           for table in tables):
        return False
    try:
        return all(_carrier(path) == [("h264", "yuv420p")] for path in paths)
    except (DecodeError, OSError, ValueError, SubprocessError):
        return False


def picture_difference(first, second):
    a, b = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (80, 60),
                       interpolation=cv2.INTER_AREA).astype(np.float32) for frame in (first, second)]
    return np.abs(a - b)


def summarize(differences):
    values = np.asarray(differences)
    frame_errors = values.mean(axis=(1, 2))
    cells = values.reshape(FRAME_COUNT, 12, 5, 16, 5).mean(axis=(2, 4))
    five_frames = np.lib.stride_tricks.sliding_window_view(cells, 5, axis=0)
    measured = {
        "whole_q65": round(float(np.quantile(frame_errors, .65)), 6),
        "whole_q90": round(float(np.quantile(frame_errors, .90)), 6),
        "local_five_frame_median": round(float(np.median(five_frames, axis=-1).max()), 6),
    }
    return {"measurements": measured, "points": POINTS, "compared_pictures": FRAME_COUNT,
            "within_bounds": all(measured[key] <= point for key, point in POINTS.items())}
