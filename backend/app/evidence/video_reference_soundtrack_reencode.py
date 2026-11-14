# SPDX-License-Identifier: Apache-2.0
"""Bounded picture correspondence for paired, independently encoded H.264 clips."""

from fractions import Fraction
from subprocess import SubprocessError

import cv2
import numpy as np

from app.detectors.audio.decode import SAMPLE_RATE
from app.detectors.video.decode import DecodeError
from app.evidence.video_reference_local import _carrier

BASIS = "paired_h264_soundtrack_picture_grid/1"
FRAME_COUNT = 63
PICTURE_SIZE = (480, 360)
HD_PICTURE_SIZE = (1280, 720)
HD_RATES = (24, 25, 30)
HD_GRID = (160, 90)
HD_30_PROFILE = {"width": HD_PICTURE_SIZE[0], "height": HD_PICTURE_SIZE[1],
                 "pictures": 120, "frames_per_second": 30}
POINTS = {"whole_q65": 2.5, "whole_q90": 4.0, "local_five_frame_median": 6.27}
SCOPE = (
    "This compares two supplied clips on the same verified picture and sound clocks. The picture comparison allows "
    "small H.264 re-encoding differences. A reported shift measures where matching sound moved between the copies. "
    "Ordinary edits can produce the same result. This check carries zero weight in the AI estimate."
)
HD_SCOPE = SCOPE.replace("small H.264 re-encoding differences.",
                         "small H.264 re-encoding differences on both picture grids.")


def profile(tables, *, allow_hd=False):
    """Select the tested geometry and exact rational clock of both copies."""
    candidates = [(PICTURE_SIZE, FRAME_COUNT, 25)]
    if allow_hd:
        candidates.extend((HD_PICTURE_SIZE, 4 * rate, rate) for rate in HD_RATES)
    for size, count, rate in candidates:
        if all((table.width, table.height) == size and len(table) == count for table in tables):
            return {"pictures": count, "rate": rate, "size": size, "samples": count * SAMPLE_RATE // rate}
    return None


def eligible(paths, tables, *, allow_hd=False):
    selected = profile(tables, allow_hd=allow_hd)
    if selected is None or any(
        table.rotation or table.interpolated or table.truncated
        or any((tick - table.pts[0]) * table.tb != Fraction(index, selected["rate"]) for index, tick in enumerate(table.pts))
        for table in tables
    ):
        return False
    try:
        return all(_carrier(path) == [("h264", "yuv420p")] for path in paths)
    except (DecodeError, OSError, ValueError, SubprocessError):
        return False


def scope(picture):
    return HD_SCOPE if "fine_grid" in picture["soundtrack_picture_comparison"] else SCOPE


def hd_report_profile(tables):
    selected = profile(tables, allow_hd=True)
    if selected is None or selected["size"] != HD_PICTURE_SIZE:
        raise ValueError("The paired picture grid requires the recorded HD profile")
    return {"width": HD_PICTURE_SIZE[0], "height": HD_PICTURE_SIZE[1], "pictures": selected["pictures"],
            "frames_per_second": selected["rate"]}


def report_profile(picture):
    comparison = picture["soundtrack_picture_comparison"]
    # Existing 30 fps exports only committed the profile on the timing record.
    return {"picture_profile": comparison.get("picture_profile", HD_30_PROFILE)} if "fine_grid" in comparison else {}


def picture_difference(first, second, *, grid=(80, 60)):
    a, b = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), grid,
                       interpolation=cv2.INTER_AREA).astype(np.float32) for frame in (first, second)]
    return np.abs(a - b)


def summarize(differences):
    values = np.asarray(differences)
    count, height, width = values.shape
    frame_errors = values.mean(axis=(1, 2))
    cells = values.reshape(count, height // 5, 5, width // 5, 5).mean(axis=(2, 4))
    five_frames = np.lib.stride_tricks.sliding_window_view(cells, 5, axis=0)
    measured = {
        "whole_q65": round(float(np.quantile(frame_errors, .65)), 6),
        "whole_q90": round(float(np.quantile(frame_errors, .90)), 6),
        "local_five_frame_median": round(float(np.median(five_frames, axis=-1).max()), 6),
    }
    return {"measurements": measured, "points": POINTS, "compared_pictures": len(values),
            "within_bounds": all(measured[key] <= point for key, point in POINTS.items())}
