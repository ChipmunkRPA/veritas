# SPDX-License-Identifier: Apache-2.0
"""Repeat one selected appearance window on the exact lower-detail inputs."""

from __future__ import annotations

import hashlib
import math
import re
from fractions import Fraction

from app.core.canonical import canonical_hash
from app.evidence.face_interval_binding import (
    APPEARANCE_METHOD,
    APPEARANCE_REVIEW_POINT,
    validate_sequence_record,
)

METHOD = "face-appearance-detail-repeat/1"
ENCODING = ("h264", "yuv420p")
FRAME_COUNT = 120
DIMENSIONS = (640, 360)
STEP = Fraction(1, 25)
REPLAY_TOLERANCE = 2e-6
ROLES = ("before", "during", "after")
MEASUREMENTS = ("response", "middle_to_before", "middle_to_after", "flank_change")
_HASH = re.compile(r"[0-9a-f]{64}")


def supported(table, encoding) -> bool:
    if (encoding != ENCODING or (table.width, table.height) != DIMENSIONS or len(table) != FRAME_COUNT
            or table.rotation or table.interpolated or table.truncated):
        return False
    base = table.tb
    return all((right-left)*base == STEP for left, right in zip(table.pts, table.pts[1:]))


def reduced_crop(aligned):
    import cv2

    if aligned is None or aligned.shape != (112, 112, 3) or aligned.dtype.name != "uint8":
        raise ValueError("The repeat requires the exact aligned 112-pixel face")
    smaller = cv2.resize(aligned, (56, 56), interpolation=cv2.INTER_AREA)
    return cv2.resize(smaller, (112, 112), interpolation=cv2.INTER_LINEAR)


def window_measurements(centers, vectors) -> dict[str, float]:
    """Pool the same triplets and before, during and after groups as the selected window."""
    import numpy as np

    from app.detectors.video import face_appearance as appearance

    groups = []
    for role in ROLES:
        triplets = [np.mean([vectors[center+offset] for offset in (-1, 0, 1)], axis=0) for center in centers[role]]
        groups.append(np.mean(triplets, axis=0))
    before, during, after = groups
    left, right = appearance.distance(during, before), appearance.distance(during, after)
    flank = appearance.distance(before, after)
    return dict(zip(MEASUREMENTS, (min(left, right)-appearance.FLANK_PENALTY*flank, left, right, flank), strict=True))


def _original_response(coarse) -> float:
    if coarse.get("method") != APPEARANCE_METHOD or not isinstance(coarse.get("track_id"), str) or not coarse["track_id"]:
        raise ValueError("The repeat needs a selected tracked appearance window")
    original = float.fromhex(coarse["measurement_hex"]["response"])
    if not math.isfinite(original) or original <= APPEARANCE_REVIEW_POINT:
        raise ValueError("The original tracked window did not meet the inspection rule")
    return original


def validate(measurement: dict, coarse: dict) -> None:
    """Check a measured repeat against its frozen coarse candidate without inference."""
    original = _original_response(coarse)
    if (set(measurement) != {"method", "status", "candidate_sha256", "source_encoding", "original_response_hex",
                             "measurements_hex", "frame_inputs"} or measurement["method"] != METHOD
            or measurement["candidate_sha256"] != canonical_hash(coarse)
            or measurement["source_encoding"] != list(ENCODING)):
        raise ValueError("The repeat differs from its tracked appearance window")
    values = measurement["measurements_hex"]
    if set(values) != set(MEASUREMENTS):
        raise ValueError("The repeat needs four exact appearance measurements")
    response, left, right, flank = (float.fromhex(values[key]) for key in MEASUREMENTS)
    repeated_original = float.fromhex(measurement["original_response_hex"])
    if (not math.isfinite(repeated_original) or abs(repeated_original-original) > REPLAY_TOLERANCE
            or any(not -1e-12 <= value <= 2.000000000001 for value in (left, right, flank))
            or not math.isfinite(response) or abs(response-(min(left, right)-.35*flank)) > 1e-12
            or measurement["status"] != ("withheld" if response <= APPEARANCE_REVIEW_POINT else "retained")):
        raise ValueError("The repeat has inconsistent appearance measurements or selection")
    inputs = measurement["frame_inputs"]
    if (not isinstance(inputs, list) or len(inputs) != len(coarse["frame_inputs"])
            or any(set(item) != {"frame_index", "pts", "lower_detail_rgb_sha256"}
                   or type(item["frame_index"]) is not int or type(item["pts"]) is not int
                   or (item["frame_index"], item["pts"]) != (known["frame_index"], known["pts"])
                   or not _HASH.fullmatch(str(item["lower_detail_rgb_sha256"]))
                   for item, known in zip(inputs, coarse["frame_inputs"], strict=True))):
        raise ValueError("The repeat does not bind the same native picture inputs")


def original_range(details: dict) -> dict | None:
    """Preserve the coarse interval used for distinctness even after withholding it."""
    if (repeat := details.get("face_detail_repeat")) and repeat["status"] == "withheld":
        original = details["face_sequence_windows"][0]
        validate(repeat["measurement"], original)
        return original
    return details.get("face_sequence")


def observation(status: str, start: float, end: float) -> str:
    location = f"The sampled face change at {start:.3f}–{end:.3f} s"
    if status == "withheld":
        return f"{location} weakened when the same frames were checked at lower detail. The interval was withheld from the review ranges."
    if status == "retained":
        return f"{location} remained when the same frames were checked at lower detail."
    return f"{location} could not be repeated at lower detail."


def measure(coarse, table, encoding, frames, recognizer, *, check=None) -> dict:
    """Measure on `(frame_index, verified_pts_or_None, rgb)` triples from the selected window."""
    import cv2

    from app.detectors.video import face_appearance as appearance

    if not supported(table, encoding):
        raise ValueError("The video is outside the qualified detail-repeat scope")
    source = {"time_base": table.time_base, "frame_count": len(table), "width": table.width, "height": table.height}
    validate_sequence_record(coarse, source, table_pts=table.pts, interpolated=table.interpolated)
    _original_response(coarse)
    inputs = {item["frame_index"]: item for item in coarse["frame_inputs"]}
    original, reduced, commitments = {}, {}, {}
    for index, pts, rgb in frames:
        if check is not None:
            check()
        item = inputs.get(index)
        if (type(index) is not int or item is None or index in original or type(pts) is not int or pts != item["pts"]
                or rgb.shape != (table.height, table.width, 3) or rgb.dtype.name != "uint8"
                or hashlib.sha256(rgb.tobytes()).hexdigest() != item["decoded_rgb_sha256"]):
            raise ValueError("The repeat does not bind the same verified source picture")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        aligned = recognizer.alignCrop(bgr, appearance.face_row(item["landmarks"]))
        low = reduced_crop(aligned)
        if hashlib.sha256(aligned[:, :, ::-1].tobytes()).hexdigest() != item["aligned_rgb_sha256"]:
            raise ValueError("The repeat does not reproduce the recorded aligned face crop")
        original[index] = appearance.unit(recognizer.feature(aligned).reshape(-1))
        reduced[index] = appearance.unit(recognizer.feature(low).reshape(-1))
        commitments[index] = {"frame_index": index, "pts": pts,
                              "lower_detail_rgb_sha256": hashlib.sha256(low[:, :, ::-1].tobytes()).hexdigest()}
    if original.keys() != inputs.keys():
        raise ValueError("The repeat is missing one or more selected native pictures")
    if check is not None:
        check()
    original_response = window_measurements(coarse["centers"], original)["response"]
    values = window_measurements(coarse["centers"], reduced)
    result = {"method": METHOD, "status": "withheld" if values["response"] <= APPEARANCE_REVIEW_POINT else "retained",
              "candidate_sha256": canonical_hash(coarse), "source_encoding": list(ENCODING),
              "original_response_hex": original_response.hex(),
              "measurements_hex": {key: value.hex() for key, value in values.items()},
              "frame_inputs": [commitments[index] for index in sorted(commitments)]}
    validate(result, coarse)
    return result
