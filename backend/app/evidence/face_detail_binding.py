"""Bind the compression-tested face detail check to native video inputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

METHOD = "relative-face-detail-samples/1"
SCHEDULE = tuple(range(5, 100, 6))
MINIMUM_TRACK = 8
POINT = .04553758344078262
CALIBRATION_SHA256 = "6c0bb233c94ecc633f0e4a83c904f7d154aa17ecd1ff4fccf9a87e8c006653eb"
WIDTHS = (2, 96, 81, 54)
_SHA = re.compile(r"[0-9a-f]{64}")


@lru_cache(maxsize=1)
def calibration():
    payload = (Path(__file__).with_name("data") / "face-swap-detail-v1.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != CALIBRATION_SHA256:
        raise ValueError("The frozen face detail calibration changed")
    document = json.loads(payload)
    if document["point"] != POINT or any(len(document[name]) != sum(WIDTHS) for name in ("center", "scale", "weight")):
        raise ValueError("Face detail calibration has the wrong width or operating point")
    return document


def base_scores(record):
    from app.evidence import whole_face_binding as binding

    readings = [[float.fromhex(x) if x is not None else None for x in row["responses_hex"]] for row in record["samples"]]
    first, second = zip(*readings, strict=True)
    return binding.paired_score(first, second), binding.same_frame_score(first, second)


def _contribution(values, offset):
    params = calibration()
    return math.fsum((value - params["center"][i]) * params["weight"][i] / params["scale"][i]
                     for i, value in enumerate(values, offset))


def contributions(base, relative_appearance, frequency):
    values = (base, relative_appearance, frequency["absolute"], frequency["relative"])
    if tuple(map(len, values)) != WIDTHS:
        raise ValueError("Face detail evidence has an unexpected feature width")
    return [_contribution(block, sum(WIDTHS[:i])) for i, block in enumerate(values)]


def eyes(row, width, height):
    raw = row["eyes_hex"]
    if not isinstance(raw, list) or len(raw) != 2 or any(not isinstance(point, list) or len(point) != 2 for point in raw):
        raise ValueError("Face detail eye positions have invalid geometry")
    points = tuple(tuple(float.fromhex(value) for value in point) for point in raw)
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points) or math.dist(*points) < 18:
        raise ValueError("Face detail eye positions lack native source support")
    return points


def validate(record, source, *, pts=None):
    from app.evidence import whole_face_binding as binding

    if (record.get("schema") != METHOD or type(record.get("flagged")) is not bool
            or record.get("time_base") != source["time_base"] or record.get("calibration_sha256") != CALIBRATION_SHA256
            or (source["width"], source["height"], source["frame_count"]) != (*binding.SAME_FRAME_DIMENSIONS, 100)
            or (pts is not None and not binding._fixed_clock(pts, Fraction(source["time_base"])))):
        raise ValueError("Face detail evidence is outside the supported source profile")
    prior = record["prior"]
    if prior["schema"] != binding.METHOD or prior["flagged"]:
        raise ValueError("Face detail evidence requires the earlier checks to finish without flagging")
    binding.validate(prior, source, pts=pts)
    first, second = base_scores(prior)
    if first is None or second is None or second > binding.SAME_FRAME_POINT:
        raise ValueError("Face detail evidence requires the earlier checks to finish without flagging")
    samples = record["samples"]
    indices = tuple(row["frame_index"] for row in samples)
    if (not MINIMUM_TRACK <= len(indices) <= len(SCHEDULE) or any(type(i) is not int for i in indices)
            or indices[0] not in SCHEDULE or any(b - a != 6 for a, b in zip(indices, indices[1:]))
            or indices[-1] not in SCHEDULE):
        raise ValueError("Face detail evidence requires an uninterrupted sampled face track")
    tb = Fraction(record["time_base"])
    first_tick = prior["samples"][0]["pts"] - prior["samples"][0]["frame_index"] * binding.SAME_FRAME_STEP / tb
    for index, row in zip(indices, samples, strict=True):
        x, y, w, h = (row["box"][key] for key in "xywh")
        eyes(row, source["width"], source["height"])
        if (type(row["pts"]) is not int or (row["pts"] - first_tick) * tb != index * binding.SAME_FRAME_STEP
                or (pts is not None and row["pts"] != pts[index])
                or any(type(item) is not int for item in (x, y, w, h)) or min(x, y) < 0 or min(w, h) < 56
                or x+w > source["width"] or y+h > source["height"]
                or not all(_SHA.fullmatch(row[key]) for key in ("decoded_rgb_sha256", "aligned_rgb_sha256"))):
            raise ValueError("Face detail input differs from the native face frame")
    values = [float.fromhex(value) for value in record["terms_hex"]]
    if len(values) != len(WIDTHS) or not all(map(math.isfinite, values)) or values[0] != _contribution((first, second), 0):
        raise ValueError("Recorded face detail contributions are inconsistent")
    score = math.fsum(values)
    if score.hex() != record["score_hex"] or (score > POINT) != record["flagged"]:
        raise ValueError("Face detail outcome differs from its recorded score")
    return indices
