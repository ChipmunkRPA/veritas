# SPDX-License-Identifier: Apache-2.0
"""Portable bindings for a still and embedded video comparison."""

from __future__ import annotations

import re
from fractions import Fraction

KEY = "motion_photo_comparison"
SCHEMA = "motion-photo-still-video/1"
TITLE = "Motion Photo still and video"
SCOPE = "This check compares visible detail in the still with three nearby frames on the embedded video clock."
LIMITATION = "Movement, camera processing and ordinary edits can produce differences. This check carries zero weight in the AI assessment."
ROLES = {
    "context": "Highlighted area in the still",
    "still": "Unmarked crop from the still",
    "video": "Corresponding crop from the matched video frame",
    "mask": "Measured difference (white)",
}
REPORT_PROFILES = frozenset({"standard", "summary", "forensic"})
MAX_ASSET_BYTES = 1_048_576
FIELDS = ("schema", "source_sha256", "status", "authorship_confidence_contribution", "clock", "comparison", "region")
SHA = re.compile(r"[a-f0-9]{64}")
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_BASE = re.compile(r"[1-9]\d{0,6}/[1-9]\d{0,9}")


def preview_size(width, height, *, limit=384):
    scale = min(1., limit / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def describe(record):
    if record["status"] == "unassessed":
        return "A Motion Photo comparison was unavailable. The still needs a supported embedded video, a verified frame clock and enough matching detail."
    clock = record["clock"]
    compared = record["comparison"]
    total = compared["still_grid"][0] * compared["still_grid"][1]
    times = [f"{float(tick * Fraction(clock['time_base'])):.6f} s" for tick in clock["supporting_pts"]]
    match = float(clock["frame_pts"] * Fraction(clock["time_base"]))
    observation = ("The highlighted area differs across all three checked frames." if record["status"] == "different"
                   else "No local difference met the review criteria in the compared area.")
    details = (f"The comparison covered {compared['compared_pixels'] / total:.1%} of the still. "
               f"The displayed frame is at {match:.6f} s. Checked frame times {', '.join(times)}.")
    if not record["region"]:
        return f"{observation} {details}"
    x, y, w, h = record["region"]["still"]
    video_x, video_y, video_w, video_h = record["region"]["video"]
    return (f"{observation} {details} Still crop x {x}, y {y}, {w} × {h} px. "
            f"Video comparison crop x {video_x}, y {video_y}, {video_w} × {video_h} px on the checked frame grid.")


def projection(review):
    return {**{key: review[key] for key in FIELDS},
            "previews": {role: {key: metadata[key] for key in ("exhibit_id", "sha256")}
                         for role, metadata in review["previews"].items()}}


def _size(values, *, ceiling=768):
    return (isinstance(values, list) and len(values) == 2
            and all(type(value) is int and 1 <= value <= ceiling for value in values))


def _box(values, size):
    if not isinstance(values, list) or len(values) != 4 or any(type(value) is not int for value in values):
        raise ValueError("Invalid Motion Photo crop coordinates")
    x, y, width, height = values
    if not (0 <= x < x + width <= size[0] and 0 <= y < y + height <= size[1]):
        raise ValueError("The Motion Photo crop is outside its source")
    return width, height


def _clock(record):
    if (not isinstance(record, dict) or set(record) != {"declared_timestamp_us", "time_base", "frame_index",
                                                       "frame_pts", "supporting_pts", "video_sha256"}
            or not isinstance(record["time_base"], str) or not _BASE.fullmatch(record["time_base"])
            or not isinstance(record["video_sha256"], str) or not SHA.fullmatch(record["video_sha256"])
            or type(record["declared_timestamp_us"]) is not int or not 0 <= record["declared_timestamp_us"] <= 10_000_000
            or type(record["frame_index"]) is not int or not 0 <= record["frame_index"] < 180
            or type(record["frame_pts"]) is not int):
        raise ValueError("Invalid Motion Photo clock")
    ticks = record["supporting_pts"]
    if (not isinstance(ticks, list) or len(ticks) != 3 or any(type(tick) is not int or tick < 0 for tick in ticks)
            or not ticks[0] < ticks[1] < ticks[2] or record["frame_pts"] not in ticks):
        raise ValueError("Invalid Motion Photo comparison times")
    base = Fraction(record["time_base"])
    declared = Fraction(record["declared_timestamp_us"], 1_000_000)
    match = record["frame_pts"] * base
    if abs(match - declared) > Fraction(1, 150) or (ticks[2] - ticks[0]) * base > Fraction(1, 5):
        raise ValueError("The Motion Photo frame times differ from the declared clock")


def assets(record, source):
    if (not isinstance(record, dict) or set(record) != {*FIELDS, "previews"}
            or record["schema"] != SCHEMA or record["source_sha256"] != source.get("sha256")
            or not isinstance(record["source_sha256"], str) or not SHA.fullmatch(record["source_sha256"])
            or source.get("coordinate_space") != "exif-oriented-original"
            or type(record["authorship_confidence_contribution"]) is not int or record["authorship_confidence_contribution"] != 0
            or record["status"] not in ("unassessed", "no_region", "different") or not isinstance(record["previews"], dict)):
        raise ValueError("Invalid Motion Photo comparison")
    status, clock, comparison, region, previews = (record[key] for key in ("status", "clock", "comparison", "region", "previews"))
    if status == "unassessed":
        if clock is not None or comparison is not None or region is not None or previews:
            raise ValueError("An unassessed Motion Photo has comparison evidence")
        return {}
    _clock(clock)
    native_size = [source.get("width"), source.get("height")]
    if (not _size(native_size, ceiling=24_000_000) or min(native_size) < 320 or native_size[0] * native_size[1] > 24_000_000
            or not isinstance(comparison, dict) or set(comparison) != {"still_grid", "video_grid", "compared_pixels"}
            or not _size(comparison["still_grid"]) or not _size(comparison["video_grid"])):
        raise ValueError("Invalid Motion Photo comparison dimensions")
    width, height = comparison["still_grid"]
    count = comparison["compared_pixels"]
    if type(count) is not int or not .42 * width * height <= count <= width * height:
        raise ValueError("Invalid Motion Photo comparison coverage")
    if status == "no_region":
        if region is not None or previews:
            raise ValueError("A Motion Photo without a measured difference has unexpected crops")
        return {}
    if not isinstance(region, dict) or set(region) != {"still", "video", "mask", "marked_pixels"} or set(previews) != set(ROLES):
        raise ValueError("Incomplete Motion Photo region")
    still = _box(region["still"], native_size)
    video = _box(region["video"], comparison["video_grid"])
    mask = _box(region["mask"], comparison["still_grid"])
    if type(region["marked_pixels"]) is not int or not 165 <= region["marked_pixels"] <= min(mask[0] * mask[1], .09 * count):
        raise ValueError("Invalid size of the Motion Photo difference")
    sizes = {"context": preview_size(width, height, limit=512), "still": preview_size(*still),
             "video": preview_size(*video), "mask": mask}
    indexed = {}
    for role, asset in previews.items():
        if (not isinstance(asset, dict) or set(asset) != {"exhibit_id", "sha256"}
                or not isinstance(asset["exhibit_id"], str) or not _ID.fullmatch(asset["exhibit_id"])
                or not isinstance(asset["sha256"], str) or not SHA.fullmatch(asset["sha256"])
                or asset["exhibit_id"] in indexed):
            raise ValueError("Invalid Motion Photo exhibit")
        width, height = sizes[role]
        indexed[asset["exhibit_id"]] = {"media_type": "image/png", "sha256": asset["sha256"], "width": width, "height": height}
    return indexed
