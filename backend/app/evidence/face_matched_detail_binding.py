# SPDX-License-Identifier: Apache-2.0
"""Portable clock, measurements and source records for the matched-detail face check."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from app.evidence.video_packet_versions import (
    MATCHED_DETAIL_AVC_VERSION,
    MATCHED_DETAIL_HEVC25_VERSION,
    MATCHED_DETAIL_NATIVE25_VERSION,
)

KEY = "face_matched_detail"
COVERAGE = "face_matched_detail_coverage"
COMMITMENT = "face_matched_detail_sha256"
COVERAGE_COMMITMENT = "face_matched_detail_coverage_sha256"
COMMITMENTS = (COMMITMENT, COVERAGE_COMMITMENT)
METHOD = "face-matched-detail/1"
NATIVE25_METHOD = "face-matched-detail/2"
HEVC25_METHOD = "face-matched-detail/3"
PACKET_ID = "VMD01"
TITLE = "Very brief face appearance change after matching image detail"
OBSERVATION = (
    "The face appearance differs within this short range after comparing crops at a similar level of detail."
)
INTERPRETATION = "The original frames show the face before, during, and after the selected range. Ordinary blur and other edits can produce a similar change."
LIMITATION = "This check carries zero weight in the overall assessment and only covers short ranges with comparable views."
ROLES = ("before", "during", "after")
FRAME_COUNT = 120
SIZE = (1280, 720)
TIME_BASE = "1/30000"
TICKS = tuple(i * 1000 for i in range(FRAME_COUNT))
AVC = ("h264", "yuv420p", "avc1")
HEVC = ("hevc", "yuv420p", "hvc1")
POINT = 0.10
SIGMAS = (0.0, 0.65, 1.0, 1.35, 1.8, 2.4, 3.2, 4.2, 5.0)
DETAIL_FLOOR = 0.75
FRACTION = 0.80
MAX_PAIRS = 8
MAX_FACE_READS = 60
REASONS = {
    "profile": "This check needs a four-second 720p AVC MP4 with verified 30 fps timing.",
    "profile_avc": "This check needs 120 verified 720p AVC MP4 frames at 30 fps (four seconds) or 25 fps (4.8 seconds).",
    "profile_hevc": "This check needs a verified 4.8-second 720p HEVC MP4 at 25 fps with a single hvc1 video track and no audio.",
    "prior_review": "The brief face appearance review already found a range for this file.",
    "shot_context": "The original pictures did not supply a complete shot review.",
    "face_context": "The source did not supply comparable face views around a selected range.",
    "time_budget": "This review did not complete within its time limit.",
    "source_input": "The native source pictures could not be verified for this check.",
}
STATUS_TEXT = {
    "localized": "A very brief face appearance change remained after matching image detail.",
    "no_match": "The comparable source views supplied no additional brief face change.",
}
_DIGEST = re.compile(r"[a-f0-9]{64}")
READINGS = ("near", "far", "flanks", "inside", "entry", "exit")
SUPPORT = ("detail_drop", "edge_step", "flank_patch_difference")


@dataclass(frozen=True)
class NativeProfile:
    method: str
    version: str
    timescale: int
    step: int
    point: float
    encoding: tuple[str, str, str] = AVC

    @property
    def time_base(self):
        return f"1/{self.timescale}"

    def exact_clock(self, time_base, pts):
        return time_base == self.time_base and tuple(pts) == tuple(i * self.step for i in range(FRAME_COUNT))

    def seconds(self, ticks):
        return ticks / self.timescale


PROFILE_30 = NativeProfile(METHOD, MATCHED_DETAIL_AVC_VERSION, 30000, 1000, POINT)
PROFILE_25 = NativeProfile(NATIVE25_METHOD, MATCHED_DETAIL_NATIVE25_VERSION, 90000, 3600, 0.20)
PROFILE_HEVC25 = NativeProfile(HEVC25_METHOD, MATCHED_DETAIL_HEVC25_VERSION, 90000, 3600, PROFILE_25.point, HEVC)
PROFILES = {p.method: p for p in (PROFILE_30, PROFILE_25, PROFILE_HEVC25)}
VERSIONS = frozenset(p.version for p in PROFILES.values())


def eligible(table, encoding, carrier, profile=PROFILE_30):
    return (
        tuple(encoding) == profile.encoding
        and carrier == "mp4"
        and (table.width, table.height) == (table.coded_width, table.coded_height) == SIZE
        and not (table.rotation or table.interpolated or table.truncated)
        and profile.exact_clock(table.time_base, table.pts)
    )


def input_profile(table, encoding, carrier):
    return next((p for p in PROFILES.values() if eligible(table, encoding, carrier, p)), None)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(raw, lo, hi):
    _require(isinstance(raw, str) and len(raw) <= 30, "Invalid matched-detail reading")
    try:
        value = float.fromhex(raw)
    except (ValueError, OverflowError) as error:
        raise ValueError("Invalid matched-detail reading") from error
    _require(
        value.hex() == raw and math.isfinite(value) and lo <= value <= hi, "Invalid matched-detail reading"
    )
    return value


def _digest(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def anchor_indices(record):
    a, b = record["start_frame_index"], record["end_frame_index"]
    return a - 1, a + (b - a) // 2, b


def face_landmarks(row):
    points = row["landmarks_hex"]
    _require(
        isinstance(points, list)
        and len(points) == 5
        and all(isinstance(p, list) and len(p) == 2 for p in points),
        "Matched-detail view requires five landmarks",
    )
    return [[exact(x, 0, SIZE[0]), exact(y, 0, SIZE[1])] for x, y in points]


def context_box(row):
    box = row["box"]
    _require(
        isinstance(box, dict)
        and set(box) == {"x", "y", "w", "h"}
        and all(type(n) is int for n in box.values()),
        "Matched-detail view has an invalid face box",
    )
    x, y, w, h = (box[k] for k in ("x", "y", "w", "h"))
    _require(
        0 <= x <= SIZE[0] - w and 0 <= y <= SIZE[1] - h and min(w, h) >= 48,
        "Matched-detail face box is outside its native frame",
    )
    return box


def interpolate(before, after, index, a, b):
    left, right, span = b - index, index - a + 1, b - a + 1
    box = {
        key: round(before["box"][key] * left / span + after["box"][key] * right / span)
        for key in ("x", "y", "w", "h")
    }
    xy = [
        [(x * left / span + y * right / span).hex() for x, y in zip(first, second, strict=True)]
        for first, second in zip(face_landmarks(before), face_landmarks(after), strict=True)
    ]
    return box, xy


def validate_clock(source, table_pts=None, *, interpolated=(), rotation=0, truncated=False, profile=PROFILE_30):
    _require(
        (source["width"], source["height"]) == SIZE
        and source["frame_count"] == FRAME_COUNT
        and source["time_base"] == profile.time_base
        and source["timeline_origin_s"] == 0
        and not (interpolated or rotation or truncated)
        and (table_pts is None or profile.exact_clock(source["time_base"], table_pts)),
        "Matched-detail review differs from its verified source clock",
    )


def validate_record(record, source, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    profile = PROFILES.get(record.get("method"))
    _require(profile is not None, "Invalid matched-detail method")
    validate_clock(source, table_pts, interpolated=interpolated, rotation=rotation, truncated=truncated, profile=profile)
    _require(
        set(record)
        == {
            "method",
            "time_base",
            "start_frame_index",
            "end_frame_index",
            "start_pts",
            "end_pts",
            "response_hex",
            "readings_hex",
            "proposal_hex",
            "target_detail_hex",
            "views",
            "scene_cuts",
        }
        and record["time_base"] == profile.time_base,
        "Invalid matched-detail record",
    )
    a, b = record["start_frame_index"], record["end_frame_index"]
    _require(
        type(a) is type(b) is int
        and 6 <= a < b <= 114
        and b - a in (3, 4)
        and (record["start_pts"], record["end_pts"]) == (a * profile.step, b * profile.step),
        "Invalid matched-detail interval",
    )
    cuts = record["scene_cuts"]
    _require(
        isinstance(cuts, list)
        and all(type(cut) is int and 0 < cut < FRAME_COUNT for cut in cuts)
        and cuts == sorted(set(cuts))
        and all(abs(cut - j) > 1 for cut in cuts for j in (a, b)),
        "Matched-detail interval crosses a verified adjoining shot boundary",
    )
    reading = record["readings_hex"]
    _require(set(reading) == set(READINGS), "Invalid matched-detail response arithmetic")
    near, far, flank, inside, entry, exit = (exact(reading[r], -2, 2) for r in READINGS)
    point = min(entry, exit, min(near, far) - max(flank, inside))
    _require(
        min(near, far) >= 0.08
        and 0 <= flank <= 0.33
        and 0 <= inside <= 2
        and entry <= near
        and exit <= far
        and point > profile.point
        and exact(record["response_hex"], profile.point, 2) == point,
        "Matched-detail response differs from the bounded comparison",
    )
    support = record["proposal_hex"]
    _require(set(support) == set(SUPPORT), "Invalid matched-detail proposal support")
    exact(support["detail_drop"], 0.16, 1)
    exact(support["edge_step"], 0.08, 1)
    exact(support["flank_patch_difference"], 0, 10)
    target = exact(record["target_detail_hex"], DETAIL_FLOOR, 255)
    frames = record["views"]
    _require(len(frames) == b - a + 6, "Matched-detail review needs all frame inputs")
    before, after = frames[2], frames[b - a + 3]
    raw = []
    for index, row in zip(range(a - 3, b + 3), frames, strict=True):
        _require(
            set(row)
            == {
                "index",
                "pts",
                "geometry",
                "box",
                "landmarks_hex",
                "decoded_rgb_sha256",
                "aligned_rgb_sha256",
                "matched_rgb_sha256",
                "sigma_hex",
                "raw_detail_hex",
                "matched_detail_hex",
            }
            and row["index"] == index
            and row["pts"] == index * profile.step
            and row["geometry"] == ("interpolated" if a <= index < b else "observed")
            and all(
                _digest(row[k]) for k in ("decoded_rgb_sha256", "aligned_rgb_sha256", "matched_rgb_sha256")
            ),
            "A matched-detail view differs from its native picture",
        )
        context_box(row)
        face_landmarks(row)
        if a <= index < b:
            _require(
                (row["box"], row["landmarks_hex"]) == interpolate(before, after, index, a, b),
                "Interior geometry differs from the measured flanks",
            )
        _require(exact(row["sigma_hex"], 0, 5) in SIGMAS, "Invalid matched-detail smoothing scale")
        original = exact(row["raw_detail_hex"], 0, 255)
        raw.append(original)
        matched = exact(row["matched_detail_hex"], 0, 255)
        _require(
            FRACTION * target <= matched <= target / FRACTION, "Matched detail is outside its agreed range"
        )

    def median(values):
        values = sorted(values)
        k = len(values) // 2
        return values[k] if len(values) % 2 else (values[k - 1] + values[k]) / 2

    _require(
        target == min(median(raw[:3]), median(raw[3:-3]), median(raw[-3:])),
        "The matched detail target differs from the native crop readings",
    )


def coverage(status, reason=None, *, proposed=0, usable=0, profile=PROFILE_30):
    return {
        "status": status,
        "reason": reason,
        "candidate_pairs": proposed,
        "measured_pairs": usable,
        "affects_overall_confidence": False,
        **({"method": profile.method} if profile != PROFILE_30 else {}),
    }


def coverage_text(record):
    return STATUS_TEXT[record["status"]] if record["status"] in STATUS_TEXT else REASONS[record["reason"]]


def validate(document, digest, *, table_pts=None, interpolated=(), rotation=0, truncated=False):
    source = document["source"]
    found = [
        p for p in document["packets"] if KEY in p or p.get("id") == PACKET_ID or p.get("semantics") in PROFILES
    ]
    state = document.get("coverage", {}).get(COVERAGE)
    _require(len(found) <= 1 and (not found or KEY in found[0]), "Invalid matched-detail packet count")
    _require(
        (document["version"] in VERSIONS) == bool(found),
        "The evidence version disagrees with its matched-detail interval",
    )
    record = found[0][KEY] if found else None
    _require(
        (digest(record) if record is not None else None) == source.get(COMMITMENT)
        and (digest(state) if state is not None else None) == source.get(COVERAGE_COMMITMENT),
        "Matched-detail measurements differ from their published commitments",
    )
    if state is None:
        _require(not found, "Matched-detail evidence requires a coverage record")
        return
    status, reason = state["status"], state["reason"]
    n, seen = state["candidate_pairs"], state["measured_pairs"]
    profile = PROFILES.get(state.get("method", METHOD))
    _require(
        profile is not None
        and set(state) == set(coverage("unassessed", profile=profile))
        and state["affects_overall_confidence"] is False
        and status in {"localized", "no_match", "unassessed"}
        and ((reason in REASONS) if status == "unassessed" else reason is None)
        and (status == "localized") == bool(found)
        and type(n) is type(seen) is int
        and 0 <= seen <= n <= MAX_PAIRS
        and (status == "unassessed" or seen > 0),
        "Invalid matched-detail coverage",
    )
    if status != "unassessed" or n or profile != PROFILE_30:
        validate_clock(source, table_pts, interpolated=interpolated, rotation=rotation, truncated=truncated, profile=profile)
    if not found:
        return
    validate_record(
        record, source, table_pts=table_pts, interpolated=interpolated, rotation=rotation, truncated=truncated
    )
    packet = found[0]
    _require(
        packet["id"] == PACKET_ID
        and packet["semantics"] == record["method"] == profile.method
        and document["version"] == profile.version
        and packet["title"] == TITLE
        and packet["claim_scope"] == "sampled_window"
        and packet["affects_overall_confidence"] is False
        and packet["observation"] == OBSERVATION
        and packet["interpretation"] == INTERPRETATION
        and packet["limitations"] == [LIMITATION]
        and packet["signal_refs"] == []
        and packet["finding_ids"] == [],
        "Matched-detail evidence must present a confidence-neutral source interval",
    )
    rows = {item["index"]: item for item in record["views"]}
    _require(len(packet["frames"]) == 3, "Matched-detail interval requires three source images")
    for index, frame, role in zip(anchor_indices(record), packet["frames"], ROLES, strict=True):
        item = rows[index]
        anchor = frame["anchor"]
        refs = frame["artifact_refs"]
        _require(
            frame["id"] == f"{PACKET_ID}.{role}"
            and frame["interval_role"] == role
            and frame["context_region"] == item["box"]
            and frame["region_meaning"] == "detected_face_context"
            and frame["score"] is None
            and frame["is_peak"] is False
            and anchor["frame_index"] == index
            and anchor["pts"] == item["pts"]
            and anchor["time_base"] == profile.time_base
            and anchor["decoded_rgb_sha256"] == item["decoded_rgb_sha256"]
            and anchor["original_sha256"] == source["sha256"]
            and anchor["stream_selector"] == "v:0"
            and (anchor["width"], anchor["height"]) == SIZE
            and abs(anchor["stream_time_s"] - profile.seconds(item["pts"])) <= 1e-6
            and abs(anchor["playback_time_s"] - profile.seconds(item["pts"])) <= 1e-6
            and set(refs) == {"original", "context", "crop"}
            and set(refs.values()) <= packet["artifacts"].keys(),
            "Displayed matched-detail frame differs from its native witness",
        )
