"""Validate the public interval evidence without loading detector dependencies."""
from __future__ import annotations

from fractions import Fraction
from statistics import mean

METHOD = "face-track-response/1"
MAX_GAP_S = 0.8
MAX_INTERVAL_S = 4.0
CONTEXT_COUNT = 2
MIN_CORE_COUNT = 3


def validate_packet(packet: dict, source: dict, *, table_pts: tuple[int, ...] | None = None, interpolated: tuple[int, ...] = ()) -> None:
    interval = packet.get("face_interval")
    if interval is None and packet.get("semantics") != METHOD:
        return
    if (not isinstance(interval, dict) or packet.get("semantics") != METHOD
            or interval.get("method") != METHOD or interval.get("time_base") != source["time_base"]):
        raise ValueError("Face interval has an invalid method or source clock")
    before, during, after = (interval["samples"][role] for role in ("before", "during", "after"))
    if len(before) != CONTEXT_COUNT or len(after) != CONTEXT_COUNT or len(during) < MIN_CORE_COUNT:
        raise ValueError("Face interval lacks measured context or support")
    support = [*before, *during, *after]
    indices = [s["frame_index"] for s in support]
    pts = [s["pts"] for s in support]
    base = Fraction(source["time_base"])
    if (indices != sorted(set(indices)) or any(not 0 <= i < source["frame_count"] for i in indices)
            or pts != sorted(set(pts)) or any(float((b-a)*base) > MAX_GAP_S + 1e-6 for a, b in zip(pts, pts[1:]))
            or any(not 0 <= s["score"] <= 1 for s in support)
            or (table_pts is not None and any(table_pts[i] != p or i in interpolated for i, p in zip(indices, pts, strict=True)))):
        raise ValueError("Face interval has inconsistent frame support")
    a, b = during[0]["pts"], during[-1]["pts"]
    if ((interval["start_pts"], interval["end_pts"]) != (a, b)
            or abs(interval["start_stream_s"] - float(a * base)) > 1e-6
            or abs(interval["end_stream_s"] - float(b * base)) > 1e-6):
        raise ValueError("Face interval does not match the first and last supported samples")
    lift = mean(s["score"] for s in during) - max(mean(s["score"] for s in before), mean(s["score"] for s in after))
    if float((b-a)*base) > MAX_INTERVAL_S + 1e-6 or not 0 < interval["lift"] <= 1 or abs(lift - interval["lift"]) > 1e-6:
        raise ValueError("Face interval has an invalid response difference")
    peak = max(during, key=lambda s: s["score"])
    expected = [("before", before[-1]), ("during", peak), ("after", after[0])]
    if len(packet["frames"]) != len(expected):
        raise ValueError("Face interval must show before, during and after frames")
    for frame, (role, sample) in zip(packet["frames"], expected, strict=True):
        anchor = frame["anchor"]
        if (frame.get("interval_role") != role or anchor["frame_index"] != sample["frame_index"]
                or anchor["pts"] != sample["pts"] or abs(frame["score"] - sample["score"]) > 1e-6
                or not frame.get("context_region") or frame.get("is_peak") != (role == "during")):
            raise ValueError("Face interval image differs from its measured role")
