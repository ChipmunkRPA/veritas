"""Look for an exact native RGB picture held across a bounded opening sequence."""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from subprocess import TimeoutExpired

from app.detectors.video.decode import DecodeError, FrameTable, decode_frames
from app.evidence import picture_hold_binding as binding

DECODE_TIMEOUT_S = 8.0


def inspect(path: Path, table: FrameTable, *, origin: float | None, check: Callable[[], None] | None = None,
            remaining_s: float | None = None) -> dict:
    """Return the longest verified exact hold and disclose the opening-prefix budget."""
    limit = binding.frame_limit(len(table), table.width, table.height)
    tb = table.tb
    scanned, last_pts, previous_digest = 0, None, None
    continuous_start, equal_start, longest = 0, 0, 0
    best = None
    timeout = DECODE_TIMEOUT_S if remaining_s is None else min(DECODE_TIMEOUT_S, remaining_s - 1.0)
    if timeout < 0.5 and limit:
        return {"coverage": binding.coverage("unavailable", binding.INCOMPLETE, table.time_base, limit=limit), "match": None}
    try:
        with closing(decode_frames(path, table, range(limit), check=check, timeout_s=timeout)) as frames:
            for frame in frames:
                if frame.index != scanned or frame.rgb.shape != (table.height, table.width, 3):
                    break
                scanned += 1
                point = frame.pts if frame.pts_verified else None
                if point is None or last_pts is None or not 0 < (point - last_pts) * tb <= binding.MAX_STEP:
                    continuous_start, previous_digest = frame.index, None
                last_pts = point
                if point is None:
                    continue
                longest = max(longest, point - table.pts[continuous_start])
                digest = hashlib.sha256(frame.rgb).hexdigest()
                if digest != previous_digest:
                    equal_start, previous_digest = frame.index, digest
                elapsed = point - table.pts[equal_start]
                if elapsed * tb >= binding.MIN_SPAN and (best is None or elapsed > best[1] - table.pts[best[0]]):
                    best = equal_start, point, frame.index, digest
    except (DecodeError, TimeoutExpired):
        pass  # An auxiliary evidence check has no authority to fail the main video analysis.
    match = None
    if scanned != limit:
        status, reason = "unavailable", binding.INCOMPLETE
    elif best is not None:
        start, end_pts, end, digest = best
        start_pts = table.pts[start]
        start_s, end_s = float(start_pts * tb), float(end_pts * tb)
        match = {"start_frame_index": start, "end_frame_index": end, "start_pts": start_pts, "end_pts": end_pts,
                 "time_base": table.time_base, "start_stream_s": start_s, "end_stream_s": end_s,
                 "start_playback_s": None if origin is None else start_s - origin,
                 "end_playback_s": None if origin is None else end_s - origin,
                 "span_s": float((end_pts - start_pts) * tb), "frame_count": end - start + 1,
                 "decoded_rgb_sha256": digest, "pts": list(table.pts[start:end + 1])}
        status, reason = "matched", binding.MATCH
    elif longest * tb >= binding.MIN_SPAN:
        status, reason = "no_match", binding.NO_MATCH
    else:
        status, reason = "unavailable", binding.TOO_SHORT
    return {"coverage": binding.coverage(status, reason, table.time_base, limit=limit, scanned=scanned, last_pts=last_pts),
            "match": match}
