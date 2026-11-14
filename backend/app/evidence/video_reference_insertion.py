# SPDX-License-Identifier: Apache-2.0
"""Locate additional video pictures between two reliably matched ranges."""

from __future__ import annotations

import hashlib
import subprocess
from fractions import Fraction
from itertools import pairwise

THUMB_SIDE = 96
THUMB_BYTES = THUMB_SIDE ** 2
EVIDENCE_ROLES = ("before", "during", "after")
RETENTION_WINDOW = 8
MAX_RETENTION_MEDIAN = .035
MAX_ALTERNATE_GAIN = .020
INTERPRETATION = (
    "The pictures before and after this interval match the earlier clip closely. "
    "The analyzed clip contains additional pictures between them. Ordinary edits can also add pictures. "
    "This comparison covers the picture track and carries no weight in the AI estimate."
)
RETENTION_INTERPRETATION = (
    "One interval could account for the longer picture track, but sections outside it did not match consistently. "
    "The comparison could not verify one continuous insertion. The soundtrack was outside this comparison."
)


def frame_error(first, second):
    import numpy as np

    return np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32)), axis=(1, 2))


def normalized(pixels):
    import numpy as np

    luma = pixels.astype(np.float32)
    low, mid, high = np.percentile(luma, (20, 50, 80), axis=(1, 2), keepdims=True)
    return ((luma - mid) / np.maximum(high - low, 24)).astype(np.float32)


def retention_support(earlier, submitted, selected):
    """Check local agreement across both sections that should have been retained."""
    import numpy as np

    start, stop = selected
    before, after = normalized(earlier), normalized(submitted)
    difference = stop - start
    costs = np.stack([frame_error(before, after[offset:offset + len(before)])
                      for offset in range(difference + 1)], axis=1)
    support = []
    for flank, offset in ((costs[:start], 0), (costs[start:], difference)):
        windows = np.lib.stride_tricks.sliding_window_view(flank, RETENTION_WINDOW, axis=0)
        typical = np.median(windows[:, offset, :], axis=-1)
        mean = windows.mean(axis=-1)
        support.append({"median": round(float(typical.max()), 6),
                        "alternate_gain": round(float((mean[:, offset] - mean.min(axis=1)).max()), 6)})
    return support


def retained_pictures_agree(support):
    return all(side["median"] <= MAX_RETENTION_MEDIAN and side["alternate_gain"] <= MAX_ALTERNATE_GAIN for side in support)


def pair_interval(first, second):
    import numpy as np

    added = len(second) - len(first)
    if not 24 <= added <= 240 or len(first) < 49:
        return {"status": "not_a_longer_pair", "bounds": None}
    a, b = normalized(first), normalized(second)
    before = frame_error(a, b[:len(a)])
    after = frame_error(a, b[added:])
    positions = np.arange(24, len(a) - 23)
    profile = np.r_[0., np.cumsum(before - after, dtype=np.float64)]
    best = int(positions[np.argmin(profile[positions])])
    before_stats, after_stats = ([round(float(value), 6) for value in np.percentile(values, (50, 90))]
                                 for values in (before[:best], after[best:]))
    close = positions[profile[positions] <= profile[best] + 0.035]
    selected = [best, best + added]
    details = {"best": selected, "uncertain_starts": [int(close.min()), int(close.max())],
               "before_error_p50_p90": before_stats, "after_error_p50_p90": after_stats}
    if any(mid > .10 or upper > .22 for mid, upper in (before_stats, after_stats)):
        return {"status": "insufficient_matching_frames", "bounds": None, **details}
    if int(close.max() - close.min()) > 3:
        return {"status": "ambiguous_insertion", "bounds": None, **details}
    return {"status": "paired_insertion", "bounds": selected, **details}


def eligible(paths, tables):
    from app.detectors.video.decode import DecodeError
    from app.evidence.video_reference_local import _carrier

    earlier, submitted = tables
    if (len(earlier) < 49 or not 24 <= len(submitted) - len(earlier) <= 240
            or any(table.truncated or table.interpolated or table.rotation
                   or max(table.width, table.height) != 480 or table.pts[0] != 0
                   or any((b - a) * table.tb != Fraction(1, 24) for a, b in pairwise(table.pts))
                   for table in tables)):
        return False
    if (earlier.width, earlier.height) != (submitted.width, submitted.height):
        return False
    try:
        return all(_carrier(path) == [("h264", "yuv420p")] for path in paths)
    except (DecodeError, OSError, ValueError, subprocess.SubprocessError):
        return False


def _luma(path, table, max_pixels):
    import numpy as np

    from app.detectors.video.decode import DecodeError, _input, _tool

    def decode(fmt):
        cmd = [_tool("ffmpeg"), "-v", "error", "-nostdin", "-threads", "2", "-filter_threads", "1",
               "-copyts", "-max_pixels", str(max_pixels), "-protocol_whitelist", "file", "-i", _input(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
               "-vf", f"scale={THUMB_SIDE}:{THUMB_SIDE}:flags=area,format=gray"]
        try:
            return subprocess.run([*cmd, "-frames:v", str(len(table) + 1), "-threads", "1", *fmt, "pipe:1"], capture_output=True,
                                  check=True, timeout=30, stdin=subprocess.DEVNULL).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise DecodeError("Video could not be decoded for insertion comparison.") from error

    raw = decode(["-f", "rawvideo"])
    framehash = decode(["-enc_time_base", table.time_base, "-f", "framehash", "-hash", "sha256"])
    lines = framehash.decode("ascii").splitlines()
    rows = [line.replace(" ", "").split(",") for line in lines if line and not line.startswith("#")]
    if (f"#tb 0: {table.time_base}" not in lines or len(rows) != len(table)
            or len(raw) != THUMB_BYTES * len(table)):
        raise DecodeError("Video could not be decoded with every recorded picture and timestamp.")
    for index, fields in enumerate(rows):
        block = raw[index * THUMB_BYTES:(index + 1) * THUMB_BYTES]
        if (len(fields) != 6 or fields[0] != "0" or fields[2] != str(table.pts[index])
                or fields[4] != str(THUMB_BYTES) or fields[5] != hashlib.sha256(block).hexdigest()):
            raise DecodeError("Video could not be decoded with verified picture timestamps.")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, THUMB_SIDE, THUMB_SIDE)


def locate(paths, tables, max_pixels, *, verify_retention=False):
    from app.detectors.video.decode import DecodeError

    if not eligible(paths, tables):
        return None
    try:
        first, second = (_luma(path, table, max_pixels) for path, table in zip(paths, tables, strict=True))
    except DecodeError:
        return None
    result = pair_interval(first, second)
    if result["bounds"] is None:
        return None
    if verify_retention:
        support = retention_support(first, second, result["bounds"])
        result["retention"] = {"window_frames": RETENTION_WINDOW, "before": support[0], "after": support[1], "audio_checked": False}
        if not retained_pictures_agree(support):
            result.update(status="insertion_retention_uncertain", bounds=None)
    return result
