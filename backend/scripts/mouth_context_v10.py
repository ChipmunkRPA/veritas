"""Source-relative raw margins, enumerated on the actual production video clock."""

from __future__ import annotations

import math

import numpy as np

METHOD = "mouth-surrounding-context/10.0"


def windows(centers, tracks, values, table):
    result = []
    for track, samples in tracks.items():
        indices = {r.frame_index for r in samples}
        eligible = [
            i
            for i in sorted(centers)
            if all(k in indices and k in values for k in (i - 1, i, i + 1))
            and table.seconds(i + 1) - table.seconds(i - 1) <= 0.13
        ]
        if not eligible:
            continue
        # Algebraically identical M-L/M-R with less cancellation from a
        # large common raw offset. This anchor has no authenticity meaning.
        anchor = values[eligible[0]]
        blocks = [(i, float(np.mean([values[k] - anchor for k in (i - 1, i, i + 1)]))) for i in eligible]
        if len(blocks) < 7:
            continue
        times = np.array([table.seconds(i) for i, _ in blocks])
        responses = np.array([v for _, v in blocks])
        for start in range(2, len(blocks) - 4):
            for end in range(start + 3, len(blocks) - 1):
                span = times[end - 1] - times[start]
                if span > 2.6:
                    break
                if span < 0.52 or np.max(np.diff(times[start - 2 : end + 2])) > 0.55:
                    continue
                middle = float(responses[start:end].mean())
                left = float(responses[start - 2 : start].mean())
                right = float(responses[end : end + 2].mean())
                response = min(middle - left, middle - right) - 0.25 * abs(left - right)
                result.append(
                    {
                        "track_id": track,
                        "start_s": float(times[start]),
                        "end_s": float(times[end - 1]),
                        "response": response,
                        "middle_margin": middle + anchor,
                        "before_margin": left + anchor,
                        "after_margin": right + anchor,
                        "reference_margin": float(anchor),
                        "centers": [i for i, _ in blocks[start - 2 : end + 2]],
                    }
                )
    return sorted(result, key=lambda w: (-w["response"], w["start_s"], w["track_id"]))


def point(rows, arm):
    expected = {"original", "local-light", "local-resample", "muse-resample", "local-blur-2", "local-blur-4"}
    groups = {r["group"] for r in rows}
    if len(rows) != 24 or len(groups) != 4 or len({r["id"] for r in rows}) != 24:
        raise ValueError("Exactly all24 calibration cases/four point actors are required")
    if any({r["condition"] for r in rows if r["group"] == g} != expected for g in groups):
        raise ValueError("Calibration actor is missing an ordinary condition")
    if any(
        r["label"] != 0
        or r["role"] != "point"
        or r["status"] != "ready"
        or r["heads"][arm]["status"] != "ok"
        or not r["heads"][arm]["probe_windows"]
        for r in rows
    ):
        raise ValueError("Every calibration input must be ordinary, separately reserved and assessed")
    return math.nextafter(
        max(0.0, max(r["heads"][arm]["probe_windows"][0]["response"] for r in rows)), math.inf
    )


def selected_signal(absolute, absolute_point, relative, relative_point):
    if absolute and absolute[0]["response"] >= absolute_point:
        return "absolute"
    if relative and relative[0]["response"] >= relative_point:
        return "relative"
    return None
