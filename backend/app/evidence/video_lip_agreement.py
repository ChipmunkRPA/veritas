# SPDX-License-Identifier: Apache-2.0
"""Validate two local speech and mouth measurements against the source clocks."""

from __future__ import annotations

from fractions import Fraction

KEY = "motion_sound_agreement"
TITLE = "Local sound and mouth movement mismatch"
LIMITATION = (
    "Dubbing, recording delays, offscreen speech, and ordinary variation in pronunciation can affect this check. "
    "It carries zero weight in the overall AI assessment."
)
RATE = 16_000
UNITS = 10_000_000
POINTS = {"pcm": (5306971, 5554583), "aac48": (5318680, 5740689)}
WEIGHT_PINS = {
    "vision": "bc58acf5926044b6680e7b1c5588f67894b1cbdd43bb6dfe54ff25c5b6ab0a70",
    "shared": "d924e379f8a6a2d861401a8f25504c73012c4ab50d37e04b5aac5ecb19d9bbd4",
    "phonetic": "2432df9a68dcf5a78c31be3b9f1a708bf7ca79097435c26b55baafc8de19dfab",
}


def carrier(mapping, *, origin, stereo_scope):
    if origin != 0 or stereo_scope is not None or mapping.get("source_channels") != 1:
        return None
    return {("pcm_s16le", 16_000): "pcm", ("aac", 48_000): "aac48"}.get(
        (mapping.get("source_codec"), mapping.get("source_sample_rate"))
    )


def encode(candidate):
    if candidate is None:
        return None
    return {key: value for key, value in candidate.items() if key not in {"score", "agreement"}} | {
        "score_units": round(candidate["score"] * UNITS),
        "agreement_units": [round(value * UNITS) for value in candidate["agreement"]],
    }


def _proposal(row, frames, samples):
    if row is None:
        return
    if (
        set(row)
        != {"picture_span", "span", "bracket_frames", "lag_samples", "score_units", "agreement_units"}
        or any(type(row[key]) is not int for key in ("lag_samples", "score_units"))
        or any(
            not isinstance(row[k], list) or len(row[k]) != n or any(type(v) is not int for v in row[k])
            for k, n in (("picture_span", 2), ("span", 2), ("bracket_frames", 3), ("agreement_units", 3))
        )
    ):
        raise ValueError("Invalid local sound and mouth comparison")
    begin, end = row["picture_span"]
    a, b = row["span"]
    before, during, after = row["bracket_frames"]
    lag = row["lag_samples"]
    left, center, right = row["agreement_units"]
    if (
        not 0 <= before < begin <= during < end <= after < len(frames)
        or not 0 <= a < b <= samples
        or [a, b] != [frames[begin]["sample"] + lag, frames[end]["sample"] + lag]
        or not -1920 <= lag <= 1920
        or lag % 320
        or any(frames[i]["gap_units"] is None for i in (before, during, after))
        or not all(-UNITS <= value <= UNITS for value in (left, center, right))
        or not 0 < row["score_units"] <= 2 * UNITS
        or abs(min(left, right) - center - row["score_units"]) > 3
    ):
        raise ValueError("Local sound and mouth evidence disagrees with the native timestamps")


def confirmed(records, file_type, frames, samples):
    if file_type not in POINTS or not isinstance(records, list) or len(records) != 2:
        raise ValueError("Unexpected audio profile or incomplete local comparisons")
    for row in records:
        _proposal(row, frames, samples)
    first, second = records
    if any(
        row is None or row["score_units"] <= point
        for row, point in zip(records, POINTS[file_type], strict=True)
    ):
        return None
    a, b = first["span"]
    c, d = second["span"]
    overlap = max(0, min(b, d) - max(a, c))
    if 5 * overlap < 2 * ((b - a) + (d - c) - overlap):
        return None
    return first


def record(candidate, frames):
    begin, end = candidate["picture_span"]
    return {
        "audio_source_pts": candidate["span"],
        "picture_source_samples": [frames[begin]["sample"], frames[end]["sample"]],
        "baseline_lag_samples": candidate["lag_samples"],
        "change_samples": 0,
        "picture_frames": dict(zip(("before", "during", "after"), candidate["bracket_frames"], strict=True)),
        KEY: {"agreement_units": candidate["agreement_units"]},
    }


def observation(witness):
    a, b = (sample / RATE for sample in witness["audio_source_pts"])
    start, end = (sample / RATE for sample in witness["picture_source_samples"])
    return (
        f"Spoken sound and mouth movement agreed less in the highlighted interval than in the surrounding speech. "
        f"Audio source time {a:.3f}–{b:.3f} s. Corresponding picture time {start:.3f}–{end:.3f} s."
    )


def source_frames(times):
    return [{"sample": at, "gap_units": 0} for at in times]


def eligible_picture_clock(frames, time_base):
    tb = Fraction(time_base)
    return (
        len(frames) >= 34
        and frames[0]["sample"] == 0
        and all(
            Fraction(1, 28) <= (b["pts"] - a["pts"]) * tb <= Fraction(1, 22)
            for a, b in zip(frames, frames[1:])
        )
    )
