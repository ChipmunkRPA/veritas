# SPDX-License-Identifier: Apache-2.0
"""Validate a source-bound stereo delay measurement without analysis imports."""

from app.evidence.audio_channel_alignment_binding import (
    NATIVE_FACTOR,
    NATIVE_RATE,
    aac_presented_samples,
    native_scope,
)
from app.evidence.audio_repetition_binding import _sample_range, _valid_hash

KEY = "stereo_timing_change"
INPUT = "stereo_timing_change_pcm_sha256"
SCHEMA = "chipmunk.stereo-timing-change/1"
CLIP_SAMPLES = 8640
ROLES = ("before", "during", "after")


def extent(source):
    if native_scope(source):
        return source["samples"] * NATIVE_FACTOR
    return aac_presented_samples(source)


def ranges(start, stop, count):
    return ((max(0, start - CLIP_SAMPLES), start), (start, stop), (stop, min(count, stop + CLIP_SAMPLES)))


def validate(record, source):
    try:
        _validate(record, source)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Invalid stereo timing change evidence") from error


def _validate(record, source):
    count = extent(source)
    if (count is None or record["input_samples"] != count or record["sample_rate"] != NATIVE_RATE
            or record["schema"] != SCHEMA or record["source_sha256"] != source["sha256"]
            or not _valid_hash(record["stereo_pcm_sha256"]) or record["affects_overall_confidence"] is not False
            or record["status"] not in {"measured", "unavailable"}):
        raise ValueError("Stereo timing change differs from its source or status")
    measured = record["measurement"]
    if measured is None:
        if record["clips"] != [] or record["artifacts"] != {}:
            raise ValueError("Stereo timing evidence has no measured interval")
        return
    start, stop = measured["start_sample"], measured["end_sample"]
    before, during, after = [measured[f"{role}_offset_samples"] for role in ("before", "inside", "after")]
    votes = measured["support"]
    if (record["status"] != "measured" or any(type(i) is not int for i in (start, stop, before, during, after))
            or not 0 < start < stop < count or not all(abs(shift) < 36 for shift in (before, during, after))
            or abs(before - after) > 3 or min(abs(during - before), abs(during - after)) < 8
            or type(votes) is not list or len(votes) != 3 or any(type(v) is not int for v in votes)
            or not (5 <= votes[0] <= 8 and 6 <= votes[1] <= 51 and 5 <= votes[2] <= 8)
            or type(measured["score"]) not in (int, float) or not 0 < measured["score"] < 72):
        raise ValueError("Stereo timing measurements are outside the supported range")
    source_pts = source["timeline"][0]["source_pts"] * NATIVE_FACTOR
    for clip, role, bounds in zip(record["clips"], ROLES, ranges(start, stop, count), strict=True):
        native = _sample_range(clip, samples=count, source_pts=source_pts, time_base=f"1/{NATIVE_RATE}")
        asset = record["artifacts"][role]
        if (clip["role"] != role or native != bounds or not _valid_hash(clip["pcm_s16le_sha256"])
                or not _valid_hash(asset["sha256"]) or asset["media_type"] != "audio/wav"
                or asset["samples"] != bounds[1] - bounds[0] or asset["sample_rate"] != NATIVE_RATE or asset["channels"] != 2):
            raise ValueError("Stereo timing excerpt differs from its original sample interval")
    if set(record["artifacts"]) != {*ROLES, "timing_plot", "waveform"}:
        raise ValueError("Stereo timing evidence requires both original plots")
    for name in ("timing_plot", "waveform"):
        asset = record["artifacts"][name]
        if not _valid_hash(asset["sha256"]) or (asset["media_type"], asset["width"], asset["height"]) != ("image/png", 960, 280):
            raise ValueError("Stereo timing evidence has an invalid plot")
