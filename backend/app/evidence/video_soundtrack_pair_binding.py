"""Portable verification for timing evidence from two tracks in one video."""
from __future__ import annotations

import math
import re
from copy import deepcopy

from app.core.canonical import canonical_hash

KEY = "soundtrack_pair"
COMMITMENT = "video_soundtrack_pair_sha256"
SCHEMA = "chipmunk.video-soundtrack-pair/1"
RATE = 16_000
LIMITATION = ("The check compares the first and second audio streams. It cannot tell which stream changed or whether either "
              "one matches the speaker's mouth. Dubbing, music and ordinary editing can affect the result. "
              "It carries zero weight in the AI estimate.")
REASONS = {
    "duration_outside_scope": "This check covers soundtracks from 1.5 to 12 seconds.",
    "second_stream_unavailable": "The second audio track could not be decoded on a verified sample clock.",
    "unsupported_clock": "This check needs two continuous audio tracks with the same start and duration, from 1.5 to 12 seconds.",
    "insufficient_same_time_support": "The tracks shared too little distinctive sound at the same times to assess a local shift.",
    "no_local_shift": "No qualifying local shift was found within the matched audio windows.",
    "local_shift": "This excerpt occurs at different times in the two soundtracks. Other matched sound is aligned on both sides.",
}
_SHA = re.compile(r"[0-9a-f]{64}")


def commitment(doc):
    if doc is None:
        return None
    data = deepcopy(doc)
    for asset in assets(data).values():
        asset.pop("url", None)
    return canonical_hash(data)


def assets(doc):
    return {f"stream_{track['stream_index']}_{role}": track[role]
            for track in (doc.get("location") or {}).get("excerpts", []) for role in ("clip", "waveform")}


def validate(doc, source, expected):
    if doc is None and expected is None:
        return
    try:
        _validate(doc, source, expected)
    except (TypeError, KeyError, AttributeError, IndexError) as error:
        raise ValueError("Soundtrack comparison contains missing or invalid fields") from error


def _validate(doc, source, expected):
    status = doc["status"]
    location = doc["location"]
    coverage = doc["coverage"]
    streams = doc["stream_indices"]
    if (commitment(doc) != expected or doc["schema"] != SCHEMA or doc["source_sha256"] != source["sha256"]
            or doc["primary_pcm_sha256"] != source["pcm_sha256"] or streams != source["audio_stream_indices"]
            or len(streams) != 2 or doc["affects_overall_confidence"] is not False or doc["limitation"] != LIMITATION
            or status not in REASONS or (location is not None) != (status == "local_shift")):
        raise ValueError("Soundtrack comparison differs from its source or recorded disposition")
    in_scope = 24000 <= source["samples"] <= 12 * RATE
    if (status == "duration_outside_scope") == in_scope:
        raise ValueError("Soundtrack comparison duration differs from the committed samples")
    if status in {"duration_outside_scope", "second_stream_unavailable"}:
        if "secondary_pcm_sha256" in doc or coverage:
            raise ValueError("An unreadable soundtrack has unexplained measurements")
        return
    if not _SHA.fullmatch(doc["secondary_pcm_sha256"]):
        raise ValueError("The second soundtrack lacks a sample commitment")
    if status == "unsupported_clock":
        if coverage:
            raise ValueError("Unverified soundtrack clocks have unexplained overlap measurements")
        return
    names = ("candidate_windows", "matched_windows", "same_time_windows", "sections_with_same_time_matches")
    total, matched, same, sections = (coverage[key] for key in names)
    if (set(coverage) != set(names) or not all(type(value) is int for value in (total, matched, same, sections))
            or not 0 <= sections <= min(3, same) or not 0 <= same <= matched <= total
            or total != 1 + (source["samples"] - 1280) // 320
            or (status == "insufficient_same_time_support") == (matched >= 8 and sections == 3 and 20 * same >= 11 * matched)):
        raise ValueError("Invalid matched-sound coverage")
    if location is None:
        return
    if len(location["excerpts"]) != 2:
        raise ValueError("Soundtrack comparison needs one excerpt per stream")
    first, second = location["excerpts"]
    shift, witnesses, score = (location[name] for name in ("shift_samples", "supporting_windows", "minimum_correlation"))
    if (not all(type(value) is int for value in (shift, witnesses)) or not 1920 <= abs(shift) <= 5760
            or not 4 <= witnesses <= matched - same or type(score) not in (int, float) or not math.isfinite(score)
            or not .86 <= score <= 1 or len(source["timeline"]) != 1
            or second["start_sample"] - first["start_sample"] != shift or second["end_sample"] - first["end_sample"] != shift):
        raise ValueError("Invalid local soundtrack observation")
    start_pts = source["timeline"][0]["source_pts"]
    for index, span in zip(streams, (first, second), strict=True):
        a, b = span["start_sample"], span["end_sample"]
        if (span["stream_index"] != index or type(a) is not int or type(b) is not int or not 0 <= a < b <= source["samples"]
                or b-a < 2240 or span["source_timestamp_ranges"] != [{"start_sample": a, "end_sample": b,
                "source_start_pts": start_pts+a, "source_end_pts": start_pts+b, "time_base": "1/16000"}]):
            raise ValueError("Soundtrack excerpt differs from its stream or timestamp")
        clip, waveform = span["clip"], span["waveform"]
        if (not all(_SHA.fullmatch(asset["sha256"]) for asset in (clip, waveform))
                or tuple(clip[key] for key in ("media_type", "sample_rate", "channels", "samples")) != ("audio/wav", RATE, 1, b-a)
                or tuple(waveform[key] for key in ("media_type", "width", "height")) != ("image/png", 960, 280)):
            raise ValueError("Soundtrack excerpt has invalid listening assets")
