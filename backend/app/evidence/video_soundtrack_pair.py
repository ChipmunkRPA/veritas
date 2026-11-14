"""Measure local timing differences between two soundtracks in one video."""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256

from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, source_ranges, wav_bytes
from app.evidence import video_reference_soundtrack as timing
from app.evidence.video_soundtrack_pair_binding import LIMITATION, SCHEMA

MAX_SECONDS = 12
DECODE_SECONDS = MAX_SECONDS + 1
MIN_SAME_TIME_FRACTION = Fraction(11, 20)


def scope(primary, secondary):
    """Both tracks must have the same continuous and bounded source clock."""
    mappings = primary.mapping, secondary.mapping
    streams = mappings[0].get("audio_stream_indices", [])
    if (len(streams) != 2 or mappings[1].get("audio_stream_indices") != streams
            or [item["source_stream_index"] for item in mappings] != streams
            or any(len(item["timeline"]) != 1 for item in mappings)):
        return False
    starts = [item["timeline"][0]["source_pts"] for item in mappings]
    counts = [len(track.pcm) for track in (primary, secondary)]
    return (starts[0] == starts[1] and counts[0] == counts[1]
            and timing.MIN_SAMPLES <= counts[0] <= MAX_SECONDS * SAMPLE_RATE)


def measure(primary, secondary, *, require_shared_track=True):
    """Keep local candidates with two same-time flanks and clip-wide support."""
    if not scope(primary, secondary):
        return {"status": "unsupported_clock"}
    matches = list(timing._matches(primary.pcm, secondary.pcm, window=timing.SHORT_WINDOW,
                                  step=timing.SHORT_STEP, min_rms=timing.QUIET_RMS))
    assessed = [(start, lag) for start, lag, _ in matches if lag is not None]
    same_time = [start for start, lag in assessed if abs(lag) <= timing.STABLE]
    thirds = {min(2, 3 * start // len(primary.pcm)) for start in same_time}
    coverage = {"candidate_windows": len(matches), "matched_windows": len(assessed), "same_time_windows": len(same_time),
                "sections_with_same_time_matches": len(thirds)}
    if (len(assessed) < 2 * timing.FLANK_WINDOWS + timing.MIN_WINDOWS
            or require_shared_track and (len(thirds) != 3 or len(same_time) < MIN_SAME_TIME_FRACTION * len(assessed))):
        return {"status": "insufficient_same_time_support", **coverage}
    span = min(timing._brackets(matches, window=timing.SHORT_WINDOW, step=timing.SHORT_STEP),
               key=lambda item: (-item["supporting_windows"], item["submitted_samples"][0]), default=None)
    return {"status": "local_shift" if span else "no_local_shift", **coverage, "range": span}


def decode(path, source_sha256):
    return decode_audio(path, source_sha256=source_sha256, stream_position=1, max_seconds=DECODE_SECONDS)


def analyze(path, source_sha256, primary):
    if not timing.MIN_SAMPLES <= len(primary.pcm) <= MAX_SECONDS * SAMPLE_RATE:
        return {"status": "duration_outside_scope"}, None
    try:
        secondary = decode(path, source_sha256)
    except (ValueError, RuntimeError):
        return {"status": "second_stream_unavailable"}, None
    return measure(primary, secondary), secondary


def describe(source, reading, primary, secondary, make_asset):
    """Use the recorded source clocks for the two listening excerpts."""
    from app.evidence.audio_packets import _chart

    span = reading.get("range")
    doc = {"schema": SCHEMA, "status": reading["status"], "source_sha256": source["sha256"],
           "stream_indices": source["audio_stream_indices"], "primary_pcm_sha256": source["pcm_sha256"],
           "affects_overall_confidence": False, "limitation": LIMITATION, "location": None,
           "coverage": {key: value for key, value in reading.items() if key not in {"status", "range"}}}
    if secondary is not None:
        doc["secondary_pcm_sha256"] = secondary.mapping["pcm_sha256"]
    if span is None:
        return doc
    excerpts = []
    for track, label, (start, end) in zip((primary, secondary), ("First", "Second"),
                                         (span["earlier_samples"], span["submitted_samples"]), strict=True):
        clock = source_ranges(track.mapping, start, end)
        waveform = _chart(track.pcm, [], start=start, end=end, selected=None, scores=False, time_precision=3,
                          title=f"{label} audio stream waveform", source_offset=clock[0]["source_start_pts"]-start)
        excerpts.append({"stream_index": track.mapping["source_stream_index"], "start_sample": start, "end_sample": end,
                         "source_timestamp_ranges": clock,
                         "clip": make_asset(wav_bytes(track.pcm[start:end]), "audio/wav", samples=end-start,
                                            sample_rate=SAMPLE_RATE, channels=1),
                         "waveform": make_asset(waveform, "image/png", width=960, height=280)})
    doc["location"] = {"shift_samples": span["shift_samples"], "supporting_windows": span["supporting_windows"],
                       "minimum_correlation": span["minimum_correlation"], "excerpts": excerpts}
    return doc


def digested_asset(data, media_type, **metadata):
    return {"sha256": sha256(data).hexdigest(), "media_type": media_type, **metadata}
