"""Build and replay a descriptive repetition check on the frozen mono input."""

from __future__ import annotations

import hashlib
from copy import deepcopy

import numpy as np

from app.detectors.audio.decode import source_ranges, wav_bytes
from app.evidence import audio_exact_repetition as exact
from app.evidence.audio_repetition_binding import METHOD, SCHEMA, validate


def build(pcm, mapping, source, make_asset):
    samples = len(pcm)
    scope = exact.MIN_SAMPLES <= samples <= exact.MAX_SAMPLES
    continuous = len(mapping["timeline"]) == 1
    context = {
        "schema": SCHEMA,
        "evidence_type": "context",
        "affects_overall_confidence": False,
        "source_sha256": source["sha256"],
        "pcm_sha256": source["pcm_sha256"],
        "sample_rate": exact.SAMPLE_RATE,
        "input_samples": samples,
        "method": dict(METHOD),
        "searched_interval": None,
        "status": "unavailable",
        "reason": "source_clock_discontinuous" if scope else "duration_out_of_scope",
        "pairs": [],
    }
    if scope and continuous:
        payload = np.rint(pcm * 32768.0).astype("<i2").tobytes()
        result = exact.find_repeated_spans(payload)
        context.update(searched_interval={"start_sample": 0, "end_sample": samples},
                       status=result.status, reason=result.reason)
        for index, span in enumerate(result.spans, 1):
            first, second = span.first, span.second
            start, stop = first
            first_bytes = payload[2 * start:2 * stop]
            if first_bytes != payload[2 * second[0]:2 * second[1]]:
                raise ValueError("Repetition intervals do not contain identical samples")
            clip = make_asset(wav_bytes(pcm[start:stop]), "audio/wav", samples=span.sample_count,
                              sample_rate=exact.SAMPLE_RATE, channels=1)

            def anchor(interval):
                a, b = interval
                return {"start_sample": a, "end_sample": b, "source_timestamp_ranges": source_ranges(mapping, a, b)}

            context["pairs"].append({
                "id": f"R{index:02d}", "first": anchor(first), "second": anchor(second),
                "sample_count": span.sample_count, "lag_samples": span.lag_samples,
                "pcm_s16le_sha256": hashlib.sha256(first_bytes).hexdigest(),
                "artifacts": {"first_clip": clip, "second_clip": dict(clip)},
            })
    validate(context, source)
    return context


def verify(context, *, pcm, mapping, source, read_asset):
    """Recompute ranges and compare each declared clip to its frozen PCM bytes."""
    validate(context, source)

    def checked_asset(data, media_type, **metadata):
        sha = hashlib.sha256(data).hexdigest()
        if read_asset(sha) != data:
            raise ValueError("Repetition playback clip differs from the frozen source samples")
        return {"sha256": sha, "media_type": media_type, **metadata}

    expected = build(pcm, mapping, source, checked_asset)
    recorded = deepcopy(context)
    for pair in recorded["pairs"]:
        for clip in pair["artifacts"].values():
            clip.pop("url", None)
    if expected != recorded:
        raise ValueError("Repetition context differs from the measured source audio")
