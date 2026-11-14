# SPDX-License-Identifier: Apache-2.0
"""Measure directly observable periodicity in the mono soundtrack of a source-bound mouth hold."""

from __future__ import annotations

import math
from functools import lru_cache
from hashlib import file_digest

import numpy as np

from app.evidence import mouth_hold_sound as binding

ACTIVITY_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"


@lru_cache(maxsize=1)
def _load_activity(path, sha256):
    from app.detectors.audio.upgraded import SpeechActivity

    if sha256 != ACTIVITY_SHA256:
        raise ValueError("The speech-activity asset differs from the verified file")
    return SpeechActivity(path)


def activity_detector():
    from app.core.config import settings

    path = settings.weights_dir / "audio-upgrade" / "silero_vad.jit"
    with path.open("rb") as stream:
        return _load_activity(path, file_digest(stream, "sha256").hexdigest())


def speech_samples(pcm, context, interval, activity):
    left, right = context
    start, stop = interval
    if not np.any(pcm[start:stop]):
        return 0
    covered, cursor = 0, start
    for a, b in activity.spans(pcm[left:right]):
        first = max(cursor, left + round(a * binding.RATE))
        last = min(stop, left + round(b * binding.RATE))
        if first < last:
            covered += last - first
            cursor = last
    return covered


def measure(pcm, mapping, review, time_base, *, schema=binding.SCHEMA, activity=None):
    clock = binding.bounds(review, mapping, time_base)
    if clock is None or len(pcm) != mapping["samples"]:
        return None
    start, stop, first, last = clock
    hop = binding.HOP
    starts = binding.block_starts(start, stop)
    measured = len(starts)
    if measured < binding.MIN_BLOCKS:
        return None
    periodic = 0
    lags = np.arange(35, 221)
    for i in starts:
        values = np.asarray(pcm[i:i + 2 * hop], dtype=np.float64)
        signal = values - values.mean()
        if np.sqrt(np.mean(signal ** 2)) < .007:
            continue
        # Mean-centered periodicity for 73–457 Hz, with local peaks to reject slowly changing backgrounds.
        lagged = np.correlate(signal, signal, "full")[len(signal) - 1:][35:221]
        squared = np.r_[0., np.cumsum(signal ** 2)]
        norm = np.sqrt(squared[len(signal) - lags] * np.maximum(0., squared[-1] - squared[lags]))
        values = lagged / (1e-12 + norm)
        if np.any((values[1:-1] >= .78) & (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])):
            periodic += 1
    record = {"schema": schema, "samples": [start, stop], "source_pts": [first, last],
              "time_base": f"1/{binding.RATE}", "blocks": measured, "periodic_blocks": periodic,
              "sustained_periodicity": 3 * periodic >= 2 * measured}
    if schema in {binding.LEVEL_SCHEMA, binding.ACTIVITY_SCHEMA}:
        selected = np.asarray(pcm[start:stop], dtype=np.float64)
        power = float(np.mean(selected ** 2))
        record["rms_dbfs"] = round(10 * math.log10(power), 3) if power else None
    if schema == binding.ACTIVITY_SCHEMA:
        context = binding.activity_context(mapping, start, stop)
        if context is None:
            record["schema"] = binding.LEVEL_SCHEMA
        else:
            activity = activity if activity is not None else activity_detector()
            record["speech_activity"] = {"samples": speech_samples(pcm, context, (start, stop), activity),
                                         "context": list(context)}
    return record


def source_assets(pcm, mapping, review, time_base, *, time_axis):
    """Reproduce the waveform and sound clip from the committed source samples."""
    from app.detectors.audio.decode import wav_bytes
    from app.evidence.audio_packets import mouth_hold_waveform

    sound = review["sound_context"]
    if measure(pcm, mapping, review, time_base, schema=sound["schema"]) != sound:
        raise ValueError("The soundtrack does not reproduce the measured mouth-hold interval")
    start, stop = sound["samples"]
    return {binding.WAV: wav_bytes(pcm[start:stop]),
            binding.WAVEFORM: mouth_hold_waveform(pcm, sound, time_axis=time_axis)}
