"""Bounded channel-level measurements before and after mono downmix."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from app.detectors.audio.decode import SAMPLE_RATE, source_ranges
from app.evidence import stereo_binding as binding
from app.evidence.audio_silence_binding import FLANK, MIN_FLANK_DBFS, MINIMUM


def supports_stereo(channels: int, layout: str | None) -> bool:
    return binding.unsupported_status(channels, layout) is None


def _level_dbfs(power: float) -> float:
    return round(10 * math.log10(power), 3) if power else binding.SILENCE_FLOOR_DBFS


def _mono_dbfs(pcm: np.ndarray) -> float:
    values = pcm.astype(np.float64)
    return _level_dbfs(float(np.mean(values * values)))


def _metrics(window: np.ndarray) -> tuple[float, float, float | None]:
    values = window.astype(np.float64) / 32768.0
    levels = [_level_dbfs(power) for power in np.mean(values * values, axis=0)]
    centered = values - np.mean(values, axis=0)
    power = np.sum(centered * centered, axis=0)
    denominator = math.sqrt(float(power[0] * power[1]))
    correlation = (
        round(float(np.clip(np.sum(centered[:, 0] * centered[:, 1]) / denominator, -1, 1)), 6)
        if denominator else None
    )
    return levels[0], levels[1], correlation


def channel_silence(stereo: np.ndarray, timeline: list[dict], ranges) -> dict:
    """Measure bounded channel loss while the other channel remains audible."""
    spans = binding.channel_silence_spans(timeline, len(stereo))
    candidates = []
    for a, b in spans:
        for index, channel in enumerate(("left", "right")):
            signal = stereo[:, index]
            other = stereo[:, 1 - index]
            edges = np.flatnonzero(np.diff(np.r_[False, signal[a:b] == 0, False])) + a
            starts, stops = edges[::2], edges[1::2]
            eligible = ((stops - starts >= MINIMUM) & (stops - starts <= binding.MAXIMUM_SILENCE)
                        & (starts >= a + FLANK) & (stops <= b - FLANK))
            for start, stop in zip(starts[eligible].tolist(), stops[eligible].tolist(), strict=True):
                flanks = [_mono_dbfs(block.astype(np.float64) / 32768)
                          for block in (signal[start - FLANK:start], signal[stop:stop + FLANK])]
                if min(flanks) < MIN_FLANK_DBFS:
                    continue
                # Cover the tail with a full 100 ms block, overlapping the prior block if needed.
                blocks = range(start, stop - FLANK + 1, FLANK)
                remaining = min(_mono_dbfs(other[point:point + FLANK].astype(np.float64) / 32768)
                                for point in (*blocks, stop - FLANK))
                if remaining >= MIN_FLANK_DBFS:
                    candidates.append({"start_sample": start, "end_sample": stop, "silent_channel": channel,
                                       "flank_rms_dbfs": flanks, "minimum_other_rms_dbfs": remaining})
    candidates.sort(key=lambda row: (row["start_sample"] - row["end_sample"], row["start_sample"], row["silent_channel"]))
    selected = sorted(candidates[:binding.MAXIMUM_SILENCE_EVENTS],
                      key=lambda row: (row["start_sample"], row["end_sample"], row["silent_channel"]))
    for row in selected:
        row["source_timestamp_ranges"] = ranges(row["start_sample"], row["end_sample"])
    return {"schema": binding.CHANNEL_SILENCE_SCHEMA, "method": binding.CHANNEL_SILENCE_METHOD,
            "affects_overall_confidence": False, "checked_samples": sum(b - a for a, b in spans),
            "qualifying_intervals": len(candidates), "selected": selected}


def build(mapping: dict[str, Any], mono: np.ndarray, layout: str | None, stereo: np.ndarray | None,
          stereo_mapping: dict[str, Any] | None) -> dict[str, Any]:
    """Bind a bounded PCM prefix to the exact input and analysed source clock."""
    count = min(mapping["samples"] // SAMPLE_RATE, binding.MAX_SECONDS)
    context: dict[str, Any] = {
        "schema": binding.SCHEMA,
        "evidence_type": "context",
        "affects_overall_confidence": False,
        "source_sha256": mapping["source_sha256"],
        "source_stream_index": mapping["source_stream_index"],
        "source_channel_count": mapping["source_channels"],
        "source_channel_layout": layout,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": 1,
        "max_seconds": binding.MAX_SECONDS,
        "quiet_floor_dbfs": binding.QUIET_DBFS,
        "strong_level_gap_db": binding.REVIEW_GAP_DB,
        "strong_downmix_attenuation_db": binding.REVIEW_DOWNMIX_DB,
        "silence_floor_dbfs": binding.SILENCE_FLOOR_DBFS,
        "mono_pcm_sha256": mapping["pcm_sha256"],
        "stereo_pcm_sha256": None,
        "windows": [],
    }
    status = binding.unsupported_status(mapping["source_channels"], layout)
    if status or count == 0:
        context["status"] = status or "too_short"
        return context
    end = count * SAMPLE_RATE
    if (stereo is None or stereo_mapping is None or len(stereo) < end
            or source_ranges(stereo_mapping, 0, end) != source_ranges(mapping, 0, end)):
        context["status"] = "unavailable"
        return context
    windows = context["windows"]
    for start in range(0, end, SAMPLE_RATE):
        stop = start + SAMPLE_RATE
        left, right, correlation = _metrics(stereo[start:stop])
        status, gap, louder = binding.classify(left, right)
        windows.append({
            "start_sample": start,
            "end_sample": stop,
            "source_timestamp_ranges": source_ranges(mapping, start, stop),
            "status": status,
            "left_rms_dbfs": left,
            "right_rms_dbfs": right,
            "level_gap_db": gap,
            "louder_channel": louder,
            "correlation": None if status == "quiet" else correlation,
            "downmix": binding.downmix(left, right, _mono_dbfs(mono[start:stop])),
        })
    context["status"] = "measured" if any(w["status"] != "quiet" for w in windows) else "quiet"
    context["stereo_pcm_sha256"] = hashlib.sha256(stereo[:end].astype("<i2", copy=False).tobytes()).hexdigest()
    context[binding.CHANNEL_SILENCE] = channel_silence(stereo[:end], mapping["timeline"],
                                                      lambda a, b: source_ranges(mapping, a, b))
    return context


def validate(context: dict[str, Any], source: dict[str, Any]) -> None:
    binding.validate(context, source, ranges=lambda start, stop: source_ranges(source, start, stop))


def verify_mono_levels(context: dict[str, Any], pcm: np.ndarray) -> None:
    """Replay the mono half of the comparison against the committed voice input."""
    if context["schema"] != binding.SCHEMA:
        return
    for window in context["windows"]:
        level = _mono_dbfs(pcm[window["start_sample"]:window["end_sample"]])
        if window["downmix"]["mono_rms_dbfs"] != level:
            raise ValueError("Stereo downmix level differs from the analyzed mono samples")
