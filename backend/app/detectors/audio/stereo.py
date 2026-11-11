"""Bounded channel-level measurements from the standalone audio input."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

import numpy as np

from app.detectors.audio.decode import SAMPLE_RATE, source_ranges

SCHEMA = "chipmunk.stereo-channel-context/1"
MAX_SECONDS = 30
QUIET_DBFS = -50
REVIEW_GAP_DB = 12
SILENCE_FLOOR_DBFS = -120
_HASH = re.compile(r"[a-f0-9]{64}")


def _status(channels: int, layout: str | None) -> str | None:
    if channels == 1:
        return "mono"
    if channels != 2 or layout not in (None, "unknown", "stereo"):
        return "unsupported"
    return None


def supports_stereo(channels: int, layout: str | None) -> bool:
    return _status(channels, layout) is None


def _metrics(window: np.ndarray) -> tuple[float, float, float | None]:
    values = window.astype(np.float64) / 32768.0
    levels = [
        round(10 * math.log10(power), 3) if power else SILENCE_FLOOR_DBFS
        for power in np.mean(values * values, axis=0)
    ]
    centered = values - np.mean(values, axis=0)
    power = np.sum(centered * centered, axis=0)
    denominator = math.sqrt(float(power[0] * power[1]))
    correlation = (
        round(float(np.clip(np.sum(centered[:, 0] * centered[:, 1]) / denominator, -1, 1)), 6)
        if denominator else None
    )
    return levels[0], levels[1], correlation


def _classification(left: float, right: float) -> tuple[str, float | None, str | None]:
    if max(left, right) < QUIET_DBFS:
        return "quiet", None, None
    gap = round(abs(left - right), 3)
    if gap >= REVIEW_GAP_DB:
        return "strong_imbalance", gap, "left" if left > right else "right"
    return "measured", gap, None


def build(mapping: dict[str, Any], layout: str | None, stereo: np.ndarray | None,
          stereo_mapping: dict[str, Any] | None) -> dict[str, Any]:
    """Bind a bounded PCM prefix to the exact input and analysed source clock."""
    count = min(mapping["samples"] // SAMPLE_RATE, MAX_SECONDS)
    context: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_type": "context",
        "affects_overall_confidence": False,
        "source_sha256": mapping["source_sha256"],
        "source_stream_index": mapping["source_stream_index"],
        "source_channel_count": mapping["source_channels"],
        "source_channel_layout": layout,
        "sample_rate": SAMPLE_RATE,
        "window_seconds": 1,
        "max_seconds": MAX_SECONDS,
        "quiet_floor_dbfs": QUIET_DBFS,
        "strong_level_gap_db": REVIEW_GAP_DB,
        "silence_floor_dbfs": SILENCE_FLOOR_DBFS,
        "stereo_pcm_sha256": None,
        "windows": [],
    }
    status = _status(mapping["source_channels"], layout)
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
        status, gap, louder = _classification(left, right)
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
        })
    context["status"] = "measured" if any(w["status"] != "quiet" for w in windows) else "quiet"
    context["stereo_pcm_sha256"] = hashlib.sha256(stereo[:end].astype("<i2", copy=False).tobytes()).hexdigest()
    return context


def validate(context: dict[str, Any], source: dict[str, Any]) -> None:
    """Check the displayed measurements' binding, ranges, units and disposition."""
    expected = {
        "schema": SCHEMA,
        "evidence_type": "context",
        "affects_overall_confidence": False,
        "source_sha256": source.get("source_sha256", source.get("sha256")),
        "source_stream_index": source["source_stream_index"],
        "source_channel_count": source["source_channels"],
        "sample_rate": SAMPLE_RATE,
        "window_seconds": 1,
        "max_seconds": MAX_SECONDS,
        "quiet_floor_dbfs": QUIET_DBFS,
        "strong_level_gap_db": REVIEW_GAP_DB,
        "silence_floor_dbfs": SILENCE_FLOOR_DBFS,
    }
    if (context.get("affects_overall_confidence") is not False
            or any(context.get(key) != value for key, value in expected.items())):
        raise ValueError("Stereo channel evidence differs from the recorded source or method")
    status = context.get("status")
    layout = context["source_channel_layout"]
    if layout is not None and not isinstance(layout, str):
        raise ValueError("Stereo channel evidence has an invalid channel layout")
    reason = _status(source["source_channels"], layout)
    count = min(source["samples"] // SAMPLE_RATE, MAX_SECONDS)
    windows = context["windows"]
    pcm_sha = context["stereo_pcm_sha256"]
    if reason or count == 0 or status == "unavailable":
        if status != (reason or ("too_short" if count == 0 else "unavailable")) or windows != [] or pcm_sha is not None:
            raise ValueError("Unassessed stereo evidence contains inconsistent measurements")
        return
    if not _HASH.fullmatch(str(pcm_sha)) or not isinstance(windows, list) or len(windows) != count:
        raise ValueError("Stereo evidence does not cover its declared PCM prefix")
    audible = False
    for index, window in enumerate(windows):
        start, stop = index * SAMPLE_RATE, (index + 1) * SAMPLE_RATE
        if (window["start_sample"] != start or window["end_sample"] != stop
                or window["source_timestamp_ranges"] != source_ranges(source, start, stop)):
            raise ValueError("Stereo evidence differs from its recorded source timestamps")
        left, right = window["left_rms_dbfs"], window["right_rms_dbfs"]
        if any(not isinstance(level, (int, float)) or isinstance(level, bool)
               or not SILENCE_FLOOR_DBFS <= level <= 0 for level in (left, right)):
            raise ValueError("Stereo evidence has invalid channel levels")
        window_status, gap, louder = _classification(left, right)
        correlation = window["correlation"]
        if ((window["status"], window["level_gap_db"], window["louder_channel"]) != (window_status, gap, louder)
                or correlation is not None and (
                    window_status == "quiet" or not isinstance(correlation, (int, float))
                    or isinstance(correlation, bool) or not -1 <= correlation <= 1)):
            raise ValueError("Stereo evidence has inconsistent channel measurements")
        audible |= window_status != "quiet"
    if status != ("measured" if audible else "quiet"):
        raise ValueError("Stereo evidence summary differs from its measured windows")
