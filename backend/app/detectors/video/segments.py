"""Temporal segments (V-7; ARCHITECTURE §3.1 `temporal` and `second_pass`, VF §8.1): window scores per lane and track →
smoothed series → segments.

- **Windows.** A detector's window units carry a time extent: `start_s` / `end_s` (voice windows, per-window track scorers)
  or a point `time_s` (face-swap crops, frame channels), which spans half the median step to its neighbours. Units without
  `@` and without a `window:` / `frame:` / `clip:` index are track or whole-media aggregates and are ignored. Only
  probability-like outputs are segmented: `probability` as it is, `logit` through the logistic function. Scores without a
  probability scale (D3 σ, SyncNet confidence, forensic ratios) are not; the lip-sync lane publishes its own offset-step and
  confidence-collapse segments.
- **Lane series.** A lane and track pools its detectors: windows with the same extent average the detectors' probabilities
  (unweighted and uncalibrated until the V-8 lane sub-fusers). A series splits where the gap between two windows exceeds
  max(1 s, 2 × the median window span), so a segment never bridges a stretch without scores.
- **Segments.** Median over 3 windows (edges replicated), hysteresis (enter at ≥ 0.60, stay while ≥ 0.45), minimum 0.5 s.
  A segment spans its first window's start to its last window's end; its peak is the highest smoothed probability inside.
- **Clip features** (ML §4.3) per lane and track: mean and p90 of the window logits and the fraction above 0.
- **Second pass.** The top segments of per-track lanes, padded and intersected with the track's frames, become dense frame
  sets for a windowed re-run (at most `max_frames` frames per segment, evenly thinned).
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

METHOD = "median3-hysteresis-v1"
MEDIAN_WIDTH = 3
ENTER, EXIT = 0.60, 0.45
MIN_DURATION_S = 0.5
MIN_GAP_S = 1.0
LOGIT_CLIP = 12.0
PARAMS = {"median_width": MEDIAN_WIDTH, "enter": ENTER, "exit": EXIT, "min_duration_s": MIN_DURATION_S, "min_gap_s": MIN_GAP_S}
_INDEXED = re.compile(r"^(window|frame|clip):\d+$")


@dataclass(frozen=True)
class Window:
    start_s: float
    end_s: float
    p: float
    # Original point timestamp, separate from the interpolation cell used by existing segmentation.
    sample_s: float | None = field(default=None, compare=False)


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    peak: float  # highest smoothed probability inside the segment
    windows: int


@dataclass(frozen=True)
class LaneSeries:
    lane: str
    track_id: str | None
    channel: str  # visual | audio | av
    detectors: tuple[str, ...]
    families: tuple[str, ...]
    windows: tuple[Window, ...]
    smoothed: tuple[float, ...]
    segments: tuple[Segment, ...]

    def features(self) -> dict[str, float]:
        """ML §4.3 clip features over the window logits."""
        z = np.asarray([logit(w.p) for w in self.windows], dtype=np.float64)
        if len(z) == 0:
            return {"windows": 0.0}
        return {"windows": float(len(z)), "mean_logit": round(float(z.mean()), 6), "p90_logit": round(float(np.percentile(z, 90)), 6),
                "fraction_positive": round(float(np.mean(z > 0.0)), 6)}


def logistic(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit(p: float) -> float:
    q = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(np.clip(math.log(q / (1.0 - q)), -LOGIT_CLIP, LOGIT_CLIP))


def window_probability(raw: float, raw_kind: str) -> float | None:
    if not math.isfinite(raw):
        return None
    if raw_kind == "probability":
        return min(max(float(raw), 0.0), 1.0)
    if raw_kind == "logit":
        return logistic(float(raw))
    return None


def unit_track(unit_ref: str) -> tuple[bool, str | None]:
    """(is a window unit, track id): `track:0@pts=1200` → (True, "track:0"); `frame@pts=40` and `window:3` → (True, None);
    `track:0` and `global` → (False, None)."""
    head, at, _ = unit_ref.partition("@")
    if at:
        return True, head if head.startswith("track:") else None
    return bool(_INDEXED.match(unit_ref)), None


def point_windows(points: Iterable[tuple[float, float]]) -> list[Window]:
    """Point samples (time, p) → windows spanning half the median step to each side (zero width for a single sample)."""
    ordered = sorted(points)
    if not ordered:
        return []
    steps = np.diff([t for t, _ in ordered])
    positive = steps[steps > 0]
    half = float(np.median(positive)) / 2.0 if len(positive) else 0.0
    return [Window(round(max(0.0, t - half), 6), round(t + half, 6), p, sample_s=t) for t, p in ordered]


def entry_windows(entry: Mapping[str, Any]) -> dict[str | None, list[Window]]:
    """A detector entry's window units per track (None: not bound to a track), sorted by time."""
    spans: dict[str | None, list[Window]] = {}
    points: dict[str | None, list[tuple[float, float]]] = {}
    for unit in entry.get("unit_scores") or ():
        is_window, track = unit_track(str(unit["unit_ref"]))
        p = window_probability(float(unit["raw"]), str(unit["raw_kind"])) if is_window else None
        if p is None:
            continue
        quality = unit.get("quality") or {}
        if "start_s" in quality and "end_s" in quality and quality["end_s"] > quality["start_s"]:
            spans.setdefault(track, []).append(Window(float(quality["start_s"]), float(quality["end_s"]), p))
        elif "time_s" in quality:
            points.setdefault(track, []).append((float(quality["time_s"]), p))
    for track, samples in points.items():
        spans.setdefault(track, []).extend(point_windows(samples))
    return {track: sorted(ws, key=lambda w: (w.start_s, w.end_s)) for track, ws in spans.items()}


def pool(windows: Iterable[Window]) -> list[Window]:
    """Windows with the same extent (to the millisecond) average their probabilities."""
    groups: dict[tuple[float, float], list[Window]] = {}
    for w in windows:
        groups.setdefault((round(w.start_s, 3), round(w.end_s, 3)), []).append(w)
    return [Window(start, end, float(np.mean([w.p for w in ws])),
                   sample_s=min(w.sample_s for w in ws) if all(w.sample_s is not None for w in ws) else None)
            for (start, end), ws in sorted(groups.items())]


def split_runs(windows: Sequence[Window]) -> list[list[Window]]:
    """Consecutive runs of time-sorted windows, split where the gap exceeds max(1 s, 2 × the median span)."""
    if not windows:
        return []
    limit = max(MIN_GAP_S, 2.0 * float(np.median([w.end_s - w.start_s for w in windows])))
    runs: list[list[Window]] = [[windows[0]]]
    reach = windows[0].end_s
    for w in windows[1:]:
        if w.start_s - reach > limit:
            runs.append([])
        runs[-1].append(w)
        reach = max(reach, w.end_s)
    return runs


def median3(values: Sequence[float]) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return v
    padded = np.concatenate([v[:1], v, v[-1:]])
    return np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)


def hysteresis(values: Sequence[float], enter: float = ENTER, exit: float = EXIT) -> list[tuple[int, int]]:
    """Inclusive index runs that start at a value ≥ `enter` and last while values stay ≥ `exit`."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, p in enumerate(values):
        if start is None:
            if p >= enter:
                start = i
        elif p < exit:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def series_segments(windows: Sequence[Window]) -> tuple[list[float], list[Segment]]:
    """Smoothed probabilities (in window order) and the segments of a pooled, time-sorted lane series."""
    smoothed: list[float] = []
    out: list[Segment] = []
    for run in split_runs(windows):
        values = median3([w.p for w in run])
        smoothed += [float(v) for v in values]
        for a, b in hysteresis(values):
            start, end = run[a].start_s, max(w.end_s for w in run[a:b + 1])
            if end - start + 1e-9 >= MIN_DURATION_S:
                out.append(Segment(round(start, 6), round(end, 6), round(float(values[a:b + 1].max()), 6), b - a + 1))
    return smoothed, out


def _natural(track: str | None) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", track or "")]


def collect(entries: Iterable[Mapping[str, Any]], *, lane_of: Callable[[Mapping[str, Any]], str | None],
            channel_of: Callable[[Mapping[str, Any]], str]) -> list[LaneSeries]:
    """Lane series per (lane, track) over the ok detector entries a lane is known for, in lane and natural track order."""
    grouped: dict[tuple[str, str | None], dict[str, Any]] = {}
    for entry in entries:
        if entry.get("status") != "ok":
            continue
        lane = lane_of(entry)
        if lane is None:
            continue
        for track, ws in entry_windows(entry).items():
            group = grouped.setdefault((lane, track), {"channel": channel_of(entry), "detectors": [], "families": [], "windows": []})
            group["detectors"].append(str(entry["detector_id"]))
            if entry["family"] not in group["families"]:
                group["families"].append(str(entry["family"]))
            group["windows"] += ws
    out = []
    for (lane, track), group in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1] is not None, _natural(kv[0][1]))):
        windows = pool(group["windows"])
        smoothed, segments = series_segments(windows)
        out.append(LaneSeries(lane, track, group["channel"], tuple(group["detectors"]), tuple(group["families"]), tuple(windows),
                              tuple(smoothed), tuple(segments)))
    return out


def dense_windows(segments: Iterable[Mapping[str, Any]], frames_by_track: Mapping[str, Sequence[Mapping[str, Any]]],
                  seconds: Callable[[int], float], *, top: int = 3, pad_s: float = 0.5, max_frames: int = 64) -> list[dict[str, Any]]:
    """The `top` segments by peak (ties: earlier start, then track) whose track has frames, padded by `pad_s` on each side:
    every frame of the track inside the padded span, evenly thinned to `max_frames`."""
    ranked = sorted((s for s in segments if s.get("track_id") in frames_by_track),
                    key=lambda s: (-float(s["peak"]), float(s["start_s"]), _natural(s["track_id"])))[:top]
    out = []
    for s in ranked:
        lo, hi = float(s["start_s"]) - pad_s, float(s["end_s"]) + pad_s
        frames = sorted({int(f["frame_index"]) for f in frames_by_track[s["track_id"]] if lo <= seconds(int(f["frame_index"])) <= hi})
        capped = len(frames) > max_frames
        if capped:
            frames = [frames[int(k)] for k in np.unique(np.linspace(0, len(frames) - 1, num=max_frames).round().astype(int))]
        if frames:
            out.append({"track_id": s["track_id"], "lane": s.get("lane"), "segment_start_s": float(s["start_s"]),
                        "segment_end_s": float(s["end_s"]), "peak": float(s["peak"]), "start_s": round(max(0.0, lo), 6),
                        "end_s": round(hi, 6), "frames": frames, "capped": capped})
    return out
