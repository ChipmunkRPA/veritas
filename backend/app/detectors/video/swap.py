"""Face-swap lane per track (V-3; VF §3.3, §8.1): which crops of a face track are scored, and how their scores aggregate.

- **Windows.** A track's windows are its frames on the plan's sparse sampler (2–4 fps on the pts time axis) that carry five
  landmarks, detected or interpolated between two detections. A track too short to meet the sampler gets its middle
  detected frame, so every track is scored at least once. At most `MAX_WINDOWS` windows per track, evenly spaced.
- **Track score.** The mean of the top 30 % window logits (at least one window; the v0.3 tail rule), the fraction of
  windows at logit ≥ 0 (the detector's decision boundary), and the median face size over the windows. Below 120 px the
  crops are upscaled into the 224 px input, which the VF size gate treats as unreliable: the track carries `small_face`.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

TOP_FRACTION = 0.3
MAX_WINDOWS = 240  # 60 s at 4 fps
SMALL_FACE_PX = 120.0


@dataclass(frozen=True)
class Window:
    track_id: str
    frame_index: int
    pts: int
    time_s: float
    landmarks: tuple[tuple[float, float], ...]
    face_px: float  # min(w, h) of the box, original px
    interpolated: bool


def _window(track_id: str, frame: dict[str, Any], time_s: float) -> Window | None:
    landmarks = frame.get("landmarks") or ()
    if len(landmarks) != 5:
        return None
    box = frame["bbox"]
    return Window(track_id, int(frame["frame_index"]), int(frame["pts"]), float(time_s),
                  tuple((float(x), float(y)) for x, y in landmarks), float(min(box["w"], box["h"])), frame.get("det_score") is None)


@dataclass(frozen=True)
class Selection:
    windows: list[Window]
    capped: bool  # more on-sampler windows than `max_windows`
    fallback: bool  # no on-sampler window: the middle detected frame


def select_windows(track_id: str, frames: Sequence[dict[str, Any]], sparse: Iterable[int], seconds: Any,
                   max_windows: int = MAX_WINDOWS) -> Selection:
    """`seconds(frame_index)` gives pts × time_base."""
    on_sampler = {int(i) for i in sparse}
    windows = [w for f in frames if int(f["frame_index"]) in on_sampler
               for w in [_window(track_id, f, seconds(int(f["frame_index"])))] if w is not None]
    fallback = False
    if not windows:
        detected = [f for f in frames if f.get("det_score") is not None and len(f.get("landmarks") or ()) == 5]
        if detected:
            middle = detected[len(detected) // 2]
            windows = [w for w in [_window(track_id, middle, seconds(int(middle["frame_index"])))] if w is not None]
            fallback = bool(windows)
    capped = len(windows) > max_windows
    if capped:
        keep = np.unique(np.linspace(0, len(windows) - 1, num=max_windows).round().astype(int))
        windows = [windows[int(k)] for k in keep]
    return Selection(windows, capped, fallback)


@dataclass(frozen=True)
class TrackScore:
    track_id: str
    top_mean: float  # mean of the top 30 % window logits
    flagged_fraction: float  # windows at logit ≥ 0
    windows: int
    median_face_px: float
    small_face: bool
    upscaled_fraction: float
    interpolated_fraction: float


def top_fraction_mean(values: Sequence[float], fraction: float = TOP_FRACTION) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, math.ceil(len(ordered) * fraction))
    return float(ordered[-k:].mean())


def score_track(track_id: str, windows: Sequence[Window], logits: Sequence[float], upscaled: Sequence[bool]) -> TrackScore:
    if not windows or len(windows) != len(logits) or len(logits) != len(upscaled):
        raise ValueError("a track score needs one logit and one upscale flag per window")
    z = np.asarray(logits, dtype=np.float64)
    median_px = float(np.median([w.face_px for w in windows]))
    return TrackScore(track_id, top_fraction_mean(z), float(np.mean(z >= 0.0)), len(windows), median_px,
                      median_px < SMALL_FACE_PX, float(np.mean(upscaled)), float(np.mean([w.interpolated for w in windows])))
