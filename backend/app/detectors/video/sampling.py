"""Seeded, profile-specific sampling plan over the frame table (ARCHITECTURE §3.1 `decode`, §3.3 rule 4; VF §8.1).

The plan lists frame-table indices per purpose. It is fixed before any frame is decoded and persisted with the probe
output, so a resumed run decodes identical frames. Randomness comes from numpy generators seeded by the media sha256 and
a per-component stream id, so adding a component never shifts another component's draws.

- **legacy:** the v0.3 frame count (4 per second within [16, 40]; 16 in quick), uniform over the table. The v0.3 frame
  detectors score these until V-3 and V-4 move them to per-track and per-shot sampling.
- **sparse:** 2–4 fps on the pts time axis with one seeded phase: 4 fps up to 120 s, then 4 × 120 / duration, floored at
  2 fps. It is the per-track crop sampler; V-2 and V-3 intersect it with each track's frames.
- **clips:** per shot, up to `clips_per_shot` clips of `clip_frames` consecutive frames at 8 fps on the pts time axis (the
  D3 protocol, V-4), one per equal time segment of the shot at a seeded offset; a shot shorter than one clip gives one clip
  over the whole shot with fewer frames. Until V-2 detects cuts the whole video is one shot.
- **speaking:** `speaking_window_s` windows per speaking-track interval (V-5 supplies the intervals).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from app.detectors.video.decode import FrameTable
from app.engine.detector import Profile
from app.engine.profiles import SAMPLING

PLAN_VERSION = "pts-plan-3"  # 3: pre-dedup target counts; 2: explicit clip frames; 1: contiguous start + length
FULL_RATE_UNTIL_S = 120.0
CLIP_RATE_FPS = 8.0  # upstream D3 extracts frames at 8 fps (utils/video2frame.py)
STREAM_SPARSE, STREAM_CLIPS = 1, 2


def plan_seed(sha256: str) -> int:
    return int(sha256[:8], 16)


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, stream])


def duration_s(table: FrameTable) -> float:
    """Span of the pts times plus the median frame step."""
    t = table.times()
    if len(t) == 0:
        return 0.0
    steps = np.diff(np.sort(t))
    positive = steps[steps > 0]
    return float(t.max() - t.min()) + (float(np.median(positive)) if len(positive) else 0.0)


def uniform_indices(n: int, k: int) -> list[int]:
    """The v0.3 rule: `linspace(0, n - 1, min(k, n))` truncated to int."""
    if n <= 0 or k <= 0:
        return []
    return [int(i) for i in np.unique(np.linspace(0, n - 1, num=min(k, n), dtype=int))]


def nearest_indices(times: np.ndarray, targets: np.ndarray) -> list[int]:
    """Table index of the frame nearest in time to each target, deduplicated, in index order."""
    if len(times) == 0 or len(targets) == 0:
        return []
    order = np.argsort(times, kind="stable")
    sorted_t = times[order]
    if len(sorted_t) == 1:
        return [int(order[0])]
    pos = np.clip(np.searchsorted(sorted_t, targets), 1, len(sorted_t) - 1)
    left, right = sorted_t[pos - 1], sorted_t[pos]
    choose = np.where(targets - left <= right - targets, pos - 1, pos)
    return sorted({int(i) for i in order[choose]})


def sparse_rate(duration: float, fps_range: tuple[float, float]) -> float:
    lo, hi = fps_range
    if duration <= FULL_RATE_UNTIL_S:
        return hi
    return max(lo, hi * FULL_RATE_UNTIL_S / duration)


def sparse_frames(table: FrameTable, rate: float, seed: int) -> tuple[float, list[int]]:
    t = table.times()
    step = 1.0 / rate
    phase = float(_rng(seed, STREAM_SPARSE).uniform(0.0, step))
    start, span = float(t.min()), float(t.max() - t.min())
    count = int(np.floor((span - phase) / step + 1e-9)) + 1 if span >= phase else 1  # no target after the last frame
    return phase, nearest_indices(t, start + phase + step * np.arange(count))


def shot_clips(table: FrameTable, shots: Sequence[tuple[int, int]], clips_per_shot: int, clip_frames: int, seed: int,
               rate: float = CLIP_RATE_FPS) -> list[dict[str, Any]]:
    """Per shot, up to `clips_per_shot` clips of `clip_frames` frames at `rate` on the pts time axis, one per equal time
    segment at a seeded offset. A shot shorter than one clip gives one clip over the whole shot (fewer frames)."""
    rng = _rng(seed, STREAM_CLIPS)
    t = table.times()
    step = 1.0 / rate
    out: list[dict[str, Any]] = []
    for s, (start, end) in enumerate(shots):
        if end <= start or clips_per_shot <= 0 or clip_frames <= 0:
            continue
        times = t[start:end]
        first, span = float(times.min()), float(times.max() - times.min())
        if span < (clip_frames - 1) * step:
            starts, count = [first], int(np.floor(span / step + 1e-9)) + 1
        else:
            k = min(clips_per_shot, int(np.floor((span + step) / (clip_frames * step) + 1e-9)))
            segment = (span + step) / k
            starts = [first + j * segment + float(rng.uniform(0.0, segment - clip_frames * step)) for j in range(k)]
            count = clip_frames
        for t0 in starts:
            frames = [start + i for i in nearest_indices(times, t0 + step * np.arange(count))]
            # Preserve requested slots separately: deduplication is not evidence of a genuinely short shot.
            out.append({"shot": s, "start_s": round(t0, 6), "rate_fps": rate,
                        "target_frame_count": count, "frames": frames})
    return out


def speaking_windows(table: FrameTable, intervals: Iterable[tuple[str, float, float]], window_s: float) -> list[dict[str, Any]]:
    """Consecutive `window_s` windows inside each (track_id, start_s, end_s) interval, with the frames inside each."""
    t = table.times()
    out: list[dict[str, Any]] = []
    for track_id, start_s, end_s in intervals:
        for j in range(int(np.floor((end_s - start_s) / window_s + 1e-9))):
            w0 = start_s + j * window_s
            frames = [int(i) for i in np.flatnonzero((t >= w0) & (t < w0 + window_s))]
            if frames:
                out.append({"track_id": track_id, "start_s": round(w0, 6), "end_s": round(w0 + window_s, 6), "frames": frames})
    return out


def build_plan(table: FrameTable, profile: Profile | str, *, seed: int, legacy_frames: int,
               shots: Sequence[tuple[int, int]] | None = None,
               speaking: Iterable[tuple[str, float, float]] = ()) -> dict[str, Any]:
    profile = Profile(profile)
    params = SAMPLING[profile]
    n = len(table)
    shot_list = [(0, n)] if shots is None and n else [(int(a), int(b)) for a, b in shots or ()]
    duration = duration_s(table)
    plan: dict[str, Any] = {
        "version": PLAN_VERSION, "profile": profile.value, "seed": seed, "time_base": table.time_base,
        "frames_in_table": n, "table_truncated": table.truncated, "duration_s": round(duration, 6),
        "legacy": {"method": "uniform-v0.3", "frames": uniform_indices(n, legacy_frames)},
        "sparse": None,
        "shots": {"source": "whole-video" if shots is None else "detected", "ranges": [list(s) for s in shot_list]},
        "clips": [], "speaking": [],
    }
    if n and params.track_fps:
        rate = sparse_rate(duration, params.track_fps)
        phase, frames = sparse_frames(table, rate, seed)
        plan["sparse"] = {"rate_fps": round(rate, 6), "phase_s": round(phase, 6), "frames": frames}
    if n and params.clips_per_shot:
        plan["clips"] = shot_clips(table, shot_list, params.clips_per_shot, params.clip_frames, seed)
    if n and params.speaking_window_s:
        plan["speaking"] = speaking_windows(table, speaking, params.speaking_window_s)
    return plan


def plan_frames(plan: dict[str, Any]) -> list[int]:
    """Every frame index the plan references, sorted (clips of a pts-plan-1 plan: `start` + `length`)."""
    frames = set(plan["legacy"]["frames"])
    if plan.get("sparse"):
        frames.update(plan["sparse"]["frames"])
    for clip in plan.get("clips") or ():
        frames.update(clip["frames"] if "frames" in clip else range(clip["start"], clip["start"] + clip["length"]))
    for window in plan.get("speaking") or ():
        frames.update(window["frames"])
    return sorted(frames)
