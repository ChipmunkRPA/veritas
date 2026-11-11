"""Lip-sync lane, model-free parts (V-5; VF §4.1–4.3, §8.1, §9.5).

- **Speech.** An in-house voice-activity detector on the 16 kHz mono sound track. Per 40 ms hop (one tick of the 25 fps
  grid), it takes the RMS level in dBFS and the spectral flatness of 100–4000 Hz. A hop is speech when its level is at
  least 12 dB above the track's 10th-percentile level and above −50 dBFS, and its flatness is below 0.4 (voiced speech is
  harmonic, noise is flat). Gaps under 0.3 s are closed, then runs under 0.2 s are dropped.
- **Tick grid.** SyncNet works at 25 fps. Tick k is source time t0 + k / 25 (t0 is the first decoded audio sample) and audio
  samples k × 640 onwards. Each tick takes the nearest decoded frame, so a 24 or 30 fps video repeats or skips frames.
  Source timestamps align the two streams. A tick without a picture within one grid period (40 ms) has no frame and
  cannot form a scoring window, including ticks inside a source timestamp gap.
- **Windows.** Per face track, the grid's consecutive 1 s windows (25 ticks) in which the track is present on every
  tick and at least half the ticks are speech; at most 30 per track, evenly thinned (VF §8.2).
- **Active speaker.** Mouth motion is the mean absolute change of a 32 × 24 grey mouth patch between consecutive ticks.
  A track's attribution score is the Pearson correlation of its mouth motion with the speech level (linear RMS) over its
  windows, maximised over lags of ±15 ticks. A track below 0.2 is not attributed; where the windows of two attributed
  tracks overlap, the track with the higher score keeps the window.
- **Offset and confidence** (syncnet_python's protocol). For each start of a W-tick visual window inside a 1 s window,
  the distances between its visual embedding and the audio embeddings at shifts d = −15 … 15 ticks are averaged over the
  starts. The offset is the argmin d, and the confidence is the median distance minus the minimum. A positive offset
  means the sound is later than the picture (syncnet_python prints the opposite sign).
- **Segments.** Windows below the collapse threshold are *collapsed*: below 0.5 × the track's median confidence when it
  has at least 4 windows, or below the embedder's absolute `collapse_confidence`. Two or more consecutive collapsed
  windows make a confidence-collapse segment. The other windows at or above the embedder's `min_confidence` are split
  into runs whose offsets stay within ±1 tick of the run median. Runs of one window are dropped as unstable, and adjacent
  runs with medians within ±1 tick are merged. The longest run (the earliest on a tie) sets the baseline offset. A run
  whose median differs from it by at least 3 ticks (120 ms) is an offset-step segment; its boundaries with neighbouring
  runs are placed by a cumulative-sum change point on the per-start distance difference between the two offsets.
- **Reading.** A step or a collapse is the only positive reading, and it is weak evidence by design (VF §4.2): generators
  optimise SyncNet agreement, so a high confidence says nothing about manipulation. With no step, a baseline offset of
  at least 2 ticks (80 ms) is "constant offset: consistent with mux delay (benign)"; a smaller one is in sync.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import groupby
from typing import Any, Protocol

import cv2
import numpy as np

SR = 16000
FPS = 25.0
HOP = int(SR / FPS)  # 640 samples: one tick
MAX_PICTURE_SKEW_S = 1.0 / FPS  # input coverage tolerance, not a manipulation decision threshold
VSHIFT = 15  # ± ticks searched (± 0.6 s)
WINDOW_TICKS = 25  # 1 s windows (SAMPLING speaking_window_s)
MAX_WINDOWS = 30  # per track (VF §8.2)
MAX_TRACKS = 4  # candidate tracks per video, by median face size
MIN_SPEECH_FRACTION = 0.5
SMALL_FACE_PX = 96.0  # VF §4.1: frontal faces of at least 96 px
PATCH = (32, 24)  # grey mouth patch (w, h)
ATTRIBUTION_MIN = 0.2
COLLAPSE_RATIO = 0.5
COLLAPSE_MIN_TRACK_WINDOWS = 4
MIN_COLLAPSE_WINDOWS = 2
STABLE_TICKS = 1
MIN_RUN_WINDOWS = 2
STEP_TICKS = 3
BENIGN_OFFSET_TICKS = 2
VAD_ABOVE_FLOOR_DB = 12.0
VAD_MIN_DBFS = -50.0
VAD_MAX_FLATNESS = 0.4
VAD_CLOSE_GAP_S = 0.3
VAD_MIN_RUN_S = 0.2
CORRELATION_VARIATION_MIN = 1e-9
BENIGN = "constant offset: consistent with mux delay (benign)"


class SyncEmbedder(Protocol):
    """A SyncNet-class model: per-tick crops, visual windows of `frames` ticks and audio windows of the same span."""

    frames: int  # visual window W in ticks (16 for StableSyncNet, 5 for syncnet_v2)
    min_confidence: float  # confidence below which an offset is not trusted
    collapse_confidence: float | None  # absolute collapse threshold (None: not calibrated)

    def crop(self, rgb: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray | None: ...

    def visual(self, windows: np.ndarray) -> np.ndarray: ...  # (N, W, …) crops → (N, D) L2-normalised

    def audio(self, pcm: np.ndarray, starts: Sequence[int]) -> np.ndarray: ...  # tick starts → (N, D) L2-normalised


# --- speech ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Speech:
    level_db: np.ndarray  # per tick, dBFS
    level: np.ndarray  # per tick, linear RMS
    flatness: np.ndarray  # per tick, 100–4000 Hz
    speech: np.ndarray  # per tick, bool
    threshold_db: float

    @property
    def ticks(self) -> int:
        return len(self.speech)

    def intervals(self) -> list[tuple[float, float]]:
        """(start_s, end_s) of the speech runs on the audio time axis."""
        return [(a / FPS, b / FPS) for a, b in _runs(self.speech)]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def vad(pcm: np.ndarray, sr: int = SR) -> Speech:
    hop = int(sr / FPS)
    n = len(pcm) // hop
    frames = np.asarray(pcm[:n * hop], np.float64).reshape(n, hop)
    level = np.sqrt(np.mean(frames ** 2, axis=1)) if n else np.zeros(0)
    level_db = 20.0 * np.log10(np.maximum(level, 1e-5))
    power = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1.0 / sr)
    band = power[:, (freqs >= 100.0) & (freqs <= 4000.0)] + 1e-12
    flatness = np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1) if n else np.zeros(0)
    threshold = max(float(np.percentile(level_db, 10)) + VAD_ABOVE_FLOOR_DB, VAD_MIN_DBFS) if n else VAD_MIN_DBFS
    mask = (level_db > threshold) & (flatness < VAD_MAX_FLATNESS)
    runs = _runs(mask)
    for (_, b), (c, _) in zip(runs, runs[1:], strict=False):
        if (c - b) / FPS < VAD_CLOSE_GAP_S:
            mask[b:c] = True
    for a, b in _runs(mask):
        if (b - a) / FPS < VAD_MIN_RUN_S:
            mask[a:b] = False
    return Speech(level_db, level, flatness, mask, threshold)


# --- tick grid and windows ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TickGrid:
    t0: float
    frames: tuple[int, ...]  # frame-table index per tick (repeats allowed; -1 without nearby source picture coverage)
    audio_ticks: int

    @property
    def ticks(self) -> int:
        return len(self.frames)

    def seconds(self, tick: float) -> float:
        return self.t0 + tick / FPS


def tick_grid(times: np.ndarray, duration: float, audio_samples: int, *, audio_start_s: float | None = None) -> TickGrid:
    """Nearest source picture within one grid period; uncovered ticks remain -1.

    A source may hold a picture for longer, but that supplies no observed mouth
    motion at the missing timestamps. Such ticks cannot support a lip-sync window.
    """
    if len(times) == 0:
        return TickGrid(0.0, (), audio_samples // HOP)
    video_start = float(times.min())
    # The optional default preserves callers of the legacy helper; production AV adapters require a verified mapping.
    t0 = video_start if audio_start_s is None else float(audio_start_s)
    audio_ticks = audio_samples // HOP
    n = min(max(0, int(np.floor((video_start + duration - t0) * FPS + 1e-9))), audio_ticks)
    order = np.argsort(times, kind="stable")
    sorted_t = times[order]
    targets = t0 + np.arange(max(n, 0)) / FPS
    if len(sorted_t) == 1:
        return TickGrid(t0, tuple(int(order[0]) if video_start <= t <= video_start + MAX_PICTURE_SKEW_S + 1e-9
                                 else -1 for t in targets), audio_ticks)
    pos = np.clip(np.searchsorted(sorted_t, targets), 1, len(sorted_t) - 1)
    left, right = sorted_t[pos - 1], sorted_t[pos]
    choose = np.where(targets - left <= right - targets, pos - 1, pos)
    covered = (targets >= video_start) & (np.abs(targets - sorted_t[choose]) <= MAX_PICTURE_SKEW_S + 1e-9)
    return TickGrid(t0, tuple(int(i) if ok else -1 for ok, i in zip(covered, order[choose], strict=True)), audio_ticks)


@dataclass(frozen=True)
class Window:
    track_id: str
    index: int  # grid window index j: ticks ⌊hop j⌋ … ⌊hop j⌋ + 24
    speech_fraction: float
    face_px: float  # median short side of the track's boxes in the window
    hop: float = WINDOW_TICKS  # ticks between window starts (V-5a: 12.5, overlapping windows)

    @property
    def start_tick(self) -> int:
        return int(self.index * self.hop)

    @property
    def ticks(self) -> range:
        return range(self.start_tick, self.start_tick + WINDOW_TICKS)


def track_windows(track_id: str, frames: Mapping[int, Mapping[str, Any]], grid: TickGrid, speech: Speech, *,
                  hop: float = WINDOW_TICKS, max_windows: int = MAX_WINDOWS) -> list[Window]:
    out = []
    limit = min(grid.ticks, speech.ticks)
    j = 0
    while int(j * hop) + WINDOW_TICKS <= limit:
        ticks = range(int(j * hop), int(j * hop) + WINDOW_TICKS)
        boxes = [frames.get(grid.frames[k]) for k in ticks]
        fraction = float(np.mean(speech.speech[ticks.start:ticks.stop]))
        if all(b is not None for b in boxes) and fraction >= MIN_SPEECH_FRACTION:
            px = float(np.median([min(b["bbox"]["w"], b["bbox"]["h"]) for b in boxes]))
            out.append(Window(track_id, j, fraction, px, hop))
        j += 1
    if len(out) > max_windows:
        out = [out[i] for i in np.linspace(0, len(out) - 1, max_windows).round().astype(int)]
    return out


# --- mouth motion and attribution ------------------------------------------------------------------------------------

def mouth_box(frame: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """(x, y, w, h) around the mouth: from the two YuNet mouth corners, or the lower face box without landmarks."""
    landmarks = frame.get("landmarks")
    if landmarks and len(landmarks) >= 5:
        (x1, y1), (x2, y2) = landmarks[3], landmarks[4]
        d = max(float(np.hypot(x2 - x1, y2 - y1)), 2.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return cx - 0.9 * d, cy - 0.6 * d, 1.8 * d, 1.2 * d
    b = frame["bbox"]
    return b["x"] + 0.2 * b["w"], b["y"] + 0.62 * b["h"], 0.6 * b["w"], 0.33 * b["h"]


def mouth_patch(rgb: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray:
    height, width = rgb.shape[:2]
    x, y, w, h = mouth_box(frame)
    x0, y0 = int(np.clip(np.floor(x), 0, width - 1)), int(np.clip(np.floor(y), 0, height - 1))
    x1, y1 = int(np.clip(np.ceil(x + w), x0 + 1, width)), int(np.clip(np.ceil(y + h), y0 + 1, height))
    grey = cv2.cvtColor(np.ascontiguousarray(rgb[y0:y1, x0:x1]), cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, PATCH, interpolation=cv2.INTER_AREA).astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < CORRELATION_VARIATION_MIN or np.std(b) < CORRELATION_VARIATION_MIN:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def speaker_score(windows: Sequence[Window], patches: Mapping[int, np.ndarray], level: np.ndarray) -> float:
    """Pearson correlation of mouth motion with the speech level over the windows, maximised over lags of ±15 ticks."""
    motion, ticks = [], []
    for window in windows:
        for k in window.ticks[1:]:
            if k in patches and k - 1 in patches:
                motion.append(float(np.mean(np.abs(patches[k] - patches[k - 1]))))
                ticks.append(k)
    if len(motion) < 3:
        return 0.0
    m, t = np.asarray(motion), np.asarray(ticks)
    best = 0.0
    for lag in range(-VSHIFT, VSHIFT + 1):
        valid = (t + lag >= 0) & (t + lag < len(level))
        best = max(best, _pearson(m[valid], level[t[valid] + lag]))
    return best


def attribute(windows: Mapping[str, Sequence[Window]], scores: Mapping[str, float]) -> dict[str, list[Window]]:
    """The windows each attributed track keeps (score ≥ 0.2; the higher score wins an overlapping window)."""
    taken: set[int] = set()
    kept: dict[str, list[Window]] = {}
    for track_id in sorted(windows, key=lambda t: (-scores.get(t, 0.0), t)):
        if scores.get(track_id, 0.0) < ATTRIBUTION_MIN:
            continue
        mine = [w for w in windows[track_id] if w.index not in taken]
        taken.update(w.index for w in mine)
        if mine:
            kept[track_id] = mine
    return kept


# --- offset and confidence -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowSync:
    window: Window
    starts: tuple[int, ...]  # visual start ticks
    distances: np.ndarray  # (starts, 2 VSHIFT + 1), inf where the shifted audio falls outside the track
    offset: int  # ticks; positive: the sound is later than the picture
    min_distance: float
    confidence: float

    @property
    def start_tick(self) -> int:
        return self.window.start_tick

    @property
    def end_tick(self) -> int:
        return self.window.start_tick + WINDOW_TICKS


def visual_starts(window: Window, frames: int) -> tuple[int, ...]:
    return tuple(range(window.start_tick, window.start_tick + WINDOW_TICKS - frames + 1))


def audio_starts(starts: Sequence[int], frames: int, audio_ticks: int) -> list[int]:
    """Every audio start the shift search needs for these visual starts, inside the track."""
    return sorted({s + d for s in starts for d in range(-VSHIFT, VSHIFT + 1) if s + d >= 0 and s + d + frames <= audio_ticks})


def offset_confidence(distances: np.ndarray) -> tuple[int, float, float]:
    """(offset ticks, min distance, confidence) of a window's (starts, shifts) distances."""
    finite = np.isfinite(distances)
    counts = finite.sum(axis=0)
    profile = np.where(counts > 0, np.where(finite, distances, 0.0).sum(axis=0) / np.maximum(counts, 1), np.inf)
    valid = np.isfinite(profile)
    if not valid.any():
        return 0, float("inf"), 0.0
    k = int(np.argmin(np.where(valid, profile, np.inf)))
    return k - VSHIFT, float(profile[k]), float(np.median(profile[valid]) - profile[k])


def window_sync(window: Window, frames: int, visual: np.ndarray, audio: Mapping[int, np.ndarray]) -> WindowSync:
    """Distances of each visual start's embedding (rows of `visual`) against the audio embeddings at every shift."""
    starts = visual_starts(window, frames)
    distances = np.full((len(starts), 2 * VSHIFT + 1), np.inf)
    for i, s in enumerate(starts):
        for j, d in enumerate(range(-VSHIFT, VSHIFT + 1)):
            a = audio.get(s + d)
            if a is not None:
                distances[i, j] = float(np.linalg.norm(visual[i] - a))
    offset, min_distance, confidence = offset_confidence(distances)
    return WindowSync(window, starts, distances, offset, min_distance, confidence)


# --- segments and reading ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    kind: str  # offset_step | confidence_collapse
    start_tick: float
    end_tick: float
    windows: int
    offset: int | None = None  # the segment's median offset (offset_step)
    baseline: int | None = None
    confidence: float | None = None  # the lowest window confidence (confidence_collapse)


@dataclass(frozen=True)
class TrackSync:
    windows: tuple[WindowSync, ...]
    collapsed: tuple[bool, ...]
    runs: tuple[tuple[WindowSync, ...], ...]
    baseline: int | None
    steps: tuple[Segment, ...]
    collapses: tuple[Segment, ...]
    collapse_threshold: float | None
    reading: str  # offset_step | confidence_collapse | offset_step_and_collapse | constant_offset | in_sync | no_stable_offset
    flagged_fraction: float
    confidence_median: float
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def flagged(self) -> bool:
        return bool(self.steps or self.collapses)


def _median_offset(run: Sequence[WindowSync]) -> int:
    return int(np.round(np.median([w.offset for w in run])))


def offset_runs(windows: Sequence[WindowSync]) -> list[list[WindowSync]]:
    runs: list[list[WindowSync]] = []
    for w in windows:
        if runs and abs(w.offset - _median_offset(runs[-1])) <= STABLE_TICKS:
            runs[-1].append(w)
        else:
            runs.append([w])
    merged: list[list[WindowSync]] = []
    for run in (r for r in runs if len(r) >= MIN_RUN_WINDOWS):
        if merged and abs(_median_offset(run) - _median_offset(merged[-1])) <= STABLE_TICKS:
            merged[-1] += run
        else:
            merged.append(list(run))
    return merged


def change_tick(before: WindowSync, after: WindowSync, old: int, new: int, frames: int) -> float:
    """Tick of the offset change between two windows: argmax of the cumulative distance difference new − old."""
    starts = [*before.starts, *after.starts]
    diffs = []
    for sync in (before, after):
        for row in sync.distances:
            a, b = row[new + VSHIFT], row[old + VSHIFT]
            diffs.append(a - b if np.isfinite(a) and np.isfinite(b) else 0.0)
    k = int(np.argmax(np.cumsum(diffs)))
    if k + 1 < len(starts):
        return (starts[k] + starts[k + 1]) / 2.0 + frames / 2.0
    return starts[k] + frames / 2.0


def track_sync(windows: Sequence[WindowSync], embedder: SyncEmbedder, *,
               unmeasured_indices: Sequence[int] = ()) -> TrackSync:
    ordered = sorted(windows, key=lambda w: w.start_tick)
    confidences = np.asarray([w.confidence for w in ordered])
    median = float(np.median(confidences)) if len(ordered) else 0.0
    thresholds = [t for t in (COLLAPSE_RATIO * median if len(ordered) >= COLLAPSE_MIN_TRACK_WINDOWS else None,
                              embedder.collapse_confidence) if t is not None]
    threshold = max(thresholds) if thresholds else None
    collapsed = [threshold is not None and w.confidence < threshold for w in ordered]

    collapses = []
    run: list[WindowSync] = []
    for w, down in [*zip(ordered, collapsed, strict=True), (None, False)]:
        if down and (not run or w.window.index == run[-1].window.index + 1):
            run.append(w)
            continue
        if len(run) >= MIN_COLLAPSE_WINDOWS:
            collapses.append(Segment("confidence_collapse", run[0].start_tick, run[-1].end_tick, len(run),
                                     confidence=min(x.confidence for x in run)))
        run = [w] if down else []

    confident = [w for w, down in zip(ordered, collapsed, strict=True) if not down and w.confidence >= embedder.min_confidence]
    gaps = sorted(unmeasured_indices)
    run_groups = [offset_runs(list(group)) for _, group in groupby(
        confident, key=lambda w: bisect_left(gaps, w.window.index))]
    runs = [run for group in run_groups for run in group]
    baseline = _median_offset(max(runs, key=len)) if runs else None  # max keeps the earliest on a tie
    steps = []
    flagged_ids = set()
    for local in run_groups:
        if not local:
            continue
        local_baseline = _median_offset(max(local, key=len))
        for i, r in enumerate(local):
            median_offset = _median_offset(r)
            if abs(median_offset - local_baseline) < STEP_TICKS:
                continue
            start = (change_tick(local[i - 1][-1], r[0], _median_offset(local[i - 1]), median_offset, embedder.frames)
                     if i > 0 else r[0].start_tick)
            end = (change_tick(r[-1], local[i + 1][0], median_offset, _median_offset(local[i + 1]), embedder.frames)
                   if i + 1 < len(local) else r[-1].end_tick)
            steps.append(Segment("offset_step", start, end, len(r), offset=median_offset, baseline=local_baseline))
            flagged_ids.update(id(w) for w in r)
    flagged_ids |= {id(w) for w, down in zip(ordered, collapsed, strict=True) if down and any(
        s.start_tick <= w.start_tick < s.end_tick for s in collapses)}
    if steps and collapses:
        reading = "offset_step_and_collapse"
    elif steps:
        reading = "offset_step"
    elif collapses:
        reading = "confidence_collapse"
    elif baseline is None:
        reading = "no_stable_offset"
    else:
        reading = "constant_offset" if abs(baseline) >= BENIGN_OFFSET_TICKS else "in_sync"
    return TrackSync(tuple(ordered), tuple(collapsed), tuple(tuple(r) for r in runs), baseline, tuple(steps), tuple(collapses),
                     threshold, reading, len(flagged_ids) / len(ordered) if ordered else 0.0, median)


def ticks_to_frame(grid: TickGrid, tick: float) -> int:
    """Frame-table index of a (fractional) tick, clamped to the grid."""
    return grid.frames[int(np.clip(round(tick), 0, max(grid.ticks - 1, 0)))]


def batched(items: Sequence[Any], size: int, fn: Callable[[Sequence[Any]], np.ndarray]) -> np.ndarray:
    parts = [fn(items[i:i + size]) for i in range(0, len(items), size)]
    return np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)
