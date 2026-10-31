"""Licence-clean lip-sync lane, model-free (V-5a; D-37; VF §4.1–4.3). The plugin is `adapters/lipsync_xcorr.py`.

- **Mouth signal.** D-37 names a landmark model's inner-lip aperture. MediaPipe Face Landmarker is not used (ADR 0003): the
  Face Mesh V2 model card says Apache-2.0, but the `mediapipe` wheel installs `opencv-contrib-python`, which replaces the
  backend's `cv2`. The lane takes the D-37 fallback: YuNet's two mouth corners (from the face tracker) place the V-5 grey
  mouth patch (`lipsync.mouth_patch`, 32 × 24), and the aperture proxy per tick is the patch median minus the mean of the
  patch centre (the middle half of the columns, the middle 60 % of the rows), divided by the median. An open mouth shows
  the dark oral cavity in the centre, so the proxy rises with the opening; teeth, tongue, beards and lighting move it too.
- **Audio signal.** The V-5 voice-activity pass: the RMS level per 40 ms tick (25 Hz) and the speech decision that gates
  the windows.
- **Windows.** Per face track, 1 s windows (25 ticks) starting every 0.5 s (tick ⌊12.5 j⌋) in which the track has five
  landmarks on every tick and at least half the ticks are speech; at most 240 per track (2 min), evenly thinned. Windows
  whose ±15-tick search would leave the sound track (the first 0.6 s and the last 1.6 s) are skipped, since a one-sided
  search biases the offset.
- **Correlation.** Per window, the Pearson correlation of its 25 aperture values with the 25 level values from tick
  start + d, for d = −15 … 15 (shifts outside the sound track are skipped). The offset is the argmax d (positive: the
  sound is later than the picture), the peak correlation is the window's confidence, and 1 − r are its distances.
- **Segments and reading.** `lipsync.track_sync` as in V-5: a correlation collapse is two or more consecutive windows
  below half the track's median peak (at least 4 windows); offsets are not trusted below a peak of 0.4; steps are runs at
  least 3 ticks from the baseline; a constant offset of at least 2 ticks is benign.
- **Weighting** (D-37; VF §4.2). In sync, a constant offset and no stable offset are informational (no probability). An
  offset step or a correlation collapse is explanatory with a hand-set, uncalibrated probability: 0.55 for a collapse,
  0.65 for a step, 0.70 for both, outcome weight 0.4.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from app.detectors.video import lipsync

HOP_TICKS = lipsync.WINDOW_TICKS / 2.0  # 0.5 s
MAX_WINDOWS = 240
MIN_PEAK = 0.4
CENTRE_COLUMNS, CENTRE_ROWS = (0.25, 0.75), (0.2, 0.8)
P_COLLAPSE, P_STEP, P_BOTH = 0.55, 0.65, 0.70
OUTCOME_WEIGHT = 0.4


@dataclass(frozen=True)
class Settings:
    """The `lipsync.SyncEmbedder` fields `track_sync` reads: one correlation per 25-tick window."""

    frames: int = lipsync.WINDOW_TICKS
    min_confidence: float = MIN_PEAK
    collapse_confidence: float | None = None


SETTINGS = Settings()


def aperture(patch: np.ndarray) -> float:
    """Mouth-opening proxy of a grey mouth patch: (median − mean of the centre) / median."""
    h, w = patch.shape[:2]
    centre = patch[int(h * CENTRE_ROWS[0]):int(np.ceil(h * CENTRE_ROWS[1])), int(w * CENTRE_COLUMNS[0]):int(np.ceil(w * CENTRE_COLUMNS[1]))]
    median = float(np.median(patch))
    return (median - float(np.mean(centre))) / max(median, 1.0)


def correlation_profile(mouth: np.ndarray, level: np.ndarray, start: int) -> np.ndarray:
    """r(d) for d = −15 … 15: the window's aperture values against the level from tick start + d (NaN outside the track)."""
    n = len(mouth)
    out = np.full(2 * lipsync.VSHIFT + 1, np.nan)
    for j, d in enumerate(range(-lipsync.VSHIFT, lipsync.VSHIFT + 1)):
        a = start + d
        if a >= 0 and a + n <= len(level):
            out[j] = lipsync._pearson(mouth, level[a:a + n])
    return out


def full_search(window: lipsync.Window, audio_ticks: int) -> bool:
    """Whether every shift of the ±15-tick search stays inside the sound track (otherwise the search is one-sided)."""
    return window.start_tick - lipsync.VSHIFT >= 0 and window.start_tick + lipsync.WINDOW_TICKS + lipsync.VSHIFT <= audio_ticks


def window_sync(window: lipsync.Window, apertures: Mapping[int, float], level: np.ndarray) -> lipsync.WindowSync:
    r = correlation_profile(np.asarray([apertures[k] for k in window.ticks], np.float64), level, window.start_tick)
    finite = np.isfinite(r)
    distances = np.where(finite, 1.0 - r, np.inf)[None, :]
    if not finite.any():
        return lipsync.WindowSync(window, (window.start_tick,), distances, 0, 1.0, 0.0)
    k = int(np.argmax(np.where(finite, r, -np.inf)))
    return lipsync.WindowSync(window, (window.start_tick,), distances, k - lipsync.VSHIFT, 1.0 - float(r[k]), float(r[k]))


def probability(sync: lipsync.TrackSync) -> float | None:
    """The hand-set explanatory probability of a flagged track; None (informational) otherwise."""
    if sync.steps and sync.collapses:
        return P_BOTH
    if sync.steps:
        return P_STEP
    return P_COLLAPSE if sync.collapses else None


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float64)
    std = float(values.std())
    return (values - values.mean()) / std if std > 1e-9 else np.zeros_like(values)
