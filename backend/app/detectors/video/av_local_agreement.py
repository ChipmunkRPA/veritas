# SPDX-License-Identifier: Apache-2.0
"""Compare short mouth and sound trajectories on the original video clock."""

from __future__ import annotations

from functools import cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.detectors.video.viseme_patch import VisualSpeech, normalize

RATE = 16_000
LEAD = 320
HOP = 160
FFT = 512
WINDOW = 400
BANDS = 32
CONTEXT = np.arange(-2, 3)
SOUND_CONTEXT = np.arange(-3, 4) * HOP
LAGS = tuple(range(-6400, 6401, 320))
FLANK = 12
WIDTHS = (5, 6, 8, 10, 12, 14, 16)
ARMS = ("shared", "phonetic")


@cache
def _filters():
    hz = np.fft.rfftfreq(FFT, 1 / RATE)
    limits = 2595 * np.log10(1 + np.asarray([65.0, 7400.0]) / 700)
    bounds = 700 * (10 ** (np.linspace(*limits, BANDS + 2) / 2595) - 1)
    lo, mid, hi = (value[:, None] for value in (bounds[:-2], bounds[1:-1], bounds[2:]))
    filters = np.maximum(0.0, np.minimum((hz - lo) / (mid - lo), (hi - hz) / (hi - mid)))
    return filters / filters.sum(axis=1, keepdims=True)


def sound(pcm):
    samples = np.asarray(pcm, dtype=np.float32) / 32768.0
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(samples, WINDOW // 2), WINDOW)[::HOP]
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(WINDOW), n=FFT)) ** 2
    bands = 10 * np.log10((spectrum @ _filters().T).clip(1e-9))
    level = 10 * np.log10(np.mean(windows**2, axis=1).clip(1e-10))
    active = level >= max(-59.0, float(np.quantile(level, 0.8)) - 25.0)
    center = np.median(bands[active], axis=0) if active.any() else np.median(bands, axis=0)
    return ((bands - center) / 14.0).clip(-5.0, 5.0).astype(np.float32), active


def acoustic_windows(reading, sample_times):
    features, active = reading
    positions = np.rint((np.asarray(sample_times)[..., None] + LEAD + SOUND_CONTEXT) / HOP).astype(int)
    covered = (positions[..., 0] >= 0) & (positions[..., -1] < len(features))
    positions = positions.clip(0, len(features) - 1)
    return features[positions].reshape(*positions.shape[:-1], -1), covered & (
        active[positions].sum(axis=-1) >= 3
    )


def picture_windows(model, patches, valid):
    positions = np.arange(2, len(patches) - 2)
    indexes = positions[:, None] + CONTEXT
    with torch.inference_mode():
        frames = model(torch.from_numpy(normalize(patches, valid))).numpy()
    return positions, frames[indexes].reshape(len(positions), -1), valid[indexes].all(axis=1)


def load_picture(state):
    model = VisualSpeech()
    model.load_state_dict(state, strict=True)
    return model.picture.eval()


class Agreement(nn.Module):
    def __init__(self):
        super().__init__()
        self.picture = self._tower(48 * len(CONTEXT))
        self.sound = self._tower(BANDS * len(SOUND_CONTEXT))
        self.phonetic = nn.Linear(32, 7)

    @staticmethod
    def _tower(inputs):
        return nn.Sequential(nn.Linear(inputs, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 32))

    def visual(self, values):
        return F.normalize(self.picture(values), dim=-1)

    def acoustic(self, values):
        return F.normalize(self.sound(values), dim=-1)


def similarities(model, picture, sound_features, times):
    positions, values, visible = picture
    audio, audible = acoustic_windows(sound_features, times[positions, None] + LAGS)
    with torch.inference_mode():
        eye = model.visual(torch.from_numpy(values)).numpy()
        ear = model.acoustic(torch.from_numpy(audio)).numpy()
    evidence = np.full((len(times), len(LAGS)), np.nan, dtype=np.float64)
    evidence[positions] = np.where(visible[:, None] & audible, np.sum(eye[:, None, :] * ear, axis=-1), np.nan)
    return evidence


def _totals(evidence):
    valid = np.isfinite(evidence)
    blank = np.zeros((1, evidence.shape[1]))
    sums = np.vstack((blank, np.cumsum(np.where(valid, evidence, 0.0), axis=0)))
    counts = np.vstack((blank, np.cumsum(valid, axis=0)))
    return sums, counts


def _window(sums, counts, start, end, minimum):
    length = counts[end] - counts[start]
    return np.where(length >= minimum, (sums[end] - sums[start]) / length.clip(1), -np.inf)


def _compensated_delay(sums, counts, frame_count):
    middle = frame_count // 2
    halves = np.array((_window(sums, counts, 0, middle, 9), _window(sums, counts, middle, frame_count, 9)))
    close = np.abs(LAGS) <= 960
    far = np.abs(LAGS) >= 2240
    local = halves[:, close].max(axis=1)
    late = halves[:, far]
    if not np.isfinite(local).all():
        return False
    return bool(np.any(((late - local[:, None]) > 0.10).all(axis=0)))


def localize(times, evidence, samples):
    if len(times) < 34 or len(times) != len(evidence) or np.any(np.diff(times) <= 0):
        return "coverage", None
    sums, counts = _totals(evidence)
    global_reading = _window(sums, counts, 0, len(times), 23)
    if not np.isfinite(global_reading).any():
        return "coverage", None
    if _compensated_delay(sums, counts, len(times)):
        return "whole_track_delay", None
    # Keep candidate offsets close to the well-supported shared alignment.
    available = global_reading >= global_reading.max() - 0.13
    permitted = np.abs(LAGS) <= 1920
    candidates = []
    for width in WIDTHS:
        for first in range(8, len(times) - width - 8):
            end = first + width
            before = _window(sums, counts, max(0, first - FLANK), first, 6)
            after = _window(sums, counts, end, min(len(times), end + FLANK), 6)
            combined = np.where(available & permitted, np.minimum(before, after), -np.inf)
            k = int(combined.argmax())
            if not np.isfinite(combined[k]) or combined[k] < -0.15:
                continue
            inside = _window(sums, counts, first, end, max(4, (2 * width) // 3))
            if not np.isfinite(inside[k]):
                continue
            nearby = inside[k - 2 : k + 3].max()
            score = float(combined[k] - nearby)
            if score <= 0:
                continue
            lag = LAGS[k]
            a, b = int(times[first] + lag), int(times[end] + lag)
            if not 0 <= a < b <= samples:
                continue
            left = np.flatnonzero(np.isfinite(evidence[max(0, first - FLANK) : first, k]))
            right = np.flatnonzero(np.isfinite(evidence[end : end + FLANK, k]))
            middle = int(np.nanargmin(evidence[first:end, k])) + first
            frames = [int(max(0, first - FLANK) + left[-1]), middle, int(end + right[0])]
            candidates.append(
                {
                    "score": round(score, 7),
                    "span": [a, b],
                    "bracket_frames": frames,
                    "picture_span": [first, end],
                    "lag_samples": lag,
                    "agreement": [round(float(n), 7) for n in (before[k], nearby, after[k])],
                }
            )
    if not candidates:
        return "no_local_dip", None
    return "measured", min(candidates, key=lambda row: (-row["score"], row["span"]))
