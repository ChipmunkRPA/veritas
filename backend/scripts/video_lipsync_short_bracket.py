# SPDX-License-Identifier: Apache-2.0
"""Short, same-track timing intervals supported by flanking lag measurements."""

from __future__ import annotations

from itertools import product

import numpy as np

from app.detectors.video import lipsync, xcorr
from app.detectors.video.adapters.lipsync_xcorr import _track_sync

LAGS = np.arange(-6, 7)
WIDTHS = (12, 16, 20)
HOP = 2
MIN_RELATIVE_LAG = 3
MIN_WINDOW_CORRELATION = .15
MIN_FLANK_GAIN = .015
MIN_SPEECH_FRACTION = .35
MAX_FLANK_LAG_DISTANCE = 1
ARMS = ("short_aperture", "short_joint")
POINT_FLOOR = .10
NEGATIVE_MARGIN = .025


def motion(patches, n):
    """Central-mouth change minus surrounding crop change, over adjacent native inputs."""
    signal = np.full(n, np.nan, dtype=np.float64)
    for tick in range(1, n):
        if tick not in patches or tick - 1 not in patches:
            continue
        a, b = (patches[key].astype(np.float64) for key in (tick - 1, tick))
        a -= np.median(a)
        b -= np.median(b)
        d = np.abs(b - a)
        signal[tick] = max(0., float(d[5:19, 8:24].mean()) - float(np.mean(d[:, [*range(4), *range(28, 32)]])))
    return signal


def audio_flux(pcm):
    n = len(pcm) // lipsync.HOP
    if n < 2:
        return np.zeros(n)
    windows = pcm[:n*lipsync.HOP].reshape(n, lipsync.HOP).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(windows * np.hanning(lipsync.HOP), axis=1))
    hz = np.fft.rfftfreq(lipsync.HOP, 1. / lipsync.SR)
    spectrum = np.asarray([np.linalg.norm(spectrum[:, (hz >= lo) & (hz < hi)], axis=1)
                           for lo, hi in ((150, 350), (350, 700), (700, 1400), (1400, 2700), (2700, 4100))]).T
    unit = spectrum / np.maximum(np.linalg.norm(spectrum, axis=1, keepdims=True), 1e-5)
    flux = np.linalg.norm(np.diff(unit, axis=0), axis=1) * np.sqrt(np.minimum(np.linalg.norm(spectrum[:-1], axis=1),
                                                                           np.linalg.norm(spectrum[1:], axis=1)))
    return np.concatenate(([0.], flux))


def _profile(visual, audio, start, width):
    return np.asarray([lipsync._pearson(visual[start:start+width], audio[start+d:start+d+width])
                       for d in LAGS], dtype=np.float64)


def _eligible(values, start, width):
    chunk = values[start:start+width]
    return len(chunk) == width and np.isfinite(chunk).all() and chunk.std() >= lipsync.CORRELATION_VARIATION_MIN


def _correlations(aperture, move, speech, flux, arm):
    n = len(aperture)
    result = {}
    for width in WIDTHS:
        for start in range(int(-LAGS[0]), min(n, len(speech.level) - int(LAGS[-1])) - width + 1):
            if (not _eligible(aperture, start, width)
                    or np.mean(speech.speech[start:start+width]) < MIN_SPEECH_FRACTION):
                continue
            basic = _profile(aperture, speech.level, start, width)
            if arm == "short_joint":
                if not _eligible(move, start, width):
                    continue
                second = _profile(move, flux, start, width)
                result[start, width] = .5 * (basic + second)
            else:
                result[start, width] = basic
    return result


def _choice(three):
    before, middle, after = three
    left, right = np.argmax(before), np.argmax(after)
    if abs(int(left) - int(right)) > MAX_FLANK_LAG_DISTANCE:
        return None
    r = int(np.argmax(np.minimum(before, after)))
    if min(before[r], after[r]) < MIN_WINDOW_CORRELATION:
        return None
    k = int(np.argmax(middle))
    lag = int(LAGS[k] - LAGS[r])
    if abs(lag) < MIN_RELATIVE_LAG or middle[k] < MIN_WINDOW_CORRELATION:
        return None
    flank = min(float(before[r] - before[k]), float(after[r] - after[k]))
    if flank < MIN_FLANK_GAIN:
        return None
    score = min(float(middle[k] - middle[r]), 2 * flank)
    return score, lag, int(LAGS[r]), int(LAGS[k])


def candidate(patches, pcm, tick_count):
    """Measure both frozen profiles without supplying known edit positions."""
    speech = lipsync.vad(pcm)
    ap = np.full(tick_count, np.nan, dtype=np.float64)
    for tick, patch in patches.items():
        ap[tick] = xcorr.aperture(patch)
    move, flux = motion(patches, tick_count), audio_flux(pcm)
    result = {}
    for arm in ARMS:
        profiles = _correlations(ap, move, speech, flux, arm)
        best, measured = None, 0
        for width, start in product(WIDTHS, range(int(-LAGS[0]), tick_count, HOP)):
            keys = [(start - width, width), (start, width), (start + width, width)]
            if not all(key in profiles for key in keys):
                continue
            measured += 1
            three = [profiles[key] for key in keys]
            picked = _choice(three)
            if picked is None:
                continue
            score, delta, baseline, changed = picked
            if best is None or score > best["score"]:
                best = {"score": score, "relative_lag_ticks": delta, "flank_lag_ticks": baseline,
                        "changed_lag_ticks": changed, "extent_ticks": [start, start+width],
                        "roles": {role: [a, a + width] for role, (a, _) in zip(("before", "during", "after"), keys, strict=True)},
                        "lag_correlations": [profile.tolist() for profile in three]}
        result[arm] = {"compared_brackets": measured, "best": best}
    return result


def current(faces, patches, pcm, grid):
    """Use the current serving comparator's admission, attribution and timing arithmetic."""
    speech = lipsync.vad(pcm)
    plan = [window for window in lipsync.track_windows("track:0", faces, grid, speech, hop=xcorr.HOP_TICKS,
                                                       max_windows=xcorr.MAX_WINDOWS) if xcorr.full_search(window, speech.ticks)]
    ap = {tick: xcorr.aperture(value) for tick, value in patches.items()}
    windows = [window for window in plan if all(tick in ap for tick in window.ticks)
               and np.std([ap[tick] for tick in window.ticks]) >= lipsync.CORRELATION_VARIATION_MIN]
    if not windows or lipsync.speaker_score(windows, patches, speech.level) < lipsync.ATTRIBUTION_MIN:
        return {"measured_windows": 0, "ranges": []}
    scored = _track_sync(windows, ap, speech.level)
    return {"measured_windows": len(windows),
            "ranges": [{"kind": item.kind, "extent_ticks": [item.start_tick, item.end_tick],
                        "relative_lag_ticks": None if item.offset is None else item.offset - item.baseline}
                       for item in (*scored.steps, *scored.collapses)]}
