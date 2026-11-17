"""Frame-aligned mouth and acoustic measurements for the local speech replacement study."""
from __future__ import annotations

from functools import cache

import numpy as np
from scipy.fft import dct

from app.evidence.video_viseme_articulation import visual_features

RATE = 16_000
HOP = 160
FFT = 512
WINDOW = 400
LANES = tuple(range(-6, 7))
CONTEXT = np.arange(-2, 3)
ACOUSTIC_CONTEXT = CONTEXT * 320
FRAMES = (9, 12, 15, 18, 20)


@cache
def _mel_filters():
    hz = np.fft.rfftfreq(FFT, 1 / RATE)
    mel_limits = 2595 * np.log10(1 + np.asarray((90, 7200)) / 700)
    boundaries = 700 * (10 ** (np.linspace(*mel_limits, 34) / 2595) - 1)
    lo, mid, hi = boundaries[:-2, None], boundaries[1:-1, None], boundaries[2:, None]
    filters = np.maximum(0, np.minimum((hz - lo) / (mid - lo), (hi - hz) / (hi - mid)))
    return filters / filters.sum(axis=-1, keepdims=True)


def acoustic(samples):
    values = np.asarray(samples, dtype=np.float64) / 32_768
    padded = np.pad(values, (WINDOW // 2, WINDOW // 2))
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP]
    power = np.abs(np.fft.rfft(frames * np.hanning(WINDOW), n=FFT)) ** 2
    bands = np.log(np.maximum(power @ _mel_filters().T, 1e-9))
    level = 10 * np.log10(np.maximum(np.mean(frames ** 2, axis=1), 1e-10))
    useful = level >= max(-61., float(np.quantile(level, .8)) - 24.)
    if useful.sum() < 24:
        return None
    spectral = dct(bands, norm="ortho", axis=1)[:, 1:14]
    relative_level = np.clip((level - np.quantile(level, .8)) / 20., -2., .9)
    center = np.median(spectral[useful], axis=0)
    features = np.column_stack((np.clip((spectral - center) / 5., -4., 4.), relative_level))
    return {"features": features.round(6), "active": useful}


def visual(frames, *, texture):
    measured = visual_features(frames, shape=True)
    if measured is None:
        return None
    _, indices, values = measured
    columns = np.asarray(values)
    eligible = np.zeros(len(frames), dtype=bool)
    features = np.zeros((len(frames), columns.shape[1] + (15 if texture else 0)))
    for index, row in zip(indices, columns, strict=True):
        appearance = frames[index].get("texture") if texture else []
        if appearance is not None:
            eligible[index] = True
            features[index] = [*row, *appearance]
    return {"features": features.round(6), "active": eligible}


def picture_context(values):
    positions = np.arange(2, len(values["active"]) - 2)
    indices = positions[:, None] + CONTEXT
    qualified = values["active"][indices].all(axis=1)
    positions, indices = positions[qualified], indices[qualified]
    return positions, values["features"][indices].reshape(len(indices), 5 * values["features"].shape[1])


def voice_context(values, centers):
    hops = np.rint((np.asarray(centers)[..., None] + ACOUSTIC_CONTEXT) / HOP).astype(int)
    available = (hops[..., 0] >= 0) & (hops[..., -1] < len(values["active"]))
    hops = np.clip(hops, 0, len(values["active"]) - 1)
    active = values["active"][hops].sum(axis=-1) >= 3
    shape = tuple(hops.shape[:-1]) + (hops.shape[-1] * values["features"].shape[1],)
    return values["features"][hops].reshape(shape), available & active


def fit_pairs(rows, arm, seed):
    """Read the same picture window with its own audio and two mismatched contexts."""
    rng = np.random.default_rng(seed)
    numbers, sounds, labels = [], [], []
    for row in rows:
        visual_data = row["visual"][arm]
        voice = row["audio"]
        if visual_data is None or voice is None:
            continue
        indices, pictures = picture_context(visual_data)
        times = row["times"][indices]
        matched, match_ok = voice_context(voice, times)
        for delay in (rng.integers(6, 14, size=len(times)), rng.integers(-13, -5, size=len(times))):
            other, other_ok = voice_context(voice, times + delay * 640)
            keep = match_ok & other_ok
            numbers.extend((pictures[keep], pictures[keep]))
            sounds.extend((matched[keep], other[keep]))
            labels.extend((np.ones(keep.sum()), np.zeros(keep.sum())))
    return tuple(np.concatenate(entries, axis=0).astype(np.float32) for entries in (numbers, sounds, labels))


def _scale(rows, mean, sigma):
    return ((rows - np.asarray(mean)) / np.asarray(sigma)).clip(-5, 5)


def predict(pictures, audio, fit):
    def embed(values, name):
        return np.tanh(_scale(values, fit[name + "_mean"], fit[name + "_scale"]) @ np.asarray(fit[name + "_weight"]).T
                       + np.asarray(fit[name + "_bias"]))

    visual, sound = embed(pictures, "visual"), embed(audio, "audio")
    joint = np.concatenate((visual * sound, np.abs(visual - sound)), axis=-1)
    hidden = np.tanh(joint @ np.asarray(fit["joint_weight"]).T + np.asarray(fit["joint_bias"]))
    return (hidden @ np.asarray(fit["output_weight"]).T + np.asarray(fit["output_bias"]))[..., 0]


def lane_scores(row, audio, fit, arm):
    view = row["visual"][arm]
    if view is None or audio is None:
        return None
    positions, windows = picture_context(view)
    if len(positions) < 24:
        return None
    times = row["times"]
    hop = round(np.median(np.diff(times)))
    sound, eligible = voice_context(audio, times[positions, None] + np.asarray(LANES) * hop)
    prices = predict(windows[:, None, :], sound, fit)
    filled = np.full((len(times), len(LANES)), np.nan)
    filled[positions] = np.where(eligible, prices, np.nan)
    return filled


def locate(times, scores):
    if scores is None:
        return None
    finite = np.isfinite(scores)
    counts = np.vstack((np.zeros(scores.shape[1], dtype=int), np.cumsum(finite, axis=0)))
    sums = np.vstack((np.zeros(scores.shape[1]), np.cumsum(np.where(finite, scores, 0.), axis=0)))

    def window(start, end, minimum):
        number = counts[end] - counts[start]
        values = (sums[end] - sums[start]) / np.maximum(1, number)
        return np.where(number >= minimum, values, -np.inf)

    best = None
    flank = 13
    for size in FRAMES:
        for begin in range(8, len(times) - size - 7, 2):
            end = begin + size
            before = window(max(0, begin - flank), begin, 6)
            after = window(end, min(len(times), end + flank), 6)
            both = np.minimum(before, after)
            lane = max(range(len(LANES)), key=lambda i: (both[i], -abs(LANES[i]), -LANES[i]))
            middle = window(begin, end, max(6, size // 2))
            # Allow one frame of additional timing variation before calling a mismatch.
            middle_value = max(middle[max(0, lane - 1):lane + 2])
            if not np.isfinite(both[lane]) or not np.isfinite(middle_value):
                continue
            score = float(both[lane] - middle_value)
            if best is None or score > best[0]:
                best = score, begin, end, lane
    if best is None:
        return None
    score, begin, end, lane = best
    offset = LANES[lane] * round(np.median(np.diff(times)))
    start, stop = int(times[begin] + offset), int(times[end] + offset)
    return {"score": round(score, 6), "span": [start, stop], "picture_frames": [begin, end], "lag": offset}
