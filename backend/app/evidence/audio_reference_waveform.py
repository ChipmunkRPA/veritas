"""Find localized losses of matching waveform on two recording clocks."""
from __future__ import annotations

from dataclasses import dataclass

from app.evidence.audio_aligned_speech import PROCESSING_LIMIT, RATE, processing_residual

BASIS = "earlier_copy_waveform_continuity/1"
WIDTH = 1_920
HOP = 640
SEARCH = 12_800
FLANK = 10
MIN_MATCH = 0.25
PROMINENCE = 0.09
LAG_TOLERANCE = 5
MIN_ANCHORS = 6
MIN_DELAY_CHANGE = 256
MAX_RETAINED_SHARE = .38


@dataclass(frozen=True)
class Match:
    step: int
    lag: int
    gain: float


def whiten(first, second):
    import numpy as np
    from scipy.signal import welch

    hz, power = welch(second, fs=RATE, nperseg=512, average="median")
    floor = np.percentile(power[(hz >= 180) & (hz <= 6500)], 60) * .1
    if not floor > 0:
        return None
    weights = 1 / np.sqrt(np.maximum(power, floor))

    def filtered(signal):
        spectrum = np.fft.rfft(signal)
        hz_out = np.fft.rfftfreq(len(signal), 1 / RATE)
        gain = np.interp(hz_out, hz, weights)
        gain *= np.clip((hz_out - 150) / 100, 0, 1) * np.clip((6800 - hz_out) / 400, 0, 1)
        return np.fft.irfft(spectrum * gain, n=len(signal))

    return filtered(first), filtered(second)


def matches(first, second):
    import numpy as np
    from scipy.signal import correlate

    frames = np.lib.stride_tricks.sliding_window_view(first, WIDTH)[::HOP]
    powers = np.einsum("ij,ij->i", frames, frames)
    # A local projection is informative only when there is enough earlier sound.
    active = powers >= np.percentile(powers, 80) * .10
    totals = np.r_[0., np.cumsum(second**2)]
    target_powers = totals[WIDTH:] - totals[:-WIDTH]
    result = []
    for step in np.flatnonzero(active):
        if powers[step] <= 0:
            continue
        start = int(step) * HOP
        low = max(0, start - SEARCH)
        high = min(len(second) - WIDTH, start + SEARCH)
        if high < low:
            continue
        products = correlate(second[low:high + WIDTH], frames[step], mode="valid", method="fft")
        normalized = products / np.sqrt(powers[step] * np.maximum(target_powers[low:high + 1], 1e-15))
        index = int(np.argmax(normalized))
        strongest = float(normalized[index])
        if strongest < MIN_MATCH:
            continue
        runner_up = max(np.max(normalized[:max(0, index - 64)], initial=0),
                        np.max(normalized[index + 65:], initial=0))
        if strongest - runner_up >= PROMINENCE:
            result.append(Match(int(step), low + index - start, float(products[index] / powers[step])))
    return frames, powers, active, result


def stable_runs(anchors):
    runs = []
    for anchor in anchors:
        if runs and anchor.step - runs[-1][-1].step <= 4 and abs(anchor.lag - runs[-1][0].lag) <= LAG_TOLERANCE:
            runs[-1].append(anchor)
        else:
            runs.append([anchor])
    return [run for run in runs if len(run) >= 3]


def _measure(first, second):
    import numpy as np

    whitened = whiten(first, second)
    if whitened is None:
        return 0, []
    a, b = whitened
    frames, powers, active, anchors = matches(a, b)
    stable = stable_runs(anchors)
    support = sum(map(len, stable))
    candidates = []
    for before, after in zip(stable, stable[1:]):
        edge_before, edge_after = before[-1].step, after[0].step
        left = [anchor for anchor in before if edge_before - anchor.step < FLANK]
        right = [anchor for anchor in after if anchor.step - edge_after < FLANK]
        if any(len(flank) < 3 or (flank[-1].step - flank[0].step) * HOP < WIDTH + HOP
               for flank in (left, right)):
            continue
        lag_left = round(np.median([anchor.lag for anchor in left]))
        lag_right = round(np.median([anchor.lag for anchor in right]))
        if abs(lag_right - lag_left) < MIN_DELAY_CHANGE:
            continue
        # The last window before and first window after the gap already contain
        # matching waveform. Bound the missing interval by their centers.
        start = edge_before * HOP + WIDTH // 2
        end = edge_after * HOP + WIDTH // 2
        submitted = [start + lag_left, end + lag_right]
        if not (0 <= submitted[0] < submitted[1] <= len(b)
                and 3_200 <= end - start <= 38_400 and 1_920 <= submitted[1] - submitted[0] <= 38_400):
            continue
        gains_left = np.median([anchor.gain for anchor in left])
        gains_right = np.median([anchor.gain for anchor in right])
        expected = min(gains_left, gains_right)
        if expected <= 0 or expected / max(gains_left, gains_right) < .45:
            continue
        positions = [step for step in range(edge_before + 2, edge_after - 1)
                     if active[step] and powers[step] > 0
                     and all(0 <= step * HOP + lag <= len(b) - WIDTH for lag in (lag_left, lag_right))]
        if len(positions) < 3:
            continue
        ratios = []
        for step in positions:
            offset = step * HOP
            supported = max(np.dot(frames[step], b[offset+lag:offset+lag+WIDTH])
                            for lag in (lag_left, lag_right))
            ratios.append(max(0., supported / powers[step] / expected))
        matching = float(np.median(ratios))
        candidates.append({"earlier_span": [start, end], "span": submitted,
                           "lags": [lag_left, lag_right], "gains": [float(gains_left), float(gains_right)],
                           "matching_ratio": matching, "supports": [len(left), len(positions), len(right)]})
    candidates.sort(key=lambda item: (item["matching_ratio"], item["span"][0]))
    return support, candidates


def locate(first, second):
    if len(first) != len(second):
        return "equal_duration_required", 0, None
    anchors, candidates = _measure(first, second)
    if anchors < MIN_ANCHORS:
        return "insufficient_matching_waveform", anchors, None
    selected = next((candidate for candidate in candidates if candidate["matching_ratio"] < MAX_RETAINED_SHARE), None)
    if selected is None:
        return "no_local_difference", anchors, None
    residual = processing_residual(first, second)
    if residual is not None and residual <= PROCESSING_LIMIT:
        return "broad_processing_difference", anchors, None
    selected["gains"] = [round(gain, 6) for gain in selected["gains"]]
    selected["matching_ratio"] = round(selected["matching_ratio"], 6)
    return "localized_difference", anchors, selected
