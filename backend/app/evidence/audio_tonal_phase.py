"""Locate temporary phase changes in persistent narrowband tones."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.signal import resample_poly

from app.evidence.audio_tonal_phase_binding import (
    FAMILIES,
    MAX_HARMONIC_RESIDUAL,
    MAX_INSIDE_DIFFERENCE,
    MAX_OUTSIDE_OFFSET,
    MIN_ABOVE_NEIGHBORS_DB,
    MIN_TONE_DBFS,
    PARTIALS,
    PHASE_FLOOR,
    WIDTHS,
)

RATE, ANALYSIS_RATE = 16000, 1000
HOP = 40
FRAME = 320
SIDE = 6
TONES = tuple(family * harmonic for family in FAMILIES for harmonic in PARTIALS)
PROBE_OFFSETS = (0, -9, 9, -14, 14)
MIN_PERSISTENT_FRAMES = 8
PERSISTENT_SHARE = .60


def _phase(value):
    return np.angle(np.exp(1j * value))


def _frames(pcm):
    track = resample_poly(pcm.astype(np.float64), 1, RATE // ANALYSIS_RATE)
    window = np.hanning(FRAME)
    signal = np.lib.stride_tricks.sliding_window_view(track, FRAME)[::HOP]
    starts = np.arange(len(signal)) * HOP
    frequencies = np.asarray(TONES)[:, None] + np.asarray(PROBE_OFFSETS)
    basis = np.exp(-2j * np.pi * np.arange(FRAME)[:, None] * frequencies.ravel()[None, :] / ANALYSIS_RATE)
    unit = ((signal * window) @ basis).reshape(len(signal), len(TONES), len(PROBE_OFFSETS)) * (2 / window.sum())
    tone = unit[:, :, 0] * np.exp(-2j * np.pi * starts[:, None] * np.asarray(TONES) / ANALYSIS_RATE)
    return tone, np.median(np.abs(unit[:, :, 1:]), axis=2)


def _supports(pcm):
    tone, sideband = _frames(pcm)
    strength = np.abs(tone)
    snr = 20 * np.log10(np.maximum(strength, 1e-9) / np.maximum(sideband, 1e-9))
    rms_dbfs = 20 * np.log10(np.maximum(strength / np.sqrt(2), 1e-9))
    unit = tone / np.maximum(strength, 1e-9)
    supported = (rms_dbfs[1:-1] >= MIN_TONE_DBFS) & (snr[1:-1] >= MIN_ABOVE_NEIGHBORS_DB)
    return (
        np.angle(unit[:-2] + unit[1:-1] + unit[2:]),
        supported & (np.abs(unit[:-2] + unit[1:-1] + unit[2:]) >= 2.75),
        snr[1:-1],
        rms_dbfs[1:-1],
    )


def _proposal(begin, end, column, deltas, delta, quality, levels):
    positions = [begin - SIDE, begin + SIDE, end - SIDE, end + SIDE]

    def samples(index):
        return (index * HOP + HOP + FRAME // 2) * (RATE // ANALYSIS_RATE)

    return {
        "samples": [samples(begin), samples(end)],
        "support_centers": [samples(index) for index in positions],
        "frequency_hz": TONES[column],
        "inside_offsets_rad": [round(float(value), 6) for value in deltas[:, column]],
        "outside_offset_rad": round(float(delta[column]), 6),
        "tone_above_neighbors_db": round(float(min(quality[positions, column])), 6),
        "tone_level_dbfs": round(float(min(levels[positions, column])), 6),
    }


def measure(pcm, *, persistent=False):
    """Return best clock-bound evidence for each rule, including pre-cutoff readings."""
    samples = np.asarray(pcm)
    if (samples.ndim != 1 or not 3 * RATE <= len(samples) <= 12 * RATE
            or not np.isfinite(samples).all() or np.max(np.abs(samples)) > 1):
        raise ValueError("Expected 3–12 seconds of continuous bounded mono PCM")
    phases, present, quality, levels = _supports(samples)
    arms = ("single", "harmonic", "persistent") if persistent else ("single", "harmonic")
    counts = dict.fromkeys(arms, 0)
    winners = dict.fromkeys(arms)
    ranks = dict.fromkeys(arms)
    cumulative = np.pad(np.cumsum(present, axis=0), ((1, 0), (0, 0))) if persistent else None
    for width in WIDTHS:
        start = np.arange(SIDE, len(phases) - width - SIDE)
        end = start + width
        left, middle_a, middle_b, right = (phases[indices] for indices in
                                            (start - SIDE, start + SIDE, end - SIDE, end + SIDE))
        supported = present[start - SIDE] & present[start + SIDE] & present[end - SIDE] & present[end + SIDE]
        outer = _phase(right - left)
        fraction = (2 * SIDE) / (width + 2 * SIDE)
        delta_a = _phase(middle_a - left - outer * fraction)
        delta_b = _phase(middle_b - left - outer * (1 - fraction))
        inner_variation = np.abs(_phase(delta_a - delta_b))
        raw_strength = np.minimum(np.abs(delta_a), np.abs(delta_b))
        coherent = supported & (np.abs(outer) <= MAX_OUTSIDE_OFFSET) & (inner_variation <= MAX_INSIDE_DIFFERENCE)
        # Prefer intervals with smaller unexplained motion at comparable offsets.
        scores = raw_strength - .45 * (inner_variation + np.abs(outer))
        counts["single"] += int(np.count_nonzero(supported))
        eligible = coherent & (raw_strength >= PHASE_FLOOR)
        single = np.where(eligible, scores, -np.inf)
        single_row, column = np.unravel_index(single.argmax(), single.shape)

        proposals = [("single", single_row, [column], single[single_row, column])]
        if cumulative is not None:
            left_size, right_size = start - SIDE + 1, len(phases) - end - SIDE
            left_count = cumulative[left_size]
            right_count = cumulative[-1] - cumulative[end + SIDE]
            enough_left = (left_size[:, None] >= MIN_PERSISTENT_FRAMES) & (left_count >= PERSISTENT_SHARE * left_size[:, None])
            enough_right = (right_size[:, None] >= MIN_PERSISTENT_FRAMES) & (right_count >= PERSISTENT_SHARE * right_size[:, None])
            stable = supported & enough_left & enough_right
            counts["persistent"] += int(np.count_nonzero(stable))
            scored = np.where(eligible & stable, scores, -np.inf)
            row, column = np.unravel_index(scored.argmax(), scored.shape)
            proposals.append(("persistent", row, [column], scored[row, column]))
        for family_idx in range(len(FAMILIES)):
            for i, j in combinations(range(len(PARTIALS)), 2):
                first, second = family_idx * len(PARTIALS) + i, family_idx * len(PARTIALS) + j
                counts["harmonic"] += int(np.count_nonzero(supported[:, first] & supported[:, second]))
                residual = np.maximum(
                    np.abs(_phase(PARTIALS[j] * delta_a[:, first] - PARTIALS[i] * delta_a[:, second])),
                    np.abs(_phase(PARTIALS[j] * delta_b[:, first] - PARTIALS[i] * delta_b[:, second])),
                )
                comparable = eligible[:, first] & eligible[:, second] & (residual <= MAX_HARMONIC_RESIDUAL)
                strength = np.where(comparable, np.minimum(scores[:, first], scores[:, second]), -np.inf)
                row = int(strength.argmax())
                proposals.append(("harmonic", row, [first, second], strength[row]))
        for name, row, columns, score in proposals:
            if not np.isfinite(score):
                continue
            key = (round(float(score), 6), -int(start[row]), -width)
            if ranks[name] is None or key > ranks[name]:
                ranks[name] = key
                readings = [_proposal(int(start[row]), int(end[row]), i, np.stack([delta_a[row], delta_b[row]]),
                                      outer[row], quality, levels) for i in columns]
                if name == "persistent":
                    column, = columns
                    readings[0]["outside_support_frames"] = [
                        [int(left_count[row, column]), int(left_size[row])],
                        [int(right_count[row, column]), int(right_size[row])],
                    ]
                winners[name] = {"score": key[0], "tones": readings}
    return counts, winners
