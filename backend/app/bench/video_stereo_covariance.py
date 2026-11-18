# SPDX-License-Identifier: Apache-2.0
"""Measure a selective loss of stereo energy against both surrounding intervals."""

from __future__ import annotations

import numpy as np

from app.detectors.audio.stereo_ambience import BANDS, FRAME, HOP, RATE
from app.detectors.video.shared_sound import FLANK, GAP, WIDTHS
from app.evidence.video_silent_mouth import mouth_motion

ARMS = ("real", "complex")
LOAD = .04
FLOOR = 1e-10
MIN_POWER = 10 ** (-60 / 10)
MAX_OTHER_CHANGE_DB = 3.0
MAX_FLANK_SPREAD_DB = 9.0


def covariances(stereo: np.ndarray) -> np.ndarray:
    """Four bandwise, Hermitian 2-channel power matrices per analysis frame."""
    if stereo.ndim != 2 or stereo.shape[1] != 2 or len(stereo) < FRAME:
        raise ValueError("Expected continuous stereo samples on the 16 kHz clock")
    starts = np.arange(0, len(stereo) - FRAME + 1, HOP)
    frames = stereo[starts[:, None] + np.arange(FRAME)].astype(np.float64) / 32768
    window = np.hanning(FRAME)
    spectra = np.fft.rfft(frames * window[None, :, None], axis=1)
    bins = np.fft.rfftfreq(FRAME, 1 / RATE)
    scale = 2.0 / (FRAME * np.dot(window, window))
    bands = [spectra[:, (bins >= low) & (bins < high)] for low, high in BANDS]
    return scale * np.stack([np.einsum("nfc,nfd->ncd", band, band.conj()) for band in bands], axis=1)


def _mean(prefix, starts, width):
    return (prefix[starts + width] - prefix[starts]) / width


def _ratio_db(values):
    return 10 * np.log10(np.maximum(values, FLOOR))


def _directional_reading(before, center, after):
    # Use diagonal loading to learn the directions. Use measured power, without
    # the loading, to decide whether energy in either direction actually changed.
    load = np.maximum(LOAD * center.trace(axis1=-2, axis2=-1).real / 2, FLOOR)
    diagonal = load[..., None, None] * np.eye(2)
    values, vectors = np.linalg.eigh(center + diagonal)
    inverse_root = (vectors * (1 / np.sqrt(np.maximum(values, FLOOR)))[..., None, :]) @ vectors.conj().swapaxes(-1, -2)
    flank = (before + after) / 2 + diagonal
    _, axes = np.linalg.eigh(inverse_root @ flank @ inverse_root)
    transform = inverse_root @ axes
    transform /= np.linalg.norm(transform, axis=-2, keepdims=True)
    transpose = transform.conj().swapaxes(-1, -2)
    powers = [(transpose @ part @ transform).diagonal(axis1=-2, axis2=-1).real
              for part in (before, center, after)]
    left, right = (_ratio_db(powers[index] / np.maximum(powers[1], FLOOR)) for index in (0, 2))
    remaining = np.maximum(np.abs(left[..., 0]), np.abs(right[..., 0]))
    loss = np.minimum(left[..., 1], right[..., 1])
    stable = ((powers[1].min(axis=-1) >= MIN_POWER)
              & (remaining <= MAX_OTHER_CHANGE_DB) & (np.abs(left[..., 1] - right[..., 1]) <= MAX_FLANK_SPREAD_DB))
    margin = np.where(stable, loss - remaining, -140.)
    return margin, stable


def scan(stereo: np.ndarray, frames: list[dict]) -> dict:
    """Return the strongest audio-only and mouth-supported range for each arm."""
    values = covariances(stereo)
    picture = [frame for frame in frames if frame["gap_units"] is not None]
    output = {}
    distance = FLANK + GAP
    for arm in ARMS:
        data = values.real if arm == "real" else values
        prefix = np.concatenate((np.zeros_like(data[:1]), np.cumsum(data, axis=0)))
        ranked = []
        audio_assessed = picture_assessed = False
        for width in WIDTHS:
            index = np.arange(distance, len(data) - width - distance + 1)
            if not len(index):
                break
            before = _mean(prefix, index - distance, FLANK)
            center = _mean(prefix, index, width)
            after = _mean(prefix, index + width + GAP, FLANK)
            margins, stable = _directional_reading(before, center, after)
            eligible = stable.sum(axis=1) >= 2
            audio_assessed |= bool(eligible.any())
            edge = (FRAME - HOP) // 2
            if not picture_assessed:
                picture_assessed = any(mouth_motion(picture, int(start) * HOP + edge,
                                                     (int(start) + width) * HOP + edge) is not None
                                       for start in index[eligible])
            qualified = np.sort(margins, axis=1)[:, -2]
            for offset in np.flatnonzero(qualified > 0):
                start = int(index[offset]) * HOP + edge
                stop = start + width * HOP
                score = int(round(qualified[offset] * 1000))
                ranked.append((score, start, stop, margins[offset]))
        ranked.sort(key=lambda row: (-row[0], row[1] - row[2], row[1]))
        audio = visual = None
        for score, start, stop, bands in ranked:
            row = {"span": [start, stop], "margin_millidb": score,
                   "band_margins_millidb": np.rint(bands * 1000).astype(int).tolist()}
            if audio is None:
                audio = row
            if support := mouth_motion(picture, start, stop):
                measures, brackets = support
                visual = {**row, **measures, "picture_frames": brackets}
                break
        output[arm] = {"audio_assessed": audio_assessed, "picture_assessed": picture_assessed,
                       "audio_only": audio, "with_picture": visual}
    return output
