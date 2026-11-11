"""Fixed two-microphone reference suppression for a speech-admission comparison."""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

SAMPLE_RATE = 16_000
SAMPLES = 48_000
FRAME = 512
HOP = 128
MIN_FREQUENCY = 150
MAX_FREQUENCY = 6000
RIDGE = 0.1
MAX_FILTER_MAGNITUDE = 1.5
SUBTRACTION = 0.9
ARMS = ("whole_clip", "quiet_third")


def _spectra(mono: np.ndarray, stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (mono.shape != (SAMPLES,) or stereo.shape != (SAMPLES, 2)
            or not np.isfinite(mono).all() or not np.isfinite(stereo).all()):
        raise ValueError("Three seconds on one finite 16 kHz stereo and mono clock are required")
    difference = (stereo[:, 0] - stereo[:, 1]) / 2
    values = np.stack([mono, difference]).astype(np.float64)
    frequency, time, spec = stft(values, fs=SAMPLE_RATE, window="hann", nperseg=FRAME, noverlap=FRAME-HOP,
                                 boundary="zeros", padded=True, axis=-1)
    return frequency, time, spec


def fit(mono: np.ndarray, stereo: np.ndarray, arm: str) -> np.ndarray:
    """Estimate a bounded noise transfer from the observed mixture alone."""
    if arm not in ARMS:
        raise ValueError("Unknown reference-suppression arm")
    frequency, time, (primary, reference) = _spectra(mono, stereo)
    fit_frames = np.flatnonzero((time * SAMPLE_RATE >= FRAME / 2) & (time * SAMPLE_RATE <= SAMPLES - FRAME / 2))
    if arm == "quiet_third":
        energies = np.sum(np.abs(primary[:, fit_frames]) ** 2, axis=0)
        fit_frames = fit_frames[np.argsort(energies, kind="stable")[:len(fit_frames) // 3]]
    signal, noise = primary[:, fit_frames], reference[:, fit_frames]
    cross_power = np.mean(signal * noise.conj(), axis=1)
    reference_power = np.mean(np.abs(noise) ** 2, axis=1)
    primary_power = np.mean(np.abs(signal) ** 2, axis=1)
    coefficient = cross_power / (reference_power + RIDGE * primary_power + 1e-12)
    magnitude = np.abs(coefficient)
    coefficient *= MAX_FILTER_MAGNITUDE / np.maximum(magnitude, MAX_FILTER_MAGNITUDE)
    coefficient[(frequency < MIN_FREQUENCY) | (frequency > MAX_FREQUENCY)] = 0
    return SUBTRACTION * coefficient


def apply(mono: np.ndarray, stereo: np.ndarray, coefficient: np.ndarray) -> np.ndarray:
    """Apply the frozen filter on the same clock without gain normalization."""
    _, _, (primary, reference) = _spectra(mono, stereo)
    if coefficient.shape != (len(primary),) or not np.isfinite(coefficient).all():
        raise ValueError("Reference-suppression coefficient geometry differs")
    conditioned = primary - coefficient[:, None] * reference
    _, samples = istft(conditioned, fs=SAMPLE_RATE, window="hann", nperseg=FRAME, noverlap=FRAME-HOP,
                       input_onesided=True, boundary=True)
    if samples.shape != (SAMPLES,) or not np.isfinite(samples).all():
        raise ValueError("Reference suppression changed the sample clock")
    return samples
