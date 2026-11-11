"""Two-channel frequency-domain separation for an offline voice study."""

from dataclasses import dataclass

import numpy as np

from scripts.audio_noise_conditioning_v211 import FFT, reconstruct, spectrum

RATE = 16_000
SAMPLES = 48_000
ITERATIONS = 18
RIDGE_RATIO = 1e-4
MIN_STEREO_RANK = 1e-4
MIN_CENTER_ALIGNMENT = 0.8
MIN_ALIGNMENT_MARGIN = 0.03
MIN_PROJECTED_RMS = 0.001
SPEECH_BAND = (250.0, 5000.0)
FREQUENCIES = np.fft.rfftfreq(FFT, 1.0 / RATE)
BAND = (SPEECH_BAND[0] <= FREQUENCIES) & (SPEECH_BAND[1] >= FREQUENCIES)


@dataclass(frozen=True)
class Projection:
    wave: np.ndarray
    coefficients: np.ndarray
    alignment: tuple[float, float]


def stereo_spectrum(stereo):
    left, size = spectrum(stereo[:, 0])
    right, _ = spectrum(stereo[:, 1])
    return np.stack((left, right), axis=-1), size


def apply_projection(stereo, coefficients):
    bins, size = stereo_spectrum(stereo)
    return reconstruct(np.einsum("tfc,fc->tf", bins, coefficients), size, SAMPLES)


def demix(bins):
    """Auxiliary-function iterative projection with one shared radius per source and frame."""
    identity = np.eye(2, dtype=np.complex128)
    initial = np.array(((1.0, 1.0), (1.0, -1.0))) / np.sqrt(2.0)
    transforms = np.broadcast_to(initial, (len(FREQUENCIES), 2, 2)).astype(np.complex128)
    ridge = (RIDGE_RATIO * np.mean(abs(bins) ** 2, axis=(0, 2)) + 1e-12)[:, None, None]
    for _ in range(ITERATIONS):
        sources = np.einsum("fsc,tfc->tfs", transforms, bins)
        radii = np.sqrt(np.mean(abs(sources[:, BAND]) ** 2, axis=1))
        for source in range(2):
            weights = 1.0 / np.maximum(radii[:, source], 1e-6)
            covariance = np.einsum("tfc,tfd,t->fcd", bins, bins.conj(), weights) / len(bins)
            covariance += ridge * identity
            target = np.broadcast_to(identity[:, source], transforms.shape[:2])
            update = np.linalg.solve(transforms @ covariance, target[..., None])[..., 0]
            scale = np.sqrt(np.einsum("fc,fcd,fd->f", update.conj(), covariance, update).real)
            transforms[:, source] = update.conj() / scale[:, None]
    return transforms


def separate(stereo):
    if stereo.shape != (SAMPLES, 2) or not np.isfinite(stereo).all():
        return "invalid_stereo", None
    centered = stereo - np.mean(stereo, axis=0)
    powers = np.linalg.eigvalsh(centered.T @ centered / SAMPLES)
    if powers[-1] < 1e-10 or powers[0] / powers[-1] < MIN_STEREO_RANK:
        return "rank_deficient", None

    bins, size = stereo_spectrum(stereo)
    try:
        transforms = demix(bins)
        mixing = np.linalg.inv(transforms)
    except np.linalg.LinAlgError:
        return "unstable_projection", None

    estimates = np.einsum("fsc,tfc->tfs", transforms, bins)
    if not (np.isfinite(mixing).all() and np.isfinite(estimates).all()):
        return "unstable_projection", None
    coefficients = mixing.mean(axis=1)[..., None] * transforms
    if np.max(abs(coefficients.sum(axis=1) - 0.5)) > 1e-8:
        return "unstable_projection", None

    # Weight alignment by physical image energy, including the individual microphone energy
    # of a side component which would otherwise cancel in the average microphone.
    physical_gain = np.linalg.norm(mixing, axis=1)
    center_alignment = abs(mixing.sum(axis=1)) / (np.sqrt(2.0) * physical_gain)
    energy = np.mean(abs(estimates) ** 2, axis=0) * physical_gain**2
    band_energy = energy[BAND].sum(axis=0)
    if np.any(band_energy <= 1e-10):
        return "ambiguous_center", None
    alignments = (center_alignment[BAND] * energy[BAND]).sum(axis=0) / band_energy
    center = int(np.argmax(alignments))
    if alignments[center] < MIN_CENTER_ALIGNMENT or abs(alignments[0] - alignments[1]) < MIN_ALIGNMENT_MARGIN:
        return "ambiguous_center", None

    selected = coefficients[:, center]
    selected_wave = reconstruct(np.einsum("tfc,fc->tf", bins, selected), size, SAMPLES)
    if (
        np.max(abs(selected_wave)) >= 32767.5 / 32768
        or np.sqrt(np.mean(selected_wave**2)) < MIN_PROJECTED_RMS
    ):
        return "out_of_range_projection", None
    return "available", Projection(selected_wave, selected, tuple(float(value) for value in alignments))


def supplement(current, candidate):
    """Use an admitted AI second opinion only for a currently inconclusive view."""
    return "ai" if current == "inconclusive" and candidate == "ai" else current
