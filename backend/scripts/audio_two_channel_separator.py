# SPDX-License-Identifier: Apache-2.0
"""Blind two-input separation for comparing source-time speech evidence."""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

FFT_SIZE = 1024
OVERLAP = 768
ITERATIONS = 24
RIDGE = 1e-5
MIN_RANK_RATIO = 1e-4


def separate(stereo: np.ndarray, *, frame_size: int = FFT_SIZE, whiten: bool = False) -> np.ndarray | None:
    """Estimate two tracks whose sum preserves the original stereo mean."""
    if stereo.ndim != 2 or stereo.shape[1] != 2 or len(stereo) < frame_size or not np.isfinite(stereo).all():
        return None
    stereo = stereo.astype(np.float64, copy=False)
    identity = np.eye(2)
    covariance = stereo.T @ stereo
    small, large = np.linalg.eigvalsh(covariance)
    if large <= 0. or small < MIN_RANK_RATIO * large:
        return None
    overlap = frame_size * OVERLAP // FFT_SIZE
    _, _, spectrum = stft(stereo, nperseg=frame_size, noverlap=overlap, axis=0)
    delivery = np.moveaxis(spectrum, 1, 2)
    channels = np.broadcast_to(identity, (delivery.shape[0], 2, 2)).astype(np.complex128).copy()
    power = np.mean(np.abs(delivery) ** 2, axis=(1, 2))
    regularize = (RIDGE * np.maximum(power, power.max() * 1e-8))[:, None, None] * identity
    if whiten:
        cross = np.einsum("ftc,ftd->fcd", delivery, delivery.conj()) / delivery.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(cross + regularize)
        channels = eigenvectors.conj().transpose(0, 2, 1) / np.sqrt(eigenvalues)[..., None]
    # Iterative projection updates one demixing row at a time.
    for _ in range(ITERATIONS):
        direct = np.einsum("fsc,ftc->fts", channels, delivery, optimize=True)
        radius = np.sqrt(np.mean(np.abs(direct) ** 2, axis=0))
        inverse = 1. / np.maximum(radius, radius.max(axis=0, keepdims=True) * 1e-5)
        for source in range(2):
            weighted = np.einsum("ftc,t,ftd->fcd", delivery, inverse[:, source], delivery.conj(), optimize=True)
            weighted = weighted / delivery.shape[1] + regularize * inverse[:, source].mean()
            basis = np.broadcast_to(identity[:, source:source+1], (len(channels), 2, 1))
            vector = np.linalg.solve(channels @ weighted, basis)[..., 0]
            norm = np.sqrt(np.real(np.einsum("fc,fcd,fd->f", vector.conj(), weighted, vector)))
            channels[:, source] = vector.conj() / norm[:, None]
    direct = np.einsum("fsc,ftc->fts", channels, delivery, optimize=True)
    # The mixing matrix maps both estimates back onto the submitted microphone clocks.
    # Averaging its rows restores the exact original mono mixture when summed.
    restored = direct * np.linalg.inv(channels).mean(axis=1)[:, None, :]
    output = np.stack([istft(restored[..., lane], nperseg=frame_size, noverlap=overlap)[1][:len(stereo)]
                       for lane in range(2)], axis=1)
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), stereo.mean(axis=1), rtol=0., atol=1e-8):
        return None
    return output
