"""LatentSync StableSyncNet as the lip-sync lane's embedder (V-5; VF §4.1; `video/lipsync.SyncEmbedder`).

Pre-processing follows LatentSync (github.com/bytedance/LatentSync @ a229c394: `latentsync/data/syncnet_dataset.py`,
`latentsync/utils/image_processor.py`, `latentsync/utils/affine_transform.py`, `latentsync/utils/audio.py`,
`configs/audio.yaml`, `configs/syncnet/syncnet_16_pixel_attn.yaml`, read 2026-09-13):

- **Face.** A similarity transform (upstream `transformation_from_points`, Procrustes, without its per-frame smoothing)
  maps the face into a 210 × 280 crop (template × 2.8, grey fill 127), resized to 256 × 256 with Lanczos. LatentSync takes
  the eyebrow and nose centres from InsightFace's 106 landmarks; Veritas has YuNet's five, so the eye centres and mouth
  corners are placed under the template's eyebrow line with ArcFace five-point proportions (inter-ocular distance = the
  template's eyebrow spacing, eyes 0.3 × that distance below the eyebrow line). The lower half (128 × 256) is kept.
- **Visual window.** 16 consecutive ticks, RGB scaled to [-1, 1], stacked as 48 channels.
- **Audio window.** The Wav2Lip mel spectrogram of the whole track (pre-emphasis 0.97, STFT n_fft 800, hop 200, Hann
  window 800, 80 mel bands 55–7600 Hz, dB − 20, symmetric normalisation clipped to ±4): 52 frames from 80 × tick / 25.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

SR = 16000
FRAMES = 16
RATIO = 2.8
FACE_SIZE = (210, 280)  # (w, h): upstream AlignRestore face_size (75 × 2.8, 100 × 2.8)
RESOLUTION = 256
FILL = 127
# eye centres (left, right in the image) and mouth corners in template units (upstream eyebrow centres x 17 and 58, y 20)
TEMPLATE = np.asarray([[17.0, 32.3], [58.0, 32.07], [20.79, 79.62], [54.74, 79.43]]) * RATIO
N_FFT, HOP, WIN, N_MELS, FMIN, FMAX = 800, 200, 800, 80, 55.0, 7600.0
PREEMPHASIS, MIN_LEVEL_DB, REF_LEVEL_DB, MAX_ABS = 0.97, -100.0, 20.0, 4.0
MEL_PER_TICK = 80.0 / 25.0
MEL_FRAMES = 52  # ceil(16 / 5 × 16)


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """2 × 3 matrix of the least-squares similarity src → dst (upstream `transformation_from_points`, no smoothing)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    c1, c2 = src.mean(axis=0), dst.mean(axis=0)
    p1, p2 = src - c1, dst - c2
    s1, s2 = p1.std(ddof=1), p2.std(ddof=1)
    u, _, vt = np.linalg.svd((p1 / s1).T @ (p2 / s2))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    sr = (s2 / s1) * r
    return np.hstack([sr, (c2 - sr @ c1)[:, None]])


def source_points(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """YuNet five landmarks → eye centres and mouth corners, each pair ordered left to right in the image."""
    eyes = sorted((tuple(landmarks[0]), tuple(landmarks[1])))
    mouth = sorted((tuple(landmarks[3]), tuple(landmarks[4])))
    return np.asarray([*eyes, *mouth], np.float64)


def align_lower_half(rgb: np.ndarray, landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = similarity(source_points(landmarks), TEMPLATE)
    face = cv2.warpAffine(np.ascontiguousarray(rgb), matrix, FACE_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(FILL, FILL, FILL))
    face = cv2.resize(face, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_LANCZOS4)
    return face[RESOLUTION // 2:]


def melspectrogram(pcm: np.ndarray) -> np.ndarray:
    import librosa
    from scipy import signal

    wav = signal.lfilter([1.0, -PREEMPHASIS], [1.0], np.asarray(pcm, np.float64))
    spectrum = np.abs(librosa.stft(y=wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN))
    mel = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX) @ spectrum
    db = 20.0 * np.log10(np.maximum(np.exp(MIN_LEVEL_DB / 20.0 * np.log(10.0)), mel)) - REF_LEVEL_DB
    return np.clip(2.0 * MAX_ABS * ((db - MIN_LEVEL_DB) / -MIN_LEVEL_DB) - MAX_ABS, -MAX_ABS, MAX_ABS).astype(np.float32)


def mel_windows(mel: np.ndarray, starts: Sequence[int]) -> np.ndarray:
    """(N, 1, 80, 52) mel windows at 80 × tick / 25; a window past the end repeats the last frame."""
    out = np.empty((len(starts), 1, mel.shape[0], MEL_FRAMES), np.float32)
    for i, s in enumerate(starts):
        a = int(MEL_PER_TICK * s)
        idx = np.clip(np.arange(a, a + MEL_FRAMES), 0, mel.shape[1] - 1)
        out[i, 0] = mel[:, idx]
    return out


class StableSyncNetEmbedder:
    frames = FRAMES
    min_confidence = 0.0
    collapse_confidence: float | None = None

    def __init__(self, model: Any, device: Any):
        self.model, self.device = model, device
        self._mel: tuple[int, int, np.ndarray] | None = None

    def crop(self, rgb: np.ndarray, frame: Mapping[str, Any]) -> np.ndarray | None:
        landmarks = frame.get("landmarks")
        if not landmarks or len(landmarks) < 5:
            return None
        return align_lower_half(rgb, landmarks)

    def _run(self, encoder: Any, x: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        dtype = next(self.model.parameters()).dtype
        with torch.inference_mode():
            out = encoder(torch.from_numpy(np.ascontiguousarray(x)).to(self.device, dtype))
            return F.normalize(out.reshape(out.shape[0], -1).float(), p=2, dim=1).cpu().numpy()

    def visual(self, windows: np.ndarray) -> np.ndarray:
        """(N, 16, 128, 256, 3) uint8 lower halves → (N, 2048)."""
        x = np.asarray(windows, np.float32).transpose(0, 1, 4, 2, 3) / 127.5 - 1.0  # b f c h w
        n, f, c, h, w = x.shape
        return self._run(self.model.visual_encoder, x.reshape(n, f * c, h, w))

    def audio(self, pcm: np.ndarray, starts: Sequence[int]) -> np.ndarray:
        """Audio embeddings of the windows at these tick starts; the track's mel spectrogram is computed once."""
        key = (id(pcm), len(pcm))
        if self._mel is None or self._mel[:2] != key:
            self._mel = (*key, melspectrogram(pcm))
        return self._run(self.model.audio_encoder, mel_windows(self._mel[2], starts))
