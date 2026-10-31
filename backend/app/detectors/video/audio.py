"""Audio deepfake (synthetic / cloned voice) detection on a video's sound track.

Three open classifiers vote on ≤10 s chunks of 16 kHz mono audio extracted with ffmpeg:
- MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection — Audio Spectrogram Transformer fine-tuned on ASVspoof 5
- Gustking/wav2vec2-large-xlsr-deepfake-audio-classification — wav2vec2-XLSR fine-tuned (real=0, fake=1)
- mo-thecreator/Deepfake-audio-detection — wav2vec2-base fine-tuned (fake=0, real=1)
Weights are set from scripts/eval_audio.py. Output is a Signal on the video verdict; it does not decide alone.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from app.core.config import settings
from app.core.device import DEVICE
from app.core.evidence import Signal
from app.core.logging import get_logger
from app.detectors.common.fusion import fuse_probability, prob_to_signal_fields
from app.detectors.common.registry import registry
from app.detectors.video.decode import _input

log = get_logger(__name__)
SR = 16000
CHUNK_S = 10.0
MAX_CHUNKS = 12
CLASSIFIER_BATCH = 4
VOICE_WINDOW_S = 4.0  # V-7 / DC-05: the v2 path scores speech in windows of at most 4 s
MAX_VOICE_WINDOWS = 60  # 120 s of extracted audio at a 2 s hop


def extract_audio(path: Path, max_seconds: float = 120.0) -> np.ndarray | None:
    """16 kHz mono float32 PCM via ffmpeg (fixed argument list, no shell, `file` protocol only). None if no audio stream."""
    if not shutil.which("ffmpeg"):
        return None
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file", "-i", _input(path), "-t", str(max_seconds),
               "-vn", "-ac", "1", "-ar", str(SR), "-f", "wav", str(wav)]
        try:
            subprocess.run(cmd, check=True, timeout=120, capture_output=True, stdin=subprocess.DEVNULL)
        except Exception as e:
            log.info("audio.extract_failed", error=str(e)[:120])
            return None
        if not wav.exists() or wav.stat().st_size < 1000:
            return None
        import wave

        with wave.open(str(wav), "rb") as w:
            frames = w.readframes(w.getnframes())
            width = w.getsampwidth()
        if width == 2:
            pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        return pcm


def is_mostly_silence(pcm: np.ndarray, thresh: float = 0.005) -> bool:
    return float(np.sqrt(np.mean(pcm**2))) < thresh


class AudioClassifier:
    def __init__(self, repo: str, positive: str):
        self.repo = repo
        self.device: Any = DEVICE
        self.fe = AutoFeatureExtractor.from_pretrained(repo)
        self.model = AutoModelForAudioClassification.from_pretrained(repo).to(DEVICE).eval()
        self._bind(positive)

    @classmethod
    def from_parts(cls, feature_extractor: Any, model: Any, positive: str, device: Any, repo: str = "") -> AudioClassifier:
        """A view over an already-loaded feature extractor and model (v2 adapters pass the leased model and its device)."""
        clf = cls.__new__(cls)
        clf.repo, clf.device, clf.fe, clf.model = repo, device, feature_extractor, model
        clf._bind(positive)
        return clf

    def _bind(self, positive: str) -> None:
        id2 = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.pos = next(i for i, l in id2.items() if positive in l)

    @torch.inference_mode()
    def predict(self, chunks: list[np.ndarray]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(chunks), CLASSIFIER_BATCH):
            batch = chunks[i : i + CLASSIFIER_BATCH]
            inputs = self.fe(batch, sampling_rate=SR, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            probs = torch.softmax(self.model(**inputs).logits.float(), dim=-1)[:, self.pos]
            out += probs.cpu().tolist()
        return out


def specs() -> list[tuple[str, str, float]]:
    out = []
    for raw in settings.audio_classifiers:
        repo, pos, w = (raw.split("|") + ["", "1.0"])[:3]
        out.append((repo.strip(), pos.strip().lower(), float(w)))
    return out


def get_audio_classifier(repo: str, positive: str) -> AudioClassifier:
    """Load-only compatibility getter; inference must use the scoped borrow."""
    return registry.get(f"audio:{repo}", lambda: AudioClassifier(repo, positive))


def borrow_audio_classifier(repo: str, positive: str) -> AbstractContextManager[AudioClassifier]:
    """Keep the legacy audio bundle leased across all classifier batches."""
    return registry.borrow(f"audio:{repo}", lambda: AudioClassifier(repo, positive))


def chunk_windows(pcm: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(start seconds, samples) of ≤ 10 s chunks of at least 1 s, thinned evenly to at most MAX_CHUNKS."""
    n = int(CHUNK_S * SR)
    starts = [i for i in range(0, len(pcm), n) if len(pcm[i : i + n]) >= SR]  # ≥ 1 s
    if len(starts) > MAX_CHUNKS:
        idx = np.linspace(0, len(starts) - 1, MAX_CHUNKS, dtype=int)
        starts = [starts[i] for i in idx]
    return [(i / SR, pcm[i : i + n]) for i in starts]


def chunk(pcm: np.ndarray) -> list[np.ndarray]:
    return [c for _, c in chunk_windows(pcm)]


def voice_windows(pcm: np.ndarray, hop_s: float, *, window_s: float = VOICE_WINDOW_S, max_windows: int = MAX_VOICE_WINDOWS,
                  keep: Callable[[float, float], bool] | None = None) -> list[tuple[float, np.ndarray]]:
    """(start seconds, samples) of ≤ `window_s` windows every `hop_s`, each of at least 1 s (V-7 voice lane, DC-05). A later
    window starts only while it reaches past the previous window's end, so the tail is covered once. `keep(start_s, end_s)`
    filters the windows first (A-1 speech gating). Above `max_windows` the windows with the most signal energy are kept, in
    time order (quick scores one; a short tail window loses to a full window of the same level)."""
    n, hop = int(round(window_s * SR)), max(1, int(round(hop_s * SR)))
    starts = [s for s in range(0, len(pcm), hop) if len(pcm) - s >= SR and (s == 0 or s - hop + n < len(pcm))]
    if keep is not None:
        starts = [s for s in starts if keep(s / SR, min(len(pcm), s + n) / SR)]
    if len(starts) > max_windows:
        energy = [float(np.sum(np.square(pcm[s:s + n], dtype=np.float64))) for s in starts]
        keep = sorted(sorted(range(len(starts)), key=lambda k: (-energy[k], k))[:max_windows])
        starts = [starts[k] for k in keep]
    return [(s / SR, pcm[s:s + n]) for s in starts]


def silent_signal(pcm_samples: int) -> tuple[Signal, dict]:
    return Signal(
        id="audio_silent", name="Sound track is silent", category="metadata", observed=f"RMS < 0.005 over {pcm_samples/SR:.1f}s",
        reference="Deepfake datasets and screen exports often carry a silent or stripped track", ai_probability=None, weight=0.0,
        direction="inconclusive", strength="none", reasoning="No audible content to assess; voice-clone detectors were not applied.",
    ), {"audio": "silent", "seconds": round(pcm_samples / SR, 1)}


def voice_signal(preds: dict[str, list[float]], weights: dict[str, float], pcm_samples: int, *,
                 windows_label: str = f"chunk(s) of ≤{int(CHUNK_S)}s") -> tuple[Signal, dict]:
    """The fused synthetic-voice signal over per-model chunk predictions (keys: model short names, in spec order)."""
    per_model = {k: [round(p, 4) for p in v] for k, v in preds.items()}
    p = fuse_probability([(float(np.mean(v)), weights[k]) for k, v in preds.items()])
    top = max(per_model.items(), key=lambda kv: float(np.mean(kv[1])))
    n_chunks = len(next(iter(preds.values())))
    sig = Signal(
        id="audio_synthetic_voice", name="Synthetic-voice detectors on the sound track", category="model",
        observed=f"fused p(synthetic)={p:.3f} over {n_chunks} {windows_label}; per model: " + ", ".join(f"{k}={np.mean(v):.2f}" for k, v in per_model.items()),
        reference="Trained on ASVspoof / in-the-wild spoofed speech vs bona fide recordings; 0.5 decision boundary",
        weight=settings.audio_weight, **prob_to_signal_fields(p),
        reasoning=("The voice carries spectral and prosodic artefacts that anti-spoofing models associate with text-to-speech or voice-conversion output." if p >= 0.5 else "The voice is classified as a bona fide recording by the anti-spoofing models."),
        details={"per_model_chunks": per_model, "seconds_analysed": round(pcm_samples / SR, 1), "strongest": top[0]},
    )
    return sig, {"audio": "analysed", "seconds": round(pcm_samples / SR, 1), "chunks": n_chunks, "p_synthetic": round(p, 4)}


def audio_signals(path: Path, has_audio: bool) -> tuple[list[Signal], dict]:
    """Signals for the video's sound track. Returns ([], info) when there is no usable speech audio."""
    if not has_audio or not settings.audio_enabled:
        return [], {"audio": "absent" if not has_audio else "disabled"}
    pcm = extract_audio(path)
    if pcm is None or len(pcm) < SR:
        return [], {"audio": "unreadable"}
    if is_mostly_silence(pcm):
        sig, info = silent_signal(len(pcm))
        return [sig], info
    chunks = chunk(pcm)
    preds: dict[str, list[float]] = {}
    weights: dict[str, float] = {}
    for repo, pos, w in specs():
        try:
            with borrow_audio_classifier(repo, pos) as classifier:
                try:
                    preds[repo.split("/")[-1]] = classifier.predict(chunks)
                    weights[repo.split("/")[-1]] = w
                finally:
                    del classifier
        except Exception as e:
            log.warning("audio.model_failed", model=repo, error=str(e)[:160])
    if not preds:
        return [], {"audio": "models_failed"}
    sig, info = voice_signal(preds, weights, len(pcm))
    return [sig], info
