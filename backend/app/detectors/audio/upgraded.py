"""Pinned local audio inspection runtimes; no downloads or remote code execution.

The voice candidate follows the publisher's four-second preprocessing. Sound
tags describe the mixed source, and never enter the synthetic-voice decision.
"""
from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from torch import nn

SR = 16000


def prepare_voice_window(pcm: np.ndarray) -> np.ndarray:
    import pyloudnorm as pyln

    wav = np.asarray(pcm, np.float32).copy()
    if not 0 < len(wav) <= 4 * SR or not np.isfinite(wav).all():
        raise ValueError("Voice candidate requires a finite window of at most four seconds")
    wav -= wav.mean()
    if len(wav) > 480:
        wav = sosfiltfilt(butter(2, 20.0, btype="highpass", fs=SR, output="sos"), wav).astype(np.float32)
    loudness = pyln.Meter(SR).integrated_loudness(wav.astype(np.float64)) if len(wav) >= 6400 else -np.inf
    if not np.isfinite(loudness):
        loudness = 20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-9) - .691
    gain = float(np.clip(-23.0 - loudness, -40.0, 40.0))
    wav = np.clip(wav * 10 ** (gain / 20), -1, 1).astype(np.float32)
    return np.pad(wav, (0, 4 * SR - len(wav)))


class VoiceModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

        self.backbone = Wav2Vec2BertModel(Wav2Vec2BertConfig(**config))
        self.layer_weights = nn.Parameter(torch.zeros(self.backbone.config.num_hidden_layers + 1))
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, wav):
        from app.third_party.kaldi.fbank import fbank

        feats, lens = [], []
        for x in wav.float():
            nz = torch.nonzero(x).flatten()
            size = int(nz[-1]) + 1 if len(nz) else len(x)
            f = fbank(x[None, :max(size, 400)] * 32768.0, num_mel_bins=80,
                      sample_frequency=SR, dither=0.0, window_type="povey")
            f = (f - f.mean(0, keepdim=True)) / torch.sqrt(f.var(0, keepdim=True) + 1e-7)
            count = f.shape[0] // 2 * 2
            feats.append(f[:count].reshape(count // 2, 160))
            lens.append(count // 2)
        features = torch.zeros(len(feats), max(lens), 160, device=wav.device)
        mask = torch.zeros(len(feats), max(lens), dtype=torch.long, device=wav.device)
        for i, (f, length) in enumerate(zip(feats, lens, strict=True)):
            features[i, :length], mask[i, :length] = f, 1
        states = self.backbone(input_features=features, attention_mask=mask, output_hidden_states=True).hidden_states
        weights = torch.softmax(self.layer_weights, dim=0)
        hidden = torch.stack(states).mul(weights[:, None, None, None]).sum(0)
        masked = mask[:, :, None].float()
        hidden = (hidden * masked).sum(1) / masked.sum(1).clamp(min=1)
        return self.head(hidden).squeeze(-1)

    @torch.inference_mode()
    def predict(self, chunks: list[np.ndarray]) -> list[float]:
        device = next(self.parameters()).device
        # Batch one bounds the 25 retained 600M-backbone hidden states on laptops.
        return [float(torch.sigmoid(self(torch.from_numpy(prepare_voice_window(x))[None].to(device))).item())
                for x in chunks]


class SoundEventsModel:
    def __init__(self, checkpoint: Path, labels: Path, device: str):
        from app.third_party.panns.cnn10 import Cnn10

        self.model = Cnn10(32000, 1024, 320, 64, 50, 14000, 527)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        self.model.to(device).eval()
        self.device = device
        with labels.open() as source:
            self.labels = [row["display_name"] for row in csv.DictReader(source)]
        if len(self.labels) != 527:
            raise ValueError("Sound-label vocabulary differs from the checkpoint")

    @torch.inference_mode()
    def predict(self, pcm: np.ndarray) -> list[dict]:
        values = resample_poly(np.asarray(pcm, np.float32), 2, 1).astype(np.float32)
        values = np.pad(values, (0, max(0, 32000 - len(values))))
        scores = self.model(torch.from_numpy(values)[None].to(self.device))["clipwise_output"][0].cpu().numpy()
        order = np.argsort(-scores, kind="stable")[:3]
        return [{"label": self.labels[int(i)], "score": float(scores[i]), "class_index": int(i)} for i in order]


class SpeechActivity:
    def __init__(self, path: Path):
        self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.lock = threading.Lock()

    @torch.inference_mode()
    def spans(self, pcm: np.ndarray) -> list[tuple[float, float]]:
        # The recurrent state must not leak between recordings or concurrent runs.
        with self.lock:
            self.model.reset_states()
            positive = []
            for start in range(0, len(pcm), 512):
                chunk = np.pad(np.asarray(pcm[start:start+512], np.float32), (0, max(0, start+512-len(pcm))))
                if self.model(torch.from_numpy(chunk.copy()), SR).item() >= .5:
                    positive.append((start, min(len(pcm), start+512)))
            self.model.reset_states()
        merged = []
        for start, end in positive:
            if merged and start-merged[-1][1] <= 3200:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return [(max(0, a-480)/SR, min(len(pcm), b+480)/SR) for a, b in merged if b-a >= 4000]


def load_voice(model_id: str, device: str, dtype: str):
    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file, _revision

    entry = _entry(model_id)
    model = VoiceModel(json.loads(_file(entry, "w2v-bert-config.json").read_text()))
    model.load_state_dict(torch.load(_file(entry, "pellav2_detector.pt"), map_location="cpu", weights_only=True), strict=True)
    return LoadedModel(model.to(device).eval(), _revision(entry))


def load_sound_events(model_id: str, device: str, dtype: str):
    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file, _revision

    entry = _entry(model_id)
    return LoadedModel(SoundEventsModel(_file(entry, "panns-Cnn10.pth"), _file(entry, "class_labels_indices.csv"), device), _revision(entry))


def load_speech_activity(model_id: str, device: str, dtype: str):
    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file, _revision

    entry = _entry(model_id)
    return LoadedModel(SpeechActivity(_file(entry, "silero_vad.jit")), _revision(entry))
