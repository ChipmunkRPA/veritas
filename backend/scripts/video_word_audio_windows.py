# SPDX-License-Identifier: Apache-2.0
"""A short-interval acoustic reader with separate cepstral word features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from scipy.fft import dct
from torch import nn
from torch.nn import functional as F

from app.bench.video_word_content import overlap
from scripts.lombard_mouth_source_intake import index_zip
from scripts.video_lip_articulation_transfer import phone
from scripts.video_word_content_model import HOP, MEL, RATE, SYMBOLS, pin, word_spans

WINDOW = 320
STEP = 128
FFT = 512
COEFFICIENTS = 14
FRAMES = 16
INPUTS = FRAMES * (2 * COEFFICIENTS)
JITTER = ((0, 0), (-480, -480), (480, 480), (-640, 640), (640, -640))
SHIFTS = (0, -320, 320)
CLASS_CODES = "?" + SYMBOLS
WIN = np.hanning(WINDOW).astype(np.float32)
SCAN_HOP = 640
SCAN_DURATIONS = (3200, 4480, 5760, 7040, 8320, 9600)
SCAN_FRAME_MIN = 0.14
SCAN_FRAME_MARGIN = 0.05
SCAN_CROP_MARGIN = 0.20
SCAN_MAX_RECHECKS = 12
SCAN_AMBIGUITY = 0.045


class WindowWord(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.LayerNorm(INPUTS),
            nn.Linear(INPUTS, 96),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.16),
            nn.Linear(64, len(CLASS_CODES)),
        )


def crop_features(pcm, span):
    start, end = span
    if not 0 <= start < end <= len(pcm) or not int(0.09 * RATE) <= end - start <= int(0.72 * RATE):
        return None
    view = pcm[start:end].astype(np.float32) / 32768.0
    view = np.pad(view, (WINDOW // 2, WINDOW // 2), mode="reflect")
    voiced = np.max(np.abs(view))
    if voiced < 1e-3:
        return None
    view = np.diff(view, prepend=view[:1]) * 0.92 + view * 0.08
    cells = np.lib.stride_tricks.sliding_window_view(view, WINDOW)[::STEP]
    fft = np.fft.rfft(cells * WIN, n=FFT, axis=1)
    response = (fft.real**2 + fft.imag**2) @ MEL.T
    log = np.log(np.maximum(np.max(response) * 1e-5, response))
    spectrum = dct(log, type=2, norm="ortho", axis=1)[:, 1 : COEFFICIENTS + 1]
    times = np.linspace(0.0, len(spectrum) - 1, FRAMES)
    chosen = np.stack([np.interp(times, np.arange(len(spectrum)), values) for values in spectrum.T], axis=1)
    velocity = np.gradient(chosen, axis=0)
    return np.concatenate((chosen, velocity * 2), axis=1).reshape(-1).astype(np.float32)


def batch(model, observations):
    with torch.inference_mode():
        out = model(torch.from_numpy(np.stack(observations))).softmax(-1)
    return out.numpy()


def read_crop(model, pcm, span, expected):
    first, last = span
    views = []
    for shift in SHIFTS:
        view = crop_features(pcm, (first + shift, last + shift))
        if view is None:
            return None
        views.append(view)
    out = batch(model, views)
    ordered = out.argsort(axis=1)
    major = ordered[:, -1]
    label = CLASS_CODES[major[0]]
    expected_idx = CLASS_CODES.index(expected)
    return {
        "digit": label,
        "repeat": bool(np.all(major == major[0])),
        "score": float(np.min(out[:, expected_idx])),
        "margin": float(np.min(out[:, expected_idx] - np.max(np.delete(out, expected_idx, axis=1), axis=1))),
    }


def scan_soundtrack(model, pcm, frame_reading):
    """Propose one interval without the frame reader's selected word or bounds."""
    if frame_reading is None:
        return None
    logits, frames = frame_reading
    responses = logits[:frames, :].softmax(-1).numpy()[:, 1:]
    cumulative = np.vstack((np.zeros((1, len(SYMBOLS))), np.cumsum(responses, axis=0, dtype=np.float64)))
    proposals, observations = [], []
    padding = max(map(abs, SHIFTS))
    for duration in SCAN_DURATIONS:
        for start in range(padding, len(pcm) - duration - padding + 1, SCAN_HOP):
            span = (start, start + duration)
            inner = round(0.15 * duration)
            a, b = (start + inner + HOP - 1) // HOP, (span[1] - inner + HOP - 1) // HOP
            probabilities = (cumulative[b] - cumulative[a]) / (b - a)
            second, first = np.argsort(probabilities)[-2:]
            probability = float(probabilities[first])
            if probability < SCAN_FRAME_MIN or probability - probabilities[second] < SCAN_FRAME_MARGIN:
                continue
            view = crop_features(pcm, span)
            if view is not None:
                observations.append(view)
                proposals.append((span, int(first), probability))
    if not observations:
        return None

    ranked = []
    for (span, digit, density), observed in zip(proposals, batch(model, observations), strict=True):
        if int(np.argmax(observed)) == digit + 1:
            ranked.append((float(observed[digit + 1]) * density, span, digit, density))
    ranked.sort(key=lambda item: item[0], reverse=True)
    confirmed = []
    for _, span, digit, density in ranked[:SCAN_MAX_RECHECKS]:
        repeated = read_crop(model, pcm, span, SYMBOLS[digit])
        if repeated and repeated["repeat"] and repeated["margin"] >= SCAN_CROP_MARGIN:
            crop_score = repeated["score"]
            confirmed.append(
                {
                    "digit": SYMBOLS[digit],
                    "span": list(span),
                    "score": round((crop_score * density) ** 0.5, 6),
                    "frame_probability": round(density, 6),
                    "crop_probability": round(crop_score, 6),
                    "crop_margin": round(repeated["margin"], 6),
                }
            )
    if not confirmed:
        return None
    confirmed.sort(key=lambda item: item["score"], reverse=True)
    first = confirmed[0]
    if any(
        first["score"] - other["score"] < SCAN_AMBIGUITY
        and (first["digit"] != other["digit"] or overlap(first["span"], other["span"]) < 0.40)
        for other in confirmed[1:]
    ):
        return None
    return first


def prepare(plan, source, output):
    fit = json.loads(plan.read_bytes())
    vectors, labels, splits, ids = [], [], [], []
    with ZipFile(source / "alignment.zip") as align, ZipFile(source / "lombardgrid_audio.zip") as wave:
        readings = index_zip(align, ".json")
        sounds = index_zip(wave, ".wav")
        for item in fit:
            uid = item["uid"]
            wav, raw = wave.read(sounds[uid]), align.read(readings[uid])
            if pin(wav) != item["audio_sha256"] or pin(raw) != item["alignment_sha256"]:
                raise ValueError(f"Published audio or alignment differs for {uid}")
            pcm = phone.pcm(wav)
            words = word_spans(json.loads(raw)[uid])
            if words is None or list(words[4]) != item["span"]:
                raise ValueError("Wrong publisher digit interval")
            for slot, (start, end) in enumerate(words):
                label = CLASS_CODES.index(uid[-2]) if slot == 4 else 0
                for before, after in JITTER if item["split"] == "fit" else JITTER[:1]:
                    a, b = start + before, end + after
                    if (vector := crop_features(pcm, (a, b))) is None:
                        continue
                    vectors.append(vector)
                    labels.append(label)
                    splits.append(item["split"])
                    ids.append(uid)
    np.savez_compressed(
        output / "audio-word-inputs.npz",
        samples=np.stack(vectors),
        labels=np.asarray(labels),
        phases=np.asarray(splits),
        sources=np.asarray(ids),
    )
    print("Prepared", len(vectors), "audio word observations", flush=True)


def train(output):
    torch.manual_seed(271117)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(output / "audio-word-inputs.npz", allow_pickle=False) as data:
        values, labels = torch.from_numpy(data["samples"]), torch.from_numpy(data["labels"])
        fit = torch.from_numpy(np.flatnonzero(data["phases"] == "fit"))
        calibration = torch.from_numpy(np.flatnonzero(data["phases"] == "calibration"))
    model = WindowWord()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0016, weight_decay=0.009)
    counts = torch.bincount(labels[fit], minlength=len(CLASS_CODES)).float()
    weights = (counts[1:].mean() / counts).sqrt().clip(0.3, 2.0)
    logged = []
    for epoch in range(1, 23):
        model.train()
        losses = []
        for indexes in fit[torch.randperm(len(fit))].split(128):
            x = values[indexes] + torch.randn_like(values[indexes]) * 0.10
            logits = model(x)
            loss = F.cross_entropy(logits, labels[indexes], weight=weights, label_smoothing=0.04)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch in {4, 8, 12, 16, 20, 22}:
            model.eval()
            with torch.inference_mode():
                pred = model(values[calibration]).argmax(-1)
            words = labels[calibration] != 0
            entry = {
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 6),
                "digits": int(words.sum()),
                "correct_digits": int((pred[words] == labels[calibration][words]).sum()),
                "controls": int((~words).sum()),
                "non_digit_flags": int((pred[~words] != 0).sum()),
            }
            logged.append(entry)
            print(entry, flush=True)
    save_file(model.state_dict(), str(output / "audio-word.safetensors"))
    (output / "audio-word-fit.json").write_text(json.dumps(logged, indent=2) + "\n")


def load(path):
    model = WindowWord()
    model.load_state_dict(load_file(str(path)))
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.fit, args.source, args.output)
    train(args.output)
