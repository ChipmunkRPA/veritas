# SPDX-License-Identifier: Apache-2.0
"""Train separate audible and visible word readers on genuine public recordings."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import defaultdict
from fractions import Fraction
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from app.bench.video_word_content import overlap
from app.detectors.video.viseme_patch import VisualSpeech
from scripts.video_lip_articulation_transfer import phone

RATE = 16_000
LEAD = 320
FRAMES = 12
SYMBOLS = "z123456789"
HOP = 160
FREQ_STEPS = 48
WINDOW = 400
FFT = 512
SCHEMA = "chipmunk.acoustic-digits/1"
LENGTH = 375


def json_file(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def pin(data):
    return hashlib.sha256(data).hexdigest()


def word_spans(phones):
    words = []
    start = None
    for info in phones:
        value = info["phone"]
        if value.startswith("SIL"):
            if start is not None:
                return None
            continue
        suffix = value.rsplit("_", 1)[-1]
        begin = round(RATE * Fraction(str(info["offset"])))
        end = round(RATE * (Fraction(str(info["offset"])) + Fraction(str(info["duration"]))))
        if suffix in ("S", "B"):
            if start is not None:
                return None
            start = begin
        if start is None or end <= begin or suffix not in ("S", "B", "I", "E"):
            return None
        if suffix in ("S", "E"):
            words.append((start, end))
            start = None
    if len(words) == 6 and all(a[1] <= b[0] for a, b in zip(words, words[1:])) and start is None:
        return words
    return None


def word_tensor(patches, valid, centers, span):
    a, b = span
    if not 1500 <= b - a <= 14300:
        return None
    desired = np.linspace(a - 320, b + 320, FRAMES) - LEAD
    location = np.searchsorted(centers, desired)
    right = np.clip(location, 1, len(centers) - 1)
    left = right - 1
    if len(centers) < FRAMES or (not valid[left].all()) or (not valid[right].all()):
        return None
    if min(desired) < centers[0] - 680 or max(desired) > centers[-1] + 680:
        return None
    fraction = np.clip((desired - centers[left]) / (centers[right] - centers[left]), 0, 1)[:, None, None]
    view = patches[left].astype(np.float32) * (1 - fraction) + patches[right].astype(np.float32) * fraction
    base = np.median(patches[valid].astype(np.float32), axis=0)
    offset = (view - base) / 32.0
    level = view.mean(axis=(1, 2), keepdims=True)
    absolute = (view - level) / view.std(axis=(1, 2), keepdims=True).clip(12)
    return np.stack((absolute, offset), axis=1).clip(-4.0, 4.0).astype(np.float32)


def visual_examples(plan, source, fit_capture, development_capture):
    plan = json_file(plan)
    records = {p["target"]["front_sha256"]: p for p in plan["rows"]}
    with ZipFile(source / "alignment.zip") as aligns:
        members = {Path(n).stem: n for n in aligns.namelist() if n.endswith(".json")}
        for fold in (fit_capture, development_capture):
            capture = json_file(fold / "capture.json.gz")
            for row in capture["rows"]:
                p = records[row["picture_sha256"]]
                if p["split"] not in {"fit", "calibration"}:
                    continue
                ref = p["target"]
                uid = ref["id"]
                raw = aligns.read(members[uid])
                if pin(raw) != ref["alignment_sha256"]:
                    raise ValueError("Alignment bytes differ")
                intervals = word_spans(json.loads(raw)[uid])
                if not intervals:
                    continue
                data = (fold / row["file"]).read_bytes()
                if pin(data) != row["sha256"]:
                    raise ValueError("Cached image bytes differ")
                with np.load(BytesIO(data), allow_pickle=False) as pack:
                    for i, (symbol, span) in enumerate(zip(uid.rsplit("_", 1)[-1], intervals, strict=True)):
                        item = word_tensor(pack["patches"], pack["valid"], pack["samples"], span)
                        if item is not None:
                            yield (p["split"], i, symbol, item, uid, p["speaker"], span)


def prepare_visual(plan, source, fit_capture, development_capture, output):
    data = list(visual_examples(plan, source, fit_capture, development_capture))
    np.savez_compressed(
        output / "word-tensors.npz",
        images=np.stack([row[3].astype(np.float16) for row in data]),
        indices=np.array([row[1] for row in data]),
        symbols=np.array([row[2] for row in data]),
        splits=np.array([row[0] for row in data]),
    )
    print(f"Prepared {len(data)} labeled word crops", flush=True)


class WordSequence(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.picture = VisualSpeech().picture
        self.context = nn.Sequential(
            nn.Conv1d(48, 72, 5, padding=2),
            nn.GroupNorm(6, 72),
            nn.GELU(),
            nn.Conv1d(72, 72, 3, padding=1),
            nn.GroupNorm(6, 72),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(72 * 4, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(0.18),
            nn.Linear(96, count),
        )

    def forward(self, array):
        batch, time = array.shape[:2]
        view = self.picture(array.flatten(0, 1)).reshape(batch, time, -1)
        return self.context(view.transpose(1, 2))


def evaluate(logits, labels, indices, names):
    score = {}
    for slot in range(6):
        mask = indices == slot
        classes = [i for i, name in enumerate(names) if name.startswith(str(slot) + ":")]
        subset = logits[mask].clone()
        positive = subset[:, classes].argsort(dim=1, descending=True)
        candidates = torch.tensor(classes)[positive]
        truth = labels[mask][:, None]
        score[slot] = [
            int(mask.sum()),
            int((truth[:, 0] == candidates[:, 0]).sum()),
            int((truth == candidates[:, :3]).any(dim=1).sum()),
        ]
    return score


def train_visual(folder, base_weights):
    torch.manual_seed(280517)
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(folder / "word-tensors.npz", allow_pickle=False) as data:
        train = data["splits"] == "fit"
        images = torch.from_numpy(data["images"])
        slots = data["indices"]
        words = [f"{slot}:{word}" for slot, word in zip(slots, data["symbols"], strict=True)]
    names = sorted({w for w, fit in zip(words, train, strict=True) if fit})
    diction = {word: i for i, word in enumerate(names)}
    labels = torch.tensor([diction.get(word, -100) for word in words]).long()
    fit = torch.from_numpy(np.flatnonzero(train))
    cal = torch.from_numpy(np.flatnonzero(~train))
    weights = labels[fit].bincount(minlength=len(names)).float()
    weights = (weights.mean() / weights).sqrt().clip(0.5, 2)
    model = WordSequence(len(names))
    original = load_file(str(base_weights))
    model.picture.load_state_dict(
        {k.removeprefix("picture."): v for k, v in original.items() if k.startswith("picture.")}
    )
    optim = torch.optim.AdamW(
        [
            {"params": model.picture.parameters(), "lr": 0.0006},
            {"params": model.context.parameters(), "lr": 0.0025},
        ],
        weight_decay=0.006,
    )
    epochs = []
    start = time.monotonic()
    for iteration in range(1, 25):
        model.train()
        scores = []
        for items in fit[torch.randperm(len(fit))].split(36):
            x = images[items].float()
            flipped = torch.rand(len(items)) < 0.5
            x[flipped] = x[flipped].flip(-1)
            x = x * (0.83 + 0.34 * torch.rand(len(x), 1, 1, 1, 1))
            x = x + torch.randn_like(x) * 0.035
            if torch.rand(()) < 0.6:
                shift = int(torch.randint(-1, 2, ()).item())
                x = torch.roll(x, shift, dims=-1)
            scores_out = model(x)
            loss = F.cross_entropy(scores_out, labels[items], weight=weights, label_smoothing=0.045)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scores.append(float(loss.detach()))
        if iteration % 3 == 0 or iteration == 1:
            model.eval()
            with torch.inference_mode():
                scores_out = model(images[cal].float())
            summary = evaluate(scores_out, labels[cal], torch.from_numpy(slots)[cal], names)
            row = {"epoch": iteration, "loss": round(float(np.mean(scores)), 4), "dev": summary}
            epochs.append(row)
            print(json.dumps(row), round(time.monotonic() - start, 1), flush=True)
            save_file(model.state_dict(), str(folder / f"word-epoch-{iteration:02d}.safetensors"))
    (folder / "word-fit.json").write_text(json.dumps({"classes": names, "epochs": epochs}, indent=2) + "\n")


def mel_bank():
    coords = np.arange(FFT // 2 + 1) * RATE / FFT
    limits = 2595 * np.log10(1 + np.asarray([55.0, 7600.0]) / 700)
    bands = np.linspace(*limits, FREQ_STEPS + 2)
    edges = 700 * (10 ** (bands / 2595) - 1)
    return np.stack(
        [
            np.maximum(0, np.minimum((coords - a) / (b - a), (c - coords) / (c - b)))
            for a, b, c in zip(edges, edges[1:], edges[2:])
        ]
    ).astype(np.float32)


MEL = mel_bank()
WIN = np.hanning(WINDOW).astype(np.float32)


def features(pcm):
    sound = np.asarray(pcm, dtype=np.float32) / 32768
    window = np.lib.stride_tricks.sliding_window_view(np.pad(sound, WINDOW // 2), WINDOW)[::HOP]
    fft = np.fft.rfft(window * WIN, n=FFT, axis=1)
    db = (np.log10((fft.real**2 + fft.imag**2) @ MEL.T + 1e-7) * 10.0).astype(np.float32)
    value = (db - np.median(db, axis=0)) / 14.0
    relative = (db - db.mean(axis=1, keepdims=True)) / 20.0
    delta = np.diff(value, axis=0, prepend=value[:1])
    return np.concatenate((value, relative, delta), axis=1).clip(-6.0, 6.0).astype(np.float32)


class AcousticDigits(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(FREQ_STEPS * 3, 96, 5, padding=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv1d(96, 96, 5, padding=4, dilation=2),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Conv1d(96, 96, 5, padding=8, dilation=4),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Conv1d(96, 64, 3, padding=7, dilation=7),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, len(SYMBOLS) + 1, 1),
        )

    def forward(self, x):
        return self.layers(x.transpose(1, 2)).transpose(1, 2)


def acoustic_examples(plan, source):
    rows = json_file(plan)["rows"]
    fit_speakers = {row["target"]["id"].split("_")[0] for row in rows if row["split"] == "fit"}
    original = {row["target"]["id"]: row["target"] for row in rows if row["split"] in ("fit", "calibration")}
    calibrate = [row["target"]["id"] for row in rows if row["split"] == "calibration"]
    with (
        ZipFile(source / "publisher-json.zip") as info,
        ZipFile(source / "alignment.zip") as labeled,
        ZipFile(source / "lombardgrid_audio.zip") as wav,
    ):
        media = {Path(path).stem: path for path in wav.namelist() if path.endswith(".wav")}
        labels = {Path(path).stem: path for path in labeled.namelist() if path.endswith(".json")}
        pool = defaultdict(set)
        for path in info.namelist():
            if not path.endswith(".json") or Path(path).name.startswith("._"):
                continue
            for row in json.loads(info.read(path)):
                actor, cond = row["SPKR"], row["COND"]
                if actor in fit_speakers and cond in {"p", "l"} and row["STATUS"] == "CORRECT":
                    uid = f"{actor}_{cond}_{row['UTTERANCE']}"
                    if uid in media and uid in labels:
                        pool[actor, cond].add(uid)
        output = []
        for key, values in sorted(pool.items()):
            indices = sorted(values, key=lambda value: pin(f"{SCHEMA}/{value}".encode()))[:42]
            output.extend(("fit", uid) for uid in indices)
        output.extend(("calibration", uid) for uid in calibrate)
        for phase, uid in output:
            audio = wav.read(media[uid])
            timing = labeled.read(labels[uid])
            if uid in original:
                record = original[uid]
                assert pin(audio) == record["audio_sha256"] and pin(timing) == record["alignment_sha256"]
            span = word_spans(json.loads(timing)[uid])
            if not span:
                continue
            pcm = phone.pcm(audio)
            word = uid[-2]
            if len(pcm) // HOP >= LENGTH or word not in SYMBOLS or span[4][1] > len(pcm):
                continue
            yield (
                {
                    "split": phase,
                    "uid": uid,
                    "digit": word,
                    "span": span[4],
                    "audio_sha256": pin(audio),
                    "alignment_sha256": pin(timing),
                },
                features(pcm),
            )


def prepare_acoustic(plan, source, output):
    rows = list(acoustic_examples(plan, source))
    record = [row for row, _ in rows]
    n = len(rows)
    mel = np.zeros((n, LENGTH, FREQ_STEPS * 3), dtype=np.float16)
    target = np.full((n, LENGTH), -100, dtype=np.int8)
    for i, (info, value) in enumerate(rows):
        mel[i, : len(value)] = value
        target[i, : len(value)] = 0
        a, b = info["span"]
        selected = (a <= np.arange(len(value)) * HOP) & (np.arange(len(value)) * HOP < b)
        target[i, : len(value)][selected] = 1 + SYMBOLS.index(info["digit"])
    np.savez_compressed(output / "acoustic.npz", value=mel, target=target)
    (output / "acoustic-source.json").write_text(json.dumps(record, sort_keys=True) + "\n")
    print(
        "Acoustic corpus",
        n,
        {k: sum(row["split"] == k for row in record) for k in ("fit", "calibration")},
        flush=True,
    )


def predict(scores, length, *, mask_point=0.5):
    pr = scores.softmax(dim=-1).detach().cpu().numpy()[:length]
    present = 1.0 - pr[:, 0]
    axes = np.flatnonzero(present >= mask_point)
    if len(axes) == 0:
        return None
    pieces = np.split(axes, np.flatnonzero(np.diff(axes) > 3) + 1)
    rank = []
    for part in pieces:
        n = int(part[-1] - part[0] + 1)
        if not 11 <= n <= 62 or len(part) < 0.75 * n:
            continue
        avg = pr[part, 1:].mean(axis=0)
        best = np.argsort(avg)[-2:]
        main = int(best[-1])
        margin = float(avg[main] - avg[best[0]])
        value = float(avg[main])
        a, b = int(part[0]) * HOP, max(0, int(part[-1] + 1)) * HOP
        rank.append(
            {
                "digit": SYMBOLS[main],
                "span": [a, b],
                "posterior": value,
                "margin": margin,
                "measure": float(present[part].mean()),
            }
        )
    rank.sort(key=lambda row: row["posterior"], reverse=True)
    if rank and (len(rank) == 1 or rank[0]["posterior"] - rank[1]["posterior"] > 0.10):
        return rank[0]
    return None


def infer(model, pcm):
    x = features(pcm)
    if len(x) >= LENGTH:
        return None
    item = np.zeros((LENGTH, FREQ_STEPS * 3), dtype=np.float32)
    item[: len(x)] = x
    with torch.inference_mode():
        scores = model(torch.from_numpy(item)[None])[0]
    return predict(scores, len(x))


def train_acoustic(output):
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101974)
    rows = json.loads((output / "acoustic-source.json").read_text())
    fit = torch.tensor([i for i, row in enumerate(rows) if row["split"] == "fit"])
    trial = torch.tensor([i for i, row in enumerate(rows) if row["split"] != "fit"])
    with np.load(output / "acoustic.npz", allow_pickle=False) as pack:
        x = torch.from_numpy(pack["value"])
        y = torch.from_numpy(pack["target"])
    weight = torch.tensor([0.48] + [1.0] * 10)
    model = AcousticDigits()
    optim = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.006)
    start = time.monotonic()
    journal = []
    for epoch in range(1, 19):
        model.train()
        losses = []
        for indexes in fit[torch.randperm(len(fit))].split(32):
            audio = x[indexes].float()
            audio = audio * (0.85 + 0.3 * torch.rand(len(audio), 1, 1)) + torch.randn_like(audio) * 0.045
            logits = model(audio)
            loss = F.cross_entropy(
                logits.flatten(0, 1),
                y[indexes].flatten().long(),
                ignore_index=-100,
                weight=weight,
                label_smoothing=0.03,
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            losses.append(float(loss.detach()))
        if epoch % 2 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                outcome = model(x[trial].float())
            result = [
                predict(scores, int((y[i] >= 0).sum()))
                for i, scores in zip(trial.tolist(), outcome, strict=True)
            ]
            named = sum(item is not None for item in result)
            correct = sum(
                item and item["digit"] == rows[i]["digit"] and overlap(item["span"], rows[i]["span"]) >= 0.4
                for i, item in zip(trial.tolist(), result, strict=True)
                if item
            )
            row = {
                "epoch": epoch,
                "loss": round(float(np.mean(losses)), 4),
                "qualified": named,
                "correct": correct,
                "total": len(trial),
            }
            journal.append(row)
            print(row, round(time.monotonic() - start, 1), flush=True)
            save_file(model.state_dict(), str(output / f"acoustic-epoch-{epoch:02d}.safetensors"))
    (output / "acoustic-fit.json").write_text(json.dumps(journal, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    phases = {
        "visual-inputs": (prepare_visual, ("plan", "source", "fit_capture", "development_capture", "output")),
        "fit-visual": (train_visual, ("folder", "base_weights")),
        "audio-inputs": (prepare_acoustic, ("plan", "source", "output")),
        "fit-audio": (train_acoustic, ("output",)),
    }
    for name, (_, fields) in phases.items():
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    phases[action][0](**args)


if __name__ == "__main__":
    main()
