# SPDX-License-Identifier: Apache-2.0
"""Compare raw separated voices with views made from the original mixture spectrum."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.signal import istft, stft

from scripts.audio_focused_context_transfer import read, save
from scripts.audio_two_three_separator_transfer import load_two, run_model

RATE = 16000
SAMPLES = 6 * RATE
REVISION = "21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5"
ASSETS = {
    "README.md": "5ed6424bf51523df4c10341b060812926982c5e0b5139ed3088e7cd2a83dc58a",
    "hyperparams.yaml": "eeef8d1450e1e06417eda07aa87ded71c6d9f3394852a92805e6aaf5229a50c6",
    "encoder.ckpt": "31d23d395a408b887b8f6ac01e477f00bcba27f95785426359fb52d52f1dc6ed",
    "decoder.ckpt": "2d959cb46de5b15f008bf5476cf7c19bc680b310c7e10883e3eddfdbde533cb8",
    "masknet.ckpt": "e5fb8c690668e5d1bbbc9a8256974577093a09cd08e845e9f30024fdd33472ce",
}
TRANSFORM = {"nperseg": 512, "noverlap": 384, "window": "hann"}
METHODS = ("separated", "mixture_phase")


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def open_pcm(bank, row):
    file = bank / "pcm" / f"{row['pcm_sha256']}.pcm"
    if file.stat().st_size != row["samples"] * 2:
        raise ValueError("PCM byte length differs from the source clock")
    body = file.read_bytes()
    if hashlib.sha256(body).hexdigest() != row["pcm_sha256"]:
        raise ValueError("PCM digest differs from the locked source")
    a, b = row["scope"]
    if not 0 <= a == b - SAMPLES <= len(body) // 2 - SAMPLES:
        raise ValueError("Expected exactly six seconds on the delivered sample clock")
    return np.frombuffer(body, dtype="<i2")[a:b].astype(np.float32) / 32768


def mixture_phase(mixture, estimates):
    """Take magnitudes from the two estimates and keep the recording's complex spectrum."""
    raw = stft(mixture, fs=RATE, **TRANSFORM)[2]
    proposed = stft(estimates, fs=RATE, **TRANSFORM)[2]
    power = np.abs(proposed) ** 2
    total = power.sum(axis=0)
    fraction = np.divide(power, total, out=np.full_like(power, .5), where=total > 1e-12)
    weights = .08 + .84 * fraction
    signals = istft(raw * weights, fs=RATE, **TRANSFORM)[1][:, :len(mixture)]
    if signals.shape != estimates.shape or np.max(np.abs(signals.sum(axis=0) - mixture)) > 2e-6:
        raise ValueError("The reconstructed views do not reproduce the recording on its sample clock")
    return signals


def decoded_views(model, mixture):
    estimates = run_model(model, mixture, 2).astype("<f4")
    recovered = mixture_phase(mixture, estimates).astype("<f4")
    return {"separated": estimates, "mixture_phase": recovered}


def extract(plan_path, data_root, assets, folder, split):
    import torch

    from scripts.audio_strong_traffic_view import network_free

    if {name: file_sha(assets / name) for name in ASSETS} != ASSETS:
        raise ValueError("Expected exact pinned Apache-2.0 model assets")
    network_free()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    plan = read(plan_path)
    folder.mkdir(mode=0o700, exist_ok=True)
    journal = folder / f"{split}.json"
    construction = {"plan_sha256": file_sha(plan_path), "revision": REVISION, "assets": ASSETS,
                    "code_sha256": file_sha(Path(__file__)), "runtime": str(torch.__version__)}
    if journal.exists():
        previous = read(journal)
        if previous["construction"] != construction:
            raise ValueError("Existing views belong to a different frozen experiment")
        packets = previous["waveforms"]
    else:
        packets = {}
    model = load_two(assets)
    rows = [row for row in plan["rows"] if row["split"] == split]
    started = time.monotonic()
    for index, row in enumerate(rows, 1):
        name = row["view_id"]
        destination = folder / f"{name}.npz"
        if name not in packets:
            mixture = open_pcm(data_root / plan["banks"][row["bank"]], row)
            views = decoded_views(model, mixture)
            with (temporary := destination.with_suffix(".partial")).open("wb") as stream:
                np.savez(stream, **views)
            temporary.replace(destination)
            packets[name] = {"file_sha256": file_sha(destination),
                             "mixture_sha256": hashlib.sha256(mixture.tobytes()).hexdigest()}
            snapshot = journal.with_suffix(".partial.json")
            snapshot.unlink(missing_ok=True)
            save(snapshot, {"construction": construction, "waveforms": packets})
            snapshot.replace(journal)
        if index == 1 or index % 12 == 0 or index == len(rows):
            print(f"Built {index}/{len(rows)} pairs of views in {time.monotonic() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "data", "assets", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("split", choices=("calibration", "held", "matched"))
    args = parser.parse_args()
    extract(args.plan, args.data, args.assets, args.output, args.split)
