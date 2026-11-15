# SPDX-License-Identifier: Apache-2.0
"""Load and run the byte-pinned speech phone checkpoint, entirely offline."""

from __future__ import annotations

import hashlib
import json
from itertools import groupby

RATE, STRIDE, CENTER = 16000, 320, 200
MODEL_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"
WEIGHT_PIN = "3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e"
MODEL_BLOBS = {
    "README.md": "d3d413893434f36cef6042639ab0be56769b95fe",
    "config.json": "c8fc04b8bd95c7a3361e606579a36f335a2b0c57",
    "preprocessor_config.json": "36ebe8b7c1cc967b3059f0494ae8a1069dd67655",
    "vocab.json": "14929da774c31eccf9e8c25fa3a966d8fc17d3fa",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def model_assets(path):
    for name, expected in MODEL_BLOBS.items():
        contents = (path / name).read_bytes()
        prefix = f"blob {len(contents)}\0".encode()
        require(hashlib.sha1(prefix + contents, usedforsecurity=False).hexdigest() == expected,
                f"Configuration file differs from the pinned publisher release: {name}")
    with (path / "pytorch_model.bin").open("rb") as handle:
        require(hashlib.file_digest(handle, "sha256").hexdigest() == WEIGHT_PIN, "Published checkpoint hash changed")
    readme = (path / "README.md").read_text()
    require("license: apache-2.0" in readme, "Commercial checkpoint terms were not verified")


def neural_decoder(path, *, names=("b", "bʲ", "bʰ", "bː", "p", "pʲ", "pʰ", "pː")):
    import numpy as np
    import torch
    from transformers import AutoModelForCTC, Wav2Vec2FeatureExtractor

    model_assets(path)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCTC.from_pretrained(path, local_files_only=True, trust_remote_code=False, dtype=dtype).to(device).eval()
    feature = Wav2Vec2FeatureExtractor.from_pretrained(path, local_files_only=True)
    vocabulary = json.loads((path / "vocab.json").read_text())
    phones = {vocabulary[name]: name[0] for name in names}
    hop, width = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride, strict=True):
        width += hop * (kernel - 1)
        hop *= step
    require((hop, width) == (STRIDE, 2 * CENTER) and feature.sampling_rate == RATE, "Acoustic clock changed")

    def infer(samples):
        material = samples.astype(np.float32) / 32768
        inputs = feature(material, sampling_rate=RATE, return_tensors="pt").input_values.to(device, dtype)
        with torch.inference_mode():
            outputs = model(inputs).logits.float().cpu()[0].softmax(-1)
        require(len(outputs) == (len(samples) - width) // STRIDE + 1, "CTC frame count does not match the input")
        maxima = outputs.argmax(1).tolist()
        events = []
        for token, positions in groupby(range(len(maxima)), key=maxima.__getitem__):
            if token not in phones:
                continue
            index = max(positions, key=lambda p: float(outputs[p, token]))
            value = float(outputs[index, token])
            if value >= .35:
                events.append([phones[token], index * STRIDE + CENTER, round(value, 6)])
        return events

    return infer, device
