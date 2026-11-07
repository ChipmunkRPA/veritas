"""Pinned, frozen noise-aware speech features; no serving adapter or AI head."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
ASSETS = {
    "pytorch_model.bin": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
    "config.json": "fea6df1c2700a3954fc07e70588aecc9055eeb28db2ff57151a2db0d19180ed4",
    "preprocessor_config.json": "99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2",
    "README.md": "d417fc16b08e2f66f0af6b371e47a5f9f1f596f6c1c593f1214d360c41d7a27e",
    "UniSpeech-LICENSE": "74295d561f60f770a4aa7525b71c0d119ec70422e9f5c601ee3c77e1b7822c91",
    "unilm-LICENSE": "904dc4d8749877f1dba1cda48200d2462dccbeb7c134d5e4ef6fa75e0198c8fe",
}
LAYER_IDS = (4, 8, 12)
WIDTH = 768
DIMENSIONS = 3 * 2 * WIDTH
SAMPLES = 64000
FRAMES = 199
SPEC = {
    "repository": "microsoft/wavlm-base-plus", "revision": REVISION,
    "assets_sha256": ASSETS, "parameters": 94381936,
    "weight_terms": "Model-card-linked UniSpeech CC-BY-SA-3.0; retain attribution/share-alike. Separate unilm code MIT.",
    "input": "Exact original mono16kHz64000 float32 samples; no normalization, denoising, padding or resampling",
    "feature_clock": {"frames": FRAMES, "stride_samples": 320, "receptive_field_samples": 400},
    "states": 13, "layer_ids": list(LAYER_IDS), "width": WIDTH,
    "pooling": "Layer order4,8,12; each masked population mean then standard deviation; float32",
    "runtime": "CPU float32, eager attention, eval, all parameters frozen; no custom remote code",
}


def assets(root):
    result = {}
    for name, expected in ASSETS.items():
        path = root / name
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != expected:
            raise ValueError("Pinned source, preprocessing, weight or license bytes changed: " + name)
        result[name] = {"sha256": checksum, "bytes": path.stat().st_size}
    return result


def load(root):
    from transformers import WavLMConfig, WavLMModel

    root = Path(root)
    assets(root)
    preprocessing = json.loads((root / "preprocessor_config.json").read_text())
    if preprocessing["sampling_rate"] != 16000 or preprocessing["do_normalize"] is not False:
        raise ValueError("Original16kHz unnormalized waveform contract required")
    if torch.serialization.get_unsafe_globals_in_checkpoint(root / "pytorch_model.bin"):
        raise ValueError("Checkpoint requires unexpected pickle globals")
    state = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
    config = WavLMConfig.from_json_file(root / "config.json")
    config._attn_implementation = "eager"
    model = WavLMModel(config).eval().requires_grad_(False)
    model.load_state_dict(state, strict=True)
    if (sum(p.numel() for p in model.parameters()) != SPEC["parameters"]
            or any(p.requires_grad or p.device.type != "cpu" or p.dtype != torch.float32
                   or not torch.isfinite(p).all().item() for p in model.parameters())):
        raise ValueError("Exact frozen finite CPU/float32 architecture required")
    return model


def pool(states, mask):
    if (not isinstance(states, (tuple, list)) or len(states) != 13
            or not isinstance(mask, torch.Tensor) or mask.ndim != 2 or mask.shape[0] != 1
            or not torch.all((mask == 0) | (mask == 1)).item() or mask.sum().item() <= 0):
        raise ValueError("Thirteen states and one nonempty binary feature mask required")
    output = []
    for layer in LAYER_IDS:
        state = states[layer]
        if (not isinstance(state, torch.Tensor) or state.shape != (*mask.shape, WIDTH)
                or not state.is_floating_point() or not torch.isfinite(state).all().item()):
            raise ValueError("Finite clock-matched768-wide selected states required")
        value = state.detach().float()[0, mask[0].bool()]
        output.extend((value.mean(0), value.std(0, unbiased=False)))
    vector = torch.stack(output).reshape(-1).cpu().numpy().astype(np.float32, copy=True)
    return {"wavlm": vector.tolist(),
            "wavlm_sha256": hashlib.sha256(np.asarray(vector, "<f4").tobytes()).hexdigest(),
            "feature_frames": int(mask.shape[1]), "valid_feature_frames": int(mask.sum().item())}


def extract(pcm, model):
    pcm = np.asarray(pcm)
    if pcm.dtype != np.float32 or pcm.shape != (SAMPLES,) or not np.isfinite(pcm).all():
        raise ValueError("Exact finite float32 original four-second PCM required")
    # Clone prevents a model from modifying the caller's source array.
    values = torch.from_numpy(np.ascontiguousarray(pcm).copy())[None]
    with torch.inference_mode():
        result = model(values, attention_mask=torch.ones_like(values, dtype=torch.long),
                       output_hidden_states=True, return_dict=True)
        if result.last_hidden_state.shape != (1, FRAMES, WIDTH):
            raise ValueError("Original four-second convolution clock changed")
        record = pool(result.hidden_states, torch.ones((1, FRAMES), dtype=torch.long))
    record["input_pcm_sha256"] = hashlib.sha256(np.asarray(pcm, "<f4").tobytes()).hexdigest()
    return record


def validate(record, source):
    if record["status"] != source["status"]:
        raise ValueError("Unassessed inputs cannot become assessed or original")
    keys = ("wavlm", "wavlm_sha256", "feature_frames", "valid_feature_frames", "input_pcm_sha256")
    if source["status"] == "not_applicable":
        if any(record[k] is not None for k in keys):
            raise ValueError("Unassessed inputs cannot carry candidate features")
        return
    if source["status"] != "ok":
        raise ValueError("Failed source cannot enter a candidate fit")
    values = np.asarray(record["wavlm"], np.float32)
    if (values.shape != (DIMENSIONS,) or not np.isfinite(values).all()
            or values.astype(np.float64).tolist() != record["wavlm"]
            or hashlib.sha256(np.asarray(values, "<f4").tobytes()).hexdigest() != record["wavlm_sha256"]
            or type(record["feature_frames"]) is not int or type(record["valid_feature_frames"]) is not int
            or record["feature_frames"] != FRAMES or record["valid_feature_frames"] != FRAMES
            or record["input_pcm_sha256"] != source["pcm_sha256"]):
        raise ValueError("Exact original PCM, clock and lossless feature payload required")
