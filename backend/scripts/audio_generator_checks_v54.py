"""Strict read-only checks for generator qualification; no serving imports."""
from __future__ import annotations

import hashlib
import math


def need(value, message):
    if not value:
        raise ValueError(message)


def checkpoint_keys(saved, expected):
    """Accept exact keys or a uniform documented DataParallel prefix, never partial loading."""
    keys, target = set(saved), set(expected)
    if keys == target:
        return {key: key for key in keys}
    if keys and all(key.startswith("module.") for key in keys):
        mapping = {key[7:]: key for key in keys}
        if set(mapping) == target:
            return mapping
    raise ValueError("Checkpoint does not cover every expected generator tensor exactly")


def strict_loaded_state(model, checkpoint):
    import torch

    modules = dict(model.named_children())
    need(isinstance(checkpoint, dict) and set(checkpoint) == set(modules), "Generator checkpoint has missing or unexpected modules")
    tensors, elements = 0, 0
    for name, module in modules.items():
        loaded, saved = module.state_dict(), checkpoint[name]
        need(isinstance(saved, dict), "Generator state must be a tensor mapping")
        mapping = checkpoint_keys(saved, loaded)
        for key, original in mapping.items():
            a, b = loaded[key], saved[original]
            need(isinstance(b, torch.Tensor) and a.device.type == b.device.type == "cpu"
                 and a.shape == b.shape and a.dtype == b.dtype, "Generator tensor shape, dtype or device differs")
            need(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and torch.equal(a, b),
                 "Generator tensor was partially loaded or differs from its pinned checkpoint")
            tensors += 1
            elements += b.numel()
    return {"tensors": tensors, "elements": elements, "all_tensors_exact": True}


def checked_phonemes(text, tokens, vocab):
    need(isinstance(text, str) and bool(text.strip()), "Authored nonempty text required")
    need(bool(tokens), "G2P returned no source tokens")
    for token in tokens:
        if any(character.isalnum() for character in token.text):
            need(isinstance(token.phonemes, str) and bool(token.phonemes), "G2P dropped a spoken source token")
    phonemes = "".join((token.phonemes or "") + (" " if token.whitespace else "") for token in tokens).strip()
    need(0 < len(phonemes) <= 510 and all(character in vocab for character in phonemes),
         "Unknown or overlong phonemes cannot be silently filtered/truncated")
    return phonemes


def waveform_summary(wave, sample_rate):
    import numpy as np

    values = np.asarray(wave)
    need(values.dtype == np.float32 and values.ndim == 1 and values.size > 0
         and bool(np.isfinite(values).all()), "Generator must return a finite mono float32 waveform")
    need(type(sample_rate) is int and sample_rate == 24000, "Declared native sample clock required")
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    need(math.isfinite(rms) and rms > 0, "Generator produced no audio energy")
    return {"samples": int(values.size), "sample_rate": sample_rate, "seconds": values.size / sample_rate,
            "rms": rms, "peak": float(np.abs(values).max()),
            "clipped_samples_if_pcm16": int((np.abs(values) > 1).sum()),
            "float32_sha256": hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()}
