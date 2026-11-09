"""Verify complete saved weights and only the upstream's exact AdaIN identity state."""
from __future__ import annotations

import hashlib

from scripts.audio_generator_checks_v54 import need


def normalized_keys(saved):
    keys = set(saved)
    prefixed = {key for key in keys if key.startswith("module.")}
    need(not prefixed or prefixed == keys, "Mixed checkpoint prefixes are unsupported")
    return {key[7:] if prefixed else key: key for key in keys}


def strict_loaded_state(model, checkpoint, *, expected_identity_tensors=140):
    import torch

    modules = dict(model.named_children())
    need(isinstance(checkpoint, dict) and set(checkpoint) == set(modules), "Generator checkpoint modules differ")
    need(not model.training and not any(p.requires_grad for p in model.parameters()), "Generator must be frozen for inference")
    tensors = elements = identities = 0
    for name, module in modules.items():
        loaded, saved = module.state_dict(), checkpoint[name]
        need(isinstance(saved, dict), "Saved state must be a tensor mapping")
        mapping = normalized_keys(saved)
        need(set(mapping).issubset(loaded), "Unexpected saved generator tensors")
        missing = set(loaded) - set(mapping)
        for key in missing:
            need("." in key, "A learned generator tensor is missing")
            path, leaf = key.rsplit(".", 1)
            need(path.endswith(".norm") or path == "norm", "A learned generator tensor is missing")
            owner = module.get_submodule(path)
            parent = module.get_submodule(path.rsplit(".", 1)[0]) if "." in path else module
            need(type(parent).__name__ == "AdaIN1d" and type(parent).__module__ == "kokoro.istftnet"
                 and type(owner) is torch.nn.InstanceNorm1d and owner.affine and not owner.track_running_stats,
                 "Only the pinned upstream AdaIN identity workaround may add unsaved state")
            need({path + ".weight", path + ".bias"}.issubset(missing) and leaf in {"weight", "bias"},
                 "Identity normalization must be an entirely unsaved pair")
            value = loaded[key]
            identity = torch.ones_like(value) if leaf == "weight" else torch.zeros_like(value)
            need(value.device.type == "cpu" and value.dtype == torch.float32
                 and torch.equal(value, identity), "Additional normalization state is not exact identity")
            identities += 1
        for key, original in mapping.items():
            a, b = loaded[key], saved[original]
            need(isinstance(b, torch.Tensor) and a.device.type == b.device.type == "cpu"
                 and a.shape == b.shape and a.dtype == b.dtype, "Saved generator shape/dtype/device differs")
            need(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()) and torch.equal(a, b),
                 "A saved generator tensor was not loaded exactly")
            tensors += 1
            elements += b.numel()
    need(identities == expected_identity_tensors, "Unexpected number of upstream identity tensors")
    return {"checkpoint_tensors": tensors, "checkpoint_elements": elements,
            "all_checkpoint_tensors_exact": True, "extra_identity_tensors_verified": identities,
            "all_parameters_frozen": True}


def state_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        data = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0" + str(data.dtype).encode() + b"\0" + str(data.shape).encode() + b"\0")
        digest.update(data.tobytes())
    return digest.hexdigest()
