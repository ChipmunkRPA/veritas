"""Read three preselected frozen speech layers without altering model outputs.

Research prototype: no production adapter, calibration or confidence admission.
The execution protocol must be frozen before any project-source neural forward.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

LAYER_IDS = (8, 16, 24)
HIDDEN_STATES = 25
WIDTH = 1024


def pool_layers(states, mask):
    if (
        not isinstance(states, (tuple, list))
        or len(states) != HIDDEN_STATES
        or not isinstance(mask, torch.Tensor)
        or mask.ndim != 2
        or mask.shape[0] != 1
        or not torch.all((mask == 0) | (mask == 1)).item()
        or mask.sum().item() <= 0
    ):
        raise ValueError("Exact25 states, one nonempty original binary attention mask required")
    pooled = []
    for index in LAYER_IDS:
        value = states[index]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (*mask.shape, WIDTH)
            or not value.is_floating_point()
            or not torch.isfinite(value).all().item()
        ):
            raise ValueError("Finite original per-layer clock/width required")
        value = value.detach().float()
        masked = mask.detach().to(value.device, dtype=torch.float32)[:, :, None]
        mean = (value * masked).sum(1) / masked.sum(1)
        pooled.append(mean[0].cpu().numpy().astype(np.float32, copy=True))
    values = np.stack(pooled)
    return {
        "layer_ids": list(LAYER_IDS),
        "layer_embeddings": values.tolist(),
        "layer_embedding_sha256": hashlib.sha256(np.asarray(values, "<f4").tobytes()).hexdigest(),
        "attention_mask_sha256": hashlib.sha256(
            np.asarray(mask.detach().cpu().numpy(), "<i8").tobytes()
        ).hexdigest(),
        "feature_frames": int(mask.shape[1]),
        "valid_feature_frames": int(mask.sum().item()),
    }


class LayerCapture:
    """Temporary read-only backbone hook, bounded to one original prediction."""

    def __init__(self, voice):
        self.records = []
        self.handle = voice.backbone.register_forward_hook(self._capture, with_kwargs=True)

    def _capture(self, _module, _args, kwargs, output):
        if "attention_mask" not in kwargs or not hasattr(output, "hidden_states"):
            raise ValueError("Original backbone mask and hidden-state output required")
        self.records.append(pool_layers(output.hidden_states, kwargs["attention_mask"]))
        # Returning None leaves the actual model output object untouched.

    def clear(self):
        self.records.clear()

    def take(self):
        if len(self.records) != 1:
            raise ValueError("Exactly one original four-second prediction required")
        return self.records.pop()

    def close(self):
        self.handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
