"""Offline inference for the GenD Perception Encoder research checkpoint.

The published model uses dynamic 224-pixel crops, the pooled pre-projection
features of PE-Core Large, and a linear head on L2-normalized features. This
module has no network access and is not registered in the serving detector set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

BACKBONE = "vit_pe_core_large_patch14_336"
CROP = 224
_TENSOR_PREFIXES = {"feature_extractor.backbone.": "backbone.", "model.linear.": "head."}


class GenDPerceptionEncoder(torch.nn.Module):
    def __init__(self) -> None:
        import timm

        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=False, dynamic_img_size=True)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        data_config["input_size"] = (3, CROP, CROP)
        self.preprocess = timm.data.create_transform(**data_config, is_training=False)
        self.backbone.head = torch.nn.Identity()
        self.head = torch.nn.Linear(self.backbone.num_features, 2)

    def to_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        return torch.stack([self.preprocess(Image.fromarray(np.ascontiguousarray(crop))) for crop in crops])

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixels)
        return self.head(torch.nn.functional.normalize(features, p=2, dim=1))


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state = {}
    for key, tensor in load_file(str(path)).items():
        for source, target in _TENSOR_PREFIXES.items():
            if key.startswith(source):
                state[target + key.removeprefix(source)] = tensor
                break
        else:
            raise ValueError(f"Unexpected GenD PE tensor: {key}")
    return state


def load_model(path: Path, device: str) -> GenDPerceptionEncoder:
    """Load an already verified local safetensors file strictly, in float32."""
    model = GenDPerceptionEncoder()
    state = _checkpoint_state(path)
    if any(tensor.dtype != torch.float32 for tensor in state.values()):
        raise ValueError("The GenD PE checkpoint must contain float32 tensors")
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device).eval()
