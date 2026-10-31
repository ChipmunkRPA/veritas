"""DDA, Dual Data Alignment (Chen, Xi, Yan et al., NeurIPS 2025): the inference path, written from the upstream code.

- **Model.** DINOv2 ViT-L/14 with rank-8 LoRA (alpha 1) on every linear layer named `attn.qkv`, `attn.proj`, `mlp.fc1`
  or `mlp.fc2`, and a 1024 → 1 head on the normalised class token: upstream `Inference/models/dinov2_models_lora.py`,
  `dinov2_models.py` and `lora.py` (github.com/roy-ch/Dual-Data-Alignment, Apache-2.0, commit 8b9c06e7). No upstream
  code is vendored, and the upstream `inference.py`, which hard-codes `cuda:0`, is never imported: the device and dtype
  come from the lease.
- **Backbone.** Upstream calls `torch.hub.load('facebookresearch/dinov2', ...)` over the network. Here the architecture
  comes from the verified `dinov2.vitl14` archive through `torch.hub.load(<dir>, source="local", pretrained=False)`.
  The pinned checkpoint carries every tensor (537, float32), so no DINOv2 pretrain checkpoint is read.
- **Preprocessing (ML N1, N2).** Native-resolution 336 px crops, never resampled (`native_crops`): 8-bit RGB scaled to
  [0, 1] and normalised with the CLIP mean and std, as upstream `test_transform` (CenterCrop 336, no resize).
- **Aggregation.** Mean logit over the crops (N2); the adapter keeps the extremes for the explanation.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from app.detectors.image import native_crops

CROP = 336
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
ENTRYPOINT = "dinov2_vitl14"
BACKBONE_MODEL = "dinov2.vitl14"
LORA_RANK, LORA_ALPHA = 8, 1.0
LORA_TARGETS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


class LoRALayer(nn.Module):
    """Low-rank update `(alpha / rank) · B A x` (upstream `LoRALayer`; parameter names `lora_A`, `lora_B`)."""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scale


class LoRALinear(nn.Module):
    """A frozen linear layer plus its LoRA update (upstream `LoRALinear`; children `original_layer`, `lora`)."""

    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.lora(x)


def add_lora(backbone: nn.Module, rank: int = LORA_RANK, alpha: float = LORA_ALPHA,
             targets: tuple[str, ...] = LORA_TARGETS) -> nn.Module:
    """Wrap every `nn.Linear` whose qualified name contains a target (upstream `apply_lora_to_linear_layers`)."""
    names = [name for name, module in backbone.named_modules()
             if isinstance(module, nn.Linear) and any(target in name for target in targets)]
    for name in names:
        parent_name, _, child = name.rpartition(".")
        parent = backbone.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha))
    return backbone


class DDANet(nn.Module):
    """The module tree of upstream `DINOv2ModelWithLoRA`: state-dict keys `base_model.model.*` and `base_model.fc.*`."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.model = add_lora(backbone)
        self.base_model.fc = nn.Linear(int(backbone.embed_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model.model.forward_features(x)["x_norm_clstoken"]
        return self.base_model.fc(features)


def build_backbone(hub_dir: Path) -> nn.Module:
    """DINOv2 ViT-L/14 architecture from a local, verified torch.hub directory (random init; no download, no bytecode)."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # the extracted tree is compared with the archive before every load: keep it exact
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available")
            return torch.hub.load(str(hub_dir), ENTRYPOINT, source="local", pretrained=False)
    finally:
        sys.dont_write_bytecode = previous


def load_model(checkpoint: Path, backbone: nn.Module, device: Any, dtype: str = "float32") -> nn.Module:
    """Load the pinned checkpoint's `model` state strictly (every key present, none left over); the optimizer state is
    discarded."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model = DDANet(backbone)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device=device, dtype=getattr(torch, dtype)).eval()


def crop_origins(width: int, height: int, crop: int = CROP) -> list[tuple[int, int, str]]:
    return native_crops.crop_origins(width, height, crop)


def to_batch(pixels: np.ndarray, origins: list[tuple[int, int, str]], crop: int = CROP) -> torch.Tensor:
    return native_crops.to_batch(pixels, origins, crop, MEAN, STD)


def logits(model: Any, batch: torch.Tensor, device: Any) -> list[float]:
    dtype = next((p.dtype for p in model.parameters()), torch.float32)
    with torch.inference_mode():
        out = model(batch.to(device=device, dtype=dtype))
    return [float(v) for v in out.detach().to("cpu", torch.float32).reshape(-1)]
