"""Pinned, offline IML-ViT evaluation; not enabled in the customer service.

The checkpoint owner permits MIT commercial use in upstream issue 60. The
publisher's decoder file cites a separately restricted SegFormer implementation;
this module does not import it. The small checkpoint-compatible head below uses
ordinary PyTorch resize/concatenate/convolution operations. The upstream ViTDet
encoder/FPN file is separately pinned and retains its Detectron2 attribution.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHECKPOINT_SHA256 = "7631fe852e139d904ad96fd766a37120d703c966cb80dfad70ffae4ff426cce4"
ENCODER_SHA256 = "382eabf15045420c827c5ad4bd1e44fa32b254b3c1139d6040f32df6a6bcdbde"
SOURCE_REVISION = "07dd2be0f4ea27a5c97c9fa5ffbe236733833eac"
SIDE = 1024
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def verify(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Pinned candidate asset changed: {path.name}")


def encoder_source(root: Path):
    path = root / "source/modules/window_attention_ViT.py"
    verify(path, ENCODER_SHA256)
    # Append, so isolated evaluation dependencies never replace the app's numpy,
    # torch, Pillow, or other installed runtime packages.
    if str(root / "deps") not in sys.path:
        sys.path.append(str(root / "deps"))
    spec = importlib.util.spec_from_file_location("chipmunk_iml_vit_encoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PredictionHead(nn.Module):
    """Checkpoint interface for a five-scale 256-channel prediction head."""

    def __init__(self) -> None:
        super().__init__()
        self.linear_fuse = nn.Conv2d(1280, 256, 1)
        self.norm = nn.BatchNorm2d(256)
        self.linear_predict = nn.Conv2d(256, 1, 1)

    def forward(self, scales: list[torch.Tensor]) -> torch.Tensor:
        target = scales[0].shape[-2:]
        combined = torch.cat([F.interpolate(s, target, mode="bilinear", align_corners=False) for s in scales], dim=1)
        return self.linear_predict(self.norm(self.linear_fuse(combined)))


class Candidate(nn.Module):
    def __init__(self, root: Path) -> None:
        super().__init__()
        architecture = encoder_source(root)
        self.encoder_net = architecture.ViT(
            img_size=SIDE, patch_size=16, embed_dim=768, depth=12, num_heads=12,
            drop_path_rate=.1, window_size=14, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            residual_block_indexes=[], use_rel_pos=True, out_feature="last_feat",
        )
        self.featurePyramid_net = architecture.SimpleFeaturePyramid(
            input_dim=768, out_channels=256, scale_factors=(4., 2., 1., .5),
            top_block=architecture.LastLevelMaxPool(), norm="LN",
        )
        self.predict_head = PredictionHead()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pyramid = self.featurePyramid_net(self.encoder_net(pixels))
        logits = self.predict_head(list(pyramid.values()))
        return F.interpolate(logits, (SIDE, SIDE), mode="bilinear", align_corners=False).sigmoid()


def preprocess(rgb: np.ndarray) -> torch.Tensor:
    """Publisher's top-left RGB padding, then ImageNet normalization; no resize."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Candidate expects uint8 RGB pixels")
    h, w = rgb.shape[:2]
    if not 0 < h <= SIDE or not 0 < w <= SIDE:
        raise ValueError("Candidate native-image scope is at most 1024 px per side")
    padded = np.zeros((SIDE, SIDE, 3), dtype=np.uint8)
    padded[:h, :w] = rgb
    values = (padded.astype(np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(values.transpose(2, 0, 1).copy()).unsqueeze(0)


def load(root: Path, device: str = "cpu") -> Candidate:
    checkpoint = root / "iml-vit_checkpoint.pth"
    verify(checkpoint, CHECKPOINT_SHA256)
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):
        raise ValueError("Checkpoint contains unsupported pickle globals")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model = Candidate(root)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


@torch.inference_mode()
def predict(model: Candidate, rgb: np.ndarray, device: str = "cpu") -> np.ndarray:
    h, w = rgb.shape[:2]
    response = model(preprocess(rgb).to(device))[0, 0, :h, :w].float().cpu().numpy()
    if not np.isfinite(response).all() or response.min() < 0 or response.max() > 1:
        raise ValueError("Candidate returned an invalid native mask")
    return response
