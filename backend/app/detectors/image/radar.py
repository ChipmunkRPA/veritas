"""RADAR, diffusion-inpainting localisation (Costanzino, Bayliss, Sock et al., NeurIPS 2025): the inference path, written
from the upstream code.

- **Model.** Two ViT-B/14 encoders over the same ImageNet-normalised 896 × 896 input (64 × 64 patch tokens each): DINOv2
  ViT-B/14 with four registers (`x_norm_patchtokens`, the semantic encoder) and the Depth Anything V2 Base encoder (the
  last backbone hidden state without its class token, the geometry encoder). A double cross-attention fusion block mixes
  the two token sets; the projector's localisation branch (three linear layers with GELU) feeds a 3 × 3 convolution
  without padding, so the logit map is 62 × 62. A sigmoid gives the map. Upstream `C_infer_single-image.py`,
  `models/{feature_extractor,fuser,projector}.py`, `loaders/loader_utils.py` (github.com/alex-costanzino/RADAR-code @
  06dba26e, Apache-2.0); the authors' Space `radar-ifdl/RADAR-demo` runs the same code at 896 px. No upstream code is
  vendored; the contrastive branch of the projector is loaded (strict) but not evaluated.
- **Checkpoints.** Two plain `state_dict` pickles (`fusion_block_…pth` 16 tensors, `projector_…pth` 14 tensors), read with
  `weights_only=True`. DINOv2 ViT-B/14-reg: the architecture from the pinned torch.hub archive (`pretrained=False`) and the
  Meta pretrain checkpoint loaded strictly. Depth Anything V2 Base: the transformers snapshot; only `backbone` is kept (the
  DPT neck and head never contribute to the hidden state RADAR reads).
- **Preprocessing (upstream).** PIL RGB → `SquarePad` (edge padding, centred) → `Resize((896, 896))` bicubic on PIL →
  `ToTensor` → ImageNet mean and std. **Postprocessing (upstream).** The 62 × 62 map is resized bilinearly
  (`align_corners=False`) to the padded square and centre-cropped to the image, so the map stays registered to the pixels.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

INPUT = 896
EMBED_DIM, HEADS = 768, 8
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ENTRYPOINT = "dinov2_vitb14_reg"
FUSION_FILE = "fusion_block_multimodal_15k_10inp_120ep_4bs.pth"
PROJECTOR_FILE = "projector_multimodal_15k_10inp_120ep_4bs.pth"
DINOV2_FILE = "dinov2_vitb14_reg4_pretrain.pth"
MAX_MAP_SIDE = 2048  # the padded square the map is resized to before the crop (upstream: the image's long side)


class DoubleCrossAttentionFusion(nn.Module):
    """Per-modality LayerNorm → cross-attention both ways with residuals → linear mixer with GELU → LayerNorm."""

    def __init__(self, hidden_dim: int = EMBED_DIM, num_heads: int = HEADS, dropout: float = 0.1):
        super().__init__()
        self.norm_rgb = nn.LayerNorm(hidden_dim)
        self.norm_depth = nn.LayerNorm(hidden_dim)
        self.cross_attn_depth = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_rgb = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mixer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        rgb, depth = self.norm_rgb(rgb), self.norm_depth(depth)
        depth_attn = depth + self.cross_attn_depth(query=depth, key=rgb, value=rgb, need_weights=False)[0]
        rgb_attn = rgb + self.cross_attn_rgb(query=rgb, key=depth, value=depth, need_weights=False)[0]
        return self.out_norm(self.mixer(torch.cat([depth_attn, rgb_attn], dim=-1)))


class SiameseProjector(nn.Module):
    """The localisation branch and 3 × 3 probe; the contrastive branch (`*_con`) is kept for a strict load only."""

    def __init__(self, inner_features: int = EMBED_DIM):
        super().__init__()
        self.inner_features = inner_features
        self.act_fcn = nn.GELU()
        self.input = nn.Linear(inner_features, inner_features)
        self.projection = nn.Linear(inner_features, inner_features)
        self.output = nn.Linear(inner_features, inner_features)
        self.input_con = nn.Linear(inner_features, inner_features)
        self.projection_con = nn.Linear(inner_features, inner_features)
        self.output_con = nn.Linear(inner_features, inner_features)
        self.probe = nn.Conv2d(inner_features, 1, kernel_size=3)

    def forward_segmentation(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output(self.act_fcn(self.projection(self.act_fcn(self.input(x)))))
        side = int(round(x.shape[1] ** 0.5))
        return self.probe(x.permute(0, 2, 1).reshape(x.shape[0], self.inner_features, side, side))


class RadarNet(nn.Module):
    def __init__(self, semantic: nn.Module, geometry: nn.Module):
        super().__init__()
        self.semantic = semantic  # DINOv2 ViT-B/14-reg (torch.hub module)
        self.geometry = geometry  # Depth Anything V2 Base backbone (transformers Dinov2Backbone)
        self.fusion = DoubleCrossAttentionFusion()
        self.projector = SiameseProjector()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised B × 3 × 896 × 896 input → B × 1 × 62 × 62 logits."""
        rgb = self.semantic.forward_features(x)["x_norm_patchtokens"]
        depth = self.geometry(x, output_hidden_states=True).hidden_states[-1][:, 1:, :]
        return self.projector.forward_segmentation(self.fusion(rgb, depth))


# --- loading ---------------------------------------------------------------------------------------------------------

def build_semantic(hub_dir: Path) -> nn.Module:
    """DINOv2 ViT-B/14-reg architecture from a local, verified torch.hub directory (random init; no download, no bytecode)."""
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # the extracted tree is compared with the archive before every load: keep it exact
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="xFormers is not available")
            return torch.hub.load(str(hub_dir), ENTRYPOINT, source="local", pretrained=False)
    finally:
        sys.dont_write_bytecode = previous


def build_geometry(snapshot: Path) -> nn.Module:
    from transformers import DepthAnythingForDepthEstimation

    return DepthAnythingForDepthEstimation.from_pretrained(snapshot).backbone


def load_model(fusion: Path, projector: Path, dinov2: Path, semantic: nn.Module, geometry: nn.Module, device: Any,
               dtype: str = "float32") -> RadarNet:
    """Load every checkpoint strictly (every key present, none left over) into `RadarNet`."""
    net = RadarNet(semantic, geometry)
    net.semantic.load_state_dict(torch.load(dinov2, map_location="cpu", weights_only=True), strict=True)
    net.fusion.load_state_dict(torch.load(fusion, map_location="cpu", weights_only=True), strict=True)
    net.projector.load_state_dict(torch.load(projector, map_location="cpu", weights_only=True), strict=True)
    return net.to(device=device, dtype=getattr(torch, dtype)).eval()


# --- preprocessing and inference -------------------------------------------------------------------------------------

def square_pad(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Upstream `SquarePad`: centred edge padding to the long side; returns the image and (left, top, right, bottom)."""
    from torchvision.transforms import functional as TF

    side = max(image.size)
    left, top = [(side - s) // 2 for s in image.size]
    right, bottom = [side - (s + p) for s, p in zip(image.size, (left, top), strict=True)]
    padding = (left, top, right, bottom)
    return TF.pad(image, list(padding), padding_mode="edge"), padding


def to_batch(pixels: list[np.ndarray] | np.ndarray, size: int = INPUT) -> torch.Tensor:
    """RGB uint8 images → normalised B × 3 × size × size float32 (edge pad to a square, PIL bicubic resize)."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF

    images = [pixels] if isinstance(pixels, np.ndarray) else pixels
    out = []
    for image in images:
        padded, _ = square_pad(Image.fromarray(np.ascontiguousarray(image)))
        resized = TF.resize(padded, [size, size], interpolation=InterpolationMode.BICUBIC)
        out.append(TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD))
    return torch.stack(out)


def to_image_map(probability: torch.Tensor, width: int, height: int, max_side: int = MAX_MAP_SIDE) -> np.ndarray:
    """Upstream postprocessing: resize the square map to the padded square and centre-crop the image area.

    The square side is the image's long side (upstream), capped at `max_side`; the crop keeps the image's aspect ratio.
    """
    long_side = max(width, height)
    side = min(long_side, max_side)
    crop_w, crop_h = max(1, int(round(width * side / long_side))), max(1, int(round(height * side / long_side)))
    square = F.interpolate(probability[None, None].float(), size=[side, side], mode="bilinear")[0, 0]
    top, left = int(round((side - crop_h) / 2.0)), int(round((side - crop_w) / 2.0))  # torchvision center_crop offsets
    return square[top:top + crop_h, left:left + crop_w].cpu().numpy()


def predict(model: Any, images: list[np.ndarray], device: Any) -> list[np.ndarray]:
    """Maps in [0, 1] registered to each image (float32 on the CPU)."""
    dtype = next((p.dtype for p in model.parameters()), torch.float32) if isinstance(model, nn.Module) else torch.float32
    batch = to_batch(images)
    with torch.inference_mode():
        probability = torch.sigmoid(model(batch.to(device=device, dtype=dtype)).float())[:, 0].cpu()
    return [to_image_map(p, int(image.shape[1]), int(image.shape[0])) for p, image in zip(probability, images, strict=True)]
