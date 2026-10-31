"""Mesorch, image manipulation localisation (Zhu, Ma, Su et al., AAAI 2025): the inference path, written from the upstream
code.

- **Model.** Two encoders over the ImageNet-normalised RGB image plus a DCT band: a ConvNeXt-T (timm `ConvNeXt`, 6-channel
  stem) on RGB + high frequencies, and a SegFormer MiT-B3 (6-channel first patch embedding) on RGB + low frequencies. Their
  eight multi-scale feature maps are upsampled to 1/4 resolution, reduced to one channel each, and mixed per pixel by a
  softmax gate computed from RGB + both bands. The logit map is resized to 512 px (bilinear, `align_corners=True`) and a
  sigmoid gives the map. Upstream `mesorch.py` (`MesorchFull`) and `extractor/{high,low}_frequency_feature_extraction.py`
  (github.com/scu-zjz/Mesorch, MIT, commit 33454cb6). No upstream code is vendored; `IMDLBenCo` (registry, datasets,
  training), `albumentations` and the loss are not needed.
- **DCT bands.** An orthonormal 2-D DCT over the whole input per channel. High band: coefficients in the top-left
  ⌊0.05 H⌋ × ⌊0.05 W⌋ block zeroed; low band: the bottom-right ⌊0.95 H⌋ × ⌊0.95 W⌋ block zeroed. Each band is min-max
  scaled per image over all channels (a constant band would divide by zero upstream; here it becomes zero).
- **Checkpoint.** A torch pickle `{model, optimizer, epoch 98, scaler, args}` whose `args` is an `argparse.Namespace`, which
  `weights_only=True` refuses. A restricted unpickler admits only the globals the pinned file uses (tensor rebuild,
  `OrderedDict`, and the `Namespace` read into a plain dict); anything else raises `UnpicklingError`. All 804 model tensors
  load strictly, including the ConvNeXt classification head that inference never calls.
- **Preprocessing (upstream test path).** PIL RGB, `albu.Resize(512, 512)` (OpenCV bilinear, aspect ratio not kept),
  `albu.Normalize` with the ImageNet mean and std, float32. `to_batch` equals the albumentations 1.4.24 transform within
  4.8e-07 on two fixtures (I-10 check).
"""
from __future__ import annotations

import collections
import math
import pickle
import types
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INPUT = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HIGH_ALPHA, LOW_ALPHA = 0.05, 0.95
CHECKPOINT_FILE = "mesorch-98.pth"
CONVNEXT_DEPTHS, CONVNEXT_DIMS = (3, 3, 9, 3), (96, 192, 384, 768)
MIT_DIMS, MIT_HEADS, MIT_DEPTHS, MIT_SR = (64, 128, 320, 512), (1, 2, 5, 8), (3, 4, 18, 3), (8, 4, 2, 1)


# --- checkpoint ------------------------------------------------------------------------------------------------------

class _Args(dict):
    """Stand-in for the pickled `argparse.Namespace`: its attributes are read into a plain dict."""

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.update(state)


_ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("argparse", "Namespace"): _Args,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # torch resolves `*Storage` names before calling this
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a Mesorch checkpoint") from None


def _refuse_legacy(*args: Any, **kwargs: Any) -> Any:
    raise pickle.UnpicklingError("legacy (non-zip) torch checkpoints are not read")


_RESTRICTED_PICKLE = types.SimpleNamespace(__name__="mesorch_restricted_pickle", Unpickler=_RestrictedUnpickler,
                                           load=_refuse_legacy)


def read_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    """The checkpoint's `model` state through the restricted unpickler; the optimizer, scaler and arguments are discarded."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, pickle_module=_RESTRICTED_PICKLE)
    return dict(ck["model"])


# --- DCT bands -------------------------------------------------------------------------------------------------------

def dct_matrix(n: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix (rows = frequencies), built like upstream `create_dct_matrix` in float32."""
    k = torch.arange(n, dtype=torch.float32).reshape(n, 1)
    i = torch.arange(n, dtype=torch.float32).reshape(1, n)
    matrix = torch.sqrt(torch.tensor(2.0 / n)) * torch.cos(math.pi * k * (2 * i + 1) / (2 * n))
    matrix[0, :] = 1 / math.sqrt(n)
    return matrix.to(device=device, dtype=dtype)


def dct_band(x: torch.Tensor, alpha: float, band: str) -> torch.Tensor:
    """Zero a corner block of the 2-D DCT (`high`: top-left, `low`: bottom-right), invert, min-max scale per image."""
    h, w = x.shape[-2:]
    dh, dw = dct_matrix(h, x.device, x.dtype), dct_matrix(w, x.device, x.dtype)
    coefficients = dh @ x @ dw.t()
    mask = torch.ones(h, w, device=x.device, dtype=x.dtype)
    ah, aw = int(alpha * h), int(alpha * w)
    if band == "high":
        mask[:ah, :aw] = 0
    elif band == "low":
        mask[-ah:, -aw:] = 0
    else:
        raise ValueError(f"unknown DCT band {band!r}")
    restored = dh.t() @ (coefficients * mask) @ dw
    flat = restored.reshape(restored.shape[0], -1)
    low, high = flat.min(dim=1).values.view(-1, 1, 1, 1), flat.max(dim=1).values.view(-1, 1, 1, 1)
    span = high - low
    return torch.where(span > 0, (restored - low) / torch.where(span > 0, span, torch.ones_like(span)),
                       torch.zeros_like(restored))


# --- ConvNeXt branch -------------------------------------------------------------------------------------------------

def _convnext() -> nn.Module:
    from timm.models.convnext import ConvNeXt

    net = ConvNeXt(depths=CONVNEXT_DEPTHS, dims=CONVNEXT_DIMS)
    first = net.stem[0]
    net.stem[0] = nn.Conv2d(6, first.out_channels, kernel_size=first.kernel_size, stride=first.stride, padding=first.padding,
                            bias=False)
    return net


def _convnext_stages(net: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    x = net.stem(x)
    outs = []
    for stage in net.stages:
        x = stage(x)
        outs.append(x)
    return outs


# --- SegFormer MiT-B3 branch -----------------------------------------------------------------------------------------

class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, _, c = x.shape
        return self.dwconv(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)


class MixMlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        return self.fc2(self.act(self.dwconv(self.fc1(x), h, w)))


class EfficientAttention(nn.Module):
    """SegFormer attention: keys and values from a sequence reduced by a strided convolution (`sr_ratio`)."""

    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        self.heads, self.sr_ratio = heads, sr_ratio
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, n, c = x.shape
        q = self.q(x).reshape(b, n, self.heads, c // self.heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x = self.norm(self.sr(x.permute(0, 2, 1).reshape(b, c, h, w)).reshape(b, c, -1).permute(0, 2, 1))
        k, v = self.kv(x).reshape(b, -1, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).float().softmax(dim=-1)
        return self.proj((attn.to(v.dtype) @ v).transpose(1, 2).reshape(b, n, c))


class MixBlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm(dim)
        self.attn = EfficientAttention(dim, heads, sr_ratio)
        self.norm2 = norm(dim)
        self.mlp = MixMlp(dim, 4 * dim)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w)
        return x + self.mlp(self.norm2(x), h, w)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch: int, stride: int, in_chans: int, dim: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=stride, padding=patch // 2, bias=bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, h, w = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), h, w


class MixVisionTransformer(nn.Module):
    """MiT-B3 with a 6-channel, bias-free first patch embedding; returns the four stage outputs (B × C × H × W)."""

    def __init__(self, dims: tuple[int, ...] = MIT_DIMS, heads: tuple[int, ...] = MIT_HEADS, depths: tuple[int, ...] = MIT_DEPTHS,
                 sr_ratios: tuple[int, ...] = MIT_SR, in_chans: int = 6):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-6)
        for i in range(4):
            embed = (OverlapPatchEmbed(7, 4, in_chans, dims[0], bias=False) if i == 0
                     else OverlapPatchEmbed(3, 2, dims[i - 1], dims[i]))
            setattr(self, f"patch_embed{i + 1}", embed)
            setattr(self, f"block{i + 1}", nn.ModuleList([MixBlock(dims[i], heads[i], sr_ratios[i]) for _ in range(depths[i])]))
            setattr(self, f"norm{i + 1}", norm(dims[i]))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        b, outs = x.shape[0], []
        for i in range(1, 5):
            x, h, w = getattr(self, f"patch_embed{i}")(x)
            for block in getattr(self, f"block{i}"):
                x = block(x, h, w)
            x = getattr(self, f"norm{i}")(x).reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


# --- decoder, gate, network ------------------------------------------------------------------------------------------

def _up(*channels: int) -> nn.Module:
    layers = [nn.ConvTranspose2d(a, b, kernel_size=4, stride=2, padding=1) for a, b in zip(channels, channels[1:], strict=False)]
    return layers[0] if len(layers) == 1 else nn.Sequential(*layers)


class UpsampleConcat(nn.Module):
    """Every stage output of both encoders upsampled to 1/4 resolution (ConvNeXt → 96 channels, MiT → 64)."""

    def __init__(self):
        super().__init__()
        self.upsamplec2, self.upsamplec3, self.upsamplec4 = _up(192, 96), _up(384, 192, 96), _up(768, 384, 192, 96)
        self.upsamples2, self.upsamples3, self.upsamples4 = _up(128, 64), _up(320, 128, 64), _up(512, 320, 128, 64)

    def forward(self, c: list[torch.Tensor], s: list[torch.Tensor]) -> list[torch.Tensor]:
        return [c[0], self.upsamplec2(c[1]), self.upsamplec3(c[2]), self.upsamplec4(c[3]),
                s[0], self.upsamples2(s[1]), self.upsamples3(s[2]), self.upsamples4(s[3])]


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ScoreNetwork(nn.Module):
    """The gate: RGB + both bands (9 channels) → per-pixel softmax weights over the eight reduced feature maps."""

    def __init__(self, experts: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(9, 192, kernel_size=7, stride=2, padding=3)
        self.invert = nn.Sequential(LayerNorm2d(192), nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(192, 768, kernel_size=1), nn.Conv2d(768, 192, kernel_size=1), nn.GELU())
        self.conv2 = nn.Conv2d(192, experts, kernel_size=7, stride=2, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(x + self.invert(x)).float().softmax(dim=1)


class MesorchNet(nn.Module):
    def __init__(self, image_size: int = INPUT):
        super().__init__()
        self.image_size = image_size
        self.convnext = _convnext()
        self.segformer = MixVisionTransformer()
        self.upsample = UpsampleConcat()
        self.inverse = nn.ModuleList([nn.Conv2d(96, 1, 1) for _ in range(4)] + [nn.Conv2d(64, 1, 1) for _ in range(4)])
        self.gate = ScoreNetwork()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised B × 3 × 512 × 512 input → B × 512 × 512 manipulation map in [0, 1] (float32)."""
        high, low = dct_band(x, HIGH_ALPHA, "high"), dct_band(x, LOW_ALPHA, "low")
        features = self.upsample(_convnext_stages(self.convnext, torch.cat([x, high], dim=1)),
                                 self.segformer(torch.cat([x, low], dim=1)))
        weights = self.gate(torch.cat([x, high, low], dim=1))
        reduced = torch.cat([head(f) for head, f in zip(self.inverse, features, strict=True)], dim=1)
        logits = torch.sum(weights * reduced.float(), dim=1, keepdim=True)
        logits = F.interpolate(logits, size=(self.image_size, self.image_size), mode="bilinear", align_corners=True)
        return torch.sigmoid(logits.float())[:, 0]


def load_model(checkpoint: Path, device: str, dtype: str = "float32") -> MesorchNet:
    net = MesorchNet()
    net.load_state_dict(read_state(checkpoint), strict=True)
    return net.to(device=device, dtype=getattr(torch, dtype)).eval()


# --- preprocessing and inference -------------------------------------------------------------------------------------

def to_batch(pixels: list[np.ndarray] | np.ndarray, size: int = INPUT) -> torch.Tensor:
    """RGB uint8 images → normalised B × 3 × size × size float32 (OpenCV bilinear resize, like `albu.Resize`)."""
    images = [pixels] if isinstance(pixels, np.ndarray) else pixels
    mean, std = np.asarray(IMAGENET_MEAN, np.float32), np.asarray(IMAGENET_STD, np.float32)
    out = []
    for image in images:
        resized = image if image.shape[:2] == (size, size) else cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        out.append(((resized.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(np.stack(out)))


def predict(model: Any, batch: torch.Tensor, device: Any) -> np.ndarray:
    """B × 512 × 512 float32 maps on the CPU."""
    dtype = next((p.dtype for p in model.parameters()), torch.float32) if isinstance(model, nn.Module) else torch.float32
    with torch.inference_mode():
        return model(batch.to(device=device, dtype=dtype)).float().cpu().numpy()
