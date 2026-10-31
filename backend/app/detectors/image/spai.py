"""SPAI, spectral AI-generated image detection (Karageorgiou, Papadopoulos, Kompatsiaris, Gavves, CVPR 2025): the inference
path, written from the upstream code.

- **Model.** A BEiT-style ViT-B/16 (MFM pre-training) encodes each 224 px patch and its low- and high-frequency parts
  (circular spectral mask, radius 16). Per-block cosine similarities between the three views (mean and std over tokens)
  plus an original-image branch give a 1,096-d patch vector. Spectral context attention (a learnable query over the patch
  vectors) and an MLP head give one logit. Upstream `spai/models/sid.py` (`PatchBasedMFViT`, `MFViT`,
  `FrequencyRestorationEstimator`), `vision_transformer.py` and `filters.py`, configured by `configs/spai.yaml`
  (github.com/mever-team/spai, Apache-2.0, commit 8ff7b3b6). No upstream code is vendored; `apex` and `tsnecuda` (training
  and t-SNE plots), `yacs`, `einops`, `timm` and `clip` are not needed.
- **Checkpoint.** A torch pickle `{model, optimizer, lr_scheduler, max_accuracy, epoch, config, amp}` whose `config` is a
  yacs `CfgNode`, which `weights_only=True` refuses. A restricted unpickler admits only the globals the pinned file uses
  (tensor rebuild, `OrderedDict`, `set`, and the `CfgNode` read into a plain dict); anything else raises `UnpicklingError`.
  The MFM classification parts that inference never calls are dropped by name; everything else loads strictly.
- **Preprocessing (ML N1, N2).** Native pixels scaled to [0, 1] (upstream `positive_0_1`), never resampled or padded.
  Non-overlapping 224 px patches from the top-left corner (upstream `patchify_image`, stride 224). An image with fewer than
  four whole patches is scored on torchvision's `five_crop` (upstream `MINIMUM_PATCHES: 4`). A short side below 224 px is
  not assessed (upstream pads).
- **Pixel cap (ML N1).** 18 MP: a larger image is scored on a centred native window of at most 358 whole patches.
- **Explanation.** Each patch's share of the spectral context attention (mean over heads) goes to the exhibit.
"""
from __future__ import annotations

import collections
import math
import pickle
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PATCH = 224
VIT_PATCH = 16
EMBED_DIM, DEPTH, HEADS = 768, 12, 12
PROJ_DIM = 1024
ATTN_DIM, ATTN_HEADS = 1536, 12
HEAD_MLP_RATIO = 3
MASK_RADIUS = 16
MINIMUM_PATCHES = 4  # configs/spai.yaml and the README test command; the checkpoint's stored training config says 1
MAX_PIXELS = 18_000_000
MAX_PATCHES = MAX_PIXELS // (PATCH * PATCH)  # 358
FEATURE_BATCH = 16  # patches per encoder call (upstream 400 on a 48 GB GPU)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UNUSED_KEYS = ("mfvit.vit.cls_pos_embed", "mfvit.vit.actual_cls_token", "mfvit.vit.cls_blocks.", "mfvit.vit.fc_norm.",
               "mfvit.vit.intermediate_fc_norm.", "mfvit.vit.head.")  # MFM classification parts inference never calls
CHECKPOINT_FILE = "spai.pth"


# --- checkpoint ------------------------------------------------------------------------------------------------------

class _Config(dict):
    """Stand-in for the pickled yacs `CfgNode`: its items are read into a plain dict, its attribute state is discarded."""

    def __setstate__(self, state: Any) -> None:
        return None


_ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("builtins", "set"): set,
    ("__builtin__", "set"): set,
    ("yacs.config", "CfgNode"): _Config,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:  # torch resolves `*Storage` names before calling this
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed in a SPAI checkpoint") from None


def _refuse_legacy(*args: Any, **kwargs: Any) -> Any:
    raise pickle.UnpicklingError("legacy (non-zip) torch checkpoints are not read")


_RESTRICTED_PICKLE = types.SimpleNamespace(__name__="spai_restricted_pickle", Unpickler=_RestrictedUnpickler,
                                           load=_refuse_legacy)


def read_state(checkpoint: Path, *, keep_unused: bool = False) -> dict[str, torch.Tensor]:
    """The checkpoint's `model` state through the restricted unpickler; the optimizer, scheduler and config are discarded."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, pickle_module=_RESTRICTED_PICKLE)
    state = ck["model"]
    return dict(state) if keep_unused else {k: v for k, v in state.items() if not k.startswith(UNUSED_KEYS)}


# --- model -----------------------------------------------------------------------------------------------------------

class Attention(nn.Module):
    """BEiT attention: one `qkv` projection without bias plus learnable q and v biases (k has none)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        q, k, v = F.linear(x, self.qkv.weight, bias).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        return self.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm transformer block without layer scale (upstream `INIT_VALUES: None`)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class IntermediateViT(nn.Module):
    """The MFM ViT with `use_intermediate_layers` over every block and `return_features`: B × depth × tokens × dim."""

    def __init__(self, dim: int, depth: int, heads: int, img: int, patch: int):
        super().__init__()
        self.patch_embed = PatchEmbed(dim, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img // patch) ** 2 + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1) + self.pos_embed
        tokens = []
        for block in self.blocks:
            x = block(x)
            tokens.append(x[:, 1:])
        return torch.stack(tokens, dim=1)


class Projector(nn.Module):
    """Upstream `Projector` with two linear layers: keys `norm1` (when the input is normalised), `projector.{1,4}`, `norm2`."""

    def __init__(self, in_dim: int, out_dim: int, last_activation: type[nn.Module] = nn.GELU, input_norm: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim) if input_norm else nn.Identity()
        self.projector = nn.Sequential(nn.Dropout(), nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(),
                                       nn.Linear(out_dim, out_dim), last_activation(), nn.Dropout())
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.projector(self.norm1(x)))


class FeatureSpecificProjector(nn.Module):
    def __init__(self, n: int, in_dim: int, out_dim: int):
        super().__init__()
        self.projectors = nn.ModuleList([Projector(in_dim, out_dim, nn.Identity) for _ in range(n)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([projector(x[:, i]) for i, projector in enumerate(self.projectors)], dim=1)


class FeatureImportanceProjector(nn.Module):
    """Original-image branch: token mean and std per block, projected, softmax-weighted over blocks, projected again."""

    def __init__(self, n: int, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, n, dim))
        self.proj1 = Projector(2 * dim, dim, input_norm=False)
        self.proj2 = Projector(dim, dim, input_norm=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj1(torch.cat([x.mean(dim=2), x.std(dim=2)], dim=-1))
        return self.proj2(torch.sum(torch.softmax(self.alpha, dim=1) * x, dim=1))


class FrequencyRestorationEstimator(nn.Module):
    def __init__(self, n: int, in_dim: int, proj_dim: int):
        super().__init__()
        self.patch_projector = FeatureSpecificProjector(n, in_dim, proj_dim)
        self.original_features_processor = FeatureImportanceProjector(n, proj_dim)

    def forward(self, x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        """B × n × tokens × dim views → B × (6n + proj_dim)."""
        orig, low, high = self.patch_projector(x), self.patch_projector(low), self.patch_projector(high)
        similarities = (F.cosine_similarity(orig, low, dim=-1), F.cosine_similarity(orig, high, dim=-1),
                        F.cosine_similarity(low, high, dim=-1))
        stats = [t for s in similarities for t in (s.mean(dim=-1), s.std(dim=-1))]
        return torch.cat([*stats, self.original_features_processor(orig)], dim=1)


def circular_mask(size: int, radius: int) -> torch.Tensor:
    """1 where the centred spectral radius is below `radius` (upstream `generate_circular_mask`; int64, even size)."""
    half = torch.arange(0, size // 2, dtype=torch.float)
    axis = torch.cat([half.flip(dims=(0,)), half]).unsqueeze(0).expand(size, -1)
    return torch.where(torch.linalg.vector_norm(torch.stack([axis, axis.t()], dim=2), dim=-1) < radius, 1, 0)


def split_frequencies(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-pass and residual images from the centred 2-D spectrum (upstream `filter_image_frequencies`)."""
    spectrum = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    low = torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask, dim=(-2, -1))).real
    high = torch.fft.ifft2(torch.fft.ifftshift(spectrum * (1 - mask), dim=(-2, -1))).real
    return low, high


class MFViT(nn.Module):
    def __init__(self, vit: IntermediateViT, features_processor: FrequencyRestorationEstimator, size: int, radius: int):
        super().__init__()
        self.vit = vit
        self.features_processor = features_processor
        self.frequencies_mask = nn.Parameter(circular_mask(size, radius), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patches in [0, 1], B × 3 × size × size → B × (6n + proj_dim)."""
        low, high = split_frequencies(x.float(), self.frequencies_mask)
        low, high = low.clamp(0.0, 1.0).to(x.dtype), high.clamp(0.0, 1.0).to(x.dtype)
        mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        x, low, high = ((t - mean) / std for t in (x, low, high))
        return self.features_processor(self.vit(x), self.vit(low), self.vit(high))


class ClassificationHead(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int):
        super().__init__()
        hidden = dim * mlp_ratio
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SPAINet(nn.Module):
    """The module tree of upstream `PatchBasedMFViT` (state-dict keys `mfvit.*`, `to_kv`, `patch_aggregator`, `to_out.0`,
    `norm`, `cls_head.head.*`), split into `encode` (patches → vectors) and `classify` (one image's vectors → logit)."""

    def __init__(self, dim: int = EMBED_DIM, depth: int = DEPTH, heads: int = HEADS, proj_dim: int = PROJ_DIM,
                 attn_dim: int = ATTN_DIM, attn_heads: int = ATTN_HEADS, mlp_ratio: int = HEAD_MLP_RATIO, img: int = PATCH,
                 patch: int = VIT_PATCH, radius: int = MASK_RADIUS):
        super().__init__()
        cls_dim = 6 * depth + proj_dim
        self.mfvit = MFViT(IntermediateViT(dim, depth, heads, img, patch), FrequencyRestorationEstimator(depth, dim, proj_dim),
                           img, radius)
        self.heads = attn_heads
        self.to_kv = nn.Linear(cls_dim, 2 * attn_dim, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros(attn_heads, 1, attn_dim // attn_heads))
        self.to_out = nn.Sequential(nn.Linear(attn_dim, cls_dim, bias=False), nn.Dropout())
        self.norm = nn.LayerNorm(cls_dim)
        self.cls_head = ClassificationHead(cls_dim, mlp_ratio)

    def encode(self, patches: torch.Tensor) -> torch.Tensor:
        return self.mfvit(patches)

    def classify(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Spectral context attention over one image's patch vectors (L × cls_dim): the logit and each patch's attention
        share, mean over heads (upstream `patches_attention`, `norm`, `cls_head`)."""
        x = features.unsqueeze(0)
        b, n, _ = x.shape
        k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2) for t in self.to_kv(x).chunk(2, dim=-1))
        query = self.patch_aggregator.expand(b, -1, -1, -1)
        attention = torch.softmax(query @ k.transpose(-1, -2) * k.size(-1) ** -0.5, dim=-1)  # B × heads × 1 × L
        x = self.to_out((attention @ v).transpose(1, 2).reshape(b, 1, -1)).squeeze(1)
        return self.cls_head(self.norm(x)).reshape(()), attention.mean(dim=1).reshape(-1)


def load_model(checkpoint: Path, device: Any, dtype: str = "float32", **dims: int) -> SPAINet:
    state = read_state(checkpoint)
    model = SPAINet(**dims)
    model.load_state_dict(state, strict=True)
    del state
    return model.to(device=device, dtype=getattr(torch, dtype)).eval()  # the int64 spectral mask keeps its dtype


# --- patches ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchPlan:
    origins: list[tuple[int, int, str]]  # (x, y, name)
    mode: str  # grid | five_crop | undersize
    window: tuple[int, int, int, int]  # (x, y, w, h) of the scored area
    capped: bool = False


def patch_plan(width: int, height: int, *, patch: int = PATCH, minimum: int | None = None,
               max_patches: int | None = None) -> PatchPlan:
    minimum = MINIMUM_PATCHES if minimum is None else minimum
    max_patches = MAX_PATCHES if max_patches is None else max_patches
    if min(width, height) < patch:
        return PatchPlan([], "undersize", (0, 0, 0, 0))
    cols, rows = width // patch, height // patch
    if cols * rows < minimum:
        # torchvision `five_crop`: top left, top right, bottom left, bottom right, centre (centre offsets use `round`)
        cx, cy = int(round((width - patch) / 2.0)), int(round((height - patch) / 2.0))
        origins = [(0, 0, "top left"), (width - patch, 0, "top right"), (0, height - patch, "bottom left"),
                   (width - patch, height - patch, "bottom right"), (cx, cy, "centre")]
        return PatchPlan(origins, "five_crop", (0, 0, width, height))
    x0 = y0 = 0
    capped = cols * rows > max_patches
    if capped:
        scale = math.sqrt(max_patches / (cols * rows))
        cols, rows = max(1, math.floor(cols * scale)), max(1, math.floor(rows * scale))
        while cols * rows > max_patches:
            cols, rows = (cols - 1, rows) if cols >= rows else (cols, rows - 1)
        x0, y0 = (width - cols * patch) // 2, (height - rows * patch) // 2
    origins = [(x0 + c * patch, y0 + r * patch, f"row {r + 1}, column {c + 1}") for r in range(rows) for c in range(cols)]
    return PatchPlan(origins, "grid", (x0, y0, cols * patch, rows * patch), capped)


def to_batch(pixels: np.ndarray, origins: list[tuple[int, int, str]], patch: int = PATCH) -> torch.Tensor:
    """8-bit RGB patches → float32 NCHW in [0, 1]."""
    crops = np.stack([pixels[y:y + patch, x:x + patch, :3] for x, y, _ in origins])
    return torch.from_numpy(np.ascontiguousarray(crops)).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)


def score(model: Any, pixels: np.ndarray, origins: list[tuple[int, int, str]], device: Any, *,
          feature_batch: int | None = None, check: Callable[[str], None] | None = None) -> tuple[float, list[float]]:
    """Encode the patches in small batches (only one batch of pixels on the device at a time), then classify the image."""
    batch = feature_batch or FEATURE_BATCH
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    with torch.inference_mode():
        features = []
        for start in range(0, len(origins), batch):
            if check is not None:
                check("spai patches")
            features.append(model.encode(to_batch(pixels, origins[start:start + batch]).to(device=device, dtype=dtype)))
        logit, attention = model.classify(torch.cat(features, dim=0))
    return float(logit.to("cpu", torch.float32)), [float(a) for a in attention.to("cpu", torch.float32)]
