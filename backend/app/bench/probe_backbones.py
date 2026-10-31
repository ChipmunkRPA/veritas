"""Frozen backbones for the Veritas probe (ML §7.2; BUILD_SPEC I-13).

- `fake`: `probe.FakeBackbone`, model-free (tests and dry runs).
- `clip_l14`: OpenAI CLIP ViT-L/14 from the pinned `clip.vit_l14_openai` export, built with QuickGELU by the ModelManager
  loader (B-4a). The feature is the projected 768-d image embedding (UnivFD's probe input) of the view resized to 224 px
  on the short side (bicubic), centre-cropped, CLIP mean and std, as open_clip's validation transform does.
- `siglip2_so400m`: SigLIP2 so400m/14 at 384 px (`siglip2.so400m_p14_384`), vision tower only. The feature is the 1152-d
  MAP-head `pooler_output` of the view resized to 384 × 384 (bilinear), mean and std 0.5, as `SiglipImageProcessor` does.
- `pe_core_l14`: Perception Encoder Core L/14 at 336 px (`pe_core.large_p14_336`, timm). The feature is the projected
  1024-d image embedding of the view resized to 336 px on the short side (bicubic), centre-cropped, mean and std 0.5
  (timm `pretrained_cfg`, crop_pct 1.0).

A backbone key is `<model_id>@<sha12 over the pinned files>` (`score.models_sha12`), computed from the manifest without
loading weights; missing weights raise `BackboneNotInstalled`.
"""
from __future__ import annotations

import gc
from collections.abc import Sequence

import numpy as np
from PIL import Image

from app.bench.probe import FakeBackbone
from app.bench.score import models_sha12

BACKBONES = ("fake", "clip_l14", "siglip2_so400m", "pe_core_l14")
MODEL_IDS = {"clip_l14": "clip.vit_l14_openai", "siglip2_so400m": "siglip2.so400m_p14_384", "pe_core_l14": "pe_core.large_p14_336"}


class BackboneNotInstalled(RuntimeError):
    """The backbone's pinned weights are not installed."""


def _entry(name: str):
    from app.engine.registry import get_registry

    manifest = get_registry().manifest
    entry = manifest.model(MODEL_IDS[name])
    if entry is None:
        raise BackboneNotInstalled(f"{MODEL_IDS[name]} is not in the manifest")
    return manifest, entry


def backbone_key(name: str) -> str:
    if name == "fake":
        return FakeBackbone.key
    manifest, entry = _entry(name)
    return f"{entry.id}@{models_sha12(manifest, [entry.id], owner=f'probe backbone {name}')}"


def backbone_licence(name: str) -> dict:
    if name == "fake":
        return dict(FakeBackbone.licence)
    _, entry = _entry(name)
    return {"model": entry.id, **entry.licence.model_dump(by_alias=True, exclude_none=True)}


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class _TorchBackbone:
    """Loads the pinned model through its ModelManager loader and embeds a batch of PIL views."""

    name: str
    dim: int

    def __init__(self, *, device: str, fp16: bool):
        import torch

        from app.engine.fetchers import installed

        _, entry = _entry(self.name)
        if not installed(entry):
            raise BackboneNotInstalled(f"{entry.id} is not installed in the HF cache (make weights-fetch)")
        self.key = backbone_key(self.name)
        self.licence = backbone_licence(self.name)
        self.device = resolve_device(device)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.model = self._load(entry.id, "float16" if fp16 else "float32").to(self.device, dtype=self.dtype).eval()
        self.transform = self._transform()

    def _load(self, model_id: str, dtype: str):
        raise NotImplementedError

    def _transform(self):
        raise NotImplementedError

    def _forward(self, x):
        raise NotImplementedError

    def embed(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        x = torch.stack([self.transform(im.convert("RGB")) for im in images]).to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            return self._forward(x).float().cpu().numpy()

    def close(self) -> None:
        import torch

        self.model = None
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()


class ClipL14Backbone(_TorchBackbone):
    name, dim = "clip_l14", 768

    def _load(self, model_id: str, dtype: str):
        from app.detectors.image.adapters.loaders import load_clip_vit_l14

        return load_clip_vit_l14(model_id, self.device, dtype).obj

    def _transform(self):
        import torchvision.transforms as T

        from app.detectors.image.univfd import CLIP_MEAN, CLIP_STD, CROP

        return T.Compose([T.Resize(CROP, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(CROP), T.ToTensor(),
                          T.Normalize(CLIP_MEAN, CLIP_STD)])

    def _forward(self, x):
        return self.model.encode_image(x)


class Siglip2So400mBackbone(_TorchBackbone):
    name, dim = "siglip2_so400m", 1152
    size = 384

    def _load(self, model_id: str, dtype: str):
        from app.detectors.image.adapters.loaders import load_siglip_vision

        return load_siglip_vision(model_id, self.device, dtype).obj

    def _transform(self):
        import torchvision.transforms as T

        return T.Compose([T.Resize((self.size, self.size), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                          T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def _forward(self, x):
        return self.model(pixel_values=x).pooler_output


class PeCoreL14Backbone(_TorchBackbone):
    name, dim = "pe_core_l14", 1024
    size = 336

    def _load(self, model_id: str, dtype: str):
        from app.detectors.image.adapters.loaders import load_pe_core

        return load_pe_core(model_id, self.device, dtype).obj

    def _transform(self):
        import torchvision.transforms as T

        return T.Compose([T.Resize(self.size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(self.size), T.ToTensor(),
                          T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def _forward(self, x):
        return self.model(x)


_TORCH_BACKBONES = {cls.name: cls for cls in (ClipL14Backbone, Siglip2So400mBackbone, PeCoreL14Backbone)}


def load_backbone(name: str, *, device: str = "auto", fp16: bool = False, allow_unsafe_precision: bool = False):
    if name == "pe_core_l14" and fp16 and not allow_unsafe_precision:
        raise ValueError("PE-Core float16 failed final-score device parity (2026-09-18); use float32 and a new feature "
                         "directory. allow_unsafe_precision is for isolated numerical diagnostics only.")
    if name == "fake":
        return FakeBackbone()
    if name in _TORCH_BACKBONES:
        return _TORCH_BACKBONES[name](device=device, fp16=fp16)
    raise ValueError(f"unknown backbone {name!r} (known: {', '.join(BACKBONES)})")
