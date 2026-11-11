"""`adapter_loader` functions (models.yaml `load.adapter_loader`) for the v0.3 image models.

The ModelManager calls `loader(model_id, device, dtype)` only after `ensure_verified` has checked every pinned file.
These functions read the verified local files: HF snapshots at the pinned revision, and `data/weights/<file>` for URL
sources. Nothing here downloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.engine.fetchers import file_path
from app.engine.manifest import ModelEntry
from app.engine.models import ModelUnavailable


@dataclass(frozen=True)
class LoadedModel:
    obj: Any
    revision: str  # HF commit, or the pinned sha256 of the first file for URL sources


def _entry(model_id: str) -> ModelEntry:
    from app.engine.registry import get_registry

    entry = get_registry().manifest.model(model_id)
    if entry is None:
        raise ModelUnavailable(f"model unavailable: {model_id} is not in the manifest")
    return entry


def _file(entry: ModelEntry, name: str | None = None) -> Path:
    for weight in entry.source.files:
        if name is None or weight.path == name:
            return file_path(entry, weight)
    raise ModelUnavailable(f"model unavailable: {entry.id} lists no file {name}")


def _revision(entry: ModelEntry) -> str:
    return entry.source.revision or entry.source.files[0].sha256 or ""


def load_yunet(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image.faces import load_yunet as create

    entry = _entry(model_id)
    return LoadedModel(create(_file(entry)), _revision(entry))


def load_clip_vit_l14(model_id: str, device: str, dtype: str) -> LoadedModel:
    import open_clip

    entry = _entry(model_id)
    # The OpenAI weights were trained with QuickGELU; open_clip 3.3 builds "ViT-L-14" with standard GELU even for the `openai`
    # tag (max |Δ feature| 3.19 on a random input). B-4a builds "ViT-L-14-quickgelu" (img.univfd and vid.frames.univfd 0.4.0).
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained=str(_file(entry)))
    return LoadedModel(model.to(device).eval(), _revision(entry))


def load_siglip_vision(model_id: str, device: str, dtype: str) -> LoadedModel:
    import torch
    from transformers import SiglipVisionModel

    entry = _entry(model_id)
    # The checkpoint holds both towers; SiglipVisionModel keeps the `vision_model.*` tensors and drops the text tower.
    model = SiglipVisionModel.from_pretrained(_file(entry, "config.json").parent, dtype=getattr(torch, dtype))
    return LoadedModel(model.to(device).eval(), _revision(entry))


def load_pe_core(model_id: str, device: str, dtype: str) -> LoadedModel:
    import timm
    from safetensors.torch import load_file

    entry = _entry(model_id)
    model = timm.create_model("vit_pe_core_large_patch14_336", pretrained=False)
    model.load_state_dict(load_file(str(_file(entry, "model.safetensors"))), strict=True)
    return LoadedModel(model.to(device).eval(), _revision(entry))


def load_gend_pe(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image.gend_pe import load_model

    entry = _entry(model_id)
    return LoadedModel(load_model(_file(entry, "model.safetensors"), device), _revision(entry))


def load_univfd_probe(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image.univfd import load_probe

    entry = _entry(model_id)
    return LoadedModel(load_probe(_file(entry), device), _revision(entry))


def load_commfor_vit(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image.commfor import load_model

    entry = _entry(model_id)
    return LoadedModel(load_model(_file(entry, "model.safetensors"), device), _revision(entry))


def load_dda(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image import dda
    from app.engine.fetchers import torch_hub_dir

    entry = _entry(model_id)
    backbone = dda.build_backbone(torch_hub_dir(_entry(dda.BACKBONE_MODEL)))  # verifies the archive; ModelNotInstalled if absent
    return LoadedModel(dda.load_model(_file(entry, "DDA_ckpt.pth"), backbone, device, dtype), _revision(entry))


def load_spai(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image import spai

    entry = _entry(model_id)
    return LoadedModel(spai.load_model(_file(entry, spai.CHECKPOINT_FILE), device, dtype), _revision(entry))


def load_gend(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image import gend

    entry = _entry(model_id)
    return LoadedModel(gend.load_model(_file(entry, "model.safetensors"), device), _revision(entry))


def load_mesorch(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image import mesorch

    entry = _entry(model_id)
    return LoadedModel(mesorch.load_model(_file(entry, mesorch.CHECKPOINT_FILE), device, dtype), _revision(entry))


def load_radar(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image import radar
    from app.engine.fetchers import ensure_verified, torch_hub_dir

    entry = _entry(model_id)
    deps = {dep: _entry(dep) for dep in entry.depends_on}  # projector, DINOv2 weights and code, Depth Anything V2 Base
    by_type = {e.source.type: e for e in deps.values()}
    for dep in deps.values():
        if dep.source.type != "torch_hub":
            ensure_verified(dep)  # ModelNotInstalled when a dependency is absent
    projector = next(e for e in deps.values() if any(f.path == radar.PROJECTOR_FILE for f in e.source.files))
    dinov2 = next(e for e in deps.values() if any(f.path == radar.DINOV2_FILE for f in e.source.files))
    semantic = radar.build_semantic(torch_hub_dir(by_type["torch_hub"]))  # verifies the archive
    geometry = radar.build_geometry(_file(by_type["hf"], "config.json").parent)
    model = radar.load_model(_file(entry, radar.FUSION_FILE), _file(projector, radar.PROJECTOR_FILE), _file(dinov2, radar.DINOV2_FILE),
                             semantic, geometry, device, dtype)
    return LoadedModel(model, _revision(entry))


def load_hf_image_classifier(model_id: str, device: str, dtype: str) -> LoadedModel:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    entry = _entry(model_id)
    snapshot = _file(entry, "config.json").parent
    processor = AutoImageProcessor.from_pretrained(snapshot)
    model = AutoModelForImageClassification.from_pretrained(snapshot).to(device).eval()
    return LoadedModel((processor, model), _revision(entry))


def load_dfdc_b7(model_id: str, device: str, dtype: str) -> LoadedModel:
    from app.detectors.image.dfdc import load_member

    entry = _entry(model_id)
    return LoadedModel(load_member(_file(entry), device), _revision(entry))
