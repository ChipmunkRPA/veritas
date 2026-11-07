"""`adapter_loader` functions (models.yaml `load.adapter_loader`) for the v0.3 voice models. The video frame detectors
reuse the image loaders. These functions read only verified local HF snapshots at the pinned revision."""
from __future__ import annotations

from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file, _revision


def load_face_appearance(model_id: str, device: str, dtype: str) -> LoadedModel:
    """Qualified native BGR112 contract, pinned Apache-2.0 OpenCV model; CPU only."""
    import cv2

    entry = _entry(model_id)
    recognizer = cv2.FaceRecognizerSF.create(str(_file(entry)), '',
        cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU)
    return LoadedModel(recognizer, _revision(entry))


def load_audio_classifier(model_id: str, device: str, dtype: str) -> LoadedModel:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    entry = _entry(model_id)
    snapshot = _file(entry, "config.json").parent
    feature_extractor = AutoFeatureExtractor.from_pretrained(snapshot)
    model = AutoModelForAudioClassification.from_pretrained(snapshot).to(device).eval()
    return LoadedModel((feature_extractor, model), _revision(entry))


def load_dinov2_small(model_id: str, device: str, dtype: str) -> LoadedModel:
    """DINOv2-small as the face-track appearance embedder (V-2)."""
    import torch
    from transformers import Dinov2Model

    from app.detectors.video.tracking import DinoEmbedder

    entry = _entry(model_id)
    model = Dinov2Model.from_pretrained(_file(entry, "config.json").parent, dtype=getattr(torch, dtype))
    return LoadedModel(DinoEmbedder(model.to(device).eval(), device), _revision(entry))


def load_xclip_vision(model_id: str, device: str, dtype: str) -> LoadedModel:
    """The X-CLIP B/16 vision tower as the D3 frame encoder (V-4), loaded strictly from the full X-CLIP checkpoint.

    transformers 5.15 `XCLIPVisionModel` has no `vision_model.` prefix, so `from_pretrained` on this checkpoint leaves every
    tower weight randomly initialised (found by the V-4 slow test). The loader strips the prefix from the checkpoint's
    `vision_model.*` tensors and loads them with `strict=True`; the text tower, prompt generator and MIT are not read."""
    import torch
    from safetensors import safe_open
    from transformers import XCLIPConfig, XCLIPVisionModel

    from app.detectors.video.d3 import XClipEncoder

    entry = _entry(model_id)
    config = XCLIPConfig.from_pretrained(_file(entry, "config.json").parent)
    prefix = "vision_model."
    with safe_open(str(_file(entry, "model.safetensors")), "pt") as f:
        state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys()  # noqa: SIM118 (a safetensors handle, not a dict)
                 if k.startswith(prefix) and not k.endswith(".position_ids")}
    model = XCLIPVisionModel(config.vision_config)
    model.load_state_dict(state, strict=True)
    return LoadedModel(XClipEncoder(model.to(device, getattr(torch, dtype)).eval(), device), _revision(entry))


def load_stable_syncnet(model_id: str, device: str, dtype: str) -> LoadedModel:
    """LatentSync StableSyncNet as the lip-sync lane's embedder (V-5): the verified checkpoint pickle read with
    `weights_only=True`, its `state_dict` loaded with `strict=True` into the vendored architecture."""
    import torch

    from app.detectors.video.syncnet import StableSyncNetEmbedder
    from app.third_party.latentsync.stable_syncnet import SYNCNET_16_PIXEL_ATTN, StableSyncNet

    entry = _entry(model_id)
    checkpoint = torch.load(_file(entry, "stable_syncnet.pt"), map_location="cpu", weights_only=True)
    model = StableSyncNet(SYNCNET_16_PIXEL_ATTN)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return LoadedModel(StableSyncNetEmbedder(model.to(device, getattr(torch, dtype)).eval(), device), _revision(entry))
