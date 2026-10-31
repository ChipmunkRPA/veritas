# Copyright (c) 2025 Siyuan Cheng; local-only adaptation, Veritas, 2026.
# SPDX-License-Identifier: MIT
"""CO-SPY sd-v1_4 inference arithmetic; see NOTICE.md for pinned sources/licences.

This is an isolated research implementation, not a registered detector. Callers
verify the acquired weight bytes against their frozen provenance before loading.
The loader independently requires the pinned configs and exact finite F32 states;
it never downloads, fits, normalizes features, or substitutes missing parameters.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

VAE_CONFIG_SHA256 = "0e5966d4d52c8077648754ffd2c14ab5e0b26a4bdfa6af1d207c27550fa6d2af"
SIGLIP_CONFIG_SHA256 = "befc34b883ae561a59afc3207f2342379d82e1cec657a791465bb1b4550ca936"


class CospyLoadError(ValueError):
    """The local candidate cannot be loaded without changing its contract."""


class CospyDependencyError(RuntimeError):
    """A caller-provisioned optional research dependency is unavailable."""


def _local_file(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise CospyLoadError(f"Required local CO-SPY file is missing: {result.name}")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CospyLoadError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_config(path: Path, expected_sha: str) -> dict:
    if path.stat().st_size > 65_536:
        raise CospyLoadError("CO-SPY configuration exceeds the size bound")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise CospyLoadError("CO-SPY configuration does not match the pinned bytes")
    config = json.loads(data, object_pairs_hook=_unique_object)
    if not isinstance(config, dict):
        raise CospyLoadError("CO-SPY configuration must be an object")
    return config


def _device(device: str | torch.device) -> torch.device:
    result = torch.device(device)
    if result.type not in {"cpu", "mps"} or result.index not in {None, 0}:
        raise CospyLoadError("The isolated CO-SPY runtime supports CPU or MPS float32 only")
    if torch.get_default_dtype() != torch.float32:
        raise CospyLoadError("CO-SPY construction requires the float32 default dtype")
    return result


def _validate_state(state: object, *, max_tensors: int, max_elements: int) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or not state or len(state) > max_tensors:
        raise CospyLoadError("Invalid or oversized CO-SPY state dictionary")
    total = 0
    for name, value in state.items():
        # Released CPU-resident head saves can retain the built-in Parameter
        # wrapper. Permit only these two exact torch types, never subclasses.
        if not isinstance(name, str) or type(value) not in (Tensor, nn.Parameter):
            raise CospyLoadError("CO-SPY states require named exact Tensor or Parameter values")
        if value.device.type != "cpu" or value.layout != torch.strided or value.ndim > 4:
            raise CospyLoadError(f"Unsupported tensor storage: {name}")
        if value.dtype != torch.float32 and not (
            value.dtype == torch.int64 and value.ndim == 0 and name.endswith(".num_batches_tracked")
        ):
            raise CospyLoadError(f"Unsupported tensor dtype: {name}")
        total += value.numel()
        if total > max_elements:
            raise CospyLoadError("CO-SPY state exceeds its element bound")
    # Check bounds for every tensor before doing any potentially large reduction.
    for name, value in state.items():
        if not bool(torch.isfinite(value).all()):
            raise CospyLoadError(f"Non-finite checkpoint tensor: {name}")
    return dict(state)


def _pth_state(path: Path, *, max_tensors: int, max_elements: int) -> dict[str, Tensor]:
    if path.stat().st_size > 400_000_000:
        raise CospyLoadError("CO-SPY checkpoint exceeds the size bound")
    if torch.serialization.get_unsafe_globals_in_checkpoint(path):
        raise CospyLoadError("Checkpoint requires unsupported pickle globals")
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return _validate_state(state, max_tensors=max_tensors, max_elements=max_elements)


def _strict_load(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    expected = module.state_dict()
    if set(state) != set(expected):
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        raise CospyLoadError(f"Checkpoint keys differ (missing={missing[:5]}, extra={extra[:5]})")
    for name, value in state.items():
        target = expected[name]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise CospyLoadError(f"Checkpoint shape/dtype differs: {name}")
        if value.device.type != "cpu" or value.layout != torch.strided:
            raise CospyLoadError(f"Unsupported checkpoint storage: {name}")
        if not bool(torch.isfinite(value).all()):
            raise CospyLoadError(f"Non-finite checkpoint tensor: {name}")
    # Nothing is copied until the complete state has passed validation.
    module.load_state_dict(state, strict=True)


def _make_vae(config: dict) -> nn.Module:
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise CospyDependencyError("CO-SPY needs the caller-provisioned diffusers research overlay") from exc
    # Config-only: embedded artifact_encoder.vae.* supplies every VAE weight.
    return AutoencoderKL.from_config(config)


def _make_siglip(config: dict) -> nn.Module:
    try:
        from open_clip.model import CustomTextCLIP
    except ImportError as exc:
        raise CospyDependencyError("CO-SPY needs the caller-provisioned OpenCLIP/timm runtime") from exc
    cfg = dict(config["model_cfg"])
    if cfg.pop("custom_text", None) is not True:
        raise CospyLoadError("The pinned SigLIP model requires CustomTextCLIP")
    vision = cfg["vision_cfg"]
    if vision.get("timm_model_pretrained") is not False or cfg["text_cfg"].get("hf_model_name"):
        raise CospyLoadError("Pretrained or remote constructors are forbidden")
    # The validated pinned config constructs both towers without a tokenizer or hub call.
    return CustomTextCLIP(**cfg, cast_dtype=None)


def _safe_state(path: Path, module: nn.Module) -> dict[str, Tensor]:
    from safetensors import safe_open

    expected = module.state_dict()
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise CospyLoadError("Truncated safetensors header")
        length = int.from_bytes(length_bytes, "little")
        if length < 2 or length > 8_000_000:
            raise CospyLoadError("Safetensors header exceeds its bound")
        header_bytes = stream.read(length)
        if len(header_bytes) != length:
            raise CospyLoadError("Truncated safetensors metadata")
    header = json.loads(header_bytes, object_pairs_hook=_unique_object)
    if not isinstance(header, dict):
        raise CospyLoadError("Invalid safetensors metadata")
    header.pop("__metadata__", None)
    if set(header) != set(expected):
        raise CospyLoadError("Full SigLIP safetensors keys differ from the local architecture")
    for name, tensor in expected.items():
        metadata = header[name]
        if not isinstance(metadata, dict) or metadata.get("shape") != list(tensor.shape):
            raise CospyLoadError(f"SigLIP safetensors shape differs: {name}")
        if tensor.dtype != torch.float32 or metadata.get("dtype") != "F32":
            raise CospyLoadError(f"SigLIP safetensors dtype differs: {name}")
    # Exact full-tower metadata is checked before materializing any tensor values.
    with safe_open(path, framework="pt", device="cpu") as source:
        if set(source.keys()) != set(expected):
            raise CospyLoadError("SigLIP safetensors keys changed while loading")
        state = {name: source.get_tensor(name) for name in expected}
    return _validate_state(state, max_tensors=2_000, max_elements=1_000_000_000)


def preprocess(image: Image.Image) -> Tensor:
    """Released PIL resize/crop/ToTensor path; deliberately no EXIF transpose."""
    if not isinstance(image, Image.Image):
        raise TypeError("CO-SPY preprocessing requires a PIL image")
    transform = transforms.Compose([
        transforms.Resize(384, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def _input(x: Tensor, side: int) -> None:
    if not isinstance(x, Tensor) or x.shape != (1, 3, side, side) or x.dtype != torch.float32:
        raise ValueError(f"CO-SPY requires a float32 [1, 3, {side}, {side}] tensor")
    if x.device.type not in {"cpu", "mps"}:
        raise ValueError("CO-SPY inputs must use CPU or MPS")
    if not bool(torch.isfinite(x).all()) or bool((x < 0).any()) or bool((x > 1).any()):
        raise ValueError("CO-SPY inputs must be finite values in [0, 1]")
    if torch.is_autocast_enabled(x.device.type):
        raise ValueError("CO-SPY inference requires autocast disabled")


def _finite(value: Tensor, shape: tuple[int, ...], name: str) -> None:
    if value.shape != shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Invalid CO-SPY {name}: expected finite float32 {shape}")


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class VAEReconEncoder(nn.Module):
    def __init__(self, vae: nn.Module):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        # Register after CNN initialization, exactly as upstream; never reinitialize the VAE.
        self.vae = vae

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, *, return_details: bool = False):
        with torch.no_grad():
            posterior_mean = self.vae.encode(x).latent_dist.mean
            reconstruction = self.vae.decode(posterior_mean).sample
        _finite(reconstruction, tuple(x.shape), "reconstruction")
        if posterior_mean.dtype != torch.float32 or not bool(torch.isfinite(posterior_mean).all()):
            raise ValueError("Invalid CO-SPY posterior mean")
        residual = (x - reconstruction) / 7.0 * 100.0
        out = self.maxpool(self.relu(self.bn1(self.conv1(residual))))
        out = self.layer2(self.layer1(out))
        features = self.avgpool(out).view(x.size(0), -1)
        _finite(features, (1, 512), "artifact features")
        if return_details:
            return {
                "posterior_mean": posterior_mean,
                "reconstruction": reconstruction,
                "residual": residual,
                "artifact_features": features,
            }
        return features


class ArtifactDetector(nn.Module):
    """Released artifact branch; accepts the already resized 224-square tensor."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.artifact_encoder = VAEReconEncoder(vae)
        self.fc = nn.Linear(512, 1)

    @torch.inference_mode()
    def forward(self, x: Tensor, *, return_details: bool = False):
        _input(x, 224)
        if self.training:
            raise ValueError("CO-SPY inference requires eval mode")
        encoded = self.artifact_encoder(x, return_details=return_details)
        features = encoded["artifact_features"] if return_details else encoded
        logit = self.fc(features)
        _finite(logit, (1, 1), "artifact logit")
        return {**encoded, "art_logit": logit} if return_details else logit


class LinearHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc = nn.Linear(width, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class CospyModel(nn.Module):
    def __init__(self, artifact: nn.Module, siglip: nn.Module, semantic: nn.Module, fusion: nn.Module):
        super().__init__()
        self.artifact = artifact
        self.siglip = siglip
        self.semantic = semantic
        self.fusion = fusion

    @torch.inference_mode()
    def forward(self, x: Tensor, *, return_details: bool = False) -> dict[str, Tensor]:
        _input(x, 384)
        if self.training:
            raise ValueError("CO-SPY inference requires eval mode")
        # The released fusion preprocessing supersedes the base's bicubic/squash metadata.
        x_sem = TF.normalize(x, [0.5] * 3, [0.5] * 3)
        x_art = TF.resize(x, 224, interpolation=InterpolationMode.BILINEAR, antialias=False)
        semantic_features = self.siglip.encode_image(x_sem, normalize=False)
        _finite(semantic_features, (1, 1152), "semantic features")
        sem_logit = self.semantic(semantic_features)
        artifact = self.artifact(x_art, return_details=return_details)
        art_logit = artifact["art_logit"] if return_details else artifact
        _finite(sem_logit, (1, 1), "semantic logit")
        _finite(art_logit, (1, 1), "artifact logit")
        fusion_logit = self.fusion(torch.cat([sem_logit, art_logit], dim=1))
        _finite(fusion_logit, (1, 1), "fusion logit")
        result = {
            "sem_logit": sem_logit,
            "art_logit": art_logit,
            "fusion_logit": fusion_logit,
            "raw_sigmoid": fusion_logit.sigmoid(),
        }
        if return_details:
            result.update(artifact)
            result["semantic_features"] = semantic_features
        return result


def load_artifact(
    artifact_path: str | Path, vae_config_path: str | Path, device: str | torch.device = "cpu",
) -> ArtifactDetector:
    """Strict local artifact + embedded-VAE gate, independent of SigLIP acquisition."""
    target = _device(device)
    weights = _local_file(artifact_path)
    config = _read_config(_local_file(vae_config_path), VAE_CONFIG_SHA256)
    state = _pth_state(weights, max_tensors=600, max_elements=90_000_000)
    with torch.device("cpu"):
        model = ArtifactDetector(_make_vae(config))
    _strict_load(model, state)
    return model.requires_grad_(False).eval().to(target)


def load_local(
    *, artifact_path: str | Path, semantic_path: str | Path, fusion_path: str | Path,
    vae_config_path: str | Path, siglip_config_path: str | Path, siglip_weights_path: str | Path,
    device: str | torch.device = "cpu", vision_only: bool = False,
) -> CospyModel:
    """Load all released branches locally; the full SigLIP state must match exactly."""
    if vision_only:
        raise CospyLoadError("Vision-only optimization is not part of this verified runtime")
    target = _device(device)
    paths = [_local_file(path) for path in (
        artifact_path, semantic_path, fusion_path, vae_config_path, siglip_config_path, siglip_weights_path,
    )]
    artifact_file, semantic_file, fusion_file, vae_config_file, siglip_config_file, siglip_file = paths
    # Refuse configuration differences before any large model construction.
    _read_config(vae_config_file, VAE_CONFIG_SHA256)
    siglip_config = _read_config(siglip_config_file, SIGLIP_CONFIG_SHA256)
    semantic_state = _pth_state(semantic_file, max_tensors=2, max_elements=1_153)
    fusion_state = _pth_state(fusion_file, max_tensors=2, max_elements=3)
    with torch.device("cpu"):
        semantic, fusion = LinearHead(1152), LinearHead(2)
    _strict_load(semantic, semantic_state)
    _strict_load(fusion, fusion_state)
    artifact = load_artifact(artifact_file, vae_config_file)
    with torch.device("cpu"):
        siglip = _make_siglip(siglip_config)
    _strict_load(siglip, _safe_state(siglip_file, siglip))
    model = CospyModel(artifact, siglip, semantic, fusion)
    return model.requires_grad_(False).eval().to(target)
