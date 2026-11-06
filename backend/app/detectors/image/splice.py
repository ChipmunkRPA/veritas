"""Offline multi-quality recompression localizer for the pinned splice checkpoint."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageChops, ImageEnhance
from torch import nn
from torch.nn import functional as F

INPUT_SIZE = 384
PIXEL_THRESHOLD = 0.70
QUALITIES = (75, 85, 95)
MEAN = (0.0684, 0.0605, 0.0402)
STD = (0.0656, 0.0604, 0.0471)
CHECKPOINT_SHA256 = "ca3aa144e6ca08f2bb2eae93ac45d76c27a2922770e028bf90ef76ce974ee061"
CHECKPOINT_BYTES = 97_958_262
CHECKPOINT_FILE = "tidal-v30.1.pt"
MODEL_ID = "tidal.splice_v30_1"
MINIMUM_ACTIVE_CELLS = 7993  # Frozen on 24 development source pairs, see docs/image-localizer-upgrade.md.


def recompression_features(image: Image.Image) -> torch.Tensor:
    """Stack contrast-scaled grayscale JPEG residuals at three qualities."""
    rgb = image.convert("RGB")
    channels = []
    for quality in QUALITIES:
        with io.BytesIO() as buffer:
            rgb.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                difference = ImageChops.difference(rgb, encoded.convert("RGB"))
        peak = max(maximum for _, maximum in difference.getextrema())
        scaled = ImageEnhance.Brightness(difference).enhance(255.0 / max(1, peak))
        grayscale = scaled.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        channels.append(np.asarray(grayscale, dtype=np.uint8))
    tensor = torch.from_numpy(np.stack(channels).astype(np.float32) / 255.0)
    mean = tensor.new_tensor(MEAN)[:, None, None]
    std = tensor.new_tensor(STD)[:, None, None]
    return ((tensor - mean) / (std + 1e-8)).unsqueeze(0)


class _ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(1, channels // 16)
        self.fc = nn.Sequential(nn.Linear(channels, hidden, bias=False), nn.ReLU(),
                                nn.Linear(hidden, channels, bias=False))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(feature, 1).flatten(1)
        pooled = self.fc(average) + self.fc(maximum)
        return feature * pooled.sigmoid()[..., None, None]


class _SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat((feature.mean(dim=1, keepdim=True), feature.amax(dim=1, keepdim=True)), dim=1)
        return feature * self.conv(pooled).sigmoid()


class _DecoderAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attn = _ChannelAttention(channels)
        self.spatial_attn = _SpatialAttention()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.spatial_attn(self.channel_attn(feature))


def load_model(checkpoint: Path) -> nn.Module:
    """Load the exact published state on CPU without network or pickle execution."""
    if checkpoint.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError("Unexpected image splice checkpoint size")
    with checkpoint.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Unexpected image splice checkpoint digest")
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):
        raise ValueError("Unexpected pickle globals in the image splice checkpoint")

    from segmentation_models_pytorch import Unet

    decoder_channels = (256, 128, 64, 32, 16)
    model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1,
                 decoder_channels=decoder_channels)
    for block, channels in zip(model.decoder.blocks, decoder_channels, strict=True):
        block.attention2 = _DecoderAttention(channels)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True), strict=True)
    return model.eval()


@torch.inference_mode()
def predict(model: nn.Module, image: Image.Image) -> np.ndarray:
    raw = model(recompression_features(image)).sigmoid()[0, 0].numpy().copy()
    if raw.shape != (INPUT_SIZE, INPUT_SIZE) or not np.isfinite(raw).all():
        raise RuntimeError("Image splice inference produced an invalid response map")
    return raw


def load_splice(model_id: str, device: str, dtype: str):
    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file
    from app.engine.models import ModelUnavailable

    if model_id != MODEL_ID or device != "cpu" or dtype != "float32":
        raise ModelUnavailable("The local editing check uses its pinned CPU float32 model")
    entry = _entry(model_id)
    return LoadedModel(load_model(_file(entry, CHECKPOINT_FILE)), CHECKPOINT_SHA256)
