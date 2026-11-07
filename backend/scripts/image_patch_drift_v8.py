# SPDX-License-Identifier: MIT
# Copyright (c) 2026 TRAIL authors
"""Research patch drift for regional editing; never an AI-origin probability.

The four NumPy primitives below are copied unchanged from TRAIL at
0fce0f1de2ac0d219baf91592ee2bca95ec0a219. Retained license:
docs/third-party/TRAIL-v8-MIT.txt. The backbone/readout, own genuine-only operating
points and source/display policy are separately frozen project adaptations.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 448
LAYER = 10
MODEL_ID = 'dinov2.vitb14_reg4'
ARCHITECTURE_ID = 'dinov2.vitl14'
WEIGHT_SHA256 = '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
MEAN = np.asarray([.485, .456, .406], np.float32)
STD = np.asarray([.229, .224, .225], np.float32)
ARMS = ('raw', 'within_image')

def as_float_rgb(image: np.ndarray) -> np.ndarray:
    """Return an HWC float32 RGB array in [0, 1]."""
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB input, got {value.shape}")
    if value.dtype == np.uint8:
        value = value.astype(np.float32) / 255.0
    else:
        value = value.astype(np.float32, copy=True)
        if value.size and (float(value.min()) < -1e-6 or float(value.max()) > 1.0 + 1e-6):
            raise ValueError("floating-point RGB input must be in [0, 1]")
    return np.clip(value, 0.0, 1.0)

def haar_attenuate(image: np.ndarray, alpha: float = 0.2, levels: int = 2) -> np.ndarray:
    """Attenuate Haar detail while retaining its blockwise approximation."""
    if not 0.0 <= alpha <= 1.0 or levels < 1:
        raise ValueError("require alpha in [0, 1] and positive levels")
    value = as_float_rgb(image)
    block = 2**levels
    height, width, channels = value.shape
    if height % block or width % block:
        raise ValueError(f"image geometry must be divisible by Haar block {block}")
    low = value.reshape(height // block, block, width // block, block, channels).mean(axis=(1, 3))
    low = np.repeat(np.repeat(low, block, axis=0), block, axis=1)
    return np.clip(value - alpha * (value - low), 0.0, 1.0).astype(np.float32)

def token_drift(left: np.ndarray, right: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute one-minus-cosine between corresponding token vectors."""
    left_value = np.asarray(left, dtype=np.float32)
    right_value = np.asarray(right, dtype=np.float32)
    if left_value.shape != right_value.shape:
        raise ValueError(f"token shapes differ: {left_value.shape} != {right_value.shape}")
    left_norm = np.linalg.norm(left_value, axis=axis, keepdims=True)
    right_norm = np.linalg.norm(right_value, axis=axis, keepdims=True)
    denominator = np.maximum(left_norm * right_norm, np.finfo(np.float32).eps)
    cosine = np.sum(left_value * right_value, axis=axis, keepdims=True) / denominator
    return (1.0 - np.clip(cosine, -1.0, 1.0)).squeeze(axis).astype(np.float32)

def median3(score_map: np.ndarray) -> np.ndarray:
    """Apply the paper's reflect-padded 3x3 spatial median readout."""
    value = np.asarray(score_map, dtype=np.float32)
    if value.ndim < 2:
        raise ValueError("score map must have at least two dimensions")
    padding = [(0, 0)] * (value.ndim - 2) + [(1, 1), (1, 1)]
    padded = np.pad(value, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(-2, -1))
    return np.median(windows, axis=(-2, -1)).astype(np.float32)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def views(rgb):
    resized = np.asarray(Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
    clean = as_float_rgb(resized)
    changed = haar_attenuate(clean, alpha=.2, levels=2)
    normalized = ((np.stack([clean, changed]) - MEAN) / STD).transpose(0, 3, 1, 2).copy()
    return clean, changed, normalized


def load(main, device='mps'):
    import os

    import torch

    from app.detectors.image.radar import build_semantic
    from app.engine.fetchers import ensure_verified, torch_hub_dir
    from app.engine.registry import get_registry

    os.environ['XFORMERS_DISABLED'] = '1'
    reg = get_registry()
    weight, architecture = reg.manifest.model(MODEL_ID), reg.manifest.model(ARCHITECTURE_ID)
    if any(m.licence.class_ != 'commercial' for m in (weight, architecture)):
        raise ValueError('Both native architecture and backbone weights require commercial admission')
    files = ensure_verified(weight)
    if len(files) != 1 or digest(files[0]) != WEIGHT_SHA256:
        raise ValueError('Exact reviewed register-backbone weights required')
    if torch.serialization.get_unsafe_globals_in_checkpoint(files[0]):
        raise ValueError('Unexpected pickle globals in the native backbone checkpoint')
    model = build_semantic(torch_hub_dir(architecture))
    model.load_state_dict(torch.load(files[0], map_location='cpu', weights_only=True, mmap=True), strict=True)
    if (len(model.blocks) != 12 or model.num_register_tokens != 4 or model.embed_dim != 768
            or model.patch_embed.patch_size != (14, 14)
            or sum(p.numel() for p in model.parameters()) != 86583552):
        raise ValueError('Unexpected native register-backbone architecture')
    model.eval().requires_grad_(False)
    return model.to(device)


def infer(model, rgb, *, native_reference=False, retain_features=False):
    import torch

    clean, changed, normalized = views(rgb)
    device = next(model.parameters()).device
    tensor = torch.from_numpy(normalized).to(device)
    with torch.inference_mode():
        values = model.get_intermediate_layers(tensor, n=[LAYER - 1], norm=True)[0].float()
    if values.shape != (2, 1024, 768) or not torch.isfinite(values).all():
        raise ValueError('Unexpected native layer/patch/register output')
    features = values.cpu().numpy().copy()
    raw = token_drift(features[0], features[1]).reshape(32, 32)
    filtered = median3(raw)
    if not np.isfinite(filtered).all() or filtered.shape != (32, 32):
        raise ValueError('Invalid patch-drift response map')
    proof = {'clean_float32_sha256': hashlib.sha256(clean.tobytes()).hexdigest(),
        'changed_float32_sha256': hashlib.sha256(changed.tobytes()).hexdigest(),
        'normalized_nchw_float32_sha256': hashlib.sha256(normalized.tobytes()).hexdigest(),
        'native_features_float32_sha256': hashlib.sha256(features.tobytes()).hexdigest(),
        'native_raw_map_float32_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
        'filtered_map_float32_sha256': hashlib.sha256(filtered.tobytes()).hexdigest(),
        'native_shape': list(features.shape), 'dtype': 'float32', 'readout_one_based_block': LAYER,
        'reference_compared': False}
    if native_reference:
        captured = {}

        def capture(_module, _inputs, output):
            captured['features'] = model.norm(output)[:, -1024:, :].float().detach().cpu().numpy().copy()

        hook = model.blocks[LAYER - 1].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                model(tensor)
        finally:
            hook.remove()
        difference = float(np.max(np.abs(features - captured['features'])))
        if difference != 0:
            raise ValueError('Native intermediate-layer extraction differs from full-forward hook')
        proof.update(reference_compared=True, native_hook_feature_max_difference=difference)
    arrays = {'raw': raw, 'filtered': filtered}
    if retain_features:
        arrays['private_features'] = features
    return arrays, proof


def scores(filtered, arm):
    value = np.asarray(filtered, np.float32)
    if value.shape != (32, 32) or not np.isfinite(value).all():
        raise ValueError('A complete native filtered map is required')
    if arm == 'raw':
        return value.copy()
    if arm != 'within_image':
        raise ValueError('Unknown predeclared patch-drift arm')
    low, high = np.quantile(value.astype(np.float64), [.25, .75], method='linear')
    scale = max(float(high - low), float(np.finfo(np.float32).eps))
    return ((value.astype(np.float64) - float(np.median(value))) / scale).astype(np.float32)


def screening(scores, rgb, image_bytes):
    from app.detectors.image import jpeg_bitstream
    from app.detectors.image import localiser as loc

    height, width = rgb.shape[:2]
    grid = loc.grid_shape(width, height, loc.GRID_MAX_SIDE['screening'])
    floating = cv2.resize(scores, grid, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    structure = jpeg_bitstream.read(image_bytes)
    excluded, _ = loc.conditions_mask(rgb, structure.quality_luma if structure else None)
    reliable = loc.reliable_grid(excluded, grid)
    if floating.shape != reliable.shape or not np.isfinite(floating).all():
        raise ValueError('Source screening/reliability geometry differs')
    return floating, reliable


def encoded_grid(floating, point):
    """Threshold-aware display encoding; raw distances are never probabilities.

    The high bit preserves strict eligibility exactly across uint8 serialization.
    Colour magnitude is relative to twice the genuine-only operating point.
    """
    if not np.isfinite(floating).all() or not np.isfinite(point) or point <= 0:
        raise ValueError('Finite scores and positive genuine-only point required')
    scaled = np.rint(np.clip(floating.astype(np.float64) / (2 * point), 0, 1) * 255).astype(np.uint8)
    return np.where(floating > point, np.maximum(scaled, 128), np.minimum(scaled, 127)).astype(np.uint8)


def predict_regions(floating, reliable, point, width, height, truth):
    from scripts.localization_benchmark import component_masks, mask_metrics, policy

    encoded = encoded_grid(floating, point)
    values = encoded.astype(np.float64) / 255
    masks, facts = component_masks((encoded >= 128) & reliable, width=width, height=height,
        rule=policy(), values=values)
    displayed = masks['per_map_display_cap']
    if displayed is None:
        raise ValueError('Complete scalar ordering is required for the display cap')
    projected = cv2.resize(displayed.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    return mask_metrics(projected, truth), {'display_grid': encoded, 'reliable': reliable, 'selected_grid': displayed}, facts
