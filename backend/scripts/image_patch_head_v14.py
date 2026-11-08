"""Research-only supervised patch head; no serving or confidence imports.

Frozen licensed DINOv2 features supply a learned representation. A small head
learns regional edit supervision rather than treating token drift as edit truth.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image
from scipy.ndimage import label, median_filter

SIDE, INPUT_SIDE, DIMENSION, SAMPLES = 32, 448, 2304, 64
LAYERS = (3, 7, 11)  # zero-based blocks 4, 8 and 12
SEED, EPOCHS, BATCH, HIDDEN = 20261108, 30, 512, 64
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def normalize(rgb):
    require(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3, 'Stored uint8 RGB required')
    values = np.asarray(Image.fromarray(rgb).resize((INPUT_SIDE, INPUT_SIDE), Image.Resampling.BILINEAR))
    return (((values.astype(np.float32) / 255) - MEAN) / STD).transpose(2, 0, 1)[None].copy()


def extract(backbone, rgb, hook=False):
    import torch

    tensor = torch.from_numpy(normalize(rgb))
    with torch.inference_mode():
        values = backbone.get_intermediate_layers(tensor, n=list(LAYERS), norm=True)
    require(len(values) == 3 and all(v.shape == (1, SIDE * SIDE, 768) for v in values),
            'Exact three-layer patch geometry required')
    features = torch.cat(values, dim=-1)[0].float().numpy().copy()
    require(np.isfinite(features).all(), 'Complete finite native features required')
    if hook:
        captured, handles = {}, []
        for layer in LAYERS:
            def capture(_module, _input, output, index=layer):
                captured[index] = backbone.norm(output)[:, 5:, :].detach().float().numpy().copy()
            handles.append(backbone.blocks[layer].register_forward_hook(capture))
        try:
            with torch.inference_mode():
                backbone(tensor)
        finally:
            for handle in handles:
                handle.remove()
        reference = np.concatenate([captured[k][0] for k in LAYERS], axis=1)
        require(np.array_equal(reference, features), 'Full-forward hooks differ from native layer extraction')
    return features, features.mean(axis=0, dtype=np.float32), sha(tensor.numpy())


def pool_truth(truth):
    """Exact source pixels assigned to edge-aligned cells, including odd sizes."""
    require(truth.dtype == bool and min(truth.shape) >= SIDE, 'Exact boolean source truth required')
    height, width = truth.shape
    index = (np.arange(height)[:, None] * SIDE // height) * SIDE + np.arange(width)[None, :] * SIDE // width
    counts = np.bincount(index.ravel(), minlength=SIDE * SIDE)
    positive = np.bincount(index.ravel(), weights=truth.ravel(), minlength=SIDE * SIDE)
    return (positive / counts).reshape(SIDE, SIDE).astype(np.float32)


def indices(target, key):
    flat = target.ravel()
    rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    positive, negative = np.flatnonzero(flat > 0), np.flatnonzero(flat == 0)
    if len(positive) and len(negative):
        return np.r_[rng.choice(positive, SAMPLES // 2, replace=len(positive) < SAMPLES // 2),
                     rng.choice(negative, SAMPLES // 2, replace=len(negative) < SAMPLES // 2)]
    return rng.choice(len(flat), SAMPLES, replace=False)


def new_head(dimension=DIMENSION):
    import torch

    return torch.nn.Sequential(torch.nn.Linear(dimension * 2, HIDDEN), torch.nn.ReLU(),
                               torch.nn.Linear(HIDDEN, 1))


def train(samples, means, targets, *, epochs=EPOCHS):
    import torch
    from torch.nn import functional as f

    require(samples.ndim == 3 and samples.shape[1] == SAMPLES
            and means.shape == (samples.shape[0], samples.shape[2])
            and targets.shape == samples.shape[:2] and np.isfinite(targets).all()
            and ((targets >= 0) & (targets <= 1)).all(), 'Complete fit-only samples required')
    y = targets.reshape(-1)
    positive, negative = float(y.sum(dtype=np.float64)), float((1 - y).sum(dtype=np.float64))
    require(positive > 0 and negative > 0, 'Both soft supervision classes required')
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    head = new_head(samples.shape[2])
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    data, rng, losses = samples.reshape(-1, samples.shape[-1]), np.random.default_rng(SEED), []
    for epoch in range(epochs):
        total = 0.
        for selected in np.array_split(rng.permutation(len(y)), math.ceil(len(y) / BATCH)):
            base = torch.from_numpy(np.array(data[selected], copy=True))
            average = torch.from_numpy(np.array(means[selected // SAMPLES], copy=True))
            target = torch.from_numpy(np.array(y[selected], copy=True))
            prediction = head(torch.cat([base, base - average], dim=1)).flatten()
            loss = (f.softplus(-prediction) * target * len(y) / (2 * positive)
                    + f.softplus(prediction) * (1 - target) * len(y) / (2 * negative)).mean()
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss retained as failure')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(selected) / len(y)
        losses.append(total)
        if (epoch + 1) % 5 == 0:
            print({'head_epoch': epoch + 1, 'balanced_soft_loss': total}, flush=True)
    head.eval().requires_grad_(False)
    return head, {'epochs': epochs, 'steps': epochs * math.ceil(len(y) / BATCH), 'losses': losses,
                  'soft_positive_mass': positive, 'soft_negative_mass': negative,
                  'parameters': sum(p.numel() for p in head.parameters()), 'calibrated_confidence': False}


def response(head, features, average):
    import torch

    require(features.shape == (SIDE * SIDE, DIMENSION) and average.shape == (DIMENSION,),
            'Exact native patch vectors and full-image mean required')
    with torch.inference_mode():
        values = np.concatenate([features, features - average], axis=1)
        scores = torch.sigmoid(head(torch.from_numpy(values))).numpy().reshape(SIDE, SIDE)
    require(np.isfinite(scores).all(), 'Finite ranking grid required')
    return median_filter(scores, size=3, mode='reflect').astype(np.float32)


def selected(scores, point, width, height):
    require(scores.shape == (SIDE, SIDE) and np.isfinite(scores).all() and 0 <= point <= 1
            and min(width, height) >= SIDE, 'Valid ranking grid, point and source geometry required')
    components, _ = label(scores > point, structure=np.ones((3, 3)))
    keep = np.bincount(components.ravel()) >= 4
    keep[0] = False
    grid = keep[components]
    return grid[np.arange(height) * SIDE // height][:, np.arange(width) * SIDE // width]
