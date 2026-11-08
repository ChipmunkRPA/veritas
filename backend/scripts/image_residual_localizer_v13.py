"""Independent research localizer: noise/texture features and nuisance projection.

No pretrained weights, provider calls, paired original at inference, or serving
imports. Responses are experimental ranking scores, not calibrated confidence.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter, label, median_filter, uniform_filter
from scipy.optimize import minimize
from scipy.special import expit

SIDE = 64
MIN_COMPONENT = 16
SAMPLE_PIXELS = 128
FEATURES = tuple(
    f"{name}.{form}"
    for form in ("raw", "within_image")
    for name in (
        "variance.3",
        "variance.7",
        "variance.15",
        "noise.3",
        "noise.7",
        "noise.15",
        "chroma_noise.7",
        "noise_texture_ratio.3",
        "noise_texture_ratio.7",
        "noise_scale_ratio",
        "variance_scale_ratio",
        "noise_adjacent_x",
        "noise_adjacent_y",
        "noise_absolute_ratio",
        "noise_kurtosis",
        "chroma_luma_ratio",
    )
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_array(value):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def grid(value):
    return np.asarray(
        Image.fromarray(np.asarray(value, np.float32)).resize((SIDE, SIDE), Image.Resampling.BOX), np.float32
    )


def features(rgb):
    require(
        rgb.dtype == np.uint8
        and rgb.ndim == 3
        and rgb.shape[2] == 3
        and min(rgb.shape[:2]) >= SIDE
        and max(rgb.shape[:2]) <= 4096,
        "Supported source RGB required",
    )
    image = rgb.astype(np.float32) / 255.0
    y = image @ np.array([0.299, 0.587, 0.114], np.float32)
    noise = y - gaussian_filter(y, 0.8, mode="reflect", truncate=3.0)
    power = noise * noise
    chroma = (image[:, :, 0] - image[:, :, 1], image[:, :, 2] - image[:, :, 1])
    chroma_power = sum((c - gaussian_filter(c, 0.8, mode="reflect", truncate=3.0)) ** 2 for c in chroma) / 2
    variance, energy = {}, {}
    for radius in (3, 7, 15):
        variance[radius] = np.maximum(
            uniform_filter(y * y, radius, mode="reflect") - uniform_filter(y, radius, mode="reflect") ** 2,
            0.0,
        )
        energy[radius] = uniform_filter(power, radius, mode="reflect")
    eps = np.float32(1e-6)

    def log(x):
        return np.log(np.maximum(x, 0.0) + eps)

    raw = [log(variance[r]) for r in (3, 7, 15)] + [log(energy[r]) for r in (3, 7, 15)]
    cn = uniform_filter(chroma_power, 7, mode="reflect")
    shifted_x = np.pad(noise[:, :-1], ((0, 0), (1, 0)), mode="reflect")
    shifted_y = np.pad(noise[:-1, :], ((1, 0), (0, 0)), mode="reflect")
    raw += [
        log(cn),
        log(energy[3]) - log(variance[3]),
        log(energy[7]) - log(variance[7]),
        log(energy[3]) - log(energy[15]),
        log(variance[3]) - log(variance[15]),
        uniform_filter(noise * shifted_x, 7, mode="reflect") / (energy[7] + eps),
        uniform_filter(noise * shifted_y, 7, mode="reflect") / (energy[7] + eps),
        uniform_filter(np.abs(noise), 7, mode="reflect") / np.sqrt(energy[7] + eps),
        np.minimum(uniform_filter(power * power, 7, mode="reflect") / (energy[7] ** 2 + eps**2), 100.0),
        log(cn) - log(energy[7]),
    ]
    small = np.stack([grid(v) for v in raw], axis=-1)
    flat = small.reshape(-1, small.shape[-1])
    center = np.median(flat, axis=0)
    q25, q75 = np.quantile(flat, [0.25, 0.75], axis=0)
    relative = np.clip((small - center) / np.maximum(q75 - q25, 0.05), -12.0, 12.0)
    result = np.concatenate([small, relative], axis=-1).astype(np.float32)
    require(
        result.shape == (SIDE, SIDE, len(FEATURES)) and np.isfinite(result).all(),
        "Finite source feature grid required",
    )
    return result


def intervene(rgb, kind):
    require(kind in ("original", "lighting", "jpeg55", "ordinary_patch"), "Unknown intervention")
    if kind == "original":
        return rgb.copy()
    if kind == "lighting":
        return np.clip(np.rint((rgb.astype(np.float32) - 127.5) * 0.88 + 127.5 + 12.0), 0, 255).astype(
            np.uint8
        )
    image = Image.fromarray(rgb)
    if kind == "jpeg55":
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=55, subsampling=2, optimize=False)
        with Image.open(io.BytesIO(out.getvalue())) as decoded:
            return np.asarray(decoded.convert("RGB")).copy()
    w, h = image.size
    box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    patch = image.crop(box).filter(ImageFilter.GaussianBlur(1.2))
    pixels = np.asarray(patch).astype(np.float32)
    patch = Image.fromarray(np.clip(np.rint(pixels * 0.94 + 5.0), 0, 255).astype(np.uint8))
    image.paste(patch, box)
    return np.asarray(image).copy()


def sample_indices(truth, seed):
    rng = np.random.default_rng(seed)
    flat = truth.reshape(-1)
    positive, negative = np.flatnonzero(flat > 0), np.flatnonzero(flat == 0)
    # Keep source cells, including partially edited cells, as soft targets.
    if len(positive) and len(negative):
        return np.concatenate(
            [
                rng.choice(positive, SAMPLE_PIXELS // 2, replace=len(positive) < SAMPLE_PIXELS // 2),
                rng.choice(negative, SAMPLE_PIXELS // 2, replace=len(negative) < SAMPLE_PIXELS // 2),
            ]
        )
    return rng.choice(len(flat), SAMPLE_PIXELS, replace=False)


def projector(deltas, scale, remove):
    require(remove in (0, 2), "Only predeclared projection ranks supported")
    dimension = len(scale)
    if remove == 0:
        return np.eye(dimension)
    covariance = np.zeros((dimension, dimension), np.float64)
    count = 0
    for delta in deltas:
        z = np.asarray(delta, np.float64) / scale
        covariance += z.T @ z
        count += len(z)
    require(count > 0, "Fit-source intervention deltas required")
    _, vectors = np.linalg.eigh(covariance / count)
    nuisance = vectors[:, -remove:]
    result = np.eye(dimension) - nuisance @ nuisance.T
    require(
        np.allclose(result @ result, result, atol=1e-10) and np.allclose(result, result.T),
        "Orthogonal projector required",
    )
    return result


def fit(x, y, deltas, remove):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    require(
        x.ndim == 2
        and x.shape[1] == len(FEATURES)
        and len(y) == len(x)
        and np.isfinite(x).all()
        and np.isfinite(y).all()
        and np.all((y >= 0) & (y <= 1))
        and y.sum() > 0
        and (1 - y).sum() > 0,
        "Valid fit-only samples required",
    )
    center, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.05)
    projection = projector(deltas, scale, remove)
    design = (x - center) / scale @ projection
    # Balanced soft positive/negative mass; outputs are ranking responses only.
    pos, neg = 0.5 / y.sum(), 0.5 / (1 - y).sum()
    mass = y * pos + (1 - y) * neg
    target = y * pos

    def objective(parameters):
        weight, bias = parameters[:-1], parameters[-1]
        z = design @ weight + bias
        loss = np.dot(mass, np.logaddexp(0.0, z)) - np.dot(target, z) + 0.01 * np.dot(weight, weight) / 2
        error = mass * expit(z) - target
        derivative = np.r_[design.T @ error + 0.01 * weight, error.sum()]
        return loss, derivative

    outcome = minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-7},
    )
    require(outcome.success and np.isfinite(outcome.fun), "Fit did not converge")
    return {
        "center": center.tolist(),
        "scale": scale.tolist(),
        "projection": projection.tolist(),
        "weight": outcome.x[:-1].tolist(),
        "bias": float(outcome.x[-1]),
        "removed_dimensions": remove,
        "objective": float(outcome.fun),
        "iterations": int(outcome.nit),
        "calibrated_confidence": False,
    }


def response(values, head):
    require(
        values.shape == (SIDE, SIDE, len(FEATURES)) and np.isfinite(values).all(),
        "Bound feature grid required",
    )
    z = (values.astype(np.float64) - head["center"]) / head["scale"] @ np.asarray(head["projection"])
    z = z @ np.asarray(head["weight"]) + head["bias"]
    return median_filter(expit(z), size=3, mode="reflect")


def point(group_maxima):
    values = sorted(group_maxima, reverse=True)
    require(len(values) >= 25 and np.isfinite(values).all(), "At least 25 calibration-source maxima required")
    budget = math.floor(0.05 * len(values))
    return {
        "value": float(values[budget]),
        "comparison": "strict_greater_than",
        "calibration_groups": len(values),
        "maximum_flagged_calibration_groups": budget,
        "scope": "Fixed research comparison point, not an AI confidence or public screening threshold",
    }


def mask(scores, threshold, width, height):
    require(
        scores.shape == (SIDE, SIDE)
        and np.isfinite(scores).all()
        and 0 <= threshold <= 1
        and min(width, height) >= SIDE,
        "Valid response and source geometry required",
    )
    binary = scores > threshold
    regions, _ = label(binary, structure=np.ones((3, 3)))
    sizes = np.bincount(regions.ravel())
    keep = sizes >= MIN_COMPONENT
    keep[0] = False
    selected = keep[regions]
    # Edge-aligned cells: source pixel (x,y) maps to floor(x*SIDE/w), floor(y*SIDE/h).
    return selected[np.arange(height) * SIDE // height][:, np.arange(width) * SIDE // width]


def metrics(prediction, truth):
    require(
        prediction.dtype == truth.dtype == bool and prediction.shape == truth.shape,
        "Matching exact source masks required",
    )
    tp, fp, fn = (
        int((prediction & truth).sum()),
        int((prediction & ~truth).sum()),
        int((~prediction & truth).sum()),
    )
    return {
        "intersection": tp,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "reference_positive_pixels": int(truth.sum()),
        "predicted_positive_pixels": int(prediction.sum()),
        "pixels": int(truth.size),
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "flag": bool(prediction.any()),
        "selected_area_fraction": float(prediction.mean()),
    }
