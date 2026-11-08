"""Pinned MIT public-family spatial watermark research, never an AI-origin verdict."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

LOCK = Path(__file__).with_name("public_visual_watermark_assets_v1.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verify(main, source):
    lock = read(LOCK)
    for row in lock["source_files"]:
        p = source / "watermark-anything" / row["path"]
        require(
            p.stat().st_size == row["bytes"] and digest(p) == row["sha256"], "Pinned source/config changed"
        )
    dep = main / lock["dependencies"]["manifest"]
    require(digest(dep) == lock["dependencies"]["manifest_sha256"], "Pinned dependency manifest changed")
    root = main / lock["dependencies"]["private_root"]
    for row in read(dep)["files"]:
        require(digest(root / row["path"]) == row["sha256"], "Pinned dependency file changed")
    return lock, root


def load_model(main, source):
    lock, dependencies = verify(main, source)
    # The explicitly MIT S3 object must also match the pinned publisher mirror.
    # Keep it in memory; do not fill local storage or change serving weights.
    asset = lock["checkpoint"]
    stream = io.BytesIO()
    h = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(asset["source"], timeout=60) as response:
        require(response.status == 200, "Complete public checkpoint response required")
        while block := response.read(1024**2):
            count += len(block)
            require(count <= asset["bytes"], "Checkpoint exceeds pinned byte count")
            h.update(block)
            stream.write(block)
    require(
        count == asset["bytes"] and h.hexdigest() == asset["sha256"],
        "Explicit MIT object does not match pinned publisher byte identity",
    )
    stream.seek(0)
    sys.path[:0] = [str(dependencies), str(source / "watermark-anything")]
    import torch
    import yaml
    from omegaconf import OmegaConf
    from watermark_anything.augmentation.augmenter import Augmenter
    from watermark_anything.data.transforms import normalize_img, unnormalize_img
    from watermark_anything.models import Wam, build_embedder, build_extractor
    from watermark_anything.modules.jnd import JND

    require(
        torch.serialization.get_unsafe_globals_in_checkpoint(stream) == [],
        "Checkpoint must contain data-only tensors without extra pickle globals",
    )
    stream.seek(0)
    state = torch.load(stream, map_location="cpu", weights_only=True)
    require(
        isinstance(state, dict)
        and state
        and all(
            isinstance(k, str)
            and isinstance(v, torch.Tensor)
            and v.dtype == torch.float32
            and torch.isfinite(v).all()
            for k, v in state.items()
        ),
        "Finite float32 tensor-only state required",
    )
    stream.close()

    def forbidden(*args, **kwargs):
        raise AssertionError("No additional network, checkpoint loader or pretrained hub permitted")

    torch.load = forbidden
    torch.hub.load = forbidden
    torch.hub.load_state_dict_from_url = forbidden
    urllib.request.urlopen = forbidden
    base = source / "watermark-anything"
    params = read(base / "checkpoints/params.json")

    def cfg(name):
        return OmegaConf.create(yaml.safe_load((base / name).read_text()))

    e = cfg(params["embedder_config"])
    d = cfg(params["extractor_config"])
    a = cfg(params["augmentation_config"])
    j = cfg(params["attenuation_config"])
    embedder = build_embedder(params["embedder_model"], e[params["embedder_model"]], params["nbits"])
    detector = build_extractor(d.model, d[params["extractor_model"]], params["img_size"], params["nbits"])
    augmenter = Augmenter(**a)
    attenuation = JND(**j[params["attenuation"]], preprocess=unnormalize_img, postprocess=normalize_img)
    model = Wam(
        embedder,
        detector,
        augmenter,
        attenuation,
        params["scaling_w"],
        params["scaling_i"],
        img_size_extractor=256,
    )
    model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    require(
        set(actual) == set(state) and all(torch.equal(actual[k], v) for k, v in state.items()),
        "Every final tensor must match the pinned original",
    )
    receipt = {
        "checkpoint_sha256": asset["sha256"],
        "checkpoint_bytes": count,
        "state_tensors": len(state),
        "loaded_tensors_verified": True,
        "parameters": sum(p.numel() for p in model.parameters()),
        "weights_only": True,
        "unrestricted_pickle_used": False,
        "stored_checkpoint_on_disk": False,
    }
    del state
    model.eval().requires_grad_(False)
    return model, receipt


def spatial(logits):
    require(
        logits.shape == (256, 256) and logits.dtype == np.float32 and np.isfinite(logits).all(),
        "Exact finite source-coordinate logits required",
    )
    mask = logits > 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = []
    for i in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area >= 500:
            regions.append({"x": x, "y": y, "w": w, "h": h, "pixels": area})
    return {
        "selected_pixels": int(mask.sum()),
        "selected_fraction": float(mask.mean()),
        "regions": regions,
        "flagged_for_inspection": bool(regions),
        "score_meaning": "Public watermark-pattern mask, not AI-authorship confidence.",
        "provider_authenticated": False,
        "used_in_confidence": False,
        "early_stop_eligible": False,
    }


def overlap(mask, truth):
    union = np.count_nonzero(mask | truth)
    inside = np.count_nonzero(mask & truth)
    return {
        "iou": float(inside / union) if truth.any() else None,
        "coverage": float(inside / np.count_nonzero(truth)) if truth.any() else None,
        "pixels_selected_outside": int(np.count_nonzero(mask & ~truth)),
    }


def authored():
    y, x = np.mgrid[:256, :256]
    rng = np.random.default_rng(20261108)
    images = {
        "gradient": np.stack((40 + 180 * x / 255, 40 + 180 * y / 255, np.full_like(x, 128)), axis=-1),
        "checker": np.repeat((40 + 170 * ((x // 16 + y // 16) % 2))[..., None], 3, axis=-1),
        "waves": np.stack(
            (128 + 70 * np.sin(x / 8), 128 + 70 * np.cos(y / 11), 128 + 70 * np.sin((x + y) / 15)), axis=-1
        ),
        "texture": np.clip(rng.normal(128, 30, (256, 256, 3)), 20, 235),
        "flat": np.full((256, 256, 3), 128),
        "rings": np.repeat((128 + 90 * np.sin(np.hypot(x - 128, y - 128) / 5))[..., None], 3, axis=-1),
    }
    return {k: np.ascontiguousarray(np.rint(v).astype(np.uint8)) for k, v in images.items()}
