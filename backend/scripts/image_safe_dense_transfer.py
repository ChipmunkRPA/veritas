# SPDX-License-Identifier: Apache-2.0
"""Test local heatmaps derived from SAFE's published image-level checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from scripts.image_local_spectral_peaks import cutoffs, plan, summarize
from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.image-safe-dense-transfer/1"
REVISION = "4e998724651b227def64f5be0cd60c0aa1552c35"
CHECKPOINT_SHA256 = "b3f5ecfb46a154ed553aaaf4bf3ba59182310726ddb0cbb1fe42bd0e22d2f20e"
CHECKPOINT_BYTES = 5_840_638
CELL = 8
WINDOWS = (5, 9, 13, 17)
RING_WIDTH = 4


def load_model(checkpoint):
    import torch
    from torch import nn
    from torchvision.models import resnet50

    if checkpoint.stat().st_size != CHECKPOINT_BYTES or sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Expected the official digest-pinned SAFE checkpoint")
    backbone = resnet50(weights=None)
    model = nn.ModuleDict({"conv1": nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
                           "bn1": backbone.bn1, "layer1": backbone.layer1, "layer2": backbone.layer2,
                           "fc1": nn.Linear(512, 2)})
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(state) != {"model"}:
        raise ValueError("Unexpected official checkpoint structure")
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


def local_logits(model, rgb, *, verify_head=False):
    import numpy as np
    import pywt
    import torch
    from torch.nn import functional as F

    pixels = rgb.astype(np.float32).transpose(2, 0, 1) / np.float32(255.)
    diagonal = pywt.dwt2(pixels, "bior1.3", mode="symmetric", axes=(-2, -1))[1][2]
    with torch.inference_mode():
        wavelet = torch.from_numpy(diagonal[None])
        wavelet = F.interpolate(wavelet, size=rgb.shape[:2], mode="bilinear", align_corners=False, antialias=True)
        features = model["bn1"](model["conv1"](wavelet)).relu()
        features = F.max_pool2d(features, kernel_size=3, stride=2, padding=1)
        features = model["layer2"](model["layer1"](features))
        dense = model["fc1"](features.permute(0, 2, 3, 1))
        if verify_head:
            torch.testing.assert_close(dense.mean((1, 2)), model["fc1"](features.mean((2, 3))), atol=3e-5, rtol=1e-5)
        return (dense[0, :, :, 1] - dense[0, :, :, 0]).numpy()


def box_average(integral, size):
    return (integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]) / size**2


def peak(field, height, width):
    import cv2
    import numpy as np

    integral = cv2.integral(field.astype(np.float32), sdepth=cv2.CV_64F)
    proposals = []
    for size in WINDOWS:
        outer = size + 2 * RING_WIDTH
        if outer > min(field.shape):
            continue
        core = box_average(integral, size)[RING_WIDTH:-RING_WIDTH, RING_WIDTH:-RING_WIDTH]
        enclosure = box_average(integral, outer)
        ring = (enclosure * outer**2 - core * size**2) / (outer**2 - size**2)
        # Require both a positive trained margin and an increase relative to the same picture.
        contrast = np.maximum(0., np.minimum(core, core - ring))
        i, j = np.unravel_index(contrast.argmax(), contrast.shape)
        x, y = CELL * (int(j) + RING_WIDTH) - CELL // 2, CELL * (int(i) + RING_WIDTH) - CELL // 2
        w, h = min(width - x, CELL * size), min(height - y, CELL * size)
        proposals.append((float(contrast[i, j]), [x, y, w, h]))
    if not proposals:
        raise ValueError("The native image is too small for the frozen geometry")
    return max(proposals, key=lambda item: item[0])


def baseline(reference, manifest_path):
    from app.detectors.image import jpeg_dq, regional
    from app.detectors.image.adapters.regional import repeat_components
    from app.detectors.image.regional_support import readout
    from scripts import image_region_rank_reference

    expected = {"manifest_sha256": sha(manifest_path), "code_sha256": sha(Path(image_region_rank_reference.__file__)),
                "regional_sha256": regional.CHECKPOINT_SHA256, "jpeg_sha256": sha(Path(jpeg_dq.__file__)),
                "regional_rule_sha256": sha(Path(readout.__code__.co_filename)), "repeat_jpeg90": True,
                "repeat_rule_sha256": sha(Path(repeat_components.__code__.co_filename))}
    with reference.open() as stream:
        if json.loads(next(stream)) != expected:
            raise ValueError("Existing checks differ from the pinned source and model rules")
        return {row["file"]: row for row in map(json.loads, stream)}


def measure(model, bank, reference, manifest, groups):
    import numpy as np
    from PIL import Image

    from scripts.image_cross_quality_fit import load_mask

    current = baseline(reference, bank / "manifest.json")
    if set(current) != {row["file"] for row in manifest["images"]}:
        raise ValueError("Baseline did not measure the same complete manifest")
    rows, cache = [], {}
    for index, entry in enumerate(manifest["images"], 1):
        body = (bank / entry["file"]).read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        original = current[entry["file"]]
        truth = load_mask(bank, entry)
        height, width = truth.shape
        if digest != entry["sha256"] or digest != original["sha256"] or list(truth.shape) != groups[entry["group"]]["shape"]:
            raise ValueError("An input differs from its baseline or truth commitment")
        if digest not in cache:
            with Image.open(io.BytesIO(body), formats=["JPEG"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            if rgb.shape[:2] != truth.shape:
                raise ValueError("Native model coordinates differ from the truth mask")
            field = local_logits(model, rgb, verify_head=index == 1)
            cache[digest] = peak(field, height, width)
        score, box = cache[digest]
        x, y, w, h = box
        rows.append({key: entry[key] for key in ("file", "role", "quality", "sha256", "mask_sha256")} | {
            "source": groups[entry["group"]]["source"], "family": groups[entry["group"]]["family"],
            "shape": [height, width], "current": {key: original[key] for key in ("jpeg", "regional")},
            "candidate": {"score": score, "box": box, "selected": w*h,
                          "intersection": int(truth[y:y + h, x:x + w].sum()), "truth": int(truth.sum())},
        })
        if index % 48 == 0 or index == len(manifest["images"]):
            print(f"Measured {index}/{len(manifest['images'])} JPEGs.", flush=True)
    return rows


def freeze(checkpoint, bank, reference, output, held):
    manifest, groups, development = plan(bank / "manifest.json")
    holdout = [plan(path / "manifest.json")[2] for path in held]
    for record in holdout:
        if any(set(identities) & set(record["identities"][key]) for key, identities in development["identities"].items()):
            raise ValueError("A held source overlaps calibration")
    rows = measure(load_model(checkpoint), bank, reference, manifest, groups)
    points = cutoffs(rows)
    results = summarize(rows, points)
    save(output, {"schema": SCHEMA, "code_sha256": sha(Path(__file__)), "checkpoint_sha256": CHECKPOINT_SHA256,
                  "development": development, "holdout": holdout, "reference_sha256": sha(reference),
                  "points": points, "summary": results, "rows": rows})
    print(results, flush=True)


def validate(checkpoint, frozen, output, banks):
    lock = read(frozen)
    manifests = [(bank, reference, *plan(bank / "manifest.json")) for bank, reference in banks]
    if (lock["schema"] != SCHEMA or lock["code_sha256"] != sha(Path(__file__)) or lock["points"] != cutoffs(lock["rows"])
            or lock["checkpoint_sha256"] != CHECKPOINT_SHA256 or any(source not in lock["holdout"] for *_, source in manifests)):
        raise ValueError("The held sources, evaluation program or cutoffs changed")
    model = load_model(checkpoint)
    rows = [row for bank, ref, manifest, groups, _ in manifests for row in measure(model, bank, ref, manifest, groups)]
    results = summarize(rows, lock["points"])
    save(output, {"schema": SCHEMA, "freeze_sha256": sha(frozen), "summary": results, "rows": rows,
                  "banks": [{"manifest_sha256": source["manifest_sha256"], "reference_sha256": sha(ref)}
                            for _, ref, _, _, source in manifests]})
    print(results, flush=True)


def review(frozen, report):
    lock, held = read(frozen), read(report)
    points = cutoffs(lock["rows"])
    if (lock["schema"] != SCHEMA or held["schema"] != SCHEMA or held["freeze_sha256"] != sha(frozen)
            or lock["points"] != points or lock["summary"] != summarize(lock["rows"], points)
            or held["summary"] != summarize(held["rows"], points)):
        raise ValueError("The reported cutoffs or outcomes differ from the ledger")
    print(held["summary"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    calibrate = actions.add_parser("freeze")
    for field in ("checkpoint", "bank", "reference", "output"):
        calibrate.add_argument(field, type=Path)
    calibrate.add_argument("--held", type=Path, action="append", required=True)
    compare = actions.add_parser("validate")
    for field in ("checkpoint", "frozen", "output"):
        compare.add_argument(field, type=Path)
    compare.add_argument("--banks", type=Path, nargs=2, action="append", required=True)
    recount = actions.add_parser("review")
    recount.add_argument("frozen", type=Path)
    recount.add_argument("report", type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action != "review":
        import cv2
        import torch

        torch.set_num_threads(2)
        cv2.setNumThreads(1)
    {"freeze": freeze, "validate": validate, "review": review}[action](**args)
