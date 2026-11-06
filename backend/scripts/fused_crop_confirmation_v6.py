"""Source-bound fresh confirmation of the unchanged image-crop method.

Freezing reads images and reference masks but performs no detector inference.
No raw response, scalar, confidence rule or development gate is refitted here.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, features

from app.detectors.image import fused, splice
from scripts.localization_benchmark import read_manifest, sha, source
from scripts.score_fused_crop_v6 import code_pins as development_code_pins

PLAN_SHA256 = "024d8837c9ec07601f3a41efcc91d90679d9c31d3c27f9c66bcfd49e97b2b3a5"
DEVELOPMENT_REVIEW_SHA256 = "9c9572eb66af942b4e94f262714b9ecc89d50fbf742d57e664dc3ba6e05642cc"
DEVELOPMENT_PREDICTIONS_SHA256 = "64eab2ae475e81802002b1a20bd415665a2cb1159cb2ffb1ff1844d52fec2ece"
DEVELOPMENT_FREEZE_SHA256 = "d4692a6d1ee68a2dcb74588b53ac8a9a3868c19bf43885be1801b6d249227f17"
HISTORIES = ("tgif-pairs-v2", "tgif-confirmation-v3", "tgif-confirmation-v4", "tgif-confirmation-v5")
WEB_IDS = {"10393", "15440", "15810", "24260", "26501", "3983", "6954", "7938"}
SCHEMA = "chipmunk.fused-crop-confirmation/6"


def code_pins() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    out = development_code_pins()
    old = root / "docs/evaluation-panels/fused-crop-v6-development-freeze.json"
    if sha(old.read_bytes()) != DEVELOPMENT_FREEZE_SHA256 or json.loads(old.read_text())["code_sha256"] != out:
        raise ValueError("Original frozen development method changed")
    for name in ("backend/scripts/fused_crop_confirmation_v6.py",
                 "backend/scripts/run_fused_crop_confirmation_v6.py",
                 "backend/tests/test_fused_crop_confirmation_v6.py",
                 "backend/tests/test_fused_crop_v6_review.py", "backend/tests/test_fused_production_grid.py",
                 "backend/scripts/prepare_tgif_localization.py", "docs/fused-crop-v6-confirmation-protocol.md"):
        out[name] = sha((root / name).read_bytes())
    return out


def weights() -> dict[str, str]:
    return {"inpainting": fused.CHECKPOINT_SHA256, "semantic": fused.BACKBONE_SHA256,
            "native_baseline": splice.CHECKPOINT_SHA256}


def runtime() -> dict:
    return {"packages": {n: importlib.metadata.version(n) for n in
                         ("torch", "torchvision", "timm", "numpy", "Pillow", "safetensors")},
            "opencv": {"version": cv2.__version__, "distributions": {
                n: importlib.metadata.version(n) for n in importlib.metadata.packages_distributions().get("cv2", [])}},
            "jpeg_codec": features.version_codec("jpg"), "device": "mps", "dtype": "float32"}


def phash(pixels: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    scaled = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    ac = cv2.dct(scaled)[:8, :8].ravel()[1:]
    return ac > np.median(ac)


def paired_items(manifest: dict, plan: dict, historical_ids: set[str], historical_hashes: set[str]) -> dict:
    pairs = {str(p["source_id"]): p for group in plan["groups"].values() for p in group}
    items = manifest["items"]
    if (len(pairs) != 48 or len(items) != 96 or {i["source_id"] for i in items} != pairs.keys()
            or len({(i["source_id"], i["label"]) for i in items}) != 96
            or any(i["label"] not in (0, 1) or isinstance(i["label"], bool) for i in items)
            or {f: sum(i["family"] == f and i["label"] == 0 for i in items)
                for f in ("firefly", "sd2", "flux-fill")} != {"firefly": 16, "sd2": 16, "flux-fill": 16}
            or len({i["image_sha256"] for i in items}) != 96
            or set(pairs) & historical_ids
            or {i["image_sha256"] for i in items} & historical_hashes):
        raise ValueError("Incomplete, duplicated or historically reused confirmation sources")
    for item in items:
        pair = pairs[item["source_id"]]
        role = "image" if item["label"] else "original"
        reference = item["archive_source"]
        if (item["family"] != pair["family"] or [reference["archive"], reference["member"]] != pair[role]
                or reference["sha256"] != item["image_sha256"]):
            raise ValueError("Source provenance differs from its fixed public plan")
    return pairs


def source_audit(root: Path, historical_root: Path) -> dict:
    plan_path = root / "tgif-plan.json"
    if sha(plan_path.read_bytes()) != PLAN_SHA256:
        raise ValueError("Changed prospective confirmation source plan")
    plan = json.loads(plan_path.read_text())
    manifest, digest = read_manifest(root)
    if manifest["plan_sha256"] != PLAN_SHA256 or manifest["revision"] != plan["metadata_revision"]:
        raise ValueError("Confirmation manifest is not bound to the public source plan")
    previous_ids, previous_hashes, previous_originals = set(), set(), []
    historical_manifest_sha256 = {}
    for name in HISTORIES:
        directory = historical_root / name
        earlier, old_digest = read_manifest(directory)
        if old_digest != plan["historical_manifest_sha256"][name]:
            raise ValueError("Historical exclusion catalogue changed")
        historical_manifest_sha256[name] = old_digest
        previous_ids.update(i["source_id"] for i in earlier["items"])
        previous_hashes.update(i["image_sha256"] for i in earlier["items"])
        for item in earlier["items"]:
            if item["label"] == 0:
                _, pixels, _ = source(directory, item)
                previous_originals.append(phash(pixels))
    if (len(previous_ids) != 240 or len(previous_originals) != 240
            or set(plan["excluded_coco_ids"]) != previous_ids | WEB_IDS):
        raise ValueError("Every earlier source and historical web example must remain excluded")
    paired_items(manifest, plan, previous_ids | WEB_IDS, previous_hashes)
    by_id = {(i["source_id"], i["label"]): i for i in manifest["items"]}
    old_hashes = np.stack(previous_originals)
    near, geometries, secondary, current_hashes = [], [], [], []
    for source_id in sorted({i["source_id"] for i in manifest["items"]}):
        genuine, edited = by_id[source_id, 0], by_id[source_id, 1]
        original_bytes, original, _ = source(root, genuine)
        edit_bytes, edit, truth = source(root, edited)
        with Image.open(root / genuine["image"]) as a, Image.open(root / edited["image"]) as b:
            if a.getexif().get(274, 1) != 1 or b.getexif().get(274, 1) != 1:
                raise ValueError("Source orientation differs from reference coordinates")
        if original.shape != edit.shape or not np.array_equal(np.any(original != edit, axis=2), truth):
            raise ValueError("Primary reference does not equal the planned original/edit pixel changes")
        current = phash(original)
        current_hashes.append(current)
        distance = int(np.count_nonzero(old_hashes != current, axis=1).min())
        near.append(distance)
        if distance <= 4:
            raise ValueError("Potential historical near duplicate; stop the panel without resampling")
        geometries.append({"source_id": source_id, "width": original.shape[1], "height": original.shape[0],
                           "original_sha256": sha(original_bytes), "edited_sha256": sha(edit_bytes),
                           "primary_mask_sha256": edited["mask_sha256"], "nearest_historical_phash_distance": distance})
        author = edited.get("author_mask")
        if author:
            path = root / author["file"]
            if sha(path.read_bytes()) != author["sha256"]:
                raise ValueError("Changed secondary publisher mask")
            with Image.open(path) as im:
                if im.size != (original.shape[1], original.shape[0]):
                    raise ValueError("Secondary publisher mask geometry differs")
        secondary.append("available" if author else "unavailable")
    current = np.stack(current_hashes)
    distances = np.count_nonzero(current[:, None, :] != current[None, :, :], axis=2)
    np.fill_diagonal(distances, 64)
    within = int(distances.min())
    if within <= 4:
        raise ValueError("Potential within-panel near duplicate; stop without resampling")
    return {"manifest_sha256": digest, "source_plan_sha256": PLAN_SHA256,
            "historical_manifest_sha256": historical_manifest_sha256,
            "source_groups": 48, "source_images": 96, "prior_source_ids": 240, "historical_web_ids": 8,
            "source_or_edit_byte_reuse": 0, "nearest_historical_phash_distance": min(near),
            "nearest_within_panel_phash_distance": within,
            "secondary_publisher_masks_available": secondary.count("available"),
            "secondary_publisher_masks_unavailable": secondary.count("unavailable"), "sources": geometries}


def freeze(root: Path, historical_root: Path, development: Path, output: Path) -> dict:
    review_path = development / "development-review.json"
    predictions = development / "development/predictions.jsonl"
    review = json.loads(review_path.read_text())
    if (sha(review_path.read_bytes()) != DEVELOPMENT_REVIEW_SHA256
            or sha(predictions.read_bytes()) != DEVELOPMENT_PREDICTIONS_SHA256
            or not all(r["development_count_tradeoff"] for r in review["conditions"].values())
            or review["may_enable_service"] is not False):
        raise ValueError("Changed development result, gates or eligibility")
    audit = source_audit(root, historical_root)
    outcome = {"schema": SCHEMA, "stage": "prospective_fresh_confirmation", "implementation_sha256": code_pins(),
               "runtime": runtime(), "weights": weights(), "source_audit": audit,
               "development_review_sha256": DEVELOPMENT_REVIEW_SHA256,
               "development_predictions_sha256": DEVELOPMENT_PREDICTIONS_SHA256,
               "rules": review["rules"], "planned_inputs": 192, "conditions": ["native", "jpeg75"],
               "acceptance": {"more_overlapping_edits": True, "higher_mean_edit_iou": True,
                              "maximum_additional_genuine_highlights_per_format": 1,
                              "no_increased_unassessed": True, "required_crop_failures": 0},
               "fresh_detector_forwards_started": False, "may_enable_service": False,
               "scope": "Source-disjoint TGIF/TGIF2 localization component comparison; no overall confidence weight. "
                        "Formats share48 sources. RGB-change references do not independently prove semantic edit boundaries."}
    with output.open("x") as f:
        json.dump(outcome, f, indent=2, allow_nan=False)
        f.write("\n")
    return outcome


def check(lock_path: Path, root: Path, freeze_sha256: str) -> tuple[dict, dict]:
    if sha(lock_path.read_bytes()) != freeze_sha256:
        raise ValueError("Confirmation freeze differs from its published commitment")
    lock = json.loads(lock_path.read_text())
    manifest, digest = read_manifest(root)
    if (lock.get("schema") != SCHEMA or lock["implementation_sha256"] != code_pins()
            or lock["runtime"] != runtime() or lock["weights"] != weights()
            or lock["source_audit"]["manifest_sha256"] != digest
            or sha((root / "tgif-plan.json").read_bytes()) != PLAN_SHA256
            or lock["planned_inputs"] != 192 or lock["may_enable_service"] is not False):
        raise ValueError("Changed frozen method, runtime or input manifest")
    return lock, manifest
