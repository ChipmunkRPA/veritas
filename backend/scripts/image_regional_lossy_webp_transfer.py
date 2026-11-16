# SPDX-License-Identifier: MIT
"""Compare source-aligned regional responses on matched PNG and lossy WebP images."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-regional-lossy-webp-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
BANKS = ("tgif-confirmation-v4", "tgif-confirmation-v5")
FAMILIES = ("firefly", "flux-fill", "sd2")
FORMATS = ("png", "webp68", "webp90")
ROLES = ("original", "local_tone", "local_blur", "edited")
METHOD = {
    "sources": "Four unique source photos per edit family per retained publisher bank, selected by identity hash. "
               "Keep original dimensions, 512 to 1024 pixels on each side, and edits covering 0.5 to 30 percent "
               "of exact source pixels. Freeze all eligible inputs before inference.",
    "controls": "Untouched photographs and conventional local tone and blur applied within the exact edit footprint.",
    "delivery": "Encode identical pre-delivery pixels as PNG, or opaque simple VP8 WebP at quality 68 or 90, effort 4.",
    "native": "Existing pinned regional localizer with its 0.90 response point, 1 percent active-area rule, "
              "reliability mask computed on the submitted pixels, and two-region limit.",
    "recompression_support": "For a highlighted WebP, decode the submitted pixels and re-encode them as JPEG75. "
                             "Keep each original component with at least half of its cells retained by the same model. "
                             "The two views use the original submitted image's reliability mask.",
    "location": "At least 30 percent IoU and 50 percent recovery of the exact pre-encode changed-pixel footprint.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def save(path, data):
    raw = json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw + b"\n")


def _rank(row):
    return sha(f"{SCHEMA}/{row['source_id']}".encode())


def inputs(root, *, banks=BANKS, per_family=4):
    import cv2
    import numpy as np
    from PIL import Image

    seen = set()
    for bank in banks:
        directory = root / bank
        manifest = read(directory / "manifest.json")
        original = {entry["source_id"]: entry for entry in manifest["items"] if entry["label"] == 0}
        for family in FAMILIES:
            pool = sorted((entry for entry in manifest["items"] if entry["label"] == 1 and entry["family"] == family), key=_rank)
            count = 0
            for edit in pool:
                source_id = edit["source_id"]
                if source_id in seen:
                    continue
                rows = original[source_id], edit
                photos = []
                for row in rows:
                    raw = (directory / row["image"]).read_bytes()
                    assert sha(raw) == row["image_sha256"]
                    with Image.open(io.BytesIO(raw)) as picture:
                        assert picture.format == "PNG" and picture.mode == "RGB"
                        photos.append(np.asarray(picture))
                base, changed = photos
                assert base.shape == changed.shape
                h, w = base.shape[:2]
                if not 512 <= min(w, h) <= max(w, h) <= 1024:
                    continue
                label = np.any(base != changed, axis=2)
                if not .005 <= float(label.mean()) <= .30:
                    continue
                pin = file_sha(directory / edit["mask"])
                assert edit["mask_sha256"] == pin
                with Image.open(directory / edit["mask"]) as supplied:
                    assert np.array_equal(np.asarray(supplied.convert("L")) > 0, label)
                shaped = label[:, :, None]
                tone = np.clip(np.rint(base.astype(np.float32) * .86 + 11), 0, 255).astype(np.uint8)
                blur = cv2.GaussianBlur(base, (0, 0), 1.5)
                versions = (base, np.where(shaped, tone, base), np.where(shaped, blur, base), changed)
                scene = {"bank": bank, "family": family, "source_id": source_id, "original_sha256": rows[0]["image_sha256"],
                         "edit_sha256": edit["image_sha256"], "publisher_mask_sha256": pin,
                         "size": [w, h], "truth_sha256": sha(label.tobytes()), "true_pixels": int(label.sum())}
                ordinary = np.zeros_like(label)
                yield scene, dict(zip(ROLES, zip(versions, (ordinary, ordinary, ordinary, label), strict=True), strict=True))
                count += 1
                seen.add(source_id)
                if count == per_family:
                    break
            assert count == per_family, f"Not enough qualified inputs in {bank}/{family}"


def encode(pixels, form):
    from PIL import Image

    stream = io.BytesIO()
    image = Image.fromarray(pixels)
    if form == "png":
        image.save(stream, format="PNG")
    elif form == "jpeg75":
        image.save(stream, format="JPEG", quality=75, subsampling="4:2:0", optimize=False, progressive=False)
    else:
        image.save(stream, format="WEBP", quality=int(form[4:]), method=4, lossless=False)
    return stream.getvalue()


def deliveries(root):
    for source, views in inputs(root):
        for role, (pixels, truth) in views.items():
            for form in FORMATS:
                body = encode(pixels, form)
                yield source | {"role": role, "format": form, "input_sha256": sha(body)}, body, truth


def _bounds(mask, truth):
    import numpy as np

    return {"pixels": int(mask.sum()), "overlap": int(np.count_nonzero(mask & truth)),
            "sha256": sha(mask.tobytes())}


def useful(reading, target):
    shared = reading["overlap"]
    return bool(target and 2 * shared >= target and 10 * shared >= 3 * (reading["pixels"] + target - shared))


def summarize(rows):
    tallies = defaultdict(Counter)
    flagged_groups = defaultdict(set)
    useful_families = defaultdict(set)
    for row in rows:
        positive = row["role"] == "edited"
        for name, measurement in row["measurements"].items():
            key = row["format"], name
            record = tallies[key]
            label = "edited" if positive else "ordinary"
            record[f"{label}_files"] += 1
            mark = bool(measurement["pixels"])
            record[f"{label}_flags"] += mark
            correct = useful(measurement, row["true_pixels"] if positive else 0)
            record["useful_locations"] += correct
            record["misplaced_locations"] += positive and mark and not correct
            record["unassessed"] += not row["assessed"]
            if mark and not positive:
                flagged_groups[key].add(row["source_id"])
            if correct:
                useful_families[key].add(row["family"])
    return {"/".join(key): dict(counts, flagged_ordinary_groups=len(flagged_groups[key]),
                               useful_families=sorted(useful_families[key]))
            for key, counts in sorted(tallies.items())}


def validate(receipt):
    freeze, rows = receipt["freeze"], receipt["rows"]
    assert freeze["schema"] == SCHEMA and freeze["method"] == METHOD
    assert len(rows) == len(freeze["views"]) == 288
    assert len({row["source_id"] for row in rows}) == 24
    for case, row in zip(freeze["views"], rows, strict=True):
        assert all(row[key] == value for key, value in case.items())
        truth = row["true_pixels"] if row["role"] == "edited" else 0
        w, h = row["size"]
        assert row["assessed"]
        for field in row["measurements"].values():
            assert 0 <= field["overlap"] <= min(truth, field["pixels"]) <= max(truth, field["pixels"]) <= w * h
    result = summarize(rows)
    assert result == receipt["summary"]
    return result


def score(args):
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import regional, regional_support
    from app.detectors.image.reference_lossy_webp import webp_size
    from scripts.image_regional_jpeg_transfer import project, retained_components

    args.output.mkdir(mode=0o700)
    freeze = {"schema": SCHEMA, "method": METHOD, "checkpoint_sha256": regional.CHECKPOINT_SHA256,
              "manifests": {name: file_sha(args.sources / name / "manifest.json") for name in BANKS},
              "views": [row for row, _, _ in deliveries(args.sources)]}
    assert len(freeze["views"]) == 288
    save(args.output / "freeze.json", freeze)
    print("Froze 288 deliveries from 24 source images.", flush=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = regional.load_model(args.checkpoint)
    rows = []
    with (args.output / "measurements.jsonl").open("x", buffering=1) as journal:
        for index, (case, body, truth) in enumerate(deliveries(args.sources), 1):
            assert case == freeze["views"][index - 1] and sha(body) == case["input_sha256"]
            image = Image.open(io.BytesIO(body)).convert("RGB")
            pixels = np.asarray(image)
            if case["format"].startswith("webp"):
                assert webp_size(body) == image.size
            prediction = model.predict(image)
            native = regional_support.readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                                              regional.PIXEL_THRESHOLD, body, pixels)
            selected = native.source_mask(*image.size)
            reference = {"native": _bounds(selected, truth)}
            repeat_hash = None
            if case["format"] != "png":
                supported = np.zeros_like(selected)
                if native.component_masks:
                    alternate = Image.open(io.BytesIO(encode(pixels, "jpeg75"))).convert("RGB")
                    second = model.predict(alternate)
                    review = regional_support.readout(second.raw_mask, second.candidate_eligibility["eligible"],
                                                       regional.PIXEL_THRESHOLD, body, pixels)
                    agreed, _ = retained_components(native.component_masks, review.selected_grid_mask)
                    supported = project(agreed, image)
                    repeat_hash = second.geometry["input_float32_sha256"]
                reference["recompression_support"] = _bounds(supported, truth)
            row = case | {"assessed": True, "input_rgb_sha256": prediction.geometry["source_rgb_sha256"],
                          "model_map_sha256": sha(prediction.raw_mask.tobytes()), "repeat_hash": repeat_hash,
                          "measurements": reference}
            rows.append(row)
            print(json.dumps(row, sort_keys=True, separators=(",", ":")), file=journal)
            if index % 24 == 0:
                print(f"Compared {index}/288 deliveries.", flush=True)
    receipt = {"freeze": freeze, "rows": rows, "summary": summarize(rows)}
    validate(receipt)
    save(args.output / "receipt.json.gz", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("score")
    for key in ("sources", "checkpoint", "output"):
        run.add_argument(key, type=Path)
    commands.add_parser("review").add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        score(args)
    else:
        print(json.dumps(validate(read(args.receipt)), indent=2, sort_keys=True))
