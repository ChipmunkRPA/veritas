"""Compare permitted-source JPEG pixels to publisher originals; no model calls."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from scripts.image_source_rights_v19 import ROOT, require, sha, write

OLD = ROOT / "backend/data/independent-evaluation/image-source-rights-v19"
ARCHIVES = {
    "training": (5652563073, '"db7c6f701d3b5a6c6f375f2fd60a1bd2"'),
    "validation": (859947874, '"d95a1202a7445d79872735f60cbeaa95"'),
    "testing": (806962390, '"fc2934fed45674aa89479640d0030beb"'),
}
CODE = ("backend/scripts/image_pixel_lineage_v20.py", "backend/tests/test_image_pixel_lineage_v20.py", "backend/scripts/image_attribution_runtime_v20.py", "backend/scripts/image_source_rights_v19.py", "backend/scripts/specialist_runtime_v4.py")


def rgb(data):
    with Image.open(io.BytesIO(data)) as image:
        require(image.format in {"JPEG", "PNG"} and 0 < image.width <= 2048 and 0 < image.height <= 2048, "Bounded source JPEG/PNG required")
        require(image.getexif().get(274, 1) == 1, "Unhandled source EXIF orientation")
        return np.asarray(image.convert("RGB")).copy()


def transformed_source(source, coco_size):
    width, height = coco_size
    require(source.ndim == 3 and source.shape[2] == 3 and source.dtype == np.uint8, "Native RGB bytes required")
    rotate = abs(width / height - source.shape[1] / source.shape[0]) > 0.01
    output = np.swapaxes(source, 0, 1)[::-1, :, :] if rotate else source
    return np.ascontiguousarray(output), rotate


def pixel_comparison(source_rgb, published_rgb, coco_size, full_size, crop):
    image, rotated = transformed_source(source_rgb, coco_size)
    require(list(image.shape[1::-1]) == full_size, "Current high-resolution dimensions differ from publisher metadata")
    require(len(crop) == 4 and all(type(n) is int for n in crop), "Integer publisher crop required")
    x, y, w, h = crop
    require(x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= image.shape[1] and y + h <= image.shape[0], "Publisher crop outside source")
    if published_rgb.shape == image.shape:
        target, geometry = image, "full_source"
    elif list(published_rgb.shape[1::-1]) == [w, h]:
        target, geometry = image[y:y + h, x:x + w], "declared_metadata_crop"
    else:
        raise ValueError("Publisher original dimensions match neither full source nor declared crop")
    exact = bool(np.array_equal(target, published_rgb))
    return {
        "status": "exact_source_rgb" if exact else "pixel_mismatch", "geometry": geometry,
        "publisher_rotation_applied": rotated, "source_size": list(source_rgb.shape[1::-1]),
        "compared_size": list(target.shape[1::-1]), "different_rgb_pixels": int(np.any(target != published_rgb, axis=2).sum()),
        "compared_rgb_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "published_rgb_sha256": hashlib.sha256(published_rgb.tobytes()).hexdigest(),
        "training_admitted": False,
    }


def plan(attribution, folder):
    attribution = attribution.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    review = json.loads((attribution / "review.json").read_text())
    require(review["complete_counts_and_source_correspondence_match"] and review["source_requests_reviewed"] == 523, "Complete attribution review required")
    require(review["program_sha256"] == sha(ROOT / "backend/scripts/review_image_attribution_v20.py"), "Attribution review implementation changed")
    require(review["summary_sha256"] == sha(attribution / "summary.json"), "Attribution summary changed")
    meta_plan = json.loads((attribution / "plan.json").read_text())
    require(all(sha(ROOT / name) == value for name, value in meta_plan["code_sha256"].items()), "Attribution acquisition code changed")
    original_summary = json.loads((OLD / "audit/summary.json").read_text())
    for name, expected in original_summary["input_sha256"].items():
        require(sha(OLD / "source" / name) == expected, "Original source metadata changed")
    tables = {}
    for role in ARCHIVES:
        tables[role] = {}
        for name in (f"metadata/orig_{role}.csv", f"metadata/ps_{role}.csv", f"metadata/sd2_{role}.csv", f"metadata-flux/flux1filldev_{role}.csv"):
            with (OLD / "source" / name).open(newline="") as stream:
                tables[role][name] = list(csv.DictReader(stream))
    journals = [json.loads(line) for line in (attribution / "responses.jsonl").read_text().splitlines()]
    cases = []
    for response, source_record in zip(journals, meta_plan["items"], strict=True):
        require(response["index"] == source_record["index"] and response["source_id"] == source_record["source"]["source_id"], "Source/attribution record mismatch")
        if response["status"] != "matched_current_provider_metadata":
            continue
        source, identity = source_record["source"], response["attribution"]
        require(sha(attribution / response["body"]) == response["sha256"], "Current attribution body changed")
        role, identifier = source["publisher_role"], str(source["source_id"])
        original = [r for r in tables[role][f"metadata/orig_{role}.csv"] if r["image_id"] == identifier]
        common = {r["category"] for r in original}
        for stem in ("metadata/ps", "metadata/sd2", "metadata-flux/flux1filldev"):
            common &= {r["category"] for r in tables[role][f"{stem}_{role}.csv"] if r["image_id"] == identifier and r["mask_variation"] == "bbox" and r["gen_variation_id"] == "0"}
        require(common, "No shared base/category/bbox source for all three generators")
        category = min(common, key=lambda c: hashlib.sha256(f"image-lineage-v20/{identifier}/{c}".encode()).hexdigest())
        rows = [r for r in original if r["category"] == category]
        require(len(rows) == 1, "Unique source/category original metadata required")
        row = rows[0]
        crop = ast.literal_eval(row["bbox_1024"])
        require(isinstance(crop, list) and len(crop) == 4 and all(type(n) is int for n in crop), "Literal integer source crop required")
        require("/" not in category and category not in {".", ".."}, "Safe exact publisher category required")
        cases.append({
            "index": response["index"], "source_id": source["source_id"], "publisher_role": role,
            "jpeg_url": identity["expected_large_source_url"], "license_id": identity["license_id"],
            "attribution_body_sha256": response["sha256"], "coco_size": [source["coco_image"]["width"], source["coco_image"]["height"]],
            "publisher_full_size": [int(row["width"]), int(row["height"])], "publisher_crop": crop,
            "archive_member": f"{role}/{category}/{identifier}_orig.png",
        })
    require(len(cases) == review["matched_source_attributions_reconstructed"] == 465, "Complete465 matched-metadata sources required")
    protocol = {
        "schema": "chipmunk.image-pixel-lineage-plan/20", "attribution_folder": str(attribution.relative_to(ROOT)),
        "attribution_plan_sha256": sha(attribution / "plan.json"), "attribution_journal_sha256": sha(attribution / "responses.jsonl"), "attribution_review_sha256": sha(attribution / "review.json"),
        "code_sha256": {name: sha(ROOT / name) for name in CODE}, "cases": cases,
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "requests")},
        "archives": {role: {"url": f"https://cloud.ilabt.imec.be/public.php/dav/files/xEeAzrY7ES9KA8o/orig/orig_{role}.tar.gz", "bytes": size, "etag": etag} for role, (size, etag) in ARCHIVES.items()},
        "policy": "All465 current exact-license/photo matches once, no replacement or implicit retry. One hash-selected shared category/photo. Stream publisher originals one role at a time; retain selected PNGs only. Source JPEG limit5MB; PNG30MB; keep failures. Decode native RGB without EXIF orientation; use only inspected publisher aspect-ratio rotation and full source or exact metadata crop. Require exact every-pixel match, not visual similarity. No model, camera-authenticity or complete training-rights claim.",
    }
    write(folder / "plan.json", protocol)
    print(json.dumps({"planned_sources": len(cases), "publisher_roles": dict(Counter(c["publisher_role"] for c in cases)), "plan_sha256": sha(folder / "plan.json")}), flush=True)


def run(folder, role):
    protocol = json.loads((folder / "plan.json").read_text())
    require(all(sha(ROOT / name) == value for name, value in protocol["code_sha256"].items()), "Planned source/geometry code changed")
    require(all(importlib.metadata.version(name) == version for name, version in protocol["runtime_versions"].items()), "Planned decoder/runtime version changed")
    cases = [c for c in protocol["cases"] if c["publisher_role"] == role]
    require(cases, "Nonempty publisher role required")
    output = folder / role
    output.mkdir(exist_ok=False)
    (output / "objects").mkdir()
    archive_info = protocol["archives"][role]
    wanted = {c["archive_member"] for c in cases}
    require(len(wanted) == len(cases), "One original per selected source required")
    originals = {}
    started = time.monotonic()
    with (output / "originals.jsonl").open("x") as log:
        with requests.get(archive_info["url"], headers={"X-Requested-With": "XMLHttpRequest"}, stream=True, timeout=(15, 90)) as response:
            response.raise_for_status()
            observed = {"content_length": response.headers.get("Content-Length"), "etag": response.headers.get("ETag"), "url": response.url}
            write(output / "archive-response.json", observed)
            require(int(observed["content_length"]) == archive_info["bytes"] and observed["etag"] == archive_info["etag"], "Publisher archive object differs from observed plan")
            with tarfile.open(fileobj=response.raw, mode="r|gz", bufsize=1024 * 1024) as archive:
                for member in archive:
                    if member.name not in wanted:
                        continue
                    require(member.name not in originals and member.isfile() and 0 < member.size <= 30_000_000, "Unique bounded regular publisher PNG required")
                    with archive.extractfile(member) as stream:
                        data = stream.read(30_000_001)
                    require(len(data) == member.size, "Incomplete publisher member")
                    rgb(data)
                    digest = hashlib.sha256(data).hexdigest()
                    name = f"objects/{digest}.png"
                    path = output / name
                    if not path.exists():
                        path.write_bytes(data)
                    require(sha(path) == digest, "Retained publisher object changed")
                    entry = {"member": member.name, "file": name, "sha256": digest, "bytes": len(data)}
                    originals[member.name] = entry
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                    if len(originals) % 50 == 0:
                        print(json.dumps({"phase": "publisher_originals", "role": role, "retained": len(originals), "planned": len(cases)}), flush=True)
                    if len(originals) == len(wanted):
                        break
    write(output / "archive-members-completed.json", {"planned": len(wanted), "retained": len(originals), "missing_members": sorted(wanted - originals.keys()), "duration_s": time.monotonic() - started, "archive_full_sha256_or_gzip_tail_verified": False, "scope": "Selected members decoded and hashed; stream may stop when all are found. Archive ETag/size are observational object bindings, not a publisher-signed digest."})
    previous = 0.0
    with (output / "comparisons.jsonl").open("x") as log:
        for case in cases:
            entry = {"index": case["index"], "source_id": case["source_id"], "publisher_role": role, "status": "unresolved", "training_admitted": False}
            try:
                require(case["archive_member"] in originals, "Publisher original member unavailable")
                entry["original"] = originals[case["archive_member"]]
                time.sleep(max(0, 0.25 - (time.monotonic() - previous)))
                previous = time.monotonic()
                with requests.get(case["jpeg_url"], stream=True, timeout=(10, 20), allow_redirects=False) as response:
                    entry["jpeg_http_status"] = response.status_code
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_content(65536):
                        data.extend(chunk)
                        require(len(data) <= 5_000_000, "Source JPEG exceeds bounded size")
                digest = hashlib.sha256(data).hexdigest()
                name = f"objects/{digest}.jpg"
                path = output / name
                if not path.exists():
                    path.write_bytes(data)
                require(sha(path) == digest, "Source JPEG object changed")
                entry["jpeg"] = {"file": name, "sha256": digest, "bytes": len(data)}
                result = pixel_comparison(rgb(data), rgb((output / entry["original"]["file"]).read_bytes()), case["coco_size"], case["publisher_full_size"], case["publisher_crop"])
                entry.update(result)
            except Exception as error:
                entry["error"] = {"type": type(error).__name__, "message": str(error)}
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
    records = [json.loads(line) for line in (output / "comparisons.jsonl").read_text().splitlines()]
    require(len(records) == len(cases), "Complete source denominator required")
    summary = {
        "schema": "chipmunk.image-pixel-lineage-role-summary/20", "publisher_role": role, "planned": len(cases), "completed": len(records),
        "statuses": dict(sorted(Counter(r["status"] for r in records).items())),
        "geometry": dict(sorted(Counter(r.get("geometry", "unresolved") for r in records).items())),
        "rotated_sources": sum(r.get("publisher_rotation_applied", False) for r in records),
        "unresolved_reasons": dict(sorted(Counter(r["error"]["message"] if r["error"]["type"] == "ValueError" else r["error"]["type"] for r in records if "error" in r).items())),
        "plan_sha256": sha(folder / "plan.json"), "comparisons_sha256": sha(output / "comparisons.jsonl"), "originals_sha256": sha(output / "originals.jsonl"),
        "training_admitted": 0, "neural_forwards": 0, "outside_human_review": False,
    }
    write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--attribution", type=Path)
    parser.add_argument("--role", choices=tuple(ARCHIVES))
    args = parser.parse_args()
    if args.command == "plan":
        require(args.attribution is not None, "Attribution input required")
        plan(args.attribution, args.folder)
    else:
        require(args.role is not None, "Explicit publisher role required")
        run(args.folder, args.role)
