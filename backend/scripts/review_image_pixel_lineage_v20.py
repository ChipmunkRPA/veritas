"""Full local source-pixel replay with separate geometry and OpenCV decode."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode(data):
    with Image.open(io.BytesIO(data)) as image:
        need(image.format in {"PNG", "JPEG"} and image.width <= 2048 and image.height <= 2048, "Unsupported source image")
        need(image.getexif().get(274, 1) == 1, "Unhandled source orientation")
        canonical = np.array(image.convert("RGB"))
    separate = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    need(separate is not None and np.array_equal(canonical, separate[:, :, ::-1]), "Pillow/OpenCV native RGB decode differs")
    return canonical


def geometry(source, reference, case):
    cw, ch = case["coco_size"]
    sh, sw, _ = source.shape
    # Separate integer version of the publisher's >0.01 aspect-ratio test.
    rotated = 100 * abs(cw * sh - sw * ch) > ch * sh
    image = np.stack([source[:, x] for x in range(sw - 1, -1, -1)], axis=0) if rotated else source.copy()
    need([image.shape[1], image.shape[0]] == case["publisher_full_size"], "High-resolution dimensions differ")
    crop = case["publisher_crop"]
    need(len(crop) == 4 and all(type(v) is int for v in crop), "Invalid crop metadata")
    left, top, width, height = crop
    need(left >= 0 and top >= 0 and width > 0 and height > 0 and left + width <= image.shape[1] and top + height <= image.shape[0], "Crop outside source")
    if image.shape == reference.shape:
        target, kind = image, "full_source"
    else:
        need([reference.shape[1], reference.shape[0]] == [width, height], "Neither full source nor metadata crop")
        target = np.stack([image[y, left:left + width] for y in range(top, top + height)])
        kind = "declared_metadata_crop"
    differences = int(np.count_nonzero(np.logical_or.reduce(target != reference, axis=2)))
    return {
        "status": "exact_source_rgb" if differences == 0 else "pixel_mismatch", "geometry": kind,
        "publisher_rotation_applied": rotated, "source_size": [sw, sh],
        "compared_size": [target.shape[1], target.shape[0]], "different_rgb_pixels": differences,
        "compared_rgb_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
        "published_rgb_sha256": hashlib.sha256(reference.tobytes()).hexdigest(), "training_admitted": False,
    }


def review(folder, role, output):
    protocol = json.loads((folder / "plan.json").read_text())
    root = Path(__file__).resolve().parents[2]
    need(all(sha(root / name) == expected for name, expected in protocol["code_sha256"].items()), "Planned source-pixel code changed")
    inputs = [c for c in protocol["cases"] if c["publisher_role"] == role]
    metadata_folder = root / protocol["attribution_folder"]
    for name, expected in (("plan.json", protocol["attribution_plan_sha256"]), ("responses.jsonl", protocol["attribution_journal_sha256"]), ("review.json", protocol["attribution_review_sha256"])):
        need(sha(metadata_folder / name) == expected, "Attribution source commitment differs")
    metadata_plan = json.loads((metadata_folder / "plan.json").read_text())
    attribution = [json.loads(line) for line in (metadata_folder / "responses.jsonl").read_text().splitlines()]
    eligible = {r["index"]: r for r in attribution if r["status"] == "matched_current_provider_metadata"}
    need({c["index"] for c in protocol["cases"]} == set(eligible) and len(protocol["cases"]) == len(eligible) == 465, "Complete matched-photo source plan differs")
    old = root / "backend/data/independent-evaluation/image-source-rights-v19"
    pinned = json.loads((old / "audit/summary.json").read_text())["input_sha256"]
    tables = {}
    for stem in ("metadata/orig", "metadata/ps", "metadata/sd2", "metadata-flux/flux1filldev"):
        name = stem + "_" + role + ".csv"
        need(sha(old / "source" / name) == pinned[name], "Pinned publisher source/crop metadata differs")
        with (old / "source" / name).open(newline="") as stream:
            tables[stem] = list(csv.DictReader(stream))
    for case in inputs:
        source = metadata_plan["items"][case["index"]]["source"]
        provider = eligible[case["index"]]
        need(source["publisher_role"] == role and source["source_id"] == case["source_id"], "Source role differs")
        need(case["jpeg_url"] == provider["attribution"]["expected_large_source_url"] and case["attribution_body_sha256"] == provider["sha256"], "Provider pixel URL/source binding differs")
        need(case["coco_size"] == [source["coco_image"]["width"], source["coco_image"]["height"]], "COCO geometry differs")
        rows_for_id = [r for r in tables["metadata/orig"] if int(r["image_id"]) == case["source_id"]]
        categories = set(r["category"] for r in rows_for_id)
        for stem in ("metadata/ps", "metadata/sd2", "metadata-flux/flux1filldev"):
            categories.intersection_update(r["category"] for r in tables[stem] if int(r["image_id"]) == case["source_id"] and r["mask_variation"] == "bbox" and r["gen_variation_id"] == "0")
        need(categories, "Shared generator/category source missing")
        category = sorted(categories, key=lambda c: hashlib.sha256(f"image-lineage-v20/{case['source_id']}/{c}".encode()).hexdigest())[0]
        records = [r for r in rows_for_id if r["category"] == category]
        need(len(records) == 1, "Ambiguous publisher original")
        record = records[0]
        need(case["publisher_full_size"] == [int(record["width"]), int(record["height"])] and case["publisher_crop"] == ast.literal_eval(record["bbox_1024"]), "Source dimensions/crop differ from publisher record")
        need(case["archive_member"] == f"{role}/{category}/{case['source_id']}_orig.png", "Source/archive member correspondence differs")
    work = folder / role
    summary = json.loads((work / "summary.json").read_text())
    rows = [json.loads(line) for line in (work / "comparisons.jsonl").read_text().splitlines()]
    originals = [json.loads(line) for line in (work / "originals.jsonl").read_text().splitlines()]
    source_members = {r["member"]: r for r in originals}
    need(len(source_members) == len(originals), "Duplicate original archive member")
    need(summary["planned"] == summary["completed"] == len(rows) == len(inputs), "Incomplete role denominator")
    need(summary["plan_sha256"] == sha(folder / "plan.json") and summary["comparisons_sha256"] == sha(work / "comparisons.jsonl") and summary["originals_sha256"] == sha(work / "originals.jsonl"), "Input/journal commitments differ")
    errors, replayed, image_decodes, metrics, refusals = [], 0, 0, [], 0
    for case, row in zip(inputs, rows, strict=True):
        try:
            need(row["source_id"] == case["source_id"] and row["index"] == case["index"] and row["publisher_role"] == role, "Case identity differs")
            for key in ("original", "jpeg"):
                if key in row:
                    record = row[key]
                    relative = Path(record["file"])
                    need(relative.parts[0] == "objects" and len(relative.parts) == 2 and relative.name == record["sha256"] + (".png" if key == "original" else ".jpg"), "Unexpected source object path")
                    need(sha(work / relative) == record["sha256"] and (work / relative).stat().st_size == record["bytes"], "Source bytes differ")
            need(row.get("original") == source_members.get(case["archive_member"]), "Original member binding differs")
            if "original" in row and "jpeg" in row:
                try:
                    source = decode((work / row["jpeg"]["file"]).read_bytes())
                    image_decodes += 1
                    published = decode((work / row["original"]["file"]).read_bytes())
                    image_decodes += 1
                    result = geometry(source, published, case)
                except ValueError as error:
                    if str(error) == "Pillow/OpenCV native RGB decode differs":
                        raise
                    need(row["status"] == "unresolved" and "error" in row, "Independent decoder/geometry refusal differs")
                    need(row["error"]["type"] == "ValueError", "Unexpected original source refusal")
                    refusals += 1
                    continue
                need(all(row.get(k) == v for k, v in result.items()), "Full independent geometry/pixel result differs")
                metrics.append(result)
                replayed += 1
            else:
                need(row["status"] == "unresolved" and "error" in row, "Unavailable source was not retained as unresolved")
        except Exception as error:
            errors.append({"index": case["index"], "type": type(error).__name__, "message": str(error)})
    statuses = dict(sorted(Counter(r["status"] for r in rows).items()))
    need(statuses == summary["statuses"], "Status counts differ")
    need(dict(sorted(Counter(r.get("geometry", "unresolved") for r in rows).items())) == summary["geometry"], "Geometry counts differ")
    need(sum(r.get("publisher_rotation_applied", False) for r in rows) == summary["rotated_sources"], "Rotation counts differ")
    reasons = dict(sorted(Counter(r["error"]["message"] if r["error"]["type"] == "ValueError" else r["error"]["type"] for r in rows if "error" in r).items()))
    need(reasons == summary["unresolved_reasons"], "Unresolved counts differ")
    need(summary["training_admitted"] == summary["neural_forwards"] == 0, "No training/model inference permitted by this audit")
    receipt = {
        "schema": "chipmunk.image-pixel-lineage-review/20", "publisher_role": role,
        "cases_reviewed": len(inputs), "full_pixel_geometry_replays": replayed,
        "native_pillow_opencv_decode_comparisons": image_decodes,
        "unresolved_source_refusals_reproduced": refusals,
        "exact_source_rgb_replays": sum(m["status"] == "exact_source_rgb" for m in metrics),
        "passed": not errors, "failed_case_count": len(errors), "errors": errors,
        "training_admitted": 0, "independent_remote_archive_reacquisition": False,
        "outside_human_review": False, "program_sha256": sha(__file__), "summary_sha256": sha(work / "summary.json"),
        "scope": "Retained publisher/JPEG bytes, dual local decoders and separately implemented geometry/counts; not independently authenticated camera content or complete rights clearance.",
    }
    with output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "errors"}), flush=True)
    if errors:
        raise ValueError("Full source-pixel review failed; original receipt retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("role", choices=("training", "validation", "testing"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.folder, args.role, args.output)
