# SPDX-License-Identifier: Apache-2.0
"""Evaluate exact earlier-copy orientation on pinned, licensed photo sources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.detectors.image import reference_png_orientation as orientation
from scripts import review_image_reference_export as reviewer

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/evaluation-panels/image-affine-native-scale-2026-11-11.json"
SCHEMA = "chipmunk.image-reference-orientation-transfer/1"
ROLES = ("unchanged", "local_edit", "small_angle", "whole_image_tone", "unrelated", "jpeg_delivery")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, content):
    with path.open("x") as stream:
        json.dump(content, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def binding(archive):
    published = json.loads(SOURCE.read_bytes())
    manifest = json.loads((archive / "plan.json").read_bytes())
    intake = json.loads((archive / "acquisition.json").read_bytes())
    require(published["source_roster"] == manifest and published["source_intake"] == intake,
            "Source roster differs from the licensed published panel")
    photos = []
    for item, acquired in zip(manifest["sources"], intake["attempts"], strict=True):
        slot, filename = acquired["slot"], acquired["name"]
        require(item["slot"] == slot == len(photos) and acquired["status"] == "qualified"
                and filename == f"source-{slot:02d}.jpg" and sha((archive / filename).read_bytes()) == acquired["sha256"],
                "Photo identity or publisher bytes differ")
        photos.append({"slot": slot, "sha256": acquired["sha256"], "file": filename})
    require(len(photos) == 38, "Expected all 38 licensed photo sources")
    return {
        "schema": SCHEMA, "public_source_receipt": sha(SOURCE.read_bytes()), "sources": photos,
        "code": {path.name: sha(path.read_bytes()) for path in (
            Path(__file__), Path(orientation.__file__), Path(reviewer.__file__),
        )},
        "design": {
            "roles": list(ROLES), "orientations": list(reviewer.ORIENTATIONS),
            "earlier_long_edge": [768, 1536], "earlier_format": ["PNG", "JPEG90"],
            "criteria": "At least 23 of 38 usable unchanged pairs and local edits, at least one correct edit per orientation, zero wrong orientation proofs, and zero orientation proofs on controls.",
        },
    }


def encode(image, carrier):
    output = io.BytesIO()
    if carrier == "JPEG":
        image.save(output, format=carrier, quality=90, subsampling=2)
    else:
        image.save(output, format=carrier, compress_level=1)
    return output.getvalue()


def earlier_source(archive, item):
    slot = item["slot"]
    with Image.open(archive / item["file"]) as raw, ImageOps.exif_transpose(raw) as exif:
        image = exif.convert("RGB")
    with image:
        side = 768 if slot % 2 == 0 else 1536
        image.thumbnail((side, side), Image.Resampling.LANCZOS)
        return encode(image, "PNG" if slot // 2 % 2 == 0 else "JPEG")


def input_views(reference, donor, transform):
    op = reviewer.ORIENTATIONS[transform]
    with reference.transpose(op) as projected:
        original = np.asarray(projected)
        height, width = original.shape[:2]
        y, x = np.ogrid[:height, :width]
        truth = ((x - width * .42) / (width * .09)) ** 2 + ((y - height * .58) / (height * .08)) ** 2 < 1
        changed = original.copy()
        changed[truth] ^= np.array([41, 27, 57], dtype=np.uint8)
        with (Image.fromarray(changed) as local, projected.rotate(1.3, resample=Image.Resampling.BICUBIC) as angled,
              projected.point([(value * 3 + 11) // 4 for value in range(256)] * 3) as toned,
              ImageOps.fit(donor, projected.size, Image.Resampling.LANCZOS) as unrelated):
            for role, image in zip(ROLES, (projected, local, angled, toned, unrelated, projected), strict=True):
                yield role, encode(image, "JPEG" if role == "jpeg_delivery" else "PNG"), truth if role == "local_edit" else None


def run(args, lock, verify):
    photos = lock["sources"]
    measurements = []
    projections = tuple(reviewer.ORIENTATIONS)
    for item in photos:
        slot = item["slot"]
        transform = projections[slot % len(projections)]
        before = earlier_source(args.archive, item)
        with (Image.open(io.BytesIO(before)) as ref,
              Image.open(io.BytesIO(earlier_source(args.archive, photos[(slot + 17) % len(photos)]))) as donor):
            for role, after, truth in input_views(ref, donor, transform):
                with Image.open(io.BytesIO(after)) as sub:
                    record, mask = orientation.compare(np.asarray(ref), np.asarray(sub), sub)
                    matched, correct = record is not None, False
                    if matched:
                        proof = record["exact_orientation"]
                        correct = proof["transform"] == transform and role in {"unchanged", "local_edit"}
                        if role == "unchanged":
                            correct &= not record["regions"] and proof["matched_pixels"] == sub.width * sub.height
                        elif role == "local_edit":
                            correct &= np.array_equal(mask, truth) and len(record["regions"]) == 1
                        if verify and correct:
                            record["coordinate_space"] = "exif-oriented-submitted"
                            mismatch, inverse = reviewer._orientation_replay(record, ref, sub, sub)
                            with mismatch:
                                require(np.array_equal(np.asarray(mismatch) != 0, mask), "Independent exact mask differs")
                            for region in record["regions"]:
                                box = reviewer._bounds(region, sub.size)
                                earlier = reviewer._bounds(region["reference_region"], ref.size)
                                require(reviewer._orientation_bounds(box, sub.size, inverse) == earlier,
                                        "Independent earlier crop differs")
                    measurements.append({"slot": slot, "role": role, "transform": transform, "accepted": matched,
                                         "correct": bool(correct), "earlier_sha256": sha(before), "submitted_sha256": sha(after)})
    return measurements


def aggregate(rows, freeze_sha256):
    counts = {role: dict(Counter(planned=0, accepted=0, correct=0)) for role in ROLES}
    local = Counter()
    for row in rows:
        cell = counts[row["role"]]
        cell["planned"] += 1
        cell["accepted"] += row["accepted"]
        cell["correct"] += row["correct"]
        if row["role"] == "local_edit" and row["correct"]:
            local[row["transform"]] += 1
    incorrect = sum(row["accepted"] and not row["correct"] for row in rows)
    return {"schema": SCHEMA, "freeze_sha256": freeze_sha256, "outcomes": counts, "correct_local_edits_by_orientation": dict(local),
            "incorrect_proofs": incorrect,
            "advances": (not incorrect and min(counts[role]["correct"] for role in ROLES[:2]) >= 23
                         and local.keys() == reviewer.ORIENTATIONS.keys())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "verify"))
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    lock = binding(args.archive)
    frozen, ledger, result = (args.output / name for name in ("freeze.json", "measurements.json", "aggregate.json"))
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(frozen, lock)
        return
    require(json.loads(frozen.read_bytes()) == lock, "Frozen photo study changed")
    verified = args.phase == "verify"
    rows = run(args, lock, verified)
    outcome = aggregate(rows, sha(frozen.read_bytes()))
    if not verified:
        write(ledger, rows)
        write(result, outcome)
    else:
        require(json.loads(ledger.read_bytes()) == rows and json.loads(result.read_bytes()) == outcome,
                "Transfer outcomes differ in the independent replay")
        write(args.output / "audit.json", {"verified_source_files": len(lock["sources"]), "verified_views": len(rows),
                    "reconstructed_orientation_proofs": sum(row["correct"] for row in rows),
                    "result_sha256": sha(result.read_bytes())})
    print(json.dumps(outcome, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
