# SPDX-License-Identifier: Apache-2.0
"""Audit earlier-copy orientation and resize commitments with optional native replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-reference-oriented-resize-transfer/1"
POSITIVES = {"clean", "local", "generated"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA-256 commitment")


def ltrb(box):
    x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
    return x, y, x+w, y+h


def check_box(box, size):
    require(len(box) == 4 and all(type(k) is int for k in box) and 0 <= box[0] < box[2] <= size[0]
            and 0 <= box[1] < box[3] <= size[1], "Box exceeds source coordinates")


def outcomes(lock, rows):
    counts, transformed, kernels = defaultdict(Counter), set(), set()
    advances = Counter()
    for row in rows:
        expected, actual = row["expected"], row["proof"]
        correct = bool(expected and actual and expected["transform"] == actual["transform"]
                       and expected["kernel"] in actual["kernels"] and tuple(expected["projected_crop"]) == ltrb(actual["projected_crop"])
                       and tuple(expected["native_crop"]) == ltrb(actual["native_crop"]))
        if correct:
            target, reported, overlap = row["counts"]
            correct &= ((not reported and actual["matched_pixels"] == math.prod(row["submitted_size"])) if row["role"] == "clean"
                        else target > 0 and overlap == reported and 2*overlap >= target
                        and 10*overlap >= 3*(reported + target - overlap))
        key = f'{row["domain"]}/{row["role"]}'
        current = row["current"]["status"] == "comparable"
        counts[key].update({"views": 1, "current_comparable": current, "accepted": bool(actual), "correct": correct,
                            "incorrect": bool(actual) and not correct, "added_correct": correct and not current})
        if actual and not correct:
            advances["incorrect"] += 1
        if correct:
            advances[row["role"]] += 1
            advances["added"] += not current
            transformed.add(actual["transform"])
            kernels.add(expected["kernel"])
    limits = lock["criteria"]
    return {"roles": {key: dict(value) for key, value in counts.items()}, "correct_orientations": sorted(transformed),
            "correct_kernels": sorted(kernels), "correct_clean": advances["clean"], "correct_local": advances["local"],
            "correct_generated": advances["generated"], "added_correct": advances["added"], "incorrect": advances["incorrect"],
            "advances": (advances["clean"] >= limits["minimum_clean"] and advances["local"] >= limits["minimum_local"]
                         and advances["generated"] >= limits["minimum_generated"] and advances["added"] >= limits["minimum_improved"]
                         and advances["incorrect"] <= limits["maximum_incorrect"] and len(transformed) >= limits["minimum_orientations"])}


def readback(receipt):
    lock = receipt["freeze"]
    require(receipt["schema"] == lock["schema"] == SCHEMA, "Unexpected study schema")
    freeze = (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode()
    require(digest(freeze) == receipt["freeze_sha256"], "Frozen study commitment differs")
    expected = {(domain, slot, role) for domain, groups in (("photo", 21), ("generated", 7), ("ui", 12))
                for slot in range(groups) for role in lock["roles"][domain]}
    seen_sources = {}
    captures = {(r["slot"], r["variant"]): r["png_sha256"] for r in lock["chromium"]["rows"]}
    for row in receipt["rows"]:
        domain, slot, role = (row[key] for key in ("domain", "slot", "role"))
        require((domain, slot, role) in expected, "Unexpected or repeated case")
        expected.remove((domain, slot, role))
        for field in ("earlier_sha256", "submitted_sha256"):
            check_digest(row[field])
        prev = seen_sources.setdefault((domain, slot), row["earlier_sha256"])
        require(prev == row["earlier_sha256"], "One source group refers to different earlier bytes")
        if domain == "ui":
            require(captures[slot, 0] == row["earlier_sha256"], "Screen control uses a different earlier capture")
        native, submitted = row["earlier_size"], row["submitted_size"]
        truth, proof = row["expected"], row["proof"]
        require(bool(truth) == (role in POSITIVES), "Wrong truth status")
        if truth:
            transform = lock["transforms"][slot % len(lock["transforms"])]
            require(truth["transform"] == transform and truth["kernel"] == lock["kernels"][slot % len(lock["kernels"])],
                    "Authored delivery differs from assigned orientation and kernel")
            check_box(truth["native_crop"], native)
            check_digest(truth["truth_sha256"])
            target, reported, overlap = row["counts"]
            require(0 <= overlap <= min(reported, target) <= max(reported, target) <= math.prod(submitted)
                    and (target == 0) == (role == "clean") and (proof or reported == 0), "Impossible native overlap")
        if proof:
            require(proof["transform"] in lock["transforms"] and proof["kernels"] and set(proof["kernels"]) <= set(lock["kernels"]),
                    "Invalid transform or kernels")
            check_box(ltrb(proof["native_crop"]), native)
            for field in ("projected_rgb_sha256", "difference_mask_sha256"):
                check_digest(proof[field])
            check_digest(row["selected_mask_sha256"])
            check_tiles(proof, submitted, lock)
            for region in proof["regions"]:
                check_box(ltrb(region), submitted)
                check_box(ltrb(region["reference_region"]), native)
        else:
            require(row["selected_mask_sha256"] is None, "Unexpected selected mask commitment")
    require(not expected and len(seen_sources) == 40 and len(set(seen_sources.values())) == 40, "Incomplete or overlapping source groups")
    require(outcomes(lock, receipt["rows"]) == receipt["outcomes"], "Study counts differ")
    return receipt["outcomes"]


def check_tiles(proof, size, lock):
    limits = lock["candidate_requirements"]
    points = [tuple(point) for point in proof["matched_tiles"]]
    rows, cols = {r for r, _ in points}, {c for _, c in points}
    quadrants = {(r > 2, c > 2) for r, c in points if r != 2 and c != 2}
    require(len(set(points)) == len(points) >= limits["min_tiles"] and min(len(rows), len(cols)) >= limits["min_spread"]
            and rows | cols <= set(range(5)) and len(quadrants) == 4
            and limits["min_exact_fraction"] * math.prod(size) <= proof["matched_pixels"] <= math.prod(size),
            "Incomplete native exact pixel support")


def verify_native(receipt, run, commons, generated, ui):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import find_objects, label

    from scripts import image_reference_oriented_resize_transfer as assay

    args = argparse.Namespace(commons=commons, generated=generated, ui=ui)
    lock = receipt["freeze"]
    require(lock == assay.freeze(args), "Frozen source receipts differ")
    files = run / "inputs"
    donor = assay.crop.source(commons, lock["unrelated_donor"], (688, 548))
    inputs = {}
    for domain, slot, source, edited, truth in assay.source_rows(args, lock):
        name = f"{domain}-{slot:02d}"
        require((files / f"{name}-earlier").read_bytes() == source, "Earlier source replay differs")
        with Image.open(files / f"{name}-earlier") as encoded:
            earlier = encoded.convert("RGB")
        for role, image, *_geometry, mask in assay.deliveries(domain, slot, earlier, edited, truth, donor):
            data = assay.prior.encode(image, "JPEG" if role == "jpeg" else "PNG")
            require((files / f"{name}-{role}").read_bytes() == data, "Prepared delivery replay differs")
            inputs[domain, slot, role] = mask
    crop_records, confirmed = [], 0
    for row in receipt["rows"]:
        domain, slot, role = (row[key] for key in ("domain", "slot", "role"))
        stem = f"{domain}-{slot:02d}"
        earlier, after = files / f"{stem}-earlier", files / f"{stem}-{role}"
        require(digest(earlier.read_bytes()) == row["earlier_sha256"] and digest(after.read_bytes()) == row["submitted_sha256"],
                "Encoded input differs from the receipt")
        proof = row["proof"]
        if row["expected"]:
            truth = inputs[domain, slot, role]
            require(digest(truth.tobytes()) == row["expected"]["truth_sha256"] and int(truth.sum()) == row["counts"][0],
                    "Exact changed-pixel truth differs")
        if proof is None:
            continue
        with Image.open(earlier) as raw, Image.open(after) as examined:
            ref, sub = raw.convert("RGB"), np.asarray(examined.convert("RGB"))
        transform = proof["transform"]
        operation = assay.prior.reviewer.ORIENTATIONS[transform]
        box = ltrb(proof["projected_crop"])
        require(assay.native_bounds(box, ref.size, transform) == ltrb(proof["native_crop"]), "Native earlier overlap differs")
        with ref.transpose(operation) as oriented:
            projections = [np.asarray(assay.project(oriented, box, tuple(row["submitted_size"]), method)) for method in proof["kernels"]]
        require(all(digest(item.tobytes()) == proof["projected_rgb_sha256"] for item in projections), "Reproduced RGB projection differs")
        difference = (projections[0] != sub).any(axis=2)
        require(digest(difference.tobytes()) == proof["difference_mask_sha256"] and int((~difference).sum()) == proof["matched_pixels"],
                "Exact native RGB difference differs")
        verify_tiles(difference, sub, proof, np)
        labels, components = label(difference, structure=np.ones((3, 3), dtype=bool))
        component_areas = np.bincount(labels.ravel(), minlength=components+1)
        floor = max(48, round(difference.size*.0004))
        qualifying = {index for index in range(1, components+1) if component_areas[index] >= floor}
        indices, selected = set(), np.zeros(difference.shape, dtype=bool)
        for region in proof["regions"]:
            x, y, r, b = ltrb(region)
            component = next((i for i, (rs, cs) in enumerate(find_objects(labels), 1)
                              if i in qualifying - indices and (cs.start, rs.start, cs.stop, rs.stop) == (x, y, r, b)
                              and component_areas[i] == region["selected_pixels"]), None)
            require(component is not None, "Selected region does not match an independently extracted component")
            indices.add(component)
            selected |= labels == component
            w, h = box[2]-box[0], box[3]-box[1]
            sw, sh = row["submitted_size"]
            mapped = (box[0] + math.floor(x*w/sw), box[1] + math.floor(y*h/sh), box[0] + math.ceil(r*w/sw), box[1] + math.ceil(b*h/sh))
            require(assay.native_bounds(mapped, ref.size, transform) == ltrb(region["reference_region"]), "Earlier evidence crop is misregistered")
            earlier_crop = ref.crop(ltrb(region["reference_region"]))
            crop_records.append({"domain": domain, "slot": slot, "role": role, "earlier_rgb_sha256": digest(earlier_crop.tobytes()),
                                 "submitted_rgb_sha256": digest(sub[y:b, x:r].tobytes())})
        require(len(indices) == min(5, len(qualifying)) and len(qualifying-indices) == proof["omitted_regions"]
                and max((component_areas[i] for i in qualifying-indices), default=0) <= min((component_areas[i] for i in indices), default=0),
                "Selected component count or ranking differs")
        require(digest(selected.tobytes()) == row["selected_mask_sha256"], "Exact selected-mask commitment differs")
        if row["expected"]:
            require(row["counts"][1:] == [int(selected.sum()), int((selected & inputs[domain, slot, role]).sum())], "Overlap counts differ")
        confirmed += 1
    return {"native_views": len(receipt["rows"]), "confirmed_projections": confirmed,
            "native_crop_pairs": len(crop_records), "crop_commitment_sha256": digest(json.dumps(crop_records, sort_keys=True).encode())}


def verify_tiles(difference, sub, proof, np):
    height, width = difference.shape
    supported = []
    for row in range(5):
        for col in range(5):
            x, y = round((col+.5)/5*width), round((row+.5)/5*height)
            patch = sub[y-10:y+10, x-10:x+10].astype(np.int16)
            texture = sum(np.any(abs(np.diff(patch, axis=axis)) >= 6, axis=2).mean() for axis in (0, 1)) / 2
            if texture >= .12 and not difference[y-10:y+10, x-10:x+10].any():
                supported.append([row, col])
    require(supported == proof["matched_tiles"], "Exact native tile witnesses differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native", nargs=4, type=Path, metavar=("RUN", "COMMONS", "GENERATED", "UI"))
    arguments = parser.parse_args()
    record = json.loads(arguments.receipt.read_bytes())
    result = readback(record)
    if arguments.native:
        result = {"outcomes": result, "replay": verify_native(record, *arguments.native)}
    print(json.dumps(result, indent=2, sort_keys=True))
