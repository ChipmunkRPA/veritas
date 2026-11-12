# SPDX-License-Identifier: Apache-2.0
"""Independently reconstruct native crop proofs with Pillow and exact byte comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_image_small_boundary_control_stress import encoded, require, sha  # noqa: E402

POSITIVE = {"unedited_crop", "local_edit"}


def summary(lock, rows):
    cases = {row["case"]: row["transform"] for row in lock["reference_groups"]}
    expected = {(case, role) for case in cases for role in lock["roles"]}
    outcomes = defaultdict(Counter)
    hits = Counter()
    for row in rows:
        case, role = row["case"], row["role"]
        require((case, role) in expected, "Unknown or repeated held view")
        expected.remove((case, role))
        require(row["transform"] == ("mirror_horizontal" if role == "ambiguous" else cases[case]), "Orientation assignment changed")
        result = row["candidate"]
        accepted = result is not None
        correct = False
        if accepted:
            proof = result["proof"]
            x, y, w, h = (proof["projected_crop"][key] for key in ("x", "y", "w", "h"))
            correct = role in POSITIVE and [x, y, x + w, y + h] == row["expected_projected_crop"] and proof["transform"] == row["transform"]
            if role == "unedited_crop":
                correct &= not result["regions"] and proof["matched_pixels"] == w * h
            elif role == "local_edit":
                correct &= (len(result["regions"]) == 1 and row["candidate_mask_sha256"] == row["truth_sha256"]
                            and proof["difference_mask_sha256"] == row["truth_sha256"])
                if correct:
                    hits[row["transform"]] += 1
        old = row["existing"]
        require((old is not None) == (role in POSITIVE), "Current app did not measure the planned pair")
        outcomes[role].update({"views": 1, "accepted": accepted, "correct": correct, "wrong_proofs": accepted and not correct,
                               "existing_comparable": old is not None and old["status"] == "comparable"})
    require(not expected, "Required held view missing")
    wrong = sum(cell["wrong_proofs"] for cell in outcomes.values())
    criterion = lock["criteria"]
    gates = {
        "correct_unchanged": outcomes["unedited_crop"]["correct"] >= criterion["minimum_correct_unchanged"],
        "correct_edits": outcomes["local_edit"]["correct"] >= criterion["minimum_correct_edits"],
        "all_orientations": all(hits[transform] >= criterion["minimum_correct_per_orientation"] for transform in set(cases.values())),
        "wrong_proofs": wrong <= criterion["maximum_incorrect_proofs"],
        "gain_over_current": outcomes["local_edit"]["correct"] - outcomes["local_edit"]["existing_comparable"]
                            >= criterion["minimum_improvement_over_existing"],
    }
    return {"outcomes": outcomes, "localized_by_orientation": hits, "incorrect_proofs": wrong, "criteria": gates, "advances": all(gates.values())}


def offsets(reference, submitted, boxes):
    """Propose offsets from exact first-row matches rather than five sampled RGB pixels."""
    if reference.width < submitted.width or reference.height < submitted.height:
        return []
    stride = reference.width * 3
    width, height = submitted.size
    raw = reference.tobytes()
    votes = Counter()
    for _, _, (x, y, right, _) in boxes:
        pattern = submitted.crop((x, y, right, y + 1)).tobytes()
        position = raw.find(pattern)
        while position >= 0:
            top, left = divmod(position, stride)
            if left % 3 == 0:
                dx, dy = left // 3 - x, top - y
                if 0 <= dx <= reference.width - width and 0 <= dy <= reference.height - height:
                    votes[dx, dy] += 1
            position = raw.find(pattern, position + 1)
    return [(x, y, x + width, y + height) for (x, y), count in votes.items() if count >= 12]


def projections(source, target):
    from PIL import Image
    from review_image_reference_export import ORIENTATIONS, _textured_boxes

    boxes = list(_textured_boxes(target))
    target_bytes = target.tobytes()
    for transform, operation in [(None, None), *ORIENTATIONS.items()]:
        with source.copy() if operation is None else source.transpose(operation) as oriented:
            for crop in offsets(oriented, target, boxes):
                with oriented.crop(crop) as projected:
                    tiles = [[y, x] for y, x, box in boxes if projected.crop(box).tobytes() == target.crop(box).tobytes()]
                    quadrants = {(y > 2, x > 2) for y, x in tiles if y != 2 and x != 2}
                    if (len(tiles) < 12 or len({y for y, _ in tiles}) < 4 or len({x for _, x in tiles}) < 4 or len(quadrants) != 4):
                        continue
                    pixels = projected.tobytes()
                    mask = bytes(pixels[i:i + 3] != target_bytes[i:i + 3] for i in range(0, len(pixels), 3))
                    width, height = target.size
                    with Image.frombytes("L", target.size, mask) as changes:
                        mismatch = changes.getbbox()
                    if 100 * (width * height - sum(mask)) < 65 * width * height:
                        continue
                    if mismatch and 5 * (mismatch[2] - mismatch[0]) * (mismatch[3] - mismatch[1]) > 2 * width * height:
                        continue
                    yield transform, crop, oriented.size, sha(pixels), mask, tiles


def native_audit(run, rows):
    from PIL import Image
    from review_image_reference_export import ORIENTATIONS, _bounds, _orientation_bounds

    checked, exact_tiles, crops = 0, 0, []
    for row in rows:
        earlier_path, submitted_path = (run / "inputs" / row[key] for key in ("earlier_file", "submitted_file"))
        require(sha(earlier_path.read_bytes()) == row["earlier_sha256"] and sha(submitted_path.read_bytes()) == row["submitted_sha256"],
                "Encoded input bytes differ")
        if not row["candidate"]:
            continue
        result = row["candidate"]
        proof = result["proof"]
        with (Image.open(earlier_path) as earlier, Image.open(submitted_path) as submitted,
              earlier.convert("RGB") as source, submitted.convert("RGB") as target):
            winners = list(projections(source, target))
            require(len(winners) == 1, "Orientation and offset are not uniquely proven")
            transform, crop, projected_size, projected_sha, difference, tiles = winners[0]
            require(proof["transform"] == transform and _bounds(proof["projected_crop"], projected_size) == crop
                    and proof["projected_rgb_sha256"] == projected_sha and proof["difference_mask_sha256"] == sha(difference)
                    and proof["matched_pixels"] == target.width * target.height - sum(difference) and proof["matched_tiles"] == tiles,
                    "Exact source proof differs")
            operation = ORIENTATIONS[transform]
            inverse = {Image.Transpose.ROTATE_90: Image.Transpose.ROTATE_270,
                       Image.Transpose.ROTATE_270: Image.Transpose.ROTATE_90}.get(operation, operation)
            require(_bounds(proof["native_crop"], source.size) == _orientation_bounds(crop, projected_size, inverse), "Native source crop differs")
            for region in result["regions"]:
                box = _bounds(region, target.size)
                original = _bounds(region["reference_region"], source.size)
                shifted = tuple(value + crop[i % 2] for i, value in enumerate(box))
                require(original == _orientation_bounds(shifted, projected_size, inverse), "Inverse region differs")
                require(sha(source.crop(original).transpose(operation).tobytes()) ==
                        sha(source.transpose(operation).crop(shifted).tobytes()), "Native source crop pixels differ")
                crops.append({"case": row["case"], "earlier_crop_sha256": sha(source.crop(original).tobytes()),
                              "submitted_crop_sha256": sha(target.crop(box).tobytes())})
            if row["role"] == "local_edit":
                require(sha(difference) == row["truth_sha256"] == row["candidate_mask_sha256"], "Held mask differs from the authored change")
            checked += 1
            exact_tiles += len(tiles)
    return {"verified_source_pairs": len(rows), "unique_native_proofs": checked, "exact_tile_witnesses": exact_tiles, "paired_crop_hashes": crops}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.source.is_file():
        package = json.loads(args.source.read_bytes())
        require(sha(encoded(package["rows"])) == package["audit"]["rows_sha256"]
                and sha(encoded(package["freeze"])) == package["audit"]["freeze_sha256"], "Published commitments differ")
        result = summary(package["freeze"], package["rows"])
        require(result == package["summary"], "Published result differs")
    else:
        lock, rows = (json.loads((args.source / f"{name}.json").read_bytes()) for name in ("freeze", "rows"))
        result = summary(lock, rows)
        audit = native_audit(args.source, rows) | {"rows_sha256": sha(encoded(rows)), "freeze_sha256": sha(encoded(lock))}
        if args.publish:
            with args.publish.open("xb") as stream:
                stream.write(encoded({"schema": lock["schema"], "freeze": lock, "rows": rows, "summary": result, "audit": audit}))
        print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in audit.items()}, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
