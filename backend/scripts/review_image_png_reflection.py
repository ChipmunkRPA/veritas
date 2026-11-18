# SPDX-License-Identifier: Apache-2.0
"""Recount the PNG reflection transfer and optionally replay the source pixels."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.review_image_webp_reflection import EDITED, _delivery, _iou, _validate_pair

SCHEMA = "chipmunk.image-exact-png-reflection/1"
CLEAR = {"no_pair", "unsupported_geometry"}
ABSTAINED = {"unassessed_match_budget", "ambiguous_pairs"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _useful(pairs, overlaps, truth, total, size):
    for pair in pairs:
        _validate_pair(pair, size)
    if truth is None:
        if overlaps or total is not None:
            raise ValueError("Ordinary PNG has an edit overlap")
        return False
    _validate_pair(truth, size)
    if not 0 < total <= truth[1][2] * truth[1][3]:
        raise ValueError("Changed pixels extend beyond the target")
    useful = False
    for (a, b), (iou, pixels) in zip(pairs, overlaps, strict=True):
        recomputed = max(min(_iou(a, left), _iou(b, right)) for left, right in (truth, truth[::-1]))
        if abs(recomputed - iou) > 1e-11 or not 0 <= pixels <= min(total, max(a[2] * a[3], b[2] * b[3])):
            raise ValueError("Stored overlap differs from the PNG rectangles")
        useful |= recomputed >= .5 and 2 * pixels >= total
    return useful


def review(receipt):
    if receipt["schema"] != SCHEMA:
        raise ValueError("Wrong PNG reflection receipt")
    plan, rows = receipt["plan"], receipt["rows"]
    if plan["criteria"] != {"edited_pixel_recall": .5, "min_pair_iou": .5}:
        raise ValueError("The frozen location criteria differ")
    expected = {}
    if len({source["sha256"] for source in plan["sources"]}) != len(plan["sources"]):
        raise ValueError("Repeated source file")
    for index, source in enumerate(plan["sources"]):
        size = plan["dimensions"][index % len(plan["dimensions"])]
        for role in source["roles"]:
            key = source["bank"], source["tag"], role
            if key in expected:
                raise ValueError("Duplicate planned PNG")
            expected[key] = size
    if len(rows) != len(expected) or {(row["bank"], row["tag"], row["role"]) for row in rows} != set(expected):
        raise ValueError("Assessments differ from the frozen PNG roster")

    total = Counter()
    roles = defaultdict(Counter)
    groups = defaultdict(set)
    for row in rows:
        bank, tag, role = row["bank"], row["tag"], row["role"]
        size, truth = row["size"], row["truth"]
        if (size != expected[bank, tag, role] or bool(truth) != (role in EDITED)
                or not all(SHA256.fullmatch(row[k]) for k in ("png_sha256", "rgb_sha256"))
                or (bool(row["mask_sha256"]) != bool(truth))
                or (row["mask_sha256"] and not SHA256.fullmatch(row["mask_sha256"]))):
            raise ValueError("PNG geometry, source commitments or truth labels differ")
        earlier = row["baseline"]
        candidate = row["candidate"]
        found = []
        if candidate["status"] == "supported_pair":
            item, = candidate["pairs"]
            a, b = item["pair"]
            measurement = item["measurement"]
            perimeter = measurement["perimeter_mean_absolute_difference"]
            if (item["transform"] not in {"horizontal_reflection", "vertical_reflection"} or a[2:] != b[2:]
                    or measurement["matching_pixels"] != a[2] * a[3] or len(perimeter) != 4
                    or sum(value >= 3 for value in perimeter) < 3):
                raise ValueError("The exact PNG comparison fails its source witness")
            found = [item["pair"]]
        elif candidate["pairs"] or candidate["status"] not in ABSTAINED | {"no_pair"}:
            raise ValueError("PNG reflection status disagrees with the reported pair")
        coarse, _, exact, affine = earlier["states"]
        added = bool(found and not earlier["pairs"] and coarse in CLEAR and exact == "no_pair" and affine in CLEAR)
        if added != row["added"]:
            raise ValueError("The PNG fallback differs from the recorded selection")
        baseline_useful = _useful(earlier["pairs"], earlier["overlaps"], truth, row["mask_pixels"], size)
        candidate_useful = _useful(found, candidate["overlaps"], truth, row["mask_pixels"], size)
        counts = {"files": 1, "prior_alerts": bool(earlier["pairs"]), "prior_useful": baseline_useful,
                  "added": added, "added_useful": added and candidate_useful,
                  "added_misplaced": bool(truth) and added and not candidate_useful,
                  "added_controls": not truth and added, "abstentions": candidate["status"] in ABSTAINED}
        if counts["added_useful"]:
            groups["added_useful"].add((bank, tag))
        if counts["added_controls"]:
            groups["added_controls"].add((bank, tag))
        total.update(counts)
        roles[role].update(counts)
    return {"totals": dict(total), "roles": {role: dict(counts) for role, counts in roles.items()},
            "added_useful_groups": len(groups["added_useful"]), "added_control_groups": len(groups["added_controls"])}


def native_replay(receipt, photographs, patterns):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    from app.evidence.image_exact_small_copy_source import review_reflection

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    cv2.setNumThreads(1)
    plan = receipt["plan"]
    rows = {(row["bank"], row["tag"], row["role"]): row for row in receipt["rows"]}
    for index, source in enumerate(plan["sources"]):
        folder = photographs if source["bank"] == "photograph" else patterns
        original = (folder / source["path"]).read_bytes()
        if sha(original) != source["sha256"]:
            raise ValueError("An original does not match its source commitment")
        size = plan["dimensions"][index % len(plan["dimensions"])]
        with Image.open(io.BytesIO(original)) as image:
            pixels = np.array(ImageOps.fit(image.convert("RGB"), tuple(size), Image.Resampling.LANCZOS))
        for role in source["roles"]:
            row = rows[source["bank"], source["tag"], role]
            delivered = _delivery(pixels, role, index)
            buffer = io.BytesIO()
            Image.fromarray(delivered).save(buffer, format="PNG", compress_level=3)
            encoded = buffer.getvalue()
            if sha(encoded) != row["png_sha256"] or sha(delivered.tobytes()) != row["rgb_sha256"]:
                raise ValueError("The delivered PNG differs from its recorded bytes")
            if row["truth"]:
                mask = np.any(delivered != pixels, axis=2)
                _, (x, y, w, h) = row["truth"]
                if (int(mask.sum()) != row["mask_pixels"] or int(mask[y:y+h, x:x+w].sum()) != row["mask_pixels"]
                        or sha(mask.tobytes()) != row["mask_sha256"]):
                    raise ValueError("The native changed pixels differ from their recorded geometry")
                candidate = row["candidate"]
                comparisons = ((row["baseline"]["pairs"], row["baseline"]["overlaps"]),
                               ([item["pair"] for item in candidate["pairs"]], candidate["overlaps"]))
                for pairs, overlaps in comparisons:
                    for two_boxes, (_, claimed) in zip(pairs, overlaps, strict=True):
                        measured = max(int(mask[top:top+height, left:left+width].sum())
                                       for left, top, width, height in two_boxes)
                        if measured != claimed:
                            raise ValueError("The native changed-pixel overlap differs from the receipt")
            item, pair, transform = review_reflection(encoded, webp=False)
            candidate = row["candidate"]
            if item["status"] != candidate["status"] or bool(pair) != bool(candidate["pairs"]):
                raise ValueError("Recorded PNG reflection coverage differs from the implementation")
            if pair:
                expected, = candidate["pairs"]
                if ([list(box) for box in pair[0]] != expected["pair"] or transform != expected["transform"]
                        or item["measurement"] != expected["measurement"]):
                    raise ValueError("The implementation selected a different native PNG pair")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--photographs", type=Path)
    parser.add_argument("--patterns", type=Path)
    args = parser.parse_args()
    if bool(args.photographs) != bool(args.patterns):
        parser.error("Provide both source folders for native replay")
    with gzip.open(args.receipt, "rt", encoding="utf-8") as file:
        receipt = json.load(file)
    result = review(receipt)
    if result != receipt["summary"]:
        raise ValueError("Published totals differ from the individual PNG measurements")
    if args.photographs:
        result["native_files_verified"] = native_replay(receipt, args.photographs, args.patterns)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
