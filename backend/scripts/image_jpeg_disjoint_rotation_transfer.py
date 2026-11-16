# SPDX-License-Identifier: Apache-2.0
"""Compare one versus two distinct rotated-copy pairs on large JPEG deliveries."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts import image_jpeg_native_rotation_transfer as prior

SCHEMA = "chipmunk.image-jpeg-disjoint-rotation/1"
ROLES = ("two_quarter", "quarter_and_half", "shared_source", "two_straight")
METHOD = {
    "source": "Use pinned original quality-90 and quality-95 camera JPEGs. Make two fresh copy regions on the decoded native grid, then export at the same profile. Retain the earlier ordinary, structural and single-edit controls.",
    "selection": "Keep existing copy proposals and color measurements. Take at most two independent pairs. The four crops need at least 16 pixels of separation across the two pairs. Decline shared crops or more than two distinct pairs.",
    "advance": "Require at least four complete new double locations across at least three held cameras, no new ordinary flags, and no false double claim on shared-source copies. Retain earlier single locations.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(phase):
    return prior.PANELS / f"image-jpeg-native-rotation-{phase}-2026-11-15.json"


def inputs(directory, phase):
    published = prior.read(reference(phase))
    plan = prior.read(directory / "freeze.json")
    committed = {row["file"]: row for row in published["freeze"]["rows"]}
    for row in plan["rows"]:
        approved = committed[row["file"]]
        require(all(row[key] == approved[key] for key in ("sha256", "group", "geometry", "role", "size", "quality", "truth")),
                "Earlier JPEG input differs from the public source")
    require(len(plan["rows"]) == len(committed), "Earlier input roster is incomplete")
    return plan


def sources(rgb, group):
    import cv2
    import numpy as np

    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pairs = []
    for band, side in enumerate((88, 80)):
        top, bottom = (height // 2 * band + 32, height // 2 * (band + 1) - side - 32)
        texture = []
        for y in range(top, bottom + 1, 67):
            for x in range(width // 2 + 32, width - side - 31, 71):
                patch = gray[y:y + side, x:x + side]
                contrast = float(np.abs(np.diff(patch, axis=1)).mean() + np.abs(np.diff(patch, axis=0)).mean())
                texture.append((contrast, x, y))
        _, sx, sy = max(texture)
        tx = width // 6 + group * 29 % 117
        ty = top + (bottom - top) // 3 + group * 23 % 83
        pairs.append(((tx, ty, side, side), (sx, sy, side, side)))
    return pairs


def delivered(rgb, placements, role, quality):
    import numpy as np

    from scripts.image_jpeg_turned_copy_transfer import encode_jpeg

    modified = rgb.copy()
    truth = []
    turns = {"two_quarter": (1, 3), "quarter_and_half": (1, 2), "shared_source": (1, 3), "two_straight": (0, 0)}[role]
    for index, ((x, y, w, h), (sx, sy, sw, sh)) in enumerate(placements):
        if index and role == "shared_source":
            sx, sy, sw, sh = placements[0][1]
            w = h = sw
        rotated = np.rot90(rgb[sy:sy + sh, sx:sx + sw], turns[index])
        target = modified[y:y + h, x:x + w]
        require(bool(np.any(target != rotated)), "The staged copy produced no change")
        target[:] = rotated
        truth.append([[x, y, w, h], [sx, sy, sw, sh]])
    return encode_jpeg(modified, quality), truth


def work(source, row):
    import numpy as np
    from PIL import Image

    from app.detectors.image import jpeg_native_turned_copy as detector
    from app.detectors.image import jpeg_turned_copy

    def decode(data):
        with Image.open(io.BytesIO(data), formats=["JPEG"]) as image:
            require(list(image.size) == row["size"], "Original source geometry changed")
            return np.asarray(image)

    def classify(data, role, truth):
        pixels = decode(data)
        state, matches = detector.candidates(pixels)
        old_status, old = detector.select(matches) if matches else (state, [])
        new_status, new = detector.select(matches, allow_disjoint=True) if matches else (state, [])
        for finding in new:
            require(jpeg_turned_copy.witness(pixels, finding["pair"], finding["turns"]) == finding["measurement"],
                    "Selected witness cannot be reconstructed from the source")
        return {"group": row["group"], "geometry": row["geometry"], "quality": row["quality"], "role": role,
                "sha256": prior.digest(data), "source_sha256": row["sha256"], "truth": truth,
                "prior": {"status": old_status, "pairs": old}, "candidate": {"status": new_status, "pairs": new}}

    data = (source / row["file"]).read_bytes()
    require(prior.digest(data) == row["sha256"], "An earlier encoded JPEG differs")
    results = [classify(data, row["role"], [row["truth"]] if row["truth"] else [])]
    if row["role"] == "ordinary":
        pixels = decode(data)
        positions = sources(pixels, int(row["group"]))
        for role in ROLES:
            jpeg, truth = delivered(pixels, positions, role, row["quality"])
            results.append(classify(jpeg, role, truth))
    return results


def counts(rows):
    result = {}
    for label in ("prior", "candidate"):
        counters, strongest_groups = Counter(), set()
        for row in rows:
            role = row["role"]
            pairs = [item["pair"] for item in row[label]["pairs"]]
            key = "dual" if role in ROLES[:2] else "shared" if role == "shared_source" else "single" if row["truth"] and role != "two_straight" else "ordinary"
            useful = sum(prior.useful(pairs, known) for known in row["truth"])
            counters.update({f"{key}_files": 1, f"{key}_flagged": bool(pairs), f"{key}_two_pairs": len(pairs) == 2,
                             f"{key}_unassessed": row[label]["status"] not in {"no_pair", "supported_pair"}})
            if key == "dual":
                counters.update(dual_useful_pairs=useful, dual_both=useful == 2, dual_misplaced_pairs=len(pairs) - useful)
                if useful == 2:
                    strongest_groups.add(row["group"])
            if key == "single":
                counters["single_useful"] += useful
                require(label != "candidate" or not prior.useful([item["pair"] for item in row["prior"]["pairs"]], row["truth"][0])
                        or useful, "A prior correct single location was lost")
        result[label] = dict(counters, dual_cameras=len(strongest_groups))
    candidate, old = result["candidate"], result["prior"]
    result["advances"] = (candidate["dual_both"] >= 4 and candidate["dual_cameras"] >= 3
                           and candidate["ordinary_flagged"] <= old["ordinary_flagged"] and candidate["shared_two_pairs"] == 0)
    return result


def capture(args):
    import cv2

    cv2.setNumThreads(1)
    plan = inputs(args.inputs, args.phase)
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        rows = []
        for index, observed in enumerate(workers.map(lambda row: work(args.inputs, row), plan["rows"]), 1):
            rows.extend(observed)
            if index % 20 == 0:
                print(f"Verified measurements for {index}/{len(plan['rows'])} earlier source files and their staged deliveries.", flush=True)
    report = {"schema": SCHEMA, "phase": args.phase, "method": METHOD, "source_sha256": prior.sha(reference(args.phase)),
              "rows": rows, "result": counts(rows)}
    prior.write(args.output, report)
    return report["result"]


def review(path):
    from app.evidence.image_copy_similarity_binding import _validate_jpeg_measurement, disjoint_copy_regions

    report = prior.read(path)
    require(report["schema"] == SCHEMA and report["method"] == METHOD
            and report["source_sha256"] == prior.sha(reference(report["phase"])), "The published source roster changed")
    previous = prior.read(reference(report["phase"]))
    source_by_sha = {row["sha256"]: (row, item["native"])
                     for row, item in zip(previous["freeze"]["rows"], previous["observed"], strict=True)}
    require(len(source_by_sha) == len(previous["freeze"]["rows"]), "Earlier source hashes are ambiguous")
    expected = [(row["sha256"], role) for row in previous["freeze"]["rows"]
                for role in ((row["role"], *ROLES) if row["role"] == "ordinary" else (row["role"],))]
    require([(row["source_sha256"], row["role"]) for row in report["rows"]] == expected, "The measured roster differs")
    for row in report["rows"]:
        source, baseline = source_by_sha[row["source_sha256"]]
        require(all(row[key] == source[key] for key in ("group", "geometry", "quality")), "The delivery profile changed")
        if row["role"] not in ROLES:
            require(row["sha256"] == source["sha256"] and row["truth"] == ([source["truth"]] if source["truth"] else [])
                    and row["prior"] == baseline, "An earlier single-pair decision changed")
        w, h = source["size"]
        for label, limit in (("prior", 1), ("candidate", 2)):
            result = row[label]
            pairs = result["pairs"]
            require(len(pairs) <= limit and bool(pairs) == (result["status"] == "supported_pair"), "Incorrect pair count or status")
            for item in pairs:
                require(item["turns"] in {1, 2} and _validate_jpeg_measurement(item["measurement"])
                        and len(item["pair"]) == 2 and all(0 <= x < x + bw <= w and 0 <= y < y + bh <= h
                                                           for x, y, bw, bh in item["pair"]), "Invalid original JPEG crop")
            if len(pairs) == 2:
                require(disjoint_copy_regions(*(item["pair"] for item in pairs)), "Two selected pairs share a source crop")
    result = counts(report["rows"])
    require(result == report["result"], "The published outcomes differ from replay")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="action", required=True)
    take = phases.add_parser("capture")
    take.add_argument("phase", choices=("development", "held"))
    take.add_argument("inputs", type=Path)
    take.add_argument("output", type=Path)
    take.add_argument("--workers", type=int, default=3)
    replay = phases.add_parser("review")
    replay.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = capture(args) if args.action == "capture" else review(args.receipt)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
