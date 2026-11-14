# SPDX-License-Identifier: Apache-2.0
"""Test high-frequency JPEG changes against a matched earlier copy."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import image_reference_same_jpeg_confirmation as confirmation
from scripts import image_reference_same_jpeg_transfer as prior

SCHEMA = "chipmunk.image-jpeg-ac-difference-transfer/1"
DESIGN = {
    "assessed_scope": "Upright same-size JPEG90 pairs, with at least 85% exact decoded RGB and distributed unique anchors.",
    "evidence": "Quantized luma AC coefficient differences, dequantized in both copies. Select full native 8x8 blocks at 2.5 luminance levels RMS.",
    "combination": "Retain all incumbent regions. Accept connected AC components of at least 9 native blocks, without filling gaps.",
    "adoption": "At least two added useful generated-edit locations in separate held source groups. Tabulate any lost useful masks, mislocated edits, and additional global or incompatible-control highlights.",
    "interpretation": "Local frequency content differs between supplied JPEGs. AI weight is zero.",
}
read, write, require, sha, digest = prior.read, prior.write, prior.require, prior.sha, prior.digest


def coefficient_mask(earlier, submitted, before, after):
    import cv2
    import numpy as np

    from app.detectors.image import (
        jpeg_dq,
        reference_exact_jpeg,
        reference_same_jpeg,
        reference_structure_jpeg,
    )

    if (earlier.size != submitted.size or not 512 <= min(earlier.size) <= max(earlier.size) <= 768
            or not all(reference_structure_jpeg._jpeg_profile(image, payload, (reference_same_jpeg.QUALITY,))
                       for image, payload in ((earlier, before), (submitted, after)))):
        return None
    a, b = np.asarray(earlier), np.asarray(submitted)
    proof = reference_exact_jpeg.alignment_proof(a, b)
    if proof is None or 20 * proof["exact_tiles"] < 17 * proof["full_tiles"]:
        return None
    height, width = a.shape[:2]
    mask = np.zeros((height, width), bool)
    if before == after:
        return mask

    left, right = (jpeg_dq.read_coefficients(payload) for payload in (before, after))
    if (left is None or right is None or left.y.shape != right.y.shape
            or not np.array_equal(left.table, right.table)):
        return None
    rows, columns = height // 8, width // 8
    delta = (left.y[:rows, :columns] - right.y[:rows, :columns]) * left.table
    delta[:, :, 0, 0] = 0
    # The orthonormal inverse DCT preserves energy. A block has 64 samples.
    squared_energy = np.square(delta).sum(axis=(-2, -1))
    changed = squared_energy >= 400
    _, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 9
    keep[0] = False
    mask[:rows * 8, :columns * 8] = keep[labels].repeat(8, axis=0).repeat(8, axis=1)
    return mask


def measure(case, scratch):
    import numpy as np
    from PIL import Image

    from app.evidence import image_reference

    key, before, after, truth = case
    paths = [(scratch / name) for name in ("earlier.jpg", "submitted.jpg")]
    for path, payload in zip(paths, (before, after), strict=True):
        path.write_bytes(payload)
    evidence = scratch / "evidence"
    output = image_reference.inspect(*paths, evidence_dir=evidence, allow_same_jpeg=True, **prior.OPTIONS)
    current = np.zeros_like(truth)
    if artifact := output.get("mask"):
        with Image.open(evidence / artifact["file"]) as image:
            current = np.asarray(image) > 0
    with Image.open(io.BytesIO(before)) as left, Image.open(io.BytesIO(after)) as right:
        candidate = coefficient_mask(left, right, before, after)
    combined = current if candidate is None else (current | candidate)

    def stats(mask):
        return {"pixels": int(mask.sum()), "overlap": int((mask & truth).sum()), "sha256": sha(mask.tobytes())}

    return {**key, "current_assessed": output["status"] == "comparable",
            "ac_assessed": candidate is not None, "current": stats(current), "combined": stats(combined),
            "ac": stats(candidate) if candidate is not None else None}


def review(lock, rows):
    require(lock["schema"] == SCHEMA and lock["design"] == DESIGN, "Unexpected frozen method")
    counts = {}
    added = set()
    for row, key in zip(rows, lock["cases"], strict=True):
        require(all(row[k] == v for k, v in key.items()), "Result input differs")
        role = row["role"]
        group = f"{row['family']}/{role}"
        counters = counts.setdefault(group, Counter())
        target = row["truth_pixels"]
        measured = row["ac_assessed"]
        require(measured == (row["ac"] is not None), "AC availability and measure disagree")
        require(measured or row["current"] == row["combined"], "Unavailable check changed result")
        pixels = row["shape"][0] * row["shape"][1]
        totals = {}
        for name in ("current", "ac", "combined"):
            values = row[name]
            if values is None:
                continue
            area, overlap = values["pixels"], values["overlap"]
            require(0 <= overlap <= min(area, target) and area + target - overlap <= pixels, "Impossible native overlap")
            totals[name] = prior.prior.useful(area, overlap, target)
        current, combined = row["current"], row["combined"]
        require(all(combined[k] >= current[k] for k in ("pixels", "overlap")), "Incumbent pixels disappeared")
        if measured:
            ac = row["ac"]
            require(ac["pixels"] % 64 == 0 and (ac["pixels"] == 0 or ac["pixels"] >= 9 * 64), "Partial native JPEG block")
            require(all(max(current[k], ac[k]) <= combined[k] <= current[k] + ac[k] for k in ("pixels", "overlap")),
                    "Impossible pixel union")
        gained = totals["combined"] and not totals["current"]
        lost = totals["current"] and not totals["combined"]
        counters.update(files=1, current_assessed=int(row["current_assessed"]), ac_assessed=int(measured),
                        current_highlight=int(current["pixels"] > 0), combined_highlight=int(combined["pixels"] > 0),
                        current_useful=int(totals["current"]), combined_useful=int(totals["combined"]),
                        gained=int(gained), lost=int(lost))
        if gained and role in prior.POSITIVE:
            added.add(row["source_sha256"])
    return {"by_family_and_role": counts, "groups_with_added_useful_generated_locations": len(added)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--stage", choices=("diagnostic", "confirmation"), default="confirmation")
    for name in set(confirmation.QUALIFICATION) | {"main", "external", "confirmation"}:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.run)
        require(receipt["freeze_sha256"] == sha(json.dumps(receipt["freeze"], sort_keys=True).encode()), "Freeze differs")
        result = review(receipt["freeze"], receipt["rows"])
        require(receipt["summary"] == result, "Result differs")
        print(json.dumps(result, sort_keys=True))
        return

    import cv2

    from app.detectors.image import (
        jpeg_dq,
        reference_compare,
        reference_exact_jpeg,
        reference_same_jpeg,
        reference_structure_jpeg,
    )
    from app.evidence import image_reference

    cv2.setNumThreads(1)
    if args.stage == "diagnostic":
        args.lama = prior.prior.PRIVATE / "image-bg-real-acquisition-v225"

        def cases():
            return prior.cases(args, "diagnostic")
    else:
        roster = confirmation.sources(args)

        def cases():
            return confirmation.cases(roster)

    lock = {"schema": SCHEMA, "stage": args.stage, "design": DESIGN,
            "code": {Path(module.__file__).name: digest(Path(module.__file__)) for module in
                     (confirmation, prior, jpeg_dq, reference_compare, reference_exact_jpeg, reference_same_jpeg,
                      reference_structure_jpeg, image_reference)},
            "code_sha256": digest(Path(__file__)), "cases": [key for key, *_ in cases()]}
    predecessor = read(prior.PANEL / f"image-reference-same-jpeg-{args.stage}-2026-11-13.json")
    require(lock["cases"] == predecessor["freeze"]["cases"], "The existing source and truth commitments changed")
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        print(f"Froze {len(lock['cases'])} {args.stage} inputs", flush=True)
        return
    require(lock == read(args.run / "freeze.json"), "Frozen code or prepared images differ")
    rows = []
    for case in cases():
        with TemporaryDirectory(dir=args.run) as temporary:
            rows.append(measure(case, Path(temporary)))
        if len(rows) % 50 == 0:
            print(f"Measured {len(rows)} of {len(lock['cases'])} pairs", flush=True)
    result = review(lock, rows)
    receipt = {"freeze": lock, "freeze_sha256": sha(json.dumps(lock, sort_keys=True).encode()), "rows": rows, "summary": result}
    if args.phase == "replay":
        require(read(args.run / "receipt.json") == receipt, "Native replay differs")
    else:
        write(args.run / "receipt.json", receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
