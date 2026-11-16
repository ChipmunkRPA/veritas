# SPDX-License-Identifier: MIT
"""Test paired local image changes across earlier JPEG and submitted lossy WebP."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_regional_lossy_webp_transfer as study

SCHEMA = "chipmunk.image-reference-webp-earlier-jpeg/1"
PROFILES = (
    ("standard", 75, "4:2:0", False, 65, 4),
    ("progressive", 82, "4:2:0", True, 82, 0),
    ("full_chroma", 94, "4:4:4", False, 92, 6),
)
ROLES = ("unchanged", "reencoded", "whole_picture_offset", "local_tone", "local_blur", "edited", "shifted")
ARMS = ("brightness", "brightness_and_color")
METHOD = {
    "sources": "Use eight exact retained photos per editor in each of two source-disjoint publisher banks. "
               "Use every selected case before inspecting a comparison. Keep the native dimensions.",
    "comparison": "Use the existing low-frequency brightness comparison and strict framing proof. "
                  "Test its optional low-frequency color output on the same two decoded inputs.",
    "controls": "Encode untouched source pixels or the earlier JPEG's decoded pixels as lossy WebP. "
                "Test a whole-image intensity offset, local tone and blur of the true editing footprint, "
                "and a four-pixel translation as a framing mismatch.",
    "useful": "Source mask IoU of at least 0.30 and at least 50 percent exact changed-pixel recall.",
}


def encode(pixels, codec, **options):
    import io

    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format=codec, **options)
    return stream.getvalue()


def decode(body):
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(body)) as image:
        return np.asarray(image.convert("RGB"))


def deliveries(root):
    import numpy as np

    for source, views in study.inputs(root, per_family=8):
        original = views["original"][0]
        for profile, jq, chroma, progressive, wq, effort in PROFILES:
            earlier = encode(original, "JPEG", quality=jq, subsampling=chroma, optimize=False, progressive=progressive)
            history = decode(earlier)
            versions = {"unchanged": original, "reencoded": history,
                        "whole_picture_offset": np.clip(history.astype(np.int16) + 6, 0, 255).astype(np.uint8),
                        **{key: views[key][0] for key in ("local_tone", "local_blur", "edited")},
                        "shifted": np.roll(original, 4, axis=1)}
            for name, image in versions.items():
                submitted = encode(image, "WEBP", quality=wq, method=effort, lossless=False)
                truth = views["edited"][1] if name in {"edited", "local_tone", "local_blur"} else views["original"][1]
                yield source | {"profile": profile, "role": name, "reference_sha256": study.sha(earlier),
                                "submitted_sha256": study.sha(submitted)}, earlier, submitted, truth


def summarize(rows):
    counts = defaultdict(Counter)
    for row in rows:
        for arm, measured in row["scores"].items():
            case = counts[f"{row['bank']}/{row['profile']}/{arm}"]
            role = row["role"]
            case[role] += 1
            case[f"{role}_comparable"] += row["status"] == "comparable"
            case[f"{role}_flags"] += measured["pixels"] > 0
            case[f"{role}_useful"] += study.useful(measured, row["true_pixels"] if role in {"edited", "local_tone", "local_blur"} else 0)
    return {key: dict(value) for key, value in sorted(counts.items())}


def validate(report):
    verify, rows = report["freeze"], report["rows"]
    assert verify["schema"] == SCHEMA and verify["method"] == METHOD
    assert verify["profiles"] == [list(profile) for profile in PROFILES]
    assert len(verify["cases"]) == len(rows) == 48 * 3 * 7
    cases = {(row["source_id"], row["profile"], row["role"]) for row in rows}
    assert len({source for source, _, _ in cases}) == 48
    assert len(cases) == len(rows) and all(profile in {item[0] for item in PROFILES} and role in ROLES
                                         for _, profile, role in cases)
    for row, case in zip(rows, verify["cases"], strict=True):
        assert all(row[key] == value for key, value in case.items())
        assert set(row["scores"]) == set(ARMS)
        assert row["status"] in {"comparable", "webp_lossy_delivery_unassessed"}
        target = row["true_pixels"] if row["role"] in {"edited", "local_tone", "local_blur"} else 0
        for value in row["scores"].values():
            assert 0 <= value["overlap"] <= min(target, value["pixels"]) <= max(target, value["pixels"]) <= row["size"][0] * row["size"][1]
            assert row["status"] == "comparable" or not value["pixels"]
    summary = summarize(rows)
    assert summary == report["summary"]
    return summary


def score(root, output):
    import cv2
    import numpy as np

    from app.detectors.image import reference_lossy_webp, reference_webp_color

    cv2.setNumThreads(1)
    output.mkdir(mode=0o700)
    freeze = {"schema": SCHEMA, "method": METHOD, "profiles": [list(profile) for profile in PROFILES],
              "cases": [row for row, _, _, _ in deliveries(root)]}
    assert len(freeze["cases"]) == 48 * 3 * 7
    study.save(output / "freeze.json", freeze)
    rows = []
    for index, (case, earlier, submitted, truth) in enumerate(deliveries(root)):
        assert case == freeze["cases"][index] and reference_lossy_webp.webp_size(submitted) == tuple(case["size"])
        before, after = decode(earlier), decode(submitted)
        result, luma = reference_lossy_webp.compare_pixels(before, after)
        combined = np.zeros_like(truth)
        luma = luma if luma is not None else np.zeros_like(truth)
        if result["status"] == "comparable":
            _, _, other, _ = reference_webp_color.compare(before, after, result)
            combined = luma | other
        rows.append(case | {"status": result["status"], "scores": {
            "brightness": study._bounds(luma, truth), "brightness_and_color": study._bounds(combined, truth)}})
        if (index + 1) % 168 == 0:
            print(f"Compared {index + 1}/{len(freeze['cases'])} pairs with exact source bindings.", flush=True)
    report = {"freeze": freeze, "rows": rows, "summary": summarize(rows)}
    validate(report)
    study.save(output / "receipt.json.gz", report)


def replay(root, report):
    """Verify the legacy and new production comparison functions on the exact panel."""
    import io

    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_lossy_webp

    validate(report)
    cv2.setNumThreads(1)
    for row, (case, earlier_bytes, submitted_bytes, truth) in zip(report["rows"], deliveries(root), strict=True):
        assert all(row[key] == value for key, value in case.items())
        with Image.open(io.BytesIO(earlier_bytes), formats=["JPEG"]) as earlier, \
                Image.open(io.BytesIO(submitted_bytes), formats=["WEBP"]) as submitted:
            legacy, _ = reference_lossy_webp.compare(earlier, submitted, earlier_bytes, submitted_bytes)
            assert legacy["status"] == reference_lossy_webp.UNASSESSED
            current, mask = reference_lossy_webp.compare_earlier_jpeg(earlier, submitted, earlier_bytes, submitted_bytes)
        assert current["status"] == row["status"] and current["affects_overall_confidence"] is False
        selected = np.zeros_like(truth) if mask is None else mask
        assert study._bounds(selected, truth) == row["scores"]["brightness"]
    return {"replayed_pairs": len(report["rows"]), "legacy_pairs_unassessed": len(report["rows"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("score")
    run.add_argument("sources", type=Path)
    run.add_argument("output", type=Path)
    commands.add_parser("review").add_argument("receipt", type=Path)
    check = commands.add_parser("replay")
    check.add_argument("receipt", type=Path)
    check.add_argument("sources", type=Path)
    args = parser.parse_args()
    if args.action == "score":
        score(args.sources, args.output)
    elif args.action == "replay":
        print(json.dumps(replay(args.sources, study.read(args.receipt)), indent=2))
    else:
        print(json.dumps(validate(study.read(args.receipt)), indent=2))
