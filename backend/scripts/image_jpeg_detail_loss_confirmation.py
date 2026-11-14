# SPDX-License-Identifier: Apache-2.0
"""Verify an earlier-copy JPEG detail-loss check on separate photo groups."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import image_reference_same_jpeg_confirmation as source
from scripts import image_reference_same_jpeg_transfer as earlier

SCHEMA = "chipmunk.image-jpeg-detail-loss-confirmation/1"
STAGE = "source_disjoint_transfer"
BLUR = ("local_blur", "local_strong_blur", "local_weak_blur")
CONTROLS = ("unchanged", "strong_global_blur", "global_sharpen", "local_unsharp", "local_tone", "local_copy",
            "recompressed", "tandem", "resized", "shift", "unrelated", "generated_sharp", "generated_feathered")
ROLES = BLUR + CONTROLS
GATE = {"minimum_useful_added_moderate_or_strong_groups": 8, "maximum_control_flags": 2,
        "maximum_unrelated_or_whole_image_flags": 0}
DESIGN = {
    "gate": GATE,
    "scope": "Loss of fine luminance detail relative to a matching earlier JPEG90. AI weight zero.",
    "selection": "Retain incumbent regions. Try the new check if the incumbent returns no regions.",
    "truth": "Changed native pixels before final JPEG encode. Useful localization requires 0.30 IoU and 0.50 recall.",
    "role_limit": "Generated inpainting can also reduce detail. Generated controls test whether the measured signature generalizes to this confound.",
}
read, write, require, sha, digest = source.read, source.write, source.require, source.sha, source.digest


def cases(roster):
    import cv2
    import numpy as np
    from PIL import Image, ImageFilter

    from scripts import image_residual_encoder_transfer as delivery

    for slot, (_, record, views, truth, _) in enumerate(roster):
        pin, original = record["original"]["sha256"], views["original"]
        height, width = original.shape[:2]
        photo = Image.fromarray(original)
        before = delivery.encode(original, 90)
        extent = cv2.distanceTransform(truth.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = ((1 - np.cos(np.minimum(extent / 8, 1) * np.pi)) / 2).astype(np.float32)[..., None]

        def local(candidate, original=original, alpha=alpha):
            return np.rint(original * (1 - alpha) + np.asarray(candidate) * alpha).astype(np.uint8)

        prior, generated = delivery.authored(views, truth, pin)
        mild, moderate, strong = (photo.filter(ImageFilter.GaussianBlur(radius)) for radius in (.7, 1.35, 2.1))
        unsharp = photo.filter(ImageFilter.UnsharpMask(1.5, 120, 0))
        wrong = cv2.resize(roster[(slot + 1) % len(roster)][2]["original"], (width, height), interpolation=cv2.INTER_AREA)
        recipes = (prior["local_blur"], local(strong), local(mild), original, np.asarray(strong), np.asarray(unsharp),
                   local(unsharp), prior["local_tone"], prior["local_copy"], prior["second_jpeg"],
                   delivery.decode(before), prior["global_resize"], np.roll(original, (3, 6), axis=(0, 1)), wrong,
                   generated["sharp"], generated["feathered"])
        require(np.array_equal(local(moderate), prior["local_blur"]), "Moderate blur recipe differs")
        for role, pixels in zip(ROLES, recipes, strict=True):
            after = delivery.encode(pixels, 90)
            exact_truth = np.any(pixels != original, axis=2) if role in BLUR else np.zeros(truth.shape, bool)
            yield ({"slot": slot, "source_sha256": pin, "family": record["generator"], "role": role,
                    "shape": list(exact_truth.shape), "before_sha256": sha(before), "after_sha256": sha(after),
                    "truth_sha256": sha(exact_truth.tobytes()), "truth_pixels": int(exact_truth.sum())},
                   before, after, exact_truth)


def score(case, scratch):
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_jpeg_detail_loss
    from app.evidence import image_reference

    key, before, after, truth = case
    old, new = scratch / "before.jpg", scratch / "submitted.jpg"
    old.write_bytes(before)
    new.write_bytes(after)
    exhibits = scratch / "regions"
    previous = image_reference.inspect(old, new, evidence_dir=exhibits, allow_same_jpeg=True, **earlier.OPTIONS)
    incumbent = np.zeros_like(truth)
    if previous.get("mask"):
        with Image.open(exhibits / previous["mask"]["file"]) as image:
            incumbent = np.asarray(image) > 0
    with Image.open(io.BytesIO(before)) as a, Image.open(io.BytesIO(after)) as b:
        details, selected = reference_jpeg_detail_loss.compare(a, b, before, after)

    def stats(mask):
        return {"pixels": int(mask.sum()), "overlap": int((mask & truth).sum()), "mask_sha256": sha(mask.tobytes())}

    fine = (stats(selected) | {"regions": len(details["regions"]),
                              "measured_tiles": sum(region["directly_measured_tiles"] for region in details["regions"])}
            if selected is not None else None)
    return key | {"incumbent": stats(incumbent), "new_check": fine}


def review(lock, rows):
    require(lock["schema"] == SCHEMA and lock["stage"] == STAGE and lock["design"] == DESIGN, "Unexpected evaluation")
    require(len(rows) == 34 * len(ROLES) and len({row["source_sha256"] for row in rows}) == 34, "Bad denominator")
    by_role = {role: Counter() for role in ROLES}
    groups = set()
    for slot in range(34):
        group = rows[slot * len(ROLES):(slot + 1) * len(ROLES)]
        require(tuple(row["role"] for row in group) == ROLES and len({row["source_sha256"] for row in group}) == 1
                and all(row["slot"] == slot for row in group), "Missing files for a source")
    for row, sealed in zip(rows, lock["cases"], strict=True):
        require(all(row[name] == expected for name, expected in sealed.items()), "Case binding differs")
        role, truth = row["role"], row["truth_pixels"]
        incumbent, candidate = row["incumbent"], row["new_check"]
        fallback = bool(candidate and not incumbent["pixels"])
        combined = candidate if fallback else incumbent
        for evidence in (incumbent, candidate):
            if evidence is None:
                continue
            selected, overlap = evidence["pixels"], evidence["overlap"]
            require(0 <= overlap <= min(truth, selected) <= row["shape"][0] * row["shape"][1]
                    and truth + selected - overlap <= row["shape"][0] * row["shape"][1], "Impossible pixel counts")
        if candidate:
            selected = candidate["pixels"]
            require(selected % 256 == 0 and 0 < candidate["measured_tiles"] <= selected // 256
                    and selected >= 1024 * candidate["regions"] and 1 <= candidate["regions"] <= 5,
                    "Impossible native tile evidence")
        old, new = (earlier.prior.useful(item["pixels"], item["overlap"], truth) for item in (incumbent, combined))
        by_role[role].update(files=1, incumbent_useful=int(old), combined_useful=int(new),
                             candidate_useful=int(bool(candidate and earlier.prior.useful(candidate["pixels"], candidate["overlap"], truth))),
                             standalone_flags=int(bool(candidate)), fallback_flags=int(fallback), gains=int(new and not old))
        if role in BLUR[:2] and new and not old:
            groups.add(row["source_sha256"])
    flags = sum(by_role[key]["standalone_flags"] for key in CONTROLS)
    broad = sum(by_role[key]["standalone_flags"] for key in CONTROLS if not key.startswith(("local_", "generated_")))
    return {"by_role": by_role, "new_moderate_or_strong_groups": len(groups), "control_flags": flags,
            "whole_image_or_incompatible_flags": broad,
            "qualified": len(groups) >= GATE["minimum_useful_added_moderate_or_strong_groups"]
            and flags <= GATE["maximum_control_flags"] and broad <= GATE["maximum_unrelated_or_whole_image_flags"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay", "review"))
    parser.add_argument("run", type=Path)
    for name in source.QUALIFICATION:
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.run)
        lock = receipt["freeze"]
        require(receipt["freeze_sha256"] == sha(json.dumps(lock, sort_keys=True).encode()), "Freeze digest differs")
        result = review(lock, receipt["rows"])
        require(receipt["summary"] == result, "Published totals differ")
        print(json.dumps(result, sort_keys=True))
        return

    import cv2

    from app.detectors.image import reference_jpeg_detail_loss
    from app.evidence import image_reference

    require(all(getattr(args, name) for name in source.QUALIFICATION), "Supply the seven source rosters")
    cv2.setNumThreads(1)
    roster = source.sources(args)
    lock = {"schema": SCHEMA, "stage": STAGE, "design": DESIGN,
            "code_sha256": {Path(m.__file__).name: digest(Path(m.__file__))
                            for m in (source, earlier, reference_jpeg_detail_loss, image_reference)},
            "versions": {name: version(name) for name in ("numpy", "opencv-python-headless", "Pillow", "jpeglib")},
            "runner_sha256": digest(Path(__file__)), "cases": [row for row, *_ in cases(roster)]}
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        print(f"Froze {len(lock['cases'])} inputs on {len(roster)} source groups", flush=True)
        return
    require(lock == read(args.run / "freeze.json"), "Native candidate or case commitments differ")
    rows = []
    for case in cases(roster):
        with TemporaryDirectory(dir=args.run) as folder:
            rows.append(score(case, Path(folder)))
        if len(rows) % (3 * len(ROLES)) == 0:
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
