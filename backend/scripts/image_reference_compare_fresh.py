"""Test reference comparison on disjoint source images and unseen JPEG histories."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from app.detectors.image import reference_compare
from scripts import image_background_patch_score as panel
from scripts import image_bg_real_insertion_score as metrics
from scripts import image_relative_patch_transfer as shared

SCHEMA = "chipmunk.image-reference-delivery-holdout/1"
PRIOR_QUALITIES = (83, 92)
sha, write = shared.file_digest, shared.write


def material(intake):
    _, plan, journal = metrics.checked_inputs(intake)
    for slot, _, native, truth in metrics.panels(intake, plan, journal):
        original = native["original"]
        distance = cv2.distanceTransform(truth.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha = np.clip((distance - 1) / 8, 0, 1)[:, :, None]
        soft = np.rint(original * (1 - alpha) + native["edited"] * alpha).astype(np.uint8)
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(original)).enhance(1.2))
        views = {"original": original, "edited": native["edited"], "feathered_edit": soft,
                 "global_contrast": tone, "global_processing": native["processed"], "translated": np.roll(original, 3, axis=1)}
        for quality in PRIOR_QUALITIES:
            with io.BytesIO() as buffer:
                Image.fromarray(original).save(buffer, format="JPEG", quality=quality, subsampling=2)
                buffer.seek(0)
                with Image.open(buffer) as lossy:
                    views[f"intermediate_jpeg{quality}"] = np.array(lossy.convert("RGB"))
        for role, pixels in views.items():
            expected = np.any(original != pixels, axis=2) if role in {"edited", "feathered_edit"} else np.zeros_like(truth)
            assert not (expected & ~truth).any()
            for fmt in panel.FORMATS:
                body, delivered = panel.encoded(pixels, fmt)
                yield slot, role, fmt, original, body, np.asarray(delivered), expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for key in ("intake", "earlier", "later", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    folders = (args.intake, args.earlier, args.later)
    source_ids = [{case["flickr_photo_id"] for case in json.loads((folder / "plan.json").read_bytes())["cases"]} for folder in folders]
    assert [len(group) for group in source_ids] == [8, 12, 12] and len(set.union(*source_ids)) == 32
    freeze = {"schema": SCHEMA, "runner_sha256": sha(Path(__file__)), "method_sha256": sha(Path(reference_compare.__file__)),
              "source_receipts": [sha(folder / "result.json") for folder in folders], "intermediate_qualities": list(PRIOR_QUALITIES),
              "tested_qualities": list(reference_compare.PRIOR_JPEG_QUALITIES), "useful_iou": .30, "useful_recall": .50}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", freeze)
        return
    assert freeze == json.loads((args.output / "freeze.json").read_bytes())
    cv2.setNumThreads(1)
    rows = []
    for slot, role, fmt, original, body, submitted, truth in material(args.intake):
        arms = {}
        for name, method in (("delivery_matched", reference_compare.compare_delivered),
                             ("delivery_agreement", reference_compare.compare_across_deliveries)):
            with Image.open(io.BytesIO(body)) as carrier:
                decision, mask = method(original, submitted, carrier)
            measure = metrics.measure(mask if mask is not None else np.zeros_like(truth), truth)
            assert measure["reported"] == sum(region["selected_pixels"] for region in decision["regions"])
            arms[name] = {"status": decision["status"], "measure": measure, "regions": decision["regions"],
                          "mask_sha256": shared.digest(mask.tobytes()) if mask is not None else None}
        rows.append({"slot": slot, "role": role, "format": fmt, "input_sha256": shared.digest(body),
                     "source_rgb_sha256": shared.digest(original.tobytes()), "truth_sha256": shared.digest(truth.tobytes()), "arms": arms})
    assert len(rows) == 128
    write(args.output / "rows.json", rows)
    write(args.output / "result.json", {"schema": SCHEMA, "views": len(rows), "rows_sha256": sha(args.output / "rows.json"),
          "freeze_sha256": sha(args.output / "freeze.json")})


if __name__ == "__main__":
    main()
