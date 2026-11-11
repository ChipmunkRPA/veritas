"""Evaluate native paired image comparison on previously source-qualified edits."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from app.detectors.image import reference_compare
from scripts import image_background_patch_score as panel
from scripts import image_bg_real_insertion_score as metrics
from scripts import image_boundary_blend_transfer as blend
from scripts import image_relative_patch_transfer as shared

SCHEMA = "chipmunk.image-reference-compare-transfer/1"
EDITS = {"edited", "feathered_edit", "local_processing", "feathered_processing", "local_contrast", "blur"}
sha, write = shared.file_digest, shared.write


def compare(intake, receipt):
    bindings = {(row["slot"], row["generator"], row["role"], row["format"]): row for row in receipt}
    assert len(bindings) == 192
    for case, native, _ in panel.inputs(intake):
        source, rect = native["original"], case["box"]
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(source)).enhance(1.2))
        _, previous_jpeg = shared.flux.previous.encode(Image.fromarray(source), 90)
        views = native | {"feathered_edit": blend.local(source, native["edited"], rect, feather=True),
                          "feathered_processing": blend.local(source, native["local_processing"], rect, feather=True),
                          "local_contrast": blend.local(source, tone, rect), "global_contrast": tone,
                          "translated": np.roll(source, 3, axis=1), "recompressed_original": np.asarray(previous_jpeg)}
        for role, pixels in views.items():
            truth = np.any(pixels != source, axis=2) if role in EDITS else np.zeros(source.shape[:2], bool)
            for fmt in panel.FORMATS:
                body, image = panel.encoded(pixels, fmt)
                if role not in {"global_contrast", "translated", "recompressed_original"}:
                    binding = bindings.pop((case["slot"], case["generator"], role, fmt))
                    assert shared.digest(body) == binding["input_sha256"]
                view = np.asarray(image)
                baseline, baseline_mask = reference_compare.compare(source, view)
                with Image.open(io.BytesIO(body)) as carrier:
                    decision, mask = reference_compare.compare_delivered(source, view, carrier)
                measurements = {arm: metrics.measure(value if value is not None else np.zeros_like(truth), truth)
                                for arm, value in (("delivery_matched", mask), ("unconditioned", baseline_mask),
                                                   ("exact_pixel_difference", np.any(source != view, axis=2)))}
                assert measurements["delivery_matched"]["reported"] == sum(item["selected_pixels"] for item in decision["regions"])
                yield {"slot": case["slot"], "generator": case["generator"], "role": role, "format": fmt,
                       "input_sha256": shared.digest(body), "reference_rgb_sha256": shared.digest(source.tobytes()),
                       "truth_sha256": shared.digest(truth.tobytes()), "status": decision["status"], "baseline_status": baseline["status"],
                       "mask_sha256": shared.digest(mask.tobytes()) if mask is not None else None, "regions": decision["regions"],
                       "measurements": measurements}
    assert not bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for name in ("intake", "panel", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    rows_file = args.panel / "rows.json"
    result_file = args.panel / "result.json"
    assert sha(rows_file) == json.loads(result_file.read_bytes())["journal_sha256"]
    frozen = {"schema": SCHEMA, "implementation_sha256": sha(Path(__file__)),
              "comparison_sha256": sha(Path(reference_compare.__file__)), "source_intake_sha256": sha(args.intake / "result.json"),
              "previous_rows_sha256": sha(rows_file), "scope": "same dimensions and unchanged source geometry",
              "additional_controls": ["global_contrast", "translated", "recompressed_original"], "formats": list(panel.FORMATS),
              "useful_region": {"iou": .30, "recall": .50}}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", frozen)
        return
    assert frozen == json.loads((args.output / "freeze.json").read_bytes())
    cv2.setNumThreads(1)
    start = time.monotonic()
    observations = list(compare(args.intake, json.loads(rows_file.read_bytes())))
    assert len(observations) == 264
    write(args.output / "rows.json", observations)
    write(args.output / "result.json", {"schema": SCHEMA, "views": len(observations), "source_groups": 12,
          "freeze_sha256": sha(args.output / "freeze.json"), "rows_sha256": sha(args.output / "rows.json"),
          "elapsed_s": round(time.monotonic() - start, 3)})


if __name__ == "__main__":
    main()
