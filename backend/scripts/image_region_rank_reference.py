"""Read the current source-aligned JPEG evidence on the exact comparison inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

from scripts.video_face_temporal_fit import read, sha


def intersections(selected, truth):
    return {"selected": int(selected.sum()), "intersection": int((selected & truth).sum()), "truth": int(truth.sum())}


def measure(study, output, *, repeat_jpeg90=False):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import jpeg_dq, regional
    from app.detectors.image.adapters.regional import RegionalDetector, evaluated_jpeg, repeat_components
    from app.detectors.image.regional_support import readout
    from app.evidence.image_regional_jpeg import reencode

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    manifest = study / "manifest.json"
    plan = read(manifest)
    roles = {row["tag"]: row["split"] for row in plan["groups"]}
    current = [row for row in plan["images"] if roles[row["group"]] != "fit"]
    model = regional.load_regional(regional.MODEL_ID, "cpu", "float32").obj
    tag = {"manifest_sha256": sha(manifest), "code_sha256": sha(Path(__file__)),
           "regional_sha256": regional.CHECKPOINT_SHA256, "jpeg_sha256": sha(Path(jpeg_dq.__file__)),
           "regional_rule_sha256": sha(Path(readout.__code__.co_filename))}
    if repeat_jpeg90:
        tag["repeat_jpeg90"] = True
        tag["repeat_rule_sha256"] = sha(Path(repeat_components.__code__.co_filename))
    existing = set()
    if output.exists():
        with output.open() as stream:
            if json.loads(next(stream)) != tag:
                raise ValueError("Cannot reuse different incumbent measurements")
            existing = {json.loads(line)["file"] for line in stream}
    with output.open("a" if existing else "w") as stream:
        if not existing:
            stream.write(json.dumps(tag, sort_keys=True) + "\n")
        for index, row in enumerate(current, 1):
            if row["file"] in existing:
                continue
            data = (study / row["file"]).read_bytes()
            mask_file = study / row["mask"]
            if hashlib.sha256(data).hexdigest() != row["sha256"] or sha(mask_file) != row["mask_sha256"]:
                raise ValueError("The original incumbent delivery or truth changed")
            with Image.open(mask_file) as labeled:
                truth = np.array(labeled.convert("L")) > 0
            old = jpeg_dq.analyse(data)
            physical = None
            if old is not None:
                selected = np.zeros_like(truth)
                if box := old.localised:
                    selected[8*box.y:8*(box.y + box.h), 8*box.x:8*(box.x + box.w)] = True
                physical = intersections(selected, truth)
            learned = None
            jpeg90 = repeat_jpeg90 and evaluated_jpeg(data, quality=90)
            if jpeg90 or evaluated_jpeg(data):
                with Image.open(study / row["file"]) as source:
                    image = source.convert("RGB")
                prediction = model.predict(image)
                rgb = np.asarray(image)
                measured = readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                                   RegionalDetector.REGION_THRESHOLD, data, rgb)
                if jpeg90 and measured.regions:
                    with Image.open(io.BytesIO(reencode(image))) as encoded:
                        second = model.predict(encoded.convert("RGB"))
                    repeated = readout(second.raw_mask, second.candidate_eligibility["eligible"],
                                       RegionalDetector.REGION_THRESHOLD, data, rgb)
                    aligned = repeated.selected_grid_mask
                    _, _, support = repeat_components(measured, aligned)
                    selected = np.zeros_like(aligned)
                    for row_support, component in zip(support["candidates"], measured.component_masks, strict=True):
                        if row_support["retained"]:
                            selected |= component
                    measured = replace(measured, selected_grid_mask=selected)
                learned = intersections(measured.source_mask(*image.size), truth)
            stream.write(json.dumps({"file": row["file"], "sha256": row["sha256"], "jpeg": physical, "regional": learned},
                                    sort_keys=True) + "\n")
            stream.flush()
            if index % 48 == 0 or index == len(current):
                print(f"Replayed the current checks on {index}/{len(current)} JPEGs.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deliveries", type=Path)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--repeat-jpeg90", action="store_true", help="Include the app's quality-90 repeat confirmation when applicable")
    args = parser.parse_args()
    measure(args.deliveries, args.journal, repeat_jpeg90=args.repeat_jpeg90)
