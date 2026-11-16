"""Read the current source-aligned JPEG evidence on the exact comparison inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.video_face_temporal_fit import read, sha


def intersections(selected, truth):
    return {"selected": int(selected.sum()), "intersection": int((selected & truth).sum()), "truth": int(truth.sum())}


def measure(study, output):
    import cv2
    import numpy as np
    import torch
    from app.detectors.image import jpeg_dq, regional
    from app.detectors.image.adapters.regional import RegionalDetector, evaluated_jpeg
    from app.detectors.image.regional_support import readout
    from PIL import Image

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
            if evaluated_jpeg(data):
                with Image.open(study / row["file"]) as source:
                    image = source.convert("RGB")
                prediction = model.predict(image)
                measured = readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                                   RegionalDetector.REGION_THRESHOLD, data, np.array(image))
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
    args = parser.parse_args()
    measure(args.deliveries, args.journal)
