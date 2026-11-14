"""Verify regional evidence across lossless baseline-to-progressive JPEG transport."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.regional-progressive-transfer/1"
PLAN = {"sources": "Held TGIF/TGIF2 source groups with short/long dimension ratio at least 2/3, first six by original SHA-256 in each family",
        "delivery": "Downsample the full image to a 768-pixel long side, preserve aspect, encode baseline quality-75 4:2:0",
        "roles": ["ordinary", "local_blur", "local_tone", "sharp", "feather"],
        "comparison": "Losslessly transcode JPEG coefficients. Require exact decoded RGB, quantization tables, and sampling. "
                      "Independently run the regional check on a preselected edited and ordinary pair in each family.",
        "pixel_point": .9, "minimum_active_fraction": .01, "minimum_iou": .30, "minimum_recall": .50}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def summarize(rows):
    total = Counter()
    for row in rows:
        total.update(deliveries=1, edited=row["edited_pixels"] > 0,
                     exact_rgb=row["baseline_rgb_sha256"] == row["progressive_rgb_sha256"],
                     exact_profile=row["tables_and_sampling_equal"], model_replays="model" in row)
        if model := row.get("model"):
            count, overlap, target = model["measured_pixels"], model["overlap"], row["edited_pixels"]
            useful = bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (count + target - overlap))
            total.update(useful_model_replays=useful, ordinary_model_highlights=bool(not target and count))
    return dict(total)


def audit(record):
    rows = record["rows"]
    require(record["schema"] == SCHEMA and record["design"] == PLAN and len(rows) == 90, "Unexpected frozen image set")
    groups = {row["group"] for row in rows}
    require(groups == set(range(1, 19))
            and all({row["role"] for row in rows if row["group"] == group} == set(PLAN["roles"]) for group in groups),
            "Every source requires all five delivery variants")
    for row in rows:
        require(row["baseline_sha256"] != row["progressive_sha256"] and row["scan_count"] >= 2
                and row["baseline_rgb_sha256"] == row["progressive_rgb_sha256"] and row["tables_and_sampling_equal"] is True,
                "Progressive delivery changed the measured pixels or compression profile")
        h, w = row["shape"]
        target = row["edited_pixels"]
        require(max(h, w) == 768 and min(h, w) >= 512 and 0 <= target <= h * w, "Unexpected native dimensions or mask")
        require(bool(target) == (row["role"] in {"sharp", "feather"}), "Unexpected edit truth for delivery")
        expected_model = row["group"] in {1, 7, 13} and row["role"] in {"ordinary", "feather"}
        require(("model" in row) == expected_model, "Unexpected selection for model replay")
        if model := row.get("model"):
            count, overlap = model["measured_pixels"], model["overlap"]
            require(model["baseline_mask_sha256"] == model["progressive_mask_sha256"]
                    and model["baseline_input_sha256"] == model["progressive_input_sha256"]
                    and 0 <= count <= h * w and max(0, target + count - h * w) <= overlap <= min(count, target),
                    "Live model responses or native overlaps differ")
    summary = summarize(rows)
    require(summary["model_replays"] == 6 and record["summary"] == summary, "Progressive comparison counts differ")
    return summary


def deliveries(sources):
    import cv2
    import numpy as np
    from PIL import Image

    from scripts import image_jpeg_signed_transitions as archive

    held = archive.sources(argparse.Namespace(sources=sources))["held"]
    selected = []
    for family in archive.FAMILIES:
        rows = []
        for folder, original, edit in held:
            if edit["family"] == family:
                with Image.open(folder / original["image"]) as image:
                    if min(image.size) * 3 >= max(image.size) * 2:
                        rows.append((folder, original, edit))
        rows.sort(key=lambda row: row[1]["image_sha256"])
        require(len(rows) >= 6, "Not enough images in the supported geometry")
        selected.extend(rows[:6])
    for group, (key, family, original, edited, exact) in enumerate(archive.archive_pairs({"held": selected}, "held"), 1):
        h, w = exact.shape
        size = (w * 768 // max(h, w), h * 768 // max(h, w))

        original, edited = (cv2.resize(pixels, size, interpolation=cv2.INTER_AREA) for pixels in (original, edited))
        exact = np.any(original != edited, axis=2)
        for row, jpeg, mask in archive.views_from_pairs([(key, family, original, edited, exact)], "held"):
            if row["quality"] == 75:
                yield group, row, jpeg, mask


def run(args):
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import jpeg_bitstream, regional, regional_support

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    binary = shutil.which("jpegtran")
    require(binary is not None, "The coefficient transcode requires local jpegtran")
    model = regional.load_model(args.checkpoint)
    rows = []
    for group, key, body, mask in deliveries(args.sources):
        truth = mask if key["role"] in ("sharp", "feather") else np.zeros_like(mask)
        converted = subprocess.run([binary, "-copy", "all", "-progressive"], input=body,
                                   capture_output=True, timeout=10, check=True).stdout
        before, after = jpeg_bitstream.read(body), jpeg_bitstream.progressive_frame(converted)
        require(before and after is not None, "Expected standard progressive JPEG")
        structure, scans = after
        images, rgb = [], []
        for carrier in (body, converted):
            with Image.open(io.BytesIO(carrier)) as image:
                images.append(image.convert("RGB"))
                rgb.append(np.asarray(images[-1]))
        row = {"group": group, "family": key["family"], "role": key["role"], "shape": list(truth.shape),
               "baseline_sha256": sha(body), "progressive_sha256": sha(converted), "truth_sha256": sha(truth.tobytes()),
               "baseline_rgb_sha256": sha(rgb[0].tobytes()), "progressive_rgb_sha256": sha(rgb[1].tobytes()),
               "edited_pixels": int(truth.sum()), "scan_count": scans, "tables_and_sampling_equal": before.fingerprint == structure.fingerprint}
        if (group - 1) % 6 == 0 and key["role"] in ("ordinary", "feather"):
            inputs, maps = [], []
            for image, pixels, carrier in zip(images, rgb, (body, converted), strict=True):
                prediction = model.predict(image)
                inputs.append(prediction.geometry["input_float32_sha256"])
                maps.append(regional_support.readout(prediction.raw_mask, prediction.candidate_eligibility["eligible"],
                                                     regional.PIXEL_THRESHOLD, carrier, pixels).source_mask(*image.size))
            row["model"] = {"baseline_input_sha256": inputs[0], "progressive_input_sha256": inputs[1],
                            "baseline_mask_sha256": sha(maps[0].tobytes()), "progressive_mask_sha256": sha(maps[1].tobytes()),
                            "measured_pixels": int(maps[0].sum()), "overlap": int(np.count_nonzero(truth & maps[0]))}
        rows.append(row)
        if len(rows) % 30 == 0:
            print(f"Verified {len(rows)} JPEG pairs", flush=True)
    version = subprocess.run([binary, "-version"], capture_output=True, text=True, timeout=5, check=True)
    record = {"schema": SCHEMA, "design": PLAN, "checkpoint_sha256": regional.CHECKPOINT_SHA256,
              "jpegtran": (version.stdout or version.stderr).strip(), "rows": rows, "summary": summarize(rows)}
    audit(record)
    with args.output.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(record["summary"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", type=Path, default=Path("backend/data/localization-eval"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/weights/native-regional-v1/regional-model.safetensors"))
    args = parser.parse_args()
    if args.phase == "review":
        print(json.dumps(audit(json.loads(args.output.read_bytes())), sort_keys=True))
        return
    run(args)


if __name__ == "__main__":
    main()
