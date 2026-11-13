"""Replay the paired-source JPEG90 reflection diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "docs/evaluation-panels"
SCHEMA = "chipmunk.image-jpeg90-reflection-source-context/1"
CONFIRMATION = PANEL / "image-regional-jpeg90-quality-confirmation-2026-11-13.json"
REFLECTION = PANEL / "image-regional-jpeg90-reflection-2026-11-13.json"
RECEIPT = PANEL / "image-jpeg90-reflection-source-context-2026-11-13.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def useful(area, overlap, target):
    return bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (area + target - overlap))


def summarize(rows, confirmation, reflection):
    require(len(rows) == len(confirmation["rows"]) == 88 and len(reflection["rows"]) == 288, "Held extent differs")
    positive, ordinary = Counter(), Counter()
    ordinary_roles, positive_families = Counter(), Counter()
    hard_groups = set()
    for index, (row, key, baseline, reflected) in enumerate(zip(rows, confirmation["freeze"]["rows"],
        confirmation["rows"], reflection["rows"][200:], strict=True), 200):
        require(row["index"] == reflected["index"] == index and
                all(row[field] == key[field] for field in ("group", "role", "family")) and
                reflected["input_sha256"] == key["q90_sha256"] and reflected["truth_sha256"] == key["truth_sha256"] and
                reflected["native_mask_sha256"] == baseline["mask_sha256"]["direct_q90"],
                "Published JPEG90 source or mask binding differs")
        height, width = key["geometry"]
        target = row["truth_pixels"]
        require(target == baseline["measures"]["direct_q90"]["reference_positive_pixels"], "Known edit area differs")
        pair = row["earlier_copy"]
        area, overlap = pair["pixels"], pair["overlap"]
        require(pair["status"] == "comparable" and type(area) is int and type(overlap) is int
                and 0 <= overlap <= min(area, target) and area + target - overlap <= height * width,
                "Earlier-copy comparison totals differ")
        hit = useful(area, overlap, target)
        require(pair["useful"] is hit, "Earlier-copy useful location differs")
        if target:
            positive.update(planned=1, useful=int(hit))
            positive_families[row["family"]] += int(hit)
        else:
            ordinary.update(planned=1, earlier_copy_observations=int(area > 0))
            ordinary_roles[row["role"]] += int(area > 0)
        components = row["native_components"]
        require(len(components) == len(reflected["components"]), "Native component count differs")
        ordinary_hard = []
        for current, saved in zip(components, reflected["components"], strict=True):
            hard = 2 * saved["hard_overlap"] >= saved["area"]
            raw, supported = current["raw_diff_pixels"], current["supported_earlier_copy_diff_pixels"]
            require(current["mask_sha256"] == saved["mask_sha256"] and current["hard"] is hard and
                    current["pixels"] == saved["area"] and current["truth_overlap"] == saved["intersection"] and
                    type(raw) is int and type(supported) is int and 0 <= raw <= saved["area"] and
                    0 <= supported <= min(area, saved["area"]) and 0 <= current["mean_raw_rgb_difference"] <= 255 and
                    (raw != 0 or current["mean_raw_rgb_difference"] == 0), "Paired component measurements differ")
            if not target and hard:
                ordinary_hard.append(current)
                ordinary.update(hard_components=1, pixel_identical_components=int(raw == 0),
                                no_supported_difference_components=int(supported == 0),
                                supported_difference_components=int(supported > 0))
                if supported:
                    require(row["role"] == "local_copy" and supported == current["pixels"],
                            "Supported ordinary component differs from the known copy control")
        if ordinary_hard:
            hard_groups.add(key["group"])
            ordinary.update(hard_flagged_views=1, all_components_without_supported_difference=int(
                all(c["supported_earlier_copy_diff_pixels"] == 0 for c in ordinary_hard)))
    return {"positive": dict(positive) | {"useful_by_family": dict(positive_families)},
            "ordinary": dict(ordinary) | {"hard_flagged_source_groups": len(hard_groups),
                                         "earlier_copy_observations_by_role": dict(ordinary_roles)}}


def native_rows(args, confirmation, reflection):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import reference_compare, regional
    from scripts import image_regional_jpeg90_quality_confirmation as cases
    from scripts import image_regional_jpeg_transfer as prior

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    pins = confirmation["freeze"]["source_sha256"]
    for path, expected in ((args.main_roster, pins["main"]), (args.external_roster, pins["external"]),
                           (args.confirmation_roster / "plan.json", pins["confirmation_plan"]),
                           (args.confirmation_roster / "acquired.json", pins["confirmation_acquisition"]),
                           (prior.CHECKPOINT, reflection["freeze"]["checkpoint_sha256"])):
        require(digest(path) == expected, f"Native source or weight commitment differs: {path.name}")
    model = regional.load_model(prior.CHECKPOINT)
    output = []
    earlier, earlier_group = None, None
    for delivered, frozen, row in zip(cases.cases(args), confirmation["freeze"]["rows"], reflection["rows"][200:], strict=True):
        key, _, _, jpg, image, truth = delivered
        require(key == frozen and row["input_sha256"] == sha(jpg) and row["truth_sha256"] == sha(truth.tobytes()),
                "Frozen JPEG or changed-pixel mask differs")
        pixels = np.asarray(image)
        if key["role"] == "original":
            earlier, earlier_group = pixels.copy(), key["group"]
        require(earlier_group == key["group"], "Missing aligned earlier source view")
        with Image.open(io.BytesIO(jpg), formats=["JPEG"]) as carrier:
            compared, selection = reference_compare.compare_across_deliveries(earlier, pixels, carrier)
        require(selection is not None and compared["status"] == "comparable", "Earlier-copy alignment differs")
        area, target, overlap = int(selection.sum()), int(truth.sum()), int((selection & truth).sum())
        record = {"index": row["index"], "group": key["group"], "role": key["role"], "family": key["family"],
                  "truth_pixels": target, "earlier_copy": {"status": "comparable", "pixels": area, "overlap": overlap,
                                                           "useful": useful(area, overlap, target)},
                  "native_components": []}
        if row["components"]:
            out = prior.native_readout(model.predict(image), jpg, pixels)
            require(sha(out.source_mask(*image.size).tobytes()) == row["native_mask_sha256"] and
                    len(out.component_masks) == len(row["components"]), "Direct native mask failed exact replay")
            by_mask = {sha(component.tobytes()): component for component in out.component_masks}
            different = np.any(earlier != pixels, axis=2)
            raw_delta = np.abs(earlier.astype(np.int16) - pixels.astype(np.int16))
            for comp in row["components"]:
                mask = by_mask[comp["mask_sha256"]]
                require(int(mask.sum()) == comp["area"] and int((mask & truth).sum()) == comp["intersection"],
                        "Native component area or truth overlap differs")
                record["native_components"].append({
                    "mask_sha256": comp["mask_sha256"], "hard": 2 * comp["hard_overlap"] >= comp["area"],
                    "pixels": comp["area"], "truth_overlap": comp["intersection"],
                    "raw_diff_pixels": int((different & mask).sum()),
                    "mean_raw_rgb_difference": round(float(raw_delta[mask].mean()), 6),
                    "supported_earlier_copy_diff_pixels": int((selection & mask).sum()),
                })
        output.append(record)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("review", "native"))
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    for name in ("main-roster", "external-roster", "confirmation-roster"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    try:
        confirmation, reflection = read(CONFIRMATION), read(REFLECTION)
        pins = {"confirmation_sha256": digest(CONFIRMATION), "reflection_sha256": digest(REFLECTION)}
        if args.phase == "native":
            require(all((args.main_roster, args.external_roster, args.confirmation_roster)),
                    "Native replay needs the three frozen roster paths")
            rows = native_rows(args, confirmation, reflection)
        else:
            require(args.record is None, "Recording is available only with native replay")
            rows = read(args.receipt)["rows"]
        result = {"schema": SCHEMA, "input_ledgers": pins, "rows": rows, "summary": summarize(rows, confirmation, reflection)}
        if args.record:
            with args.record.open("x") as stream:
                json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
        else:
            require(result == read(args.receipt), "Paired source-context replay differs from the receipt")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(1, f"Reflection source-context review failed: {error}\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
