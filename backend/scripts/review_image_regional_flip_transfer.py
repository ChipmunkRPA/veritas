"""Rebuild image input commitments, masks and decisions without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops
from scipy import ndimage

from scripts import image_edge_spread_transfer as sources

STORED = ("native", "mirrored", "recompression")
METHODS = (*STORED, "shared_pixels", "supported_components")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def prepared_digest(photo):
    """Rebuild the recorded CHW model input using independent per-channel planes."""
    width, height = photo.size
    side = max(width, height)
    if side > 1024:
        size = (max(1, width * 1024 // side), max(1, height * 1024 // side))
        photo = photo.resize(size, Image.Resampling.BILINEAR)
    inputs = []
    for plane, mean, std in zip(photo.split(), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), strict=True):
        values = np.zeros((1024, 1024), np.float32)
        values[:photo.height, :photo.width] = np.asarray(plane)
        inputs.append((values / np.float32(255) - np.float32(mean)) / np.float32(std))
    return digest(np.stack(inputs).tobytes())


def restore_masks(path, row):
    body = path.read_bytes()
    pixels = math.prod(row["shape"])
    stride = (pixels + 7) // 8
    require(len(body) == len(STORED) * stride and digest(body) == row["mask_bytes_sha256"],
            "Stored mask bytes or dimensions changed")
    values = np.unpackbits(np.frombuffer(body, dtype=np.uint8).reshape(len(STORED), stride), axis=1)
    require(not values[:, pixels:].any(), "Packed mask has nonzero padding")
    masks = dict(zip(STORED, values[:, :pixels].reshape(len(STORED), *row["shape"]).astype(bool), strict=True))
    native, mirrored = masks["native"], masks["mirrored"]
    shared = native & mirrored
    labels, number = ndimage.label(native, structure=np.ones((3, 3), dtype=bool))
    supported = np.zeros_like(native)
    for label, extent in enumerate(ndimage.find_objects(labels, max_label=number), 1):
        component = labels[extent] == label
        if np.count_nonzero(shared[extent] & component) >= (np.count_nonzero(component) + 1) // 2:
            supported[extent] |= component
    return {**masks, "shared_pixels": shared, "supported_components": supported}


def measured(mask, truth):
    selected = int(np.count_nonzero(mask))
    target = int(np.count_nonzero(truth))
    overlap = int(np.count_nonzero(mask[truth]))
    return {"reported": selected, "target": target, "overlap": overlap,
            "iou": overlap / (selected + target - overlap) if target else None,
            "recall": overlap / target if target else None}


def useful(m):
    return (m["target"] > 0 and 10 * m["overlap"] >= 3 * (m["reported"] + m["target"] - m["overlap"])
            and 2 * m["overlap"] >= m["target"])


def review_counts(rows, arms, rule):
    grouped = {}
    for key, panel_rows in rows.items():
        ordinary = [row for row in panel_rows if row["role"] in sources.CONTROLS]
        edits = [row for row in panel_rows if row["role"] in sources.POSITIVE]
        require(len(ordinary) == 72 and len(edits) == 24, "Missing genuine or edited views")
        counts = {}
        for arm in arms:
            retained = [row for row in edits if useful(row["measurements"][arm])]
            alarms = [row for row in ordinary if row["measurements"][arm]["reported"] > 0]
            counts[arm] = {
                "useful_sharp": sum(row["role"] == "edited" for row in retained),
                "useful_feathered": sum(row["role"] == "feathered_edit" for row in retained),
                "useful": len(retained),
                "useful_by_editor": {editor: sum(row["generator"] == editor for row in retained)
                                     for editor in {row["generator"] for row in edits}},
                "ordinary_flags": len(alarms),
                "ordinary_groups": len({row["slot"] for row in alarms}),
                "untouched_flags": sum(row["role"] == "original" for row in alarms),
                "ordinary_by_treatment": {role: sum(row["role"] == role for row in alarms) for role in sources.CONTROLS},
                "removed_native_useful": sum(row not in retained and useful(row["measurements"]["native"]) for row in edits),
            }
        candidate = counts["supported_components"]
        passes = (candidate["useful"] >= counts["native"]["useful"] * rule["minimum_native_useful_retention"]
                  and candidate["useful"] - counts["recompression"]["useful"] >= rule["minimum_additional_useful_over_recompression_per_panel"]
                  and candidate["ordinary_flags"] <= counts["native"]["ordinary_flags"] * (1 - rule["minimum_ordinary_relative_reduction"]))
        grouped[key] = {"arms": counts, "passes": passes}
    return grouped


def audit(run, earlier, later, prior_later=None):
    freeze_path, journal = run / "freeze.json", run / "rows.json"
    freeze, rows, result = read(freeze_path), read(journal), read(run / "result.json")
    require(digest(freeze_path.read_bytes()) == result["freeze_sha256"]
            and digest(journal.read_bytes()) == result["rows_sha256"], "Experiment receipt differs")
    pending = {(r["panel"], r["slot"], r["role"]): r for r in rows}
    require(len(pending) == len(rows) == result["views"] == 192 and result["inference_calls"] == 576,
            "A view or inference slot is missing")
    previous = {}
    if prior_later is not None:
        prior_journal, prior_result = prior_later / "rows.json", read(prior_later / "result.json")
        require(digest(prior_journal.read_bytes()) == prior_result["rows_sha256"], "Earlier study was altered")
        previous = {(r["slot"], r["role"]): r for r in read(prior_journal)}
        require(len(previous) == 96, "Earlier study must have 96 measured views")
    repeated = 0
    verified = {"earlier": [], "later": []}
    for key, folder in (("earlier", earlier), ("later", later)):
        for file, expected in freeze["intakes"][key].items():
            require(digest((folder / file).read_bytes()) == expected, "Frozen source intake changed")
        for case, native, _ in sources.paired.inputs(folder):
            original = Image.fromarray(native["original"])
            for role, pixels in sources.native_variants(native, case["box"]).items():
                item = pending.pop((key, case["slot"], role))
                photo = Image.fromarray(pixels)
                reflected = photo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                source_bytes, _ = sources.paired.encoded(np.asarray(photo), "png")
                reflected_bytes, _ = sources.paired.encoded(np.asarray(reflected), "png")
                if role in sources.POSITIVE:
                    red, green, blue = ImageChops.difference(original, photo).split()
                    truth = np.asarray(ImageChops.lighter(red, ImageChops.lighter(green, blue))) != 0
                else:
                    truth = np.zeros((photo.height, photo.width), dtype=bool)
                require(digest(source_bytes) == item["input_sha256"] and digest(reflected_bytes) == item["reflected_sha256"]
                        and digest(truth.tobytes()) == item["truth_sha256"] and list(truth.shape) == item["shape"]
                        and item["generator"] == case["generator"], "Decoded input, delivery or truth differs")
                require(prepared_digest(photo) == item["native_model_input_sha256"]
                        and prepared_digest(reflected) == item["reflected_model_input_sha256"],
                        "A normalized model input differs from the recorded digest")
                masks = restore_masks(run / "masks" / f"{key}-{case['slot']:02d}-{role}.bin", item)
                require({arm: measured(mask, truth) for arm, mask in masks.items()} == item["measurements"],
                        "Native-pixel mask overlap differs")
                if key == "later" and prior_later is not None:
                    prior = previous.pop((item["slot"], role))
                    require(prior["input_sha256"] == item["input_sha256"] and prior["truth_sha256"] == item["truth_sha256"]
                            and prior["measurements"]["regional"] == item["measurements"]["native"]
                            and prior["measurements"]["recompression"] == item["measurements"]["recompression"],
                            "Native arms differ from the previous actual inference")
                    repeated += 2
                verified[key].append(item)
    require(not pending and not previous, "Some recorded cases were not source checked")
    require(review_counts(verified, METHODS, freeze["method"]["rule"]) == result["counts"],
            "Reported counts or eligibility differ")
    return {"source_views_verified": len(rows), "original_and_reflected_encodings_verified": 2 * len(rows),
            "normalized_model_inputs_verified": 2 * len(rows),
            "mask_measurements_verified": len(rows) * len(METHODS), "prior_detector_replays_matched": repeated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("--prior-later", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.earlier, args.later, args.prior_later), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
