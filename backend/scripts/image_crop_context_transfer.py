"""Compare absolute and contextual crop scores on qualified regional image edits."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.bench import independent
from scripts import image_relative_patch_transfer as source

SIDE = 8
STARTS = (0, 4, 8, 12, 16)
ARMS = ("absolute_crop", "relative_crop")


def boundaries(length, start):
    return tuple((length * value + source.GRID - 1) // source.GRID for value in (start, start + SIDE))


def crops(image):
    for y in STARTS:
        top, bottom = boundaries(image.height, y)
        for x in STARTS:
            left, right = boundaries(image.width, x)
            box = (left, top, right, bottom)
            yield box, np.asarray(image.crop(box).resize((384, 384), Image.Resampling.BILINEAR))


def scores(logits):
    values = np.asarray(logits, dtype=np.float64)
    assert values.shape == (len(STARTS)**2,) and np.isfinite(values).all()
    peak = int(np.argmax(values))
    return peak, {ARMS[0]: float(values[peak]), ARMS[1]: float(values[peak] - np.median(values))}


def loader(weight):
    import torch

    from app.detectors.image import commfor

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = commfor.load_model(weight, "cpu")
    mean = torch.tensor(commfor.MEAN)[None, :, None, None]
    std = torch.tensor(commfor.STD)[None, :, None, None]

    def apply(image):
        positions, pixels = zip(*crops(image), strict=True)
        logits = []
        for start in range(0, len(pixels), 5):
            tensor = torch.from_numpy(np.stack(pixels[start:start + 5])).permute(0, 3, 1, 2).float() / 255
            logits.extend(commfor.logits(model, (tensor - mean) / std, "cpu"))
        peak, responses = scores(logits)
        return positions[peak], responses

    return apply


def identity(weight):
    return source.identity(weight) | {
        "schema": "chipmunk.image-crop-context-transfer/1",
        "crop_code_sha256": source.file_digest(Path(__file__)),
        "receipt_code_sha256": source.file_digest(Path(independent.__file__)),
        "crop_side": SIDE,
        "crop_starts": STARTS,
        "arms": ARMS,
        "views": {"development_controls": 32, "transfer": 60},
        "one_control_group_above_each_point": True,
    }


def observed(inputs, split, predict):
    rows = {}
    for group, encoding, kind, image, masks in inputs:
        if split == "development" and (group not in source.POINT_GROUPS or kind == "ai"):
            continue
        box, responses = predict(image)
        payload, _ = source.flux.previous.encode(image, 0)
        target = masks["ai"] if kind == "ai" else None
        left, top, right, bottom = box
        key = f"{split}/{group}"
        rows[key, encoding, kind] = {
            "id": f"{key}/{encoding}/{kind}",
            "group": key,
            "condition": encoding,
            "kind": kind,
            "input_sha256": source.digest(payload),
            "label": int(kind == "ai"),
            "box": box,
            "source_units": image.height * image.width,
            "target_units": int(target.sum()) if target is not None else 0,
            "candidate_overlap": int(target[top:bottom, left:right].sum()) if target is not None else 0,
            "responses": responses,
        }
        if kind == "copy_move":
            print(f"{split} {group} {encoding} complete", flush=True)
    return rows


def threshold(rows, arm):
    maxima = {}
    for row in rows.values():
        maxima[row["group"]] = max(maxima.get(row["group"], float("-inf")), row["responses"][arm])
    # The portable flag is inclusive, so use the next float above the second-highest group.
    return float(np.nextafter(sorted(maxima.values())[-2], np.inf))


def manifest_and_receipts(development, confirmation, arm, point):
    manifest = {"schema": independent.SCHEMA, "cases": []}
    predictions = {"development": [], "confirmation": []}
    for split, inputs in (("development", development), ("confirmation", confirmation)):
        for row in inputs.values():
            left, top, right, bottom = row["box"]
            value = row["responses"][arm]
            flag = value >= point
            manifest["cases"].append({
                "id": row["id"], "modality": "image", "split": split, "group": row["group"],
                "label": row["label"], "family": "flux-klein" if row["label"] else row["kind"],
                "condition": row["condition"],
                "source_sha256": inputs[row["group"], "source", "original"]["input_sha256"],
                "input_sha256": row["input_sha256"],
                "license": {"id": "CC BY 2.0 source; panel terms in v262/v285", "evaluation_permitted": True,
                            "redistribution_permitted": False},
                "localization_truth": {"unit": "pixels", "source_units": row["source_units"],
                                       "target_units": row["target_units"]},
            })
            predictions[split].append({
                "id": row["id"], "input_sha256": row["input_sha256"], "status": "ok", "response": value,
                "localization": {"status": "ok", "reported_units": (right-left) * (bottom-top) if flag else 0,
                                 "overlap_units": row["candidate_overlap"] if flag else 0},
            })
    receipt = {"manifest_sha256": independent.digest(manifest), "system": arm, "score_kind": "raw",
               "response_units": "classifier logit or logit difference", "weight_sha256": [source.WEIGHT_SHA],
               "implementation_sha256": source.file_digest(Path(__file__))}
    return manifest, {key: receipt | {"split": key, "predictions": values} for key, values in predictions.items()}


def readout(development, confirmation, arm, point):
    manifest, receipts = manifest_and_receipts(development, confirmation, arm, point)
    lock = independent.freeze(manifest, receipts["development"], thresholds={"image": point})
    summary = independent.summarize(manifest, receipts["confirmation"], lock)
    result = {}
    for row in summary["strata"]:
        encoding = row["condition"]
        samples = [item for item in confirmation.values() if item["condition"] == encoding]
        found = 0
        for item in samples:
            if not item["label"] or item["responses"][arm] < point:
                continue
            left, top, right, bottom = item["box"]
            overlap, target = item["candidate_overlap"], item["target_units"]
            found += overlap / ((right-left) * (bottom-top) + target - overlap) >= .30 and overlap / target >= .50
        result[encoding] = row | {"useful_localizations": found}
    return {"point": point, "summary_receipt_sha256": independent.digest(summary), "transfer": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("weight", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    locked = json.loads(json.dumps(identity(args.weight)))
    if args.action == "freeze":
        args.output.mkdir(mode=0o700)
        source.write(args.output / "freeze.json", locked)
        return
    assert locked == json.loads((args.output / "freeze.json").read_bytes())
    start = time.monotonic()
    predict = loader(args.weight)
    development = observed(source.lama_inputs(), "development", predict)
    assert len(development) == 32
    points = {arm: threshold(development, arm) for arm in ARMS}
    source.write(args.output / "points.json", points)
    confirmation = observed(source.flux_inputs(), "confirmation", predict)
    assert len(confirmation) == 60
    source.write(args.output / "result.json", {
        "schema": locked["schema"], "status": "complete",
        "freeze_sha256": source.file_digest(args.output / "freeze.json"),
        "points_sha256": source.file_digest(args.output / "points.json"),
        "image_views": len(development) + len(confirmation),
        "crop_inputs": (len(development) + len(confirmation)) * len(STARTS)**2,
        "batches": (len(development) + len(confirmation)) * len(STARTS),
        "arms": {arm: readout(development, confirmation, arm, points[arm]) for arm in ARMS},
        "elapsed_s": round(time.monotonic() - start, 3),
        "peak_rss_bytes_mac": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "serving_admitted": False,
    })


if __name__ == "__main__":
    main()
