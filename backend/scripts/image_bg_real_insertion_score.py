"""Compare native PNG regional outputs on fresh licensed object insertions."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import image_crop_context_transfer as crop
from scripts import image_flux_product_comparison_v288 as previous
from scripts import image_relative_patch_transfer as shared

POINTS_SHA256 = "19ba3baa137d04fd51ff88d31040cd7c3c7cd1dc7350b5729dcf9c633424511a"
ROLES = ("original", "processed", "edited")


def checked_inputs(trial):
    intake = json.loads((trial / "result.json").read_bytes())
    plan = json.loads((trial / "plan.json").read_bytes())
    log = trial / "attempts.jsonl"
    shared.flux.require(shared.file_digest(log) == intake["acquisition_journal_sha256"]
                        and shared.file_digest(trial / "plan.json") == intake["plan_sha256"], "Intake changed")
    attempts = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(attempts) == len(plan["cases"]) == intake["planned"]
    assert sum(row["status"] == "qualified" for row in attempts) == intake["qualified"]
    return intake, plan, attempts


def panels(trial, plan, attempts):
    from scripts.image_bg_real_acquisition_v221 import edit_truth, png_pixels

    for case, row in zip(plan["cases"], attempts, strict=True):
        assert case["slot"] == row["slot"]
        if row["status"] != "qualified":
            continue
        raw, pixels = {}, {}
        for role in (*ROLES, "semantic_mask"):
            raw[role] = (trial / "assets" / f"{row['slot']:02d}-{role}.png").read_bytes()
            assert shared.digest(raw[role]) == row["files"][role]["sha256"]
            pixels[role] = png_pixels(raw[role], case["size"])
        truth, support = edit_truth(pixels["original"], pixels["edited"], pixels["semantic_mask"],
                                    case["semantic_edit_area"])
        assert int(truth.sum()) == row["exact_changed_pixels"] and support == row["semantic_mask_pixels"]
        assert int(np.any(pixels["original"] != pixels["processed"], axis=2).sum()) == row["processing_changed_pixels"]
        yield case["slot"], raw, pixels, truth


def measure(mask, truth):
    overlap = int((mask & truth).sum())
    predicted, target = int(mask.sum()), int(truth.sum())
    return {"reported": predicted, "target": target, "overlap": overlap,
            "iou": overlap / (predicted + target - overlap) if target else None,
            "recall": overlap / target if target else None}


def summarize(rows):
    sections = defaultdict(list)
    for row in rows:
        for arm, result in row["measurements"].items():
            sections[row["delivery"], arm, row["role"]].append(result)
    aggregate = defaultdict(dict)
    for (delivery, arm, role), values in sorted(sections.items()):
        item = {"views": len(values), "region_flags": sum(row["reported"] > 0 for row in values)}
        if role == "edited":
            item["useful_localizations"] = sum(row["iou"] >= .30 and row["recall"] >= .50 for row in values)
            item["mean_iou"] = float(np.mean([row["iou"] for row in values]))
            item["mean_recall"] = float(np.mean([row["recall"] for row in values]))
        aggregate[delivery].setdefault(arm, {})[role] = item
    return aggregate


def scores(trial, weights, point_file, output, stage):
    import torch

    from app.detectors.image import regional_support, splice
    from scripts.image_native_comparison_readout_v45 import source_mask

    torch.set_num_threads(2)
    points = json.loads(point_file.read_bytes())
    intake, plan, attempts = checked_inputs(trial)
    checkpoint = shared.flux.ROOT / previous.SPLICE
    identity = {
        "schema": "chipmunk.image-bg-real-insertion-score/1",
        "script_sha256": shared.file_digest(Path(__file__)),
        "intake_sha256": shared.file_digest(trial / "result.json"),
        "crop_sha256": shared.file_digest(Path(crop.__file__)),
        "receipt_sha256": shared.file_digest(Path(crop.independent.__file__)),
        "checkpoint_sha256": shared.file_digest(checkpoint),
        "crop_weight_sha256": shared.file_digest(weights),
        "prior_points_sha256": shared.file_digest(point_file),
        "operating_points": points,
        "reference_route": previous.ROUTES[0],
        "format_scope": {"png": ["current", *crop.ARMS, "current_or_relative"], "jpeg75": list(crop.ARMS)},
        "planned": intake["planned"],
        "qualified": intake["qualified"],
    }
    assert identity["checkpoint_sha256"] == previous.SPLICE_SHA
    assert identity["crop_weight_sha256"] == shared.WEIGHT_SHA
    assert identity["prior_points_sha256"] == POINTS_SHA256
    if stage == "freeze":
        output.mkdir(mode=0o700)
        shared.write(output / "freeze.json", identity)
        return
    assert identity == json.loads((output / "freeze.json").read_bytes())
    start = time.monotonic()
    model = splice.load_model(checkpoint)
    scorer = crop.loader(weights)
    rows = []
    with (output / "observations.jsonl").open("x") as log:
        for slot, payloads, native, truth in panels(trial, plan, attempts):
            for role in ROLES:
                truth_mask = truth if role == "edited" else np.zeros_like(truth)
                image = Image.fromarray(native[role])
                raw_map = splice.predict(model, image)
                assert raw_map.shape == (384, 384) and np.isfinite(raw_map).all()
                gate = bool(np.count_nonzero(raw_map > splice.PIXEL_THRESHOLD) >= splice.MINIMUM_ACTIVE_CELLS)
                reference = regional_support.readout(raw_map, gate, splice.PIXEL_THRESHOLD, payloads[role],
                                                    native[role]).source_mask(image.width, image.height)
                replay = source_mask(raw_map, gate, splice.PIXEL_THRESHOLD, payloads[role], native[role], truth_mask)
                assert np.array_equal(reference, replay)
                jpeg_bytes, jpeg = shared.flux.previous.encode(image, 75)
                for delivery, prepared, file_sha in (("png", image, shared.digest(payloads[role])),
                                                       ("jpeg75", jpeg, shared.digest(jpeg_bytes))):
                    box, responses = scorer(prepared)
                    left, top, right, bottom = box
                    candidate = np.zeros_like(truth)
                    candidate[top:bottom, left:right] = True
                    regions = {arm: candidate if responses[arm] >= points[arm] else np.zeros_like(truth)
                               for arm in crop.ARMS}
                    if delivery == "png":
                        regions |= {"current": reference, "current_or_relative": reference | regions["relative_crop"]}
                    row = {"slot": slot, "role": role, "delivery": delivery, "input_sha256": file_sha,
                           "measurements": {arm: measure(mask, truth_mask) for arm, mask in regions.items()}}
                    rows.append(row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
            print(f"qualified group {slot} complete", flush=True)
    assert len(rows) == intake["qualified"] * len(ROLES) * 2
    shared.write(output / "result.json", {
        "schema": identity["schema"], "freeze_sha256": shared.file_digest(output / "freeze.json"),
        "observations_sha256": shared.file_digest(output / "observations.jsonl"),
        "planned_groups": intake["planned"], "assessed_groups": intake["qualified"],
        "excluded_before_inference": intake["excluded_reasons"],
        "baseline_forwards": len(rows) // 2, "candidate_crop_inputs": len(rows) * len(crop.STARTS)**2,
        "candidate_forward_batches": len(rows) * len(crop.STARTS),
        "elapsed_s": round(time.monotonic() - start, 3),
        "peak_rss_bytes_mac": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": summarize(rows),
        "serving_admitted": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for arg in ("trial", "weights", "points", "output"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    scores(args.trial, args.weights, args.points, args.output, args.stage)


if __name__ == "__main__":
    main()
