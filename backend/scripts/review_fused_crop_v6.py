"""Reconstruct complete consumed-source crop results; never authorize adoption."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.detectors.image import fused, splice
from scripts.evaluate_iml_vit import input_bytes, write_new
from scripts.fused_crop_v6 import Crop, choose_gate, complement, stitch, validate
from scripts.localization_benchmark import sha, source
from scripts.score_fused_crop_v6 import code_pins, selected
from scripts.score_splice_localizers import score_map


def prepare(root: Path, selection: Path, output: Path) -> tuple[dict, list[dict]]:
    _, digest, items = selected(root, selection)
    policy = json.loads((output / "policy.json").read_text())
    expected = {(i["id"], c): i for i in items for c in ("native", "jpeg75")}
    rows = [json.loads(l) for l in (output / "predictions.jsonl").read_text().splitlines()]
    if (policy.get("schema") != "chipmunk.fused-crop-development/6"
            or policy.get("section") != "development" or policy["manifest_sha256"] != digest
            or policy["selection_sha256"] != sha(selection.read_bytes())
            or policy["code_sha256"] != code_pins()
            or policy["item_ids"] != [i["id"] for i in items]
            or policy["weights"] != {"fused": fused.CHECKPOINT_SHA256, "semantic": fused.BACKBONE_SHA256,
                                     "native_baseline": splice.CHECKPOINT_SHA256}
            or len(rows) != len(expected) or {(r["id"], r["condition"]) for r in rows} != expected.keys()):
        raise ValueError("Changed method or incomplete exact planned denominator")
    prepared = []
    for row in rows:
        item = expected[(row["id"], row["condition"])]
        if any(row[k] != item[k] for k in ("source_id", "family", "label", "image_sha256", "mask_sha256")):
            raise ValueError("Changed source identity or reference labels")
        if row["status"] not in {"ok", "failed", "not_applicable"}:
            raise ValueError("Unrecognized result status")
        if row["status"] != "ok":
            if not row.get("reason"):
                raise ValueError("Unexplained unassessed input")
            prepared.append(row)
            continue
        original, pixels, truth = source(root, item)
        data, rgb = input_bytes(pixels, original, row["condition"])
        if sha(data) != row["input_sha256"]:
            raise ValueError("Changed evaluated input bytes")
        name = row["arrays"]
        if Path(name).name != name or sha((output / name).read_bytes()) != row["arrays_sha256"]:
            raise ValueError("Changed native crop arrays")
        with np.load(output / name, allow_pickle=False) as saved:
            if set(saved.files) != {"baseline", "crop0", "crop1", "crop2", "crop3"}:
                raise ValueError("Changed crop count or missing baseline map")
            crops = [Crop(tuple(c["box"]), saved[f"crop{i}"].copy(), c["image_score"], c["rgb_sha256"])
                     for i, c in enumerate(row["crops"])]
            validate(crops, rgb.shape[1], rgb.shape[0])
            image = Image.fromarray(rgb)
            if any(hashlib.sha256(np.asarray(image.crop(c.box)).tobytes()).hexdigest() != c.rgb_sha256
                   for c in crops):
                raise ValueError("Crop coordinates do not identify the scored native RGB input")
            baseline = saved["baseline"]
            if row["condition"] == "native":
                gate = int((baseline > splice.PIXEL_THRESHOLD).sum()) >= splice.MINIMUM_ACTIVE_CELLS
                threshold = splice.PIXEL_THRESHOLD
            else:
                gate = fused.candidate_eligibility(row["baseline_image_score"])["eligible"]
                threshold = 0.5
            prior = score_map(baseline, gate, threshold, data, rgb, truth)["metrics"]
        if prior != row["baseline"]:
            raise ValueError("Recomputed baseline display differs from saved measurements")
        prepared.append({**row, "crops": crops, "data": data, "rgb": rgb, "truth": truth})
    return policy, prepared


def extra(row: dict, gate: float) -> dict:
    response = stitch(row["crops"], row["rgb"].shape[1], row["rgb"].shape[0], gate)
    return score_map(response, True, 0.5, row["data"], row["rgb"], row["truth"])["metrics"]


def summarize(rows: list[dict]) -> dict:
    result = {}
    edits = [r for r in rows if r["label"] == 1]
    originals = [r for r in rows if r["label"] == 0]
    for method in ("baseline", "candidate"):
        result[method] = {
            "originals": len(originals), "edits": len(edits),
            "unassessed": sum(r[method] is None for r in rows),
            "originals_with_regions": sum(r[method] is not None and r[method]["predicted_positive_pixels"] > 0
                                          for r in originals),
            "edits_with_overlap": sum(r[method] is not None and r[method]["intersection"] > 0 for r in edits),
            "mean_edit_iou": sum(r[method]["iou"] for r in edits if r[method] is not None) / len(edits),
            "mean_edit_pixel_f1": sum(r[method]["pixel_f1"] for r in edits if r[method] is not None) / len(edits),
        }
    before, after = result["baseline"], result["candidate"]
    result["development_count_tradeoff"] = (
        after["edits_with_overlap"] > before["edits_with_overlap"]
        and after["mean_edit_iou"] > before["mean_edit_iou"]
        and after["originals_with_regions"] <= before["originals_with_regions"] + 1
        and after["unassessed"] <= before["unassessed"])
    delta = [(r["candidate"]["iou"] if r["candidate"] else 0)
             - (r["baseline"]["iou"] if r["baseline"] else 0) for r in edits]
    draws = np.random.default_rng(20261106).choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
    result["paired_iou_difference"] = {"mean": float(np.mean(delta)),
                                      "source_bootstrap_95_percent": np.quantile(draws, [.025, .975]).tolist()}
    result["crop_failures"] = sum(r["crop_status"] == "failed" for r in rows)
    result["crop_unassessed"] = sum(r["crop_status"] != "ok" for r in rows)
    return result


def review(root: Path, selection: Path, maps: Path, output: Path) -> dict:
    policy, rows = prepare(root, selection, maps)
    if any(r["status"] != "ok" for r in rows):
        raise ValueError("Preserve incomplete extraction; no development gate selection is permitted")
    rules = {}
    for condition in ("native", "jpeg75"):
        originals = [r for r in rows if r["condition"] == condition and r["label"] == 0]
        rules[condition] = choose_gate(originals, lambda r, g: extra(r, g)["predicted_positive_pixels"] > 0)
    cases = []
    for row in rows:
        added = extra(row, rules[row["condition"]]["image_gate"])
        candidate, used = complement(row["baseline"], added)
        cases.append({**{k: row[k] for k in ("id", "source_id", "family", "label", "condition")},
                      "baseline": row["baseline"], "candidate": candidate,
                      "used_crops": used, "crop_status": row["status"]})
    result = {
        "schema": "chipmunk.fused-crop-development-review/6", "section": "development",
        "policy_sha256": sha((maps / "policy.json").read_bytes()),
        "predictions_sha256": sha((maps / "predictions.jsonl").read_bytes()), "rules": rules,
        "conditions": {c: summarize([r for r in cases if r["condition"] == c]) for c in ("native", "jpeg75")},
        "families": {f: {c: summarize([r for r in cases if r["condition"] == c and r["family"] == f])
                          for c in ("native", "jpeg75")} for f in sorted({r["family"] for r in cases})},
        "cases": cases, "may_enable_service": False,
        "limits": "Consumed development only; no independent accuracy or adoption evidence. "
                  "Cropping changes model context; these are uncalibrated review cues. Source-paired RGB-difference "
                  "references measure changed pixels, not independently verified semantic editing boundaries. "
                  "Formats share source groups; a separate source-disjoint frozen confirmation is required.",
    }
    write_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_root", type=Path)
    parser.add_argument("consumed_selection", type=Path)
    parser.add_argument("maps", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.manifest_root, args.consumed_selection, args.maps, args.output)["conditions"], indent=2))
