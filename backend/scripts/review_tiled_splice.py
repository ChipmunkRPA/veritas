"""Freeze and review a complementary tiled-splice check, retaining every planned case."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from app.detectors.image import splice
from scripts.evaluate_iml_vit import input_bytes, write_new
from scripts.localization_benchmark import read_manifest, sha, source
from scripts.score_splice_localizers import score_map
from scripts.tiled_splice_candidate import Tile, stitch


def receipts(root: Path, manifest: dict, digest: str) -> tuple[dict, list[dict]]:
    policy = json.loads((root / "policy.json").read_text())
    rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
    expected = {(i["id"], c): i for i in manifest["items"] for c in ("native", "jpeg75")}
    if (policy["manifest_sha256"] != digest or len(rows) != len(expected)
            or {(r["id"], r["condition"]) for r in rows} != expected.keys()):
        raise ValueError("Tile receipt does not include its exact planned denominator")
    for row in rows:
        item = expected[(row["id"], row["condition"])]
        if any(row[k] != item[k] for k in ("source_id", "family", "label", "image_sha256", "mask_sha256")):
            raise ValueError("Changed tile source identity or labels")
        if row["status"] == "ok":
            name = row["tile_maps"]
            if Path(name).name != name or sha((root / name).read_bytes()) != row["tile_maps_sha256"]:
                raise ValueError("Tile maps changed after inference")
        elif row["status"] != "failed" or not row.get("reason"):
            raise ValueError("Unsupported or unexplained tile failure")
    return policy, rows


def baseline_jpeg(path: Path, manifest: dict, digest: str) -> dict[str, dict]:
    policy = json.loads((path.parent / "policy.json").read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {i["id"]: i for i in manifest["items"]}
    if (policy["manifest_sha256"] != digest or policy["condition"] != "jpeg75" or len(rows) != len(expected)
            or {r["id"] for r in rows} != expected.keys()):
        raise ValueError("JPEG reference differs from the same-input source panel")
    for row in rows:
        if any(row[k] != expected[row["id"]][k] for k in ("source_id", "label", "image_sha256", "mask_sha256")):
            raise ValueError("JPEG reference changed source labels or identity")
    return {r["id"]: r for r in rows}


def current(row: dict, jpeg: dict[str, dict]) -> dict | None:
    if row["condition"] == "native":
        return row.get("current_native")
    reference = jpeg[row["id"]]
    if reference["input_sha256"] != row["input_sha256"]:
        raise ValueError("JPEG reference and candidate input bytes differ")
    return reference.get("metrics") if reference["status"] == "ok" else None


def fit_gate(originals: list[tuple[dict, dict]]) -> dict:
    if len(originals) < 20:
        raise ValueError("At least twenty authentic development sources are required")
    if any(row["status"] != "ok" or metric is None for row, metric in originals):
        raise ValueError("Development reference and tile scoring must be complete")
    budget = max(1, math.floor(.025 * len(originals)))
    values = sorted((row["max_active_fraction"] for row, metric in originals
                     if metric["predicted_positive_pixels"] == 0 and row["tiles"]), reverse=True)
    minimum = max(splice.MINIMUM_ACTIVE_CELLS / splice.INPUT_SIZE**2,
                  math.nextafter(values[budget], math.inf) if len(values) > budget else 0.)
    return {"enabled": minimum <= 1., "minimum_active_fraction": min(1., minimum),
            "additional_authentic_budget": budget, "development_originals": len(originals),
            "current_authentic_flags": sum(metric["predicted_positive_pixels"] > 0 for _, metric in originals),
            "selection": "Smallest tile-active fraction at or above the existing relative area gate that admits at most "
                         "the declared additional original-source budget; edited labels do not choose the gate."}


def freeze(manifest_root: Path, development: Path, jpeg_path: Path, future_plan: Path, output: Path) -> dict:
    manifest, digest = read_manifest(manifest_root)
    policy, rows = receipts(development, manifest, digest)
    if policy["section"] != "development" or any(r["status"] != "ok" for r in rows):
        raise ValueError("Only complete development receipts can select the rule")
    jpeg = baseline_jpeg(jpeg_path, manifest, digest)
    plan = json.loads(future_plan.read_text())
    future = {i["source_id"] for family in plan["groups"].values() for i in family}
    if future & set(policy["source_groups"]):
        raise ValueError("Tile confirmation reuses a consumed development source")
    rules = {condition: fit_gate([(row, current(row, jpeg)) for row in rows
                                  if row["condition"] == condition and not row["label"]])
             for condition in ("native", "jpeg75")}
    result = {"schema": "chipmunk.splice-tiles-decision/1", "rules": rules,
              "development_sources": policy["source_groups"], "development_manifest_sha256": digest,
              "development_policy_sha256": sha((development / "policy.json").read_bytes()),
              "development_predictions_sha256": sha((development / "predictions.jsonl").read_bytes()),
              "jpeg_reference_sha256": sha(jpeg_path.read_bytes()),
              "confirmation_plan_sha256": sha(future_plan.read_bytes()),
              **{k: policy[k] for k in ("runtime_sha256", "scorer_sha256", "checkpoint_sha256")},
              "reviewer_sha256": sha(Path(__file__).read_bytes()),
              "preservation": "Keep the current component when it displays a region. Otherwise use at most two "
                              "regions from admitted tiles. No extra display budget or confidence weight.",
              "adoption": "On a new same-input panel: more overlapping edits and higher mean edited IoU, at most one "
                          "additional genuine image with a region, assessed separately for native and JPEG75."}
    write_new(output, result)
    return result


def measurements(manifest_root: Path, maps_root: Path, rows: list[dict], jpeg: dict, rules: dict) -> list[dict]:
    manifest, _ = read_manifest(manifest_root)
    source_rows = {i["id"]: i for i in manifest["items"]}
    out = []
    for index, row in enumerate(rows):
        original, pixels, truth = source(manifest_root, source_rows[row["id"]])
        data, rgb = input_bytes(pixels, original, row["condition"])
        if sha(data) != row["input_sha256"]:
            raise ValueError("Recomputed scoring input differs from its frozen bytes")
        prior = current(row, jpeg)
        rule = rules[row["condition"]]
        predicted, used = prior, False
        if (row["status"] == "ok" and rule["enabled"]
                and prior is not None and prior["predicted_positive_pixels"] == 0):
            with np.load(maps_root / row["tile_maps"], allow_pickle=False) as saved:
                if set(saved.files) != {f"tile{i}" for i in range(len(row["tiles"]))}:
                    raise ValueError("Tile count differs from stored native response arrays")
                tiles = [Tile(tuple(t["box"]), saved[f"tile{i}"])
                         for i, t in enumerate(row["tiles"])]
                if any(tile.active_fraction != info["active_fraction"]
                       for tile, info in zip(tiles, row["tiles"], strict=True)):
                    raise ValueError("Native tile area differs from its saved gate input")
                response = stitch(tiles, rgb.shape[1], rgb.shape[0], rule["minimum_active_fraction"])
            measured = score_map(response, True, splice.PIXEL_THRESHOLD, data, rgb, truth)["metrics"]
            if measured["predicted_positive_pixels"]:
                predicted, used = measured, True
        out.append({"id": row["id"], "source_id": row["source_id"], "label": row["label"],
                    "family": row["family"], "condition": row["condition"],
                    "baseline": prior, "candidate": predicted, "used_tiles": used,
                    "tile_status": row["status"], "scheduled_tiles": len(row.get("tiles", []))})
        if index == 0 or (index+1) % 48 == 0 or index == len(rows)-1:
            print(f"Measured tile review: {index+1}/{len(rows)}", flush=True)
    return out


def summary(rows: list[dict]) -> dict:
    result = {}
    for name in ("baseline", "candidate"):
        originals = [r for r in rows if not r["label"]]
        edits = [r for r in rows if r["label"]]
        result[name] = {"originals": len(originals), "edits": len(edits),
                        "unassessed": sum(r[name] is None for r in rows),
                        "originals_with_regions": sum(r[name] is not None and r[name]["predicted_positive_pixels"] > 0
                                                      for r in originals),
                        "edits_with_overlap": sum(r[name] is not None and r[name]["intersection"] > 0 for r in edits),
                        "mean_edit_iou": sum(r[name]["iou"] for r in edits if r[name] is not None) / len(edits),
                        "mean_edit_pixel_f1": sum(r[name]["pixel_f1"] for r in edits if r[name] is not None) / len(edits)}
    delta = [((r["candidate"]["iou"] if r["candidate"] else 0.) -
              (r["baseline"]["iou"] if r["baseline"] else 0.)) for r in rows if r["label"]]
    draws = np.random.default_rng(20261105).choice(delta, (5000, len(delta)), replace=True).mean(axis=1)
    result["paired_iou_difference"] = {"mean": float(np.mean(delta)),
                                      "source_bootstrap_95_percent": np.quantile(draws, [.025, .975]).tolist()}
    before, after = result["baseline"], result["candidate"]
    result["meets_count_tradeoff"] = (
        after["edits_with_overlap"] > before["edits_with_overlap"]
        and after["mean_edit_iou"] > before["mean_edit_iou"]
        and after["originals_with_regions"] <= before["originals_with_regions"] + 1
        and after["unassessed"] <= before["unassessed"])
    result["tile_failure_count"] = sum(r["tile_status"] != "ok" for r in rows)
    result["cases_with_scheduled_tiles"] = sum(r["scheduled_tiles"] > 0 for r in rows)
    return result


def review(manifest_root: Path, maps_root: Path, jpeg_path: Path, lock_path: Path, output: Path) -> dict:
    manifest, digest = read_manifest(manifest_root)
    policy, rows = receipts(maps_root, manifest, digest)
    lock = json.loads(lock_path.read_text())
    if (lock["reviewer_sha256"] != sha(Path(__file__).read_bytes())
            or lock["runtime_sha256"] != policy["runtime_sha256"]
            or lock["scorer_sha256"] != policy["scorer_sha256"]):
        raise ValueError("Review implementation differs from the frozen experiment")
    if policy["section"] == "confirmation":
        if (policy["lock_sha256"] != sha(lock_path.read_bytes())
                or lock["confirmation_plan_sha256"] != sha((manifest_root / "tgif-plan.json").read_bytes())
                or set(lock["development_sources"]) & set(policy["source_groups"])):
            raise ValueError("Confirmation differs from its frozen source split or rules")
    elif lock["development_predictions_sha256"] != sha((maps_root / "predictions.jsonl").read_bytes()):
        raise ValueError("Development receipt changed")
    jpeg = baseline_jpeg(jpeg_path, manifest, digest)
    measured = measurements(manifest_root, maps_root, rows, jpeg, lock["rules"])
    result = {"schema": "chipmunk.splice-tiles-review/1", "section": policy["section"],
              "manifest_sha256": digest, "lock_sha256": sha(lock_path.read_bytes()),
              "predictions_sha256": sha((maps_root / "predictions.jsonl").read_bytes()),
              "jpeg_reference_sha256": sha(jpeg_path.read_bytes()),
              "conditions": {c: summary([r for r in measured if r["condition"] == c]) for c in ("native", "jpeg75")},
              "families": {f: {c: summary([r for r in measured if r["family"] == f and r["condition"] == c])
                               for c in ("native", "jpeg75")} for f in sorted({r["family"] for r in measured})},
              "cases": measured,
              "limits": "Localization component only, not whole-file AI confidence or the complete UI packet policy. "
                        "Generator training overlap unknown. Formats share source groups. A failed additional tile check "
                        "retains only the prior assessed result and remains recorded as a tile failure."}
    for values in result["conditions"].values():
        values["qualifies_for_scoped_local_adoption"] = (
            policy["section"] == "confirmation" and values["meets_count_tradeoff"])
    write_new(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("freeze")
    select.add_argument("manifest_root", type=Path)
    select.add_argument("development", type=Path)
    select.add_argument("jpeg_reference", type=Path)
    select.add_argument("future_plan", type=Path)
    select.add_argument("output", type=Path)
    report = sub.add_parser("review")
    report.add_argument("manifest_root", type=Path)
    report.add_argument("predictions", type=Path)
    report.add_argument("jpeg_reference", type=Path)
    report.add_argument("lock", type=Path)
    report.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.manifest_root, args.development, args.jpeg_reference, args.future_plan, args.output)
        print(json.dumps(result["rules"], indent=2))
    else:
        result = review(args.manifest_root, args.predictions, args.jpeg_reference, args.lock, args.output)
        print(json.dumps(result["conditions"], indent=2))
