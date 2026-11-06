"""Freeze format-specific image review gates on consumed development data only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(rows: list[dict]) -> dict:
    rules = {}
    for condition in ("native", "jpeg75"):
        cases = [r for r in rows if r["condition"] == condition]
        authentic = [r for r in cases if r["label"] == 0]
        edited = [r for r in cases if r["label"] == 1]
        if len(authentic) < 20 or len(edited) != len(authentic):
            raise ValueError("At least twenty paired development sources are required")
        if any(r["status"] != "ok" for r in cases):
            raise ValueError("Development coverage is incomplete; no gate will be fitted")
        budget = math.floor(.05 * len(authentic))
        values = sorted((r["active_fraction"] for r in authentic
                         if r["ungated"]["predicted_positive_pixels"] > 0), reverse=True)
        minimum = max(.001, math.nextafter(values[budget], math.inf) if len(values) > budget else .001)
        retained = [r for r in edited if r["active_fraction"] >= minimum]
        rules[condition] = {"minimum_active_fraction": minimum,
                            "development_originals": len(authentic), "development_edits": len(edited),
                            "false_alert_budget": budget,
                            "originals_with_regions": sum(r["active_fraction"] >= minimum
                                                          and r["ungated"]["predicted_positive_pixels"] > 0
                                                          for r in authentic),
                            "edits_with_overlap": sum(r["ungated"]["intersection"] > 0 for r in retained),
                            "mean_edit_iou": sum(r["ungated"]["iou"] for r in retained) / len(edited)}
    return rules


def freeze(development: Path, plan: Path, output: Path) -> dict:
    policy = json.loads((development / "policy.json").read_text())
    rows = [json.loads(line) for line in (development / "predictions.jsonl").read_text().splitlines()]
    expected = {(group, label, condition) for group in policy["source_groups"]
                for label in (0, 1) for condition in ("native", "jpeg75")}
    if policy["section"] != "development" or len(rows) != len(expected) or {
            (r["source_id"], r["label"], r["condition"]) for r in rows} != expected:
        raise ValueError("Incomplete or ambiguous development predictions")
    confirmation = json.loads(plan.read_text())
    future = {r["source_id"] for family in confirmation["groups"].values() for r in family}
    if future & set(policy["source_groups"]):
        raise ValueError("Development and confirmation share a source")
    record = {"schema": "chipmunk.iml-vit-decision/1", "rules": fit(rows),
              "development_source_groups": policy["source_groups"],
              "development_policy_sha256": sha(development / "policy.json"),
              "development_predictions_sha256": sha(development / "predictions.jsonl"),
              "confirmation_plan_sha256": sha(plan),
              **{k: policy[k] for k in ("checkpoint_sha256", "encoder_sha256", "runtime_sha256")},
              "evaluation_sha256": sha(Path(__file__).with_name("evaluate_iml_vit.py")),
              "development_evaluation_sha256": policy["script_sha256"],
              "source_change": "Confirmation adds only frozen-decision and disjointness checks to development scorer; "
                               "model, preprocessing, geometry, quantization and thresholds are unchanged.",
              "adoption": "More edits overlap displayed regions and mean IoU improves, with at most one additional "
                          "authentic input highlighted per format against the same-input current check. "
                          "This evaluates localization only, not overall AI confidence."}
    with output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("confirmation_plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze(args.development, args.confirmation_plan, args.output), indent=2))
