"""Audit complete matched receipts and summarize the frozen image-localizer trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.localization_benchmark import sha


def read_rows(root: Path, manifest: dict, digest: str, *, candidate: bool) -> tuple[dict, list[dict]]:
    policy = json.loads((root / "policy.json").read_text())
    if policy["manifest_sha256"] != digest:
        raise ValueError("Receipt belongs to a different source manifest")
    rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
    conditions = policy["conditions"] if candidate else [policy["condition"]]
    expected = {(i["id"], condition): i for i in manifest["items"] for condition in conditions}
    if len(rows) != len(expected) or {(r["id"], r["condition"]) for r in rows} != expected.keys():
        raise ValueError("Incomplete or ambiguous evaluation denominator")
    for row in rows:
        source = expected[(row["id"], row["condition"])]
        if any(row[k] != source[k] for k in ("source_id", "label", "family", "image_sha256", "mask_sha256")):
            raise ValueError("Receipt changed the known source labels or identity")
        if row["status"] not in {"ok", "failed", "not_applicable"}:
            raise ValueError("Unsupported evaluation status")
        if row["status"] == "ok":
            name = row["raw_map"]
            if Path(name).name != name or sha((root / name).read_bytes()) != row["raw_map_sha256"]:
                raise ValueError("Saved model map differs from its receipt")
            for field in ("ungated", "metrics"):
                metric = row[field]
                if not (0 <= metric["iou"] <= 1 and 0 <= metric["pixel_f1"] <= 1
                        and 0 <= metric["intersection"] <= metric["reference_positive_pixels"] <= metric["pixels"]
                        and 0 <= metric["predicted_positive_pixels"] <= metric["pixels"]):
                    raise ValueError("Invalid saved mask measurement")
    return policy, rows


def summarize(rows: list[dict], *, threshold: float | None = None) -> dict:
    originals = [r for r in rows if not r["label"]]
    edits = [r for r in rows if r["label"]]

    def metrics(row):
        if row["status"] != "ok" or (threshold is not None and row["active_fraction"] < threshold):
            return {"intersection": 0, "predicted_positive_pixels": 0, "iou": 0., "pixel_f1": 0.}
        return row["ungated"] if threshold is not None else row["metrics"]

    return {"originals": len(originals), "edits": len(edits),
            "assessed": sum(r["status"] == "ok" for r in rows),
            "failures": sum(r["status"] == "failed" for r in rows),
            "not_applicable": sum(r["status"] == "not_applicable" for r in rows),
            "originals_with_regions": sum(metrics(r)["predicted_positive_pixels"] > 0 for r in originals),
            "edits_with_overlap": sum(metrics(r)["intersection"] > 0 for r in edits),
            "missed_edits": sum(metrics(r)["intersection"] == 0 for r in edits),
            "mean_edit_iou": float(np.mean([metrics(r)["iou"] for r in edits])),
            "mean_edit_pixel_f1": float(np.mean([metrics(r)["pixel_f1"] for r in edits])),
            "median_runtime_s": float(np.median([r["runtime_s"] for r in rows if r["status"] == "ok"]))}


def paired(old: list[dict], new: list[dict], gate: float) -> dict:
    prior = {r["id"]: r for r in old}
    differences = []
    for row in new:
        before = prior[row["id"]]
        if row["input_sha256"] != before["input_sha256"]:
            raise ValueError("Baseline and candidate scored different input bytes")
        if row["label"]:
            first = before["metrics"]["iou"] if before["status"] == "ok" else 0.
            second = row["ungated"]["iou"] if row["status"] == "ok" and row["active_fraction"] >= gate else 0.
            differences.append(second - first)
    rng = np.random.default_rng(20261105)
    draws = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
    return {"metric": "source-paired mean edited IoU", "candidate_minus_current": float(np.mean(differences)),
            "source_bootstrap_95_percent": np.quantile(draws, [.025, .975]).tolist(),
            "source_groups": len(differences), "iterations": 5000, "seed": 20261105}


def run(manifest_path: Path, root: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    digest = sha(manifest_path.read_bytes())
    lock = json.loads((root / "frozen-decision.json").read_text())
    policy, candidate = read_rows(root / "confirmation", manifest, digest, candidate=True)
    if (policy["decision_lock_sha256"] != sha((root / "frozen-decision.json").read_bytes())
            or lock["confirmation_plan_sha256"] != sha((manifest_path.parent / "tgif-plan.json").read_bytes())
            or policy["script_sha256"] != lock["evaluation_sha256"]
            or policy["runtime_sha256"] != lock["runtime_sha256"]
            or policy["checkpoint_sha256"] != lock["checkpoint_sha256"]
            or set(lock["development_source_groups"]) & {r["source_id"] for r in candidate}):
        raise ValueError("The held-out trial does not match its frozen decision")
    result = {"schema": "chipmunk.iml-vit-comparison/1", "manifest_sha256": digest,
              "plan_sha256": lock["confirmation_plan_sha256"],
              "decision_sha256": policy["decision_lock_sha256"],
              "candidate_predictions_sha256": sha((root / "confirmation/predictions.jsonl").read_bytes()),
              "checkpoint_sha256": lock["checkpoint_sha256"], "conditions": {}}
    for condition in ("native", "jpeg75"):
        _, current = read_rows(root / f"current-{condition}", manifest, digest, candidate=False)
        selected = [r for r in candidate if r["condition"] == condition]
        gate = lock["rules"][condition]["minimum_active_fraction"]
        old, new = summarize(current), summarize(selected, threshold=gate)
        qualifies = (new["edits_with_overlap"] > old["edits_with_overlap"]
                     and new["mean_edit_iou"] > old["mean_edit_iou"]
                     and new["originals_with_regions"] <= old["originals_with_regions"] + 1)
        families = sorted({r["family"] for r in current})
        result["conditions"][condition] = {"current": old, "candidate": new,
              "paired": paired(current, selected, gate), "qualifies_for_local_adoption": qualifies,
              "families": {family: {"current": summarize([r for r in current if r["family"] == family]),
                         "candidate": summarize([r for r in selected if r["family"] == family], threshold=gate)}
                           for family in families}}
    result["decision"] = "No default changes; candidate remains offline unless a separately recorded scoped adoption is justified."
    result["limits"] = ("48 paired sources from one benchmark suite and three generators; training overlap unknown. "
                        "Pixel metrics measure the selected localization component, not all UI evidence packets or AI confidence. "
                        "JPEG copies share source groups; do not double independent sample size. Runtime includes concurrent "
                        "local experiments and is diagnostic, not a controlled hardware benchmark.")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.manifest, args.experiment, args.output)
    print(json.dumps({k: {"current": v["current"], "candidate": v["candidate"],
                          "qualifies": v["qualifies_for_local_adoption"]} for k, v in result["conditions"].items()}, indent=2))
