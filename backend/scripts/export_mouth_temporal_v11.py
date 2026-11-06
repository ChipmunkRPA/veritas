"""Export source-free, allowlisted fitting and inference arithmetic for reviewers."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts.export_mouth_context_v10 import safe
from scripts.liveportrait_mouth_benchmark import digest
from scripts.run_mouth_context_v10 import check_lock as parent_lock
from scripts.run_mouth_temporal_v11 import check_lock, checked_rows, load_rows, training_windows


def publish_exact(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Refusing to replace a different public artifact")
    else:
        with path.open("xb") as stream:
            stream.write(data)


def preparation(root, destination):
    lock = check_lock(root / "experiment/lock.json")
    plan = json.loads((root / "training-plan.json").read_text())
    public_plan = {
        k: v
        for k, v in plan.items()
        if k not in ("source_plan", "source_root", "task", "portrait_root", "speech_asset_manifest")
    }
    inputs = json.loads((root / "training-inputs/manifest.json").read_text())
    audit = json.loads((root / "training-source-audit.json").read_text())
    if audit["passed"] != 160 or audit["manifest_sha256"] != digest(root / "training-inputs/manifest.json"):
        raise ValueError("Every frozen fitting construction must pass before publication")
    for row in inputs["items"]:
        row.pop("path", None)
    prepared = {
        "original_plan_sha256": digest(root / "training-plan.json"),
        "original_manifest_sha256": digest(root / "training-inputs/manifest.json"),
        "original_lock_sha256": digest(root / "experiment/lock.json"),
        "lock": {
            k: v
            for k, v in lock.items()
            if k.endswith("_sha256")
            or k in ("schema", "method", "fit_groups", "feature_names", "deployment_eligible")
        },
        "plan": public_plan,
        "inputs": inputs,
        "source_audit": audit,
        "runtime": json.loads((root / "runtime-receipt.json").read_text()),
    }
    target = destination / "mouth-v11-preparation.json.gz"
    publish_exact(target, gzip.compress(safe(prepared), mtime=0))
    print(
        json.dumps({"preparation_sha256": digest(target), "cases": 160, "source_audit_passed": 160}),
        flush=True,
    )


def export(root, destination):
    experiment = root / "experiment"
    lock_path = experiment / "lock.json"
    lock = check_lock(lock_path)
    parent = parent_lock(Path(lock["parent_lock"]))
    head = json.loads((experiment / "head.json").read_text())
    summary = json.loads((experiment / "summary.json").read_text())
    source_audit = json.loads((root / "training-source-audit.json").read_text())
    if (
        source_audit["passed"] != 160
        or head["source_audit_sha256"] != digest(root / "training-source-audit.json")
        or summary["head_sha256"] != digest(experiment / "head.json")
    ):
        raise ValueError("Incomplete source audit or head/summary binding")
    fields = (
        "id",
        "group",
        "condition",
        "label",
        "role",
        "status",
        "sha256",
        "source_sha256",
        "source_url",
        "source_license",
        "family",
        "frames",
        "time_base",
        "pts",
        "edit_frame_range",
        "edit_time_range_s",
        "source_ranges",
        "source_clock_span",
        "current_reference",
        "heads",
        "measurements",
        "geometry_trace",
        "windows",
        "native_input_replay",
    )
    panels = {}
    private_fit = None
    for stage, count in (("fit", 160), ("calibration", 24), ("development", 88)):
        path = experiment / (stage + "-features.jsonl")
        rows = checked_rows(lock_path, path, stage, count)
        if stage == "fit":
            private_fit = rows
        panels[stage] = {
            "original_receipt_sha256": digest(path),
            "cases": [{k: r[k] for k in fields if k in r} for r in rows],
        }
    capsule = {
        "schema": "chipmunk.mouth-temporal-public-capsule/11",
        "lock": {
            k: v
            for k, v in lock.items()
            if k.endswith("_sha256")
            or k in ("schema", "method", "fit_groups", "feature_names", "deployment_eligible")
        },
        "original_lock_sha256": digest(lock_path),
        "head": {k: v for k, v in head.items() if k not in ("calibration_receipt", "fit_receipt")},
        "original_head_sha256": digest(experiment / "head.json"),
        "panels": panels,
        "fitting_windows": training_windows(private_fit),
        "source_audit": source_audit,
        "summary": summary,
    }
    _, absolute = load_rows(Path(parent["prior_scores"]["v8"]["path"]))
    absolute_point = json.loads(Path(lock["heads"]["v8"]["path"]).read_text())["review_point"]
    old_summary = json.loads(Path(parent["prior_summary"]).read_text())
    capsule["absolute_reference"] = {
        "review_point": absolute_point,
        "cases": [{"id": r["id"], "probe_windows": r["probe_windows"]} for r in absolute],
        "outcomes": old_summary["cases"]["candidate"],
    }
    target = destination / "mouth-v11-fitting-review.json.gz"
    with target.open("xb") as stream:
        stream.write(gzip.compress(safe(capsule), mtime=0))
    preparation(root, destination)
    print(
        json.dumps(
            {
                "capsule_sha256": digest(target),
                "bytes": target.stat().st_size,
                "cases": 272,
                "feature_variables": 49,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("destination", type=Path)
    p.add_argument("--preparation-only", action="store_true")
    a = p.parse_args()
    if a.preparation_only:
        preparation(a.root, a.destination)
    else:
        export(a.root, a.destination)
