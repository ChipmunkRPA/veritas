"""Allowlisted source/time/context metadata, without customer data, media or private paths."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts.liveportrait_mouth_benchmark import digest
from scripts.run_mouth_context_v10 import check_lock
from scripts.run_mouth_learning_v8 import write


def safe(body):
    encoded = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
    if b"/Users/" in encoded or b"/private/" in encoded:
        raise ValueError("Private filesystem path in allowlisted public artifact")
    return encoded


def export(root, destination):
    experiment = root / "experiment"
    lock_path = experiment / "lock.json"
    lock = check_lock(lock_path)
    result = json.loads((experiment / "summary.json").read_text())
    native = json.loads((experiment / "native-input-audit.json").read_text())
    if native["passed"] != 112:
        raise ValueError("Complete calibration/development native-input replay required")
    if (
        result["score_receipt_sha256"] != digest(experiment / "development-scores.jsonl")
        or native["development_scores_sha256"] != result["score_receipt_sha256"]
        or native["calibration_scores_sha256"] != digest(experiment / "calibration-scores.jsonl")
        or result["points_sha256"] != digest(experiment / "points.json")
    ):
        raise ValueError("Score, point and native-audit receipt identities differ")
    for name, source in (
        ("mouth-v10-calibration-source-audit.json", root / "calibration-source-audit.json"),
        ("mouth-v10-native-input-audit.json", experiment / "native-input-audit.json"),
        ("mouth-v10-runtime.json", root / "runtime-receipt.json"),
        ("mouth-v10-initial-preparation-failure.json", root / "calibration-inputs/manifest.json"),
    ):
        text = source.read_text()
        safe(json.loads(text))
        (destination / name).write_text(text)
    points_path = experiment / "points.json"
    points = json.loads(points_path.read_text())
    public_points = {k: v for k, v in points.items() if k != "calibration_scores"}
    public_points["original_points_sha256"] = digest(points_path)
    safe(public_points)
    write(destination / "mouth-v10-points.json", public_points)
    public_lock = {
        k: v
        for k, v in lock.items()
        if k.endswith("_sha256")
        or k in ("schema", "method", "implementation_sha256", "candidate_priority", "deployment_eligible")
    }
    public_lock["heads"] = {
        arm: {k: v for k, v in entry.items() if k != "path"} for arm, entry in lock["heads"].items()
    }
    public_lock["prior_scores"] = {
        arm: {k: v for k, v in entry.items() if k != "path"} for arm, entry in lock["prior_scores"].items()
    }
    public_lock["development"] = [
        {k: v for k, v in entry.items() if k not in ("manifest", "receipt")} for entry in lock["development"]
    ]
    public_lock["original_lock_sha256"] = digest(lock_path)
    safe(public_lock)
    write(destination / "mouth-v10-lock.json", public_lock)
    plan = json.loads((root / "calibration-plan.json").read_text())
    safe({k: v for k, v in plan.items() if k not in ("source_plan", "source_root", "task")})
    write(
        destination / "mouth-v10-calibration-plan.json",
        {k: v for k, v in plan.items() if k not in ("source_plan", "source_root", "task")},
    )
    manifest = Path(lock["calibration_manifest"])
    public_inputs = json.loads(manifest.read_text())
    for row in public_inputs["items"]:
        row.pop("path", None)
    public_inputs["original_manifest_sha256"] = digest(manifest)
    (destination / "mouth-v10-calibration-inputs.json.gz").write_bytes(
        gzip.compress(safe(public_inputs), mtime=0)
    )
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
        "failure_reason",
    )
    for stage in ("calibration", "development"):
        receipt = experiment / (stage + "-scores.jsonl")
        header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
        body = {
            "schema": "chipmunk.mouth-context-public-responses/10",
            "original_receipt_sha256": digest(receipt),
            "header": header,
            "cases": [{k: row[k] for k in fields if k in row} for row in rows],
        }
        (destination / ("mouth-v10-" + stage + "-responses.json.gz")).write_bytes(
            gzip.compress(safe(body), mtime=0)
        )
    safe(result)
    (destination.parent / "mouth-context-v10-evaluation.json").write_text(
        (experiment / "summary.json").read_text()
    )
    print(
        json.dumps(
            {"public_calibration_cases": 24, "public_development_cases": 88, "all_eligible_windows": True}
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("destination", type=Path)
    a = p.parse_args()
    export(a.root, a.destination)
