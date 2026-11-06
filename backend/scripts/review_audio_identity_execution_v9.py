"""Export/review the prospective execution freeze using only the standard library.

Explicit allowlists omit private source/cache paths and original source IDs.
This verifies published metadata/code, not neural results or outside participation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

FIELDS = (
    "schema",
    "source_plan_sha256",
    "variants_sha256",
    "reserved_confirmation_sha256",
    "method",
    "implementation_sha256",
    "runtime",
    "weights",
    "serving_calibration_sha256",
    "serving_bundle",
    "cache_sha256",
    "cache_old_manifest_sha256",
    "cache_eligible_inputs",
    "native_cache_ids",
    "planned",
    "development_sources",
    "deployment_eligible",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def review(body, plan, original, project=None):
    if set(body) != {
        "schema",
        "private_lock_sha256",
        "corrected_capsule_sha256",
        "lock",
        "new_neural_inference_run",
        "fitting_started",
        "deployment_eligible",
    }:
        raise ValueError("Only the public execution allowlist may be exported")
    lock = body["lock"]
    if set(lock) != set(FIELDS) or body["schema"] != "chipmunk.audio-identity-public-execution/9":
        raise ValueError("Exact execution schema/allowlist required")
    if any(
        body[k] is not False for k in ("new_neural_inference_run", "fitting_started", "deployment_eligible")
    ):
        raise ValueError("Prospective freeze cannot claim an executed result")
    if (
        lock["schema"] != "chipmunk.audio-identity-execution-lock/9"
        or lock["deployment_eligible"] is not False
    ):
        raise ValueError("Corrected research execution required")
    if (
        lock["source_plan_sha256"] != plan["source_plan_sha256"]
        or lock["variants_sha256"] != plan["variants_sha256"]["development"]
        or lock["reserved_confirmation_sha256"] != plan["variants_sha256"]["confirmation"]
    ):
        raise ValueError("Execution differs from the corrected published sources")
    old = original["lock"]
    for key in ("method", "runtime", "weights", "serving_calibration_sha256", "serving_bundle"):
        if lock[key] != old[key]:
            raise ValueError("Unchanged statistical method/model/runtime/service required")
    if lock["cache_old_manifest_sha256"] != old["variants_sha256"]:
        raise ValueError("Cache lineage differs from the original manifest")
    if (
        lock["planned"] != 4248
        or lock["development_sources"] != 161
        or lock["cache_eligible_inputs"] != 1846
        or lock["cache_sha256"] != "68673bfb01c3e5b80609299c4455f3e436d723312f328536bacf95fbd2a7e6f4"
    ):
        raise ValueError("Corrected denominators and terminal cache required")
    rows = {r["id"]: r for r in plan["variants"]["development"]}
    selected = lock["native_cache_ids"]
    if len(selected) not in (18, 19) or selected != sorted(set(selected)) or not set(selected) <= set(rows):
        raise ValueError("Complete unique native selection required")
    cells = Counter((rows[i]["role"], rows[i]["label"], rows[i]["condition"]) for i in selected)
    if len(cells) != 18 or any(role != "fit" for role, _, _ in cells):
        raise ValueError("Every eligible fitting truth/condition cell required")
    if len(lock["implementation_sha256"]) < 49:
        raise ValueError("Complete implementation pins required")
    if project is not None:
        for name, expected in lock["implementation_sha256"].items():
            path = project / name
            if path.resolve().is_relative_to(project.resolve()) is False or digest(path) != expected:
                raise ValueError("Frozen implementation changed")
    return {
        "schema": "chipmunk.audio-identity-execution-review/9",
        "passed": True,
        "planned": 4248,
        "reserved": 1080,
        "native_checks": len(selected),
        "implementation_files": len(lock["implementation_sha256"]),
        "scope": "Metadata/code checks only; no neural execution, performance claim or outside participation.",
    }


def export(private_path, plan_path, original_path, output, project=None):
    private = load(private_path)
    body = {
        "schema": "chipmunk.audio-identity-public-execution/9",
        "private_lock_sha256": digest(private_path),
        "corrected_capsule_sha256": digest(plan_path),
        "lock": {k: private[k] for k in FIELDS},
        "new_neural_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
    }
    review(body, load(plan_path), load(original_path), project)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"public_execution_sha256": digest(output), "bytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "review"))
    parser.add_argument("lock", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    if args.stage == "export":
        result = export(args.lock, args.plan, args.original, args.output, args.project)
    else:
        body = load(args.lock)
        if body["corrected_capsule_sha256"] != digest(args.plan):
            raise ValueError("Corrected capsule bytes changed")
        result = review(body, load(args.plan), load(args.original), args.project)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result), flush=True)
