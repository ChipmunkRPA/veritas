"""Export an allowlisted consumed-development receipt after its exact input audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.review_mouth_artifact import summarize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(catalogue_path, receipt_path, lock_path, audit_path, output):
    catalogue = json.loads(catalogue_path.read_text())
    header, *rows = [json.loads(line) for line in receipt_path.read_text().splitlines()]
    lock = json.loads(lock_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        catalogue["generated_manifest_sha256"] != header["manifest_sha256"]
        or lock["manifest_sha256"] != header["manifest_sha256"]
        or header["lock_sha256"] != digest(lock_path)
        or audit["score_receipt_sha256"] != digest(receipt_path)
        or audit["planned"] != audit["passed"]
        or audit["planned"] != len(rows)
    ):
        raise ValueError("Public review requires the exact complete source-audited comparison")
    keys = (
        "id",
        "group",
        "condition",
        "label",
        "sha256",
        "status",
        "model_inputs",
        "base_windows",
        "artifact_windows",
        "baseline_replay_parity",
        "roi_bindings",
    )
    body = {
        "schema": "chipmunk.mouth-artifact-public-review/5",
        "role": "Consumed development; not fresh confirmation, outside review or customer confidence",
        "catalogue_sha256": digest(catalogue_path),
        "source_score_receipt_sha256": digest(receipt_path),
        "source_audit_sha256": digest(audit_path),
        "frozen_recipe": lock,
        "inference": {
            k: header[k]
            for k in ("method", "weights", "candidate_device", "baseline_device", "torch_version")
        },
        "candidate_review_point": None,
        "cases": [{k: row[k] for k in keys if k in row} for row in rows],
        "separate_source_terms": catalogue["terms"],
        "limitations": [
            "Published inputs identify source positions; actual pixels and model execution need separate acquisition and replay.",
            "The standard-library reviewer checks reported windows, not completeness of window enumeration or track membership.",
            "Generic learned face responses do not establish causal mouth attribution, lip sync, or AI confidence.",
            "Eight consumed actors and one generator family; no independent reviewer has participated.",
        ],
    }
    import math

    controls = [
        r for r in body["cases"] if not r["label"] and r.get("artifact_windows") and r["status"] == "ok"
    ]
    if {r["condition"] for r in controls} == {
        "original",
        "local-light",
        "local-resample",
        "constant-audio-delay",
    }:
        body["candidate_review_point"] = math.nextafter(
            max(0.0, max(r["artifact_windows"][0]["response"] for r in controls)), math.inf
        )
    summary = summarize(catalogue, body)
    payload = json.dumps(body, indent=2, allow_nan=False) + "\n"
    if "/Users/" in payload or "PRIVATE KEY" in payload:
        raise ValueError("Unexpected private setup information in public allowlist")
    with output.open("x") as stream:
        stream.write(payload)
    print(
        json.dumps(
            {
                "public_receipt_sha256": digest(output),
                "cases": len(rows),
                "verified_source_inputs": audit["verified_frames"],
                "temporal_review": {k: summary[k] for k in ("baseline", "candidate")},
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalogue", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("lock", type=Path)
    p.add_argument("audit", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    export(a.catalogue, a.receipt, a.lock, a.audit, a.output)
