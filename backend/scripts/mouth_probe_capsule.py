"""Export/replay public source-separated fitting features without pretrained weights."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_vectors(rows, head):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate fitting identities")
    roles = {role: {r["group"] for r in rows if r["role"] == role} for role in ("fit", "point")}
    if (
        roles["fit"] & roles["point"]
        or roles["fit"] != set(head["fit_groups"])
        or roles["point"] != set(head["point_groups"])
    ):
        raise ValueError("Fitting and point-selection identities changed or overlap")
    for row in rows:
        if (
            len(row["features"]) != 768
            or hashlib.sha256(struct.pack("<768f", *row["features"])).hexdigest()
            != row["features_float32_sha256"]
        ):
            raise ValueError("A frozen little-endian feature vector changed")
        if row["role"] not in roles or row["label"] != int(row["condition"].startswith("ai-")):
            raise ValueError("A source role or class label changed")


def export(plan_path, receipt, head_path, audit_path, output):
    plan = json.loads(plan_path.read_text())
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    head = json.loads(head_path.read_text())
    audit = json.loads(audit_path.read_text())
    if (
        head["features_receipt_sha256"] != digest(receipt)
        or audit["feature_receipt_sha256"] != digest(receipt)
        or audit["passed"] != audit["planned"]
        or audit["passed"] != len(rows)
        or [r["id"] for r in audit["cases"]] != [r["id"] for r in rows]
        or any(r["status"] != "ok" for r in rows)
    ):
        raise ValueError("A complete exact-input-audited fitting receipt is required")
    source_keys = (
        "id",
        "group",
        "role",
        "download_url",
        "source_license",
        "sha256",
        "original_sha256",
        "frames",
        "selected_frames",
    )
    vector_keys = (
        "id",
        "group",
        "role",
        "condition",
        "label",
        "source_sha256",
        "sha256",
        "frame_index",
        "pts",
        "time_base",
        "features",
        "features_float32_sha256",
        "model_input_float32_sha256",
        "mouth_chip_rgb_sha256",
    )
    selected = [{k: row[k] for k in vector_keys} for row in rows]
    validate_vectors(selected, head)
    body = {
        "schema": "chipmunk.mouth-probe-fitting-capsule/6",
        "role": "Consumed fitting and review-point data, not independent validation",
        "source_plan_sha256": digest(plan_path),
        "source_features_sha256": digest(receipt),
        "source_audit_sha256": digest(audit_path),
        "original_head_sha256": digest(head_path),
        "feature_byte_order": "little",
        "sources": [{k: row[k] for k in source_keys} for row in plan["sources"]],
        "weights": header["weights"],
        "head": head,
        "cases": selected,
        "separate_data_terms": {
            "database": "ODbL 1.0",
            "contents": "Database Contents License 1.0",
            "attribution": "CREMA-D, CheyneyComputerScience; pinned source references in sources",
            "license_url": "https://github.com/CheyneyComputerScience/CREMA-D/blob/1658cd342dff90010aa843eaeebd53610a08b1dc/LICENSE.txt",
        },
        "limitations": [
            "Frozen generic mouth features and a fitted research head; no source media or pretrained vision weights included.",
            "Replaying a numerical fit does not independently certify pixel provenance, neural feature extraction or predictive accuracy.",
            "16 fitting actors and 4 distinct review-point actors from consumed studio footage; one editing family.",
            "No outside reviewer has participated; this head is not enabled or calibrated as AI confidence.",
        ],
    }
    payload = (json.dumps(body, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if b"/Users/" in payload or b"PRIVATE KEY" in payload:
        raise ValueError("Unexpected private setup data in the public allowlist")
    with output.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        compressed.write(payload)
    print(
        json.dumps(
            {
                "capsule_sha256": digest(output),
                "bytes": output.stat().st_size,
                "feature_rows": len(rows),
                "scope": "Reproducible numerical fit, not independent detection validation",
            }
        )
    )


def replay(capsule, expected_sha, output):
    import numpy as np

    from scripts.mouth_probe_v6 import train_head

    if digest(capsule) != expected_sha:
        raise ValueError("Capsule differs from the selected published digest")
    with gzip.open(capsule, "rt") as stream:
        data = json.load(stream)
    if data["schema"] != "chipmunk.mouth-probe-fitting-capsule/6" or data["feature_byte_order"] != "little":
        raise ValueError("Unknown fitting capsule schema or vector format")
    rows, expected = data["cases"], data["head"]
    validate_vectors(rows, expected)
    actual = train_head(rows)
    differences = {
        key: float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key]))))
        for key in ("mean", "scale", "coef", "intercept", "review_point")
    }
    passed = max(differences.values()) < 1e-6
    result = {
        "schema": "chipmunk.mouth-probe-fitting-replay/6",
        "capsule_sha256": expected_sha,
        "feature_rows": len(rows),
        "fit_actors": len(actual["fit_groups"]),
        "point_actors": len(actual["point_groups"]),
        "max_absolute_differences": differences,
        "passed": passed,
        "scope": "Local numerical refit; no pretrained model loaded and no outside reviewer or fresh accuracy implied",
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("export")
    for name in ("plan", "features", "head", "audit", "output"):
        e.add_argument(name, type=Path)
    r = sub.add_parser("replay")
    r.add_argument("capsule", type=Path)
    r.add_argument("output", type=Path)
    r.add_argument("--sha256", required=True)
    a = p.parse_args()
    if a.stage == "export":
        export(a.plan, a.features, a.head, a.audit, a.output)
    else:
        replay(a.capsule, a.sha256, a.output)
