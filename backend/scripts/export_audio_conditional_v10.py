"""Publish/review prospective metadata and exact quality JSONL, never private paths.

Standard-library-only source/role/code review. Does not prove descriptor values,
classifier accuracy, neural execution or outside participation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

FIELDS = (
    "schema",
    "prepared_sha256",
    "quality_sha256",
    "original_receipt_sha256",
    "original_projection_head_sha256",
    "original_responses_sha256",
    "method",
    "implementation_sha256",
    "runtime",
    "software",
    "weights",
    "serving_calibration_sha256",
    "serving_bundle",
    "role_counts",
    "reserved_confirmation_sha256",
    "fitting_started",
    "new_neural_forwards",
    "deployment_eligible",
)
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
SOURCE_SHA = "526a660d812c3afec959fb45946101d643181816e4b7c4248e93559280adbf83"
ORIGINAL = {
    "original_receipt_sha256": "315c07c1eec95553760a671e0480fb99bb53669e6f5679be0a73a1077cac512d",
    "original_projection_head_sha256": "411c4322ac35037073f2998ef0757d80c5bd348413d509b302bca90bcdcc7519",
    "original_responses_sha256": "7dff2eb4259f97731bfde0b6e1f3e38a0fb60d9920431212eeb8f9790e8fac9e",
    "reserved_confirmation_sha256": "14828469353f3b37a60da86f6588b817d946846303c275afa46e4f752bce4e26",
    "serving_calibration_sha256": "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def review(body, quality_raw, source, project=None):
    lock = body["lock"]
    if (
        set(lock) != set(FIELDS)
        or body["schema"] != "chipmunk.audio-conditional-public-freeze/10"
        or lock["schema"] != "chipmunk.audio-conditional-lock/10"
    ):
        raise ValueError("Exact public prospective schema/allowlist required")
    if any(lock[k] is not False for k in ("fitting_started", "new_neural_forwards", "deployment_eligible")):
        raise ValueError("Prospective freeze cannot claim a completed fitting or inference")
    if any(lock[k] != value for k, value in ORIGINAL.items()):
        raise ValueError("Original source/feature/projection/service commitments required")
    if lock["role_counts"] != COUNTS or hashlib.sha256(quality_raw).hexdigest() != lock["quality_sha256"]:
        raise ValueError("Exact consumed role matrix and measurement bytes required")
    header, *rows = [json.loads(line) for line in quality_raw.splitlines()]
    expected = source["variants"]["development"]
    if [r["id"] for r in rows] != [r["id"] for r in expected] or Counter(r["role"] for r in rows) != COUNTS:
        raise ValueError("Every corrected consumed measurement and role required")
    if (
        header["original_receipt_sha256"] != lock["original_receipt_sha256"]
        or header["original_responses_sha256"] != lock["original_responses_sha256"]
    ):
        raise ValueError("Measurement/reference lineage differs")
    for row, variant in zip(rows, expected, strict=True):
        if row["status"] != "ok" or any(row[k] != variant[k] for k in ("sha256", "pcm_sha256", "role")):
            raise ValueError("Every measurement source/status must be valid")
        for key in ("quality", "cached_probe_quality"):
            value = row[key]
            if key == "cached_probe_quality" and value is None:
                continue
            if (
                not isinstance(value, dict)
                or set(value) != {"flatness", "floor"}
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                    for v in value.values()
                )
            ):
                raise ValueError("Finite bounded original/probe descriptors required")
    params = lock["method"]["parameters"]
    if (
        lock["method"]["version"] != "audio-conditional-response/10.0"
        or set(lock["method"]["arms"]) != {"nonlinear", "quality", "quality_only"}
        or params["max_iter"] != 200
        or params["max_depth"] != 2
        or params["max_leaf_nodes"] != 4
        or params["early_stopping"] is not False
        or params["validation_fraction"] is not None
    ):
        raise ValueError("Fixed arm/learner/no-random-holdout design required")
    if project is not None:
        for name, value in lock["implementation_sha256"].items():
            path = project / name
            if not path.resolve().is_relative_to(project.resolve()) or digest(path) != value:
                raise ValueError("Frozen implementation source changed")
    return {
        "schema": "chipmunk.audio-conditional-public-preflight/10",
        "passed": True,
        "inputs": len(rows),
        "code_files": len(lock["implementation_sha256"]),
        "cached_probe_measurements": sum(r["cached_probe_quality"] is not None for r in rows),
        "scope": "Local public metadata/role/finite-descriptor/code review only; no waveform recomputation, neural forwards, accuracy or outside participation.",
    }


def export(root, folder, source_path):
    if (root / "trained").exists():
        raise ValueError("Cannot publish a prospective freeze after project fitting")
    private = root / "lock.json"
    lock = load(private)
    raw = (root / "quality.jsonl").read_bytes()
    body = {
        "schema": "chipmunk.audio-conditional-public-freeze/10",
        "private_lock_sha256": digest(private),
        "source_capsule_sha256": SOURCE_SHA,
        "source_capsule": "audio-v9-identity-plan.json.gz",
        "original_numeric_capsule": "audio-v9-development-review.json.gz",
        "original_numeric_capsule_sha256": "0546940a656ce0f718318975491ee041f2a429be15ac57d316e8e5e3ea6345c3",
        "lock": {k: lock[k] for k in FIELDS},
        "export_code_sha256": digest(Path(__file__)),
    }
    if digest(source_path) != SOURCE_SHA:
        raise ValueError("Original corrected public source bytes changed")
    result = review(body, raw, load(source_path), Path(__file__).resolve().parents[2])
    with (folder / "audio-v10-conditional-freeze.json").open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    quality_path = folder / "audio-v10-quality-inputs.jsonl.gz"
    with quality_path.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0))
    print(
        json.dumps(
            {
                **result,
                "freeze_sha256": digest(folder / "audio-v10-conditional-freeze.json"),
                "quality_capsule_sha256": digest(quality_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "review"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--source", type=Path, default=Path("docs/evaluation-panels/audio-v9-identity-plan.json.gz")
    )
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    if args.stage == "export":
        export(args.input, args.output, args.source)
    else:
        body = load(args.input)
        if digest(args.source) != SOURCE_SHA or body["source_capsule_sha256"] != SOURCE_SHA:
            raise ValueError("Pinned canonical source catalogue required")
        raw = gzip.decompress((args.input.parent / "audio-v10-quality-inputs.jsonl.gz").read_bytes())
        result = review(body, raw, load(args.source), args.project)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        print(json.dumps(result), flush=True)
