"""Standard-library prospective freeze export/review; no private source paths.

This checks commitments and unchanged source roles, not detector performance,
fitted parameters, biometric independence or native inference.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCE_SHA = "526a660d812c3afec959fb45946101d643181816e4b7c4248e93559280adbf83"
REFERENCE_SHA = "b73e5933deb418129f570500a90672e062d159ed84d39ddf8bc8744d2f13ff8c"
ORIGINAL = {
    "original_receipt_sha256": "315c07c1eec95553760a671e0480fb99bb53669e6f5679be0a73a1077cac512d",
    "original_projection_head_sha256": "411c4322ac35037073f2998ef0757d80c5bd348413d509b302bca90bcdcc7519",
    "original_responses_sha256": "7dff2eb4259f97731bfde0b6e1f3e38a0fb60d9920431212eeb8f9790e8fac9e",
    "conditional_responses_sha256": "a2a83749942f71df0ea923534537fae28a4b2e372a7a101b591e0097f262c06d",
    "reserved_confirmation_sha256": "14828469353f3b37a60da86f6588b817d946846303c275afa46e4f752bce4e26",
    "serving_calibration_sha256": "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
}
FIELDS = {
    "schema",
    "method",
    "implementation_sha256",
    "runtime",
    "weights",
    *ORIGINAL,
    "available_cached_probes",
    "role_counts",
    "serving_bundle",
    "fitting_started",
    "new_neural_forwards",
    "deployment_eligible",
}
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def review(body, source, reference, project=None):
    lock = body["lock"]
    if (
        body["schema"] != "chipmunk.audio-full-representation-public-freeze/11"
        or set(lock) != FIELDS
        or lock["schema"] != "chipmunk.audio-full-representation-lock/11"
    ):
        raise ValueError("Exact prospective schema and field allowlist required")
    if (
        body["source_capsule_sha256"] != SOURCE_SHA
        or body["reference_freeze_sha256"] != REFERENCE_SHA
        or any(lock[k] != v for k, v in ORIGINAL.items())
    ):
        raise ValueError("Original source/feature/calibration/probe commitments required")
    if any(lock[k] is not False for k in ("fitting_started", "new_neural_forwards", "deployment_eligible")):
        raise ValueError("Prospective freeze cannot claim fitting, inference or deployment")
    method = lock["method"]
    if (
        method["version"] != "audio-full-frozen-representation/11.0"
        or method["original_ridge"] != 0.001
        or method["residual_ridge"] != 0.05
        or method["intercept_ridge"] != 0.0
        or method["solver"] != {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000}
        or method["new_pretrained_checkpoint"] is not False
        or method["deployment_eligible"] is not False
    ):
        raise ValueError("Fixed full-representation objective and solver required")
    if any(
        lock[k] != reference["lock"][k]
        for k in ("weights", "runtime", "serving_bundle", "serving_calibration_sha256")
    ):
        raise ValueError("Original commercial checkpoints, runtime and service required")
    variants = source["variants"]["development"]
    if (
        Counter(r["role"] for r in variants) != COUNTS
        or lock["role_counts"] != COUNTS
        or lock["available_cached_probes"] != 424
    ):
        raise ValueError("Complete corrected consumed roles and cached probes required")
    for name, value in lock["implementation_sha256"].items():
        if project is not None:
            path = project / name
            if not path.resolve().is_relative_to(project.resolve()) or digest(path) != value:
                raise ValueError("Frozen implementation source bytes differ")
    if any(prefix in json.dumps(body) for prefix in ("/Users/", "/private/", "/home/")):
        raise ValueError("Private paths cannot enter a public freeze")
    return {
        "schema": "chipmunk.audio-full-representation-public-preflight/11",
        "passed": True,
        "inputs": len(variants),
        "implementation_files": len(lock["implementation_sha256"]),
        "fitting_started": False,
        "fresh_confirmation_started": False,
        "scope": "Local metadata/code-pin preflight only. No fitting, accuracy, native inference or outside human validation.",
    }


def export(path, destination, source_path, reference_path):
    if (path.parent / "trained").exists():
        raise ValueError("Cannot publish a prospective freeze after fitting")
    lock = load(path)
    if digest(source_path) != SOURCE_SHA or digest(reference_path) != REFERENCE_SHA:
        raise ValueError("Exact original public source/reference bytes required")
    body = {
        "schema": "chipmunk.audio-full-representation-public-freeze/11",
        "private_lock_sha256": digest(path),
        "source_capsule_sha256": SOURCE_SHA,
        "reference_freeze_sha256": REFERENCE_SHA,
        "lock": {k: lock[k] for k in sorted(FIELDS)},
    }
    result = review(body, load(source_path), load(reference_path), Path(__file__).resolve().parents[2])
    with destination.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({**result, "public_freeze_sha256": digest(destination)}), flush=True)


def tamper(body, source, reference, project):
    cases = []
    for kind in (
        "changed learner",
        "changed solver",
        "changed source receipt",
        "changed checkpoint",
        "changed role count",
        "fitting claim",
        "hidden path",
    ):
        changed = copy.deepcopy(body)
        lock = changed["lock"]
        if kind == "changed learner":
            lock["method"]["residual_ridge"] = 0.01
        elif kind == "changed solver":
            lock["method"]["solver"]["maxiter"] += 1
        elif kind == "changed source receipt":
            lock["original_receipt_sha256"] = "a" * 64
        elif kind == "changed checkpoint":
            lock["weights"] = []
        elif kind == "changed role count":
            lock["role_counts"]["fit"] -= 1
        elif kind == "fitting claim":
            lock["fitting_started"] = True
        else:
            changed["note"] = "/Users/placeholder/private-input.wav"
        try:
            review(changed, source, reference, project)
        except ValueError as error:
            cases.append({"alteration": kind, "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Altered actual freeze incorrectly passed: " + kind)
    return {
        "schema": "chipmunk.audio-full-representation-public-tamper-review/11",
        "passed": True,
        "cases": cases,
        "scope": "Deliberate local actual-freeze alterations; no fitting or outside participation.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("export", "review", "tamper"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--source", type=Path, default=Path("docs/evaluation-panels/audio-v9-identity-plan.json.gz")
    )
    p.add_argument(
        "--reference", type=Path, default=Path("docs/evaluation-panels/audio-v9-execution-freeze.json")
    )
    p.add_argument("--project", type=Path)
    a = p.parse_args()
    if a.stage == "export":
        export(a.input, a.output, a.source, a.reference)
    else:
        if digest(a.source) != SOURCE_SHA or digest(a.reference) != REFERENCE_SHA:
            raise ValueError("Exact original public source/reference bytes required")
        body, source, reference = load(a.input), load(a.source), load(a.reference)
        result = (
            tamper(body, source, reference, a.project)
            if a.stage == "tamper"
            else review(body, source, reference, a.project)
        )
        with a.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
