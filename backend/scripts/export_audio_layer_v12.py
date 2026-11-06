"""Standard-library prospective metadata/code review, never model inference.

The public field allowlist omits private roots. Checks declared source/runtime/
checkpoint/role commitments; does not establish feature values or accuracy.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_SHA = "526a660d812c3afec959fb45946101d643181816e4b7c4248e93559280adbf83"
REFERENCE_SHA = "b73e5933deb418129f570500a90672e062d159ed84d39ddf8bc8744d2f13ff8c"
ORIGINAL = {
    "original_receipt_sha256": "315c07c1eec95553760a671e0480fb99bb53669e6f5679be0a73a1077cac512d",
    "original_responses_sha256": "7dff2eb4259f97731bfde0b6e1f3e38a0fb60d9920431212eeb8f9790e8fac9e",
    "conditional_responses_sha256": "a2a83749942f71df0ea923534537fae28a4b2e372a7a101b591e0097f262c06d",
    "reserved_confirmation_sha256": "14828469353f3b37a60da86f6588b817d946846303c275afa46e4f752bce4e26",
    "serving_calibration_sha256": "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
    "native_plan_sha256": "48795283220152d5de07356ff54c018b8da797749957b6480d27582de97eed1e",
}
PREVIOUS = {
    "head_sha256": "bf2a363ad6e491ad4c4fac1ec18658c58bc01eb20788a2a198c198426d481607",
    "responses_sha256": "683f56f383244257ddb47edb1e3d5c87a3071db19df7d13c4cc707f48129bf0f",
    "summary_sha256": "cab6bf33a644774d565a5bc3f949143b3d3ab35c84c1e9e5425a84bef79bcdcf",
}
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
FIELDS = {
    "schema",
    "method",
    "implementation_sha256",
    "runtime",
    "weights",
    *ORIGINAL,
    "previous_control",
    "role_counts",
    "planned",
    "original_input_bytes_verified",
    "expected_unassessed",
    "available_prior_probe_references",
    "native_ids",
    "serving_bundle",
    "fitting_started",
    "new_neural_forwards",
    "deployment_eligible",
}
BODY_FIELDS = {"schema", "private_lock_sha256", "source_capsule_sha256", "reference_freeze_sha256", "lock"}
EXTRA_FILES = {
    "backend/scripts/audio_conditional_v10.py",
    "backend/scripts/audio_full_representation_v11.py",
    "backend/scripts/audio_layer_features_v12.py",
    "backend/scripts/audio_layer_response_v12.py",
    "backend/scripts/audit_audio_identity_forwards_v9.py",
    "backend/scripts/export_audio_full_representation_v11.py",
    "backend/scripts/export_audio_layer_v12.py",
    "backend/scripts/run_audio_conditional_v10.py",
    "backend/scripts/run_audio_full_representation_v11.py",
    "backend/scripts/run_audio_layer_v12.py",
    "docs/audio-conditional-v10-protocol.md",
    "docs/audio-full-representation-v11-protocol.md",
    "docs/audio-layer-v12-protocol.md",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def native_ids(rows):
    cells = defaultdict(list)
    for r in rows:
        if r["role"] == "development":
            cells[r["dataset"], r["generator"], r["label"], r["condition"]].append(r["id"])
    return sorted(min(ids) for ids in cells.values())


def review(body, source, reference, project=None):
    if set(body) != BODY_FIELDS or body["schema"] != "chipmunk.audio-layer-public-freeze/12":
        raise ValueError("Exact public prospective field allowlist required")
    lock = body["lock"]
    if set(lock) != FIELDS or lock["schema"] != "chipmunk.audio-layer-lock/12":
        raise ValueError("Exact layer execution field allowlist required")
    if (
        body["source_capsule_sha256"] != SOURCE_SHA
        or body["reference_freeze_sha256"] != REFERENCE_SHA
        or any(lock[k] != v for k, v in ORIGINAL.items())
        or lock["previous_control"] != PREVIOUS
    ):
        raise ValueError("Original source, service, reserved and full-vector control commitments required")
    if any(lock[k] is not False for k in ("fitting_started", "new_neural_forwards", "deployment_eligible")):
        raise ValueError("Prospective freeze cannot claim fitting, neural forwards or deployment")
    m = lock["method"]
    if (
        m["version"] != "audio-fixed-layer-response/12.0"
        or m["layer_ids"] != [8, 16, 24]
        or set(m["arms"]) != {"final_control", "layers"}
        or m["ridge"] != {"raw": 0.001, "representation": 0.05, "intercept": 0.0}
        or m["solver"] != {"method": "L-BFGS-B", "ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000}
        or m["reference_prior"] != 0.5
        or m["deployment_eligible"] is not False
    ):
        raise ValueError("Fixed preselected layers, objective and solver required")
    if any(
        lock[k] != reference["lock"][k]
        for k in ("weights", "runtime", "serving_bundle", "serving_calibration_sha256")
    ):
        raise ValueError("Original commercial checkpoints, runtime and service required")
    old_files = reference["lock"]["implementation_sha256"]
    if set(lock["implementation_sha256"]) != set(old_files) | EXTRA_FILES or any(
        lock["implementation_sha256"][name] != sha for name, sha in old_files.items()
    ):
        raise ValueError("Complete unchanged baseline and all new implementation files required")
    rows = source["variants"]["development"]
    if (
        Counter(r["role"] for r in rows) != COUNTS
        or lock["role_counts"] != COUNTS
        or lock["planned"] != 4248
        or lock["original_input_bytes_verified"] != 4248
        or lock["expected_unassessed"] != 62
        or lock["available_prior_probe_references"] != 450
        or lock["native_ids"] != native_ids(rows)
        or len(lock["native_ids"]) != 99
    ):
        raise ValueError("Complete original roles, statuses, references and native cells required")
    if any(prefix in json.dumps(body) for prefix in ("/Users/", "/private/", "/home/")):
        raise ValueError("Private filesystem roots cannot enter a public freeze")
    if project is not None:
        for name, sha in lock["implementation_sha256"].items():
            path = project / name
            if not path.resolve().is_relative_to(project.resolve()) or digest(path) != sha:
                raise ValueError("Frozen implementation source bytes changed")
    return {
        "schema": "chipmunk.audio-layer-public-preflight/12",
        "passed": True,
        "inputs": len(rows),
        "native_cells": len(lock["native_ids"]),
        "implementation_files": len(lock["implementation_sha256"]),
        "fitting_started": False,
        "new_neural_forwards": False,
        "scope": "Local declared metadata/code-pin check only, not waveform/model re-execution, feature verification, accuracy or outside human validation.",
    }


def export(path, destination, source_path, reference_path):
    if any((path.parent / name).exists() for name in ("representations.jsonl", "trained")):
        raise ValueError("Cannot publish a prospective freeze after source forwards or fitting")
    if digest(source_path) != SOURCE_SHA or digest(reference_path) != REFERENCE_SHA:
        raise ValueError("Exact original public source/reference bytes required")
    lock = load(path)
    body = {
        "schema": "chipmunk.audio-layer-public-freeze/12",
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
        "changed layer order",
        "changed regularization",
        "changed source receipt",
        "changed checkpoint",
        "changed missing count",
        "changed native coverage",
        "dropped implementation pin",
        "fitting claim",
        "hidden field",
    ):
        changed = copy.deepcopy(body)
        lock = changed["lock"]
        if kind == "changed layer order":
            lock["method"]["layer_ids"] = [24, 16, 8]
        elif kind == "changed regularization":
            lock["method"]["ridge"]["representation"] = 0.01
        elif kind == "changed source receipt":
            lock["original_receipt_sha256"] = "a" * 64
        elif kind == "changed checkpoint":
            lock["weights"] = {}
        elif kind == "changed missing count":
            lock["expected_unassessed"] = 0
        elif kind == "changed native coverage":
            lock["native_ids"].pop()
        elif kind == "dropped implementation pin":
            lock["implementation_sha256"].pop("backend/scripts/audio_layer_features_v12.py")
        elif kind == "fitting claim":
            lock["fitting_started"] = True
        else:
            changed["private_path"] = "/Users/placeholder/private-input.wav"
        try:
            review(changed, source, reference, project)
        except ValueError as error:
            cases.append({"alteration": kind, "rejected": True, "reason": str(error)})
        else:
            raise ValueError("Altered actual freeze incorrectly passed: " + kind)
    return {
        "schema": "chipmunk.audio-layer-public-tamper-review/12",
        "passed": True,
        "cases": cases,
        "scope": "Deliberate local actual-freeze alterations, not model execution or outside participation.",
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
            raise ValueError("Original public source/reference bytes required")
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
