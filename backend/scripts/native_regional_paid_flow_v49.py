"""Frozen real-detector app smoke: simulated prepaid checkout, queue, PDF/JSON and browser."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/native-regional-paid-flow-v49"
PUBLIC = ROOT / "docs/evaluation-panels/native-regional-paid-flow-v49-freeze.json"
CASES = ("candidate-small", "legacy-ai", "legacy-original", "candidate-ai", "candidate-original")
OWN = ("backend/scripts/native_regional_paid_flow_v49.py", "backend/scripts/native_regional_paid_flow_runtime_v49.py",
       "backend/tests/test_native_regional_paid_flow_v49.py", "backend/tests/test_stripe_billing.py",
       "frontend/e2e/native-regional-paid-flow-v49.mjs", "docs/native-regional-paid-flow-v49-protocol.md")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def prepare():
    from app.core.config import settings
    from app.engine.fetchers import file_path
    from app.engine.registry import get_registry
    from scripts.image_native_confirmation_v46 import encoded_view
    from scripts.image_regional_training_v26 import PAIRS, source_views
    from scripts.prepare_regional_calibration_v42 import runtime_versions

    parent = TRIAL.parent / "native-regional-serving-v47"
    need(read(parent / "summary.json")["exact_raw_maps"] == 236, "Completed serving parity required")
    plan, references = read(parent / "plan.json"), read(parent / "references.json")
    reference = references[0]["candidate"]
    case = next(r for r in plan["rows"] if r["source_id"] == reference["source_id"])
    qualification = next(r for r in map(json.loads, (PAIRS / "qualification/pairs.jsonl").read_text().splitlines())
                         if r["source_id"] == reference["source_id"])
    edited, controls, _ = source_views(case, qualification)
    payloads = {"ai": encoded_view(edited, "native")[0], "original": encoded_view(controls["original"], "native")[0]}
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (30, 50, 90)).save(buffer, "PNG")
    payloads["small"] = buffer.getvalue()
    TRIAL.mkdir(exist_ok=False)
    (TRIAL / "inputs").mkdir()
    inputs = {}
    for kind, data in payloads.items():
        filename = f"inputs/{kind}.png"
        with (TRIAL / filename).open("xb") as out:
            out.write(data)
        inputs[kind] = {"file": filename, "sha256": sha(TRIAL / filename), "bytes": len(data),
                        "role": "authored scope smoke" if kind == "small" else "consumed confirmation; engineering only"}
    write(TRIAL / "inputs.json", inputs)
    code_paths = [ROOT / p for p in OWN]
    for p in (ROOT / "backend/app").rglob("*"):
        if p.is_file() and p.suffix in {".py", ".json", ".yaml", ".j2", ".css", ".ttf", ".woff2"}:
            code_paths.append(p)
    for p in (ROOT / "frontend/dist").rglob("*"):
        if p.is_file():
            code_paths.append(p)
    code = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(code_paths))}
    registry = get_registry()
    enabled = {}
    before = settings.image_region_mode
    for mode in ("legacy", "candidate"):
        settings.image_region_mode = mode
        enabled[mode] = registry.enabled("image", "full")
    settings.image_region_mode = before
    model_ids = {m for d in enabled["candidate"] for m in registry.required_models(registry.manifest.detector(d))}
    assets, private_paths = [], []
    for model_id in sorted(model_ids):
        model = registry.manifest.model(model_id)
        for weight in model.source.files:
            try:
                path = file_path(model, weight)
            except Exception as error:
                assets.append({"model": model_id, "file": weight.path, "status": "unavailable", "error_type": type(error).__name__})
                continue
            if not path.is_file():
                assets.append({"model": model_id, "file": weight.path, "status": "missing"})
                continue
            digest = sha(path)
            need(weight.sha256 is None or weight.sha256 == digest, "Available model weight digest differs")
            assets.append({"model": model_id, "file": weight.path, "status": "available", "sha256": digest, "bytes": path.stat().st_size})
            private_paths.append({"path": str(path), "sha256": digest})
    write(TRIAL / "weight-paths.json", private_paths)
    frozen = {"schema": "chipmunk.native-regional-paid-flow/49", "code_sha256": code,
              "cases": list(CASES), "inputs_sha256": sha(TRIAL / "inputs.json"), "inputs": inputs,
              "weight_paths_sha256": sha(TRIAL / "weight-paths.json"), "assets": assets,
              "runtime_versions": runtime_versions(), "licence_policy": registry.policy.model_dump(),
              "enabled_detectors": enabled, "cpu_threads": 2, "device": "cpu", "model_memory_budget_mb": 4096,
              "stripe_transport": "isolated test double; USD 9.99 paid before queue and inference",
              "real_detectors_and_production_pipeline": True, "model_or_threshold_tuning": False,
              "parent_parity_summary_sha256": sha(parent / "summary.json"), "before_source_forwards": True,
              "default_changed": False, "new_accuracy_claim": False,
              "criteria": ["No analysis/model load/forward before paid confirmation", "All enabled checks recorded",
                           "Native and legacy checks complete on supported PNG inputs", "Small input never becomes original",
                           "Same whole-file authorship numbers in legacy/candidate paired flows",
                           "Identical evidence crops/masks in API/export/report snapshot", "All declared assets hash-verified",
                           "Private model names absent in public API/JSON/PDF/UI", "Ordinary browser view loads source crops and mask",
                           "Visual PDF inspection after the automatic check", "Every monitored phase passes; preserve failures"]}
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    print(json.dumps({"cases": len(CASES), "code_files": len(code), "assets": len(assets), "neural_calls": 0}), flush=True)


def admission():
    from scripts.prepare_regional_calibration_v42 import runtime_versions

    frozen = read(TRIAL / "freeze.json")
    start = read(TRIAL / "execution-start.json")
    need(sha(PUBLIC) == sha(TRIAL / "freeze.json") == start["freeze_sha256"]
         and start["remote_verified_before_first_source_forward"] is True
         and len(start["remote_freeze_blob_sha"]) == 40, "Remote exact freeze required")
    need(all(sha(ROOT / p) == digest for p, digest in frozen["code_sha256"].items()), "Frozen app/test code changed")
    need(runtime_versions() == frozen["runtime_versions"], "Runtime changed")
    need(sha(TRIAL / "inputs.json") == frozen["inputs_sha256"]
         and sha(TRIAL / "weight-paths.json") == frozen["weight_paths_sha256"], "Private bindings changed")
    for item in frozen["inputs"].values():
        need(sha(TRIAL / item["file"]) == item["sha256"], "Frozen input changed")
    profile_path = TRIAL.parent / (TRIAL.name + "-preparation-profile/profile.json")
    need(sha(profile_path) == start["preparation_profile_sha256"], "Preparation proof changed")
    profile = read(profile_path)
    need(profile["child_exit_code"] == 0 and profile["guard"] is None and profile["sample_errors"] == 0
         and not profile["resource_limit_observations"], "Bounded successful preparation required")
    return frozen


def phase(number):
    import pytest

    admission()
    need(0 <= number < len(CASES), "Explicit flow required")
    if number:
        prior = read(TRIAL / f"phase-{number - 1:03d}/completed.json")
        profile = read(TRIAL / f"phase-{number - 1:03d}-profile/profile.json")
        need(prior["status"] == "complete" and profile["child_exit_code"] == 0 and profile["guard"] is None
             and profile["sample_errors"] == 0 and not profile["resource_limit_observations"], "Previous flow must pass")
    folder = TRIAL / f"phase-{number:03d}"
    folder.mkdir(exist_ok=False)
    os.environ["CHIPMUNK_REGIONAL_FLOW_CASE"] = CASES[number]
    result = pytest.main([f"tests/test_native_regional_paid_flow_v49.py::test_real_prepaid_image_flow[{CASES[number]}]",
                          "-o", "addopts=", "-m", "slow", "-q", "--basetemp", str(folder / "work"),
                          "--junitxml", str(folder / "pytest.xml")])
    if result:
        raise RuntimeError("Full flow failed; preserve attempt, no automatic retry")
    need(read(folder / "completed.json")["case"] == CASES[number], "Complete explicit case receipt required")


def seal():
    admission()
    rows = []
    for n, name in enumerate(CASES):
        row = read(TRIAL / f"phase-{n:03d}/completed.json")
        p = read(TRIAL / f"phase-{n:03d}-profile/profile.json")
        need(row["case"] == name and row["status"] == "complete" and p["child_exit_code"] == 0
             and p["guard"] is None and p["sample_errors"] == 0 and not p["resource_limit_observations"], "All flows must pass")
        rows.append(row)
    for kind in ("ai", "original"):
        a = next(r for r in rows if r["case"] == f"legacy-{kind}")
        b = next(r for r in rows if r["case"] == f"candidate-{kind}")
        need(a["input_sha256"] == b["input_sha256"] and a["decision"] == b["decision"]
             and a["assessment"] == b["assessment"], "Regional evidence altered whole-file authorship estimate")
    result = {"schema": "chipmunk.native-regional-paid-flow-result/49", "status": "automatic_checks_complete",
              "cases": rows, "paired_authorship_estimates_unchanged": True, "pdf_visual_review_pending": True,
              "default_changed": False, "new_accuracy_claim": False, "github_workflows_started": 0,
              "simulated_stripe_only": True, "outreach_sent": False, "deployment_done": False}
    write(TRIAL / "summary.json", result)
    print(json.dumps({"flows_complete": len(rows), "visual_review_pending": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "phase", "seal"))
    parser.add_argument("number", type=int, nargs="?")
    args = parser.parse_args()
    {"prepare": prepare, "phase": lambda: phase(args.number), "seal": seal}[args.command]()
