"""Frozen consumed-source separation experiment; never reserved confirmation.

The unchanged original voice loader owns inter-op thread initialization. The CLI
sets only intra-op threads. No app detector, calibration or acceptance gate changes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration as serving
from scripts import audio_separation_features_v14 as features
from scripts import audio_separation_response_v14 as method
from scripts import audio_separation_v14 as separation
from scripts import audit_audio_separation_v14 as upstream
from scripts import run_audio_wavlm_v13 as previous
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_guard_v7 import initial
from scripts.audio_multisource_v8 import final
from scripts.audio_source_identity_v9 import digest
from scripts.audit_audio_identity_forwards_v9 import same_native
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.run_audio_multisource_v8 import input_pcm, open_journal
from scripts.score_audio_confirmation_v7 import runtime, weights

prior = previous.prior
SCHEMA = "chipmunk.audio-separation-lock/14"
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
PROJECT = Path(__file__).resolve().parents[2]
NEW_FILES = {
    "backend/scripts/audio_separation_v14.py", "backend/scripts/audio_separation_response_v14.py",
    "backend/scripts/audio_separation_features_v14.py", "backend/scripts/run_audio_separation_v14.py",
    "backend/scripts/audit_audio_separation_v14.py", "backend/tests/test_audio_separation_v14.py",
    "backend/tests/test_audio_separation_response_v14.py", "backend/tests/test_audio_separation_execution_v14.py",
    "docs/audio-separation-v14-protocol.md", "docs/third-party/asteroid-v14-MIT.txt",
}
PRIVATE_FIELDS = {"source", "weights_root", "package_root"}
AUDIT = "docs/evaluation-panels/audio-v14-separation-structural-audit.json"


def implementation():
    return {name: digest(PROJECT / name) for name in sorted(set(previous.implementation()) | NEW_FILES)}


def installed_implementation():
    names = ("transformers.models.wav2vec2_bert.modeling_wav2vec2_bert", "torch.nn.functional",
             "torch.nn.modules.conv", "scipy.optimize._lbfgsb_py", "scipy.optimize._lbfgsb",
             "scipy.special._logsumexp")
    return {name: digest(Path(importlib.import_module(name).__file__)) for name in names}


def structural_audit():
    value = json.loads((PROJECT / AUDIT).read_text())
    if (value["passed"] is not True or value["project_recordings_scored"] != 0
            or value["local_code_sha256"] != digest(PROJECT / "backend/scripts/audio_separation_v14.py")
            or value["audit_code_sha256"] != digest(PROJECT / "backend/scripts/audit_audio_separation_v14.py")
            or value["spec"] != separation.SPEC or value["model_args_retained"] != separation.ARGS
            or len(value["synthetic_probes"]) != 3 or not value["repeated_synthetic_exact"]
            or not all(v["all_outputs_exact"] for v in value["synthetic_probes"])):
        raise ValueError("Exact original synthetic source/graph preparation required")
    return digest(PROJECT / AUDIT)


def upstream_assets(root):
    result = {name: digest(root / name) for name in upstream.WHEELS}
    if result != upstream.WHEELS:
        raise ValueError("Pinned upstream reference packages changed")
    return result


def freeze(source, weights_root, package_root, output):
    if output.exists() or any((output.parent / name).exists() for name in ("features.jsonl", "trained")):
        raise ValueError("Never extracted/fitted destination required")
    body, _, rows, _, _ = prior.prepared(source)
    if len(rows) != 4248 or Counter(r["role"] for r in rows) != COUNTS:
        raise ValueError("All original consumed roles required; confirmation forbidden")
    for row in rows:
        if digest(Path(body["source_root"]) / row["path"]) != row["sha256"]:
            raise ValueError("Original waveform changed before freeze")
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, {"schema": SCHEMA, "source": str(source.resolve()), "weights_root": str(weights_root.resolve()),
        "package_root": str(package_root.resolve()), "upstream_assets": upstream_assets(package_root),
        "source_sha256": digest(source / "source-plan.json"), "variants_sha256": digest(source / "development/variants.json"),
        "method": method.METHOD, "implementation_sha256": implementation(), "runtime": runtime(),
        "installed_implementation_sha256": installed_implementation(), "baseline_weights": weights(),
        "pretrained_assets": features.assets(weights_root), "structural_audit_sha256": structural_audit(),
        "role_counts": COUNTS, "planned": 4248, "expected_unassessed": 62,
        "original_receipt_sha256": prior.prior.reuse.RECEIPT_SHA,
        "original_responses_sha256": prior.prior.reuse.RESPONSES_SHA,
        "reserved_confirmation_sha256": prior.prior.original.CONFIRMATION_SHA,
        "native_ids": prior.native_plan(rows), "serving_calibration_sha256": serving.SHA256,
        "serving_bundle": serving.load(), "project_forwards_started": False,
        "fitting_started": False, "deployment_eligible": False})
    print(json.dumps({"lock_sha256": digest(output), "inputs": 4248, "new_project_forwards": 0}), flush=True)


def validate_lock(lock):
    if (lock["schema"] != SCHEMA or lock["method"] != method.METHOD or lock["implementation_sha256"] != implementation()
            or lock["runtime"] != runtime() or lock["installed_implementation_sha256"] != installed_implementation()
            or lock["source_sha256"] != prior.prior.original.SOURCE_SHA
            or lock["variants_sha256"] != prior.prior.original.VARIANTS_SHA
            or lock["role_counts"] != COUNTS or lock["planned"] != 4248 or lock["expected_unassessed"] != 62
            or lock["structural_audit_sha256"] != structural_audit()
            or lock["upstream_assets"] != upstream.WHEELS
            or lock["serving_calibration_sha256"] != serving.SHA256 or lock["serving_bundle"] != serving.load()
            or lock["original_receipt_sha256"] != prior.prior.reuse.RECEIPT_SHA
            or lock["original_responses_sha256"] != prior.prior.reuse.RESPONSES_SHA
            or lock["reserved_confirmation_sha256"] != prior.prior.original.CONFIRMATION_SHA
            or any(lock[k] is not False for k in ("project_forwards_started", "fitting_started", "deployment_eligible"))):
        raise ValueError("Frozen source, method, roles, runtime, calibration or scope changed")


def check(path):
    lock = json.loads(path.read_text())
    validate_lock(lock)
    root = Path(lock["source"])
    if digest(root / "source-plan.json") != lock["source_sha256"] or digest(root / "development/variants.json") != lock["variants_sha256"]:
        raise ValueError("Original source tables changed")
    result = prior.prepared(root)
    if (lock["native_ids"] != prior.native_plan(result[2])
            or features.assets(Path(lock["weights_root"])) != lock["pretrained_assets"]
            or upstream_assets(Path(lock["package_root"])) != lock["upstream_assets"]
            or weights() != lock["baseline_weights"]):
        raise ValueError("Fixed native cells or original/separator checkpoint assets changed")
    return lock, result


def public_freeze(lock_path, output):
    lock, _ = check(lock_path)
    value = {"schema": "chipmunk.audio-separation-public-freeze/14", "private_lock_sha256": digest(lock_path),
             "lock": {k: v for k, v in lock.items() if k not in PRIVATE_FIELDS},
             "scope": "Prospective code/runtime/checksum metadata; no recordings, real voice features, private paths or accuracy"}
    if any(part in json.dumps(value) for part in ("/Users/", "/private/", "/home/")):
        raise ValueError("Private paths cannot enter public freeze")
    write(output, value)


def header(lock_path):
    return {"schema": "chipmunk.audio-separation-features/14", "lock_sha256": digest(lock_path), "planned": 4248}


def validate_feature(record, source, reference):
    if any(record[k] != source[k] for k in ("id", "role", "sha256", "pcm_sha256")):
        raise ValueError("Feature source/role identity changed")
    features.validate(record, source, reference)


def extract(lock_path, output):
    lock, (body, reference, rows, _, _) = check(lock_path)
    done = open_journal(output, header(lock_path), rows)
    for record, source in zip(done, rows, strict=False):
        validate_feature(record, source, reference)
    if len(done) == len(rows):
        print(json.dumps({"completed": len(rows), "new_forwards": 0}), flush=True)
        return
    # Exactly one owner initializes inter-op threads, before any separation forward.
    voice, _, captured, hook = prior.load_models(reference)
    voice.requires_grad_(False)
    model = separation.load(lock["weights_root"])
    try:
        with output.open("a") as stream:
            for index, source in enumerate(rows[len(done):], len(done)):
                started = time.monotonic()
                record = {k: source[k] for k in ("id", "role", "sha256", "pcm_sha256", "status", "raw")}
                try:
                    pcm = input_pcm(body, source)
                    record.update(features.extract(pcm, model, voice, captured, reference, source["raw"])
                                  if source["status"] == "ok" else {"stems": None, "input_pcm_sha256": features.pcm_sha(pcm)})
                    validate_feature(record, source, reference)
                except Exception as error:
                    traceback.print_exc()
                    record.update(status="failed", error_type=type(error).__name__)
                record["runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 25 == 0 or record["status"] == "failed" or index + 1 == len(rows):
                    print(json.dumps({"completed": index + 1, "planned": len(rows), "status": record["status"]}), flush=True)
                if record["status"] == "failed":
                    raise ValueError("Extraction failed; preserve terminal journal and stop before fitting")
    finally:
        hook.remove()


def feature_rows(lock_path, receipt):
    lock, result = check(lock_path)
    meta, records = load_rows(receipt)
    sources = result[2]
    if meta != header(lock_path) or [r["id"] for r in records] != [r["id"] for r in sources]:
        raise ValueError("Complete ordered original feature receipt required")
    for record, source in zip(records, sources, strict=True):
        validate_feature(record, source, result[1])
    return lock, result, [{**source, **record} for source, record in zip(sources, records, strict=True)]


def train(lock_path, receipt, output):
    _, _, rows = feature_rows(lock_path, receipt)
    output.mkdir(parents=True, exist_ok=False)
    fitted = {arm: method.fit([r for r in rows if r["role"] == "fit"], arm) for arm in method.ARMS}
    common = {"lock_sha256": digest(lock_path), "receipt_sha256": digest(receipt), "fitted": fitted, "deployment_eligible": False}
    write(output / "fit.json", common)
    guards = {arm: method.calibrate([r for r in rows if r["role"] == "cal"], fitted[arm]) for arm in method.ARMS}
    write(output / "head.json", {**common, "fit_sha256": digest(output / "fit.json"), "guards": guards})
    print(json.dumps({"head_sha256": digest(output / "head.json"), "arms": list(method.ARMS)}), flush=True)


def read_head(lock_path, receipt, head_path):
    head = json.loads(head_path.read_text())
    original = json.loads((head_path.parent / "fit.json").read_text())
    if (head["lock_sha256"] != digest(lock_path) or head["receipt_sha256"] != digest(receipt)
            or head["fit_sha256"] != digest(head_path.parent / "fit.json")
            or any(head[k] != original[k] for k in original) or set(head["fitted"]) != set(method.ARMS)
            or set(head["guards"]) != set(method.ARMS) or head["deployment_eligible"] is not False):
        raise ValueError("Complete immutable fitting-before-calibration lineage required")
    return head


def validate_response(record, source, old, head, reference, bound_band_sha, old_probes):
    if (record["status"] != source["status"] or any(record[k] != source[k] for k in prior.META)
            or any(record[k] != old[k] for k in ("original_p", "original_decision"))):
        raise ValueError("Source/assessment/current-service outcome changed")
    probabilities = {arm: method.predict(source, head["fitted"][arm]) for arm in method.ARMS}
    need = any(initial(probabilities[arm], head["guards"][arm]) == "original" for arm in method.ARMS)
    probe = record["probe"]
    if probe["status"] != ("computed" if need else "not_required"):
        raise ValueError("Missing or unexpected candidate stability probe")
    if need:
        features.validate({**probe, "status": "ok"},
                          {"status": "ok", "raw": probe["raw"], "pcm_sha256": bound_band_sha}, reference)
        cached = old_probes.get(source["id"])
        if cached is not None and (probe["raw"] != cached["raw"] or bound_band_sha != cached["filtered_pcm_sha256"]):
            raise ValueError("Known original filtered response/PCM changed")
    for arm in method.ARMS:
        band_p = method.predict(probe, head["fitted"][arm]) if need else None
        detail = final(probabilities[arm], band_p, source["raw"], probe["raw"], head["guards"][arm])
        if not same_tree(record["candidates"][arm], {"p": probabilities[arm], "decision": detail["decision"], "detail": detail}):
            raise ValueError("Raw/bag probability, own guard or stability outcome changed")


def response_header(lock_path, receipt, head_path):
    return {"schema": "chipmunk.audio-separation-responses/14", "lock_sha256": digest(lock_path),
            "receipt_sha256": digest(receipt), "head_sha256": digest(head_path), "planned": 1449}


def compare(lock_path, receipt, head_path, output):
    lock, (body, reference, _, baseline, old_probes), rows = feature_rows(lock_path, receipt)
    head = read_head(lock_path, receipt, head_path)
    sources = [r for r in rows if r["role"] == "development"]
    done = open_journal(output, response_header(lock_path, receipt, head_path), sources)
    for record, source in zip(done, sources, strict=False):
        band_sha = features.pcm_sha(band_view(input_pcm(body, source))) if record["probe"]["status"] == "computed" else None
        validate_response(record, source, baseline[source["id"]], head, reference, band_sha, old_probes)
    voice = model = hook = None
    captured = []
    try:
        with output.open("a") as stream:
            for index, source in enumerate(sources[len(done):], len(done)):
                old = baseline[source["id"]]
                record = {k: source[k] for k in prior.META}
                record.update(status=source["status"], original_p=old["original_p"], original_decision=old["original_decision"])
                probe = {"status": "not_required", "raw": None, "stems": None}
                band_sha = None
                try:
                    probabilities = {arm: method.predict(source, head["fitted"][arm]) for arm in method.ARMS}
                    if any(initial(probabilities[arm], head["guards"][arm]) == "original" for arm in method.ARMS):
                        band = band_view(input_pcm(body, source))
                        band_sha = features.pcm_sha(band)
                        if voice is None:
                            voice, _, captured, hook = prior.load_models(reference)
                            voice.requires_grad_(False)
                            model = separation.load(lock["weights_root"])
                        cached = old_probes.get(source["id"])
                        if cached is not None:
                            if band_sha != cached["filtered_pcm_sha256"]:
                                raise ValueError("Original filtered input changed")
                            raw = cached["raw"]
                        else:
                            raw, _ = capture(voice, band, captured)
                        probe.update(features.extract(band, model, voice, captured, reference, raw), raw=raw, status="computed",
                                     raw_reference="known-original-cache" if cached is not None else "new-original-forward")
                    record["candidates"] = {}
                    for arm in method.ARMS:
                        band_p = method.predict(probe, head["fitted"][arm]) if probe["status"] == "computed" else None
                        detail = final(probabilities[arm], band_p, source["raw"], probe["raw"], head["guards"][arm])
                        record["candidates"][arm] = {"p": probabilities[arm], "decision": detail["decision"], "detail": detail}
                    record["probe"] = probe
                    validate_response(record, source, old, head, reference, band_sha, old_probes)
                except Exception as error:
                    traceback.print_exc()
                    record.update(status="failed", error_type=type(error).__name__)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 25 == 0 or record["status"] == "failed" or index + 1 == len(sources):
                    print(json.dumps({"completed": index + 1, "planned": len(sources), "status": record["status"]}), flush=True)
                if record["status"] == "failed":
                    raise ValueError("Comparison failed; preserve terminal journal")
    finally:
        if hook is not None:
            hook.remove()


def review(lock_path, receipt, head_path, responses_path, output):
    _, (body, reference, _, baseline, old_probes), rows = feature_rows(lock_path, receipt)
    head = read_head(lock_path, receipt, head_path)
    meta, records = load_rows(responses_path)
    sources = [r for r in rows if r["role"] == "development"]
    if meta != response_header(lock_path, receipt, head_path) or [r["id"] for r in records] != [r["id"] for r in sources]:
        raise ValueError("All1449 ordered development outcomes required")
    for arm in method.ARMS:
        if not same_tree(method.calibrate([r for r in rows if r["role"] == "cal"], head["fitted"][arm]), head["guards"][arm]):
            raise ValueError("Own independent calibration changed")
    for record, source in zip(records, sources, strict=True):
        band_sha = features.pcm_sha(band_view(input_pcm(body, source))) if record["probe"]["status"] == "computed" else None
        validate_response(record, source, baseline[source["id"]], head, reference, band_sha, old_probes)
    summaries = {arm: summarize([{**r, "candidate_p": r["candidates"][arm]["p"],
                                   "candidate_decision": r["candidates"][arm]["decision"]} for r in records], 161) for arm in method.ARMS}
    write(output, {"schema": "chipmunk.audio-separation-summary/14", "lock_sha256": digest(lock_path),
        "receipt_sha256": digest(receipt), "head_sha256": digest(head_path), "responses_sha256": digest(responses_path),
        "arms": summaries, "selected_arm_for_possible_confirmation": method.select(summaries), "deployment_eligible": False})
    print(json.dumps({"selected": method.select(summaries), "failed_gates":
                      {arm: [key for key, value in summary["gates"].items() if not value] for arm, summary in summaries.items()}}), flush=True)


def native(lock_path, receipt, head_path, responses_path, output):
    lock, (body, reference, _, _, _), rows = feature_rows(lock_path, receipt)
    read_head(lock_path, receipt, head_path)
    meta, records = load_rows(responses_path)
    if (meta != response_header(lock_path, receipt, head_path)
            or [r["id"] for r in records] != [r["id"] for r in rows if r["role"] == "development"]):
        raise ValueError("Complete ordered comparison required before native review")
    probes = {r["id"]: r["probe"] for r in records if r["probe"]["status"] == "computed"}
    ids = sorted(set(lock["native_ids"]) | set(probes))
    by_id = {r["id"]: r for r in rows}
    voice, activity, captured, hook = prior.load_models(reference)
    voice.requires_grad_(False)
    model = separation.load(lock["weights_root"])
    native_model, _ = upstream.reference(Path(lock["package_root"]), model.state_dict(), 8000)
    cases = []
    try:
        for index, identifier in enumerate(ids):
            source = by_id[identifier]
            pcm = input_pcm(body, source)
            original = prior.prior.original.extract_one(pcm, voice, activity, captured)
            if not same_native(original, source):
                raise ValueError("Original source scope/raw/vector/VAD/window changed")
            if source["status"] == "ok":
                actual = features.extract(pcm, model, voice, captured, reference, source["raw"])
                if any(actual[k] != source[k] for k in actual):
                    raise ValueError("Full native source estimate/feature bytes differ")
                assert_upstream_estimates(pcm, native_model, source["stems"])
            if identifier in probes:
                band = band_view(pcm)
                raw, _ = capture(voice, band, captured)
                actual = features.extract(band, model, voice, captured, reference, raw)
                if raw != probes[identifier]["raw"] or any(actual[k] != probes[identifier][k] for k in actual):
                    raise ValueError("Required filtered original/source estimates/features differ")
                assert_upstream_estimates(band, native_model, probes[identifier]["stems"])
            cases.append({"id": identifier, "sha256": source["sha256"], "pcm_sha256": source["pcm_sha256"],
                          "predetermined_cell": identifier in lock["native_ids"], "computed_probe": identifier in probes, "passed": True})
            if (index + 1) % 25 == 0 or index + 1 == len(ids):
                print(json.dumps({"native_completed": index + 1, "planned": len(ids)}), flush=True)
    finally:
        hook.remove()
    write(output, {"schema": "chipmunk.audio-separation-native/14", "passed": True,
        "lock_sha256": digest(lock_path), "receipt_sha256": digest(receipt), "responses_sha256": digest(responses_path),
        "predetermined_cells": 99, "computed_probes": len(probes), "unique_cases": len(ids), "cases": cases,
        "scope": "Complete local original/source features and every computed filtered probe, with upstream graph waveform hashes; not fresh detection accuracy or outside human review"})


def assert_upstream_estimates(pcm, model, expected):
    import torch

    with torch.inference_mode():
        actual = model(torch.from_numpy(np.asarray(pcm, np.float32).copy())[None, None]).numpy()[0]
    if actual.shape != (2, 64000) or not np.isfinite(actual).all():
        raise ValueError("Complete native upstream source estimates required")
    if [features.pcm_sha(s) for s in actual] != [s["pcm_sha256"] for s in expected]:
        raise ValueError("Native upstream waveforms differ from cached local estimates")


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "public", "extract", "train", "compare", "review", "native"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--weights-root", type=Path)
    p.add_argument("--package-root", type=Path)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--responses", type=Path)
    a = p.parse_args()
    if a.stage == "freeze":
        if a.weights_root is None or a.package_root is None:
            p.error("Freeze requires --weights-root and --package-root")
        freeze(a.input, a.weights_root, a.package_root, a.output)
    elif a.stage == "public":
        public_freeze(a.input, a.output)
    elif a.stage == "extract":
        extract(a.input, a.output)
    elif a.stage == "train":
        train(a.input, a.receipt, a.output)
    elif a.stage == "compare":
        compare(a.input, a.receipt, a.head, a.output)
    else:
        (review if a.stage == "review" else native)(a.input, a.receipt, a.head, a.responses, a.output)
