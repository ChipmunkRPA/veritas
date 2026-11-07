"""Freeze, extract and compare a different frozen speech backbone, research only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration as serving
from scripts import audio_wavlm_features_v13 as features
from scripts import audio_wavlm_response_v13 as method
from scripts import run_audio_layer_v12 as prior
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_guard_v7 import initial
from scripts.audio_multisource_v8 import final
from scripts.audio_source_identity_v9 import digest
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.run_audio_multisource_v8 import input_pcm, open_journal
from scripts.score_audio_confirmation_v7 import runtime, weights

SCHEMA = "chipmunk.audio-wavlm-lock/13"
COUNTS = {"fit": 1926, "cal": 873, "development": 1449}
NEW_FILES = {
    "backend/scripts/audio_wavlm_features_v13.py", "backend/scripts/audio_wavlm_response_v13.py",
    "backend/scripts/run_audio_wavlm_v13.py", "backend/tests/test_audio_wavlm_v13.py",
    "docs/audio-wavlm-v13-protocol.md",
}
PRIVATE_FIELDS = {"source", "weights_root"}


def installed_implementation():
    modules = ("transformers.models.wavlm.modeling_wavlm", "transformers.models.wavlm.configuration_wavlm",
               "torch.nn.functional", "scipy.optimize._lbfgsb_py", "scipy.optimize._lbfgsb")
    return {name: digest(Path(importlib.import_module(name).__file__)) for name in modules}


def implementation():
    project = Path(__file__).resolve().parents[2]
    return {n: digest(project / n) for n in sorted(set(prior.implementation()) | NEW_FILES)}


def freeze(source, weights_root, output):
    if output.exists() or any((output.parent / n).exists() for n in ("features.jsonl", "trained")):
        raise ValueError("Never extracted/fitted destination required")
    body, _, rows, _, _ = prior.prepared(source)
    if Counter(r["role"] for r in rows) != COUNTS or len(rows) != 4248:
        raise ValueError("Only the complete consumed original roles are allowed")
    for row in rows:
        if digest(Path(body["source_root"]) / row["path"]) != row["sha256"]:
            raise ValueError("Original waveform bytes changed before freeze")
    write(output, {"schema": SCHEMA, "source": str(source.resolve()),
                   "weights_root": str(weights_root.resolve()), "source_sha256": digest(source / "source-plan.json"),
                   "variants_sha256": digest(source / "development/variants.json"),
                   "method": method.METHOD, "implementation_sha256": implementation(),
                   "runtime": runtime(), "installed_implementation_sha256": installed_implementation(),
                   "baseline_weights": weights(),
                   "pretrained_assets": features.assets(weights_root),
                   "role_counts": COUNTS, "planned": len(rows), "expected_unassessed": 62,
                   "original_receipt_sha256": prior.prior.reuse.RECEIPT_SHA,
                   "original_responses_sha256": prior.prior.reuse.RESPONSES_SHA,
                   "reserved_confirmation_sha256": prior.prior.original.CONFIRMATION_SHA,
                   "native_ids": prior.native_plan(rows),
                   "serving_calibration_sha256": serving.SHA256, "serving_bundle": serving.load(),
                   "project_forwards_started": False, "fitting_started": False,
                   "deployment_eligible": False})
    print(json.dumps({"lock_sha256": digest(output), "inputs": len(rows),
                      "new_project_forwards": 0, "fitting_started": False}), flush=True)


def validate_lock(lock):
    if (lock["schema"] != SCHEMA or lock["method"] != method.METHOD
            or lock["implementation_sha256"] != implementation() or lock["runtime"] != runtime()
            or lock["installed_implementation_sha256"] != installed_implementation()
            or lock["source_sha256"] != prior.prior.original.SOURCE_SHA
            or lock["variants_sha256"] != prior.prior.original.VARIANTS_SHA
            or lock["role_counts"] != COUNTS or lock["planned"] != 4248 or lock["expected_unassessed"] != 62
            or lock["serving_bundle"] != serving.load() or lock["serving_calibration_sha256"] != serving.SHA256
            or lock["original_receipt_sha256"] != prior.prior.reuse.RECEIPT_SHA
            or lock["original_responses_sha256"] != prior.prior.reuse.RESPONSES_SHA
            or lock["reserved_confirmation_sha256"] != prior.prior.original.CONFIRMATION_SHA
            or any(lock[k] is not False for k in ("project_forwards_started", "fitting_started", "deployment_eligible"))):
        raise ValueError("Prospective method/runtime/source/role/service commitments changed")


def check(path):
    lock = json.loads(path.read_text())
    validate_lock(lock)
    source = Path(lock["source"])
    if (digest(source / "source-plan.json") != lock["source_sha256"]
            or digest(source / "development/variants.json") != lock["variants_sha256"]):
        raise ValueError("Original consumed source table changed")
    result = prior.prepared(Path(lock["source"]))
    if lock["native_ids"] != prior.native_plan(result[2]):
        raise ValueError("Original99-cell native plan changed")
    if features.assets(Path(lock["weights_root"])) != lock["pretrained_assets"]:
        raise ValueError("Frozen pretrained assets changed")
    return lock, result


def public_freeze(lock_path, output):
    lock, _ = check(lock_path)
    public = {k: v for k, v in lock.items() if k not in PRIVATE_FIELDS}
    value = {"schema": "chipmunk.audio-wavlm-public-freeze/13",
             "private_lock_sha256": digest(lock_path), "lock": public,
             "scope": "Code/runtime/checksum metadata only; no media, features, private paths or accuracy result"}
    raw = json.dumps(value, allow_nan=False)
    if "/Users/" in raw or "/private/" in raw:
        raise ValueError("Private path in public prospective freeze")
    write(output, value)


def feature_header(lock_path):
    return {"schema": "chipmunk.audio-wavlm-features/13", "lock_sha256": digest(lock_path), "planned": 4248}


def validate_feature(row, source):
    if any(row[k] != source[k] for k in ("id", "role", "sha256", "pcm_sha256")):
        raise ValueError("Feature source/role identity differs")
    features.validate(row, source)


def extract(lock_path, output):
    lock, (body, _, rows, _, _) = check(lock_path)
    completed = open_journal(output, feature_header(lock_path), rows)
    for row, source in zip(completed, rows, strict=False):
        validate_feature(row, source)
    if len(completed) == len(rows):
        print(json.dumps({"completed": len(rows), "new_forwards": 0}), flush=True)
        return
    model = features.load(lock["weights_root"])
    with output.open("a") as stream:
        for index, source in enumerate(rows[len(completed):], len(completed)):
            start = time.monotonic()
            row = {k: source[k] for k in ("id", "role", "sha256", "pcm_sha256", "status")}
            try:
                pcm = input_pcm(body, source)
                row.update(features.extract(pcm, model) if source["status"] == "ok" else
                           {k: None for k in ("wavlm", "wavlm_sha256", "feature_frames", "valid_feature_frames", "input_pcm_sha256")})
                validate_feature(row, source)
            except Exception as error:
                row.update(status="failed", error_type=type(error).__name__)
            row["runtime_s"] = time.monotonic() - start
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if (index + 1) % 50 == 0 or row["status"] == "failed" or index + 1 == len(rows):
                print(json.dumps({"completed": index + 1, "planned": len(rows), "status": row["status"]}), flush=True)
            if row["status"] == "failed":
                raise ValueError("Extraction failed; preserve the journal prefix and stop before fitting")


def feature_rows(lock_path, receipt):
    lock, result = check(lock_path)
    header, rows = load_rows(receipt)
    sources = result[2]
    if header != feature_header(lock_path) or [r["id"] for r in rows] != [r["id"] for r in sources]:
        raise ValueError("Every original feature record in bound order required")
    for row, source in zip(rows, sources, strict=True):
        validate_feature(row, source)
    return lock, result, [{**s, **r} for s, r in zip(sources, rows, strict=True)]


def train(lock_path, receipt, output):
    _, _, rows = feature_rows(lock_path, receipt)
    output.mkdir(parents=True, exist_ok=False)
    fitted = {a: method.fit([r for r in rows if r["role"] == "fit"], a) for a in method.ARMS}
    common = {"lock_sha256": digest(lock_path), "receipt_sha256": digest(receipt),
              "fitted": fitted, "deployment_eligible": False}
    write(output / "fit.json", common)
    guards = {a: method.calibrate([r for r in rows if r["role"] == "cal"], fitted[a]) for a in method.ARMS}
    write(output / "head.json", {**common, "fit_sha256": digest(output / "fit.json"), "guards": guards})
    print(json.dumps({"head_sha256": digest(output / "head.json"), "arms": list(method.ARMS)}), flush=True)


def read_head(lock_path, receipt, head_path):
    head = json.loads(head_path.read_text())
    original = json.loads((head_path.parent / "fit.json").read_text())
    if (head["lock_sha256"] != digest(lock_path) or head["receipt_sha256"] != digest(receipt)
            or head["fit_sha256"] != digest(head_path.parent / "fit.json")
            or any(head[k] != original[k] for k in original)
            or set(head["fitted"]) != set(method.ARMS) or set(head["guards"]) != set(method.ARMS)
            or head["deployment_eligible"] is not False):
        raise ValueError("Original fitting-before-calibration lineage required")
    return head


def expected_response(row, source, old, head, old_probes, filtered_sha=None):
    if row["status"] == "failed" or any(row[k] != source[k] for k in prior.META):
        raise ValueError("Failed or rebound response cannot be resumed or accepted")
    if row["status"] != source["status"] or any(row[k] != old[k] for k in ("original_p", "original_decision")):
        raise ValueError("Original assessment/current-service result changed")
    probabilities = {a: method.predict(source, head["fitted"][a]) for a in method.ARMS}
    needed = any(initial(probabilities[a], head["guards"][a]) == "original" for a in method.ARMS)
    probe = row["probe"]
    if probe["status"] != ("computed" if needed else "not_required"):
        raise ValueError("Missing or unexpected candidate-stability probe")
    if needed:
        cached = old_probes.get(source["id"])
        bound_sha = cached["filtered_pcm_sha256"] if cached is not None else filtered_sha
        if bound_sha is None:
            raise ValueError("New filtered probe requires original-source PCM reconstruction")
        features.validate({**probe, "status": "ok"}, {"status": "ok", "pcm_sha256": bound_sha})
        if cached is not None and (probe["raw"] != cached["raw"]
                or probe["input_pcm_sha256"] != cached["filtered_pcm_sha256"]):
            raise ValueError("Known original filtered-probe response changed")
    for arm in method.ARMS:
        band = method.predict(probe, head["fitted"][arm]) if needed else None
        detail = final(probabilities[arm], band, source["raw"], probe["raw"], head["guards"][arm])
        if not same_tree(row["candidates"][arm], {"p": probabilities[arm], "decision": detail["decision"], "detail": detail}):
            raise ValueError("Candidate probability, independent guard or dual stability changed")


def new_filtered_sha(body, source, response, old_probes):
    if response["probe"]["status"] != "computed" or source["id"] in old_probes:
        return None
    band = band_view(input_pcm(body, source))
    return hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()


def compare(lock_path, receipt, head_path, output):
    lock, (body, reference, _, baseline, old_probes), all_rows = feature_rows(lock_path, receipt)
    if weights() != lock["baseline_weights"]:
        raise ValueError("Original commercial checkpoint bytes changed")
    head = read_head(lock_path, receipt, head_path)
    rows = [r for r in all_rows if r["role"] == "development"]
    header = {"schema": "chipmunk.audio-wavlm-responses/13", "lock_sha256": digest(lock_path),
              "receipt_sha256": digest(receipt), "head_sha256": digest(head_path), "planned": 1449}
    completed = open_journal(output, header, rows)
    for row, source in zip(completed, rows, strict=False):
        expected_response(row, source, baseline[source["id"]], head, old_probes,
                          new_filtered_sha(body, source, row, old_probes))
    candidate, original, hook, captured = None, None, None, []
    try:
        with output.open("a") as stream:
            for index, source in enumerate(rows[len(completed):], len(completed)):
                old = baseline[source["id"]]
                row = {k: source[k] for k in prior.META}
                row.update(status=source["status"], original_p=old["original_p"], original_decision=old["original_decision"])
                probe = {"status": "not_required", "raw": None, "wavlm": None}
                filtered_sha = None
                try:
                    probabilities = {a: method.predict(source, head["fitted"][a]) for a in method.ARMS}
                    if any(initial(probabilities[a], head["guards"][a]) == "original" for a in method.ARMS):
                        band = band_view(input_pcm(body, source))
                        filtered_sha = hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                        cached = old_probes.get(source["id"])
                        if cached is None:
                            if original is None:
                                original, _, captured, hook = prior.load_models(reference)
                            raw, _ = capture(original, band, captured)
                        else:
                            if hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest() != cached["filtered_pcm_sha256"]:
                                raise ValueError("Original filtered PCM changed")
                            raw = cached["raw"]
                        if candidate is None:
                            candidate = features.load(lock["weights_root"])
                        probe.update(features.extract(band, candidate), raw=raw, status="computed",
                                     raw_reference="known-original-cache" if cached is not None else "new-original-forward")
                    row["candidates"] = {}
                    for arm in method.ARMS:
                        band_p = method.predict(probe, head["fitted"][arm]) if probe["status"] == "computed" else None
                        detail = final(probabilities[arm], band_p, source["raw"], probe["raw"], head["guards"][arm])
                        row["candidates"][arm] = {"p": probabilities[arm], "decision": detail["decision"], "detail": detail}
                    row["probe"] = probe
                    expected_response(row, source, old, head, old_probes, filtered_sha)
                except Exception as error:
                    row.update(status="failed", error_type=type(error).__name__, original_p=None,
                               original_decision="inconclusive", candidates={a: {"p": None, "decision": "inconclusive"} for a in method.ARMS})
                    row["probe"] = {"status": "failed", "raw": None, "wavlm": None}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 50 == 0 or row["status"] == "failed" or index + 1 == len(rows):
                    print(json.dumps({"completed": index + 1, "planned": len(rows), "status": row["status"]}), flush=True)
                if row["status"] == "failed":
                    raise ValueError("Comparison failed; preserve the journal prefix")
    finally:
        if hook is not None:
            hook.remove()


def review(lock_path, receipt, head_path, responses_path, output):
    _, (body, _, _, baseline, old_probes), rows = feature_rows(lock_path, receipt)
    head = read_head(lock_path, receipt, head_path)
    header, responses = load_rows(responses_path)
    expected = [r for r in rows if r["role"] == "development"]
    if (header != {"schema": "chipmunk.audio-wavlm-responses/13", "lock_sha256": digest(lock_path),
                   "receipt_sha256": digest(receipt), "head_sha256": digest(head_path), "planned": 1449}
            or [r["id"] for r in responses] != [r["id"] for r in expected]):
        raise ValueError("All original1449 bound development outcomes required")
    for arm in method.ARMS:
        if not same_tree(method.calibrate([r for r in rows if r["role"] == "cal"], head["fitted"][arm]), head["guards"][arm]):
            raise ValueError("Independent calibration changed")
    for row, source in zip(responses, expected, strict=True):
        expected_response(row, source, baseline[source["id"]], head, old_probes,
                          new_filtered_sha(body, source, row, old_probes))
    summaries = {a: summarize([{**r, "candidate_p": r["candidates"][a]["p"],
                               "candidate_decision": r["candidates"][a]["decision"]} for r in responses], 161)
                 for a in method.ARMS}
    write(output, {"schema": "chipmunk.audio-wavlm-summary/13", "lock_sha256": digest(lock_path),
                   "receipt_sha256": digest(receipt), "head_sha256": digest(head_path),
                   "responses_sha256": digest(responses_path), "arms": summaries,
                   "selected_arm_for_possible_confirmation": method.select(summaries), "deployment_eligible": False})
    print(json.dumps({"selected": method.select(summaries),
                      "failed_gates": {a: [k for k, v in s["gates"].items() if not v] for a, s in summaries.items()}}), flush=True)


if __name__ == "__main__":
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "public", "extract", "train", "compare", "review"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--weights-root", type=Path)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--responses", type=Path)
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.input, a.weights_root, a.output)
    elif a.stage == "public":
        public_freeze(a.input, a.output)
    elif a.stage == "extract":
        extract(a.input, a.output)
    elif a.stage == "train":
        train(a.input, a.receipt, a.output)
    elif a.stage == "compare":
        compare(a.input, a.receipt, a.head, a.output)
    else:
        review(a.input, a.receipt, a.head, a.responses, a.output)
