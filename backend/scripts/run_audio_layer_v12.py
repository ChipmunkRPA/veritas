"""Freeze/extract/fit/compare fixed-layer audio features; never fresh data.

Every new original forward must exactly reproduce the prior raw/final-vector,
VAD and window result. No project forward/fitting before published freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration as serving
from scripts import audio_layer_response_v12 as method
from scripts import run_audio_full_representation_v11 as prior
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_guard_v7 import initial
from scripts.audio_layer_features_v12 import LAYER_IDS, LayerCapture
from scripts.audio_multisource_v8 import final
from scripts.audio_source_identity_v9 import digest
from scripts.audit_audio_identity_forwards_v9 import same_native, select
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.reproduce_audio_confirmation_v7 import same_tree
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import models, runtime, weights

PREVIOUS = {
    "head_sha256": "bf2a363ad6e491ad4c4fac1ec18658c58bc01eb20788a2a198c198426d481607",
    "responses_sha256": "683f56f383244257ddb47edb1e3d5c87a3071db19df7d13c4cc707f48129bf0f",
    "summary_sha256": "cab6bf33a644774d565a5bc3f949143b3d3ab35c84c1e9e5425a84bef79bcdcf",
}
PLAN_SHA = "48795283220152d5de07356ff54c018b8da797749957b6480d27582de97eed1e"
META = (
    "id",
    "base_id",
    "group",
    "role",
    "label",
    "generator",
    "dataset",
    "condition",
    "sha256",
    "pcm_sha256",
    "background_id",
)
LAYER_FIELDS = (
    "layer_ids",
    "layer_embeddings",
    "layer_embedding_sha256",
    "attention_mask_sha256",
    "feature_frames",
    "valid_feature_frames",
)


def implementation():
    project = Path(__file__).resolve().parents[2]
    names = set(prior.implementation()) | {
        "backend/scripts/audio_layer_features_v12.py",
        "backend/scripts/audio_layer_response_v12.py",
        "backend/scripts/run_audio_layer_v12.py",
        "backend/scripts/export_audio_layer_v12.py",
        "backend/scripts/audit_audio_identity_forwards_v9.py",
        "docs/audio-layer-v12-protocol.md",
    }
    return {name: digest(project / name) for name in sorted(names)}


def prepared(source):
    body, reference, rows, baseline, probes = prior.prepared(source)
    root = source.parent / "audio-v11"
    for name, path in (
        ("head_sha256", "trained/head.json"),
        ("responses_sha256", "responses.jsonl"),
        ("summary_sha256", "summary.json"),
    ):
        if digest(root / path) != PREVIOUS[name]:
            raise ValueError("Original completed full-vector control changed")
    header, results = load_rows(root / "responses.jsonl")
    expected = [r for r in rows if r["role"] == "development"]
    if header["head_sha256"] != PREVIOUS["head_sha256"] or [r["id"] for r in results] != [
        r["id"] for r in expected
    ]:
        raise ValueError("Complete previous-control input order required")
    for result, row in zip(results, expected, strict=True):
        if any(result[k] != row[k] for k in META):
            raise ValueError("Previous control source/truth binding differs")
        probe = result["probe"]
        if probe["status"] == "computed":
            prior.original.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]}, reference
            )
            cached = probes.get(row["id"])
            if cached is not None and any(
                probe[k] != cached[k] for k in ("raw", "embedding", "embedding_sha256", "filtered_pcm_sha256")
            ):
                raise ValueError("Original filtered-probe journals disagree")
            probes[row["id"]] = probe
    if len(probes) != 450:
        raise ValueError("All450 old final-vector probes required as parity references")
    return body, reference, rows, baseline, probes


def native_plan(rows):
    path = (
        Path(__file__).resolve().parents[2] / "docs/evaluation-panels/audio-v9-native-development-plan.json"
    )
    value = json.loads(path.read_text())
    if digest(path) != PLAN_SHA or value["ids"] != select(rows) or len(value["ids"]) != 99:
        raise ValueError("Unchanged99-cell native plan required")
    return value["ids"]


def freeze(source, output):
    body, _, rows, _, _ = prepared(source)
    if output.exists() or any((output.parent / p).exists() for p in ("representations.jsonl", "trained")):
        raise ValueError("New, never extracted/fitted destination required")
    for row in rows:
        if digest(Path(body["source_root"]) / row["path"]) != row["sha256"]:
            raise ValueError("Original waveform bytes changed before freeze")
    write(
        output,
        {
            "schema": "chipmunk.audio-layer-lock/12",
            "source_root": str(source.resolve()),
            "method": method.METHOD,
            "implementation_sha256": implementation(),
            "runtime": runtime(),
            "weights": weights(),
            "original_receipt_sha256": prior.reuse.RECEIPT_SHA,
            "original_responses_sha256": prior.reuse.RESPONSES_SHA,
            "conditional_responses_sha256": prior.CONDITIONAL_SHA,
            "previous_control": PREVIOUS,
            "role_counts": dict(Counter(r["role"] for r in rows)),
            "planned": len(rows),
            "original_input_bytes_verified": len(rows),
            "expected_unassessed": sum(r["status"] == "not_applicable" for r in rows),
            "available_prior_probe_references": 450,
            "native_plan_sha256": PLAN_SHA,
            "native_ids": native_plan(rows),
            "reserved_confirmation_sha256": prior.original.CONFIRMATION_SHA,
            "serving_calibration_sha256": serving.SHA256,
            "serving_bundle": serving.load(),
            "fitting_started": False,
            "new_neural_forwards": False,
            "deployment_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "lock_sha256": digest(output),
                "inputs": len(rows),
                "fitting_started": False,
                "new_neural_forwards": False,
            }
        ),
        flush=True,
    )


def check(path):
    lock = json.loads(path.read_text())
    if (
        lock["method"] != method.METHOD
        or lock["implementation_sha256"] != implementation()
        or lock["runtime"] != runtime()
        or lock["serving_bundle"] != serving.load()
        or lock["serving_calibration_sha256"] != serving.SHA256
        or lock["previous_control"] != PREVIOUS
        or lock["original_receipt_sha256"] != prior.reuse.RECEIPT_SHA
        or lock["original_responses_sha256"] != prior.reuse.RESPONSES_SHA
        or lock["conditional_responses_sha256"] != prior.CONDITIONAL_SHA
        or lock["reserved_confirmation_sha256"] != prior.original.CONFIRMATION_SHA
        or lock["role_counts"] != prior.original.COUNTS
        or lock["planned"] != 4248
        or lock["original_input_bytes_verified"] != 4248
        or lock["expected_unassessed"] != 62
        or lock["available_prior_probe_references"] != 450
        or lock["native_plan_sha256"] != PLAN_SHA
    ):
        raise ValueError("Frozen layer method/runtime/source/control/service commitments changed")
    result = prepared(Path(lock["source_root"]))
    if lock["native_ids"] != native_plan(result[2]):
        raise ValueError("Predetermined native cells changed")
    return lock, result


def validate_layers(record):
    if record["layer_ids"] != list(LAYER_IDS):
        raise ValueError("Original preselected layer order required")
    if record["status"] == "not_applicable":
        if any(record[k] is not None for k in LAYER_FIELDS if k != "layer_ids"):
            raise ValueError("Unassessed speech cannot carry layer measurements")
        return
    if record["status"] != "ok":
        raise ValueError("Failed layer extraction cannot enter fitting")
    value = np.asarray(record["layer_embeddings"], np.float32)
    if (
        value.shape != (3, 1024)
        or not np.isfinite(value).all()
        or value.astype(np.float64).tolist() != record["layer_embeddings"]
        or hashlib.sha256(np.asarray(value, "<f4").tobytes()).hexdigest() != record["layer_embedding_sha256"]
    ):
        raise ValueError("Exact finite float32 selected-layer payload and digest required")
    frames, valid = record["feature_frames"], record["valid_feature_frames"]
    mask_sha = record["attention_mask_sha256"]
    if (
        type(frames) is not int
        or type(valid) is not int
        or not 0 < valid <= frames
        or not isinstance(mask_sha, str)
        or len(mask_sha) != 64
        or any(c not in "0123456789abcdef" for c in mask_sha)
    ):
        raise ValueError("Original nonempty feature-frame clock and mask digest required")


def validate_original(record, expected, reference):
    if any(record[k] != expected[k] for k in ("id", "role", "sha256", "pcm_sha256")):
        raise ValueError("Layer receipt source/role binding differs")
    prior.original.validate_feature(record, reference)
    if not same_native(record, expected):
        raise ValueError("Read-only layer capture changed original status/raw/vector/VAD/window")
    validate_layers(record)


def load_models(reference):
    import torch

    model, activity = models()
    if (
        model.head.weight.detach().cpu().numpy()[0].tolist() != reference["head_weight"]
        or float(model.head.bias.detach().cpu()[0]) != reference["head_bias"]
        or model.backbone.config.num_hidden_layers != 24
        or model.backbone.config.hidden_size != 1024
        or tuple(model.layer_weights.shape) != (25,)
        or model.head.weight.dtype != torch.float32
        or model.head.weight.device.type != "cpu"
    ):
        raise ValueError("Exact original CPU/float32/25-state model and head required")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    return model, activity, captured, hook


def extract_one(pcm, model, activity, captured, layers):
    layers.clear()
    record = prior.original.extract_one(pcm, model, activity, captured)
    if record["status"] == "ok":
        record.update(layers.take())
    else:
        if layers.records:
            raise ValueError("Unassessed source unexpectedly ran a neural forward")
        record.update({k: None for k in LAYER_FIELDS})
        record["layer_ids"] = list(LAYER_IDS)
    return record


def representation_header(lock_path, lock):
    return {
        "schema": "chipmunk.audio-layer-representations/12",
        "lock_sha256": digest(lock_path),
        "original_receipt_sha256": prior.reuse.RECEIPT_SHA,
        "weights": lock["weights"],
        "runtime": lock["runtime"],
        "layer_ids": list(LAYER_IDS),
        "planned": 4248,
    }


def extract(lock_path, output):
    lock, (body, reference, rows, _, _) = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Commercial neural checkpoint bytes changed")
    header = representation_header(lock_path, lock)
    completed = prior.original.prior.open_journal(output, header, rows)
    for record, expected in zip(completed, rows, strict=False):
        validate_original(record, expected, reference)
    if len(completed) == len(rows):
        print(json.dumps({"complete": len(completed), "new_forwards": 0}), flush=True)
        return
    model, activity, captured, hook = load_models(reference)
    try:
        with LayerCapture(model) as layers, output.open("a") as stream:
            for index, row in enumerate(rows[len(completed) :], len(completed)):
                start = time.monotonic()
                record = {k: row[k] for k in ("id", "role", "sha256", "pcm_sha256")}
                try:
                    pcm = prior.original.prior.input_pcm(body, row)
                    record.update(extract_one(pcm, model, activity, captured, layers))
                    validate_original(record, row, reference)
                except Exception as error:
                    record.update(status="failed", error_type=type(error).__name__)
                record["runtime_s"] = time.monotonic() - start
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 25 == 0 or record["status"] == "failed" or index + 1 == len(rows):
                    print(
                        json.dumps(
                            {"completed": index + 1, "planned": len(rows), "status": record["status"]}
                        ),
                        flush=True,
                    )
                if record["status"] == "failed":
                    raise ValueError("Source/layer integrity failure; retain prefix and stop before fitting")
    finally:
        hook.remove()


def feature_rows(lock_path, receipt):
    lock, result = check(lock_path)
    _, reference, expected, _, _ = result
    header, records = load_rows(receipt)
    if header != representation_header(lock_path, lock) or [r["id"] for r in records] != [
        r["id"] for r in expected
    ]:
        raise ValueError("Every original layer feature record in bound order required")
    rows = []
    for record, source in zip(records, expected, strict=True):
        validate_original(record, source, reference)
        rows.append({**source, **record})
    return lock, result, rows


def train(lock_path, receipt, output):
    _, _, rows = feature_rows(lock_path, receipt)
    output.mkdir(parents=True, exist_ok=False)
    fitted = {a: method.fit([r for r in rows if r["role"] == "fit"], a) for a in method.ARMS}
    write(
        output / "fit.json",
        {
            "lock_sha256": digest(lock_path),
            "receipt_sha256": digest(receipt),
            "fitted": fitted,
            "deployment_eligible": False,
        },
    )
    guards = {a: method.calibrate([r for r in rows if r["role"] == "cal"], fitted[a]) for a in method.ARMS}
    write(
        output / "head.json",
        {
            "lock_sha256": digest(lock_path),
            "receipt_sha256": digest(receipt),
            "fit_sha256": digest(output / "fit.json"),
            "fitted": fitted,
            "guards": guards,
            "deployment_eligible": False,
        },
    )
    print(json.dumps({"head_sha256": digest(output / "head.json"), "arms": list(method.ARMS)}), flush=True)


def read_head(lock_path, receipt, path):
    head = json.loads(path.read_text())
    fit_record = json.loads((path.parent / "fit.json").read_text())
    if (
        head["lock_sha256"] != digest(lock_path)
        or fit_record["lock_sha256"] != digest(lock_path)
        or head["receipt_sha256"] != digest(receipt)
        or fit_record["receipt_sha256"] != digest(receipt)
        or head["fit_sha256"] != digest(path.parent / "fit.json")
        or head["fitted"] != fit_record["fitted"]
        or set(head["fitted"]) != set(method.ARMS)
        or set(head["guards"]) != set(method.ARMS)
        or head["deployment_eligible"] is not False
        or fit_record["deployment_eligible"] is not False
    ):
        raise ValueError("Original fitting-before-calibration/source artifact changed")
    return head


def probe_forward(pcm, model, captured, layers, reference):
    layers.clear()
    raw, feature = capture(model, pcm, captured)
    record = {
        "status": "computed",
        "raw": raw,
        "embedding": feature.tolist(),
        "embedding_sha256": prior.original.learner.digest_embedding(feature),
        **layers.take(),
    }
    prior.original.validate_feature(
        {**record, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]}, reference
    )
    validate_layers({**record, "status": "ok"})
    return record


def compare(lock_path, receipt, head_path, output):
    lock, (body, reference, _, baseline, old_probes), all_rows = feature_rows(lock_path, receipt)
    if weights() != lock["weights"]:
        raise ValueError("Commercial neural checkpoint bytes changed")
    head = read_head(lock_path, receipt, head_path)
    rows = [r for r in all_rows if r["role"] == "development"]
    header = {
        "schema": "chipmunk.audio-layer-responses/12",
        "lock_sha256": digest(lock_path),
        "receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "planned": 1449,
    }
    completed = prior.original.prior.open_journal(output, header, rows)
    model, hook, layers, captured = None, None, None, []
    try:
        with output.open("a") as stream:
            for index, row in enumerate(rows[len(completed) :], len(completed)):
                old = baseline[row["id"]]
                result = {k: row[k] for k in META}
                result.update(
                    status=row["status"],
                    original_p=old["original_p"],
                    original_decision=old["original_decision"],
                )
                probe = {
                    "status": "not_required",
                    "raw": None,
                    "embedding": None,
                    "p": {a: None for a in method.ARMS},
                }
                try:
                    pcm = prior.original.prior.input_pcm(body, row)
                    predicted = {a: method.predict(row, head["fitted"][a]) for a in method.ARMS}
                    if any(initial(predicted[a], head["guards"][a]) == "original" for a in method.ARMS):
                        band = band_view(pcm)
                        checksum = hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                        if model is None:
                            model, _, captured, hook = load_models(reference)
                            layers = LayerCapture(model)
                        probe.update(
                            probe_forward(band, model, captured, layers, reference),
                            filtered_pcm_sha256=checksum,
                        )
                        cached = old_probes.get(row["id"])
                        if cached is not None and any(
                            probe[k] != cached[k]
                            for k in ("raw", "embedding", "embedding_sha256", "filtered_pcm_sha256")
                        ):
                            raise ValueError("New layer probe changed known original raw/final-vector/PCM")
                        probe["prior_final_probe_parity_checked"] = cached is not None
                        probe["p"] = {a: method.predict(probe, head["fitted"][a]) for a in method.ARMS}
                    result["candidates"] = {}
                    for a in method.ARMS:
                        detail = final(
                            predicted[a], probe["p"][a], row["raw"], probe["raw"], head["guards"][a]
                        )
                        result["candidates"][a] = {
                            "p": predicted[a],
                            "decision": detail["decision"],
                            "detail": detail,
                        }
                except Exception as error:
                    result.update(
                        status="failed",
                        error_type=type(error).__name__,
                        original_p=None,
                        original_decision="inconclusive",
                        candidates={a: {"p": None, "decision": "inconclusive"} for a in method.ARMS},
                    )
                    probe.update(status="failed", error_type=type(error).__name__)
                result["probe"] = probe
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 50 == 0 or result["status"] == "failed" or index + 1 == len(rows):
                    print(
                        json.dumps(
                            {
                                "completed": index + 1,
                                "planned": len(rows),
                                "status": result["status"],
                                "probe": probe["status"],
                            }
                        ),
                        flush=True,
                    )
    finally:
        if layers is not None:
            layers.close()
        if hook is not None:
            hook.remove()


def bound_responses(lock_path, receipt, head_path, response_path):
    lock, result, rows = feature_rows(lock_path, receipt)
    head = read_head(lock_path, receipt, head_path)
    header, responses = load_rows(response_path)
    expected = [r for r in rows if r["role"] == "development"]
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["receipt_sha256"] != digest(receipt)
        or header["head_sha256"] != digest(head_path)
        or [r["id"] for r in responses] != [r["id"] for r in expected]
    ):
        raise ValueError("Every bound original development outcome required")
    for row, source in zip(responses, expected, strict=True):
        if any(row[k] != source[k] for k in META):
            raise ValueError("Outcome source/truth/group binding differs")
    return lock, result, rows, head, responses


def review(lock_path, receipt, head_path, response_path, output):
    _, (_, reference, _, baseline, old_probes), rows, head, responses = bound_responses(
        lock_path, receipt, head_path, response_path
    )
    cal = [r for r in rows if r["role"] == "cal"]
    if not same_tree({a: method.calibrate(cal, head["fitted"][a]) for a in method.ARMS}, head["guards"]):
        raise ValueError("Independent calibration inputs/guards changed")
    by_id = {r["id"]: r for r in rows}
    for row in responses:
        if row["status"] == "failed":
            if (
                row["original_p"] is not None
                or row["original_decision"] != "inconclusive"
                or any(
                    c["p"] is not None or c["decision"] != "inconclusive" for c in row["candidates"].values()
                )
            ):
                raise ValueError("Failed cases cannot establish directional conclusions")
            continue
        source = by_id[row["id"]]
        if row["status"] != source["status"]:
            raise ValueError("Original assessment status changed")
        predicted = {a: method.predict(source, head["fitted"][a]) for a in method.ARMS}
        needed = any(initial(predicted[a], head["guards"][a]) == "original" for a in method.ARMS)
        probe = row["probe"]
        if probe["status"] not in (("computed", "failed") if needed else ("not_required",)):
            raise ValueError("Missing or unexpected original-stability probe")
        if probe["status"] == "computed":
            prior.original.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]}, reference
            )
            validate_layers({**probe, "status": "ok"})
            cached = old_probes.get(row["id"])
            if cached is not None and any(
                probe[k] != cached[k] for k in ("raw", "embedding", "embedding_sha256", "filtered_pcm_sha256")
            ):
                raise ValueError("Known original filtered probe changed")
        for arm in method.ARMS:
            band = method.predict(probe, head["fitted"][arm]) if probe["status"] == "computed" else None
            expected = final(predicted[arm], band, source["raw"], probe["raw"], head["guards"][arm])
            if (
                not same_tree(
                    row["candidates"][arm],
                    {"p": predicted[arm], "decision": expected["decision"], "detail": expected},
                )
                or row["probe"]["p"][arm] != band
            ):
                raise ValueError("Mapped probability or raw/mapped stability conclusion differs")
    summaries = {}
    for arm in method.ARMS:
        flat = [
            {
                **r,
                "candidate_p": r["candidates"][arm]["p"],
                "candidate_decision": r["candidates"][arm]["decision"],
            }
            for r in responses
        ]
        for row in flat:
            if row["status"] != "failed" and any(
                row[k] != baseline[row["id"]][k] for k in ("original_p", "original_decision")
            ):
                raise ValueError("Actual current-service baseline changed")
        summaries[arm] = summarize(flat, 161)
    value = {
        "schema": "chipmunk.audio-layer-summary/12",
        "lock_sha256": digest(lock_path),
        "receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "responses_sha256": digest(response_path),
        "arms": summaries,
        "selected_arm_for_possible_confirmation": method.select(summaries),
        "deployment_eligible": False,
    }
    write(output, value)
    print(
        json.dumps(
            {
                "gates": {a: s["gates"] for a, s in summaries.items()},
                "selected_arm_for_possible_confirmation": value["selected_arm_for_possible_confirmation"],
            }
        ),
        flush=True,
    )


def native(lock_path, receipt, head_path, response_path, output):
    lock, (body, reference, _, _, _), all_rows, head, responses = bound_responses(
        lock_path, receipt, head_path, response_path
    )
    if weights() != lock["weights"]:
        raise ValueError("Commercial checkpoints changed")
    by_id = {r["id"]: r for r in all_rows}
    outcomes = {r["id"]: r for r in responses}
    probe_ids = {r["id"] for r in responses if r["probe"]["status"] == "computed"}
    identifiers = sorted(set(lock["native_ids"]) | probe_ids)
    model, activity, captured, hook = load_models(reference)
    cases = []
    try:
        with LayerCapture(model) as layers:
            for index, identifier in enumerate(identifiers):
                row = by_id[identifier]
                expected = outcomes[identifier]
                record = {
                    "id": identifier,
                    "sha256": row["sha256"],
                    "pcm_sha256": row["pcm_sha256"],
                    "predetermined_cell": identifier in lock["native_ids"],
                    "new_probe_case": identifier in probe_ids,
                }
                try:
                    pcm = prior.original.prior.input_pcm(body, row)
                    actual = extract_one(pcm, model, activity, captured, layers)
                    validate_original(
                        {**actual, **{k: row[k] for k in ("id", "role", "sha256", "pcm_sha256")}},
                        row,
                        reference,
                    )
                    record["original_passed"] = same_native(actual, row) and all(
                        actual[k] == row[k] for k in LAYER_FIELDS
                    )
                    record["mapped_original_passed"] = all(
                        method.predict(actual, head["fitted"][a]) == expected["candidates"][a]["p"]
                        for a in method.ARMS
                    )
                    probe = expected["probe"]
                    record.update(probe_status=probe["status"], probe_passed=None)
                    if probe["status"] == "computed":
                        band = band_view(pcm)
                        fresh = probe_forward(band, model, captured, layers, reference)
                        record["probe_passed"] = (
                            all(fresh[k] == probe[k] for k in ("raw", "embedding_sha256", *LAYER_FIELDS))
                            and hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                            == probe["filtered_pcm_sha256"]
                            and all(
                                method.predict(fresh, head["fitted"][a]) == probe["p"][a] for a in method.ARMS
                            )
                        )
                    record["passed"] = bool(
                        record["original_passed"]
                        and record["mapped_original_passed"]
                        and record["probe_passed"] is not False
                        and probe["status"] != "failed"
                    )
                except Exception as error:
                    record.update(passed=False, error_type=type(error).__name__)
                cases.append(record)
                if (index + 1) % 25 == 0 or not record["passed"] or index + 1 == len(identifiers):
                    print(
                        json.dumps(
                            {
                                "native_completed": index + 1,
                                "planned": len(identifiers),
                                "passed": record["passed"],
                            }
                        ),
                        flush=True,
                    )
    finally:
        hook.remove()
    value = {
        "schema": "chipmunk.audio-layer-native-review/12",
        "lock_sha256": digest(lock_path),
        "receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "responses_sha256": digest(response_path),
        "native_plan_sha256": PLAN_SHA,
        "predetermined_cells": 99,
        "new_probe_cases": len(probe_ids),
        "unique_cases": len(cases),
        "passed": all(r["passed"] for r in cases),
        "cases": cases,
        "scope": "Local original/layer/mask/mapped-output and every computed probe replay; extra probe cases audit provenance, not fresh accuracy or outside human validation.",
    }
    write(output, value)
    print(json.dumps({k: v for k, v in value.items() if k != "cases"}), flush=True)
    if not value["passed"]:
        raise ValueError("Native layer replay failed; retain all outcomes")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "extract", "train", "compare", "review", "native"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--responses", type=Path)
    a = p.parse_args()
    if a.stage in ("train", "compare", "review", "native") and a.receipt is None:
        p.error("Original complete layer receipt required")
    if a.stage in ("compare", "review", "native") and a.head is None:
        p.error("Original fitted head required")
    if a.stage in ("review", "native") and a.responses is None:
        p.error("Complete development responses required")
    if a.stage == "freeze":
        freeze(a.input, a.output)
    elif a.stage == "extract":
        extract(a.input, a.output)
    elif a.stage == "train":
        train(a.input, a.receipt, a.output)
    elif a.stage == "compare":
        compare(a.input, a.receipt, a.head, a.output)
    elif a.stage == "review":
        review(a.input, a.receipt, a.head, a.responses, a.output)
    else:
        native(a.input, a.receipt, a.head, a.responses, a.output)
