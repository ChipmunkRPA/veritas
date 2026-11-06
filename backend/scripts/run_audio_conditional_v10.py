"""Prepare/freeze/fit/compare the prespecified conditional response experiment.

Reuses exact audited v9 original/probe representations; no confirmation input is
accepted. Quality measurements do not change the waveforms sent to classifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration as serving
from scripts import audio_conditional_v10 as method
from scripts import run_audio_identity_v9 as original
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_guard_v7 import initial
from scripts.audio_multisource_v8 import final
from scripts.audio_quality_v5 import features as quality
from scripts.audio_source_identity_v9 import digest
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import models, runtime, weights

RECEIPT_SHA = "315c07c1eec95553760a671e0480fb99bb53669e6f5679be0a73a1077cac512d"
HEAD_SHA = "411c4322ac35037073f2998ef0757d80c5bd348413d509b302bca90bcdcc7519"
RESPONSES_SHA = "7dff2eb4259f97731bfde0b6e1f3e38a0fb60d9920431212eeb8f9790e8fac9e"
LICENSE_SHA = "fc8dc597df0e3f51da65e2a71021e7f92e0b481998ec572a53c03ac0441db040"


def software():
    distribution = importlib.metadata.distribution("scikit-learn")
    if distribution.version != "1.9.0":
        raise ValueError("Exact installed nonlinear-learner version required")
    license_file = next(f for f in distribution.files if str(f).endswith("/licenses/COPYING"))
    if digest(Path(distribution.locate_file(license_file))) != LICENSE_SHA:
        raise ValueError("Verified BSD3 software license changed")
    folder = Path(inspect.getfile(method.HistGradientBoostingClassifier)).parent
    pins = {
        "sklearn/ensemble/_hist_gradient_boosting/" + p.name: digest(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix in (".py", ".so")
    }
    return {
        "version": distribution.version,
        "license": "BSD-3-Clause",
        "license_sha256": LICENSE_SHA,
        "implementation_sha256": pins,
    }


def implementation():
    project = Path(__file__).resolve().parents[2]
    names = set(original.implementation()) | {
        "backend/scripts/audio_conditional_v10.py",
        "backend/scripts/run_audio_conditional_v10.py",
        "backend/scripts/audio_quality_v5.py",
        "docs/audio-conditional-v10-protocol.md",
    }
    return {name: digest(project / name) for name in sorted(names)}


def prior(root):
    if digest(root / "confirmation/variants.json") != original.CONFIRMATION_SHA:
        raise ValueError("Reserved confirmation inputs changed")
    development = root / "development"
    receipt, head, responses = (
        development / "representations.jsonl",
        development / "trained/head.json",
        development / "responses.jsonl",
    )
    if digest(receipt) != RECEIPT_SHA or digest(head) != HEAD_SHA or digest(responses) != RESPONSES_SHA:
        raise ValueError("Original audited v9 features/projection/responses changed")
    body, reference, rows = original.feature_rows(development / "lock.json", receipt)
    header, outcomes = load_rows(responses)
    expected = [r for r in rows if r["role"] == "development"]
    if (
        header["receipt_sha256"] != RECEIPT_SHA
        or header["head_sha256"] != HEAD_SHA
        or [r["id"] for r in outcomes] != [r["id"] for r in expected]
    ):
        raise ValueError("Complete bound baseline outcome cache required")
    for row, result in zip(expected, outcomes, strict=True):
        if any(row[k] != result[k] for k in ("role", "sha256", "pcm_sha256", "raw", "status")):
            raise ValueError("Retained baseline source/assessment changed")
        if result["probe"]["status"] == "computed":
            original.validate_feature(
                {**result["probe"], "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]},
                reference,
            )
    transform = json.loads(head.read_text())["fitted"]["transform"]
    return body, reference, rows, {r["id"]: r for r in outcomes}, transform


def prepare(source, output):
    body, _, rows, baseline, _ = prior(source)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "quality.jsonl"
    project = Path(__file__).resolve().parents[2]
    header = {
        "schema": "chipmunk.audio-conditional-quality/10",
        "original_receipt_sha256": RECEIPT_SHA,
        "original_responses_sha256": RESPONSES_SHA,
        "quality_code_sha256": digest(project / "backend/scripts/audio_quality_v5.py"),
        "planned": 4248,
    }
    completed = original.prior.open_journal(path, header, rows)
    with path.open("a") as stream:
        for index, row in enumerate(rows[len(completed) :], len(completed) + 1):
            result = {k: row[k] for k in ("id", "role", "sha256", "pcm_sha256")}
            try:
                pcm = original.prior.input_pcm(body, row)
                result.update(status="ok", quality=quality(pcm), cached_probe_quality=None)
                old = baseline.get(row["id"])
                if old is not None and old["probe"]["status"] == "computed":
                    band = band_view(pcm)
                    checksum = hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                    if checksum != old["probe"]["filtered_pcm_sha256"]:
                        raise ValueError("Cached original probe PCM changed")
                    result.update(cached_probe_quality=quality(band), cached_probe_pcm_sha256=checksum)
            except Exception as error:
                result.update(status="failed", error_type=type(error).__name__, quality=None)
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            if index % 100 == 0 or result["status"] == "failed" or index == 4248:
                print(
                    json.dumps({"quality_completed": index, "planned": 4248, "status": result["status"]}),
                    flush=True,
                )
    write(
        output / "prepared.json",
        {
            "schema": "chipmunk.audio-conditional-preparation/10",
            "prior_root": str(source.resolve()),
            "quality_sha256": digest(path),
            "planned": 4248,
            "new_neural_forwards": False,
            "fitting_started": False,
        },
    )


def quality_rows(root):
    prepared = json.loads((root / "prepared.json").read_text())
    if digest(root / "quality.jsonl") != prepared["quality_sha256"]:
        raise ValueError("Prepared signal measurements changed")
    body, reference, rows, baseline, transform = prior(Path(prepared["prior_root"]))
    header, measured = load_rows(root / "quality.jsonl")
    project = Path(__file__).resolve().parents[2]
    if (
        header["quality_code_sha256"] != digest(project / "backend/scripts/audio_quality_v5.py")
        or header["original_receipt_sha256"] != RECEIPT_SHA
        or header["original_responses_sha256"] != RESPONSES_SHA
        or [r["id"] for r in measured] != [r["id"] for r in rows]
    ):
        raise ValueError("Every unchanged measurement/input binding required")
    for row, measurement in zip(rows, measured, strict=True):
        if (
            any(row[k] != measurement[k] for k in ("role", "sha256", "pcm_sha256"))
            or measurement["status"] != "ok"
        ):
            raise ValueError("Failed or mismatched measurement blocks fitting")
        row.update(quality=measurement["quality"], cached_probe_quality=measurement["cached_probe_quality"])
        if (
            not isinstance(row["quality"], dict)
            or set(row["quality"]) != {"flatness", "floor"}
            or not all(serving.valid(value) for value in row["quality"].values())
        ):
            raise ValueError("Every planned quality measurement must be finite, including unassessed speech")
        for arm in method.ARMS:
            method.inputs(row, transform, arm)
    return body, reference, rows, baseline, transform


def freeze(root, output):
    _, _, rows, _, _ = quality_rows(root)
    write(
        output,
        {
            "schema": "chipmunk.audio-conditional-lock/10",
            "root": str(root.resolve()),
            "prepared_sha256": digest(root / "prepared.json"),
            "quality_sha256": digest(root / "quality.jsonl"),
            "original_receipt_sha256": RECEIPT_SHA,
            "original_projection_head_sha256": HEAD_SHA,
            "original_responses_sha256": RESPONSES_SHA,
            "method": method.METHOD,
            "implementation_sha256": implementation(),
            "runtime": runtime(),
            "software": software(),
            "weights": weights(),
            "serving_calibration_sha256": serving.SHA256,
            "serving_bundle": serving.load(),
            "role_counts": dict(Counter(r["role"] for r in rows)),
            "reserved_confirmation_sha256": original.CONFIRMATION_SHA,
            "fitting_started": False,
            "new_neural_forwards": False,
            "deployment_eligible": False,
        },
    )
    print(
        json.dumps({"lock_sha256": digest(output), "inputs": len(rows), "project_fitting_started": False}),
        flush=True,
    )


def check(path):
    lock = json.loads(path.read_text())
    root = Path(lock["root"])
    if (
        lock["method"] != method.METHOD
        or lock["implementation_sha256"] != implementation()
        or lock["runtime"] != runtime()
        or lock["software"] != software()
        or digest(root / "prepared.json") != lock["prepared_sha256"]
        or digest(root / "quality.jsonl") != lock["quality_sha256"]
        or lock["serving_calibration_sha256"] != serving.SHA256
        or lock["serving_bundle"] != serving.load()
    ):
        raise ValueError("Prospective conditional method/runtime/measurement/service changed")
    return lock, quality_rows(root)


def train(lock_path, output):
    lock, (_, _, rows, _, transform) = check(lock_path)
    if Counter(r["role"] for r in rows) != original.COUNTS:
        raise ValueError("Complete corrected consumed role matrix required")
    output.mkdir(parents=True, exist_ok=False)
    fitted, parity = method.fit([r for r in rows if r["role"] == "fit"], transform, parity_rows=rows)
    write(
        output / "fit.json",
        {
            "lock_sha256": digest(lock_path),
            "fitted": fitted,
            "native_fit_parity": parity,
            "deployment_eligible": False,
        },
    )
    guards = {
        arm: method.calibrate([r for r in rows if r["role"] == "cal"], fitted, arm) for arm in method.ARMS
    }
    write(
        output / "head.json",
        {
            "lock_sha256": digest(lock_path),
            "fit_sha256": digest(output / "fit.json"),
            "fitted": fitted,
            "guards": guards,
            "deployment_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "head_sha256": digest(output / "head.json"),
                "arms": list(method.ARMS),
                "native_fit_parity": {arm: p["max_absolute_error"] for arm, p in parity.items()},
                "software_license": lock["software"]["license"],
            }
        ),
        flush=True,
    )


def compare(lock_path, head_path, output):
    lock, (body, reference, all_rows, baseline, _) = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Commercial pretrained bytes changed")
    head = json.loads(head_path.read_text())
    fit_record = json.loads((head_path.parent / "fit.json").read_text())
    if (
        head["lock_sha256"] != digest(lock_path)
        or head["fit_sha256"] != digest(head_path.parent / "fit.json")
        or fit_record["fitted"] != head["fitted"]
    ):
        raise ValueError("Fitting/calibration artifact changed")
    rows = [r for r in all_rows if r["role"] == "development"]
    mapped = {arm: method.predict_rows(rows, head["fitted"], arm) for arm in method.ARMS}
    header = {
        "schema": "chipmunk.audio-conditional-development-responses/10",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "original_responses_sha256": RESPONSES_SHA,
        "planned": 1449,
    }
    completed = original.prior.open_journal(output, header, rows)
    model = None
    hook = None
    captured = []
    try:
        with output.open("a") as stream:
            for index, row in enumerate(rows[len(completed) :], len(completed)):
                old = baseline[row["id"]]
                result = {
                    k: row[k]
                    for k in (
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
                }
                result.update(
                    status=row["status"],
                    original_p=old["original_p"],
                    original_decision=old["original_decision"],
                    candidates={},
                )
                probe = {
                    "status": "not_required",
                    "raw": None,
                    "embedding": None,
                    "quality": None,
                    "p": {arm: None for arm in method.ARMS},
                }
                try:
                    pcm = original.prior.input_pcm(body, row)
                    needed = any(
                        initial(mapped[arm][index], head["guards"][arm]) == "original" for arm in method.ARMS
                    )
                    if needed:
                        band = band_view(pcm)
                        checksum = hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                        if old["probe"]["status"] == "computed":
                            if checksum != old["probe"]["filtered_pcm_sha256"] or row[
                                "cached_probe_quality"
                            ] != quality(band):
                                raise ValueError("Original cached probe/quality changed")
                            probe.update(
                                {k: old["probe"][k] for k in ("raw", "embedding", "embedding_sha256")}
                            )
                            probe.update(
                                status="computed", quality=row["cached_probe_quality"], reused_v9_probe=True
                            )
                        else:
                            probe["status"] = "failed"
                            if model is None:
                                model, _ = models()
                                if (
                                    model.head.weight.detach().cpu().numpy()[0].tolist()
                                    != reference["head_weight"]
                                    or float(model.head.bias.detach().cpu()[0]) != reference["head_bias"]
                                ):
                                    raise ValueError("Original neural head changed")
                                hook = model.head.register_forward_pre_hook(
                                    lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
                                )
                            raw, feature = capture(model, band, captured)
                            probe.update(
                                status="computed",
                                raw=raw,
                                embedding=feature.tolist(),
                                quality=quality(band),
                                embedding_sha256=original.learner.digest_embedding(feature),
                                reused_v9_probe=False,
                            )
                        probe["filtered_pcm_sha256"] = checksum
                        original.validate_feature(
                            {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]},
                            reference,
                        )
                        probe["p"] = {
                            arm: method.predict_rows([probe], head["fitted"], arm)[0] for arm in method.ARMS
                        }
                    for arm in method.ARMS:
                        p = mapped[arm][index]
                        detail = final(p, probe["p"][arm], row["raw"], probe["raw"], head["guards"][arm])
                        result["candidates"][arm] = {"p": p, "decision": detail["decision"], "detail": detail}
                except Exception as error:
                    result.update(
                        status="failed",
                        error_type=type(error).__name__,
                        original_p=None,
                        original_decision="inconclusive",
                    )
                    result["candidates"] = {
                        arm: {"p": None, "decision": "inconclusive"} for arm in method.ARMS
                    }
                    probe.update(status="failed", error_type=type(error).__name__)
                result["probe"] = probe
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 50 == 0 or result["status"] == "failed" or index + 1 == 1449:
                    print(
                        json.dumps(
                            {
                                "completed": index + 1,
                                "planned": 1449,
                                "status": result["status"],
                                "probe": probe["status"],
                            }
                        ),
                        flush=True,
                    )
    finally:
        if hook is not None:
            hook.remove()


def review(lock_path, head_path, responses, output):
    _, (_, _, expected, _, _) = check(lock_path)
    header, rows = load_rows(responses)
    development = [r for r in expected if r["role"] == "development"]
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["head_sha256"] != digest(head_path)
        or [r["id"] for r in rows] != [r["id"] for r in development]
    ):
        raise ValueError("Complete bound development responses required")
    for row, expected_row in zip(rows, development, strict=True):
        if any(
            row[key] != expected_row[key]
            for key in (
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
        ):
            raise ValueError("Development outcome source/truth/group/background changed")
    summaries = {}
    for arm in method.ARMS:
        flattened = [
            {
                **row,
                "candidate_p": row["candidates"][arm]["p"],
                "candidate_decision": row["candidates"][arm]["decision"],
            }
            for row in rows
        ]
        summaries[arm] = summarize(flattened, 161)
    result = {
        "schema": "chipmunk.audio-conditional-development-summary/10",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "responses_sha256": digest(responses),
        "arms": summaries,
        "selected_arm_for_possible_confirmation": method.select(summaries),
        "deployment_eligible": False,
    }
    write(output, result)
    print(
        json.dumps(
            {
                "gates": {arm: summary["gates"] for arm, summary in summaries.items()},
                "selected_arm_for_possible_confirmation": result["selected_arm_for_possible_confirmation"],
                "deployment_eligible": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "freeze", "train", "compare", "review"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    if args.stage in ("compare", "review") and args.head is None:
        parser.error("Comparison and review require the fixed fitted head")
    if args.stage == "review" and args.responses is None:
        parser.error("Review requires the complete response receipt")
    if args.stage == "prepare":
        prepare(args.input, args.output)
    elif args.stage == "freeze":
        freeze(args.input, args.output)
    elif args.stage == "train":
        train(args.input, args.output)
    elif args.stage == "compare":
        compare(args.input, args.head, args.output)
    else:
        review(args.input, args.head, args.responses, args.output)
