"""Freeze, extract consumed inputs, fit/calibrate separately and compare service.

Fresh confirmation is deliberately refused here. Its later runner must bind a
passing development decision and the resulting immutable fitted head first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit

from app.detectors.audio import calibration as serving
from app.detectors.audio.decode import decode_audio
from app.engine.boundary import Deadline
from scripts import audio_multisource_v8 as method
from scripts import audio_residual_head_v6 as learner
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_guard_v7 import initial
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.plan_audio_multisource_v8 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import implementation as prior_implementation
from scripts.score_audio_confirmation_v7 import models, runtime, speech_windows, weights


def implementation():
    project = Path(__file__).resolve().parents[2]
    names = set(prior_implementation()) | {
        "backend/scripts/audio_multisource_v8.py",
        "backend/scripts/run_audio_multisource_v8.py",
        "backend/scripts/evaluate_audio_multisource_v8.py",
        "backend/scripts/plan_audio_multisource_v8.py",
        "backend/scripts/prepare_audio_multisource_v8.py",
        "backend/scripts/prepare_audio_multisource_variants_v8.py",
        "backend/scripts/materialize_background_audio_v8.py",
        "backend/scripts/materialize_reserved_audio_v8.py",
        "backend/scripts/acquire_genuine_audio_v8.py",
        "backend/scripts/acquire_genuine_parquet_v8.py",
        "backend/scripts/export_audio_multisource_v8.py",
        "backend/scripts/review_audio_multisource_plan_v8.py",
        "docs/audio-multisource-v8-development.md",
    }
    return {name: digest(project / name) for name in sorted(names)}


def validate_variants(body):
    rows = body["variants"]
    if body["panel"] != "development" or any(r["role"] not in ("fit", "cal", "development") for r in rows):
        raise ValueError("This runner cannot inspect fresh confirmation")
    if len({r["id"] for r in rows}) != len(rows) or Counter(r["role"] for r in rows) != {
        "fit": 1971,
        "cal": 891,
        "development": 1422,
    }:
        raise ValueError("Every frozen consumed input and role required")
    if any(r["background_role"] not in (None, r["role"]) for r in rows):
        raise ValueError("Backgrounds cannot cross frozen roles")


def freeze(root, output):
    path = root / "development/variants.json"
    body = json.loads(path.read_text())
    validate_variants(body)
    project = Path(__file__).resolve().parents[2]
    for key, name in (
        ("preparation_code_sha256", "prepare_audio_multisource_variants_v8.py"),
        ("mixture_code_sha256", "prepare_audio_confirmation_variants_v7.py"),
    ):
        if body[key] != digest(project / "backend/scripts" / name):
            raise ValueError("Exact variant implementation changed")
    if body["prepared_sources_sha256"] != digest(root / "prepared-sources.json"):
        raise ValueError("Prepared originals changed")
    for row in body["variants"]:
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError("Frozen waveform changed")
    write(
        output,
        {
            "schema": "chipmunk.audio-multisource-development-lock/8",
            "variants_path": str(path.resolve()),
            "variants_sha256": digest(path),
            "prepared_sources_sha256": body["prepared_sources_sha256"],
            "reserved_confirmation_variants_sha256": digest(root / "confirmation/variants.json"),
            "method": method.METHOD,
            "implementation_sha256": implementation(),
            "runtime": runtime(),
            "weights": weights(),
            "serving_calibration_sha256": serving.SHA256,
            "serving_bundle": serving.load(),
            "planned": len(body["variants"]),
            "deployment_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "lock_sha256": digest(output),
                "planned": len(body["variants"]),
                "fitting_started": False,
                "detector_inference_run": False,
            }
        ),
        flush=True,
    )


def check(lock_path):
    lock = json.loads(lock_path.read_text())
    if (
        lock["implementation_sha256"] != implementation()
        or lock["runtime"] != runtime()
        or lock["method"] != method.METHOD
    ):
        raise ValueError("Frozen method/implementation/runtime changed")
    if lock["serving_calibration_sha256"] != serving.SHA256 or serving.load() != lock["serving_bundle"]:
        raise ValueError("Actual-service calibration changed")
    path = Path(lock["variants_path"])
    if digest(path) != lock["variants_sha256"]:
        raise ValueError("Exact input manifest changed")
    body = json.loads(path.read_text())
    validate_variants(body)
    return lock, body


def input_pcm(body, row):
    path = Path(body["source_root"]) / row["path"]
    if digest(path) != row["sha256"]:
        raise ValueError("Frozen input bytes changed")
    audio = decode_audio(path, source_sha256=row["sha256"])
    if len(audio.pcm) != row["samples"] or audio.mapping["pcm_sha256"] != row["pcm_sha256"]:
        raise ValueError("Frozen production PCM changed")
    return audio.pcm


def open_journal(output, header, rows):
    complete = []
    if output.exists():
        previous, complete = load_rows(output)
        if previous != header or [r["id"] for r in complete] != [r["id"] for r in rows[: len(complete)]]:
            raise ValueError("Cannot resume changed inputs or non-prefix outcomes")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")
    return complete


def extract(lock_path, output):
    lock, body = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Commercial checkpoint bytes changed")
    model, activity = models()
    direction = model.head.weight.detach().cpu().numpy()[0]
    bias = float(model.head.bias.detach().cpu()[0])
    header = {
        "schema": "chipmunk.audio-multisource-representations/8",
        "lock_sha256": digest(lock_path),
        "weights": lock["weights"],
        "runtime": lock["runtime"],
        "planned": lock["planned"],
        "head_weight": direction.tolist(),
        "head_bias": bias,
    }
    completed = open_journal(output, header, body["variants"])
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with output.open("a") as stream:
            for index, row in enumerate(body["variants"][len(completed) :], len(completed) + 1):
                started = time.monotonic()
                result = {k: row[k] for k in ("id", "role", "sha256", "pcm_sha256")}
                result.update(status="not_applicable", raw=None, embedding=None)
                try:
                    pcm = input_pcm(body, row)
                    spans, windows = speech_windows(pcm, activity)
                    if windows and (
                        len(windows) != 1 or windows[0][0] != 0 or len(windows[0][1]) != len(pcm)
                    ):
                        raise ValueError("Unexpected original speech window")
                    result.update(
                        speech_spans_s=[list(s) for s in spans],
                        windows=[{"start_sample": 0, "end_sample": len(chunk)} for _, chunk in windows],
                    )
                    if windows:
                        raw, feature = capture(model, windows[0][1], captured)
                        reconstructed = float(expit(feature.astype(float) @ direction + bias))
                        if not serving.valid(raw) or abs(raw - reconstructed) > 1e-6:
                            raise ValueError("Original head/representation parity failed")
                        result.update(
                            status="ok",
                            raw=raw,
                            embedding=feature.tolist(),
                            embedding_sha256=learner.digest_embedding(feature),
                            head_reconstructed_raw=reconstructed,
                        )
                except Exception as error:
                    result.update(status="failed", error_type=type(error).__name__, raw=None, embedding=None)
                result["runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if index % 30 == 0 or result["status"] == "failed" or index == lock["planned"]:
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "planned": lock["planned"],
                                "role": row["role"],
                                "status": result["status"],
                                "seconds": result["runtime_s"],
                            }
                        ),
                        flush=True,
                    )
    finally:
        hook.remove()


def feature_rows(lock_path, receipt):
    lock, body = check(lock_path)
    header, records = load_rows(receipt)
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["weights"] != lock["weights"]
        or [r["id"] for r in records] != [r["id"] for r in body["variants"]]
    ):
        raise ValueError("Every frozen input needs a terminal representation record")
    rows = []
    for row, record in zip(body["variants"], records, strict=True):
        if any(row[k] != record[k] for k in ("id", "role", "sha256", "pcm_sha256")):
            raise ValueError("Original role/source/PCM binding changed")
        if record["status"] not in ("ok", "not_applicable"):
            raise ValueError("A required representation failed; preserve it and stop fitting")
        if (
            record["status"] == "ok"
            and learner.digest_embedding(record["embedding"]) != record["embedding_sha256"]
        ):
            raise ValueError("Saved representation changed")
        if (record["raw"] is None) != (record["status"] == "not_applicable"):
            raise ValueError("Assessed status and response disagree")
        rows.append({**row, **record})
    return body, header, rows


def train(lock_path, receipt, output):
    _, header, rows = feature_rows(lock_path, receipt)
    output.mkdir(parents=True, exist_ok=False)
    fitted = method.fit([r for r in rows if r["role"] == "fit"], header["head_weight"])
    write(
        output / "fit.json",
        {
            "lock_sha256": digest(lock_path),
            "receipt_sha256": digest(receipt),
            "fitted": fitted,
            "deployment_eligible": False,
        },
    )
    guards = method.calibrate([r for r in rows if r["role"] == "cal"], fitted)
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
    print(
        json.dumps(
            {
                "head_sha256": digest(output / "head.json"),
                "fit_groups_by_class": fitted["fit_groups_by_class"],
                "cal_groups_by_class": [len(guards[str(c)]) for c in (0, 1)],
                "deployment_eligible": False,
            }
        ),
        flush=True,
    )


def compare_one(pcm, row, model, head, captured):
    raw = row["raw"]
    candidate_p = learner.predict(row, head["fitted"])
    windows = [(0.0, pcm)] if raw is not None else []
    probe = {"status": "not_required", "raw": None, "embedding": None, "p": None}

    class Recorder:
        def predict(self, chunks):
            if len(chunks) != 1:
                raise ValueError("One complete stability view required")
            probe["status"] = "failed"
            q, feature = capture(model, chunks[0], captured)
            if not serving.valid(q):
                raise ValueError("Nonfinite stability response")
            probe.update(
                status="computed",
                raw=q,
                embedding=feature.tolist(),
                embedding_sha256=learner.digest_embedding(feature),
                filtered_pcm_sha256=hashlib.sha256(np.asarray(chunks[0], "<f4").tobytes()).hexdigest(),
            )
            return [q]

    recorder = Recorder()
    baseline = serving.evaluate(raw, pcm, windows, recorder, Deadline(300))
    proposed = initial(candidate_p, head["guards"])
    if proposed == "original" and probe["status"] == "not_required":
        try:
            recorder.predict([band_view(pcm)])
        except Exception as error:
            probe.update(status="failed", error_type=type(error).__name__)
    if probe["status"] == "computed":
        probe["p"] = learner.predict({"raw": probe["raw"], "embedding": probe["embedding"]}, head["fitted"])
    candidate = method.final(candidate_p, probe["p"], raw, probe["raw"], head["guards"])
    return {
        "original_p": baseline["probability_ai"],
        "original_decision": baseline["decision"],
        "baseline_detail": baseline,
        "candidate_p": candidate_p,
        "candidate_decision": candidate["decision"],
        "candidate_detail": candidate,
        "probe": probe,
        "probe_needed_by_service": baseline["probe_status"] != "not_required_for_decision",
        "probe_needed_by_candidate": proposed == "original",
    }


def compare(lock_path, receipt, head_path, output):
    body, reference, all_rows = feature_rows(lock_path, receipt)
    head = json.loads(head_path.read_text())
    if head["lock_sha256"] != digest(lock_path) or head["receipt_sha256"] != digest(receipt):
        raise ValueError("Fitted head differs from frozen inputs/representations")
    lock, _ = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Commercial checkpoint bytes changed")
    rows = [r for r in all_rows if r["role"] == "development"]
    header = {
        "schema": "chipmunk.audio-multisource-development-responses/8",
        "lock_sha256": digest(lock_path),
        "receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "planned": len(rows),
    }
    completed = open_journal(output, header, rows)
    model, _ = models()
    if (
        not np.array_equal(
            model.head.weight.detach().cpu().numpy()[0], np.asarray(reference["head_weight"], np.float32)
        )
        or float(model.head.bias.detach().cpu()[0]) != reference["head_bias"]
    ):
        raise ValueError("Original frozen head changed")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with output.open("a") as stream:
            for index, row in enumerate(rows[len(completed) :], len(completed) + 1):
                started = time.monotonic()
                result = {k: v for k, v in row.items() if k not in ("path", "production_input", "embedding")}
                try:
                    result.update(compare_one(input_pcm(body, row), row, model, head, captured))
                except Exception as error:
                    result.update(
                        status="failed",
                        error_type=type(error).__name__,
                        original_p=None,
                        original_decision="inconclusive",
                        candidate_p=None,
                        candidate_decision="inconclusive",
                        probe={"status": "failed", "raw": None, "p": None, "embedding": None},
                    )
                result["comparison_runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if index % 30 == 0 or result["status"] == "failed" or index == len(rows):
                    print(
                        json.dumps({"completed": index, "planned": len(rows), "status": result["status"]}),
                        flush=True,
                    )
    finally:
        hook.remove()


def review(lock_path, receipt, head_path, responses, output):
    _, _, all_rows = feature_rows(lock_path, receipt)
    expected = [r for r in all_rows if r["role"] == "development"]
    header, actual = load_rows(responses)
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["head_sha256"] != digest(head_path)
        or header["receipt_sha256"] != digest(receipt)
        or [r["id"] for r in actual] != [r["id"] for r in expected]
    ):
        raise ValueError("Complete bound development outcomes required")
    for row, result in zip(expected, actual, strict=True):
        if any(
            row[k] != result[k]
            for k in ("sha256", "pcm_sha256", "role", "base_id", "group", "condition", "generator", "label")
        ):
            raise ValueError("Outcome binding changed")
    result = summarize(actual, 158)
    result.update(
        lock_sha256=digest(lock_path),
        receipt_sha256=digest(receipt),
        head_sha256=digest(head_path),
        responses_sha256=digest(responses),
    )
    write(output, result)
    print(
        json.dumps(
            {
                "gates": result["gates"],
                "advance_to_fresh_confirmation": result["advance_to_fresh_confirmation"],
                "deployment_eligible": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "extract", "train", "compare", "review"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--head", type=Path)
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.input, args.output)
    elif args.stage == "extract":
        extract(args.input, args.output)
    elif args.stage == "train" and args.receipt:
        train(args.input, args.receipt, args.output)
    elif args.stage == "compare" and args.receipt and args.head:
        compare(args.input, args.receipt, args.head, args.output)
    elif args.stage == "review" and args.receipt and args.head and args.responses:
        review(args.input, args.receipt, args.head, args.responses, args.output)
    else:
        parser.error("train needs --receipt; compare needs --receipt/--head; review also needs --responses")
