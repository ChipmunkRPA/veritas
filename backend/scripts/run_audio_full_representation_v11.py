"""Freeze, fit and compare one full-representation candidate, never fresh data.

Original v9 representations/service decisions and exact v9/v10 filtered probes
can be reused. All required new probes run the same commercial frozen backbone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration as serving
from scripts import audio_full_representation_v11 as method
from scripts import run_audio_conditional_v10 as reuse
from scripts import run_audio_identity_v9 as original
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_guard_v7 import initial
from scripts.audio_multisource_v8 import final
from scripts.audio_source_identity_v9 import digest
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import models, runtime, weights

CONDITIONAL_SHA = "a2a83749942f71df0ea923534537fae28a4b2e372a7a101b591e0097f262c06d"


def implementation():
    project = Path(__file__).resolve().parents[2]
    names = set(reuse.implementation()) | {
        "backend/scripts/audio_full_representation_v11.py",
        "backend/scripts/run_audio_full_representation_v11.py",
        "backend/scripts/export_audio_full_representation_v11.py",
        "docs/audio-full-representation-v11-protocol.md",
    }
    return {name: digest(project / name) for name in sorted(names)}


def prepared(source):
    body, reference, rows, baseline, _ = reuse.prior(source)
    path = source.parent / "audio-v10/responses.jsonl"
    if digest(path) != CONDITIONAL_SHA:
        raise ValueError("Exact completed v10 probe journal required")
    header, conditional = load_rows(path)
    development = [r for r in rows if r["role"] == "development"]
    if header["original_responses_sha256"] != reuse.RESPONSES_SHA or [r["id"] for r in conditional] != [
        r["id"] for r in development
    ]:
        raise ValueError("Complete original conditional input order required")
    probes = {k: v["probe"] for k, v in baseline.items() if v["probe"]["status"] == "computed"}
    for row in conditional:
        probe = row["probe"]
        if probe["status"] == "computed":
            original.validate_feature(
                {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]}, reference
            )
            if row["id"] in probes and any(
                probe[k] != probes[row["id"]][k]
                for k in ("raw", "embedding", "embedding_sha256", "filtered_pcm_sha256")
            ):
                raise ValueError("Cached probes disagree across original journals")
            probes[row["id"]] = probe
    return body, reference, rows, baseline, probes


def freeze(source, output):
    _, _, rows, _, probes = prepared(source)
    if output.exists() or (output.parent / "trained").exists():
        raise ValueError("Freeze requires a new, never fitted destination")
    write(
        output,
        {
            "schema": "chipmunk.audio-full-representation-lock/11",
            "source_root": str(source.resolve()),
            "method": method.METHOD,
            "implementation_sha256": implementation(),
            "runtime": runtime(),
            "weights": weights(),
            "original_receipt_sha256": reuse.RECEIPT_SHA,
            "original_projection_head_sha256": reuse.HEAD_SHA,
            "original_responses_sha256": reuse.RESPONSES_SHA,
            "conditional_responses_sha256": CONDITIONAL_SHA,
            "available_cached_probes": len(probes),
            "role_counts": dict(Counter(r["role"] for r in rows)),
            "reserved_confirmation_sha256": original.CONFIRMATION_SHA,
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
                "available_cached_probes": len(probes),
                "fitting_started": False,
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
        or lock["original_receipt_sha256"] != reuse.RECEIPT_SHA
        or lock["original_projection_head_sha256"] != reuse.HEAD_SHA
        or lock["original_responses_sha256"] != reuse.RESPONSES_SHA
        or lock["conditional_responses_sha256"] != CONDITIONAL_SHA
        or lock["role_counts"] != original.COUNTS
        or lock["reserved_confirmation_sha256"] != original.CONFIRMATION_SHA
    ):
        raise ValueError("Prospective method/runtime/implementation/source/service changed")
    result = prepared(Path(lock["source_root"]))
    if len(result[-1]) != lock["available_cached_probes"]:
        raise ValueError("Complete cached-probe inventory changed")
    return lock, result


def train(lock_path, output):
    _, (_, reference, rows, _, _) = check(lock_path)
    output.mkdir(parents=True, exist_ok=False)
    fitted = method.fit([r for r in rows if r["role"] == "fit"], reference["head_weight"])
    write(
        output / "fit.json",
        {"lock_sha256": digest(lock_path), "fitted": fitted, "deployment_eligible": False},
    )
    guards = method.calibrate([r for r in rows if r["role"] == "cal"], fitted)
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
                "coefficients": len(fitted["coefficients"]),
                "fit_groups_by_class": fitted["fit_groups_by_class"],
            }
        ),
        flush=True,
    )


def read_head(lock_path, path):
    head = json.loads(path.read_text())
    fit_record = json.loads((path.parent / "fit.json").read_text())
    if (
        head["lock_sha256"] != digest(lock_path)
        or fit_record["lock_sha256"] != digest(lock_path)
        or head["fit_sha256"] != digest(path.parent / "fit.json")
        or head["fitted"] != fit_record["fitted"]
    ):
        raise ValueError("Original fitting-before-calibration artifact changed")
    return head


def compare(lock_path, head_path, output):
    lock, (body, reference, all_rows, baseline, probes) = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Commercial pretrained checkpoint bytes changed")
    head = read_head(lock_path, head_path)
    rows = [r for r in all_rows if r["role"] == "development"]
    header = {
        "schema": "chipmunk.audio-full-representation-responses/11",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "original_responses_sha256": reuse.RESPONSES_SHA,
        "planned": 1449,
    }
    completed = original.prior.open_journal(output, header, rows)
    model, hook, captured = None, None, []
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
                    raw=row["raw"],
                    original_p=old["original_p"],
                    original_decision=old["original_decision"],
                )
                probe = {"status": "not_required", "raw": None, "embedding": None, "p": None}
                try:
                    pcm = original.prior.input_pcm(body, row)
                    p = method.predict(row, head["fitted"])
                    if initial(p, head["guards"]) == "original":
                        band = band_view(pcm)
                        checksum = hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                        cached = probes.get(row["id"])
                        if cached is not None:
                            if checksum != cached["filtered_pcm_sha256"]:
                                raise ValueError("Actual filtered probe PCM changed")
                            probe.update({k: cached[k] for k in ("raw", "embedding", "embedding_sha256")})
                            probe.update(status="computed", reused_exact_probe=True)
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
                                embedding_sha256=original.learner.digest_embedding(feature),
                                reused_exact_probe=False,
                            )
                        probe["filtered_pcm_sha256"] = checksum
                        original.validate_feature(
                            {**probe, "status": "ok", "windows": [{"start_sample": 0, "end_sample": 64000}]},
                            reference,
                        )
                        probe["p"] = method.predict(probe, head["fitted"])
                    detail = final(p, probe["p"], row["raw"], probe["raw"], head["guards"])
                    result.update(
                        candidate_p=p, candidate_decision=detail["decision"], candidate_detail=detail
                    )
                except Exception as error:
                    result.update(
                        status="failed",
                        error_type=type(error).__name__,
                        original_p=None,
                        original_decision="inconclusive",
                        candidate_p=None,
                        candidate_decision="inconclusive",
                    )
                    probe.update(status="failed", error_type=type(error).__name__)
                result["probe"] = probe
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if (index + 1) % 100 == 0 or result["status"] == "failed" or index + 1 == 1449:
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


def review(lock_path, head_path, response_path, output):
    _, (_, _, rows, _, _) = check(lock_path)
    read_head(lock_path, head_path)
    header, responses = load_rows(response_path)
    expected = [r for r in rows if r["role"] == "development"]
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["head_sha256"] != digest(head_path)
        or [r["id"] for r in responses] != [r["id"] for r in expected]
    ):
        raise ValueError("Every original development outcome in order required")
    for row, source in zip(responses, expected, strict=True):
        if any(
            row[k] != source[k]
            for k in (
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
                "raw",
            )
        ):
            raise ValueError("Complete source/truth/group/clock/raw binding differs")
    summary = summarize(responses, 161)
    summary.update(
        schema="chipmunk.audio-full-representation-summary/11",
        lock_sha256=digest(lock_path),
        head_sha256=digest(head_path),
        responses_sha256=digest(response_path),
    )
    write(output, summary)
    print(
        json.dumps(
            {
                "gates": summary["gates"],
                "advance_to_fresh_confirmation": summary["advance_to_fresh_confirmation"],
                "deployment_eligible": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "train", "compare", "review"))
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--responses", type=Path)
    a = p.parse_args()
    if a.stage in ("compare", "review") and a.head is None:
        p.error("Comparison/review requires the original fitted head")
    if a.stage == "review" and a.responses is None:
        p.error("Review requires the complete response journal")
    if a.stage == "freeze":
        freeze(a.input, a.output)
    elif a.stage == "train":
        train(a.input, a.output)
    elif a.stage == "compare":
        compare(a.input, a.head, a.output)
    else:
        review(a.input, a.head, a.responses, a.output)
