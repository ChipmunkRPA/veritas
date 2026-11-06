"""Predetermine and replay development native inputs without selecting by scores.

This adds verification, not a learner, threshold or advancement-gate change.
Publish the metadata plan before fitting/comparison; the source freeze is intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import run_audio_identity_v9 as execution
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_source_identity_v9 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import models, weights


def select(rows):
    cells = defaultdict(list)
    for row in rows:
        if row["role"] == "development":
            cells[row["dataset"], row["generator"], row["label"], row["condition"]].append(row["id"])
    if not cells:
        raise ValueError("Complete development metadata required")
    return sorted(min(ids) for ids in cells.values())


def same_native(actual, expected):
    keys = ("status", "raw", "speech_spans_s", "windows")
    if any(actual[k] != expected[k] for k in keys):
        return False
    return actual["status"] != "ok" or actual["embedding_sha256"] == expected["embedding_sha256"]


def plan(lock_path, output):
    lock, body = execution.check(lock_path)
    selected = select(body["variants"])
    if len(selected) != 99:
        raise ValueError("All99 fixed dataset/generator/truth/condition cells required")
    result = {
        "schema": "chipmunk.audio-identity-native-development-plan/9",
        "lock_sha256": digest(lock_path),
        "variants_sha256": lock["variants_sha256"],
        "auditor_code_sha256": digest(Path(__file__)),
        "selection": "Lexicographically first development input per dataset/generator/truth/condition cell; no response or applicability selection",
        "ids": selected,
        "planning_runs_inference": False,
        "planned_after_extraction_started": True,
        "required_before_fitting_and_comparison": True,
        "method_or_gates_changed": False,
        "deployment_eligible": False,
    }
    write(output, result)
    print(json.dumps({"plan_sha256": digest(output), "native_cases": len(selected)}), flush=True)


def run(lock_path, plan_path, receipt, responses, output):
    lock, body = execution.check(lock_path)
    selection = json.loads(plan_path.read_text())
    if (
        selection["lock_sha256"] != digest(lock_path)
        or selection["variants_sha256"] != lock["variants_sha256"]
        or selection["auditor_code_sha256"] != digest(Path(__file__))
        or selection["ids"] != select(body["variants"])
        or weights() != lock["weights"]
    ):
        raise ValueError("Predetermined native selection, code or verified weights changed")
    _, header, records = execution.feature_rows(lock_path, receipt)
    by_id = {r["id"]: r for r in records}
    response_header, results = load_rows(responses)
    expected = [r for r in records if r["role"] == "development"]
    if (
        response_header["lock_sha256"] != digest(lock_path)
        or response_header["receipt_sha256"] != digest(receipt)
        or [r["id"] for r in results] != [r["id"] for r in expected]
    ):
        raise ValueError("Every development response must remain bound")
    outcomes = {r["id"]: r for r in results}
    model, activity = models()
    if (
        not np.array_equal(
            model.head.weight.detach().cpu().numpy()[0], np.asarray(header["head_weight"], np.float32)
        )
        or float(model.head.bias.detach().cpu()[0]) != header["head_bias"]
    ):
        raise ValueError("Native pretrained checkpoint head changed")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    cases = []
    try:
        for identifier in selection["ids"]:
            row = by_id[identifier]
            result = {"id": identifier, "sha256": row["sha256"], "pcm_sha256": row["pcm_sha256"]}
            try:
                pcm = execution.prior.input_pcm(body, row)
                actual = execution.extract_one(pcm, model, activity, captured)
                result.update(
                    original_passed=same_native(actual, row),
                    probe_status=outcomes[identifier]["probe"]["status"],
                )
                probe = outcomes[identifier]["probe"]
                result["probe_passed"] = None
                if probe["status"] == "computed":
                    band = band_view(pcm)
                    raw, feature = capture(model, band, captured)
                    result["probe_passed"] = (
                        raw == probe["raw"]
                        and execution.learner.digest_embedding(feature) == probe["embedding_sha256"]
                        and hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                        == probe["filtered_pcm_sha256"]
                    )
                result["passed"] = bool(
                    result["original_passed"]
                    and result["probe_passed"] is not False
                    and probe["status"] != "failed"
                )
            except Exception as error:
                result.update(passed=False, error_type=type(error).__name__)
            cases.append(result)
    finally:
        hook.remove()
    result = {
        "schema": "chipmunk.audio-identity-native-development-review/9",
        "lock_sha256": digest(lock_path),
        "plan_sha256": digest(plan_path),
        "receipt_sha256": digest(receipt),
        "responses_sha256": digest(responses),
        "passed": len(cases) == 99 and all(r["passed"] for r in cases),
        "cases": cases,
        "scope": "Local99-cell native source/VAD/window/response/embedding and any computed-probe replay. No fresh confirmation or outside reviewer participation.",
    }
    write(output, result)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "cases": len(cases),
                "computed_probes": sum(r.get("probe_status") == "computed" for r in cases),
            }
        ),
        flush=True,
    )
    if not result["passed"]:
        raise ValueError("Native development replay failed; preserve the receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run"))
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--responses", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        plan(args.lock, args.output)
    else:
        if any(x is None for x in (args.plan, args.receipt, args.responses)):
            parser.error(
                "Native replay requires the published plan, complete features and complete responses"
            )
        run(args.lock, args.plan, args.receipt, args.responses, args.output)
