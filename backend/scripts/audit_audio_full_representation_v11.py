"""Native re-execution of99 fixed development cells and all new v11 probes.

No case selection by score, fitting, confirmation scoring or serving change.
New-probe cases audit complete provenance, not a new accuracy panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import audio_full_representation_v11 as method
from scripts import run_audio_full_representation_v11 as execution
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_source_identity_v9 import digest
from scripts.audit_audio_identity_forwards_v9 import same_native, select
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import models, weights

PLAN_SHA = "48795283220152d5de07356ff54c018b8da797749957b6480d27582de97eed1e"


def run(lock_path, head_path, response_path, output):
    lock, (body, reference, all_rows, _, _) = execution.check(lock_path)
    project = Path(__file__).resolve().parents[2]
    plan_path = project / "docs/evaluation-panels/audio-v9-native-development-plan.json"
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != PLAN_SHA or plan["ids"] != select(all_rows) or weights() != lock["weights"]:
        raise ValueError("Original native plan and commercial checkpoint bytes required")
    head = execution.read_head(lock_path, head_path)
    header, responses = load_rows(response_path)
    rows = {r["id"]: r for r in all_rows if r["role"] == "development"}
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["head_sha256"] != digest(head_path)
        or [r["id"] for r in responses] != list(rows)
    ):
        raise ValueError("Every bound development outcome required")
    outcomes = {r["id"]: r for r in responses}
    new_probes = {r["id"] for r in responses if r["probe"].get("reused_exact_probe") is False}
    identifiers = sorted(set(plan["ids"]) | new_probes)
    model, activity = models()
    if (
        model.head.weight.detach().cpu().numpy()[0].tolist() != reference["head_weight"]
        or float(model.head.bias.detach().cpu()[0]) != reference["head_bias"]
    ):
        raise ValueError("Live original predictor head changed")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    cases = []
    try:
        for index, identifier in enumerate(identifiers):
            row = rows[identifier]
            record = {
                "id": identifier,
                "sha256": row["sha256"],
                "pcm_sha256": row["pcm_sha256"],
                "predetermined_cell": identifier in plan["ids"],
                "new_probe_case": identifier in new_probes,
            }
            try:
                pcm = execution.original.prior.input_pcm(body, row)
                actual = execution.original.extract_one(pcm, model, activity, captured)
                record["original_passed"] = same_native(actual, row)
                expected = outcomes[identifier]
                record["mapped_original_passed"] = (
                    method.predict(actual, head["fitted"]) == expected["candidate_p"]
                )
                probe = expected["probe"]
                record.update(probe_status=probe["status"], probe_passed=None)
                if probe["status"] == "computed":
                    band = band_view(pcm)
                    raw, feature = capture(model, band, captured)
                    record["probe_passed"] = (
                        raw == probe["raw"]
                        and execution.original.learner.digest_embedding(feature) == probe["embedding_sha256"]
                        and hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                        == probe["filtered_pcm_sha256"]
                        and method.predict({"raw": raw, "embedding": feature.tolist()}, head["fitted"])
                        == probe["p"]
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
    result = {
        "schema": "chipmunk.audio-full-representation-native-review/11",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "responses_sha256": digest(response_path),
        "plan_sha256": PLAN_SHA,
        "audit_code_sha256": digest(Path(__file__)),
        "passed": all(r["passed"] for r in cases),
        "predetermined_cells": len(plan["ids"]),
        "new_probe_cases": len(new_probes),
        "unique_cases": len(identifiers),
        "cases": cases,
        "scope": "Local native original/VAD/window/raw/vector/mapped-output and filtered-probe replay. Predetermined99 cells plus every new probe; additional cases audit provenance, not fresh accuracy or outside human participation.",
    }
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    if not result["passed"]:
        raise ValueError("Native replay failed; retain all outcomes")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "head", "responses", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.lock, a.head, a.responses, a.output)
