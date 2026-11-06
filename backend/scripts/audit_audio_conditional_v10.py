"""Replay the predetermined99 native cells and every newly computed v10 probe.

Additional new-probe cases are an exhaustive provenance audit, not a selected
accuracy panel. No method, threshold, acceptance gate or model default changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts import audio_conditional_v10 as method
from scripts import run_audio_conditional_v10 as execution
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_source_identity_v9 import digest
from scripts.audit_audio_identity_forwards_v9 import same_native, select
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows
from scripts.score_audio_confirmation_v7 import models, weights

PLAN_SHA = "48795283220152d5de07356ff54c018b8da797749957b6480d27582de97eed1e"


def run(lock_path, head_path, responses_path, output):
    lock, (body, reference, all_rows, _, _) = execution.check(lock_path)
    project = Path(__file__).resolve().parents[2]
    plan_path = project / "docs/evaluation-panels/audio-v9-native-development-plan.json"
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != PLAN_SHA or plan["ids"] != select(all_rows) or weights() != lock["weights"]:
        raise ValueError("Original native selection or verified model bytes changed")
    header, responses = load_rows(responses_path)
    head = json.loads(head_path.read_text())
    rows = {r["id"]: r for r in all_rows if r["role"] == "development"}
    if (
        header["head_sha256"] != digest(head_path)
        or header["lock_sha256"] != digest(lock_path)
        or [r["id"] for r in responses] != list(rows)
        or head["lock_sha256"] != digest(lock_path)
    ):
        raise ValueError("Complete bound conditional outcomes required")
    outcomes = {r["id"]: r for r in responses}
    new_probes = {r["id"] for r in responses if r["probe"].get("reused_v9_probe") is False}
    identifiers = sorted(set(plan["ids"]) | new_probes)
    model, activity = models()
    if (
        model.head.weight.detach().cpu().numpy()[0].tolist() != reference["head_weight"]
        or float(model.head.bias.detach().cpu()[0]) != reference["head_bias"]
    ):
        raise ValueError("Live pretrained head changed")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    cases = []
    try:
        for identifier in identifiers:
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
                record.update(
                    original_passed=same_native(actual, row),
                    quality_passed=execution.quality(pcm) == row["quality"],
                )
                probe = outcomes[identifier]["probe"]
                record.update(probe_status=probe["status"], probe_passed=None)
                if probe["status"] == "computed":
                    band = band_view(pcm)
                    raw, feature = capture(model, band, captured)
                    q = execution.quality(band)
                    probe_row = {"raw": raw, "embedding": feature.tolist(), "quality": q}
                    predicted = {
                        a: method.predict_rows([probe_row], head["fitted"], a)[0] for a in method.ARMS
                    }
                    record["probe_passed"] = (
                        raw == probe["raw"]
                        and execution.original.learner.digest_embedding(feature) == probe["embedding_sha256"]
                        and q == probe["quality"]
                        and predicted == probe["p"]
                        and hashlib.sha256(np.asarray(band, "<f4").tobytes()).hexdigest()
                        == probe["filtered_pcm_sha256"]
                    )
                record["passed"] = bool(
                    record["original_passed"]
                    and record["quality_passed"]
                    and record["probe_passed"] is not False
                    and probe["status"] != "failed"
                )
            except Exception as error:
                record.update(passed=False, error_type=type(error).__name__)
            cases.append(record)
    finally:
        hook.remove()
    result = {
        "schema": "chipmunk.audio-conditional-native-review/10",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "responses_sha256": digest(responses_path),
        "plan_sha256": PLAN_SHA,
        "audit_code_sha256": digest(Path(__file__)),
        "passed": all(r["passed"] for r in cases),
        "predetermined_cells": 99,
        "new_probe_cases": len(new_probes),
        "unique_cases": len(cases),
        "cases": cases,
        "scope": "Local native re-execution of99 predetermined cells plus every newly computed probe, including actual quality and mapped probe outputs. Additional cases audit complete provenance, not independent accuracy or outside participation.",
    }
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)
    if not result["passed"]:
        raise ValueError("Native conditional replay failed; preserve all outcomes")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("lock", "head", "responses", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.lock, a.head, a.responses, a.output)
