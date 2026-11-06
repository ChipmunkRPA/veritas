"""Replay the18 source cells selected before fresh classification; never replace results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts.prepare_audio_confirmation_v7 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import load_rows
from scripts.score_audio_confirmation_v7 import check, models, score_one, weights


def audit(root, output):
    lock_path = root / "lock.json"
    lock = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Original checkpoint bytes changed")
    header, responses = load_rows(root / "scores.jsonl")
    if header["lock_sha256"] != digest(lock_path) or len(responses) != 1080:
        raise ValueError("Complete immutable source responses required")
    inputs = json.loads(Path(lock["variants_path"]).read_text())
    variants = {r["id"]: r for r in inputs["variants"]}
    saved = {r["id"]: r for r in responses}
    source_audit_path = root / "input-audit.json"
    source_audit = json.loads(source_audit_path.read_text())
    selected = [r["id"] for r in source_audit["cases"] if r["sampled_production_decode"]]
    if len(selected) != 18 or source_audit["variants_sha256"] != lock["variants_sha256"]:
        raise ValueError("Keep the18 pre-inference source/class/condition cells")
    model, activity = models()
    head = json.loads(Path(lock["head_path"]).read_text())
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    records, max_raw, max_embedding = [], 0.0, 0.0
    try:
        for identifier in selected:
            row, prior = variants[identifier], saved[identifier]
            path = root / row["path"]
            if digest(path) != row["sha256"]:
                raise ValueError("Preselected source bytes changed")
            decoded = decode_audio(path, source_sha256=row["sha256"])
            if decoded.mapping["pcm_sha256"] != row["pcm_sha256"]:
                raise ValueError("Preselected source PCM changed")
            actual = score_one(decoded.pcm, model, activity, head, captured)
            errors = {}
            for key in ("raw", "original_p", "candidate_p"):
                a, b = actual[key], prior[key]
                if a is None or b is None:
                    if a != b:
                        raise ValueError("Native replay applicability differs")
                    errors[key] = 0.0
                else:
                    errors[key] = abs(a - b)
                    if errors[key] > 1e-7:
                        raise ValueError("Native replay response/calibration differs")
            max_raw = max(max_raw, errors["raw"])
            for key in ("status", "original_decision", "candidate_decision", "windows", "speech_spans_s"):
                if actual[key] != prior[key]:
                    raise ValueError("Native gate/clock/decision differs")
            for a, b in (
                (actual["embedding"], prior["embedding"]),
                (actual["probe"]["embedding"], prior["probe"]["embedding"]),
            ):
                if a is None or b is None:
                    if a != b:
                        raise ValueError("Native representation missing")
                else:
                    error = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                    max_embedding = max(max_embedding, error)
                    if error > 1e-6:
                        raise ValueError("Native representation differs")
            for key in ("status", "filtered_pcm_sha256", "samples"):
                if actual["probe"].get(key) != prior["probe"].get(key):
                    raise ValueError("Native stability probe source differs")
            records.append(
                {
                    "id": identifier,
                    "passed": True,
                    "condition": row["condition"],
                    "label": row["label"],
                    "status": actual["status"],
                    "errors": errors,
                }
            )
    finally:
        hook.remove()
    result = {
        "schema": "chipmunk.audio-fresh-native-forward-audit/7",
        "lock_sha256": digest(lock_path),
        "scores_sha256": digest(root / "scores.jsonl"),
        "selection_receipt_sha256": digest(source_audit_path),
        "passed": len(records) == 18,
        "planned": 18,
        "cases": records,
        "max_raw_replay_error": max_raw,
        "max_embedding_replay_error": max_embedding,
        "scope": "Native replay of18 source/class/condition cells selected in the input audit before classifier inference, including retained unassessed cells. Not every neural forward, new evaluation or outside reviewer; original outcomes remain unchanged.",
    }
    write(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.root, args.output)
