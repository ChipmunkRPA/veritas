"""Replay the new runner on consumed sources before any fresh classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.detectors.audio.decode import decode_audio
from scripts.audio_confirmation_v7 import HEAD_SHA256
from scripts.prepare_audio_confirmation_v7 import digest
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import load_rows
from scripts.score_audio_confirmation_v7 import implementation, models, score_one, weights


def audit(development, output):
    head_path = development / "trained/head.json"
    if digest(head_path) != HEAD_SHA256:
        raise ValueError("Fixed candidate head required")
    head = json.loads(head_path.read_text())
    manifest = json.loads((development / "manifest.json").read_text())
    _, prior_rows = load_rows(development / "development-scores.jsonl")
    prior = {r["id"]: r for r in prior_rows}
    sources = [r for r in manifest["variants"] if r["role"] == "development"]
    selected = []
    for condition in ("clean", "noise_0", "noise_5", "noise_15", "music_5", "music_15"):
        for label in (0, 1):
            selected.append(
                min(
                    [
                        r
                        for r in sources
                        if r["condition"] == condition and r["label"] == label and r["raw"] is not None
                    ],
                    key=lambda r: r["id"],
                )
            )
    selected.append(min([r for r in sources if r["raw"] is None], key=lambda r: r["id"]))
    verified = weights()
    model, activity = models()
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    records = []
    try:
        for row in selected:
            source = Path(manifest["source_root"]) / row["path"]
            if digest(source) != row["sha256"]:
                raise ValueError("Consumed input bytes changed")
            decoded = decode_audio(source, source_sha256=row["sha256"])
            if decoded.mapping["pcm_sha256"] != row["pcm_sha256"]:
                raise ValueError("Consumed production decoding differs")
            actual = score_one(decoded.pcm, model, activity, head, captured)
            expected = prior[row["id"]]
            errors = {}
            for key in ("raw", "original_p", "candidate_p"):
                a, b = actual[key], expected[key]
                if a is None or b is None:
                    if a != b:
                        raise ValueError("Consumed applicability mismatch")
                    errors[key] = 0.0
                else:
                    errors[key] = abs(a - b)
                    if errors[key] > 1e-7:
                        raise ValueError("Consumed raw/calibration probability differs")
            if any(actual[k] != expected[k] for k in ("original_decision", "candidate_decision")):
                raise ValueError("Consumed original/candidate guarded decision differs")
            expected_windows = [
                {k: w[k] for k in ("start_sample", "end_sample")} for w in row["original_windows"]
            ]
            if actual["windows"] != expected_windows:
                raise ValueError("Source speech gate/window differs from original app path")
            records.append(
                {
                    "id": row["id"],
                    "condition": row["condition"],
                    "label": row["label"],
                    "passed": True,
                    "errors": errors,
                    "status": actual["status"],
                    "original_decision": actual["original_decision"],
                    "candidate_decision": actual["candidate_decision"],
                }
            )
    finally:
        hook.remove()
    body = {
        "schema": "chipmunk.audio-fresh-runner-consumed-parity/7",
        "implementation_sha256": implementation(),
        "head_sha256": HEAD_SHA256,
        "source_manifest_sha256": digest(development / "manifest.json"),
        "prior_scores_sha256": digest(development / "development-scores.jsonl"),
        "weights": verified,
        "passed": len(records) == 13,
        "cases": records,
        "fresh_confirmation_inference_run": False,
        "scope": "Thirteen consumed source replays, six conditions/two classes plus one unassessed case. Actual app calibration and source windows; not fresh or outside testing.",
    }
    write(output, body)
    print(
        json.dumps(
            {
                "passed": body["passed"],
                "cases": len(records),
                "max_errors": {
                    k: max(r["errors"][k] for r in records) for k in ("raw", "original_p", "candidate_p")
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.development, args.output)
