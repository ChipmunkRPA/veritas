"""Replay every development PCM/probe clock and a fixed coverage sample of forwards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit

from scripts import audio_residual_head_v6 as learner
from scripts.prepare_audio_guard_v7 import check, write
from scripts.run_audio_guard_v7 import capture, digest, feature_rows, load_rows, setup, source_pcm


def audit(manifest, receipt, head_path, scores, output):
    m, representation, inputs = feature_rows(manifest, receipt)
    check(manifest)
    header, results = load_rows(scores)
    head = json.loads(head_path.read_text())
    expected = [r for r in inputs if r["role"] == "development"]
    if (
        header["manifest_sha256"] != digest(manifest)
        or header["head_sha256"] != digest(head_path)
        or [r["id"] for r in expected] != [r["id"] for r in results]
    ):
        raise ValueError("Complete frozen source/result panel required")
    cells = {}
    for row in results:
        if row["probe"]["status"] == "computed":
            key = (row["condition"], row["label"])
            if (
                key not in cells
                or hashlib.sha256(row["id"].encode()).digest()
                < hashlib.sha256(cells[key]["id"].encode()).digest()
            ):
                cells[key] = row
    selected = {r["id"] for r in cells.values()}
    model, weights = setup()
    if weights != representation["weights"]:
        raise ValueError("Model identity changed")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    records = []
    samples = probes = replays = 0
    max_raw_error = max_embedding_error = 0.0
    try:
        for source, row in zip(expected, results, strict=True):
            result = {
                "id": row["id"],
                "status": "failed",
                "probe_checked": False,
                "neural_probe_replay": False,
            }
            try:
                pcm = source_pcm(m, source)
                if row["pcm_sha256"] != source["pcm_sha256"] or row["sha256"] != source["sha256"]:
                    raise ValueError("Different source evidence")
                samples += len(pcm)
                probe = row["probe"]
                if source["raw"] is None and row["candidate_decision"] != "inconclusive":
                    raise ValueError("Unassessed speech became a directional verdict")
                if probe["status"] == "computed":
                    filtered = sosfiltfilt(
                        butter(4, [100.0, 3800.0], btype="bandpass", fs=16000, output="sos"),
                        np.asarray(pcm, np.float32),
                    ).astype(np.float32)
                    if (
                        len(filtered) != probe["samples"]
                        or len(filtered) != len(pcm)
                        or hashlib.sha256(filtered.astype("<f4").tobytes()).hexdigest()
                        != probe["filtered_pcm_sha256"]
                    ):
                        raise ValueError("Sensitivity view PCM/sample clock mismatch")
                    e = np.asarray(probe["embedding"], np.float64)
                    raw = float(
                        expit(e @ np.asarray(representation["head_weight"]) + representation["head_bias"])
                    )
                    if (
                        abs(raw - probe["raw"]) > 1e-6
                        or learner.digest_embedding(probe["embedding"]) != probe["embedding_sha256"]
                    ):
                        raise ValueError("Probe embedding/head response mismatch")
                    if (
                        learner.predict(
                            {**source, "raw": probe["raw"], "embedding": probe["embedding"]}, head["fitted"]
                        )
                        != probe["p"]
                    ):
                        raise ValueError("New probe probability differs")
                    probes += 1
                    result["probe_checked"] = True
                    if row["id"] in selected:
                        fresh_raw, fresh_e = capture(model, filtered, captured)
                        raw_error = abs(fresh_raw - probe["raw"])
                        embedding_error = float(np.max(np.abs(fresh_e - e)))
                        max_raw_error = max(max_raw_error, raw_error)
                        max_embedding_error = max(max_embedding_error, embedding_error)
                        if raw_error > 1e-7 or embedding_error > 1e-6:
                            raise ValueError("Sampled neural probe forward differs")
                        replays += 1
                        result["neural_probe_replay"] = True
                elif probe["status"] != "not_required":
                    raise ValueError("Required probe failed")
                result["status"] = "ok"
            except Exception as error:
                result["reason"] = f"{type(error).__name__}: {error}"
            records.append(result)
            if len(records) % 60 == 0:
                print(
                    json.dumps(
                        {"checked": len(records), "passed": sum(r["status"] == "ok" for r in records)}
                    ),
                    flush=True,
                )
    finally:
        hook.remove()
    body = {
        "schema": "chipmunk.audio-source-guard-source-audit/7",
        "manifest_sha256": digest(manifest),
        "head_sha256": digest(head_path),
        "scores_sha256": digest(scores),
        "implementation_sha256": digest(Path(__file__)),
        "planned": 720,
        "passed": sum(r["status"] == "ok" for r in records),
        "original_pcm_samples": samples,
        "exact_probe_views": probes,
        "sampled_neural_probe_forwards": replays,
        "planned_neural_probe_forwards": len(selected),
        "max_raw_replay_error": max_raw_error,
        "max_embedding_replay_error": max_embedding_error,
        "cases": records,
        "scope": "Every development source/PCM, all computed filtered sample clocks/hashes and stored probe arithmetic; one deterministic forward per nonempty condition/true-class probe cell. Not every neural forward, independent label truth or outside review.",
    }
    write(output, body)
    print(json.dumps({k: v for k, v in body.items() if k != "cases"}), flush=True)
    if body["passed"] != 720 or replays != len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(
        a.root / "manifest.json",
        a.root / "representations.jsonl",
        a.root / "trained/head.json",
        a.root / "development-scores.jsonl",
        a.output,
    )
