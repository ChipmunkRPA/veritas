"""Separate cached-aggregation and actual CPU-runtime parity checks.

Only consumed-v2 data are read. This does not claim 520 fresh model reruns.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_calibration_v3_methods import score_views


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_speech_activity, load_voice
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    source = ROOT / "data/independent-evaluation/audio-v2"
    out = ROOT / "data/independent-evaluation/audio-v3/development"
    scores = {r["variant_id"]: r for r in map(json.loads, (source / "scores.jsonl").read_text().splitlines()) if r["detector"] == "new"}
    if len(scores) != 520:
        raise ValueError("The complete consumed-v2 cohort is required")
    maximum = 0.
    for row in scores.values():
        windows = row["windows"]
        if windows:
            mean = float(np.mean([w["raw_unrounded"] for w in windows]))
            maximum = max(maximum, abs(mean - row["raw_unrounded"]))
            assert abs(mean - row["raw_unrounded"]) <= 1e-12
            assert round(mean, 6) == row["score"]
        else:
            assert row["score"] is None
        proposed = row["proposed_windows"]
        for w in proposed:
            expected = window_in_speech(row["applied_speech_spans_s"], w["start_sample"]/16000, w["end_sample"]/16000)
            assert expected == w["accepted_by_speech_gate"]
        assert [(w["start_sample"], w["end_sample"]) for w in windows] == [
            (w["start_sample"], w["end_sample"]) for w in proposed if w["accepted_by_speech_gate"]]

    manifest = json.loads((source / "manifest.json").read_text())
    ids = {json.loads(line)["id"] for line in (out / "band-probe.jsonl").read_text().splitlines()}
    selected = [r for r in manifest["variants"] if r["id"] in ids]
    assert len(selected) == 8
    frozen = json.loads((out / "band-diagnostic-lock.json").read_text())
    weight_paths = ensure_verified(get_registry().manifest.model("pella.voice_v2"), force=True)
    weights = []
    for path in weight_paths:
        with path.open("rb") as stream:
            weights.append({"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()})
    assert weights == frozen["weights"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = load_voice("pella.voice_v2", "cpu", "float32").obj
    activity = load_speech_activity("silero.speech_activity", "cpu", "float32").obj
    checks = []
    for row in selected:
        decoded = decode_audio(source / row["path"], source_sha256=row["sha256"])
        assert decoded.mapping["pcm_sha256"] == row["pcm_sha256"]
        result = score_views(decoded.pcm, model, activity)
        original = scores[row["id"]]
        assert result["source_pcm_sha256"] == row["pcm_sha256"]
        assert result["speech_spans_s"] == original["applied_speech_spans_s"]
        assert [(w["start_sample"], w["end_sample"]) for w in result["windows"]] == [
            (w["start_sample"], w["end_sample"]) for w in original["windows"]]
        errors = [abs(a["raw"]-b["raw_unrounded"]) for a, b in zip(result["windows"], original["windows"], strict=True)]
        aggregate_delta = abs(result["raw_unrounded"] - original["raw_unrounded"])
        assert max(errors + [aggregate_delta]) <= 1e-6
        checks.append({"variant_id": row["id"], "original_source": row["base_id"],
                       "condition": row["condition"], "label": row["label"], "seconds": row["seconds"],
                       "windows": len(result["windows"]), "maximum_window_raw_delta": max(errors),
                       "aggregate_unrounded_delta": aggregate_delta,
                       "rounded_aggregate_exact": result["raw"] == original["score"],
                       "source_pcm_hash_exact": True, "speech_gate_and_windows_exact": True})
        print(f"Actual CPU-runtime parity: {len(checks)}/8", flush=True)
    report = {"status": "passed", "research_scope": "consumed-v2 development only",
              "cached_aggregation_audit": {"inputs": 520, "mean_rounding_and_gate_selection_match": True,
                  "maximum_aggregate_delta": maximum, "fresh_classifier_inferences": 0},
              "actual_cpu_runtime_parity": {"inputs": 8, "original_sources": 4,
                  "cases": "Two known erroneous AI sources plus deterministic 4s/8s real controls, clean and noise5 counterparts.",
                  "tolerance": 1e-6, "results": checks},
              "methods_code_sha256": hashlib.sha256(Path(__file__).with_name("audio_calibration_v3_methods.py").read_bytes()).hexdigest(),
              "source_manifest_sha256": frozen["source_manifest_sha256"],
              "source_scores_sha256": frozen["source_scores_sha256"],
              "weights_match_diagnostic_lock": True,
              "deployment_device_scope": "cpu only; no MPS/GPU runtime claim"}
    body = json.dumps(report, indent=2, allow_nan=False).encode() + b"\n"
    dest = out / "method-parity.json"
    if dest.exists() and dest.read_bytes() != body:
        raise ValueError("Refusing to overwrite a parity artifact")
    dest.write_bytes(body)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
