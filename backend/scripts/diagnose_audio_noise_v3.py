"""Consumed-v2 development experiment for the fixed v3 sensitivity probe.

This is NOT a fresh validation. Every old fit/cal/test source has been consumed.
Only new audio-v3/development artifacts are written. No thresholds are searched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_calibration_v3_methods import METHOD, band_view, decide, sensitivity

SOURCE = ROOT / "data/independent-evaluation/audio-v2"
OUT = ROOT / "data/independent-evaluation/audio-v3/development"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    body = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    if path.exists() and path.read_bytes() != body:
        raise ValueError(f"Refusing to overwrite a development artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def score():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_voice
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    weights = ensure_verified(get_registry().manifest.model("pella.voice_v2"), force=True)
    lock = {"method": METHOD, "research_use": "consumed-v2 development only; no confirmatory claim",
            "source_manifest_sha256": sha(SOURCE / "manifest.json"),
            "source_scores_sha256": sha(SOURCE / "scores.jsonl"),
            "source_calibration_sha256": sha(SOURCE / "calibration.json"),
            "methods_code_sha256": sha(Path(__file__).with_name("audio_calibration_v3_methods.py")),
            "weights": [{"path": str(p), "sha256": sha(p)} for p in weights],
            "threads": 4, "device": "cpu", "selection": "all 520 consumed-v2 inputs; no tuning sweep"}
    save(OUT / "band-diagnostic-lock.json", lock)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    prior = {r["variant_id"]: r for r in map(json.loads, (SOURCE / "scores.jsonl").read_text().splitlines()) if r["detector"] == "new"}
    output = OUT / "band-diagnostic-scores.jsonl"
    complete = {r["variant_id"] for r in map(json.loads, output.read_text().splitlines())} if output.exists() else set()
    todo = [r for r in manifest["variants"] if r["id"] not in complete]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = load_voice("pella.voice_v2", "cpu", "float32").obj
    for i, row in enumerate(todo):
        old = prior[row["id"]]
        result = {"variant_id": row["id"], "base_id": row["base_id"], "speaker": row["speaker"],
                  "label": row["label"], "old_split": row["split"], "condition": row["condition"],
                  "seconds": row["seconds"], "raw": old["score"], "raw_status": old["status"],
                  "raw_windows": old["windows"], "source_pcm_sha256": row["pcm_sha256"],
                  "band": None, "band_windows": [], "status": "not_applicable"}
        started = time.perf_counter()
        if old["score"] is not None:
            try:
                if sha(SOURCE / row["path"]) != row["sha256"]:
                    raise ValueError("Consumed source file changed")
                decoded = decode_audio(SOURCE / row["path"], source_sha256=row["sha256"])
                if decoded.mapping["pcm_sha256"] != row["pcm_sha256"]:
                    raise ValueError("Production PCM changed")
                band = band_view(decoded.pcm)
                predictions = model.predict([band[w["start_sample"]:w["end_sample"]] for w in old["windows"]])
                if len(predictions) != len(old["windows"]) or not predictions or not np.isfinite(predictions).all():
                    raise ValueError("Invalid probe predictions")
                result.update(status="ok", band=round(float(np.mean(predictions)), 6),
                              band_unrounded=float(np.mean(predictions)), band_windows=predictions,
                              band_pcm_sha256=hashlib.sha256(band.astype("<f4").tobytes()).hexdigest())
            except Exception as error:
                result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        result["sensitivity"] = sensitivity(result["raw"], result["band"])
        result["latency_s"] = time.perf_counter() - started
        with output.open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
        if i % 20 == 0 or i + 1 == len(todo):
            print(f"Consumed-v2 band probe: {i+1}/{len(todo)} inputs", flush=True)


def analyse():
    from scipy.special import expit

    rows = [json.loads(s) for s in (OUT / "band-diagnostic-scores.jsonl").read_text().splitlines()]
    if len(rows) != 520 or len({r["variant_id"] for r in rows}) != 520:
        raise ValueError("The complete, unique 520-input development cohort is required")
    calibration = json.loads((SOURCE / "calibration.json").read_text())
    original_scores = {r["variant_id"]: r for r in map(json.loads, (SOURCE / "scores.jsonl").read_text().splitlines()) if r["detector"] == "new"}
    pars = calibration["parameters"]["new"]
    for row in rows:
        # Probability uses the original unrounded mean, as the v2 fit does.
        # The stability perturbation separately uses round6 aggregates.
        raw = original_scores[row["variant_id"]].get("raw_unrounded", row["raw"])
        row["raw_unrounded"] = raw
        raw_clip = float(np.clip(raw, 1e-6, 1-1e-6)) if raw is not None else None
        p = float(expit(pars["slopes"][0] * np.log(raw_clip/(1-raw_clip)) + pars["intercept"])) if raw_clip is not None else None
        row.update(decide(p, row["raw"], row["band"], calibration["guards"]["new"]))

    def metrics(values, key):
        return {"rows": len(values), "original_sources": len({r["base_id"] for r in values}),
                "decisions": dict(Counter(r[key] for r in values)),
                "coverage": sum(r[key] != "inconclusive" for r in values)/len(values),
                "false_ai": sum(r["label"] == 0 and r[key] == "ai" for r in values),
                "false_original": sum(r["label"] == 1 and r[key] == "original" for r in values),
                "ai_coverage": sum(r["label"] == 1 and r[key] != "inconclusive" for r in values)/sum(r["label"] == 1 for r in values),
                "real_coverage": sum(r["label"] == 0 and r[key] != "inconclusive" for r in values)/sum(r["label"] == 0 for r in values)}

    groups = {"all_consumed": rows, "previous_test_now_consumed": [r for r in rows if r["old_split"] == "test"]}
    groups.update({"all_consumed:"+c: [r for r in rows if r["condition"] == c] for c in sorted({r["condition"] for r in rows})})
    groups.update({"previous_test_now_consumed:"+c: [r for r in rows if r["condition"] == c and r["old_split"] == "test"] for c in sorted({r["condition"] for r in rows})})
    report = {"method": METHOD, "claim": "Consumed-v2 development diagnostic only; every split is consumed. No independent validation.",
              "probability": "Original-view v2 fit mapping unchanged by veto; Brier/logloss are therefore unchanged.",
              "groups": {name: {"baseline": metrics(values, "identity_guard_decision"), "sensitivity_guard": metrics(values, "decision")} for name, values in groups.items()},
              "execution_status": dict(Counter(r["status"] for r in rows)),
              "extra_latency_s": {"total": sum(r["latency_s"] for r in rows), "median": float(np.median([r["latency_s"] for r in rows if r["status"] == "ok"]))},
              "deployment_eligible": False}
    save(OUT / "band-diagnostic-predictions.json", rows)
    save(OUT / "band-diagnostic-evaluation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["score", "analyse"])
    args = parser.parse_args()
    globals()[args.stage]()
