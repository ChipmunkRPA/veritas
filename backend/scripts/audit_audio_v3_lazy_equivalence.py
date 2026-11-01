"""Actual CPU equivalence against consumed-v2 full-probe references only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audio_calibration_v3_methods as method
from audio_calibration_v3_scoring import EXECUTION_VERSION, score_candidate


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_speech_activity, load_voice
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    source = ROOT / "data/independent-evaluation/audio-v2"
    out = ROOT / "data/independent-evaluation/audio-v3/development"
    lock = json.loads((out / "band-diagnostic-lock.json").read_text())
    original = {r["variant_id"]: r for r in map(json.loads, (source / "scores.jsonl").read_text().splitlines()) if r["detector"] == "new"}
    probes = {r["variant_id"]: r for r in map(json.loads, (out / "band-diagnostic-scores.jsonl").read_text().splitlines())}
    manifest = json.loads((source / "manifest.json").read_text())
    ids = {json.loads(line)["id"] for line in (out / "band-probe.jsonl").read_text().splitlines()}
    selected = [r for r in manifest["variants"] if r["id"] in ids]
    assert len(selected) == 8
    assert sha(source / "scores.jsonl") == lock["source_scores_sha256"]
    assert sha(ROOT / "scripts/audio_calibration_v3_methods.py") == lock["methods_code_sha256"]
    weights = ensure_verified(get_registry().manifest.model("pella.voice_v2"), force=True)
    assert [{"path": str(path), "sha256": sha(path)} for path in weights] == lock["weights"]
    cal = json.loads((source / "calibration.json").read_text())
    old_pars = cal["parameters"]["new"]
    calibration = {"parameters": {"new": {"slope": old_pars["slopes"][0], "intercept": old_pars["intercept"]}},
                   "guards": {"new": cal["guards"]["new"]}}
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = load_voice("pella.voice_v2", "cpu", "float32").obj
    activity = load_speech_activity("silero.speech_activity", "cpu", "float32").obj

    class CountCalls:
        calls = 0

        def predict(self, chunks):
            self.calls += 1
            return model.predict(chunks)

    records = []
    for row in selected:
        assert sha(source / row["path"]) == row["sha256"]
        decoded = decode_audio(source / row["path"], source_sha256=row["sha256"])
        assert decoded.mapping["pcm_sha256"] == row["pcm_sha256"]
        counted = CountCalls()
        result = score_candidate(decoded.pcm, counted, activity, calibration)
        reference, full_probe = original[row["id"]], probes[row["id"]]
        expected_p = method.probability(reference["raw_unrounded"], calibration["parameters"]["new"])
        expected = method.decide(expected_p, reference["score"], full_probe["band"], calibration["guards"]["new"])
        assert result["raw_unrounded"] == reference["raw_unrounded"]
        assert result["raw"] == reference["score"]
        assert result["speech_spans_s"] == reference["applied_speech_spans_s"]
        assert result["source_pcm_sha256"] == row["pcm_sha256"]
        assert [(w["start_sample"], w["end_sample"], w["raw"]) for w in result["windows"]] == [
            (w["start_sample"], w["end_sample"], w["raw_unrounded"]) for w in reference["windows"]]
        assert result["decision_detail"]["probability_ai"] == expected_p
        assert result["decision_detail"]["decision"] == expected["decision"]
        assert result["decision_detail"]["identity_guard_decision"] == expected["identity_guard_decision"]
        computed = result["probe_status"] == "computed"
        if computed:
            assert result["band"] == full_probe["band"]
            assert result["band_unrounded"] == full_probe["band_unrounded"]
            assert result["band_pcm_sha256"] == full_probe["band_pcm_sha256"]
            assert [w["band"] for w in result["windows"]] == full_probe["band_windows"]
        else:
            assert result["probe_status"] == "not_required_for_decision"
            assert result["probe_error"] is None and result["band"] is None
        assert counted.calls == (2 if computed else 1)
        records.append({"variant_id": row["id"], "source": row["base_id"], "condition": row["condition"],
                        "label": row["label"], "seconds": row["seconds"],
                        "probability_ai": expected_p, "identity_guard_decision": expected["identity_guard_decision"],
                        "full_probe_decision": expected["decision"], "lazy_decision": result["decision_detail"]["decision"],
                        "probe_status": result["probe_status"], "classifier_predict_calls": counted.calls,
                        "raw_window_and_aggregate_exact": True, "gate_and_source_hash_exact": True,
                        "band_values_exact_when_computed": True if computed else None})
        print(f"Actual lazy/full equivalence: {len(records)}/8", flush=True)
    assert any(r["probe_status"] == "computed" for r in records)
    assert any(r["identity_guard_decision"] == "ai" and r["probe_status"] == "not_required_for_decision" for r in records)
    report = {"schema": "chipmunk.audio-v3-lazy-equivalence/1", "status": "passed",
              "execution": EXECUTION_VERSION, "scope": "Eight actual CPU inputs from four consumed-v2 originals; no fresh-data use",
              "method_unchanged": True, "method_sha256": sha(ROOT / "scripts/audio_calibration_v3_methods.py"),
              "scoring_code_sha256": sha(ROOT / "scripts/audio_calibration_v3_scoring.py"),
              "source_v2_scores_sha256": sha(source / "scores.jsonl"),
              "full_probe_scores_sha256": sha(out / "band-diagnostic-scores.jsonl"),
              "production_admitted": False, "all_probabilities_and_final_decisions_exact": True,
              "full_probe_predict_calls_for_same_inputs": 16,
              "lazy_predict_calls": sum(r["classifier_predict_calls"] for r in records),
              "actual_band_probes": sum(r["probe_status"] == "computed" for r in records),
              "intentional_skips": sum(r["probe_status"] == "not_required_for_decision" for r in records),
              "records": records}
    body = json.dumps(report, indent=2, allow_nan=False).encode() + b"\n"
    dest = out / "optimization-equivalence.json"
    if dest.exists() and dest.read_bytes() != body:
        raise ValueError("Refusing to overwrite equivalence evidence")
    dest.write_bytes(body)
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
