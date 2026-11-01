"""Read-only checks of the six preselected audio API replays and their exports.

This is implementation diagnostics on already-seen samples, not a new accuracy
test. The original FLOAT-input experiment remains unchanged, including its
failed admission. No candidate calibrator is applied by this script.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit() -> dict:
    from app.detectors.audio.decode import read_wav
    from app.engine.fusion.artefact import Fuser
    from app.engine.fusion.features import Context

    folder = ROOT / "data/independent-evaluation/audio"
    manifest = json.loads((folder / "api-parity-manifest.json").read_text())["items"]
    variants = {v["id"]: v for v in json.loads((folder / "manifest.json").read_text())["variants"]}
    expected = {r["id"]: r for r in json.loads((folder / "api-parity-expected.json").read_text())}
    state = json.loads((folder / "api-parity-run/run-state.json").read_text())
    store = ROOT / "data/objects/sha256"
    db = sqlite3.connect(f"file:{ROOT / 'data/veritas.db'}?mode=ro", uri=True)
    fuser = Fuser.load(ROOT / "app/engine/fusion/release/fuser-audio-2.1.1.json")
    assert fuser is not None
    rows, all_assets = [], set()

    def object_bytes(sha: str) -> bytes:
        assert re.fullmatch(r"[0-9a-f]{64}", sha)
        data = (store / sha[:2] / sha[2:4] / sha).read_bytes()
        assert digest(data) == sha
        return data

    def stage(aid: str, name: str) -> dict:
        sha = db.execute("SELECT a.sha256 FROM analysis_stages s JOIN artefacts a "
                         "ON a.id=s.output_artefact_id WHERE s.analysis_id=? AND s.stage=?",
                         (aid, name)).fetchone()[0]
        return json.loads(object_bytes(sha))

    def assets(value):
        if isinstance(value, dict):
            if "url" in value and "sha256" in value:
                yield value
            for child in value.values():
                yield from assets(child)
        elif isinstance(value, list):
            for child in value:
                yield from assets(child)

    with httpx.Client(base_url="http://127.0.0.1:8160", timeout=60) as client:
        for item in manifest:
            aid = state["submitted"][item["id"]]
            path = folder / "api-parity-run/items" / item["id"]
            detail = json.loads((path / "analysis.json").read_text())
            exported = json.loads((path / "export.json").read_text())
            assert detail["status"] == "done"
            assert detail["customer_assessment"] == exported["customer_assessment"]
            public_text = json.dumps([detail, exported]).lower()
            assert not any(name in public_text for name in ("gustking", "pella", "wav2vec", "silero", "panns"))
            evidence = exported["evidence"]
            source = evidence["source"]
            variant = variants[item["variant_id"]]
            assert source["sha256"] == item["sha256"] == digest(Path(item["path"]).read_bytes())
            pcm = read_wav(object_bytes(source["wav_sha256"]))
            assert digest(pcm.astype("<f4").tobytes()) == source["pcm_sha256"]
            assert source["samples"] == len(pcm) == 64000 and source["sample_rate"] == 16000
            detect = stage(aid, "detect")
            checks = {r["detector_id"]: r for r in detect["detectors"]}
            values = {}
            for key, detector in (("old", "aud.gustking_xlsr"), ("new", "aud.pella_v2")):
                record = checks[detector]
                assert record["status"] == "ok"
                assert record["details"]["audio_input"]["pcm_sha256"] == source["pcm_sha256"]
                windows = record["unit_scores"]
                assert len(windows) == 1
                assert windows[0]["quality"]["start_sample"] == 0
                assert windows[0]["quality"]["end_sample"] == 64000
                assert record["aggregate"]["raw"] == windows[0]["raw"]
                old_value = expected[variant["id"]]["scores"][key]["score"]
                values[key] = {"api_score": record["aggregate"]["raw"], "offline_float_score": old_value,
                               "difference": record["aggregate"]["raw"] - old_value,
                               "api_device": record["device"], "offline_device": "cpu"}
            fused = stage(aid, "fuse")
            recomputed = fuser.apply({"aud.gustking_xlsr": (values["old"]["api_score"], None)}, Context(modality="audio"))
            actual = fused["decision"]["probability_ai_calibrated"]
            assert abs(actual - recomputed.probability_display) <= 0.00000051
            new_signal = next(s for s in fused["signals"] if s["id"] == "aud.pella_v2:global")
            assert new_signal.get("contribution_logit") in (None, 0)
            packets = evidence["packets"] + evidence["sound_events"].get("windows", [])
            for packet in packets:
                interval = packet.get("interval", packet)
                left, right = interval["start_sample"], interval["end_sample"]
                assert 0 <= left < right <= len(pcm)
                assert packet["start_s"] == left / 16000 and packet["end_s"] == right / 16000
                selected = packet["artifacts"]["selected_clip"]
                assert np.array_equal(read_wav(object_bytes(selected["sha256"])), pcm[left:right])
            for asset in assets(evidence):
                if asset["url"] in all_assets:
                    continue
                assert asset["url"].startswith(f"/api/v2/analyses/{aid}/evidence-packets/")
                response = client.get(asset["url"])
                response.raise_for_status()
                assert digest(response.content) == asset["sha256"]
                all_assets.add(asset["url"])
            rows.append({"sample_id": item["id"], "analysis_id": aid, "status": "done", "checks": values,
                         "source_pcm_sha256": source["pcm_sha256"], "offline_pcm_sha256": variant["pcm_sha256"],
                         "pcm_identical_to_original_experiment": source["pcm_sha256"] == variant["pcm_sha256"],
                         "fuser_replay_matches_api": True, "new_check_fusion_weight": 0,
                         "selected_clip_and_time_bounds_verified": len(packets),
                         "public_model_names_absent": True, "customer_assessment": detail["customer_assessment"]})
    db.close()
    result = {"purpose": "Post-test implementation diagnostics on six preselected repeated samples, not independent accuracy",
              "original_experiment_production_pcm_parity": False,
              "reason": "Original experiment read FLOAT WAV directly; service explicitly quantizes to signed 16-bit PCM. Old check additionally uses MPS in the service, CPU offline.",
              "production_calibration_admitted": False, "downloaded_evidence_assets_verified": len(all_assets), "items": rows}
    (folder / "api-parity-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"items_verified": len(rows), "evidence_assets_verified": len(all_assets),
            "original_experiment_pcm_parity": False, "fuser_replay_matches_api": True}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
