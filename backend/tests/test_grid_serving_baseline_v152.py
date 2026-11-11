import copy
import json
import subprocess
import sys

import pytest

from scripts import grid_serving_baseline_v152 as baseline


def test_controlled_environment_keeps_home_and_rejects_inherited_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("VERITAS_DATABASE_URL", "authored untrusted value")
    monkeypatch.setenv("VERITAS_STRIPE_SECRET_KEY", "authored untrusted value")
    monkeypatch.setenv("HF_TOKEN", "authored untrusted value")
    monkeypatch.setenv("VERITAS_DETECTOR_SET", "fake")
    monkeypatch.setenv("HOME", str(tmp_path / "original-home"))
    env = baseline.controlled_environment(
        "auto", tmp_path / "models", tmp_path / "hf", tmp_path / "data", tmp_path
    )
    assert env["HOME"] == str(tmp_path / "original-home")
    assert env["VERITAS_DETECTOR_SET"] == "manifest" and env["VERITAS_DEFAULT_PROFILE"] == "full"
    assert env["VERITAS_OFFLINE"] == "true" and env["VERITAS_BILLING_MODE"] == "disabled"
    assert env["VERITAS_RUN_WORKER"] == env["VERITAS_WEBHOOKS_ENABLED"] == "false"
    assert env["VERITAS_DEVICE"] == "auto"
    assert env["VERITAS_VIDEO_FACE_REVIEW_MODE"] == "appearance"
    assert env["VERITAS_AUDIO_DETECTOR_MODE"] == "candidate"
    assert env["HF_HUB_CACHE"] == str(tmp_path / "hf/hub")
    assert not {"VERITAS_DATABASE_URL", "VERITAS_STRIPE_SECRET_KEY", "HF_TOKEN"}.intersection(env)


def test_actual_settings_registry_and_projection_in_fresh_authored_process(tmp_path):
    script = """
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic_settings import DotEnvSettingsSource
from scripts import grid_serving_baseline_v152 as v

folder = Path(sys.argv[1])
with patch.object(DotEnvSettingsSource, "_static_read_env_file", side_effect=AssertionError("dotenv was read")):
    settings = v.fresh_settings(v.controlled_environment("cpu", folder / "models", folder / "hf", folder / "data", folder))
from app.engine.registry import get_registry
active = get_registry()
registry = v.registry_snapshot(active)
assert "vid.swap.gend_clip_l14" not in registry["enabled"]["video"]
assert "vid.lipsync.stable_syncnet" not in registry["enabled"]["video"]
assert "vid.review.face_sequence" not in registry["enabled"]["video"]
assert "aud.gustking_xlsr" not in registry["enabled"]["audio"]
for model_id in v.required_model_ids(active, registry):
    model = active.manifest.model(model_id)
    assert model.licence.class_ == "commercial" and model.source.files
    assert all(weight.sha256 for weight in model.source.files)
outcome = v.customer_readout(SimpleNamespace(status="done"), {"decision": {"code": "not_assessable"}})
assert outcome["direction"] == "unavailable" and not outcome["content_assessed"]
assert settings.sqlalchemy_url == f"sqlite:///{folder / 'data/veritas.db'}"
assert "torch" not in sys.modules
from app.storage.db import get_engine, init_db, shutdown
from app.storage.files import validate_kind
receipt = folder / "receipt"
receipt.mkdir()
v.isolate_filesystem(folder, receipt, set())
init_db()
assert get_engine().url.database == str(folder / "data/veritas.db")
header = bytes.fromhex("1a45dfa3428288") + b"matroska"
assert validate_kind("video", header, v.FILENAME) == ("video/x-matroska", ".mkv")
shutdown()
print(json.dumps({"video": len(registry["enabled"]["video"]), "audio": len(registry["enabled"]["audio"])}))
"""
    env = baseline.controlled_environment(
        "cpu", tmp_path / "models", tmp_path / "hf", tmp_path / "data", tmp_path
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    counts = json.loads(result.stdout)
    assert counts["video"] >= 6 and counts["audio"] >= 2
    assert (tmp_path / "data/veritas.db").is_file()


def test_model_links_keep_underlying_files_read_only_and_retain_missing(tmp_path):
    source = tmp_path / "author-model"
    source.write_bytes(b"authored model fixture")
    folder = tmp_path / "isolated"
    folder.mkdir()
    inventory = {
        "model_files": [
            {
                "slot": "models",
                "relative": "weights/fixture.bin",
                "source_file": str(source),
                "status": "available",
                "sha256": baseline.sha(source),
                "bytes": source.stat().st_size,
            },
            {
                "slot": "hf",
                "relative": "hub/absent.bin",
                "source_file": str(tmp_path / "absent"),
                "status": "missing",
            },
        ],
        "local_fuser_files": [],
    }
    baseline.install_assets(inventory, folder)
    mirrored = folder / "models/weights/fixture.bin"
    assert mirrored.is_symlink() and mirrored.resolve() == source
    assert not (folder / "hf/hub/absent.bin").exists()
    (tmp_path / "absent").write_bytes(b"new content")
    with pytest.raises(ValueError, match="missing model inventory changed"):
        baseline.install_assets(
            {"model_files": inventory["model_files"][1:], "local_fuser_files": []}, folder
        )
    inventory["model_files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fuser asset bytes changed"):
        baseline.install_assets(
            {"model_files": inventory["model_files"][:1], "local_fuser_files": []}, folder
        )


def test_audit_guard_refuses_python_writes_through_asset_links_and_network(monkeypatch, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    external = tmp_path / "existing-model"
    external.write_bytes(b"authored fixture")
    link = stage / "input"
    link.symlink_to(external)
    receipt = tmp_path / "receipt"
    receipt.mkdir()
    registered = []
    monkeypatch.setattr(baseline.sys, "addaudithook", registered.append)
    baseline.isolate_filesystem(stage, receipt, set())
    event = registered[0]
    event("open", (str(link), "r", baseline.os.O_RDONLY))
    event("open", (str(stage / "result.json"), "w", baseline.os.O_WRONLY))
    event("open", (str(receipt / "result.json"), "w", baseline.os.O_WRONLY))
    with pytest.raises(PermissionError, match="writes"):
        event("open", (str(link), "w", baseline.os.O_WRONLY))
    with pytest.raises(PermissionError, match="networking"):
        event("socket.connect", (None, ("localhost", 1234)))
    with pytest.raises(PermissionError, match="decoder"):
        event("subprocess.Popen", (sys.executable, [sys.executable], None, None))


def detector(name, family, *, enabled=True, modality="video", reasons=()):
    return {
        "id": name,
        "family": family,
        "enabled_for_full": enabled,
        "modality": modality,
        "registry_reasons": reasons,
    }


def test_actual_scheduled_skipped_failed_and_unobserved_detectors_remain_distinct():
    snapshot = {
        "detectors": [
            detector("provenance", "provenance"),
            detector("track", "face-tracking"),
            detector("face", "face-detection"),
            detector("appearance", "face-sequence-review"),
            detector("old-voice", "voice", enabled=False, modality="audio", reasons=("rollback",)),
        ]
    }
    stages = {"provenance": "done", "tracks": "failed", "faces": "done", "detect": "done"}
    outputs = {
        "provenance": {"detectors": [{"detector_id": "provenance", "status": "not_applicable"}]},
        "faces": {"detectors": [{"detector_id": "face", "status": "ok"}]},
        "detect": {
            "detectors": [{"detector_id": "appearance", "status": "skipped", "reason": "memory budget"}]
        },
    }
    rows = baseline.detector_inventory(snapshot, stages, outputs)
    assert [(r["state"], r["expected_stage"]) for r in rows] == [
        ("observed", "provenance"),
        ("unobserved", "tracks"),
        ("observed", "faces"),
        ("observed", "detect"),
        ("not_scheduled", None),
    ]
    assert rows[3]["runs"][0]["status"] == "skipped" and "failed" in rows[1]["reason"]
    assert rows[4]["reason"] == "rollback"
    outputs["faces"]["detectors"][0]["detector_id"] = "unfrozen"
    with pytest.raises(ValueError, match="Unexpected detector"):
        baseline.detector_inventory(snapshot, stages, outputs)


def test_real_input_readout_matches_clock_and_samples_without_filling_missing_data():
    case = {
        "status": "ready",
        "waveform_id": "original_pcm",
        "audio_output_clock_mapping": {},
        "expected": {
            "picture_pts_ms": [0, 40, 80],
            "rgb_sha256": ["a", "b", "c"],
            "audio_samples": 1280,
            "audio_start_seconds": "1/1000",
        },
    }
    waveforms = {
        "original_pcm": {
            "status": "encoded",
            "waveform": {"samples": 1280, "pcm_sha256": "ab" * 32},
        }
    }
    case = baseline.bind_expected_pcm([case], waveforms)[0]
    outputs = {
        "probe": {
            "frame_table": {
                "time_base": "1/90000",
                "pts": [0, 3600, 7200],
                "truncated": False,
                "interpolated": [],
            }
        },
        "decode": {
            "time_base": "1/90000",
            "frames": [{"frame_index": 1, "pts": 3600, "sha256": "b"}],
            "missing_frames": [0, 2],
            "audio": {
                "samples": 1280,
                "source_mapping": {
                    "timeline_time_base": "1/16000",
                    "timeline": [{"source_pts": 16}],
                    "pcm_sha256": "ab" * 32,
                },
            },
        },
    }
    observed = baseline.serving_input(case, outputs)
    assert observed["clock_exact"] and observed["sampled_rgb_exact"]
    assert observed["audio_samples_equal"] and observed["audio_origin_equal"] and observed["audio_pcm_equal"]
    assert observed["missing_sampled_frames"] == [0, 2]
    assert baseline.serving_input(case, {})["sampled_rgb_exact"] is None
    outputs["decode"]["frames"][0]["sha256"] = "wrong"
    assert baseline.serving_input(case, outputs)["sampled_rgb_exact"] is False
    case["audio_output_clock_mapping"]["pcm_sha256"] = "wrong"
    with pytest.raises(ValueError, match="whole-waveform PCM commitment"):
        baseline.bind_expected_pcm([case], waveforms)


def unavailable_case_rows():
    cases = [
        {
            "asset_index": 0,
            "case_number": 10 - i,
            "source_slot": 0,
            "component_index": 0,
            "condition": name,
            "status": "unavailable",
            "reason": "Authored unavailable input",
        }
        for i, name in enumerate(baseline.planning.CASES)
    ]
    rows = [dict(case, status="unavailable_input") for case in baseline.planned_cases(cases)]
    return rows, cases


def test_unavailable_input_cannot_be_relabelled_as_original():
    rows, cases = unavailable_case_rows()
    snapshot = {"detectors": []}
    baseline.validate_case_rows(rows, cases, snapshot)
    assert baseline.count_results(rows)["input_unavailable_cases"] == 11
    changed = copy.deepcopy(rows)
    changed[0]["customer_assessment"] = {"direction": "original"}
    with pytest.raises(ValueError, match="without a verdict"):
        baseline.validate_case_rows(changed, cases, snapshot)
    with pytest.raises(ValueError, match="Every matched case"):
        baseline.validate_case_rows(rows[:-1], cases, snapshot)


def test_completed_original_requires_assessed_production_conclusion():
    rows, cases = unavailable_case_rows()
    ready = next(case for case in cases if case["case_number"] == rows[0]["case_number"])
    ready.update(status="ready", reason=None, file_sha256="authored")
    rows[0].update(
        status="done",
        pipeline_outcome="done",
        analysis_status="done",
        result_hash="authored",
        filename=baseline.FILENAME,
        input_file_sha256="authored",
        stages=[],
        detector_inventory=[],
        detector_stage_outputs={},
        customer_assessment={
            "direction": "original",
            "content_assessed": False,
            "directional_conclusion": True,
        },
    )
    with pytest.raises(ValueError, match="assessed conclusion"):
        baseline.validate_case_rows(rows, cases, {"detectors": []})
    rows[0]["customer_assessment"]["content_assessed"] = True
    baseline.validate_case_rows(rows, cases, {"detectors": []})


def test_routing_keeps_exclusive_v108_authored_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(baseline, "TRIAL", tmp_path)
    launched = []
    monkeypatch.setattr(baseline.subprocess, "Popen", lambda argv, **kw: launched.append((argv, kw)))
    output = tmp_path / "clip-02/authored/output"
    baseline.routed_popen(
        2, ["python", "-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)], env={}
    )
    assert launched[0][0][3:] == ["scripts.grid_serving_baseline_v152", "assess", "2", str(output)]
    with pytest.raises(ValueError, match="exclusive"):
        baseline.routed_popen(
            2, ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", str(output)]
        )


def test_pending_native_clip_context_creates_no_freeze(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(baseline.clips, "PUBLIC", tmp_path / "missing")
    monkeypatch.setattr(baseline, "TRIAL", tmp_path / "trial")
    monkeypatch.setattr(baseline, "PUBLIC", tmp_path / "public")
    baseline.freeze("auto", tmp_path / "models", tmp_path / "hf")
    assert json.loads(capsys.readouterr().out)["status"] == "pending_v149_context_no_freeze"
    assert not baseline.TRIAL.exists() and not baseline.PUBLIC.exists()
