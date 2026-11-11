import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import grid_serving_baseline_v154 as corrected


def test_capture_records_actual_source_inode_for_object_store_link(tmp_path):
    objects = tmp_path / "objects"
    source = objects / "tmp/authored.part"
    source.parent.mkdir(parents=True)
    source.write_bytes(corrected.SAMPLE)
    final = objects / "sha256/authored"
    final.parent.mkdir()
    captured = []
    corrected.capture_link("os.link", (source, final, -1, -1), objects, captured)
    os.link(source, final)
    source.unlink()
    stat = final.stat()
    assert captured == [(stat.st_dev, stat.st_ino, final)]
    assert stat.st_nlink == 1 and final.read_bytes() == corrected.SAMPLE


def test_capture_refuses_an_external_source_for_owned_cas_link(tmp_path):
    with pytest.raises(ValueError, match="Unexpected object-store link source"):
        corrected.capture_link(
            "os.link", (tmp_path / "outside", tmp_path / "objects/final", -1, -1), tmp_path / "objects", []
        )


def test_work_volume_refuses_non_apfs_filesystem(monkeypatch, tmp_path):
    monkeypatch.setattr(corrected, "WORK", tmp_path / "work")
    monkeypatch.setattr(corrected, "TRIAL", tmp_path / "trial")
    monkeypatch.setattr(corrected, "SYSTEM_DATA", tmp_path / "alternate")
    (tmp_path / "alternate").mkdir()
    commands = []

    def mount(*args, **kwargs):
        commands.append((args, kwargs))
        return SimpleNamespace(stdout=f"external on {corrected.SYSTEM_DATA} (exfat, local, removable)")

    monkeypatch.setattr(corrected.subprocess, "run", mount)
    with pytest.raises(ValueError, match="local APFS"):
        corrected.work_volume()
    assert commands[0][0][0] == ["/sbin/mount"]


def test_no_admission_when_disk_reserve_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(corrected, "WORK", tmp_path / "work")
    monkeypatch.setattr(corrected, "work_volume", lambda: {"device": 123})
    monkeypatch.setattr(corrected.shutil, "disk_usage", lambda _path: SimpleNamespace(free=3 * 1024**3 - 1))
    with pytest.raises(ValueError, match="Unchanged initial disk reserve"):
        corrected.storage_smoke("cpu")
    assert not corrected.WORK.exists()


def test_real_object_store_hardlink_smoke_in_authored_workspace(tmp_path):
    if sys.platform != "darwin":
        pytest.skip("System APFS is required for the production storage smoke")
    script = """
import json
import sys
from pathlib import Path
from scripts import grid_serving_baseline_v154 as v
root = Path(sys.argv[1])
v.WORK, v.TRIAL = root / "work", root / "trial"
proof = v.storage_smoke("cpu")
assert "torch" not in sys.modules and not list(v.WORK.iterdir())
print(json.dumps(proof))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        env={"PYTHONPATH": str(corrected.ROOT / "backend"), "PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    proof = json.loads(result.stdout)
    assert proof["filesystem"] == "apfs" and proof["system_data_volume"] is True
    assert proof["production_object_store_round_trip"] and proof["same_inode_hard_link"]
    assert proof["bytes"] == len(corrected.SAMPLE)
    assert proof["sha256"] == hashlib.sha256(corrected.SAMPLE).hexdigest()
    assert not list((tmp_path / "work").iterdir())


def test_actual_historical_stage_remains_failed_and_binds_its_logs():
    prior = corrected.bootstrap.stage_folder(0)
    if not (prior / "output/result.json").is_file():
        pytest.skip("Actual v153 failed-stage receipts are local to the evaluation workspace")
    _, history = corrected.metadata("history")
    assert history["failed_ingest_cases"] == 11 and history["detector_runs"] == 0
    assert history["null_case_labels_after_field_collision"] == 11
    assert history["v153_result_sha256"] == corrected.sha(prior / "output/result.json")
    assert history["v153_profile_sha256"] == corrected.sha(prior / "profile.json")
    assert history["v153_worker_log_sha256"] == corrected.sha(prior / "worker.log")


def test_app_condition_profile_cannot_overwrite_private_case_label(monkeypatch):
    monkeypatch.setattr(
        corrected.baseline,
        "analyze_case",
        lambda: {"condition": {"slice": "authored"}, "analysis_status": "done"},
    )
    row = {"condition": "fixed_variant", **corrected.analyze_case()}
    assert row["condition"] == "fixed_variant" and row["condition_profile"] == {"slice": "authored"}
