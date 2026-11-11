from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import video_vcd2_case_archive_v202 as sealing


def setup_case(tmp_path):
    root = tmp_path.resolve()
    work, stage = root / "case-01", root / "authored"
    work.mkdir()
    stage.mkdir()
    (work / sealing.MARKER).write_text(json.dumps({"schema": sealing.WORK_SCHEMA, "case_id": work.name,
                                                  "work_path": str(work)}))
    (work / "data").mkdir()
    (work / "data" / "objects").mkdir()
    (work / "data" / "objects" / "evidence.bin").write_bytes(b"source evidence")
    (work / "case.json").write_text("{}")
    (work / "runtime.json").write_text("{}")
    (work / "models.yaml").write_text("models: []\n")
    (stage / "worker.log").write_text("guard log")
    (stage / "resources.jsonl").write_text("{}\n")
    (stage / "profile.json").write_text("{}")
    return work, stage, root / "case-01.tar"


def test_complete_case_is_verified_and_only_owned_work_is_removed(tmp_path):
    work, stage, archive = setup_case(tmp_path)
    (work / "models").mkdir()
    external = tmp_path / "external-model"
    external.write_bytes(b"large model")
    (work / "models" / "model").symlink_to(external)
    (work / "cache").mkdir()
    (work / "cache" / "temporary").write_bytes(b"temporary")
    receipt = sealing.seal_case(work, stage, archive)
    sealing.verify_archive(archive, receipt)
    paths = {row["path"] for row in receipt["entries"]}
    assert {"work/data/objects/evidence.bin", "work/case.json", "work/runtime.json", "work/models.yaml",
            "stage/profile.json", "stage/worker.log", "stage/resources.jsonl"} <= paths
    assert not any(path == "work/models" or path == "work/cache" or path.startswith(("work/models/", "work/cache/"))
                   for path in paths)
    assert {row["path"] for row in receipt["excluded"]} >= {"work/models/model", "work/cache/temporary"}
    with pytest.raises(ValueError, match="Fresh archive path"):
        sealing.seal_case(work, stage, archive)
    sealing.release_local_work(work, archive, receipt)
    assert not work.exists() and stage.exists() and external.read_bytes() == b"large model" and archive.exists()
    sealing.verify_archive(archive, receipt)


def test_incomplete_case_is_archived_without_required_result(tmp_path):
    work, stage, archive = setup_case(tmp_path)
    (work / "case.json").unlink()
    (work / "runtime.json").unlink()
    (stage / "profile.json").unlink()
    receipt = sealing.seal_case(work, stage, archive)
    assert "work/case.json" not in {row["path"] for row in receipt["entries"]}
    sealing.release_local_work(work, archive, receipt)
    assert not work.exists() and stage.exists()


def test_fuser_links_are_excluded_only_with_exact_approved_bytes(tmp_path):
    work, stage, archive = setup_case(tmp_path)
    target = tmp_path / "pinned-fuser.json"
    target.write_bytes(b"pinned calibration")
    folder = work / "data" / "fusion"
    folder.mkdir()
    (folder / "model.json").symlink_to(target)
    with pytest.raises(ValueError, match="Unapproved data link"):
        sealing.seal_case(work, stage, archive)
    approved = {"data/fusion/model.json": {"target": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}}
    receipt = sealing.seal_case(work, stage, archive, allowed_data_links=approved)
    assert any(row["kind"] == "fuser_link" for row in receipt["excluded"])
    assert "work/data/fusion/model.json" not in {row["path"] for row in receipt["entries"]}
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="fuser bytes differ"):
        sealing.release_local_work(work, archive, receipt)
    assert work.exists()


@pytest.mark.parametrize("change", ["source", "extra", "stage", "archive", "receipt", "marker", "link"])
def test_release_refuses_any_changed_or_unowned_material(tmp_path, change):
    work, stage, archive = setup_case(tmp_path)
    receipt = sealing.seal_case(work, stage, archive)
    if change == "source":
        (work / "data" / "objects" / "evidence.bin").write_bytes(b"changed")
    elif change == "extra":
        (work / "extra").write_bytes(b"unsealed")
    elif change == "stage":
        (stage / "worker.log").write_text("changed")
    elif change == "archive":
        with archive.open("ab") as stream:
            stream.write(b"tampered")
    elif change == "receipt":
        Path(receipt["receipt_path"]).write_text("{}")
    elif change == "marker":
        (work / sealing.MARKER).write_text('{}')
    else:
        (work / "data" / "escape").symlink_to(tmp_path)
    with pytest.raises((ValueError, OSError)):
        sealing.release_local_work(work, archive, receipt)
    assert work.exists() and stage.exists()


def test_partial_archive_is_retained_after_write_failure(tmp_path, monkeypatch):
    work, stage, archive = setup_case(tmp_path)
    original = sealing._tar_entry
    count = 0

    def fail_second(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("disk full")
        original(*args)

    monkeypatch.setattr(sealing, "_tar_entry", fail_second)
    with pytest.raises(OSError, match="disk full"):
        sealing.seal_case(work, stage, archive)
    assert archive.exists() and work.exists() and stage.exists()
    with pytest.raises(ValueError, match="Fresh archive path"):
        sealing.seal_case(work, stage, archive)


def test_wrong_marker_and_unexpected_stage_link_fail_closed(tmp_path):
    work, stage, archive = setup_case(tmp_path)
    (work / sealing.MARKER).write_text(json.dumps({"schema": sealing.WORK_SCHEMA, "case_id": "other", "work_path": str(work)}))
    with pytest.raises(ValueError, match="Ownership marker"):
        sealing.seal_case(work, stage, archive)
    (work / sealing.MARKER).write_text(json.dumps({"schema": sealing.WORK_SCHEMA, "case_id": work.name, "work_path": str(work)}))
    (stage / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="Unexpected symlink"):
        sealing.seal_case(work, stage, archive)
