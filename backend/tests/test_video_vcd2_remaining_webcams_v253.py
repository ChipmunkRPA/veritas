"""Verify the remaining-controls wrapper without running models."""

import json

import pytest

from scripts import video_vcd2_remaining_webcams_v253 as successor


def test_selection_binds_the_full_old_plan_and_uses_only_six_prepared_cases(tmp_path, monkeypatch):
    raw = successor.raw
    monkeypatch.setattr(raw, "TRIAL", tmp_path)
    path = tmp_path / "private-plan.json"
    full = {
        raw.case_name(slot, role): {"slot": slot, "role": role, "status": "prepared", "bytes": 10}
        for slot in raw.SLOTS
        for role in raw.ROLES
    }
    raw.write(path, full)
    lock = {"plan_sha256": successor.replay.archive_set_commitment(full)}
    assert tuple(successor.frozen_roster(lock)) == successor.CONDITIONS
    assert len(successor.frozen_roster(lock)) == 6
    full[successor.CONDITIONS[-1]]["status"] = "unsupported_upload_size"
    path.unlink()
    raw.write(path, full)
    with pytest.raises(ValueError, match="historical v202 roster"):
        successor.frozen_roster(lock)
    with pytest.raises(ValueError, match="six original prepared controls"):
        successor.frozen_roster({"plan_sha256": successor.replay.archive_set_commitment(full)})


def test_scoped_verification_checks_case_manifest_and_exact_upload(tmp_path, monkeypatch):
    condition = successor.CONDITIONS[0]
    trial = tmp_path / "trial"
    case = trial / condition
    case.mkdir(parents=True)
    prep = tmp_path / "prep"
    folder = prep / "slot-11-attempt-01"
    folder.mkdir(parents=True)
    qualification, file = folder / "qualification.json", folder / "unchanged.mp4"
    qualification.write_text("exact qualification")
    file.write_bytes(b"known test bytes")
    row = {
        "slot": 11,
        "role": "unchanged",
        "status": "prepared",
        "qualification_sha256": successor.sha(qualification),
        "bytes": file.stat().st_size,
        "sha256": successor.sha(file),
    }
    manifest = case / f"{successor.ARM}.yaml"
    manifest.write_text("patched runtime manifest")
    lock = {"app_manifest_sha256": successor.sha(manifest)}
    successor.raw.write(case / "inputs.json", row)
    successor.raw.write(case / "freeze.json", lock)
    monkeypatch.setattr(successor, "TRIAL", trial)
    monkeypatch.setattr(successor.raw, "PREPARATION", prep)
    monkeypatch.setattr(successor, "verify_plan", lambda: (lock, {condition: row}))
    assert successor.verify(condition) == (lock, file, {"file_sha256": row["sha256"], "bytes": row["bytes"]})
    with pytest.raises(ValueError, match="six fixed controls"):
        successor.verify("slot-10-unchanged")
    file.write_bytes(b"changed sample!")
    with pytest.raises(ValueError, match="pixels or metadata changed"):
        successor.verify(condition)
    manifest.write_text("different manifest")
    with pytest.raises(ValueError, match="inputs and app manifest"):
        successor.verify(condition)


def test_paired_flags_require_measured_appearance_for_both_controls():
    before, chroma, recompression = successor.raw.ROLES
    clear = {
        "status": "scored",
        "appearance_route_status": "ok",
        "appearance_check_status": "ok",
        "appearance_measured_windows": 12,
    }
    group = [
        {**clear, "role": before, "face_review_intervals": {}},
        {**clear, "role": chroma, "face_review_intervals": {"flag": [{}]}},
        {**clear, "role": recompression, "appearance_measured_windows": 0, "face_review_intervals": {}},
    ]
    assert successor.paired_flag_changes({11: group}) == {
        chroma: {"clear_to_flagged": 1},
        recompression: {"comparison_unavailable": 1},
    }
    assert group[-1]["status"] == "scored"


def test_partial_readout_verifies_sealed_upload_without_exposing_case_details(tmp_path, monkeypatch):
    for name in ("TRIAL", "WORK", "SEALED", "PUBLIC"):
        monkeypatch.setattr(successor, name, tmp_path / name.lower())
    successor.SEALED.mkdir()
    condition = successor.CONDITIONS[0]
    stage, work = successor.authored(condition), successor.work_dir(condition)
    stage.mkdir(parents=True)
    (work / "data").mkdir(parents=True)
    archives = successor.replay.archives
    successor.raw.write(
        work / archives.MARKER,
        {"schema": archives.WORK_SCHEMA, "case_id": work.name, "work_path": str(work)},
    )
    upload = work / "data" / "private-upload.mp4"
    upload.write_bytes(b"private fixture payload")
    plan = {
        successor.raw.case_name(slot, role): {"slot": slot, "role": role, "status": "prepared"}
        for slot in successor.SLOTS
        for role in successor.raw.ROLES
    }
    row = plan[condition]
    row.update(sha256=successor.sha(upload), bytes=upload.stat().st_size)
    manifest = stage / f"{successor.ARM}.yaml"
    manifest.write_text("fixture manifest")
    frozen = {
        "app_manifest_sha256": successor.sha(manifest),
        "historical_result_sha256": "historical commitment",
        "cache_fix_result_sha256": "cache fix commitment",
    }
    monkeypatch.setattr(successor, "verify_plan", lambda: (frozen, plan))
    for path, data in (
        (stage / "inputs.json", row),
        (stage / "freeze.json", frozen),
        (successor.PUBLIC, frozen),
        (stage / "profile.json", {"guard": None}),
        (
            stage / "audit.json",
            {"condition": condition, "role": row["role"], "status": "analysis_unavailable"},
        ),
    ):
        successor.raw.write(path, data)
    (stage / "worker.log").write_text("private worker text")
    archives.seal_case(work, stage, successor.archive_paths(condition)[0])
    result = successor.readout()
    assert result["status"] == "partial"
    assert result["sealed_cases"] == result["verified_sealed_uploads"] == 1
    assert result["overall"]["case_outcomes"] == {"analysis_unavailable": 1, "run_pending": 5}
    public = json.dumps(result)
    assert condition not in public and row["sha256"] not in public and "private" not in public
    (stage / "profile.json").write_text("modified resource receipt")
    with pytest.raises(ValueError, match="Sealed stage"):
        successor.readout()
