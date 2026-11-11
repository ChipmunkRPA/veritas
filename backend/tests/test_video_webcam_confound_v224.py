from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import video_webcam_confound_v224 as study


def packed(value):
    return json.dumps(value, sort_keys=True).encode()


def hashed(value):
    return hashlib.sha256(value).hexdigest()


def archived_fixture():
    lock_sha = "1" * 64
    source_sha = "2" * 64
    lock = {
        "app_manifest_sha256": "3" * 64,
        "code_sha256": {"backend/scripts/video_vcd2_full_app_v202.py": "4" * 64},
    }
    row = {"role": "unchanged", "sha256": source_sha}
    case = {"result_hash": "5" * 64, "analysis_status": "done", "pipeline_status": "done"}
    runtime = {"ok": True}
    record = {
        "method": "face-appearance-change/1",
        "centers": {"before": [2, 6], "during": [10, 14, 18], "after": [22, 26]},
    }
    raw = {
        "details": {
            "appearance_check_status": "ok",
            "face_sequence": record,
            "face_sequence_windows": [record],
        }
    }
    evidence = {
        "id": "ev",
        "analysis_id": "a",
        "revision": 1,
        "original_result_hash": case["result_hash"],
        "source": {"sha256": source_sha},
        "status": "ready",
        "packets": [{"semantics": "face-appearance-change/1", "face_sequence": record}],
    }
    probe = {"frame_table": {"pts": [0, 1]}}
    raw_bytes, evidence_bytes, probe_bytes = map(packed, (raw, evidence, probe))
    raw_sha, evidence_sha, probe_sha = map(hashed, (raw_bytes, evidence_bytes, probe_bytes))
    db = sqlite3.connect(":memory:")
    db.executescript("""CREATE TABLE analyses(id, sha256, status, result_hash, revision);
       CREATE TABLE artefacts(id, analysis_id, sha256, role, media_type, created_by, created_at);
       CREATE TABLE detector_runs(analysis_id, detector_id, status, raw_output_artefact_id);
       CREATE TABLE analysis_stages(analysis_id, stage, status, output_artefact_id);""")
    db.execute("INSERT INTO analyses VALUES (?,?,?,?,?)", ("a", source_sha, "done", case["result_hash"], 1))
    db.executemany(
        "INSERT INTO artefacts VALUES (?,?,?,?,?,?,?)",
        [
            ("raw", "a", raw_sha, "raw", "application/json", "route", "1"),
            ("ev", "a", evidence_sha, "exhibit_payload", "application/json", "video-evidence-packets/7", "2"),
            ("probe", "a", probe_sha, "stage_output", "application/json", "probe", "1"),
        ],
    )
    db.execute("INSERT INTO detector_runs VALUES (?,?,?,?)", ("a", "vid.review.face_appearance", "ok", "raw"))
    db.execute("INSERT INTO analysis_stages VALUES (?,?,?,?)", ("a", "probe", "done", "probe"))
    db.commit()
    db_bytes = db.serialize()
    db.close()
    worker = {
        "status": "complete",
        "freeze_sha256": lock_sha,
        "manifest_sha256": lock["app_manifest_sha256"],
        "private_case_sha256": hashed(packed(case)),
        "private_runtime_sha256": hashed(packed(runtime)),
    }
    resource = b"monitor data"
    profile = {
        "schema": "chipmunk.video-appearance-resource-profile/108",
        "command": "authored",
        "child_started": True,
        "child_exit_code": 0,
        "guard": None,
        "sample_errors": 0,
        "error_types": [],
        "resource_limit_observations": [],
        "runtime_sha256": lock["code_sha256"]["backend/scripts/video_vcd2_full_app_v202.py"],
        "resource_journal_sha256": hashed(resource),
        "worker_receipt_sha256": hashed(packed(worker)),
    }
    audit = {
        "condition": "slot-01-unchanged",
        "role": "unchanged",
        "status": "scored",
        "appearance_selected_window": True,
        "appearance_measured_windows": 1,
    }
    files = {
        "stage/audit.json": packed(audit),
        "stage/profile.json": packed(profile),
        "stage/resources.jsonl": resource,
        "stage/output/result.json": packed(worker),
        "stage/freeze.json": packed(lock),
        "stage/inputs.json": packed(row),
        "work/case.json": packed(case),
        "work/runtime.json": packed(runtime),
        "work/data/veritas.db": db_bytes,
    }
    for digest, body in ((raw_sha, raw_bytes), (evidence_sha, evidence_bytes), (probe_sha, probe_bytes)):
        files[f"work/data/objects/sha256/{digest[:2]}/{digest[2:4]}/{digest}"] = body
    return lock, row, record, files, lock_sha


def stream_for(files):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tar:
        for path, body in files.items():
            header = tarfile.TarInfo(path)
            header.size = len(body)
            tar.addfile(header, io.BytesIO(body))
    data.seek(0)
    return tarfile.open(fileobj=data, mode="r:")


def test_archived_analysis_and_packet_bind_selected_record(monkeypatch):
    lock, row, record, files, lock_sha = archived_fixture()
    monkeypatch.setattr(study, "read", lambda path: lock)
    monkeypatch.setattr(study, "sha", lambda path: lock_sha)
    with stream_for(files) as tar:
        selected, table, unavailable = study.select_record(tar, "slot-01-unchanged", row)
    assert selected == record and table == {"pts": [0, 1]} and unavailable is None
    files["stage/resources.jsonl"] = b"changed"
    with stream_for(files) as tar, pytest.raises(ValueError, match="binding differs"):
        study.select_record(tar, "slot-01-unchanged", row)


def test_incomplete_and_absent_selected_window_are_explicit(monkeypatch):
    lock, row, _, files, lock_sha = archived_fixture()
    audit = json.loads(files["stage/audit.json"])
    audit["status"] = "guard_or_worker_failed"
    files["stage/audit.json"] = packed(audit)
    with stream_for({"stage/audit.json": files["stage/audit.json"]}) as tar:
        assert study.select_record(tar, "slot-01-unchanged", row)[2] == "analysis_unavailable"
    audit.update(status="scored", appearance_selected_window=False)
    files["stage/audit.json"] = packed(audit)
    monkeypatch.setattr(study, "read", lambda path: lock)
    monkeypatch.setattr(study, "sha", lambda path: lock_sha)
    with stream_for(files) as tar:
        assert study.select_record(tar, "slot-01-unchanged", row)[2] == "no_selected_appearance_window"


def test_role_triplets_preserve_exact_membership_and_reject_overlap():
    record = {"centers": {"before": [2, 6], "during": [10, 14, 18], "after": [22, 26]}}
    groups = study.role_indices(record)
    assert groups["before"] == [1, 2, 3, 5, 6, 7]
    assert groups["during"] == [9, 10, 11, 13, 14, 15, 17, 18, 19]
    record["centers"]["after"] = [19, 26]
    assert study.role_indices(record) is None


def test_assess_binds_all_rgb_clock_and_metadata(monkeypatch):
    from app.detectors.video import decode
    from app.evidence import face_interval_binding

    record = {"centers": {"before": [2, 6], "during": [10, 14, 18], "after": [22, 26]}}
    groups = study.role_indices(record)
    indices = [i for role in study.diagnostic.ROLES for i in groups[role]]
    rgb = np.full((64, 64, 3), 100, np.uint8)
    digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    record["frame_inputs"] = [
        {
            "frame_index": i,
            "pts": i,
            "decoded_rgb_sha256": digest,
            "landmarks": [[26, 27], [38, 27], [32, 32], [27, 39], [37, 39]],
            "bbox": {"x": 20, "y": 20, "w": 24, "h": 24},
        }
        for i in indices
    ]
    table = {
        "time_base": "1/30",
        "pts": list(range(30)),
        "key_frames": [0],
        "pict_types": "I" * 30,
        "width": 64,
        "height": 64,
        "coded_width": 64,
        "coded_height": 64,
    }
    expected = {"pts": list(range(30)), "time_base": "1/30", "size": (64, 64), "rgb": [digest] * 30}
    monkeypatch.setattr(face_interval_binding, "validate_sequence_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        decode,
        "decode_frames",
        lambda *args, **kwargs: (SimpleNamespace(index=i, pts=i, rgb=rgb) for i in indices),
    )
    assert study.assess(record, table, expected, None)["status"] == "measured"
    expected["rgb"][indices[0]] = "0" * 64
    with pytest.raises(ValueError, match="Native RGB"):
        study.assess(record, table, expected, None)


def test_cross_arm_comparison_requires_same_selected_frames():
    results = {role: {"status": "measured"} for role in study.prior.ROLES}
    selections = {
        "unchanged": {"during": [1, 2]},
        "chroma_conversion_only": {"during": [1, 2]},
        "recompression_only": {"during": [2, 3]},
    }
    assert study.comparability(results, selections) == {
        "chroma_conversion_only": "same_role_frame_indices",
        "recompression_only": "different_selected_frames",
    }
    results["unchanged"]["status"] = "unavailable"
    assert set(study.comparability(results, {}).values()) == {"measurement_unavailable"}
