from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
from pathlib import Path

import pytest

from scripts import video_webcam_confound_resume_v227 as study


def test_checkpointed_wal_database_is_read_in_memory_without_modifying_source(tmp_path):
    path = tmp_path / "test.db"
    db = sqlite3.connect(path)
    try:
        assert db.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        db.execute("CREATE TABLE sample(value TEXT)")
        db.execute("INSERT INTO sample VALUES ('bound')")
        db.commit()
        assert db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() == (0, 0, 0)
        original = path.read_bytes()
        assert original[18:20] == b"\x02\x02"
        assert (tmp_path / "test.db-wal").stat().st_size == 0
        copied = study.normalize_header(original)
        assert copied[:18] == original[:18] and copied[18:20] == b"\x01\x01" and copied[20:] == original[20:]
        connection = study.memory_connect(":memory:")
        try:
            connection.deserialize(original)
            connection.execute("PRAGMA query_only=ON")
            assert connection.execute("SELECT value FROM sample").fetchone() == ("bound",)
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                connection.execute("INSERT INTO sample VALUES ('mutate')")
        finally:
            connection.close()
        assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(original).digest()
    finally:
        db.close()


def test_invalid_headers_and_disk_connections_are_refused():
    with pytest.raises(ValueError, match="header"):
        study.normalize_header(b"not a database")
    malformed = b"SQLite format 3\0" + b"\0\0" + b"\x01\x02" + b"\0" * 80
    with pytest.raises(ValueError, match="journal header"):
        study.normalize_header(malformed)
    with pytest.raises(ValueError, match="in-memory"):
        study.memory_connect("/tmp/not-allowed.db")


def test_original_archived_sidecar_check_precedes_normalization(monkeypatch):
    # Reuse the v224 synthetic archive fixture; no source media is opened.
    path = Path(__file__).with_name("test_video_webcam_confound_v224.py")
    spec = importlib.util.spec_from_file_location("v224_test_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock, row, record, files, lock_sha = module.archived_fixture()
    monkeypatch.setattr(study.prior, "read", lambda _: lock)
    monkeypatch.setattr(study.prior, "sha", lambda _: lock_sha)
    data = files["work/data/veritas.db"]
    files["work/data/veritas.db"] = data[:18] + b"\x02\x02" + data[20:]
    with module.stream_for(files) as tar:
        selected, _, unavailable = study.select_record(tar, "slot-01-unchanged", row)
    assert selected == record and unavailable is None
    files["work/data/veritas.db-wal"] = b"uncheckpointed"
    with module.stream_for(files) as tar, pytest.raises(ValueError, match="checkpointed"):
        study.select_record(tar, "slot-01-unchanged", row)


def test_identity_binds_original_frozen_failed_readout(monkeypatch):
    original = {"schema": "frozen"}
    failed = {
        "schema": "chipmunk.video-webcam-confound-result/224",
        "status": "complete",
        "freeze_sha256": "digest",
        "source_groups": 1,
        "model_calls": 0,
        "serving_defaults_changed": False,
        "roles": {
            role: {"status": "unavailable", "reason": "source_or_archive_binding_failed"}
            for role in study.prior.prior.ROLES
        },
    }
    monkeypatch.setattr(study.prior, "identity", lambda: original)
    monkeypatch.setattr(study, "sha", lambda _: "digest")
    monkeypatch.setattr(study, "read", lambda path: failed if path == study.FAILED else original)
    result = study.identity()
    assert result["v224_freeze_sha256"] == result["v224_failed_readout_sha256"] == "digest"
    failed["roles"]["unchanged"]["status"] = "measured"
    with pytest.raises(ValueError, match="failed v224"):
        study.identity()
