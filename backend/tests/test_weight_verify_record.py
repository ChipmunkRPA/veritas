"""Regression coverage for concurrent writers to the signed weight verification cache."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.engine import fetchers


def legacy_record(record, file):
    record.key_path.write_bytes(b"fixed legacy test key")
    pin = fetchers.sha256_file(file)
    st = file.stat()
    entries = {
        str(file.resolve()): {
            "sha256": pin,
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "verified_at": "2020-01-01T00:00:00+00:00",
        }
    }
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(record.key_path.read_bytes(), payload, hashlib.sha256).hexdigest()
    record.path.write_text(json.dumps({"entries": entries, "hmac": signature}), encoding="utf-8")
    return pin


def test_independent_writers_merge_signed_weight_entries(tmp_path, monkeypatch):
    path = tmp_path / ".veritas-verified.json"
    key = tmp_path / "verify_hmac.key"
    first = fetchers.VerifyRecord(path, key)
    second = fetchers.VerifyRecord(path, key)
    legacy_file, first_file, second_file = (tmp_path / name for name in ("legacy", "first", "second"))
    for file in (legacy_file, first_file, second_file):
        file.write_text(file.name, encoding="utf-8")
    legacy_pin = legacy_record(first, legacy_file)
    first_pin = fetchers.sha256_file(first_file)
    second_pin = fetchers.sha256_file(second_file)
    assert first.is_fresh(legacy_file, legacy_pin)

    first_loaded = threading.Event()
    second_contended = threading.Event()
    load_first = first._load
    flock = fcntl.flock

    def hold_first_after_read():
        entries = load_first()
        first_loaded.set()
        assert second_contended.wait(5), "The second writer never reached the shared record lock"
        return entries

    def observe_contention(fd, operation):
        if first_loaded.is_set():
            try:
                with pytest.raises(BlockingIOError):
                    flock(fd, operation | fcntl.LOCK_NB)
            finally:
                second_contended.set()
        flock(fd, operation)

    def write_second():
        assert first_loaded.wait(5), "The first writer never reached the read-modify-write window"
        second.record(second_file, second_pin)

    monkeypatch.setattr(first, "_load", hold_first_after_read)
    monkeypatch.setattr(fcntl, "flock", observe_contention)
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(first.record, first_file, first_pin)
        right = pool.submit(write_second)
        left.result(timeout=10)
        right.result(timeout=10)

    reader = fetchers.VerifyRecord(path, key)
    assert reader.is_fresh(legacy_file, legacy_pin)
    assert reader.is_fresh(first_file, first_pin)
    assert reader.is_fresh(second_file, second_pin)
    assert not list(tmp_path.glob(".veritas-verified.json.*.tmp"))


def test_failed_replacement_cleans_owned_temp_and_preserves_previous_record(tmp_path, monkeypatch):
    record = fetchers.VerifyRecord(tmp_path / ".veritas-verified.json", tmp_path / "verify_hmac.key")
    previous, new = tmp_path / "previous", tmp_path / "new"
    previous.write_bytes(b"old verified weight")
    new.write_bytes(b"new verified weight")
    previous_pin = legacy_record(record, previous)
    new_pin = fetchers.sha256_file(new)
    tmp_paths = []

    def fail_replace(src, dst):
        source = Path(src)
        assert dst == record.path
        assert source.parent == record.path.parent and source.is_file()
        assert source != record.path.with_name(record.path.name + ".tmp")
        tmp_paths.append(source)
        raise OSError("injected replace failure")

    monkeypatch.setattr(fetchers.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        record.record(new, new_pin)
    assert len(tmp_paths) == 1 and not tmp_paths[0].exists()
    assert record.is_fresh(previous, previous_pin)
    assert not record.is_fresh(new, new_pin)
