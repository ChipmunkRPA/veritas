"""Bounded source replay must reproduce the frozen clocks and reject changed bytes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import replay_audio_bandwidth_public as replay


def test_schedule_matches_all_published_windows_without_wavs():
    parent_samples = {"ES2007c": 38038528, "IS1004c": 36223658, "ES2012b": 35834667, "IS1001b": 33869824}
    sources = json.loads(replay.RECEIPT.read_bytes())["freeze"]["sources"]
    for row in sources:
        capture = row["session"]
        ordinal = int(row["id"].split("-")[1])
        assert replay.start_sample(capture, ordinal, parent_samples[capture]) == row["start_sample"]


def test_changed_cached_or_publisher_bytes_are_refused(tmp_path, monkeypatch):
    payload = b"abcdefgh"
    pins = tuple(replay.trial.sha(part) for part in (b"abcd", b"efgh"))

    def publisher(capture, start, size):
        assert (capture, start, size) == ("ES2007c", 44, 8)
        return payload

    monkeypatch.setattr(replay, "publisher_range", publisher)
    file = Path(tmp_path) / "capture.pcm"
    assert replay.cached_range(tmp_path, "ES2007c", file.name, 44, 8, pins, acquire=True) == payload
    assert file.read_bytes() == payload
    file.write_bytes(b"abcdefgh".replace(b"d", b"X"))
    with pytest.raises(ValueError, match="Published AMI bytes changed"):
        replay.cached_range(tmp_path, "ES2007c", file.name, 44, 8, pins, acquire=True)
    assert file.read_bytes() == b"abcXefgh"
    file.unlink()
    with pytest.raises(ValueError, match="Published AMI bytes changed"):
        replay.cached_range(tmp_path, "ES2007c", file.name, 44, 8, pins[::-1], acquire=True)
    assert not file.exists()
