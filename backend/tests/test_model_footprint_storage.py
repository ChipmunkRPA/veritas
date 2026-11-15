from __future__ import annotations

import errno
from pathlib import Path

from app.engine.models import FootprintStore


def test_full_storage_keeps_the_live_memory_observation_and_prior_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "model_footprints.json"
    store = FootprintStore(path)
    store.record_measured("text", "mps", 120, 0.1)
    prior_snapshot = path.read_bytes()

    def full(_source, _destination):
        raise OSError(errno.ENOSPC, "No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", full)
        store.record_measured("text", "mps", 180, 0.2)
        assert store.measured("text", "mps") == 180
        assert store.maximum_positive_delta("text", "mps") == 180
        assert path.read_bytes() == prior_snapshot
        assert not list(tmp_path.glob(".model_footprints.*.tmp"))

    store.record_measured("text", "mps", 130, 0.1)
    restored = FootprintStore(path)
    assert restored.measured("text", "mps") == 130
    assert restored.maximum_positive_delta("text", "mps") == 180
