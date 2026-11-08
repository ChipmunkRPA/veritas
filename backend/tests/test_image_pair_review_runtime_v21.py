"""The separate local reviewer must retain both successful and failed workers."""

import json
from types import SimpleNamespace

import pytest

from scripts import image_pair_review_runtime_v21 as runtime


def test_actual_short_reviewer_exit_is_recorded(tmp_path):
    folder = tmp_path / "success"
    record = runtime.run(["--help"], folder)
    assert record["child_exit_code"] == 0 and record["guard"] is None
    assert record["sample_errors"] == 0
    assert "folder" in (folder / "worker.log").read_text()


def test_actual_argument_failure_does_not_pass_or_overwrite(tmp_path):
    folder = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Failed/guarded"):
        runtime.run(["--not-a-valid-review-option"], folder)
    receipt = json.loads((folder / "profile.json").read_text())
    assert receipt["child_exit_code"] == 2
    with pytest.raises(FileExistsError):
        runtime.run(["--help"], folder)
    assert json.loads((folder / "profile.json").read_text()) == receipt


def test_low_initial_reserve_stops_before_worker_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("Worker must not start"))
    folder = tmp_path / "reserve"
    with pytest.raises(RuntimeError, match="Failed/guarded"):
        runtime.run(["--help"], folder)
    receipt = json.loads((folder / "profile.json").read_text())
    assert receipt["child_exit_code"] is None and receipt["guard"] == "initial_reserve_unavailable"
