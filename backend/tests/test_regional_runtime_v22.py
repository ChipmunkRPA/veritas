"""Actual short worker exits, unavailable reserves and unsupported worker refusal."""

import json
from types import SimpleNamespace

import pytest

from scripts import relayformer_learning_runtime_v22 as runtime


@pytest.mark.parametrize("module", sorted(runtime.MODULES))
def test_actual_help_worker_has_clean_retained_exit(tmp_path, module):
    folder = tmp_path / "help"
    runtime.run(module, ["--help"], folder)
    record = json.loads((folder / "profile.json").read_text())
    assert record["module"] == module and record["child_exit_code"] == 0
    assert record["sample_errors"] == 0 and record["guard"] is None


def test_actual_failed_arguments_stay_failed_and_cannot_be_overwritten(tmp_path):
    folder = tmp_path / "failure"
    with pytest.raises(RuntimeError):
        runtime.run("scripts.image_source_groups_v22", ["--unknown-option"], folder)
    record = json.loads((folder / "profile.json").read_text())
    assert record["child_exit_code"] == 2
    with pytest.raises(FileExistsError):
        runtime.run("scripts.image_source_groups_v22", ["--help"], folder)
    assert json.loads((folder / "profile.json").read_text()) == record


def test_unknown_worker_is_refused_before_any_output(tmp_path):
    folder = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="supported"):
        runtime.run("scripts.anything_else", [], folder)
    assert not folder.exists()


def test_low_reserve_stops_before_worker_start(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("Worker must not start"))
    folder = tmp_path / "reserve"
    with pytest.raises(RuntimeError):
        runtime.run("scripts.image_source_groups_v22", ["--help"], folder)
    record = json.loads((folder / "profile.json").read_text())
    assert record["guard"] == "initial_reserve_unavailable" and record["child_exit_code"] is None
