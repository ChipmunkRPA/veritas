"""Continuation preserves fitting mathematics and cannot accept a missing live PID."""

import ast
import json
import subprocess
from pathlib import Path

import psutil
import pytest

from scripts import image_regional_continuation_v41 as continuation
from scripts.image_regional_continuation_runtime_v41 import verified_terminal_transition

ROOT = Path(__file__).resolve().parents[2]


def function(path, name):
    text = path.read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(text, node)


def test_fitting_body_preserves_all_frozen_math_and_source_calls():
    expected = function(ROOT / "backend/scripts/image_regional_training_v26.py", "train")
    expected = expected.replace('state["plan_sha256"] == sha(folder / "plan.json")', 'state["plan_sha256"] == expected_predecessor_plan_sha(folder, number)')
    expected = expected.replace("chipmunk.private-regional-checkpoint/26", "chipmunk.private-regional-checkpoint/41").replace("chipmunk.regional-supervised-phase/26", "chipmunk.regional-supervised-phase/41")
    actual = function(ROOT / "backend/scripts/image_regional_continuation_v41.py", "train")
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))


@pytest.mark.parametrize("number", [0, 32, 40, True, None])
def test_other_phases_cannot_restart_or_skip_the_frozen_prefix(tmp_path, number):
    with pytest.raises(ValueError):
        continuation.checkpoint_predecessor(tmp_path, {}, number)


def test_first_continuation_uses_original_plan_then_its_own_plan(tmp_path):
    (tmp_path / "plan.json").write_text("authored continuation plan")
    assert continuation.expected_predecessor_plan_sha(tmp_path, 33) == continuation.ORIGINAL_PLAN_SHA
    assert continuation.expected_predecessor_plan_sha(tmp_path, 34) == continuation.sha(tmp_path / "plan.json")


def test_guarded_predecessor_is_rejected_before_loading_any_model(tmp_path):
    old = tmp_path / "original"
    (old / "phase-032").mkdir(parents=True)
    (old / "phase-032/completed.json").write_text("{}")
    (old / "phase-032-profile").mkdir()
    (old / "phase-032-profile/profile.json").write_text(json.dumps({"child_exit_code": 0, "guard": "live_sampling_unavailable", "sample_errors": 1, "resource_limit_observations": []}))
    with pytest.raises(ValueError, match="successfully monitored"):
        continuation.checkpoint_predecessor(tmp_path, {"original_study": str(old)}, 33)


@pytest.mark.parametrize("status", [0, 7, -15])
def test_missing_worker_requires_actual_terminal_wait_status(status):
    class Child:
        def wait(self, timeout):
            assert timeout == 0.25
            return status
    event = verified_terminal_transition(Child(), psutil.NoSuchProcess(123))
    assert event["classification"] == "verified_missing_process_terminal" and event["child_exit_code"] == status


def test_missing_pid_with_still_running_worker_stays_a_failure():
    class Child:
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("authored live worker", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        verified_terminal_transition(Child(), psutil.NoSuchProcess(123))
