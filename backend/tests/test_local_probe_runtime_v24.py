"""Every guard reason retains the actual threshold crossing; no implicit worker retry."""

import json

import psutil
import pytest

from scripts import local_probe_limits_v24 as limits
from scripts import local_probe_runtime_v24 as runtime


def healthy():
    return {"elapsed_s": 1.0, "tree_rss_bytes": 0, "ram_available_bytes": limits.START_RAM, "disk_free_bytes": limits.START_DISK}


def test_exact_limit_boundaries_are_accepted():
    row = healthy()
    row.update(elapsed_s=900, tree_rss_bytes=limits.RSS_LIMIT, ram_available_bytes=limits.RESERVE, disk_free_bytes=limits.RESERVE)
    assert limits.violations(row) == []


@pytest.mark.parametrize(("key", "value", "reason"), [("elapsed_s", 900.001, "elapsed_time_limit"), ("tree_rss_bytes", limits.RSS_LIMIT + 1, "tree_rss_limit"), ("ram_available_bytes", limits.RESERVE - 1, "ram_reserve"), ("disk_free_bytes", limits.RESERVE - 1, "disk_reserve")])
def test_each_observed_limit_has_its_own_reason(key, value, reason):
    row = healthy()
    row[key] = value
    assert limits.violations(row) == [reason]


def test_simultaneous_violations_are_all_retained():
    row = {"elapsed_s": 901, "tree_rss_bytes": limits.RSS_LIMIT + 1, "ram_available_bytes": 0, "disk_free_bytes": 0}
    assert set(limits.violations(row)) == {"elapsed_time_limit", "tree_rss_limit", "ram_reserve", "disk_reserve"}


@pytest.mark.parametrize("fault", [None, float("nan"), -1, True])
def test_invalid_time_observations_cannot_be_recorded_as_healthy(fault):
    row = healthy()
    row["elapsed_s"] = fault
    with pytest.raises(ValueError):
        limits.violations(row)


def test_initial_headroom_is_stricter_than_running_reserves():
    row = healthy()
    row["ram_available_bytes"] -= 1
    row["disk_free_bytes"] -= 1
    assert limits.violations(row) == []
    assert set(limits.violations(row, initial=True)) == {"initial_ram_headroom", "initial_disk_headroom"}


def test_unavailable_initial_space_stops_before_any_worker(tmp_path, monkeypatch):
    row = healthy()
    row["disk_free_bytes"] = limits.RESERVE - 1
    monkeypatch.setattr(limits, "observe", lambda *_: dict(row))
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("Worker must not start"))
    profile, worker = tmp_path / "profile", tmp_path / "worker"
    with pytest.raises(RuntimeError):
        runtime.run("baseline", profile, worker)
    record = json.loads((profile / "profile.json").read_text())
    assert record["guard"] == "initial_headroom_unavailable" and not record["child_started"]
    assert record["resource_limit_observations"][0]["disk_free_bytes"] == row["disk_free_bytes"]
    assert not worker.exists()


def test_healthy_preflight_does_not_claim_a_completed_model_run(tmp_path, monkeypatch):
    monkeypatch.setattr(limits, "observe", lambda *_: healthy())
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("Preflight never starts a worker"))
    record = runtime.run("checkpointed", tmp_path / "profile", tmp_path / "worker", preflight_only=True)
    assert record["eligible_at_initial_observation"] and record["preflight_only"]
    assert record["child_exit_code"] is None and not record["child_started"]


class Worker:
    pid = 24

    def __init__(self):
        self.terminated = False

    def poll(self):
        return -15 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout):
        return -15


@pytest.mark.parametrize("fault", ["disk", "live_sample"])
def test_running_stop_retains_exact_observation_or_live_sampling_failure(tmp_path, monkeypatch, fault):
    worker = Worker()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_kw: worker)
    monkeypatch.setattr(runtime.psutil, "Process", lambda _pid: object())
    count = 0

    def observe(*_args):
        nonlocal count
        count += 1
        row = healthy()
        if count > 1 and fault == "disk":
            row["disk_free_bytes"] = 123456789
        return row

    def sample(*_args):
        if fault == "live_sample":
            raise psutil.NoSuchProcess(24)
        return 123, None

    monkeypatch.setattr(limits, "observe", observe)
    monkeypatch.setattr(runtime, "sample", sample)
    folder = tmp_path / "profile"
    with pytest.raises(RuntimeError):
        runtime.run("checkpointed", folder, tmp_path / "worker")
    record = json.loads((folder / "profile.json").read_text())
    assert worker.terminated and record["child_exit_code"] == -15
    if fault == "disk":
        assert record["guard"] == "recorded_resource_limit"
        assert record["resource_limit_observations"][0]["violations"] == ["disk_reserve"]
        assert record["resource_limit_observations"][0]["disk_free_bytes"] == 123456789
    else:
        assert record["guard"] == "live_sampling_unavailable" and record["sample_errors"] == 1
    with pytest.raises(FileExistsError):
        runtime.run("checkpointed", folder, tmp_path / "worker")
