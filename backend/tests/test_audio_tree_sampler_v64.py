from types import SimpleNamespace

import psutil
import pytest

from scripts.audio_tree_sampler_v64 import sample_tree


class Process:
    def __init__(self, pid, rss=100, members=(), error=None, running=True):
        self.pid, self.rss, self.members, self.error, self.running = pid, rss, members, error, running

    def memory_info(self):
        if self.error:
            raise self.error
        return SimpleNamespace(rss=self.rss)

    def children(self, recursive):
        assert recursive is True
        return list(self.members)

    def is_running(self):
        return self.running

    def status(self):
        return psutil.STATUS_RUNNING

    def wait(self, timeout):
        assert timeout == .25
        raise psutil.TimeoutExpired(timeout, self.pid)


def test_verified_exited_decoder_does_not_kill_live_worker():
    dead = Process(2, error=psutil.NoSuchProcess(2), running=False)
    root = Process(1, members=[dead, Process(3, rss=50)])
    rss, events, terminal = sample_tree(root, SimpleNamespace(poll=lambda: None))
    assert rss == 150 and terminal is False
    assert events[0]["classification"] == "verified_descendant_exit_during_memory_sample"


def test_missing_measurement_from_live_descendant_remains_failure():
    root = Process(1, members=[Process(2, error=psutil.NoSuchProcess(2), running=True)])
    with pytest.raises(psutil.NoSuchProcess):
        sample_tree(root, SimpleNamespace(poll=lambda: None))


def test_delayed_macos_descendant_exit_requires_bounded_wait():
    process = Process(2, error=psutil.NoSuchProcess(2), running=True)
    process.wait = lambda timeout: None
    rss, events, terminal = sample_tree(Process(1, members=[process]), SimpleNamespace(poll=lambda: None))
    assert rss == 100 and len(events) == 1 and terminal is False


def test_live_worker_cannot_be_declared_finished_from_missing_pid():
    with pytest.raises(psutil.NoSuchProcess):
        sample_tree(Process(1, error=psutil.NoSuchProcess(1)), SimpleNamespace(poll=lambda: None))


def test_authoritative_terminal_worker_event_is_preserved():
    rss, events, terminal = sample_tree(Process(1, error=psutil.NoSuchProcess(1)), SimpleNamespace(poll=lambda: 0))
    assert rss == 0 and terminal is True
    assert events[-1]["child_exit_code"] == 0


def test_permission_failure_cannot_be_silently_ignored():
    with pytest.raises(psutil.AccessDenied):
        sample_tree(Process(1, members=[Process(2, error=psutil.AccessDenied(2))]), SimpleNamespace(poll=lambda: None))


def test_verified_enumeration_race_is_resampled(monkeypatch):
    root = Process(1)
    calls = []

    def children(recursive):
        calls.append(True)
        if len(calls) == 1:
            raise psutil.NoSuchProcess(2)
        return [Process(3, rss=25)]

    root.children = children
    monkeypatch.setattr(psutil, "Process", lambda pid: Process(pid, running=False))
    rss, events, terminal = sample_tree(root, SimpleNamespace(poll=lambda: None))
    assert rss == 125 and len(calls) == 2 and terminal is False
    assert len(events) == 1


def test_repeated_enumeration_races_fail_instead_of_sampling_root_only(monkeypatch):
    root = Process(1)
    root.children = lambda recursive: (_ for _ in ()).throw(psutil.NoSuchProcess(2))
    monkeypatch.setattr(psutil, "Process", lambda pid: Process(pid, running=False))
    with pytest.raises(RuntimeError, match="complete live"):
        sample_tree(root, SimpleNamespace(poll=lambda: None))
