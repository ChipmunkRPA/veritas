"""Replay cannot hide decision/count changes inside floating point tolerance."""
import pytest

from scripts.replay_scoped_writing_v6 import compare, safe


def test_parameter_roundoff_is_measured_but_integer_boolean_and_missing_changes_fail():
    assert compare({'p': .2 + 1e-10}, {'p': .2}) == pytest.approx(1e-10)
    for actual, expected in ((2, 1), (True, 1), (None, .2), ([], [1]), ({'a': 1}, {'b': 1})):
        with pytest.raises(ValueError):
            compare(actual, expected)


@pytest.mark.parametrize('name', ['/private/a', '../a', '-a', 'backend/../a', '', 'backend//a'])
def test_archive_requires_exact_repository_paths(name):
    with pytest.raises(ValueError):
        safe(name)


def test_failed_frozen_subprocess_retains_its_actual_diagnostic(monkeypatch):
    from types import SimpleNamespace

    from scripts import replay_scoped_writing_v6 as replay

    monkeypatch.setattr(replay.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stderr='authored fitting failure: exact cause', stdout=''))
    with pytest.raises(RuntimeError, match='authored fitting failure: exact cause'):
        replay.run_stage(['authored-only-command'])
