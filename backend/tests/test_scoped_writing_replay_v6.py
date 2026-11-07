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
