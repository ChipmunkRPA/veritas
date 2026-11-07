"""The external archive runner cannot change the scientific source identity."""
import pytest

from scripts.replay_frozen_specialist_release_v5 import validate


@pytest.mark.parametrize('commit', ['main', 'abc123', 'a' * 39, 'A' * 40, '--all'])
def test_mutable_or_invalid_commit_is_refused(commit):
    with pytest.raises(ValueError, match='immutable'):
        validate(commit, 'backend/scripts/real.py')


@pytest.mark.parametrize('path', ['/tmp/file', '../file', 'a/../b', '-option', 'a//b', ''])
def test_source_path_must_be_exact_and_inside_the_repository(path):
    with pytest.raises(ValueError):
        validate('a' * 40, path)


def test_explicit_commit_and_source_path_remain_unchanged():
    assert validate('a' * 40, 'docs/evaluation-panels/expected.json.gz') == 'docs/evaluation-panels/expected.json.gz'
