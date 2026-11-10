"""Completed cases survive continuation; attempted failures cannot be rerun."""
import json

import pytest

from scripts.audio_final_continuation_v90 import sha, unattempted_suffix


def complete(folder, index):
    raw = folder / f"chunk-{index:02d}.jsonl"
    raw.write_text('preserved response\n')
    profile = folder / f"chunk-{index:02d}-profile"
    profile.mkdir()
    (profile / "profile.json").write_text(json.dumps({"child_exit_code": 0, "guard": None,
        "sample_errors": 0, "resource_limit_observations": []}))
    (folder / f"chunk-{index:02d}-complete.json").write_text(json.dumps({"completed": 87, "compact_sha256": sha(raw)}))


def test_preserved_first_batch_is_not_in_the_remaining_work(tmp_path):
    complete(tmp_path, 36)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert unattempted_suffix(tmp_path) == ([36], list(range(37, 48)))
    assert {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize("partial", [".jsonl", "-profile", "-complete.json"])
def test_partial_or_before_launch_failed_attempt_is_never_retried(tmp_path, partial):
    complete(tmp_path, 36)
    path = tmp_path / ('chunk-37' + partial)
    path.mkdir() if partial == '-profile' else path.touch()
    with pytest.raises(ValueError, match="cannot be retried"):
        unattempted_suffix(tmp_path)


def test_changed_response_or_noncontiguous_completion_is_rejected(tmp_path):
    complete(tmp_path, 36)
    complete(tmp_path, 38)
    with pytest.raises(ValueError, match="contiguous"):
        unattempted_suffix(tmp_path)
    (tmp_path / 'chunk-36.jsonl').write_text('changed response\n')
    with pytest.raises(ValueError, match="changed"):
        unattempted_suffix(tmp_path)
