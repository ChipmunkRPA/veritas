import copy

import pytest

from scripts import native_mouth_clips_v120 as clips


def test_native_boundary_opacity_and_seed_are_deterministic():
    assert [clips.opacity(i, [4, 24]) for i in (4, 5, 22, 23)] == [0.5, 1, 1, 0.5]
    assert clips.seed(288, 4) == clips.seed(288, 4) and 0 <= clips.seed(288, 4) < 2**63
    assert clips.seed(288, 4) != clips.seed(288, 5)


@pytest.mark.parametrize("index", [3, 24, 4.0])
def test_out_of_edit_or_noninteger_picture_cannot_get_opacity(index):
    with pytest.raises(ValueError):
        clips.opacity(index, [4, 24])


def test_short_earlier_context_is_unavailable_without_mutating_aligned_plan():
    f = clips.clock.frame_plan(2, "1/10", "0", "0", 32000, "0")
    before = copy.deepcopy(f)
    assert clips.shifted(f) is None and f == before
    later = clips.clock.frame_plan(40, "4/3", "0", "0", 64000, "0")
    assert clips.shifted(later)["encoder_anchor"] == later["encoder_anchor"] - 16


@pytest.mark.parametrize("batch", [-1, 8, True, 1.0])
def test_undeclared_source_batch_is_rejected(batch):
    with pytest.raises(ValueError):
        clips.run(batch)


def test_fixed_generation_worker_does_not_change_original_argv(monkeypatch):
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract", "private", "0"]
    calls = []
    monkeypatch.setattr(clips.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    clips.routed_popen(args)
    assert args[3] == "scripts.video_appearance_native_v108"
    assert calls == [[*args[:3], "scripts.native_mouth_clips_v120", *args[4:]]]
