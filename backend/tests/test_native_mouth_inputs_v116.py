import copy

import pytest

from scripts import native_mouth_inputs_v116 as inputs


def fixture():
    return {
        "native20_edit": {"frames_half_open": [4, 24]},
        "frames": [{"video_frame_index": i, "source_sample_status": "supported"} for i in range(28)],
    }


def test_exact_native_edit_includes_two_before_and_after():
    plan = fixture()
    before = copy.deepcopy(plan)
    assert inputs.edit_support(plan) == list(range(2, 26))
    assert plan == before


def test_short_source_is_retained_without_invented_pictures():
    assert inputs.edit_support({"native20_edit": None}) == []


@pytest.mark.parametrize(
    "problem",
    ["short_edit", "missing_before", "missing_during", "missing_after", "outside_audio", "negative_index"],
)
def test_incomplete_edit_or_context_support_is_rejected(problem):
    plan = fixture()
    if problem == "short_edit":
        plan["native20_edit"]["frames_half_open"] = [4, 23]
    elif problem == "negative_index":
        plan["native20_edit"]["frames_half_open"] = [1, 21]
    elif problem == "outside_audio":
        plan["frames"][12]["source_sample_status"] = "unavailable"
    else:
        index = {"missing_before": 2, "missing_during": 12, "missing_after": 25}[problem]
        plan["frames"] = [f for f in plan["frames"] if f["video_frame_index"] != index]
    with pytest.raises(ValueError):
        inputs.edit_support(plan)


def test_data_worker_routing_preserves_original_argv(monkeypatch):
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", "private", "0"]
    calls = []
    monkeypatch.setattr(inputs.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)))
    inputs.routed_popen(args, cwd="backend")
    assert args[3] == "scripts.video_appearance_native_v108"
    assert calls == [([*args[:3], "scripts.native_mouth_inputs_v116", *args[4:]], {"cwd": "backend"})]
    with pytest.raises(ValueError):
        inputs.routed_popen([*args[:4], "extract", *args[5:]])


@pytest.mark.parametrize("batch", [-1, 8, True, 1.0])
def test_undeclared_data_batch_cannot_start(batch):
    with pytest.raises(ValueError):
        inputs.run(batch)


def test_data_limit_is_3gib_while_neural_limit_stays_6gib():
    row = {
        "elapsed_s": 0,
        "tree_rss_bytes": 0,
        "ram_available_bytes": 3 * 1024**3,
        "disk_free_bytes": 3 * 1024**3,
    }
    assert inputs.runtime.violations(row, "bind", initial=True) == []
    assert "initial_ram_headroom" in inputs.runtime.violations(row, "extract", initial=True)
