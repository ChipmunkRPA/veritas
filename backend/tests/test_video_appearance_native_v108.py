import ast
import inspect
import math
from copy import deepcopy

import numpy as np
import pytest

from app.detectors.video import face_appearance
from scripts import measure_video_specialists_v18 as legacy
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_native_v108 import fixed_response, validate_case
from scripts.video_appearance_plan_v107 import case_plan


def fixture(condition="swap_a"):
    n = 150
    source = ["a" * 64] * n
    swap = source[:50] + ["b" * 64] * 50 + source[100:]
    group = {
        "group": "authored",
        "role": "selection",
        "construction": {"frames": n, "frame_step_seconds": "1/25", "edit_frames": [50, 100]},
        "sources": [
            {"rgb_sha256": h, "frame_table": {"width": 1280, "height": 720}} for h in (source, swap, swap)
        ],
    }
    planned = case_plan(0, group, condition)
    constructed = {
        "condition": condition,
        "label": int(condition.startswith("swap")),
        "file": condition + ".mp4",
        "construction": deepcopy(group["construction"]),
        "encoded_duration_seconds": "6",
        "expected_rgb_sha256": swap if condition.startswith("swap") else source,
        "actual_changed_frames": [] if condition == "original" else list(range(50, 100)),
        "frame_table": {
            "time_base": "1/25",
            "pts": list(range(n)),
            "key_frames": [0],
            "pict_types": "I" + "P" * (n - 1),
            "width": 1280,
            "height": 720,
            "coded_width": 1280,
            "coded_height": 720,
            "rotation": 0,
            "truncated": False,
            "interpolated": [],
        },
    }
    return planned, constructed


@pytest.mark.parametrize(
    "condition", ["original", "swap_a", "swap_b", "ordinary_light", "ordinary_blur", "ordinary_time_step"]
)
def test_native_clock_and_truth_accept_all_conditions_without_relabelling_controls(condition):
    plan, case = fixture(condition)
    before = deepcopy((plan, case))
    validate_case(plan, case)
    assert (plan, case) == before
    assert plan["planned_scope_label"] == int(condition.startswith("swap"))


@pytest.mark.parametrize(
    "change",
    [
        "pts",
        "duration",
        "rotation",
        "truncated",
        "interpolated",
        "width",
        "frames",
        "label",
        "condition",
        "path",
        "changed",
        "boundary",
    ],
)
def test_wrong_bytes_clock_geometry_and_scope_cannot_be_bound(change):
    plan, case = fixture()
    if change == "pts":
        case["frame_table"]["pts"][75] += 1
    elif change == "duration":
        case["encoded_duration_seconds"] = "6001/1000"
    elif change == "rotation":
        case["frame_table"]["rotation"] = 90
    elif change == "truncated":
        case["frame_table"]["truncated"] = True
    elif change == "interpolated":
        case["frame_table"]["interpolated"] = [1]
    elif change == "width":
        case["frame_table"]["width"] = 1279
    elif change == "frames":
        case["expected_rgb_sha256"].pop()
    elif change == "label":
        case["label"] = 0
    elif change == "condition":
        case["condition"] = "original"
    elif change == "path":
        case["file"] = "../swap_a.mp4"
    elif change == "changed":
        case["actual_changed_frames"].pop()
    else:
        case["construction"]["edit_frames"] = [49, 100]
    with pytest.raises(ValueError):
        validate_case(plan, case)


def reference(unit, vectors):
    normalized = {}
    for i, vector in vectors.items():
        norm = math.sqrt(math.fsum(float(x) ** 2 for x in vector))
        normalized[i] = [float(x) / norm for x in vector]
    pools = []
    for key in ("before_centers", "middle_centers", "after_centers"):
        centers = unit[key]
        pools.append(
            [
                math.fsum(math.fsum(normalized[i + k][j] for k in (-1, 0, 1)) / 3 for i in centers)
                / len(centers)
                for j in range(128)
            ]
        )

    def distance(a, b):
        return 1 - max(
            -1.0,
            min(
                1.0,
                math.fsum(x * y for x, y in zip(a, b, strict=True))
                / math.sqrt(math.fsum(x * x for x in a) * math.fsum(y * y for y in b)),
            ),
        )

    a, b, c = pools
    return min(distance(b, a), distance(b, c)) - 0.35 * distance(a, c)


def legacy_appearance_readout():
    # Execute the retained function's appearance branch with its two lazy imports
    # removed. The unused feature branch is never selected or loaded (no Torch).
    module = ast.parse(inspect.getsource(legacy.fixed_change))
    imports = [n for n in module.body[0].body if isinstance(n, ast.ImportFrom)]
    assert len(imports) == 2 and {n.module for n in imports} == {"app.detectors.video"}
    module.body[0].body = [n for n in module.body[0].body if not isinstance(n, ast.ImportFrom)]
    scope = {"np": np, "math": math, "identity": face_appearance, "feature": None, "require": legacy.require}
    exec(compile(module, "retained-appearance-arithmetic", "exec"), scope)
    return scope["fixed_change"]


@pytest.mark.parametrize("seed", range(5))
def test_fixed_readout_matches_retained_arithmetic_and_separate_scalar_sum(seed):
    unit = fixture()[0]["units"][0]
    indices = set(unit["location"]["sampled_frames"] + unit["location"]["context_frames"])
    vectors = {i: np.sin(np.arange(128) * 0.03 + i * 0.007 + seed * 0.1) + 0.4 for i in indices}
    boxes = dict.fromkeys(indices, (20, 20, 100, 100))
    actual = fixed_response(unit, vectors, boxes)
    assert actual == legacy_appearance_readout()(unit, vectors, appearance=True)
    assert actual == pytest.approx(reference(unit, vectors), abs=1e-13)


def test_missing_frames_or_discontinuous_boxes_stay_unassessed():
    unit = fixture()[0]["units"][0]
    indices = sorted(set(unit["location"]["sampled_frames"] + unit["location"]["context_frames"]))
    vectors = {i: np.ones(128) for i in indices}
    boxes = dict.fromkeys(indices, (20, 20, 100, 100))
    assert fixed_response(unit, vectors, boxes) == pytest.approx(0.0, abs=1e-14)
    absent = vectors.copy()
    del absent[indices[0]]
    assert fixed_response(unit, absent, boxes) is None
    boxes[indices[-1]] = (1000, 1000, 100, 100)
    assert fixed_response(unit, vectors, boxes) is None


def test_invalid_model_vector_is_a_contract_error_not_an_original_verdict():
    unit = fixture()[0]["units"][0]
    indices = set(unit["location"]["sampled_frames"] + unit["location"]["context_frames"])
    vectors = {i: np.zeros(128) for i in indices}
    with pytest.raises(ValueError):
        fixed_response(unit, vectors, dict.fromkeys(indices, (20, 20, 100, 100)))


def snapshot(ram=6, rss=0, elapsed=0):
    return {
        "tree_rss_bytes": int(rss * 1024**3),
        "ram_available_bytes": int(ram * 1024**3),
        "disk_free_bytes": 3 * 1024**3,
        "elapsed_s": elapsed,
    }


@pytest.mark.parametrize("command", ["authored", "extract"])
def test_neural_admission_and_all_running_bounds_are_unchanged(command):
    assert runtime.violations(snapshot(), command, initial=True) == []
    assert "initial_ram_headroom" in runtime.violations(snapshot(ram=5.9), command, initial=True)
    assert "tree_rss_limit" in runtime.violations(snapshot(rss=6.01), command)
    assert "elapsed_time_limit" in runtime.violations(snapshot(elapsed=900.01), command)
    assert "ram_reserve" in runtime.violations(snapshot(ram=1.49), command)


@pytest.mark.parametrize("command", ["monitor", "bind"])
def test_data_only_bounds_do_not_change_neural_limits(command):
    assert runtime.violations(snapshot(ram=3), command, initial=True) == []
    assert "initial_ram_headroom" in runtime.violations(snapshot(ram=2.99), command, initial=True)
    assert "tree_rss_limit" in runtime.violations(snapshot(rss=1.51), command)
    assert "ram_reserve" in runtime.violations(snapshot(ram=1.49), command)


def test_short_ram_is_read_only_admission_without_model_stage_or_retry(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(runtime, "TRIAL", tmp_path)
    monkeypatch.setattr(runtime, "ARCHIVE", tmp_path)
    monkeypatch.setattr(runtime, "verify", lambda: None)
    monkeypatch.setattr(runtime, "prerequisites", lambda *args: True)
    monkeypatch.setattr(runtime.original, "observe", lambda *args: snapshot(ram=5.3))
    monkeypatch.setattr(
        runtime.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not launch a child")
    )
    runtime.run("authored")
    assert "pending_headroom_no_stage_attempt" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_existing_stage_is_preserved_before_any_child_can_start(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "TRIAL", tmp_path)
    monkeypatch.setattr(runtime, "verify", lambda: None)
    (tmp_path / "authored").mkdir()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("No retry"))
    with pytest.raises(ValueError, match="Preserve"):
        runtime.run("authored")
