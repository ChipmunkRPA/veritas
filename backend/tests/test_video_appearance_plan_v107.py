from copy import deepcopy
from fractions import Fraction

import pytest

from scripts.video_appearance_plan_v107 import CONDITIONS, case_plan, windows


def group(n=150, step="1/25"):
    original = [f"original-{i}" for i in range(n)]
    first = original[:]
    second = original[:]
    a, b = n // 3, 2 * n // 3
    for i in range(a, b):
        first[i] = f"swap-{i}"
    return {
        "group": "authored-group",
        "role": "selection",
        "construction": {"frames": n, "frame_step_seconds": step, "edit_frames": [a, b]},
        "sources": [
            {"rgb_sha256": x, "frame_table": {"width": 1280, "height": 720}}
            for x in (original, first, second)
        ],
    }


def test_half_second_support_uses_native_clock_not_assumed_frame_rate():
    result = windows(150, "1/25", set(range(50, 100)))
    assert result["centers"][:4] == [0, 12, 25, 37]
    assert len(result["units"]) == 5
    u = result["units"][0]
    assert u["middle_centers"] == [37, 50, 62]
    assert u["before_centers"] == [12, 25] and u["after_centers"] == [75, 87]
    assert len(u["location"]["sampled_frames"]) == 9 and len(u["location"]["context_frames"]) == 12
    assert u["location"]["start"] == 1_440_000_000 and u["location"]["end"] == 2_560_000_000


@pytest.mark.parametrize(
    "n,step", [(144, "1/24"), (150, "1/25"), (180, "1/30"), (300, "1/50"), (360, "1/60"), (180, "1001/30000")]
)
def test_all_source_rates_preserve_bounds_and_rational_picture_positions(n, step):
    result = case_plan(0, group(n, step), "swap_a")
    clock = result["expected_clock"]
    delta = Fraction(step)
    assert clock["frame_count"] == n and Fraction(clock["duration_seconds"]) == n * delta
    assert all(p * Fraction(clock["time_base"]) == i * delta for i, p in enumerate(clock["pts"]))
    assert all(0 <= i < n for i in result["model_frame_indices"])
    for u in result["units"]:
        frames = u["location"]["sampled_frames"]
        assert u["location"]["start"] == round(frames[0] * delta * 10**9)
        assert u["location"]["end"] == round((frames[-1] + 1) * delta * 10**9)


def test_context_only_edit_is_not_labelled_as_middle_frame_evidence():
    baseline = windows(150, "1/25", set())
    index = baseline["units"][0]["location"]["context_frames"][0]
    result = windows(150, "1/25", {index})
    assert result["units"][0]["sampled_ai_frame_presence"] == 0
    assert result["units"][0]["context_contains_ai_frame"] is True


def test_ordinary_edits_keep_genuine_labels_and_known_intervals():
    source = group()
    before = deepcopy(source)
    for condition in CONDITIONS:
        case = case_plan(0, source, condition)
        assert case["known_insert_frames"] == (None if condition == "original" else [50, 100])
        if not condition.startswith("swap"):
            assert case["publisher_manipulation_label"] == case["planned_scope_label"] == 0
            assert case["intended_changed_ai_frames"] == []
    assert source == before


def test_zero_change_swap_is_preserved_as_provenance_without_fabricated_pixels():
    case = case_plan(0, group(), "swap_b")
    assert case["publisher_manipulation_label"] == 1 and case["planned_scope_label"] == 0
    assert case["intended_changed_ai_frames"] == []


def test_short_support_is_unavailable_and_does_not_become_an_original_result():
    case = case_plan(0, group(10), "swap_a")
    assert case["support_status"] == "unsupported" and case["model_frame_indices"] == []
    assert case["publisher_manipulation_label"] == 1 and case["unobserved_changed_ai_frames"] > 0
    assert "no originality verdict" in case["support_unavailable_reason"]


@pytest.mark.parametrize(
    "n,step,changed", [(True, "1/25", set()), (10, "0", set()), (10, "1/25", {-1}), (10, "1/25", {10})]
)
def test_invalid_geometry_and_out_of_source_labels_refused(n, step, changed):
    with pytest.raises(ValueError):
        windows(n, step, changed)
