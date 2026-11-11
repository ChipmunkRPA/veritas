from fractions import Fraction

import pytest

from scripts.av_timing_engineering_fixture import run_fixture

STARTS = [25, 37, 50, 62, 75, 87, 100, 112, 125, 137, 150]


@pytest.fixture(scope="module")
def report():
    return run_fixture()


@pytest.fixture(scope="module")
def by_case(report):
    return {case["name"]: case for case in report["xcorr"]}


@pytest.mark.parametrize(
    ("name", "lag", "reading"), [("same_time", 0, "in_sync"), ("global_late", 8, "constant_offset")]
)
def test_control_is_measured_without_a_local_flag(by_case, name, lag, reading):
    result = by_case[name]
    assert result["full_search_starts"] == STARTS
    assert result["reading"] == reading
    assert result["baseline_ticks"] == lag
    assert result["segments"] == []
    assert all(
        window["selected_lag_ticks"] == window["exact_copy_lag_ticks"] == lag
        and window["peak_correlation"] == pytest.approx(1.0)
        for window in result["windows"]
    )


@pytest.mark.parametrize(
    ("name", "lag", "exact_shifted", "expected_seconds", "expected_overlap"),
    [
        ("local_late", 8, [75, 87], ("137/50", "112/25"), ("70/103", "-17/50", "-8/25")),
        ("local_early", -8, [87, 100], ("81/25", "237/50"), ("75/86", "4/25", "-3/50")),
    ],
)
def test_local_copy_recovers_signed_shift_and_retains_boundary_windows(
    by_case, name, lag, exact_shifted, expected_seconds, expected_overlap
):
    result = by_case[name]
    assert result["edit_ticks_half_open"] == [77, 120]
    assert result["copy_mapping"] == {
        "output_samples_half_open": [77 * 640, 120 * 640],
        "input_samples_half_open": [(77 - lag) * 640, (120 - lag) * 640],
        "sample_rate": 16000,
        "authored_lag_ticks": lag,
    }
    assert result["full_search_starts"] == STARTS
    phases = {"before": [], "after": [], "inside_edit": [], "cross_edit": []}
    shifted = []
    for window in result["windows"]:
        start, end = window["core_ticks"]
        phases[window["full_search_phase"]].append(start)
        assert end - start == 25
        assert window["full_search_ticks"] == [start - 15, end + 15]
        assert len(window["all_lag_correlations"]) == 31
        if window["exact_copy_lag_ticks"] == lag:
            shifted.append(start)
            assert window["selected_lag_ticks"] == lag
            assert window["peak_correlation"] == pytest.approx(1.0)
    assert phases == {
        "before": [25, 37],
        "after": [137, 150],
        "inside_edit": [],
        "cross_edit": [50, 62, 75, 87, 100, 112, 125],
    }
    assert shifted == exact_shifted
    assert result["reading"] == "offset_step"
    assert len(result["segments"]) == 1
    segment = result["segments"][0]
    assert segment["offset_ticks"] == lag
    assert (segment["start_seconds"], segment["end_seconds"]) == expected_seconds
    assert expected_seconds != tuple(str(Fraction(tick, 25)) for tick in result["edit_ticks_half_open"])
    iou, onset, end = expected_overlap
    assert segment["comparison_to_authored_audio_copy"] == {
        "authored_audio_copy_seconds_half_open": ["77/25", "24/5"],
        "flagged_timing_seconds_half_open": list(expected_seconds),
        "temporal_iou": iou,
        "signed_onset_error_seconds": onset,
        "signed_end_error_seconds": end,
    }


def test_flat_local_support_exercises_relative_collapse_rule(by_case):
    result = by_case["local_flat_correlation"]
    assert result["full_search_starts"] == [*STARTS, 162, 175, 187]
    assert result["relative_collapse_threshold"] > 0
    inside = [window for window in result["windows"] if window["wholly_flat_search"]]
    assert [window["core_ticks"][0] for window in inside] == [100, 112]
    assert all(
        window["all_lag_correlations"] == [0.0] * 31 and window["relative_collapse"] is True
        for window in inside
    )
    assert result["reading"] == "confidence_collapse"
    assert [
        (segment["kind"], segment["start_tick"], segment["end_tick"], segment["windows"])
        for segment in result["segments"]
    ] == [("confidence_collapse", 87, 150, 4)]


def test_scheduled_but_missing_speech_is_unassessed(by_case):
    result = by_case["speech_unavailable"]
    assert result["scheduled_starts_before_speech_and_full_search"]
    assert result["full_search_starts"] == result["windows"] == result["segments"] == []
    assert result["reading"] == "unassessed"


def test_articulation_keeps_the_supported_change_and_the_sparse_abstention(report):
    rows = report["articulation"]
    assert rows["same_time"]["state"] == "no_change_in_measured_comparisons"
    assert rows["local_late"]["state"] == "local_timing_change"
    assert rows["wide_gap_local_late"]["state"] == "unassessed"
    assert {row["matched_events"] for row in rows.values()} == {6}
    local = rows["local_late"]["comparisons"][0]
    assert (local["before"], local["during"], local["after"]) == ([0, 1], [2, 3], [4, 5])
    assert local["change_seconds"] == ["6/25", "2/5"]
    assert rows["wide_gap_local_late"]["comparisons"] == []
