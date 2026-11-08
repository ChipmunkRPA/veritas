"""Known sample truth, matched backgrounds and honest repeated-source denominators."""

import numpy as np
import pytest

from scripts import run_public_audioseal_speech_v2 as study


def test_partial_mark_leaves_every_other_source_sample_identical_with_matched_backgrounds():
    u = np.sin(np.arange(128000) / 10).astype(np.float32)
    marked = u + np.float32(0.03)
    backgrounds = {c: np.full_like(u, 3.0) for c in study.CONDITIONS if c not in ["clean", "mp3_64k"]}
    outputs, gain = study.mixtures(u, marked, backgrounds)
    assert gain < 1 and len(outputs) == 15
    for condition in study.CONDITIONS:
        if condition == "mp3_64k":
            continue
        before = outputs[("unmarked", condition)]
        partial = outputs[("partial", condition)]
        assert np.array_equal(before[:32000], partial[:32000])
        assert np.array_equal(before[96000:], partial[96000:])
        assert np.max(np.abs(outputs[("marked", condition)])) <= 0.800001


@pytest.mark.parametrize(
    "state,intervals,iou,outside",
    [
        ("partial", [{"start_sample": 32000, "end_sample": 96000}], 1.0, 0.0),
        ("partial", [], 0.0, 0.0),
        ("partial", [{"start_sample": 0, "end_sample": 128000}], 0.5, 4.0),
        ("unmarked", [{"start_sample": 0, "end_sample": 4000}], None, 0.25),
    ],
)
def test_interval_quality_uses_deliberate_mark_truth_not_ai_authorship(state, intervals, iou, outside):
    r = study.metrics(intervals, state)
    assert r["iou"] == iou and r["outside_mark_s"] == outside


def test_no_selected_controls_do_not_establish_a_zero_population_false_alarm_rate():
    lo, hi = study.wilson(0, 32)
    assert lo == 0 and 0.10 < hi < 0.11
    rows = [
        {"state": "unmarked", "condition": "clean", "metrics": {"any_selected": False, "outside_mark_s": 0}}
        for _ in range(32)
    ]
    result = study.aggregate(rows)[0]
    assert result["n"] == 32 and result["selected_cases"] == 0
    assert result["selected_rate_wilson95"] == [lo, hi]
    assert study.POLICY["cases"] == 576 and study.POLICY["speakers"] == 32
    assert study.POLICY["serving_admission"] is study.POLICY["confidence_admission"] is False
