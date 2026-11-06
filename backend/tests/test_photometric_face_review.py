from __future__ import annotations

import pytest

from scripts.review_photometric_face_sequence import cases, choose, metrics, overlap, summarize


def example(ident, label, *, current=None, windows=None, status="ok", family=None):
    return {"id": ident, "group": ident.split("/")[0], "label": label, "status": "ready",
            "family": family or ("inswapper" if label else "original_videos"),
            "edit_s": [2., 3.] if label else None, "current_range": current,
            "baseline_scored": True, "baseline_status": "ok",
            "proposal": {"status": status, "raw_windows": windows or [], "normalized_windows": windows or []}}


def window(score, start=2., end=3.):
    return {"response": score, "start_s": start, "end_s": end}


def test_missing_or_failed_candidate_preserves_the_current_review_and_the_denominator():
    rows = [example("a/real", 0, current=[1, 2], status="failed"),
            example("a/fake", 1, status="not_applicable"), example("b/fake", 1, current=[2, 3])]
    result = metrics(cases(rows, "normalized", .01))
    assert result["planned"] == 3 and result["edits"] == 2
    assert result["localized_edits"] == 1 and result["missed_localizations"] == 1
    assert result["original_flags"] == result["candidate_failures"] == 1
    assert result["candidate_status_counts"]["not_applicable"] == 1
    assert result["mean_edit_iou"] == .5


def test_overlap_uses_source_time_and_counts_no_range_as_zero():
    assert overlap([2., 3.], [2., 3.]) == 1
    assert overlap([1.5, 3.], [2., 3.]) == pytest.approx(2/3)
    assert overlap(None, [2, 3]) == overlap([1, 1], [1, 1]) == 0


def test_failed_candidate_cannot_replace_current_evidence_with_a_partial_high_response():
    row = example("a/fake", 1, current=[0, 1], windows=[window(.9)], status="failed")
    value = cases([row], "normalized", .1)[0]
    assert value["range_s"] == [0, 1] and not value["used_candidate"]


def test_development_can_select_disabled_when_every_extra_window_is_a_false_alarm():
    rows = [example(str(i), 0, windows=[window(.2)]) for i in range(4)]
    rows.append(example("fake", 1, windows=[window(.1)]))
    assert choose(rows, "normalized") is None


def test_the_selected_operating_point_uses_development_real_and_edit_counts():
    rows = [example("r1", 0, windows=[window(.08)]), example("r2", 0, windows=[window(.07)]),
            example("f1", 1, windows=[window(.12)]), example("f2", 1, windows=[window(.10)])]
    assert choose(rows, "normalized") == .10
    dev = summarize(rows, {"raw": .10, "normalized": .10}, confirmation=False)
    assert dev["normalized"]["meets_count_tradeoff"]
    assert not dev["normalized"]["qualifies_for_scoped_local_adoption"]
    fresh = summarize(rows, {"raw": .10, "normalized": .10}, confirmation=True)
    assert fresh["normalized"]["qualifies_for_scoped_local_adoption"]


def test_paired_uncertainty_resamples_source_groups_rather_than_two_swaps_as_independent():
    rows = [example(f"{group}/{family}", 1, windows=[window(.1)])
            for group in ("a", "b") for family in ("swap-one", "swap-two")]
    rows.extend([example("a/real", 0), example("b/real", 0)])
    result = summarize(rows, {"raw": .1, "normalized": .1}, confirmation=True)
    assert result["normalized"]["paired_iou"]["source_groups"] == 2
    assert result["normalized"]["localized_edits"] == 4
    assert result["normalized"]["paired_iou"]["source_group_bootstrap_95_percent"] == [1., 1.]
