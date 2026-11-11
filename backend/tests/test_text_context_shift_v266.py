import math

import pytest

from scripts import text_context_shift_v266 as shift


def test_shift_replaces_one_outer_focus_and_excludes_unavailable_centers():
    raw = [0.2, 0.5, 0.8, 0.2]
    assert list(shift.shift_scores(raw)) == pytest.approx([-math.log(4), None, None, math.log(4)])
    assert list(shift.shift_scores(raw[:3])) == [None] * 3


def test_shift_requires_both_flanks_when_they_exist():
    scores = list(shift.shift_scores([0.5, 0.25, 0.75, 0.5, 0.25, 0.75, 0.5]))
    assert scores[3] == pytest.approx(-2 * math.log(3))


def test_point_uses_group_maxima_and_rejects_unmeasured_groups():
    groups = {n: [n, n, n, None] for n in range(32)}
    assert shift.select_point(groups) == (math.nextafter(30.0, math.inf), 1)
    groups[31] = groups[30]
    assert shift.select_point(groups) == (math.nextafter(30.0, math.inf), 0)
    groups[31] = [None] * 4
    with pytest.raises(ValueError, match="measured support"):
        shift.select_point(groups)


def test_missing_shift_does_not_mask_current_and_ranking_uses_joint_support():
    rows = [
        ({0}, {"raw": 0.9, "score": 0.9}, None),
        ({1, 2}, {"raw": 0.2, "score": 0.2}, 0.5),
        ({3}, {"raw": 0.8, "score": 0.8}, 0.1),
    ]
    current, contrast, combined = shift.evidence(rows, 0.75, 0.25)
    assert current.mask == {0, 3}
    assert contrast.mask == {1, 2}
    assert contrast.covered == {1, 2, 3}
    assert combined.mask == combined.covered == {0, 1, 2, 3}
    assert shift.ranks(rows, {0: 1, 1: 1, 2: 0, 3: 0}) == [0.25, 0.75]
    assert shift.evidence(rows[:1], 0.75, 0.25)[1] is None
