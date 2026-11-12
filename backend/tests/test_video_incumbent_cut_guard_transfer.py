# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction

import numpy as np

from scripts import video_incumbent_cut_guard_transfer as study


def scenes():
    a = np.zeros((2, 2, 3), dtype=np.float32), np.array([1., 0., 0.], dtype=np.float32)
    b = np.full((2, 2, 3), 128, dtype=np.float32), np.array([0., 1., 0.], dtype=np.float32)
    c = np.full((2, 2, 3), 255, dtype=np.float32), np.array([0., 0., 1.], dtype=np.float32)
    return a, b, c


def data(left=14, right=28, *, returns=True):
    times = [Fraction(i, 25) for i in range(50)]
    a, b, c = scenes()
    region = {i: a if i < left else b if i < right else a if returns else c for i in range(len(times))}
    incumbent = {"start": str(times[15]), "end": str(times[27]), "score": .4}
    return times, region, incumbent


def test_paired_scene_switch_reports_exact_cut_boundaries():
    times, region, incumbent = data()
    result = study.cut_guard(incumbent, times, region)
    assert result["indices"] == [14, 28]
    assert result["span"] == ["14/25", "28/25"] and result["iou"] == "6/7"
    assert study.is_cut(result["entry"]) and study.is_cut(result["exit"])
    assert study.is_same_scene(result["flank"]) and study.is_same_scene(result["middle"])


def test_unpaired_cut_and_different_third_scene_retain_incumbent():
    times, region, incumbent = data(right=50)
    assert study.cut_guard(incumbent, times, region) is None
    times, region, incumbent = data(returns=False)
    assert study.cut_guard(incumbent, times, region) is None


def test_picture_support_and_exact_clock_proximity_are_required():
    times, region, incumbent = data()
    del region[27]
    assert study.cut_guard(incumbent, times, region) is None
    times, region, incumbent = data(left=6)
    assert study.cut_guard(incumbent, times, region) is None


def test_supplement_does_not_make_an_unflagged_case_positive():
    times, region, incumbent = data()
    assert study.cut_guard(None, times, region) is None
    assert study.cut_guard({**incumbent, "score": study.trial.prior.POINT}, times, region) is None
