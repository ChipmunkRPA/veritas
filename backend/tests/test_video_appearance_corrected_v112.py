from copy import deepcopy

import pytest

from scripts import video_appearance_corrected_v112 as corrected
from scripts import video_appearance_native_v108 as native
from scripts import video_appearance_readout_v111 as readout


def record():
    return {
        "threshold": 0.24379932164053977,
        "common_genuine_groups": 39,
        "budget": 3,
        "groups_flagged": 3,
        "comparison": "strict_greater_than",
    }


def test_versioned_saved_point_becomes_exact_same_numeric_comparison():
    p = record()
    value = corrected.point_value(p)
    assert value == p["threshold"]
    assert (value < 0.25) is True and (value > value) is False
    assert corrected.point_value(p) == 0.24379932164053977


@pytest.mark.parametrize("problem", ["scalar", "missing", "nan", "bool", "comparison", "support"])
def test_bad_record_fails_before_any_model(problem):
    p = record()
    if problem == "scalar":
        p = p["threshold"]
    elif problem == "missing":
        del p["comparison"]
    elif problem == "nan":
        p["threshold"] = float("nan")
    elif problem == "bool":
        p["threshold"] = True
    elif problem == "comparison":
        p["comparison"] = "greater_or_equal"
    else:
        p["groups_flagged"] = 4
    with pytest.raises(ValueError):
        corrected.point_value(p)


def test_derived_plan_changes_only_point_representation_and_not_source_or_roles():
    p = {
        "candidate_point": record(),
        "cases": [{"id": "authored", "role": "selection"}],
        "batch_group_indices": [[1, 2, 3]],
    }
    before = deepcopy(p)
    derived = corrected.canonical_plan(p)
    assert p == before and derived["candidate_point"] == before["candidate_point"]["threshold"]
    derived["candidate_point"] = before["candidate_point"]
    assert derived == before


def test_frozen_extractor_code_and_original_globals_are_unchanged():
    prep, trial = native.PREP, native.TRIAL
    fn = corrected.extractor()
    assert fn.__code__ is native.extract.__code__
    assert fn.__globals__ is not native.extract.__globals__
    assert fn.__globals__["PREP"] == corrected.PREP and fn.__globals__["TRIAL"] == corrected.TRIAL
    assert prep == native.PREP and trial == native.TRIAL


def test_readout_uses_same_math_with_isolated_context_paths():
    old = readout.NATIVE
    context = corrected.readout_context()
    assert context["evaluate"].__code__ is readout.evaluate.__code__
    assert context["identity"].__globals__ is context
    assert context["NATIVE"] == corrected.TRIAL and old == readout.NATIVE
    assert context["reference_probability"] is readout.reference_probability


def test_routing_changes_only_worker_module_and_does_not_mutate_argv(monkeypatch):
    calls = []
    monkeypatch.setattr(corrected.subprocess, "Popen", lambda argv, **kwargs: calls.append((argv, kwargs)))
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract", "output", "0"]
    before = args[:]
    corrected.routed_popen(args, start_new_session=True)
    assert args == before and calls[0][0][3] == "scripts.video_appearance_corrected_v112"
    expected = before[:]
    expected[3] = "scripts.video_appearance_corrected_v112"
    assert calls[0] == (expected, {"start_new_session": True})
    with pytest.raises(ValueError):
        corrected.routed_popen(["python", "-B", "-m", "other", "extract"])
