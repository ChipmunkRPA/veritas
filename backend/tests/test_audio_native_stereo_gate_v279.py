from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from scripts import audio_native_stereo_gate_v279 as study


def authored_channels():
    pcm = np.empty((study.SAMPLES, 2), dtype="<i2")
    pcm[:] = [40, 1000]
    pcm[:study.SAMPLES // 2] = [8000, 2000]
    return pcm


def test_acoustic_selection_is_physical_channel_swap_invariant():
    pcm = authored_channels()
    spans = {"mono": [], "ch0": [[0, 24_000]], "ch1": []}
    forward = study.observation(pcm, spans)
    reverse = study.observation(pcm[:, ::-1], {"mono": [], "ch0": spans["ch1"], "ch1": spans["ch0"]})
    assert forward["selected"] == 0 and reverse["selected"] == 1
    assert forward["contrast_db"] == reverse["contrast_db"][::-1]
    assert forward["admitted"] == {"mono": False, "ch0": True, "ch1": False}
    assert forward["swap_invariant"] and reverse["swap_invariant"]


def test_unsupported_quality_does_not_select():
    pcm = authored_channels()
    spans = [[[0, 24_000]], []]
    assert study.choose_channel(pcm, [[], []])["reason"] == "insufficient_active_or_quiet_support"
    assert study.choose_channel(pcm, [[[0, 47_999]], []])["selected"] is None
    duplicate = np.repeat(pcm[:, :1], 2, axis=1)
    assert study.choose_channel(duplicate, spans)["reason"] == "contrast_gap_below_minimum"
    pcm[25_000, 0] = -32768
    assert study.choose_channel(pcm, spans)["reason"] == "rail_clipping"


def test_case_binding_rejects_missing_and_substituted_rows():
    entry = {"source": {"slot": "example", "group_id": "group", "lane": "genuine"},
             "tape_start_sample": 0, "forced_quieter_channel": 1}
    base = {"case": "case-000-00", "condition": "matched_clean", "status": "prepared", "source_slot": "example",
            "source_group": "group", "lane": "genuine", "source_timeline": study.TIMELINE,
            "tape_start_sample": 0, "forced_quieter_channel": 1}
    assert study.bind_cases([entry], ["matched_clean"], [base]) == [base]
    with pytest.raises(ValueError, match="Every v277"):
        study.bind_cases([entry], ["matched_clean", "river_forced"], [base])
    substituted = deepcopy(base)
    substituted["forced_quieter_channel"] = 0
    with pytest.raises(ValueError, match="frozen development case binding"):
        study.bind_cases([entry], ["matched_clean"], [substituted])


def test_forced_assignment_is_accounted_after_selection():
    cases = [{"case": "a", "condition": "river_forced", "lane": "genuine", "forced_quieter_channel": 0},
             {"case": "b", "condition": "river_forced", "lane": "clone", "forced_quieter_channel": 1}]
    first = {"case": "a", "reason": "selected", "swap_invariant": True, "selected": 0,
             "admitted": {"mono": False, "ch0": True, "ch1": False}}
    second = {"case": "b", "reason": "contrast_gap_below_minimum", "swap_invariant": True, "selected": None,
              "admitted": {"mono": True, "ch0": False, "ch1": True}}
    totals = study.summarize(cases, [first, second])["overall"]
    assert totals["mono_admitted"] == totals["selected_admitted"] == 1
    assert totals["mono_or_selected_admitted"] == 2
    assert totals["forced_selection_available"] == totals["forced_assignment_matches"] == 1
    first["selected"] = 1
    assert study.summarize(cases, [first, second])["overall"]["forced_assignment_matches"] == 0
