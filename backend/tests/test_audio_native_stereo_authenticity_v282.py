from __future__ import annotations

import pytest

from scripts import audio_native_stereo_authenticity_v282 as study


def entry(name, condition, gain=0.5):
    return {"case": name, "source_slot": "s1", "source_group": "g1", "lane": "genuine",
            "condition": condition, "common_gain": gain}


def gate(name, mono, selected=0):
    return {"case": name, "swap_invariant": True, "selected": selected,
            "admitted": {"mono": mono, "ch0": selected == 0, "ch1": selected == 1}}


def test_selects_every_rescue_and_admitted_control_with_one_exact_clean(monkeypatch):
    river, washing = study.FORCED
    monkeypatch.setattr(study, "EXPECTED", {("genuine", "rescued", river): 1, ("genuine", "admitted", washing): 1})
    monkeypatch.setattr(study, "EXPECTED_CLEAN", {"genuine": 1})
    cases = [entry("c", study.CLEAN), entry("r", river), entry("w", washing)]
    saved = [gate("c", True, None), gate("r", False), gate("w", True, 1)]
    paired, clean = study.select(cases, saved)
    assert clean == [cases[0]]
    assert [(row[0]["case"], row[2], row[3], row[4]) for row in paired] == [
        ("r", 0, "rescued", False), ("w", 1, "admitted", True),
    ]
    with pytest.raises(ValueError, match="deterministic same-family control"):
        study.select(cases[:2], saved[:2])
    with pytest.raises(ValueError, match="Exact matched clean"):
        study.select(cases[1:], saved[1:])
    with pytest.raises(ValueError, match="common gain"):
        study.select([{**cases[0], "common_gain": 1}, *cases[1:]], saved)


def test_counterfactual_score_cannot_become_gate_respecting_result():
    cases = [
        {"case": "r-mono", "source_case": "r", "clean_case": "c-clean", "view": "mono", "speech_gate_admitted": False},
        {"case": "r-selected", "view": "selected"}, {"case": "c-clean", "view": "clean"},
    ]
    cases = [{"condition": study.FORCED[0], "arm": "genuine", "cohort": "rescued", "group": ("genuine", "g1"), **case}
             for case in cases]
    rows = [{"case": case["case"], "decision": decision, "raw": 0.5, "reference_estimate_ai": 0.5,
             "stability_probe": "not_required_for_decision", "stability_log_odds_change": None, "classifier_forwards": 1}
            for case, decision in zip(cases, ("ai", "inconclusive", "original"), strict=True)]
    result = study.summarize(cases, rows)["paired_direction_counts"][0]
    assert result["mono_gate_respecting"] == {"not_assessed": 1}
    assert result["mono_counterfactual"] == {"ai": 1}
    assert result["selected"] == {"inconclusive": 1}
    assert result["matched_clean"] == {"original": 1}


def test_view_binds_exact_pcm_and_source_case_link():
    case, clean = entry("r", study.FORCED[0]), entry("c", study.CLEAN)
    body = b"\0\0" * study.gate.SAMPLES
    view = study.view(case, clean, "selected", body, study.digest(body), True, "rescued", 1)
    assert view["clean_case"] == "c-clean" and view["selected_channel"] == 1
    with pytest.raises(ValueError, match="Exact v277 view PCM"):
        study.view(case, clean, "selected", body, "0" * 64, True, "rescued", 1)
