from __future__ import annotations

import pytest

from scripts import audio_native_stereo_holdout_v284 as study


def case(name, lane="genuine", condition="traffic_forced"):
    return {"case": name, "lane": lane, "condition": condition, "source_group": f"group-{name}",
            "mixed_pcm_sha256": "mono", "channel_pcm_sha256": ["left", "right"]}


def activity(name, mono, selected=None, selected_admitted=False):
    return {"case": name, "selected": selected, "swap_invariant": True,
            "admitted": {"mono": mono, "ch0": selected == 0 and selected_admitted, "ch1": selected == 1 and selected_admitted}}


def receipt(variant, direction):
    return {"case": variant["case"], "decision": direction, "raw": 0.5, "reference_estimate_ai": 0.5,
            "stability_probe": "not_required_for_decision", "stability_log_odds_change": None, "classifier_forwards": 1}


def test_views_require_actual_admission_and_exact_selected_channel():
    cases = [case("a"), case("b"), case("c"), case("d")]
    observed = [activity("a", False, 1, True), activity("b", True, 0, True),
                activity("c", False, 0, False), activity("d", True)]
    views = study.views_for(cases, observed)
    assert [(row["case"], row["pcm_sha256"]) for row in views] == [
        ("a-selected", "right"), ("b-mono", "mono"), ("b-selected", "left"), ("d-mono", "mono"),
    ]
    with pytest.raises(ValueError, match="Exact replayed"):
        study.views_for(cases, observed[::-1])
    with pytest.raises(ValueError):
        study.views_for(cases, observed[:-1])


def test_full_policy_retains_mono_directions_and_unassessed_coverage():
    cases = [case("a", "genuine"), case("b", "expressive"), case("c", "clone", "kitchen_equal_gain"),
             case("d", "genuine"), case("e", "stock", "matched_clean"), case("f", "stock")]
    observed = [activity("a", True, 0, True), activity("b", False, 1, True), activity("c", True, 1, True),
                activity("d", False, 0, True), activity("e", False), activity("f", False, 0, True)]
    variants = study.views_for(cases, observed)
    directions = ("original", "ai", "ai", "inconclusive", "original", "ai", "original")
    rows = [receipt(variant, direction) for variant, direction in zip(variants, directions, strict=True)]
    result = study.decision_report(cases, variants, rows)
    overall = result["all"]["all"]
    assert overall["mono"]["not_assessed"] == 4
    assert overall["ai_only_rescue"]["not_assessed"] == 2
    assert overall["ai_only_rescue"]["correct_ai"] == overall["ai_only_rescue"]["wrong_ai"] == 1
    assert overall["ai_only_rescue"]["wrong_original"] == 0
    assert overall["channel_priority"]["wrong_original"] == 2
    assert overall["groups_with_added_correct_ai"] == overall["groups_with_added_false_ai"] == 1
    assert result["treatment"]["equal_gain"]["ai_only_rescue"]["inconclusive"] == 1
    assert all(overall[route]["planned"] == 6 for route in study.ROUTES)
    with pytest.raises(ValueError, match="Every frozen rescue"):
        study.decision_report(cases, variants, rows[:-1])
