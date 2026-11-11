from types import SimpleNamespace

import pytest

from scripts import audio_rescued_speech_shadow_v258 as shadow


def test_select_rescue_only_and_reject_synthetic_or_changed_baseline():
    cases = [
        {"case": f"{condition}-{index}", "arm": "genuine", "status": "prepared", "condition": condition}
        for condition, count in shadow.PLANNED.items()
        for index in range(count)
    ]
    cases.append({"case": "accepted", "arm": "genuine", "status": "prepared", "condition": "matched_clean"})
    observed = [{"case": row["case"], "spans": [[0, 16_000]]} for row in cases[:-1]]
    observed.append({"case": "accepted", "spans": [[0, 24_000]]})
    recorded = {row["case"]: "not_applicable" for row in cases[:-1]} | {"accepted": "ok"}
    assert shadow.select(cases, recorded, observed) == cases[:-1]
    with pytest.raises(ValueError, match="previously skipped"):
        shadow.select(cases, {**recorded, cases[0]["case"]: "ok"}, observed)
    with pytest.raises(ValueError, match="genuine case"):
        shadow.select([{**cases[0], "arm": "alania"}, *cases[1:]], recorded, observed)


def test_score_submits_original_pcm_and_one_whole_window():
    pcm = [0.0] * shadow.gate.SAMPLES
    deadline = object()

    def predict(windows):
        assert len(windows) == 1 and windows[0] is pcm
        return [0.12]

    model = SimpleNamespace(predict=predict)

    def evaluate(raw, original, windows, classifier, passed_deadline):
        assert raw == 0.12 and original is pcm and classifier is model and passed_deadline is deadline
        assert len(windows) == 1 and windows[0][0] == 0 and windows[0][1] is pcm
        return {
            "scope_supported": True,
            "decision": "original",
            "probability_ai": 0.1,
            "probe_status": "computed",
            "stability_log_odds_change": 0.3,
        }

    result = shadow.score(pcm, model, SimpleNamespace(valid=lambda _: True, evaluate=evaluate), deadline)
    assert result["decision"] == "original" and result["stability_probe"] == "computed"


def test_summary_reports_false_ai_at_clip_and_group_levels():
    cases = [{"case": str(i), "group": "private", "condition": "kitchen_0db"} for i in range(2)]
    rows = [
        {
            "case": case["case"],
            "decision": "ai",
            "stability_probe": "not_required_for_decision",
            "classifier_forwards": 1,
            "stability_log_odds_change": None,
            "raw": 0.9,
            "reference_estimate_ai": 0.95,
        }
        for case in cases
    ]
    result = shadow.summarize(cases, rows)
    assert result["directions"]["ai"] == 2 and result["groups_with_false_ai"] == 1
    assert result["vad_forwards"] == 0 and "private" not in repr(result)
    with pytest.raises(ValueError, match="Invalid or incomplete classifier receipt"):
        shadow.summarize(cases, [{**rows[0], "classifier_forwards": 2}, rows[1]])
