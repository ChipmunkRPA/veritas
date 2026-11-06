import json

import pytest

from scripts.audio_confirmation_v7 import CONDITIONS
from scripts.evaluate_audio_multisource_v8 import summarize
from scripts.run_audio_multisource_v8 import open_journal, validate_variants


def variants():
    return {
        "panel": "development",
        "variants": [
            {"id": f"{role}:{i}", "role": role, "background_role": None}
            for role, count in (("fit", 1971), ("cal", 891), ("development", 1422))
            for i in range(count)
        ],
    }


def test_development_runner_refuses_fresh_inputs():
    body = variants()
    body["variants"][0]["role"] = "confirmation"
    with pytest.raises(ValueError, match="fresh confirmation"):
        validate_variants(body)


def test_input_count_and_role_cannot_be_silently_changed():
    body = variants()
    body["variants"].pop()
    with pytest.raises(ValueError, match="Every frozen"):
        validate_variants(body)


def test_background_cannot_cross_a_valid_role_assignment():
    body = variants()
    body["variants"][0]["background_role"] = "cal"
    with pytest.raises(ValueError, match="Backgrounds cannot cross"):
        validate_variants(body)


def test_resume_requires_exact_header_and_input_prefix(tmp_path):
    p = tmp_path / "responses.jsonl"
    header = {"source": "fixed"}
    assert open_journal(p, header, [{"id": "a"}]) == []
    with p.open("a") as stream:
        stream.write(json.dumps({"id": "b"}) + "\n")
    with pytest.raises(ValueError, match="non-prefix outcomes"):
        open_journal(p, header, [{"id": "a"}])


def outcomes():
    return [
        {
            "id": f"{label}:{n}:{condition}",
            "base_id": f"{label}:{n}",
            "group": f"{label}:{n}",
            "label": label,
            "generator": "fixture-ai" if label else "human",
            "dataset": "fixture",
            "condition": condition,
            "original_p": 0.9 if label else 0.1,
            "candidate_p": 0.9 if label else 0.1,
            "original_decision": "ai" if label else "original",
            "candidate_decision": "inconclusive",
            "status": "ok",
            "probe": {"status": "not_required"},
            "background_id": None,
        }
        for label in (0, 1)
        for n in range(3)
        for condition in CONDITIONS
    ]


def test_more_abstention_is_not_counted_as_an_improvement():
    result = summarize(outcomes(), 6, draws=4)
    assert result["gates"]["complete_cases"]
    assert not result["gates"]["clean_retention"]
    assert not result["gates"]["strong_condition_usefulness"]
    assert not result["advance_to_fresh_confirmation"]
    assert result["groups"]["all"]["candidate"]["inconclusive_rate"] == 1
    assert result["groups"]["all"]["candidate"]["ai_without_ai_conclusion"] == 27


def test_failures_remain_in_total_and_block_advancement():
    rows = outcomes()
    rows[0].update(status="failed", original_p=None, candidate_p=None)
    rows[0]["probe"]["status"] = "failed"
    result = summarize(rows, 6, draws=4)
    assert result["complete_inputs"] == 54
    assert not result["gates"]["no_extraction_failures"]
    assert not result["gates"]["no_probe_failures"]
    assert result["groups"]["all"]["candidate"]["real_inputs"] == 27
