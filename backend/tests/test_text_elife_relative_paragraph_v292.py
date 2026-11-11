from fractions import Fraction

import pytest

from scripts import text_elife_relative_paragraph_v292 as study


def test_paragraph_rule_uses_text_only_and_preserves_unicode_coverage():
    text = "aa\n\ncafé\n\nzz"
    scores = {i: 0.75 if 4 <= i < 8 else 0.25 for i, ch in enumerate(text) if not ch.isspace()}
    assert study.leader(text, scores) == (Fraction(1, 2), [4, 8], {4, 5, 6, 7})
    assert study.leader(text, dict.fromkeys(scores, 0.75))[0] == 0
    scores.pop(7)
    assert study.leader(text, scores) is None


def test_development_point_uses_only_originals_and_has_at_most_one_control_above(monkeypatch):
    controls = [
        {
            "case": str(i),
            "role": "development",
            "generator": None,
            "arm": "human",
            "status": "constructed",
            "text": "é\n\nx",
        }
        for i in range(13)
    ]
    responses = [{"case": str(i), "margin": i / 16} for i in range(13)]
    monkeypatch.setattr(study.rank, "current_positions", lambda _, response: {0: 0, 3: response["margin"]})
    assert study.control_point(controls, responses) == (Fraction(11, 16), 13)
    with pytest.raises(ValueError, match="Only the thirteen development originals"):
        study.control_point([*controls[:-1], {**controls[-1], "generator": "qwen"}], responses)


def test_confirmation_cannot_start_without_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "TRIAL", tmp_path)
    monkeypatch.setattr(study, "POINT", tmp_path / "point.json")
    monkeypatch.setattr(study, "verify", lambda: {})
    with pytest.raises(ValueError, match="pre-response development-only point"):
        study.authorized("confirmation", 0)


def test_existing_highlight_prevents_a_new_paragraph_and_equal_margin_abstains(monkeypatch):
    text = "a\n\nb\n\nc"
    scores = {0: 0.25, 3: 0.75, 6: 0.25}
    response = {"primary_current": []}
    monkeypatch.setattr(study.rank, "current_positions", lambda *_: scores)
    monkeypatch.setattr(study.previous.prior, "views", lambda _: ({}, {}))
    baseline = study.layout.Evidence(frozenset(), frozenset(scores))
    monkeypatch.setattr(study.layout, "project", lambda *_: baseline)
    assert study.compare(text, response, 0.5, Fraction(1, 2))[2] is None
    _, augmented, added = study.compare(text, response, 0.5, Fraction(3, 8))
    assert added == [3, 4] and augmented.mask == {3}
    baseline = study.layout.Evidence(frozenset({0}), frozenset(scores))
    assert study.compare(text, response, 0.5, Fraction(3, 8)) == (baseline, baseline, None)
