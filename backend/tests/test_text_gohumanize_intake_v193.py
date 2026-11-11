"""Focused data-only checks for the writing source and exact control offsets."""

import json

import pytest

from scripts import text_gohumanize_intake_v193 as intake

META = {
    "pair": "pair-00",
    "article_group": "article-00",
    "agency": "Federal Reserve",
    "generator": "gpt-4o-mini",
}


def article(prefix):
    return "  \n\n".join(
        " ".join(
            [
                prefix + str(index),
                "cafés",
                "record",
                "rainfall",
                "and",
                "temperatures",
                "for",
                "the",
                "whole",
                "region",
            ]
        )
        + "."
        for index in range(7)
    )


def test_construct_keeps_exact_base_text_and_matches_same_host_range():
    human, donor = article("genuine"), article("rewrite")
    original = {"output": human, "input": donor}
    rows = {row["arm"]: row for row in intake.construct(original, META)}
    assert rows == {row["arm"]: row for row in intake.construct(original, META)}
    assert tuple(rows) == intake.ARMS
    assert rows["human"]["text"] == human
    assert rows["human"]["ai_spans"] == []
    assert rows["full_ai"]["text"] == donor
    assert rows["full_ai"]["ai_spans"] == [[0, len(donor)]]
    for band, (low, high) in intake.BANDS.items():
        mixed, formatted = rows[f"{band}_ai"], rows[f"{band}_format"]
        start, end = mixed["host_range"]
        ds, de = mixed["donor_range"]
        ms, me = mixed["ai_spans"][0]
        fs, fe = formatted["format_range"]
        assert 0 < start == ms < end < len(human)
        assert 0 < ds < de < len(donor)
        assert formatted["host_range"] == mixed["host_range"]
        assert start < fs < fe < end
        assert mixed["text"][ms:me] == donor[ds:de]
        assert low <= len(intake.WORD.findall(human[start:end])) <= high
        assert low <= len(intake.WORD.findall(donor[ds:de])) <= high
        assert mixed["text"][:ms] == human[:start]
        assert mixed["text"][me:] == human[end:]
        assert human[fs:fe] == " "
        assert formatted["text"][:fs] + " " + formatted["text"][fe:] == human
        assert formatted["text"][fs:fe] == "\n"
        assert formatted["ai_spans"] == []
        assert all(row["text_sha256"] == intake.digest(row["text"]) for row in (mixed, formatted))


def test_missing_central_sentence_pair_stays_unavailable_on_both_arms():
    rows = list(intake.construct({"output": "Only one genuine sentence.", "input": article("rewrite")}, META))
    assert [row["status"] for row in rows] == ["constructed"] * 2 + ["unavailable"] * 4
    for row in rows[2:]:
        assert row["reason"] == "no_eligible_sentence_pair"
        assert all(
            row[key] is None for key in ("text", "text_sha256", "ai_spans", "host_range", "donor_range")
        )


def test_sentence_candidates_protect_abbreviations_and_use_completed_surrounds():
    source = "Dr. Lin reviewed the first record. “This field survey assessed the long northern river basin.” Next steps are pending. Fragment"
    parts = intake.sentences(source)
    assert [source[s.start : s.end] for s in parts] == [
        "Dr. Lin reviewed the first record.",
        "“This field survey assessed the long northern river basin.”",
        "Next steps are pending.",
    ]
    assert [(p.start, p.end) for p in intake.candidates(source, (8, 16))] == [(parts[1].start, parts[1].end)]


def test_rights_join_rejects_wrong_url_and_nonqualified_status():
    article_url = "https://www.federalreserve.gov/article.htm"
    row = {"id": "fixture", "url": article_url, "title": article_url, "author": "Federal Reserve"}
    audit = {
        "id": "fixture",
        "source_url": article_url,
        "agency": "Federal Reserve Board",
        "rights_assessment": intake.RIGHTS,
    }
    intake.check_rights(row, audit)
    for change in (
        {"id": "other"},
        {"source_url": article_url + "#part"},
        {"rights_assessment": "Reuse policy only"},
    ):
        with pytest.raises(SystemExit, match="Exact ID"):
            intake.check_rights(row, audit | change)


def test_exposure_audit_reports_counts_without_copying_phrases(tmp_path, monkeypatch):
    name = "writing_v7_prepared_exposures"
    previous = "The same old sentence was used for a bounded archival exact phrase check."
    data = json.dumps({"cases": [{"text": previous}]}).encode("utf-8")
    (tmp_path / "inputs.json").write_bytes(data)
    monkeypatch.setattr(intake, "BASE", tmp_path)
    monkeypatch.setattr(intake, "HISTORY", {name: ("inputs.json", "receipt", "sha")})
    result = intake.exposure_audit(
        [{"output": previous.upper(), "input": "Intro. " + previous + " Outro."}],
        {name: {"status": "available", "committed_input_sha256": intake.digest(data)}},
    )[name]
    assert result["exact_normalized_document_matches"] == {"human": 1}
    assert result["at_least_one_normalized_12_word_match"] == {"full_ai": 1, "human": 1}
    assert result["pairs_with_any_match"] == 1
    assert previous not in str(result)
