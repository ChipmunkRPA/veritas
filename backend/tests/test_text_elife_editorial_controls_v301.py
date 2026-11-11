import json
from collections import Counter

from scripts import text_elife_editorial_controls_v301 as study


def original(paragraph, donor=None):
    projection = [
        "Introduction with diacritics from São Tomé.",
        "First unchanged neighbor.",
        paragraph,
        "Second unchanged neighbor.",
        "An unchanged conclusion.",
    ]
    text = "\n\n".join(projection)
    edges = study.paragraph_offsets(text)
    row = {
        "group": "elife-example",
        "role": "confirmation",
        "text": text,
        "text_sha256": study.sha(text.encode()),
        "body_slots": [0, len(projection)],
        "edit_target_codepoints": edges[2],
        "paragraph_codepoints": edges,
    }
    return row, projection + ([donor] if donor else [])


def test_fixed_roster_and_split_precede_body_admission():
    lock = study.protocol()
    assert len(lock["works"]) == 26
    assert Counter(row["role"] for row in lock["works"]) == {"development": 13, "confirmation": 13}
    assert Counter(row["date"][:7] for row in lock["works"]) == {
        "2022-01": 4,
        "2022-02": 8,
        "2022-03": 7,
        "2022-04": 7,
    }
    assert study.partitions(study.WORKS) == study.partitions(reversed(study.WORKS))
    assert len(lock["known_exposure_receipts_sha256"]) == 5


def test_unicode_offsets_and_unchanged_flanks_for_both_ordinary_controls():
    paragraph = (
        "Méthodes vary however widely they are used. However, different conditions usually need different "
        "tools in order to measure the same features across samples. For example, this one includes café."
    )
    donor = (
        "In a separate example from the same article, the discussion focuses on another experiment "
        "with enough words to preserve the overall size of the paragraph after moving it here."
    )
    row, projection = original(paragraph, donor)
    unchanged, edited, moved = study.controls(row, projection)
    start, end = row["edit_target_codepoints"]
    assert unchanged["text"] == row["text"]
    assert unchanged["ordinary_spans"] == []
    for case in (edited, moved):
        _, new_end = case["paragraph_codepoints"][2]
        assert case["status"] == "constructed"
        assert case["text"][:start] == row["text"][:start]
        assert case["text"][new_end:] == row["text"][end:]
        assert case["ai_spans"] == []
        assert case["role"] == "confirmation"
    assert [edited["text"][a:b] for a, b in edited["ordinary_spans"]] == [
        "Nevertheless",
        "to",
        "For instance",
    ]
    assert "however widely" in edited["text"]
    assert moved["donor_body_slot"] == 5
    assert moved["ordinary_spans"] == [[start, start + len(donor)]]
    assert moved["text"][start : start + len(donor)] == donor


def test_no_eligible_copyedit_or_donor_stays_unavailable():
    row, projection = original("This paragraph has no matching editorial phrases.")
    cases = study.controls(row, projection)
    assert [row["reason"] for row in cases] == [None, "no_fixed_copyedit", "no_length_matched_source_donor"]
    assert all(row["text"] is row["ai_spans"] is None for row in cases[1:])
    assert study.copyedit("In addition to this, however large a trial, approximately similar.") == (
        "In addition to this, however large a trial, approximately similar.",
        [],
    )


def test_missing_articles_remain_in_every_denominator(tmp_path, monkeypatch):
    lock = study.protocol()
    monkeypatch.setattr(study, "historical", lambda *args: (set(), set(), ()))
    monkeypatch.setattr(study.july, "exposure_packets", lambda *args: ())
    originals, controls, receipt = study.run(lock, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    assert originals == controls == b""
    assert receipt["status"] == "complete_below_target"
    assert receipt["source_outcomes_by_role"] == {role: {"missing_direct_xml": 13} for role in study.ROLES}
    assert receipt["control_outcomes_by_role"] == {
        role: {arm: {"source_not_admitted": 13} for arm in study.ARMS} for role in study.ROLES
    }
    assert receipt["generator_calls"] == receipt["detector_calls"] == 0
    assert '"text"' not in json.dumps(receipt)


def test_intake_accepts_only_licensed_publisher_prose_and_keeps_body_private(tmp_path, monkeypatch):
    lock = study.protocol()
    body = "".join(
        "<p>However, " + " ".join(f"testword{index}x{word}" for word in range(46)) + ".</p>"
        for index in range(6)
    )
    for work, license_kind in zip(lock["works"][:2], ("by", "by-nc"), strict=True):
        source = work["source_id"]
        year, month, day = work["date"].split("-")
        authors = "".join(
            "<contrib contrib-type='author'><name>"
            f"<given-names>{given}</given-names><surname>{family}</surname></name></contrib>"
            for given, family in (name.rsplit(" ", 1) for name in work["authors"])
        )
        xml = (
            "<article article-type='article-commentary' xmlns:xlink='http://www.w3.org/1999/xlink'>"
            "<front><article-meta><article-categories><subj-group subj-group-type='display-channel'>"
            "<subject>Insight</subject></subj-group></article-categories>"
            f"<article-id pub-id-type='doi'>10.7554/eLife.{source}</article-id>"
            f"<volume>11</volume><elocation-id>e{source}</elocation-id>"
            f"<pub-date><year>{year}</year><month>{month}</month><day>{day}</day></pub-date>"
            f"<contrib-group>{authors}</contrib-group><title-group><article-title>Fixture title</article-title>"
            "</title-group><permissions>"
            f"<license xlink:href='https://creativecommons.org/licenses/{license_kind}/4.0/'>"
            f"<license-p>Fixture license</license-p></license></permissions></article-meta></front>"
            f"<body>{body}</body></article>"
        )
        (tmp_path / study.transfer.filename(source)).write_text(xml)
    monkeypatch.setattr(study, "historical", lambda *args: (set(), set(), ()))
    monkeypatch.setattr(study.july, "exposure_packets", lambda *args: ())
    originals, controls, receipt = study.run(lock, tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)
    assert [row["status"] for row in receipt["works"][:2]] == ["admitted", "license_not_cc_by_4"]
    assert len(originals.splitlines()) == 1
    assert [json.loads(line)["status"] for line in controls.splitlines()] == ["constructed"] * 3
    assert "testword" not in json.dumps(receipt)
