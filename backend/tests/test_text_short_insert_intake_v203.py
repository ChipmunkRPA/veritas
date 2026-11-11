import json
import stat

import pytest

from scripts import text_short_insert_intake_v203 as intake


def publisher_row(page=1234, suffix="a"):
    left = " ".join(f"left{suffix}{n}" for n in range(26))
    right = " ".join(f"right{suffix}{n}" for n in range(26))
    old, new = "café words", "clearer e\u0301vidence"
    return {
        "doc_id": str(page),
        "revision_depth": 1,
        "before_revision": f"{left} {old} {right}",
        "after_revision": f"{left} {new} {right}",
        "edit_actions": [
            {
                "type": "R",
                "major_intent": "clarity",
                "before": old,
                "after": new,
                "start_char_pos": len(left) + 1,
                "end_char_pos": len(left) + 1 + len(old),
            }
        ],
    }


def official(folder, row, *, after_parent=11, user=9, tags=()):
    page = int(row["doc_id"])
    rights = {
        "page_id": page,
        "before_revision": 11,
        "after_revision": 12,
        "license": intake.LICENSE,
        "license_policy": intake.POLICY,
        "wikinews_staff_or_contributor_prose": True,
        "third_party_material_excluded": True,
        "ordinary_editorial_retouch": True,
        "not_known_bot_or_revert": True,
        "story_component": intake.sha(f"story {page}".encode()),
    }

    def save(filename, key, body):
        data = json.dumps(body).encode()
        (folder / filename).write_bytes(data)
        rights[key] = intake.sha(data)

    save(
        f"history-{page}.json",
        "history_sha256",
        {
            "query": {
                "pages": [
                    {
                        "pageid": page,
                        "ns": 0,
                        "revisions": [
                            {
                                "revid": 12,
                                "parentid": after_parent,
                                "timestamp": "2020-01-03T00:00:00Z",
                                "userid": user,
                                "tags": list(tags),
                            },
                            {"revid": 11, "parentid": 10, "timestamp": "2020-01-02T00:00:00Z"},
                        ],
                    }
                ]
            }
        },
    )
    for rev, side, text in ((11, "before", row["before_revision"]), (12, "after", row["after_revision"])):
        save(
            f"render-{rev}.json",
            f"{side}_render_sha256",
            {
                "parse": {
                    "pageid": page,
                    "revid": rev,
                    "text": f"<div><p>{text}</p><blockquote><p>Excluded quote</p></blockquote></div>",
                }
            },
        )
    return rights


def test_unicode_retouch_and_word_envelope_round_trip():
    row = publisher_row()
    [case] = list(intake.proposed_retouched_excerpts(row))
    for candidate, original in (
        (case.original, row["before_revision"]),
        (case.revised, row["after_revision"]),
    ):
        start, end = candidate.target
        retouch_start, retouch_end = candidate.retouch
        assert start < retouch_start < retouch_end < end
        assert (
            original[candidate.article_offset : candidate.article_offset + len(candidate.text)]
            == candidate.text
        )
        assert candidate.target_words == len(candidate.text[start:end].split()) == 8
        assert len(candidate.text[:start].split()) >= 12
        assert len(candidate.text[end:].split()) >= 12
    a, b = case.revised.retouch
    assert case.revised.text[a:b] == "clearer e\u0301vidence"
    row["edit_actions"][0]["start_char_pos"] += 1
    assert list(intake.proposed_retouched_excerpts(row)) == []


def test_official_prose_excludes_quotes_citations_and_duplicate_matches():
    assert (
        intake.visible_prose("<p>é &amp; <b>edit</b><sup>[1]</sup></p><figure><p>ignore</p></figure>")
        == "é & edit"
    )
    assert intake.unique_prose_match("é  word", "start é word end")
    assert not intake.unique_prose_match("é word", "é word é word")
    assert not intake.unique_prose_match("é word", "prefixé words")


@pytest.mark.parametrize("invalid", [{"after_parent": 8}, {"user": 0}, {"tags": ["mw-rollback"]}])
def test_official_archive_requires_adjacent_registered_nonrevert_revisions(tmp_path, invalid):
    audit = official(tmp_path, publisher_row(), **invalid)
    with pytest.raises(ValueError, match="Adjacent visible"):
        intake.rights_lineage(tmp_path, audit)


def test_admission_requires_both_exact_prose_witnesses_and_checks_prior_overlap(tmp_path):
    row = publisher_row()
    candidates = intake.all_candidates([row])
    review = official(tmp_path, row)
    accepted, rejected = intake.admitted_sources(candidates, tmp_path, [review], set())
    assert len(accepted) == 1 and not rejected
    exposed = intake.fingerprints(accepted[1234][1].revised.text)
    accepted, rejected = intake.admitted_sources(candidates, tmp_path, [review], exposed)
    assert not accepted and rejected == {"previous_project_exposure": 1}

    review = official(tmp_path, {**row, "after_revision": "A different but valid rendered article."})
    accepted, rejected = intake.admitted_sources(candidates, tmp_path, [review], set())
    assert not accepted and rejected == {"revision_prose_not_bound": 1}
    (tmp_path / "render-12.json").write_text("{}")
    with pytest.raises(ValueError, match="privately audited copy"):
        intake.admitted_sources(candidates, tmp_path, [review], set())


def test_selection_is_deterministic_and_keeps_stories_disjoint(tmp_path, monkeypatch):
    monkeypatch.setattr(intake, "POINT_DYADS", 1)
    monkeypatch.setattr(intake, "EVALUATION_DYADS", 1)
    rows = [publisher_row(1234 + n, suffix=str(n)) for n in range(5)]
    records = intake.all_candidates(rows)
    reviewed = {}
    for i, (page, [candidate]) in enumerate(records.items()):
        review = official(tmp_path, rows[i])
        review["private_notes"] = "discarded"
        if i == 4:
            review["story_component"] = reviewed[1234][2]["story_component"]
        reviewed[page] = ((), candidate, review, intake.fingerprints(candidate.original.text))
    dyads, ready = intake.select(reviewed)
    assert ready
    assert intake.select(dict(reversed(list(reviewed.items())))) == (dyads, ready)
    assert [d["role"] for d in dyads] == ["point", "evaluation"]
    assert len({a["story_component"] for d in dyads for a in d["lineage"]}) == 4
    assert "private_notes" not in dyads[0]["lineage"][0]
    assert intake.select(dict(list(reviewed.items())[:3]))[1] is False


def test_private_output_is_exclusive_and_owner_only(tmp_path):
    output = tmp_path / "test.json"
    payload = [{"role": "point"}]
    assert intake.save_private(output, payload) == intake.sha(output.read_bytes())
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        intake.save_private(output, payload)
    with pytest.raises(ValueError, match="ignored evaluation"):
        intake.save_private(intake.ROOT / "docs" / "invalid-test-destination.json", payload)
