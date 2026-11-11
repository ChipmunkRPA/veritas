import pytest

from scripts import text_short_insert_intake_v203 as intake
from scripts import text_wikinews_direct_v209 as direct


def render(revision, document, *, published=True):
    return {
        "parse": {
            "pageid": 24,
            "revid": revision,
            "text": document,
            "categories": [{"category": "Published"}] if published else [],
        }
    }


def example(old="café earlier", new="clearer e\u0301vidence", *, quote=False, published=True):
    left = " ".join(f"left{n}" for n in range(17))
    right = " ".join(f"right{n}" for n in range(17))
    left = f'{left} "quoted"' if quote else left
    before = render(100, f"<p>First paragraph</p><p>{left} {old} {right}</p>")
    after = render(101, f"<p>First paragraph</p><p>{left} {new} {right}</p>", published=published)
    return direct.article_prose(before, 24, 100), direct.article_prose(after, 24, 101)


def test_exact_unicode_offsets_are_in_the_extracted_revision():
    before, after = example()
    [case] = direct.retouches(before, after)
    for key, revision, changed in (
        ("before", before, "café earlier"),
        ("after", after, "clearer e\u0301vidence"),
    ):
        record = case[key]
        start, end = record["article_codepoints"]
        changed_start, changed_end = record["retouch_codepoints"]
        target_start, target_end = record["target_codepoints"]
        assert start > 0
        assert revision[0][start:end] == record["excerpt"]
        assert revision[0][changed_start:changed_end] == changed
        assert start < target_start < changed_start < changed_end < target_end < end
        assert len(revision[0][target_start:target_end].split()) == record["target_words"] == 8
        assert len(revision[0][start:target_start].split()) == len(revision[0][target_end:end].split()) == 12


@pytest.mark.parametrize(
    "fixture", [{"quote": True}, {"published": False}, {"old": "Words 12", "new": "words 13"}]
)
def test_quoted_unpublished_and_nonlexical_retouches_are_excluded(fixture):
    assert direct.retouches(*example(**fixture)) == []


def test_inline_quotes_and_render_identity_are_checked():
    entry = render(100, "<p>Original <q>borrowed text</q></p><p>Author prose<sup>[1]</sup></p>")
    text, safe, published = direct.article_prose(entry, 24, 100)
    assert published
    assert [text[a:b] for a, b in safe] == ["Author prose"]
    with pytest.raises(ValueError, match="selected article"):
        direct.article_prose(entry, 25, 100)


def test_fixed_catalog_is_complete_and_excludes_earlier_sources():
    response = {
        "query": {"searchinfo": {"totalhits": 2}, "search": [{"pageid": 1, "ns": 0}, {"pageid": 2, "ns": 0}]}
    }
    assert direct.month_candidates(response, {2}) == {1}
    with pytest.raises(ValueError, match="complete unique"):
        direct.month_candidates({**response, "continue": {}}, set())


def test_revert_or_bot_tags_and_post_2019_history_are_excluded():
    rows = [
        {
            "revid": rev,
            "parentid": rev - 1,
            "userid": 5,
            "timestamp": f"2019-01-{rev:02d}T00:00:00Z",
            "tags": tags,
            "slots": {"main": {"content": "visible text"}},
        }
        for rev, tags in ((13, []), (12, ["botEdit"]), (11, []), (10, []))
    ]
    packet = {"query": {"pages": [{"pageid": 24, "ns": 0, "revisions": rows}]}}
    assert [pair[:2] for pair in direct.pre2020_pairs(24, packet)] == [(10, 11)]
    rows[0]["timestamp"] = "2020-01-13T00:00:00Z"
    with pytest.raises(ValueError, match="strictly before 2020"):
        direct.pre2020_pairs(24, packet)


def test_rights_gate_rejects_imports_and_unresolved_or_unbound_checks():
    [window] = direct.retouches(*example())
    candidate = {"page_id": 24, "before_revision": 100, "after_revision": 101, "windows": [window]}
    *_, digest = direct.review_key(candidate, window)
    review = {
        "page_id": 24,
        "before_revision": 100,
        "after_revision": 101,
        "window_sha256": digest,
        "decision": "approved",
        "license": intake.LICENSE,
        "license_policy": intake.POLICY,
        "checks": dict.fromkeys(direct.REVIEW_CHECKS, True),
        "story_component": intake.sha(b"synthetic story"),
    }
    assert direct.rights_admission([candidate], [], set())["rights_pending_windows"] == 1
    assert direct.rights_admission([candidate], [review], set())["rights_admitted_pages"] == 1
    denied = direct.rights_admission([candidate], [{**review, "decision": "imported"}], set())
    assert denied["rights_admitted_pages"] == 0 and denied["rights_exclusions"] == {"imported": 1}
    exposure = intake.fingerprints(window["after"]["excerpt"])
    assert direct.rights_admission([candidate], [review], exposure)["rights_exclusions"] == {
        "prior_project_exposure": 1
    }
    with pytest.raises(ValueError, match="exact source window"):
        direct.rights_admission([candidate], [{**review, "window_sha256": "f" * 64}], set())
    review["checks"]["licensed_original_prose"] = False
    with pytest.raises(ValueError, match="original licensed prose"):
        direct.rights_admission([candidate], [review], set())
