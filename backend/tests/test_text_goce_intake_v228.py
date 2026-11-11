import html
import json

import pytest

from scripts import text_goce_intake_v228 as goce


def archive_row(article, editor, requested="2018-01-02", completed="2018-01-08", purpose="GAN"):
    return (
        f'<tr><td>{requested}</td><td><a href="/wiki/{html.escape(article.replace(" ", "_"))}">article</a></td>'
        f'<td>Requester</td><td>{completed}</td><td><a href="/wiki/User_talk:{html.escape(editor)}">editor</a></td>'
        f"<td>{purpose}</td></tr>"
    )


def source_archive(*rows):
    return {
        "parse": {
            "title": goce.INDEX,
            "revid": goce.INDEX_REVISION,
            "text": "<table>" + "".join(rows) + "</table>",
        }
    }


def revision(revid, parent, day, *, editor="Named editor", tags=(), comment="copyedit"):
    return {
        "revid": revid,
        "parentid": parent,
        "timestamp": f"2018-01-{day:02d}T12:00:00Z",
        "user": editor,
        "userid": 12,
        "comment": comment,
        "tags": list(tags),
    }


def history(title, *rows, page_id=90):
    return {"query": {"pages": [{"title": title, "pageid": page_id, "ns": 0, "revisions": list(rows)}]}}


def renders(quoted=False):
    before = " ".join(f"left{n}" for n in range(17))
    after = " ".join(f"right{n}" for n in range(17))
    before = f'{before} "borrowed"' if quoted else before
    return tuple(
        {
            "parse": {
                "pageid": 90,
                "revid": revid,
                "text": f"<p>{before} {change} {after}</p>",
                "categories": [],
            }
        }
        for revid, change in ((100, "café earlier"), (101, "clearer e\u0301vidence"))
    )


def test_planning_pins_interval_and_excludes_prior_or_uncompleted_tasks(monkeypatch):
    monkeypatch.setattr(goce, "EDITOR_CAP", 1)
    monkeypatch.setattr(goce, "DEVELOPMENT_TITLE_SHA256", {goce.text.sha(b"inspected archive article")})
    data = source_archive(
        archive_row("Inspected Archive Article", "Known editor"),
        archive_row("Part#Only", "Named editor"),
        archive_row("February", "Named editor", completed="2018-02-01"),
        archive_row("Declined article", "Named editor", purpose="Declined"),
        archive_row("Beta", "Named editor"),
        archive_row("Alpha", "Named editor"),
        archive_row("Another", "Another editor"),
    )
    chosen, count = goce.plan(data)
    assert count == 3
    assert [(item.article, item.editor) for item in chosen] == [
        ("Alpha", "Named editor"),
        ("Another", "Another editor"),
    ]
    data["parse"]["revid"] += 1
    with pytest.raises(ValueError, match="pinned GOCE"):
        goce.plan(data)


def test_history_requires_named_actor_exact_date_and_parent_identity():
    task = goce.Task("Article", "Named editor", "2018-01-02", "2018-01-08")
    first, last = revision(100, 99, 2, editor="Earlier editor"), revision(101, 100, 6)
    wrong_actor = revision(102, 101, 7, editor="Different editor")
    wrong_date = revision(103, 102, 9)
    known_revert = revision(104, 103, 7, tags=["mw-undo"])
    page, rows, status = goce.task_history(
        task, history("Article", wrong_actor, wrong_date, known_revert, last)
    )
    assert (page, rows, status) == (90, [last], "awaiting_parent")
    resolved = goce.paired_metadata(history("Article", first, last))
    assert goce.confirmed(page, task, last, resolved)
    assert not goce.confirmed(page, task, last, {**resolved, 100: (91, first)})
    assert not goce.confirmed(page, task, last, {**resolved, 101: (90, {**last, "parentid": 99})})
    assert not goce.confirmed(page, task, last, {**resolved, 100: (90, {**first, "tags": ["botEdit"]})})


def test_lexical_witness_binds_both_rendered_unicode_offsets_and_excludes_quotes():
    before, after = renders()
    windows, hashes = goce.lexical_witnesses(before, after, 90, 100, 101)
    [window] = windows
    assert len(set(hashes)) == 2
    for key, packet, revision_id, change in (
        ("before", before, 100, "café earlier"),
        ("after", after, 101, "clearer e\u0301vidence"),
    ):
        raw = goce.prose.article_prose(packet, 90, revision_id)[0]
        a, b = window[key]["retouch_codepoints"]
        assert raw[a:b] == change
        assert window[key]["target_words"] == 8
    assert goce.lexical_witnesses(*renders(quoted=True), 90, 100, 101)[0] == []
    with pytest.raises(ValueError, match="selected article"):
        goce.lexical_witnesses(before, after, 91, 100, 101)


class FixtureArchive:
    def __init__(self, stop_at=None, *, second_pair=False):
        self.requests = 0
        self.stop_at = stop_at
        self.second_pair = second_pair

    def read(self, name, **kwargs):
        self.requests += 1
        first, last = revision(100, 99, 2, editor="Earlier editor"), revision(101, 100, 6)
        before, after = renders()
        packets = {
            "archive.json": source_archive(
                archive_row("Article A", "Named editor"), archive_row("Article B", "Second editor")
            ),
            "history-00.json": history("Article A", last),
            "history-01.json": history("Article B", page_id=92),
            "pair-metadata.json": history("Article A", first, last),
            "render-100.json": before,
            "render-101.json": after,
        }
        if self.second_pair:
            extra_first, extra_last = revision(102, 101, 6, editor="Earlier editor"), revision(103, 102, 7)
            packets["history-00.json"] = history("Article A", extra_last, last)
            packets["pair-metadata.json"] = history("Article A", first, last, extra_first, extra_last)
            packets["render-102.json"] = {"parse": {**before["parse"], "revid": 102}}
            packets["render-103.json"] = {"parse": {**after["parse"], "revid": 103}}
        if self.stop_at == name:
            raise goce.archive.ArchiveStopped("publisher_http_429")
        if name == "history-00.json":
            assert kwargs["rvuser"] == "Named editor"
            assert kwargs["rvstart"] == "2018-01-08T23:59:59Z"
        return packets[name], "synthetic archive digest"


def test_collect_preserves_all_planned_tasks_and_separate_rights_state(monkeypatch):
    monkeypatch.setattr(goce, "SAMPLE", 2)
    report, private = goce.collect(FixtureArchive())
    assert report["planned_pages"] == 2
    assert report["counts"]["mechanical_candidate"] == report["counts"]["no_exact_editor_revision"] == 1
    assert report["rights_admitted_pages"] == report["model_calls"] == 0
    assert [item["status"] for item in private] == ["mechanical_candidate", "no_exact_editor_revision"]
    assert "Named editor" not in json.dumps(report) and "Article A" not in json.dumps(report)
    stopped, preserved = goce.collect(FixtureArchive(stop_at="history-00.json"))
    assert stopped["status"] == "publisher_http_429"
    assert stopped["counts"]["not_run_after_stop"] == len(preserved) == 2


def test_rejected_second_pair_retains_first_witness_and_marks_page_incomplete(monkeypatch):
    monkeypatch.setattr(goce, "SAMPLE", 2)
    stopped, preserved = goce.collect(FixtureArchive(stop_at="render-100.json", second_pair=True))
    assert stopped["status"] == "publisher_http_429"
    assert stopped["counts"]["pair_collection_incomplete"] == 1
    assert stopped["counts"]["mechanical_windows"] == 1
    assert "mechanical_candidate" not in stopped["counts"]
    assert "no_lexical_witness" not in stopped["counts"]
    assert len(preserved) == 2 and len(preserved[0]["pairs"]) == 1


def test_rights_requires_the_historical_license_and_unquoted_source(monkeypatch):
    monkeypatch.setattr(goce, "SAMPLE", 2)
    _, plan = goce.collect(FixtureArchive())
    pair = plan[0]["pairs"][0]
    window = pair["windows"][0]
    key = goce.prose.review_key(pair, window)
    record = dict(zip(("page_id", "before_revision", "after_revision", "window_sha256"), key, strict=True))
    record.update(
        decision="approved",
        source_license=goce.SOURCE_LICENSE,
        source_policy=goce.SOURCE_POLICY,
        story_component=goce.text.sha(b"story"),
        checks=dict.fromkeys(goce.RIGHTS_CHECKS, True),
    )
    assert goce.rights_review(plan, [], set())["pending_windows"] == 1
    assert goce.rights_review(plan, [record], set())["rights_admitted_pages"] == 1
    denied = goce.rights_review(plan, [{**record, "decision": "imported_or_quoted"}], set())
    assert denied["rights_admitted_pages"] == 0 and denied["rights_exclusions"] == {"imported_or_quoted": 1}
    with pytest.raises(ValueError, match="Archived source"):
        goce.rights_review(plan, [{**record, "source_license": "CC-BY-SA-4.0"}], set())
    assert goce.rights_review(plan, [record], goce.text.fingerprints(window["before"]["excerpt"]))[
        "rights_exclusions"
    ] == {"prior_project_exposure": 1}
