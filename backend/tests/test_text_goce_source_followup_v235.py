from collections import deque

import pytest

from scripts import text_goce_source_followup_v235 as followup

prior = followup.prior
PAST = "2018-01-01T00:00:00Z"
CUTOFF = "2018-01-21T00:00:00Z"
PRIVATE_COPY = "copied from [[Private donor example]] for attribution"
ARCHIVE_TITLE = "Wikipedia:Peer review/Private subject example/archive1"


def row(identifier, parent, comment="", timestamp=PAST):
    return {
        "revid": identifier,
        "parentid": parent,
        "comment": comment,
        "timestamp": timestamp,
        "slots": {"main": {"contentmodel": "wikitext", "content": "Private example prose"}},
    }


def page(identifier, namespace, *rows, title="Private subject example"):
    return {
        "query": {"pages": [{"pageid": identifier, "ns": namespace, "title": title, "revisions": list(rows)}]}
    }


def pair(page_id, parent):
    return {
        "page_id": page_id,
        "before_revision": parent,
        "after_revision": parent + 1,
        "windows": [{"before": "Earlier prose", "after": "Revised prose"}],
    }


def fixture_plan():
    groups = {
        10: [pair(10, 100), pair(10, 101)],
        11: [pair(11, 200), pair(11, 201)],
        12: [pair(12, 300)],
        13: [pair(13, 400)],
    }
    archived_talk = row(500, 499)
    archived_talk["slots"]["main"]["content"] = (
        f"| action3link = {ARCHIVE_TITLE}\n| action4link = Wikipedia:Peer review/Unrelated subject/archive1\n"
    )
    packets = [
        ("article_attribution", {}, {"query": {"pages": [{"pageid": 10, "talkid": 50}]}}),
        ("article_history_metadata", {"pageids": 13}, page(13, 0, row(395, 394, PRIVATE_COPY))),
        (
            "historical_talk",
            {"pageids": 50, "rvstart": CUTOFF},
            page(50, 1, archived_talk, title="Talk:Private subject example"),
        ),
        (
            "supplemental_talk",
            {"pageids": 50, "rvstart": CUTOFF},
            page(50, 1, archived_talk, title="Talk:Private subject example"),
        ),
    ]
    for page_id, newest in ((10, 101), (11, 201)):
        for start in (newest - 1, newest):
            rows = [row(i, i - 1) for i in range(start, start - 16, -1)]
            packets.append(
                ("article_parent_wikitext", {"pageids": page_id, "rvstartid": start}, page(page_id, 0, *rows))
            )
    return groups, packets


class FixtureSource:
    offline = False

    def __init__(self, cache, *packets):
        self.cache = cache
        self.packets = deque(packets)
        self.queries = []

    def read(self, filename, **query):
        self.queries.append(query)
        packet = self.packets.popleft()
        if packet is None:
            raise prior.archive.ArchiveStopped("publisher_http_503")
        raw = prior.json_bytes(packet)
        prior.archive.retain(self.cache / filename, raw)
        return packet, prior.text.sha(raw)


def test_targets_bind_historical_subject_copy_credit_and_overlapping_parent_trails():
    groups, packets = fixture_plan()
    plan = followup.private_plan(groups, packets, "private-v233")
    assert plan["credited_copy"]["parent"] == 394
    assert plan["credited_copy"]["review_keys"] == [
        prior.intake.prose.review_key(groups[13][0], groups[13][0]["windows"][0])
    ]
    assert plan["historical_peer_review"]["title"] == ARCHIVE_TITLE
    assert plan["historical_peer_review"]["cutoff"] == CUTOFF
    assert [p["anchor"] for p in plan["older_article_history"]] == [84, 184]
    assert PRIVATE_COPY not in prior.json_bytes(plan).decode()
    packets.pop()
    with pytest.raises(ValueError, match="both verified v233 parent trails"):
        followup.private_plan(groups, packets, "private-v233")


def test_collection_replays_cached_copy_peer_review_and_exact_older_anchors(tmp_path):
    groups, packets = fixture_plan()
    plan = followup.private_plan(groups, packets, "private-v233")
    live = FixtureSource(
        tmp_path,
        page(13, 0, row(394, 393, timestamp="2017-12-31T23:59:59Z"), row(395, 394, PRIVATE_COPY)),
        page(51, 4, row(900, 899), title=ARCHIVE_TITLE),
        page(10, 0, row(84, 0)),
        page(11, 0, row(184, 0)),
    )
    report = followup.collect(followup.FollowupLedger(live), plan)
    assert report["status"] == "collection_complete"
    assert report["counts"] == {
        "credited_copy_wikitext_revisions": 2,
        "historical_peer_review_with_wikitext": 1,
        "older_article_history_rows": 2,
        "older_article_history_complete": 2,
    }
    assert report["requests_recorded"] == 4
    assert report["rights_decisions_recorded"] == report["model_calls"] == 0
    assert live.queries[0]["revids"] == "394|395"
    assert live.queries[1]["titles"] == ARCHIVE_TITLE and live.queries[1]["rvstart"] == CUTOFF
    assert [r["rvstartid"] for r in live.queries[2:]] == [84, 184]
    offline = prior.intake.OfficialArchive(tmp_path, offline=True)
    assert report == followup.collect(followup.FollowupLedger(offline), plan)
    assert offline.requests == 0
    raw = prior.json_bytes(report).decode()
    assert all(s not in raw for s in (ARCHIVE_TITLE, PRIVATE_COPY, "review_keys", "Private example prose"))


def test_copy_comment_change_stops_before_any_other_request(tmp_path):
    groups, packets = fixture_plan()
    plan = followup.private_plan(groups, packets, "private-v233")
    live = FixtureSource(
        tmp_path,
        page(13, 0, row(394, 393, timestamp="2017-12-31T23:59:59Z"), row(395, 394, "other summary")),
    )
    report = followup.collect(followup.FollowupLedger(live), plan)
    assert report["status"] == "source_response_binding_invalid"
    assert report["requests_recorded"] == 1 and report["counts"] == {}
    assert (tmp_path / "response-01.json").is_file()


def test_request_cap_stops_before_starting_an_extra_request(tmp_path):
    source = FixtureSource(tmp_path, *([{}] * followup.MAX_REQUESTS))
    ledger = followup.FollowupLedger(source)
    for _ in range(followup.MAX_REQUESTS):
        ledger.read("fixture")
    with pytest.raises(prior.archive.ArchiveStopped, match="source_followup_request_bound"):
        ledger.read("fixture")
    assert len(source.queries) == followup.MAX_REQUESTS
