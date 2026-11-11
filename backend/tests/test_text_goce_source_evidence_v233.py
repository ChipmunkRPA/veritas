import io
import json
from collections import deque
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import text_goce_source_evidence_v233 as review

ARTICLE, TALK = 90, 91
GROUPS = {ARTICLE: [{"before_revision": 100, "after_revision": 101}]}
ANCHOR_TIME = "2018-01-03T12:00:00Z"
PRIVATE_TITLE = "Fixture example article"
PRIVATE_TEXT = "The underlying private example prose."


def revision(identifier, parent, date=ANCHOR_TIME):
    return {
        "revid": identifier,
        "parentid": parent,
        "timestamp": date,
        "slots": {"main": {"contentmodel": "wikitext", "content": PRIVATE_TEXT}},
    }


def page(identifier, namespace, *revisions, continuation=None):
    packet = {"query": {"pages": [{"pageid": identifier, "ns": namespace, "revisions": list(revisions)}]}}
    if continuation:
        packet["continue"] = continuation
    return packet


def attribution():
    return {
        "query": {
            "rightsinfo": {"text": "Current site license"},
            "pages": [
                {
                    "pageid": ARTICLE,
                    "ns": 0,
                    "title": PRIVATE_TITLE,
                    "talkid": TALK,
                    "fullurl": "https://en.wikipedia.org/wiki/Fixture_example_article",
                }
            ],
        },
    }


def exact(parent=100):
    return page(ARTICLE, 0, revision(100, 99, "2018-01-02T12:00:00Z"), revision(101, parent))


class FixtureTransport:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.queries = []

    def open(self, request, timeout):
        url = urlsplit(request.full_url)
        assert (url.scheme, url.netloc, url.path) == ("https", "en.wikipedia.org", "/w/api.php")
        assert timeout == 12
        query = parse_qs(url.query)
        assert query["maxlag"] == ["5"] and query["formatversion"] == ["2"]
        self.queries.append({key: values[0] for key, values in query.items()})
        response = self.responses.popleft()
        if isinstance(response, int):
            raise HTTPError(request.full_url, response, "fixture rejection", {}, None)
        return io.BytesIO(review.json_bytes(response))


def sources(tmp_path, monkeypatch, *responses):
    tmp_path.chmod(0o700)
    transport = FixtureTransport(*responses)
    live = review.intake.OfficialArchive(tmp_path)
    live.opener = transport
    monkeypatch.setattr(review.archive, "MIN_REQUEST_GAP", 0)
    return live, review.intake.OfficialArchive(tmp_path, offline=True), transport


def test_fixture_pass_replays_and_leaves_private_prose_and_queries_in_owner_cache(tmp_path, monkeypatch):
    continuation = {"continue": "||", "rvcontinue": "20180102000000|100"}
    live, offline, transport = sources(
        tmp_path,
        monkeypatch,
        attribution(),
        exact(),
        page(TALK, 1, revision(201, 200, "2018-01-01T12:00:00Z")),
        page(TALK, 1),
        page(ARTICLE, 0, revision(101, 100), continuation=continuation),
        page(ARTICLE, 0, revision(100, 99), revision(99, 0)),
        page(ARTICLE, 0, revision(100, 99), revision(99, 0)),
    )
    report = review.collect(review.RequestLedger(live), GROUPS, "fixture-roster")
    assert report["status"] == "collection_complete"
    assert report["requests_recorded"] == live.requests == 7
    assert report["counts"] == {
        "article_attribution_records": 1,
        "exact_revisions_received": 2,
        "exact_revisions_with_wikitext": 2,
        "historical_talk_wikitext": 1,
        "supplemental_talk_unavailable": 1,
        "article_history_metadata_rows": 3,
        "article_history_metadata_complete": 1,
        "article_parent_history_rows": 2,
        "article_parent_history_complete": 1,
    }
    assert not transport.responses
    assert transport.queries[1]["revids"] == "100|101"
    assert "rvlimit" not in transport.queries[1]
    assert transport.queries[2]["rvstart"] == ANCHOR_TIME
    assert transport.queries[4]["rvstartid"] == "101"
    assert transport.queries[5]["rvcontinue"] == continuation["rvcontinue"]
    assert transport.queries[6]["rvstartid"] == "100"
    assert report == review.collect(review.RequestLedger(offline), GROUPS, "fixture-roster")
    assert offline.requests == 0
    public = json.dumps(report)
    assert PRIVATE_TEXT not in public and PRIVATE_TITLE not in public and "2018-01" not in public
    assert PRIVATE_TEXT in (tmp_path / "response-02.json").read_text()


def test_transport_failure_is_sealed_and_replays_without_retries(tmp_path, monkeypatch):
    live, offline, transport = sources(tmp_path, monkeypatch, attribution(), 503)
    report = review.collect(review.RequestLedger(live), GROUPS, "fixture-roster")
    assert report["status"] == "publisher_http_503"
    assert len(transport.queries) == report["requests_recorded"] == 2
    assert report["counts"] == {"article_attribution_records": 1}
    assert not (tmp_path / "response-02.json").exists()
    assert report == review.collect(review.RequestLedger(offline), GROUPS, "fixture-roster")
    assert offline.requests == 0


def test_exact_pair_mismatch_preserves_payload_and_stops(tmp_path, monkeypatch):
    live, offline, _ = sources(tmp_path, monkeypatch, attribution(), exact(parent=99))
    report = review.collect(review.RequestLedger(live), GROUPS, "fixture-roster")
    assert report["status"] == "source_response_binding_invalid"
    assert report["counts"] == {"article_attribution_records": 1}
    assert (tmp_path / "response-02.json").is_file()
    assert report == review.collect(review.RequestLedger(offline), GROUPS, "fixture-roster")


def test_replay_rejects_changed_response_bytes(tmp_path, monkeypatch):
    live, offline, _ = sources(tmp_path, monkeypatch, attribution(), 503)
    review.collect(review.RequestLedger(live), GROUPS, "fixture-roster")
    body = tmp_path / "response-01.json"
    body.write_bytes(body.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Private response bytes differ"):
        review.collect(review.RequestLedger(offline), GROUPS, "fixture-roster")


def test_history_bound_and_lineage_gap_have_distinct_receipts(tmp_path, monkeypatch):
    first = {"continue": "||", "rvcontinue": "first"}
    second = {"continue": "||", "rvcontinue": "second"}
    live, _, transport = sources(
        tmp_path,
        monkeypatch,
        page(ARTICLE, 0, revision(101, 100), continuation=first),
        page(ARTICLE, 0, revision(100, 99), continuation=second),
        page(ARTICLE, 0, revision(99, 98), revision(97, 0)),
    )
    ledger = review.RequestLedger(live)
    assert review.article_history(ledger, ARTICLE, 101) == (2, "bounded")
    assert len(transport.queries) == 2
    assert review.article_history(ledger, ARTICLE, 99, content=True) == (1, "lineage_gap")
    assert len(transport.queries) == 3
