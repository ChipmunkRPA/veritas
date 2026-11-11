import io
import json
import urllib.error

import pytest

from scripts import text_short_insert_intake_v203 as intake
from scripts import text_wikinews_archive_v205 as archive


def official_revision(revid, parent, day, content, *, user=3, tags=()):
    return {
        "revid": revid,
        "parentid": parent,
        "timestamp": f"2021-07-{day:02d}T08:00:00Z",
        "userid": user,
        "tags": list(tags),
        "slots": {"main": {"content": content}},
    }


def publisher_candidate():
    old = "Left text keeps the actual café choice close to the rest."
    new = "Left text keeps the clearer café description close to the rest."

    def passage(text, word):
        start = text.index(word)
        return intake.Excerpt(text, 0, (0, len(text)), (start, start + len(word)), 12)

    return intake.Candidate(10, 1, passage(old, "choice"), passage(new, "description"))


def test_only_direct_adjacent_registered_archive_pairs_and_raw_priority():
    candidate = publisher_candidate()
    records = [
        official_revision(16, 15, 26, "unrelated after"),
        official_revision(15, 14, 25, "unrelated before"),
        official_revision(14, 99, 24, "revision whose direct parent is outside this packet"),
        official_revision(13, 12, 23, "ignored revert", tags=["mw-undo"]),
        official_revision(12, 11, 22, candidate.revised.text.replace("café", "[[Coffee|café]]")),
        official_revision(11, 10, 21, candidate.original.text),
    ]
    history = {"query": {"pages": [{"pageid": 10, "ns": 0, "revisions": records}]}}
    pairs = list(archive.adjacent_pairs(10, history))
    assert [(a, b) for a, b, _ in pairs] == [(15, 16), (14, 15), (11, 12)]
    ranked = archive.ranked_pairs(pairs, [candidate])
    assert [pair[:2] for pair, _ in ranked] == [(11, 12), (15, 16)]
    assert ranked[0][1] == 121


def test_transport_stops_on_http_rejection_without_writing(tmp_path, monkeypatch):
    source = archive.Archive(tmp_path)

    class Rejected:
        def open(self, request, timeout):
            assert "maxlag=5" in request.full_url
            assert request.full_url.startswith(archive.API)
            assert timeout == 12
            raise urllib.error.HTTPError(request.full_url, 429, "Rate limit", {}, None)

    monkeypatch.setattr(source, "opener", Rejected())
    with pytest.raises(archive.ArchiveStopped, match="publisher_http_429"):
        source.history(10)
    assert source.requests == 1
    assert list(tmp_path.iterdir()) == []


def test_cache_reuses_exact_bytes_without_overwriting(tmp_path, monkeypatch):
    source = archive.Archive(tmp_path)
    raw = b'{"query": {"pages": []}}\n'

    class Client:
        def open(self, request, timeout):
            return io.BytesIO(raw)

    monkeypatch.setattr(source, "opener", Client())
    value, digest = source.history(10)
    path = tmp_path / "history-10.json"
    assert path.read_bytes() == raw
    assert digest == intake.sha(raw)
    assert source.history(10) == (value, digest)
    assert source.requests == 1
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="differs"):
        archive.retain(path, b"{}")
    offline = archive.Archive(tmp_path, offline=True)
    with pytest.raises(archive.ArchiveStopped, match="missing_offline_record"):
        offline.history(11)
    assert offline.requests == 0


def test_pilot_only_exact_rendered_witnesses_and_private_ids(tmp_path, monkeypatch):
    proposed = publisher_candidate()
    monkeypatch.setattr(archive, "roster", lambda _: [(10, [proposed]), (20, [proposed])])

    class Official:
        requests = 0

        def history(self, page):
            self.requests += 1
            entries = [
                official_revision(12 + page, 11 + page, 2, proposed.revised.text),
                official_revision(11 + page, 10 + page, 1, proposed.original.text),
            ]
            return {"query": {"pages": [{"pageid": page, "ns": 0, "revisions": entries}]}}, "history digest"

        def render(self, page, revision):
            self.requests += 1
            if page == 20:
                return "Other text", "private digest"
            return (proposed.revised.text if revision == 22 else proposed.original.text), "private digest"

    report, witnesses = archive.pilot(tmp_path, Official())
    assert report["status"] == "completed_fixed_sample"
    assert report["counts"]["ranked_pairs_with_both_raw_hints"] == 2
    assert report["counts"]["pages_with_exact_adjacent_witness"] == 1
    assert report["rights_admitted"] == report["model_calls"] == 0
    assert witnesses[0]["page_id"] == 10
    assert "page_id" not in json.dumps(report) and "private digest" not in json.dumps(report)


def test_fixed_pilot_stops_at_first_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "roster", lambda _: [(1, []), (2, [])])

    class Stop:
        requests = 0

        def history(self, page):
            self.requests += 1
            assert page == 1
            raise archive.ArchiveStopped("publisher_http_403")

    report, witnesses = archive.pilot(tmp_path, Stop())
    assert report["status"] == "publisher_http_403"
    assert report["actual_new_requests"] == 1
    assert witnesses == []
