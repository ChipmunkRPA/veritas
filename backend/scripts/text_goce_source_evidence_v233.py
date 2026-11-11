"""Collect private source evidence for the frozen v228 copy-edit witnesses."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

if __package__:
    from . import text_goce_intake_v228 as intake
else:
    import text_goce_intake_v228 as intake

archive, text = intake.archive, intake.text
INTAKE_RESULT = text.ROOT / "docs/evaluation-panels/text-goce-intake-complete-v228.json"
METADATA = "ids|timestamp|user|userid|comment|flags|tags|slotsha1|contentmodel"
WIKITEXT = METADATA + "|content"
HISTORY_BATCHES = 2
METADATA_LIMIT, CONTENT_LIMIT = 50, 8
SUPPLEMENTAL_TALK_CUTOFF = "2026-11-11T00:00:00Z"
MAX_REQUESTS = 1 + 4 + 4 * 2 + 4 * HISTORY_BATCHES + 6 * HISTORY_BATCHES
TRANSPORT_STOPS = re.compile(
    r"publisher_(?:http_[0-9]{3}|transport_failed|response_exceeded_bound|invalid_json|api_error_or_warning)"
)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def private_bytes(path):
    return text.read_file(path, archive.MAX_BYTES)


def frozen_candidates(cache):
    text.require(cache.is_dir(), "The completed v228 cache is required")
    archive.safe_cache(cache)
    report, roster = intake.collect(intake.OfficialArchive(cache, offline=True))
    published = json.loads(INTAKE_RESULT.read_bytes())
    raw = private_bytes(cache / "private-roster.json")
    text.require(
        report["status"] == "completed_fixed_sample"
        and report == {**published, "actual_new_requests": 0}
        and json.dumps(roster, indent=2, ensure_ascii=True).encode() == raw,
        "The v228 source roster or published result did not replay",
    )
    groups = {
        row["page_id"]: [pair for pair in row["pairs"] if pair["windows"]]
        for row in roster
        if row["status"] == "mechanical_candidate"
    }
    pairs = [pair for group in groups.values() for pair in group]
    text.require(
        len(groups) == 4
        and len(pairs) == 6
        and len({pair[key] for pair in pairs for key in ("before_revision", "after_revision")}) == 10
        and sum(len(pair["windows"]) for pair in pairs) == 7,
        "The completed v228 candidate denominator differs",
    )
    return groups, text.sha(raw)


class RequestLedger:
    def __init__(self, source):
        self.source = source
        self.cache = source.cache
        self.hashes = []

    def read(self, purpose, **query):
        ordinal = len(self.hashes) + 1
        if ordinal > MAX_REQUESTS:
            raise archive.ArchiveStopped("source_review_request_bound")
        path = self.cache / f"request-{ordinal:02d}.json"
        body = f"response-{ordinal:02d}.json"
        identity = {"purpose": purpose, "query": query, "body": body}
        if self.source.offline:
            raw = private_bytes(path)
            receipt = json.loads(raw)
            text.require(
                all(receipt.get(key) == value for key, value in identity.items()),
                "Private request identity differs",
            )
            self.hashes.append(text.sha(raw))
            if receipt["status"] != "received":
                text.require(
                    TRANSPORT_STOPS.fullmatch(receipt["status"]) is not None,
                    "Private failure receipt has an unknown status",
                )
                raise archive.ArchiveStopped(receipt["status"])
            packet, digest = self.source.read(body, **query)
            text.require(digest == receipt["response_sha256"], "Private response bytes differ")
            return packet
        try:
            packet, digest = self.source.read(body, **query)
        except archive.ArchiveStopped as error:
            self.retain(path, {**identity, "status": str(error)})
            raise
        self.retain(path, {**identity, "status": "received", "response_sha256": digest})
        return packet

    def retain(self, path, receipt):
        raw = json_bytes(receipt)
        archive.retain(path, raw)
        self.hashes.append(text.sha(raw))


def check(condition):
    if not condition:
        raise archive.ArchiveStopped("source_response_binding_invalid")


def page_rows(packet):
    query = packet.get("query")
    check(isinstance(query, dict))
    pages = query.get("pages")
    check(isinstance(pages, list) and all(isinstance(page, dict) for page in pages))
    return pages


def revisions_on_page(packet, page_id, namespace):
    pages = page_rows(packet)
    check(len(pages) == 1 and pages[0].get("pageid") == page_id and pages[0].get("ns") == namespace)
    rows = pages[0].get("revisions", [])
    check(isinstance(rows, list) and all(isinstance(row, dict) for row in rows))
    return rows


def visible_wikitext(row):
    slots = row.get("slots")
    slot = slots.get("main") if isinstance(slots, dict) else None
    return (
        isinstance(slot, dict)
        and slot.get("contentmodel") == "wikitext"
        and isinstance(slot.get("content"), str)
    )


def article_info(ledger, groups):
    packet = ledger.read(
        "article_attribution",
        action="query",
        pageids="|".join(map(str, groups)),
        prop="info",
        inprop="url|talkid",
        meta="siteinfo",
        siprop="rightsinfo",
    )
    pages = page_rows(packet)
    check(
        len(pages) == len(groups)
        and {page.get("pageid") for page in pages} == set(groups)
        and all(page.get("ns") == 0 for page in pages)
        and isinstance(packet["query"].get("rightsinfo"), dict)
    )
    for page in pages:
        url = urlsplit(page.get("fullurl", ""))
        check(url.scheme == "https" and url.netloc == "en.wikipedia.org")
        check("talkid" not in page or type(page["talkid"]) is int and page["talkid"] > 0)
    return {page["pageid"]: page.get("talkid") for page in pages}


def exact_revisions(ledger, page, pairs):
    expected = sorted({pair[key] for pair in pairs for key in ("before_revision", "after_revision")})
    packet = ledger.read(
        "exact_article_wikitext",
        action="query",
        prop="revisions",
        revids="|".join(map(str, expected)),
        rvslots="main",
        rvprop=WIKITEXT,
    )
    rows = revisions_on_page(packet, page, 0)
    revisions = {row.get("revid"): row for row in rows}
    check(len(rows) == len(expected) and set(revisions) == set(expected) and "continue" not in packet)
    for pair in pairs:
        before, after = (revisions[pair[key]] for key in ("before_revision", "after_revision"))
        first, last = before.get("timestamp"), after.get("timestamp")
        check(
            after.get("parentid") == before["revid"]
            and isinstance(first, str)
            and isinstance(last, str)
            and first < last
            and last.startswith("2018-01-")
        )
    return revisions


def talk_evidence(ledger, talk, cutoff, purpose):
    packet = ledger.read(
        purpose,
        action="query",
        prop="revisions",
        pageids=talk,
        rvdir="older",
        rvstart=cutoff,
        rvlimit=1,
        rvslots="main",
        rvprop=WIKITEXT,
    )
    rows = revisions_on_page(packet, talk, 1)
    check(
        len(rows) <= 1
        and all(isinstance(row.get("timestamp"), str) and row["timestamp"] <= cutoff for row in rows)
    )
    return bool(rows and visible_wikitext(rows[0]))


def article_history(ledger, page, anchor, content=False):
    purpose = "article_parent_wikitext" if content else "article_history_metadata"
    limit = CONTENT_LIMIT if content else METADATA_LIMIT
    query = {
        "action": "query",
        "prop": "revisions",
        "pageids": page,
        "rvstartid": anchor,
        "rvdir": "older",
        "rvlimit": limit,
        "rvslots": "main",
        "rvprop": WIKITEXT if content else METADATA,
    }
    rows_seen, previous_parent, continuation = 0, anchor, {}
    for _ in range(HISTORY_BATCHES):
        packet = ledger.read(purpose, **query, **continuation)
        rows = revisions_on_page(packet, page, 0)
        check(0 < len(rows) <= limit)
        for row in rows:
            if row.get("revid") != previous_parent:
                return rows_seen, "lineage_gap"
            previous_parent = row.get("parentid")
            check(type(previous_parent) is int and previous_parent >= 0)
            rows_seen += 1
        continuation = packet.get("continue", {})
        if not continuation:
            return rows_seen, "complete" if previous_parent == 0 else "lineage_gap"
        check(
            isinstance(continuation, dict)
            and set(continuation) == {"continue", "rvcontinue"}
            and all(isinstance(value, str) for value in continuation.values())
        )
    return rows_seen, "bounded"


def collect(ledger, groups, roster_sha256):
    counts = Counter()
    status = "collection_complete"
    try:
        talk_pages = article_info(ledger, groups)
        counts["article_attribution_records"] = len(talk_pages)
        for page, pairs in groups.items():
            revisions = exact_revisions(ledger, page, pairs)
            counts["exact_revisions_received"] += len(revisions)
            counts["exact_revisions_with_wikitext"] += sum(map(visible_wikitext, revisions.values()))
            latest = max(revisions.values(), key=lambda row: row["timestamp"])
            talk = talk_pages[page]
            if talk is None:
                counts["no_talk_page_listed"] += 1
            else:
                for cutoff, purpose in (
                    (latest["timestamp"], "historical_talk"),
                    (SUPPLEMENTAL_TALK_CUTOFF, "supplemental_talk"),
                ):
                    outcome = "wikitext" if talk_evidence(ledger, talk, cutoff, purpose) else "unavailable"
                    counts[f"{purpose}_{outcome}"] += 1
            n, outcome = article_history(ledger, page, latest["revid"])
            counts["article_history_metadata_rows"] += n
            counts[f"article_history_metadata_{outcome}"] += 1
            for parent in sorted({pair["before_revision"] for pair in pairs}):
                n, outcome = article_history(ledger, page, parent, content=True)
                counts["article_parent_history_rows"] += n
                counts[f"article_parent_history_{outcome}"] += 1
    except archive.ArchiveStopped as error:
        status = str(error)
    return {
        "schema": "chipmunk.text-goce-source-evidence/233",
        "status": status,
        "planned_page_groups": len(groups),
        "counts": dict(counts),
        "requests_recorded": len(ledger.hashes),
        "max_requests": MAX_REQUESTS,
        "private_evidence_commitment_sha256": text.sha(json_bytes([roster_sha256, *ledger.hashes])),
        "rights_decisions_recorded": 0,
        "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intake_cache", type=Path)
    parser.add_argument("review_cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    groups, roster_hash = frozen_candidates(args.intake_cache)
    text.require(
        args.review_cache.is_dir() if args.offline else not args.review_cache.exists(),
        "Use a fresh private review cache or replay an existing one offline",
    )
    archive.safe_cache(args.review_cache)
    source = intake.OfficialArchive(args.review_cache, offline=args.offline)
    report = collect(RequestLedger(source), groups, roster_hash)
    report_path = args.review_cache / "aggregate.json"
    raw = json_bytes(report)
    if args.offline:
        text.require(raw == private_bytes(report_path), "Private source evidence replay differs")
    else:
        archive.retain(report_path, raw)
    print(json.dumps({**report, "new_requests": source.requests}, indent=2))


if __name__ == "__main__":
    main()
