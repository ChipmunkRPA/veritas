"""Retrieve exact source records for the remaining v233 copy-edit questions."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

if __package__:
    from . import text_goce_source_evidence_v233 as prior
else:
    import text_goce_source_evidence_v233 as prior

archive, text = prior.archive, prior.text
PUBLIC_RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/evaluation-panels/text-goce-source-evidence-complete-v233.json"
)
PUBLIC_SHA256 = "037feef6861acac36817f07af6d5ec2ce816aa91a1c197d7804ac94a30381811"
MAX_REQUESTS = 1 + 1 + 2 * prior.HISTORY_BATCHES
COPY_LINK = re.compile(r"\bcopied from\s+\[\[[^\[\]\r\n]+\]\]", re.I)
CREDIT = re.compile(r"\b(?:attribution|credit)\b", re.I)
PEER_REVIEW = re.compile(
    r"^[ \t]*\|[ \t]*action[0-9]+link[ \t]*=[ \t]*"
    r"(Wikipedia:Peer review/([^/\[\]{}\r\n]+)/archive[ _]*[1-9][0-9]*)[ \t]*$",
    re.I | re.M,
)


class PriorLedger(prior.RequestLedger):
    def __init__(self, source):
        super().__init__(source)
        self.packets = []

    def read(self, purpose, **query):
        packet = super().read(purpose, **query)
        self.packets.append((purpose, query, packet))
        return packet


class FollowupLedger(prior.RequestLedger):
    def read(self, purpose, **query):
        if len(self.hashes) >= MAX_REQUESTS:
            raise archive.ArchiveStopped("source_followup_request_bound")
        return super().read(purpose, **query)


def verified_prior(intake_cache, evidence_cache):
    groups, roster_hash = prior.frozen_candidates(intake_cache)
    text.require(evidence_cache.is_dir(), "The completed v233 cache is required")
    archive.safe_cache(evidence_cache)
    ledger = PriorLedger(prior.intake.OfficialArchive(evidence_cache, offline=True))
    report = prior.collect(ledger, groups, roster_hash)
    public_bytes = prior.private_bytes(PUBLIC_RECEIPT)
    text.require(
        text.sha(public_bytes) == PUBLIC_SHA256
        and json.loads(public_bytes) == report
        and prior.private_bytes(evidence_cache / "aggregate.json") == prior.json_bytes(report)
        and report["status"] == "collection_complete"
        and ledger.source.requests == 0,
        "Published v233 receipt and private evidence did not replay together",
    )
    return groups, ledger.packets, report["private_evidence_commitment_sha256"]


def review_keys(groups, page):
    return [
        prior.intake.prose.review_key(pair, window) for pair in groups[page] for window in pair["windows"]
    ]


def title_key(title):
    return " ".join(title.replace("_", " ").split()).casefold()


def private_plan(groups, packets, prior_evidence_hash):
    (attribution,) = [packet for purpose, _, packet in packets if purpose == "article_attribution"]
    talk_subjects = {
        page["talkid"]: page["pageid"] for page in prior.page_rows(attribution) if "talkid" in page
    }
    copy_rows, discussions = [], []
    trails = defaultdict(lambda: defaultdict(list))
    for purpose, query, packet in packets:
        if purpose == "article_history_metadata":
            page = query["pageids"]
            for row in prior.revisions_on_page(packet, page, 0):
                comment = row.get("comment", "")
                if COPY_LINK.search(comment) and CREDIT.search(comment):
                    copy_rows.append(
                        {
                            "page": page,
                            "revision": row["revid"],
                            "parent": row["parentid"],
                            "timestamp": row["timestamp"],
                            "comment_sha256": text.sha(comment.encode()),
                            "review_keys": review_keys(groups, page),
                        }
                    )
        elif purpose == "historical_talk":
            talk = query["pageids"]
            (page,) = prior.page_rows(packet)
            rows = prior.revisions_on_page(packet, talk, 1)
            if rows and prior.visible_wikitext(rows[0]):
                source = rows[0]["slots"]["main"]["content"]
                for match in PEER_REVIEW.finditer(source):
                    if title_key(match[2]) == title_key(page["title"].split(":", 1)[-1]):
                        discussions.append(
                            {
                                "title": match[1],
                                "cutoff": query["rvstart"],
                                "review_keys": review_keys(groups, talk_subjects[talk]),
                            }
                        )
        elif purpose == "article_parent_wikitext":
            page, start = query["pageids"], query["rvstartid"]
            trails[page][start].extend(prior.revisions_on_page(packet, page, 0))
    older = []
    for page, by_parent in trails.items():
        if len(groups[page]) != 2:
            continue
        text.require(len(by_parent) == 2, "Expected both verified v233 parent trails")
        left, right = by_parent.values()
        shared = {row["revid"] for row in left} & {row["revid"] for row in right}
        text.require(
            len(left) == len(right) == prior.HISTORY_BATCHES * prior.CONTENT_LIMIT
            and len(shared) == len(left) - 1,
            "Expected the fixed overlap between v233 parent trails",
        )
        oldest = min((left[-1], right[-1]), key=lambda row: (row["timestamp"], row["revid"]))
        text.require(oldest["parentid"] > 0, "The oldest v233 row has no earlier parent")
        older.append({"page": page, "anchor": oldest["parentid"], "review_keys": review_keys(groups, page)})
    text.require(
        len(copy_rows) == len(discussions) == 1 and len(older) == 2 and copy_rows[0]["parent"] > 0,
        "The published v233 evidence does not support the fixed follow-up targets",
    )
    return {
        "schema": "chipmunk.text-goce-private-source-followup-plan/235",
        "v233_public_sha256": PUBLIC_SHA256,
        "v233_private_commitment_sha256": prior_evidence_hash,
        "credited_copy": copy_rows[0],
        "historical_peer_review": discussions[0],
        "older_article_history": older,
    }


def credited_copy(ledger, plan):
    packet = ledger.read(
        "credited_copy_pair",
        action="query",
        prop="revisions",
        revids=f"{plan['parent']}|{plan['revision']}",
        rvslots="main",
        rvprop=prior.WIKITEXT,
    )
    rows = prior.revisions_on_page(packet, plan["page"], 0)
    revisions = {row.get("revid"): row for row in rows}
    prior.check(
        len(rows) == 2 and set(revisions) == {plan["parent"], plan["revision"]} and "continue" not in packet
    )
    before, after = (revisions[plan[key]] for key in ("parent", "revision"))
    comment = after.get("comment")
    prior.check(
        after.get("parentid") == before["revid"]
        and after.get("timestamp") == plan["timestamp"]
        and isinstance(before.get("timestamp"), str)
        and before["timestamp"] < plan["timestamp"]
        and isinstance(comment, str)
        and text.sha(comment.encode()) == plan["comment_sha256"]
    )
    return sum(map(prior.visible_wikitext, rows))


def peer_review(ledger, plan):
    packet = ledger.read(
        "historical_peer_review",
        action="query",
        prop="revisions",
        titles=plan["title"],
        rvdir="older",
        rvstart=plan["cutoff"],
        rvlimit=1,
        rvslots="main",
        rvprop=prior.WIKITEXT,
    )
    pages = prior.page_rows(packet)
    prior.check(
        len(pages) == 1
        and pages[0].get("ns") == 4
        and title_key(pages[0].get("title", "")) == title_key(plan["title"])
    )
    rows = pages[0].get("revisions", [])
    prior.check(
        isinstance(rows, list)
        and len(rows) <= 1
        and all(
            isinstance(row, dict)
            and isinstance(row.get("timestamp"), str)
            and row["timestamp"] <= plan["cutoff"]
            for row in rows
        )
    )
    return bool(rows and prior.visible_wikitext(rows[0]))


def collect(ledger, plan):
    plan_hash = text.sha(prior.json_bytes(plan))
    counts = Counter()
    status = "collection_complete"
    try:
        counts["credited_copy_wikitext_revisions"] = credited_copy(ledger, plan["credited_copy"])
        counts["historical_peer_review_with_wikitext"] = int(
            peer_review(ledger, plan["historical_peer_review"])
        )
        for target in plan["older_article_history"]:
            rows, outcome = prior.article_history(ledger, target["page"], target["anchor"], content=True)
            counts["older_article_history_rows"] += rows
            counts[f"older_article_history_{outcome}"] += 1
    except archive.ArchiveStopped as error:
        status = str(error)
    return {
        "schema": "chipmunk.text-goce-source-followup/235",
        "status": status,
        "counts": dict(counts),
        "max_requests": MAX_REQUESTS,
        "requests_recorded": len(ledger.hashes),
        "v233_public_sha256": plan["v233_public_sha256"],
        "v233_private_commitment_sha256": plan["v233_private_commitment_sha256"],
        "private_plan_sha256": plan_hash,
        "private_evidence_commitment_sha256": text.sha(prior.json_bytes([plan_hash, *ledger.hashes])),
        "rights_decisions_recorded": 0,
        "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intake_cache", type=Path)
    parser.add_argument("v233_cache", type=Path)
    parser.add_argument("followup_cache", type=Path, nargs="?")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.preflight == bool(args.followup_cache):
        parser.error("Use --preflight without an output cache, or supply a follow-up cache")
    plan = private_plan(*verified_prior(args.intake_cache, args.v233_cache))
    plan_bytes = prior.json_bytes(plan)
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": "verified_preflight",
                    "private_plan_sha256": text.sha(plan_bytes),
                    "planned_older_groups": len(plan["older_article_history"]),
                    "max_requests": MAX_REQUESTS,
                    "new_requests": 0,
                },
                indent=2,
            )
        )
        return
    cache = args.followup_cache
    text.require(
        cache.is_dir() if args.offline else not cache.exists(),
        "Use a fresh private follow-up cache or replay an existing one offline",
    )
    archive.safe_cache(cache)
    plan_path = cache / "private-plan.json"
    if args.offline:
        text.require(prior.private_bytes(plan_path) == plan_bytes, "Private source follow-up plan differs")
    else:
        archive.retain(plan_path, plan_bytes)
    source = prior.intake.OfficialArchive(cache, offline=args.offline)
    report = collect(FollowupLedger(source), plan)
    report_path = cache / "aggregate.json"
    raw = prior.json_bytes(report)
    if args.offline:
        text.require(prior.private_bytes(report_path) == raw, "Private source follow-up replay differs")
    else:
        archive.retain(report_path, raw)
    print(json.dumps({**report, "new_requests": source.requests}, indent=2))


if __name__ == "__main__":
    main()
