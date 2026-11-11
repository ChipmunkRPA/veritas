"""Cache a fixed Wikinews archive sample and locate exact adjacent source witnesses."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

if __package__:
    from . import text_short_insert_intake_v203 as intake
else:
    import text_short_insert_intake_v203 as intake

API = "https://en.wikinews.org/w/api.php"
HISTORY_END = "2021-07-31T23:59:59Z"
USER_AGENT = "ChipmunkVerifySourceAudit/1.0 (https://chipmunkrpa.com) urllib"
PAGES, REVISIONS, RENDER_PAIRS = 12, 24, 2
MAX_REQUESTS = PAGES * (1 + 2 * RENDER_PAIRS)
MAX_BYTES = 2_000_000
MIN_REQUEST_GAP = 1.0
REVERTS = {"mw-rollback", "mw-undo", "mw-reverted", "mw-manual-revert"}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ArchiveStopped(Exception):
    """Publisher or local archive boundary stopped the sequential collection."""


def safe_cache(path):
    allowed = (intake.PRIVATE.resolve(), Path(tempfile.gettempdir()).resolve(), Path("/private/tmp"))
    intake.require(
        not path.is_symlink()
        and any(path.resolve() != root and path.resolve().is_relative_to(root) for root in allowed),
        "Archive cache must be a private evaluation or temporary subdirectory",
    )
    path.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    intake.require(
        info.st_uid == os.getuid() and info.st_mode & 0o077 == 0, "Archive cache must be owner-only"
    )


def retain(path, data):
    if path.is_file():
        intake.require(intake.read_file(path, MAX_BYTES) == data, "Existing archive record differs")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent) as temp:
        temp.write(data)
        temp.flush()
        os.link(temp.name, path)


class Archive:
    def __init__(self, cache, *, offline=False):
        self.cache = cache
        self.offline = offline
        self.requests = 0
        self.last_request = None
        self.opener = urllib.request.build_opener(NoRedirect())

    def read(self, filename, **params):
        path = self.cache / filename
        if path.exists():
            data = intake.read_file(path, MAX_BYTES)
        else:
            if self.offline:
                raise ArchiveStopped("missing_offline_record")
            if self.requests >= MAX_REQUESTS:
                raise ArchiveStopped("request_budget_exhausted")
            if self.last_request is not None:
                time.sleep(max(0.0, self.last_request + MIN_REQUEST_GAP - time.monotonic()))
            query = urllib.parse.urlencode({"format": "json", "formatversion": 2, "maxlag": 5, **params})
            request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
            self.requests += 1
            try:
                with self.opener.open(request, timeout=12) as response:
                    data = response.read(MAX_BYTES + 1)
            except urllib.error.HTTPError as error:
                raise ArchiveStopped(f"publisher_http_{error.code}") from None
            except (OSError, urllib.error.URLError):
                raise ArchiveStopped("publisher_transport_failed") from None
            finally:
                self.last_request = time.monotonic()
            if len(data) > MAX_BYTES:
                raise ArchiveStopped("publisher_response_exceeded_bound")
        try:
            result = json.loads(data)
        except (UnicodeError, ValueError):
            raise ArchiveStopped("publisher_invalid_json") from None
        if not isinstance(result, dict) or "error" in result or "errors" in result or "warnings" in result:
            raise ArchiveStopped("publisher_api_error_or_warning")
        retain(path, data)
        return result, intake.sha(data)

    def history(self, page):
        return self.read(
            f"history-{page}.json",
            action="query",
            prop="revisions",
            pageids=page,
            rvstart=HISTORY_END,
            rvlimit=REVISIONS,
            rvslots="main",
            rvprop="ids|timestamp|flags|tags|userid|sha1|content",
        )

    def render(self, page, revision):
        result, digest = self.read(
            f"render-{revision}.json",
            action="parse",
            oldid=revision,
            prop="text|revid|categories",
            disablelimitreport=1,
        )
        return intake.revision_prose(result, page, revision), digest


def roster(train):
    proposed = intake.all_candidates(intake.train_rows(train))
    pages = [page for page in sorted(proposed) if proposed[page]]
    intake.require(len(pages) == 62, "Mechanical publisher roster changed")
    return [(page, proposed[page]) for page in pages[:PAGES]]


def raw_anchor(source):
    """Make only retrieval hints from literal wikitext; no result is accepted on these hints."""
    cleaned = re.sub(r"<!--.*?-->|<ref\b[^>]*>.*?</ref>|<ref\b[^>]*/\s*>", " ", source, flags=re.I | re.S)
    cleaned = re.sub(r"\[\[([^\[\]]+)\]\]", lambda match: match[1].rsplit("|", maxsplit=1)[-1], cleaned)
    cleaned = re.sub(r"'{2,5}|</?\w+\b[^>]*>", "", cleaned)
    return f" {intake.canonical(html.unescape(cleaned))} "


def anchor_score(excerpt, raw):
    a, b = excerpt.target
    c, d = excerpt.retouch
    spans = ((excerpt.text, 100), (excerpt.text[a:b], 20), (excerpt.text[c:d], 1))
    return sum(weight for text, weight in spans if f" {intake.canonical(text)} " in raw)


def adjacent_pairs(page, history):
    records = history["query"]["pages"]
    intake.require(
        len(records) == 1 and records[0]["pageid"] == page and records[0]["ns"] == 0,
        "Official history does not match the requested article",
    )
    entries = records[0].get("revisions", [])
    by_revision = {item["revid"]: item for item in entries}
    times = [intake.timestamp(item["timestamp"]) for item in entries]
    intake.require(
        len(entries) == len(by_revision) <= REVISIONS
        and times == sorted(times, reverse=True)
        and all(time < intake.CUTOFF for time in times),
        "Official revision order or date search bound changed",
    )
    for after in entries:
        before = by_revision.get(after["parentid"])
        if (
            before is None
            or not intake.EARLIEST
            <= intake.timestamp(before["timestamp"])
            < intake.timestamp(after["timestamp"])
            or type(after.get("userid")) is not int
            or after["userid"] <= 0
            or REVERTS.intersection(after["tags"])
            or any("texthidden" in rev or "suppressed" in rev for rev in (before, after))
        ):
            continue
        raw = tuple(rev.get("slots", {}).get("main", {}).get("content") for rev in (before, after))
        if all(isinstance(content, str) for content in raw):
            yield before["revid"], after["revid"], tuple(raw_anchor(content) for content in raw)


def ranked_pairs(pairs, proposed):
    def score(pair):
        return max(
            min(anchor_score(case.original, pair[2][0]), anchor_score(case.revised, pair[2][1]))
            for case in proposed
        )

    # `pairs` retains newest-first official order as the deterministic tie-break.
    return [(pair, score(pair)) for pair in sorted(pairs, key=score, reverse=True)[:RENDER_PAIRS]]


def pilot(train, archive):
    fixed = roster(train)
    counts = Counter()
    witnesses = []
    status = "completed_fixed_sample"
    try:
        for page, proposed in fixed:
            history, history_hash = archive.history(page)
            pairs = list(adjacent_pairs(page, history))
            counts["histories_verified"] += 1
            counts["visible_registered_adjacent_pairs"] += len(pairs)
            page_matches = []
            for (before, after, _), hint in ranked_pairs(pairs, proposed):
                if hint:
                    counts["ranked_pairs_with_both_raw_hints"] += 1
                first, before_hash = archive.render(page, before)
                second, after_hash = archive.render(page, after)
                counts["rendered_pairs_reviewed"] += 1
                matches = [
                    case
                    for case in proposed
                    if intake.unique_prose_match(case.original.text, first)
                    and intake.unique_prose_match(case.revised.text, second)
                ]
                if matches:
                    page_matches.append(
                        {
                            "page_id": page,
                            "before_revision": before,
                            "after_revision": after,
                            "history_sha256": history_hash,
                            "before_render_sha256": before_hash,
                            "after_render_sha256": after_hash,
                            "exact_excerpt_windows": len(matches),
                        }
                    )
                    counts["exact_adjacent_revision_pairs"] += 1
                    counts["exact_excerpt_windows"] += len(matches)
            if page_matches:
                counts["pages_with_exact_adjacent_witness"] += 1
                witnesses.extend(page_matches)
            counts["pages_completed"] += 1
    except ArchiveStopped as error:
        status = str(error)
    return {
        "schema": "chipmunk.text-wikinews-archive-pilot/205",
        "status": status,
        "fixed_pages": len(fixed),
        "max_requests": MAX_REQUESTS,
        "actual_new_requests": archive.requests,
        "counts": dict(counts),
        "rights_admitted": 0,
        "model_calls": 0,
    }, witnesses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.offline:
        intake.require(args.cache.is_dir(), "Offline replay requires an existing private archive cache")
    safe_cache(args.cache)
    report, witnesses = pilot(args.train, Archive(args.cache, offline=args.offline))
    if report["status"] == "completed_fixed_sample":
        proof = args.cache / "private-witnesses.json"
        data = json.dumps(witnesses, indent=2).encode()
        if args.offline:
            intake.require(
                intake.read_file(proof, MAX_BYTES) == data, "Private archive witness replay differs"
            )
        else:
            retain(proof, data)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
