"""Verify a fixed January 2018 Wikipedia copy-edit intake without running models."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

if __package__:
    from . import text_short_insert_intake_v203 as text
    from . import text_wikinews_archive_v205 as archive
    from . import text_wikinews_direct_v209 as prose
else:
    import text_short_insert_intake_v203 as text
    import text_wikinews_archive_v205 as archive
    import text_wikinews_direct_v209 as prose

API = "https://en.wikipedia.org/w/api.php"
INDEX = "Wikipedia:WikiProject Guild of Copy Editors/Requests/Archives/2018"
INDEX_REVISION = 1271487041
START, END = date(2018, 1, 1), date(2018, 1, 31)
SAMPLE, EDITOR_CAP, HISTORY_CAP, PAIRS_PER_TASK = 12, 3, 12, 2
MAX_REQUESTS = 1 + SAMPLE + 1 + SAMPLE * PAIRS_PER_TASK * 2
SOURCE_LICENSE = "CC-BY-SA-3.0"
SOURCE_POLICY = "https://meta.wikimedia.org/wiki/Terms_of_use/Creative_Commons_4.0/FAQ/en"
DEVELOPMENT_TITLE_SHA256 = {
    # Four inspected articles, including one alias, using case-folded canonical titles.
    "d5d56bc62410ed872bb82ae703014e15af04ec1891d5c9458185d584b966d263",
    "8736b8d791f30b3718d6958e85678d1cd73eacdb524edddf9e44c986125d1716",
    "1faa9a757ea2bd536175392ba27579f6cf3006f73d15b5e20951272de54979ce",
    "acd9dfffd9ba6df764e3bf0fc0873a97297d8a2c70836dcab49b56081d7f8a23",
    "e44223035c37f2c1de954f81d3415bed18af0e6cc382112b8f454ff295f24474",
}
RIGHTS_CHECKS = {
    "original_licensed_article_prose",
    "quoted_or_imported_material_excluded",
    "human_editorial_task_supported",
    "bot_and_revert_excluded",
    "archived_attribution_retained",
}
REJECTED = {"rights_unresolved", "unlicensed", "imported_or_quoted", "bot_or_revert", "noneditorial"}


class OfficialArchive(archive.Archive):
    def read(self, filename, **params):
        path = self.cache / filename
        if path.exists():
            data = text.read_file(path, archive.MAX_BYTES)
        else:
            if self.offline:
                raise archive.ArchiveStopped("missing_offline_record")
            if self.requests >= MAX_REQUESTS:
                raise archive.ArchiveStopped("request_budget_exhausted")
            if self.last_request is not None:
                time.sleep(max(0, self.last_request + archive.MIN_REQUEST_GAP - time.monotonic()))
            query = urllib.parse.urlencode({"format": "json", "formatversion": 2, "maxlag": 5, **params})
            request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": archive.USER_AGENT})
            self.requests += 1
            try:
                with self.opener.open(request, timeout=12) as response:
                    data = response.read(archive.MAX_BYTES + 1)
            except urllib.error.HTTPError as error:
                raise archive.ArchiveStopped(f"publisher_http_{error.code}") from None
            except OSError:
                raise archive.ArchiveStopped("publisher_transport_failed") from None
            finally:
                self.last_request = time.monotonic()
            if len(data) > archive.MAX_BYTES:
                raise archive.ArchiveStopped("publisher_response_exceeded_bound")
        try:
            result = json.loads(data)
        except (UnicodeError, ValueError):
            raise archive.ArchiveStopped("publisher_invalid_json") from None
        if not isinstance(result, dict) or {"error", "errors", "warnings"}.intersection(result):
            raise archive.ArchiveStopped("publisher_api_error_or_warning")
        if not self.offline:
            archive.retain(path, data)
        return result, text.sha(data)


class ArchiveRows(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = ([], [])
        elif tag == "a" and self.cell is not None:
            href = dict(attrs).get("href", "")
            if href.startswith(("/wiki/", "./")):
                value = href.removeprefix("/wiki/").removeprefix("./")
                self.cell[1].append(urllib.parse.unquote(value).replace("_", " "))

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[0].append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append((text.canonical("".join(self.cell[0])), self.cell[1]))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


@dataclass(frozen=True)
class Task:
    article: str
    editor: str
    requested: str
    completed: str


def task_rows(document):
    parser = ArchiveRows()
    parser.feed(document)
    parser.close()
    for cells in parser.rows:
        if len(cells) != 6:
            continue
        requested, article, _, completed, editor, purpose = cells
        try:
            first, last = date.fromisoformat(requested[0]), date.fromisoformat(completed[0])
        except ValueError:
            continue
        if not START <= first <= last <= END:
            continue
        title = article[1][0] if len(article[1]) == 1 else ""
        users = sorted(
            {name.split(":", 1)[1] for name in editor[1] if name.startswith(("User:", "User talk:"))}
        )
        disposition = purpose[0].casefold()
        if (
            not title
            or "#" in title
            or text.sha(title.casefold().encode()) in DEVELOPMENT_TITLE_SHA256
            or len(users) != 1
            or not users[0]
            or any(word in disposition for word in ("declined", "withdrawn", "duplicate", "section"))
        ):
            continue
        yield Task(title, users[0], first.isoformat(), last.isoformat())


def plan(packet):
    body = packet["parse"]
    text.require(
        body["revid"] == INDEX_REVISION and body["title"] == INDEX, "Exact pinned GOCE archive required"
    )
    ordered = sorted(
        task_rows(body["text"]), key=lambda task: (task.requested, task.article.casefold(), task.editor)
    )
    chosen, articles, editors = [], set(), Counter()
    for task in ordered:
        title = task.article.casefold()
        if title in articles or editors[task.editor] == EDITOR_CAP:
            continue
        chosen.append(task)
        articles.add(title)
        editors[task.editor] += 1
    return chosen[:SAMPLE], len(ordered)


def safe_revision(item):
    return (
        "texthidden" not in item
        and "userhidden" not in item
        and "suppressed" not in item
        and type(item.get("userid")) is int
        and item["userid"] > 0
        and not item.get("user", "").casefold().endswith("bot")
        and not any(
            marker in tag.casefold()
            for tag in item["tags"]
            for marker in ("bot", "revert", "undo", "rollback")
        )
    )


def task_history(task, packet):
    lookup = packet["query"]
    pages = lookup["pages"]
    if len(pages) != 1 or pages[0].get("missing") or pages[0]["ns"] != 0:
        return None, [], "article_unavailable"
    page = pages[0]
    if text.sha(page["title"].casefold().encode()) in DEVELOPMENT_TITLE_SHA256:
        return page["pageid"], [], "prior_development_article"
    aliases = {
        item["to"]: item["from"] for field in ("normalized", "redirects") for item in lookup.get(field, [])
    }
    title = page["title"]
    chain = {title}
    while title in aliases and aliases[title] not in chain:
        title = aliases[title]
        chain.add(title)
    text.require(task.article in chain, "Official history is for a different archive task")
    if "continue" in packet:
        return page["pageid"], [], "editor_history_exceeded_bound"
    eligible = [
        row
        for row in page.get("revisions", [])
        if safe_revision(row)
        and row.get("user") == task.editor
        and row["parentid"] > 0
        and task.requested <= row["timestamp"][:10] <= task.completed
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (
            bool(re.search(r"\b(?:copy[- ]?edit\w*|c/e|ce)\b", row.get("comment", ""), re.I)),
            row["timestamp"],
            row["revid"],
        ),
        reverse=True,
    )
    return (
        page["pageid"],
        ranked[:PAIRS_PER_TASK],
        "awaiting_parent" if ranked else "no_exact_editor_revision",
    )


def paired_metadata(packet):
    return {
        row["revid"]: (page["pageid"], row)
        for page in packet["query"]["pages"]
        if page["ns"] == 0
        for row in page.get("revisions", [])
    }


def confirmed(page, task, selected, revisions):
    before, after = selected["parentid"], selected["revid"]
    if before not in revisions or after not in revisions:
        return False
    first_page, first = revisions[before]
    last_page, last = revisions[after]
    return (
        page == first_page == last_page
        and last["parentid"] == before
        and first["timestamp"] < last["timestamp"] == selected["timestamp"]
        and task.requested <= last["timestamp"][:10] <= task.completed
        and last.get("user") == task.editor
        and last.get("userid") == selected["userid"]
        and all(safe_revision(item) for item in (first, last))
    )


def lexical_witnesses(before, after, page, first, last):
    source = prose.article_prose(before, page, first)
    revised = prose.article_prose(after, page, last)
    windows = prose.retouches((*source[:2], True), (*revised[:2], True))
    return windows, [text.sha(body[0].encode()) for body in (source, revised)]


def collect(source):
    status, planned, catalog_size = "completed_fixed_sample", [], 0
    counts = Counter()
    try:
        index, index_hash = source.read(
            "archive.json", action="parse", oldid=INDEX_REVISION, prop="text|revid"
        )
        tasks, catalog_size = plan(index)
        text.require(len(tasks) == SAMPLE, "The pinned January GOCE roster must supply all planned tasks")
        planned = [
            {"task": asdict(task), "status": "not_run_after_stop", "index_sha256": index_hash, "pairs": []}
            for task in tasks
        ]
        histories = []
        for ordinal, (task, item) in enumerate(zip(tasks, planned, strict=True)):
            packet, digest = source.read(
                f"history-{ordinal:02d}.json",
                action="query",
                prop="revisions",
                titles=task.article,
                redirects=1,
                rvuser=task.editor,
                rvstart=f"{task.completed}T23:59:59Z",
                rvend=f"{task.requested}T00:00:00Z",
                rvlimit=HISTORY_CAP,
                rvprop="ids|timestamp|user|userid|comment|tags|flags",
            )
            page, selected, history_status = task_history(task, packet)
            item.update(status=history_status, history_sha256=digest)
            histories.append((page, selected))
            counts["histories_read"] += 1
        ids = sorted({row[key] for _, rows in histories for row in rows for key in ("parentid", "revid")})
        if ids:
            packet, metadata_hash = source.read(
                "pair-metadata.json",
                action="query",
                prop="revisions",
                revids="|".join(map(str, ids)),
                rvprop="ids|timestamp|user|userid|tags|flags",
            )
            revisions = paired_metadata(packet)
            seen_pages = set()
            for task, item, (page, candidates) in zip(tasks, planned, histories, strict=True):
                if page is None:
                    continue
                if page in seen_pages:
                    item["status"] = "duplicate_canonical_article"
                    continue
                seen_pages.add(page)
                if not candidates:
                    continue
                item.update(
                    status="pair_collection_incomplete", page_id=page, pair_metadata_sha256=metadata_hash
                )
                for row in candidates:
                    if not confirmed(page, task, row, revisions):
                        continue
                    first, last = row["parentid"], row["revid"]
                    renders = [
                        source.read(
                            f"render-{rev}.json",
                            action="parse",
                            oldid=rev,
                            prop="text|revid|categories",
                            disablelimitreport=1,
                        )
                        for rev in (first, last)
                    ]
                    windows, hashes = lexical_witnesses(renders[0][0], renders[1][0], page, first, last)
                    counts["exact_parent_pairs_rendered"] += 1
                    item["pairs"].append(
                        {
                            "page_id": page,
                            "before_revision": first,
                            "after_revision": last,
                            "render_sha256": [entry[1] for entry in renders],
                            "prose_sha256": hashes,
                            "windows": windows,
                        }
                    )
                    counts["mechanical_windows"] += len(windows)
                if any(pair["windows"] for pair in item["pairs"]):
                    item["status"] = "mechanical_candidate"
                elif item["pairs"]:
                    item["status"] = "no_lexical_witness"
                else:
                    item["status"] = "no_exact_safe_parent_pair"
    except archive.ArchiveStopped as error:
        status = str(error)
        for item in planned:
            if item["status"] == "awaiting_parent":
                item["status"] = "parent_check_not_run_after_stop"
    counts.update(item["status"] for item in planned)
    return {
        "schema": "chipmunk.text-goce-intake/228",
        "status": status,
        "archive_revision": INDEX_REVISION,
        "eligible_archive_tasks": catalog_size,
        "planned_pages": len(planned),
        "counts": dict(counts),
        "actual_new_requests": source.requests,
        "max_requests": MAX_REQUESTS,
        "rights_admitted_pages": 0,
        "model_calls": 0,
    }, planned


def rights_review(planned, decisions, exposures):
    windows = {
        prose.review_key(pair, window): window
        for item in planned
        for pair in item["pairs"]
        for window in pair["windows"]
    }
    reviewed, pages, stories, fragments = set(), set(), set(), set()
    reasons, approvals = Counter(), {}
    for record in decisions:
        key = tuple(
            record[field] for field in ("page_id", "before_revision", "after_revision", "window_sha256")
        )
        text.require(key in windows and key not in reviewed, "Review must bind one exact unrepeated witness")
        reviewed.add(key)
        state = record["decision"]
        if state in REJECTED:
            reasons[state] += 1
            continue
        text.require(
            state == "approved"
            and record["source_license"] == SOURCE_LICENSE
            and record["source_policy"] == SOURCE_POLICY
            and set(record["checks"]) == RIGHTS_CHECKS
            and all(value is True for value in record["checks"].values())
            and text.HEX.fullmatch(record["story_component"]),
            "Archived source, attribution, rights and editorial review required",
        )
        span = windows[key]
        original = text.fingerprints(span["before"]["excerpt"]) | text.fingerprints(span["after"]["excerpt"])
        approvals[key] = record["story_component"], original
    for key, (story, original) in sorted(approvals.items()):
        if original & exposures:
            reasons["prior_project_exposure"] += 1
        elif key[0] in pages or story in stories or original & fragments:
            reasons["duplicate_article_story_or_fragment"] += 1
        else:
            pages.add(key[0])
            stories.add(story)
            fragments.update(original)
    return {
        "reviewed_windows": len(reviewed),
        "pending_windows": len(windows) - len(reviewed),
        "rights_admitted_pages": len(pages),
        "rights_exclusions": dict(reasons),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--rights", type=Path)
    parser.add_argument("--exposure", type=Path)
    args = parser.parse_args()
    text.require(
        (args.rights is None and args.exposure is None) or (args.offline and args.rights and args.exposure),
        "Rights review requires complete offline evidence and prior-exposure input",
    )
    if args.offline:
        text.require(args.cache.is_dir(), "Offline replay requires a prior archive cache")
    archive.safe_cache(args.cache)
    report, planned = collect(OfficialArchive(args.cache, offline=args.offline))
    if planned:
        path = args.cache / "private-roster.json"
        data = json.dumps(planned, indent=2, ensure_ascii=True).encode()
        if args.offline and report["status"] == "completed_fixed_sample":
            text.require(
                text.read_file(path, archive.MAX_BYTES) == data, "Private GOCE roster replay differs"
            )
        elif not args.offline:
            archive.retain(path, data)
    if args.rights and report["status"] == "completed_fixed_sample":
        decisions = json.loads(text.read_file(args.rights, archive.MAX_BYTES))
        exposure = json.loads(text.read_file(args.exposure, 16_000_000))
        text.require(
            isinstance(exposure, list) and exposure and all(text.HEX.fullmatch(item) for item in exposure),
            "Exact prior-project fragment index required",
        )
        report.update(rights_review(planned, decisions, set(exposure)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
