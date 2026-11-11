"""Locate short editorial retouches directly in a fixed pre-2020 Wikinews sample."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path

if __package__:
    from . import text_short_insert_intake_v203 as intake
    from . import text_wikinews_archive_v205 as archive
else:
    import text_short_insert_intake_v203 as intake
    import text_wikinews_archive_v205 as archive

MONTHS = ("October 2018", "April 2019", "October 2019")
RENDER_PAIRS = (2, 2, 1, 1)
HISTORY_END = "2019-12-31T23:59:59Z"
CUTOFF = datetime(2020, 1, 1, tzinfo=UTC)
MAX_REQUESTS = len(MONTHS) * (1 + len(RENDER_PAIRS) + 2 * sum(RENDER_PAIRS))
QUOTES = frozenset('"“”„‟«»‹›')
REVIEW_CHECKS = {
    "licensed_original_prose",
    "third_party_imports_excluded",
    "quoted_material_excluded",
    "ordinary_editorial_retouch",
    "bot_or_revert_excluded",
}
REJECTION_REASONS = {
    "rights_unresolved",
    "unlicensed",
    "imported",
    "quoted",
    "bot_or_revert",
    "noneditorial",
}


def month_candidates(response, excluded):
    rows = response["query"]["search"]
    ids = [row["pageid"] for row in rows]
    intake.require(
        "continue" not in response
        and len(rows) == response["query"]["searchinfo"]["totalhits"]
        and len(set(ids)) == len(ids)
        and all(row["ns"] == 0 for row in rows)
        and all(type(page) is int and page > 0 for page in ids),
        "Expected a complete unique bounded monthly published search result",
    )
    return set(ids) - excluded


def sample(source, excluded):
    picked = set()
    roster, counts = [], {}
    for index, month in enumerate(MONTHS):
        response, digest = source.read(
            f"month-{index}.json",
            action="query",
            list="search",
            srsearch=f'deepcat:"{month}" incategory:"Published"',
            srnamespace=0,
            srlimit=100,
            srprop="",
        )
        pool = month_candidates(response, excluded | picked)
        ranked = sorted(pool, key=lambda page: intake.sha(f"v209:{month}:{page}".encode()))
        chosen = ranked[: len(RENDER_PAIRS)]
        picked.update(chosen)
        roster.extend((page, index, limit, digest) for page, limit in zip(chosen, RENDER_PAIRS, strict=False))
        counts[month] = {"eligible_catalog_pages": len(pool), "sampled": len(chosen)}
    return roster, counts


class EditorialParagraphs(intake.Paragraphs):
    marked = {"blockquote", "cite", "q"}

    def __init__(self):
        super().__init__()
        self.marked_parts = set()
        self.marked_quote = False

    def handle_starttag(self, tag, attrs):
        active = self.current is not None
        if active and tag in self.marked:
            self.marked_quote = True
        super().handle_starttag(tag, attrs)
        if not active and self.current is not None:
            self.marked_quote = False

    def handle_endtag(self, tag):
        size = len(self.parts)
        super().handle_endtag(tag)
        if len(self.parts) > size and (self.marked_quote or not QUOTES.isdisjoint(self.parts[-1])):
            self.marked_parts.add(size)


def article_prose(packet, page, revision):
    item = packet["parse"]
    intake.require(
        item["pageid"] == page
        and item["revid"] == revision
        and item.get("ns", 0) == 0
        and isinstance(item["text"], str),
        "Official render is not the selected article and revision",
    )
    parser = EditorialParagraphs()
    parser.feed(item["text"])
    parser.close()
    chunks, usable = [], []
    cursor = 0
    for i, paragraph in enumerate(parser.parts):
        if not paragraph.strip():
            continue
        if chunks:
            cursor += 2
        chunks.append(paragraph)
        end = cursor + len(paragraph)
        if i not in parser.marked_parts:
            usable.append((cursor, end))
        cursor = end
    published = any(category["category"].casefold() == "published" for category in item["categories"])
    return "\n\n".join(chunks), usable, published


def word_changes(before, after):
    left = list(intake.WORD.finditer(before))
    right = list(intake.WORD.finditer(after))
    opcodes = SequenceMatcher(
        None, [m.group() for m in left], [m.group() for m in right], autojunk=False
    ).get_opcodes()
    for previous, op, following in zip(opcodes, opcodes[1:], opcodes[2:]):
        tag, a, b, c, d = op
        if tag != "replace" or previous[0] != "equal" or following[0] != "equal":
            continue
        pad = max(0, intake.MIN_TARGET - min(b - a, d - c))
        start_pad, end_pad = pad // 2, (pad + 1) // 2
        if max(b - a, d - c) + pad <= intake.MAX_TARGET:
            yield (
                (left, a, b, a - previous[1], following[2] - b),
                (right, c, d, c - previous[3], following[4] - d),
                start_pad,
                end_pad,
            )


def exact_window(raw, ranges, changed, start_pad, end_pad):
    words, start, end, left_equal, right_equal = changed
    needed_left, needed_right = intake.FLANK + start_pad, intake.FLANK + end_pad
    if left_equal < needed_left or right_equal < needed_right:
        return None
    first, last = words[start - needed_left].start(), words[end + needed_right - 1].end()
    if not any(a <= first <= last <= b for a, b in ranges):
        return None
    window = intake.excerpt(raw, words, start, end, needed_left, needed_right, start_pad, end_pad)
    a, b = window.retouch
    changed_text = window.text[a:b]
    # Capitalization, punctuation, dates and digits alone are outside this editorial prose pilot.
    lexical = " ".join("".join(c for c in token.casefold() if c.isalpha()) for token in changed_text.split())
    return {
        "excerpt": window.text,
        "article_codepoints": [first, last],
        "target_codepoints": [first + i for i in window.target],
        "retouch_codepoints": [first + i for i in window.retouch],
        "target_words": window.target_words,
    }, lexical


def retouches(before, after):
    left, left_ranges, _ = before
    right, right_ranges, published = after
    if not published:
        return []
    windows = []
    for old, new, start_pad, end_pad in word_changes(left, right):
        first = exact_window(left, left_ranges, old, start_pad, end_pad)
        second = exact_window(right, right_ranges, new, start_pad, end_pad)
        if first and second and first[1].strip() and second[1].strip() and first[1] != second[1]:
            windows.append({"before": first[0], "after": second[0]})
    return windows


def pre2020_pairs(page, response):
    records = response["query"]["pages"][0]["revisions"]
    intake.require(
        all(intake.timestamp(record["timestamp"]) < CUTOFF for record in records),
        "Direct pilot must use revisions strictly before 2020",
    )
    flagged = {
        rev["revid"]
        for rev in records
        if any(
            any(marker in tag.casefold() for marker in ("bot", "revert", "rollback", "undo"))
            for tag in rev["tags"]
        )
    }
    return [pair for pair in archive.adjacent_pairs(page, response) if not (flagged & set(pair[:2]))]


def rank(pairs):
    def score(pair):
        left, right = pair[2]
        return sum(
            min(old[3], new[3]) >= intake.FLANK + start and min(old[4], new[4]) >= intake.FLANK + end
            for old, new, start, end in word_changes(left, right)
        )

    return sorted(pairs, key=score, reverse=True)


def pilot(source, excluded):
    counts = Counter()
    witnesses, monthly = [], {}
    status = "completed_direct_sample"
    try:
        roster, monthly = sample(source, excluded)
        counts["pages_sampled"] = len(roster)
        for page, month, pair_limit, catalog_digest in roster:
            history, history_hash = source.read(
                f"history-{page}.json",
                action="query",
                prop="revisions",
                pageids=page,
                rvstart=HISTORY_END,
                rvlimit=archive.REVISIONS,
                rvslots="main",
                rvprop="ids|timestamp|flags|tags|userid|sha1|content",
            )
            pairs = pre2020_pairs(page, history)
            counts["histories_verified"] += 1
            counts["mechanically_eligible_adjacent_pairs"] += len(pairs)
            page_witness = False
            for first, last, _ in rank(pairs)[:pair_limit]:
                renders, digests = [], []
                for revision in (first, last):
                    packet, digest = source.read(
                        f"render-{revision}.json",
                        action="parse",
                        oldid=revision,
                        prop="text|revid|categories",
                        disablelimitreport=1,
                    )
                    renders.append(article_prose(packet, page, revision))
                    digests.append(digest)
                windows = retouches(*renders)
                counts["rendered_pairs_reviewed"] += 1
                if not windows:
                    continue
                page_witness = True
                counts["mechanical_windows"] += len(windows)
                counts["pairs_with_mechanical_windows"] += 1
                witnesses.append(
                    {
                        "page_id": page,
                        "month": MONTHS[month],
                        "license_policy": intake.POLICY,
                        "before_revision": first,
                        "after_revision": last,
                        "catalog_sha256": catalog_digest,
                        "history_sha256": history_hash,
                        "before_render_sha256": digests[0],
                        "after_render_sha256": digests[1],
                        "prose_sha256": [intake.sha(text.encode()) for text, _, _ in renders],
                        "windows": windows,
                    }
                )
            counts["pages_with_mechanical_windows"] += page_witness
            counts["pages_completed"] += 1
    except archive.ArchiveStopped as error:
        status = str(error)
    intake.require(source.requests <= MAX_REQUESTS, "Direct archive request bound changed")
    return {
        "schema": "chipmunk.text-wikinews-direct-pilot/209",
        "status": status,
        "months": monthly,
        "counts": dict(counts),
        "max_requests": MAX_REQUESTS,
        "actual_new_requests": source.requests,
        "rights_admitted_pages": 0,
        "model_calls": 0,
    }, witnesses


def review_key(witness, window):
    digest = intake.sha(json.dumps(window, sort_keys=True, ensure_ascii=True).encode())
    return witness["page_id"], witness["before_revision"], witness["after_revision"], digest


def rights_admission(witnesses, reviews, exposure):
    possible = {review_key(witness, window): window for witness in witnesses for window in witness["windows"]}
    dispositions, seen = Counter(), set()
    approvals = {}
    for review in reviews:
        key = tuple(
            review[field] for field in ("page_id", "before_revision", "after_revision", "window_sha256")
        )
        intake.require(key in possible and key not in seen, "Rights review must bind one exact source window")
        seen.add(key)
        if review["decision"] in REJECTION_REASONS:
            dispositions[review["decision"]] += 1
            continue
        intake.require(
            review["decision"] == "approved"
            and review["license"] == intake.LICENSE
            and review["license_policy"] == intake.POLICY
            and set(review["checks"]) == REVIEW_CHECKS
            and all(value is True for value in review["checks"].values())
            and intake.HEX.fullmatch(review["story_component"]),
            "Rights approval requires original licensed prose and reviewed editorial lineage",
        )
        window = possible[key]
        fragments = intake.fingerprints(window["before"]["excerpt"]) | intake.fingerprints(
            window["after"]["excerpt"]
        )
        if fragments & exposure:
            dispositions["prior_project_exposure"] += 1
            continue
        approvals[key] = (review["story_component"], fragments)
    admitted, stories, consumed = set(), set(), set()
    for key, (story, fragments) in sorted(approvals.items()):
        if key[0] in admitted or story in stories or fragments & consumed:
            dispositions["duplicate_article_story_or_fragment"] += 1
            continue
        admitted.add(key[0])
        stories.add(story)
        consumed.update(fragments)
    return {
        "rights_reviewed_windows": len(seen),
        "rights_pending_windows": len(possible) - len(seen),
        "rights_admitted_pages": len(admitted),
        "rights_exclusions": dict(dispositions),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "train", type=Path, help="Pinned IteraTeR train JSONL used only to exclude earlier article IDs"
    )
    parser.add_argument("cache", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--rights", type=Path, help="Private per-window rights review")
    parser.add_argument("--exposure", type=Path, help="Private already-consumed project fragment hashes")
    args = parser.parse_args()
    intake.require(
        (args.rights is None and args.exposure is None) or (args.offline and args.rights and args.exposure),
        "Rights admission requires complete offline rights and exposure inputs",
    )
    if args.offline:
        intake.require(args.cache.is_dir(), "Offline replay requires a prior direct archive cache")
    archive.safe_cache(args.cache)
    excluded = {int(row["doc_id"]) for row in intake.train_rows(args.train)}
    report, witnesses = pilot(archive.Archive(args.cache, offline=args.offline), excluded)
    if report["status"] == "completed_direct_sample":
        path = args.cache / "private-candidates.json"
        data = json.dumps(witnesses, indent=2, ensure_ascii=True).encode()
        if args.offline:
            intake.require(
                intake.read_file(path, archive.MAX_BYTES) == data, "Private source window replay differs"
            )
        else:
            archive.retain(path, data)
        if args.rights:
            exposure = json.loads(intake.read_file(args.exposure, 16_000_000))
            intake.require(
                isinstance(exposure, list)
                and exposure
                and all(intake.HEX.fullmatch(item) for item in exposure),
                "Exact private exposure hash list required",
            )
            review = json.loads(intake.read_file(args.rights, archive.MAX_BYTES))
            report.update(rights_admission(witnesses, review, set(exposure)))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
