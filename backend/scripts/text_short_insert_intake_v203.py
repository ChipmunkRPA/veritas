"""Prepare a Wikinews human-retouch roster without running any detector or generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from tempfile import gettempdir

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "backend/data/independent-evaluation"
REVISION = "3b0bdabb090d04062ebc17e54ac889a64f5cb791"
TRAIN_SHA256 = "3f60e9989879b7a51b994edbe4536a5bd30556670f47b6677815bf0ed45c0acc"
TRAIN_BYTES = 2_088_096
POLICY = "https://en.wikinews.org/w/index.php?title=Wikinews:Copyright&oldid=4991371"
LICENSE = "CC-BY-2.5"
EARLIEST = datetime(2005, 9, 26, tzinfo=UTC)
CUTOFF = datetime(2021, 8, 1, tzinfo=UTC)
EDIT_INTENTS = {"clarity", "fluency", "style"}
POINT_DYADS, EVALUATION_DYADS = 8, 16
MIN_TARGET, MAX_TARGET, FLANK = 8, 16, 12
MIN_CONTEXT, MAX_CONTEXT = 16, 24
WORD = re.compile(r"\S+")
HEX = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(text):
    return " ".join(text.split())


def read_file(path, limit):
    require(path.is_file() and not path.is_symlink(), "Expected a direct intake input")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "Intake input exceeded its byte bound")
    return data


def train_rows(path):
    data = read_file(path, TRAIN_BYTES)
    require(len(data) == TRAIN_BYTES and sha(data) == TRAIN_SHA256, "Pinned publisher train bytes differ")
    rows = [json.loads(line) for line in data.splitlines()]
    inventory = Counter(row.get("domain") for row in rows)
    require(inventory == {"news": 169, "wiki": 154, "arxiv": 125, None: 33}, "Publisher train roster changed")
    news = [row for row in rows if row.get("domain") == "news"]
    require(
        all(re.fullmatch(r"[1-9][0-9]{3,6}", row["doc_id"]) for row in news)
        and len({row["doc_id"] for row in news}) == 156,
        "Explicitly labeled Wikinews roster changed",
    )
    return news


@dataclass(frozen=True)
class Excerpt:
    text: str
    article_offset: int
    target: tuple[int, int]
    retouch: tuple[int, int]
    target_words: int


@dataclass(frozen=True)
class Candidate:
    page_id: int
    depth: int
    original: Excerpt
    revised: Excerpt


def excerpt(raw, words, edit_start, edit_end, left, right, left_pad, right_pad):
    target_start, target_end = edit_start - left_pad, edit_end + right_pad
    start = words[edit_start - left].start()
    stop = words[edit_end + right - 1].end()
    return Excerpt(
        raw[start:stop],
        start,
        (words[target_start].start() - start, words[target_end - 1].end() - start),
        (words[edit_start].start() - start, words[edit_end - 1].end() - start),
        target_end - target_start,
    )


def proposed_retouched_excerpts(row):
    before, after = row["before_revision"], row["after_revision"]
    original, revised = [list(WORD.finditer(text)) for text in (before, after)]
    words_a = [m.group() for m in original]
    words_b = [m.group() for m in revised]
    opcodes = SequenceMatcher(None, words_a, words_b, autojunk=False).get_opcodes()
    edits = [
        edit for edit in row["edit_actions"] if edit["type"] == "R" and edit["major_intent"] in EDIT_INTENTS
    ]
    for previous, op, following in zip(opcodes, opcodes[1:], opcodes[2:]):
        tag, a, b, c, d = op
        if tag != "replace" or previous[0] != "equal" or following[0] != "equal":
            continue
        left, right = min(a - previous[1], MAX_CONTEXT), min(following[2] - b, MAX_CONTEXT)
        if min(left, right) < MIN_CONTEXT:
            continue
        start, end = original[a].start(), original[b - 1].end()
        if not any(
            start <= edit["start_char_pos"] < edit["end_char_pos"] <= end
            and before[edit["start_char_pos"] : edit["end_char_pos"]] == edit["before"]
            and canonical(edit["after"]) in " ".join(words_b[c:d])
            and canonical(edit["after"])
            for edit in edits
        ):
            continue
        padding = max(0, MIN_TARGET - min(b - a, d - c))
        left_pad, right_pad = padding // 2, (padding + 1) // 2
        if max(b - a, d - c) + padding > MAX_TARGET or min(left - left_pad, right - right_pad) < FLANK:
            continue
        yield Candidate(
            int(row["doc_id"]),
            row["revision_depth"],
            excerpt(before, original, a, b, left, right, left_pad, right_pad),
            excerpt(after, revised, c, d, left, right, left_pad, right_pad),
        )


def all_candidates(rows):
    result = defaultdict(list)
    for row in sorted(rows, key=lambda item: (int(item["doc_id"]), item["revision_depth"])):
        result[int(row["doc_id"])].extend(proposed_retouched_excerpts(row))
    return result


class Paragraphs(HTMLParser):
    """Read visible paragraph prose, excluding markup blocks and citation markers."""

    blocked = {
        "aside",
        "blockquote",
        "figcaption",
        "figure",
        "footer",
        "nav",
        "script",
        "style",
        "sup",
        "table",
    }
    void = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.parts = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "br" and self.current is not None:
            self.current.append(" ")
        if tag in self.void:
            return
        properties = dict(attrs)
        classes = (properties.get("class") or "").split()
        excluded = tag in self.blocked or bool(
            {"references", "metadata", "mw-editsection"}.intersection(classes)
        )
        self.stack.append((tag, excluded))
        if tag == "p" and not any(ignored for _, ignored in self.stack):
            self.current = []

    def handle_endtag(self, tag):
        matched = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i][0] == tag), None)
        if matched is None:
            return
        if tag == "p" and self.current is not None:
            self.parts.append("".join(self.current))
            self.current = None
        del self.stack[matched:]

    def handle_data(self, data):
        if self.current is not None and not any(ignored for _, ignored in self.stack):
            self.current.append(data)


def visible_prose(document):
    parser = Paragraphs()
    parser.feed(document)
    parser.close()
    return canonical(" ".join(parser.parts))


def unique_prose_match(source, official):
    # Pad both sides to require entire words, including punctuation as recorded.
    excerpt, article = f" {canonical(source)} ", f" {official} "
    start = article.find(excerpt)
    return start >= 0 and article.find(excerpt, start + 1) < 0


def timestamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def revision_prose(render, page_id, revision_id):
    document = render["parse"]
    require(
        document["pageid"] == page_id
        and document["revid"] == revision_id
        and document.get("ns", 0) == 0
        and isinstance(document["text"], str),
        "Official rendered article does not match its committed page and revision",
    )
    return visible_prose(document["text"])


def rights_lineage(folder, audit):
    page_id, before, after = (audit[key] for key in ("page_id", "before_revision", "after_revision"))
    require(
        all(type(value) is int and value > 0 for value in (page_id, before, after)) and before != after,
        "Invalid official page/revision identifiers",
    )
    require(
        audit["license"] == LICENSE
        and audit["license_policy"] == POLICY
        and audit["wikinews_staff_or_contributor_prose"] is True
        and audit["third_party_material_excluded"] is True
        and audit["ordinary_editorial_retouch"] is True
        and audit["not_known_bot_or_revert"] is True
        and HEX.fullmatch(audit["story_component"]),
        "Exact private source and editorial eligibility review required",
    )

    def checked(name, field):
        data = read_file(folder / name, 2_000_000)
        require(sha(data) == audit[field], "Official reply differs from the privately audited copy")
        return json.loads(data)

    history = checked(f"history-{page_id}.json", "history_sha256")
    pages = history["query"]["pages"]
    require(
        len(pages) == 1
        and pages[0]["pageid"] == page_id
        and pages[0]["ns"] == 0
        and pages[0].get("pagelanguage", "en") == "en",
        "Expected the original official English Wikinews article",
    )
    records = pages[0]["revisions"]
    revisions = {row["revid"]: row for row in records}
    require(
        len(revisions) == len(records) and before in revisions and after in revisions, "Revision pair missing"
    )
    first, last = revisions[before], revisions[after]
    require(
        last["parentid"] == before
        and EARLIEST <= timestamp(first["timestamp"]) < timestamp(last["timestamp"]) < CUTOFF
        and all("texthidden" not in row and "suppressed" not in row for row in (first, last))
        and type(last.get("userid")) is int
        and last["userid"] > 0
        and not {"mw-rollback", "mw-undo", "mw-reverted", "mw-manual-revert"}.intersection(last["tags"]),
        "Adjacent visible pre-cutoff revisions with a registered editorial change required",
    )
    return tuple(
        revision_prose(checked(f"render-{revision}.json", f"{side}_render_sha256"), page_id, revision)
        for side, revision in (("before", before), ("after", after))
    )


def fingerprints(value):
    tokens = canonical(value).casefold().split(" ")
    return {sha(" ".join(tokens[i : i + 12]).encode()) for i in range(len(tokens) - 11)}


def admitted_sources(candidates, official_folder, reviews, exposure):
    admitted = {}
    dispositions = Counter()
    seen_reviews = set()
    for review in reviews:
        page_id = review["page_id"]
        pair = page_id, review["before_revision"], review["after_revision"]
        require(pair not in seen_reviews, "Duplicate private rights review")
        seen_reviews.add(pair)
        if not candidates.get(page_id):
            dispositions["no_constructible_retouch"] += 1
            continue
        before, after = rights_lineage(official_folder, review)
        matches = [
            item
            for item in candidates[page_id]
            if unique_prose_match(item.original.text, before) and unique_prose_match(item.revised.text, after)
        ]
        if not matches:
            dispositions["revision_prose_not_bound"] += 1
            continue
        choice = matches[0]
        sig = fingerprints(choice.original.text) | fingerprints(choice.revised.text)
        if sig & exposure:
            dispositions["previous_project_exposure"] += 1
            continue
        # Preserve all private commitments even when another reviewed revision of this page already qualified.
        key = (choice.depth, choice.original.article_offset, review["after_revision"])
        if page_id not in admitted or key < admitted[page_id][0]:
            admitted[page_id] = (key, choice, review, sig)
    return admitted, dispositions


def select(admitted):
    ordered = sorted(admitted.values(), key=lambda entry: sha(f"v203:{entry[1].page_id}".encode()))
    unique, fragments, stories = [], set(), set()
    for item in ordered:
        _, _, review, sig = item
        if sig & fragments or review["story_component"] in stories:
            continue
        unique.append(item)
        fragments.update(sig)
        stories.add(review["story_component"])
    dyads = []
    while unique:
        _, source, review, _ = unique.pop(0)
        a = source.original.target_words, source.revised.target_words
        match = next(
            (
                i
                for i, item in enumerate(unique)
                if abs(a[0] - item[1].original.target_words) <= 2
                and abs(a[1] - item[1].revised.target_words) <= 2
            ),
            None,
        )
        if match is None:
            continue
        _, donor, donor_review, _ = unique.pop(match)
        lineage_keys = (
            "page_id",
            "before_revision",
            "after_revision",
            "history_sha256",
            "before_render_sha256",
            "after_render_sha256",
            "story_component",
            "license",
            "license_policy",
        )
        lineage = [{key: evidence[key] for key in lineage_keys} for evidence in (review, donor_review)]
        dyads.append({"articles": [asdict(source), asdict(donor)], "lineage": lineage})
    ready = len(dyads) >= POINT_DYADS + EVALUATION_DYADS
    if ready:
        for i, dyad in enumerate(dyads[: POINT_DYADS + EVALUATION_DYADS]):
            dyad["role"] = "point" if i < POINT_DYADS else "evaluation"
    return dyads, ready


def save_private(path, roster):
    private_roots = (PRIVATE.resolve(), Path(gettempdir()).resolve(), Path("/private/tmp"))
    require(
        any(path.resolve().is_relative_to(base) for base in private_roots),
        "Place a private roster in the ignored evaluation directory or a system temporary directory",
    )
    data = json.dumps(roster, ensure_ascii=True).encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(data)
    return sha(data)


def inventory(rows, candidates):
    return {
        "schema": "chipmunk.text-short-insert-intake/203",
        "publisher_train_sha256": TRAIN_SHA256,
        "explicit_wikinews_rows": len(rows),
        "explicit_wikinews_pages": len(candidates),
        "candidate_retouched_windows": sum(map(len, candidates.values())),
        "pages_with_retouched_window": sum(map(bool, candidates.values())),
        "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path, help="Exact pinned publisher train JSONL file")
    parser.add_argument("--official", type=Path, help="Directory of retained official JSON responses")
    parser.add_argument("--rights", type=Path, help="Private rights review JSON array")
    parser.add_argument("--exposure", type=Path, help="Private JSON array of project-source 12-word hashes")
    parser.add_argument(
        "--private-roster", type=Path, help="Optional new private output if enough dyads qualify"
    )
    args = parser.parse_args()
    rows = train_rows(args.train)
    candidates = all_candidates(rows)
    report = inventory(rows, candidates)
    if args.official is None and args.rights is None and args.exposure is None:
        require(args.private_roster is None, "Private roster requires official rights and exposure inputs")
        report["status"] = "publisher_inventory_only"
    else:
        require(
            all((args.official, args.rights, args.exposure)), "All three private evidence inputs required"
        )
        exposure = json.loads(read_file(args.exposure, 16_000_000))
        require(
            isinstance(exposure, list) and exposure and all(HEX.fullmatch(v) for v in exposure),
            "Exact private exposure hash list required",
        )
        reviews = json.loads(read_file(args.rights, 2_000_000))
        admitted, excluded = admitted_sources(candidates, args.official, reviews, set(exposure))
        dyads, ready = select(admitted)
        report.update(
            status="admitted" if ready else "insufficient_admitted_dyads",
            verified_articles=len(admitted),
            constructed_dyads=len(dyads),
            exclusions=dict(excluded),
            required_point_dyads=POINT_DYADS,
            required_evaluation_dyads=EVALUATION_DYADS,
        )
        if ready and args.private_roster is not None:
            report["private_roster_sha256"] = save_private(
                args.private_roster, dyads[: POINT_DYADS + EVALUATION_DYADS]
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
