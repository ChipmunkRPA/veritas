"""Exact local wording comparison against pinned GAO report highlights."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from array import array
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from app.detectors.text.extract import Document, text_hash
from app.evidence import document_archive
from app.evidence.document_revision import tokens

SCHEMA = "chipmunk.document-archive-comparison/2"
PACK_SCHEMA = "chipmunk.gao-highlights-pack/1"
REVISION = "32feeaede49fed993aef070bc4da09263fd0429a"
PUBLISHER = f"https://huggingface.co/datasets/launch/gov_report/tree/{REVISION}/data"
GAO = "https://www.gao.gov/products"
LICENSE = "https://gov-report-data.github.io/#license"
PACK_LIMIT = 48 * 1024 * 1024
REPORT_COUNT = 12228
SPLITS = {"train", "valid", "test"}
HASH = re.compile(r"[0-9a-f]{64}")
REPORT_ID = re.compile(r"[A-Z0-9][A-Z0-9-]{2,39}")
DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
WINDOW = 8
SLOT_COUNT = 1 << 24
SLOT_MASK = SLOT_COUNT - 1
MAX_INDEX_KEYS = SLOT_COUNT // 2
SCOPE = ("Compares wording in the bounded passage of a matching U.S. Government Accountability Office report highlight "
         "as transcribed in the pinned GovReport dataset. The comparison excludes recommendations and text outside matched boundaries.")
INTERPRETATION = ("Differences describe wording relative to the cited source. Ordinary edits can produce the same evidence. "
                  "This comparison carries zero AI-authorship weight.")
_CHARACTERS = str.maketrans({"’": "'", "“": '"', "”": '"'})


def normalize(token):
    return unicodedata.normalize("NFKC", token).casefold().translate(_CHARACTERS)


def token_stream(text):
    return [(normalize(value), start, end) for value, start, end in tokens(text, [(0, len(text))])]


def token_values(stream):
    return [value for value, _, _ in stream]


def _window(values, start):
    # Only use this key to find candidates. The exact terms are verified on lookup.
    digest = hashlib.blake2b("\0".join(values[start:start + WINDOW]).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") or 1


def build_index(reports):
    identities = list(reports)
    keys = array("Q", [0]) * SLOT_COUNT
    locations = array("I", [0]) * SLOT_COUNT
    used = 0
    for index_number, name in enumerate(identities, 1):
        values = token_values(token_stream(reports[name]["text"]))
        for start in range(len(values) - WINDOW + 1):
            key = _window(values, start)
            slot = key & SLOT_MASK
            while keys[slot] and keys[slot] != key:
                slot = (slot + 1) & SLOT_MASK
            if keys[slot] == key:
                locations[slot] = 0
            else:
                used += 1
                if used > MAX_INDEX_KEYS:
                    raise document_archive.ArchiveUnavailable("The GAO pack exceeds the local index capacity.")
                keys[slot] = key
                locations[slot] = (index_number << 16) + start
    return keys, locations, identities


@lru_cache(maxsize=1)
def library(path: Path, sha256: str):
    try:
        if not isinstance(sha256, str) or not HASH.fullmatch(sha256):
            raise ValueError("A pinned GAO pack hash is required")
        with path.open("rb") as stream:
            raw = stream.read(PACK_LIMIT + 1)
        if not 0 < len(raw) <= PACK_LIMIT or hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError("GAO pack bytes differ")
        data = json.loads(raw)
        reports = data["reports"]
        if data["schema"] != PACK_SCHEMA or data["revision"] != REVISION or len(reports) != REPORT_COUNT:
            raise ValueError("Unknown GAO pack format")
        if any(not REPORT_ID.fullmatch(key) or not 0 <= len(row["text"]) <= 20_000
               or row["split"] not in SPLITS or not DATE.fullmatch(row["published_date"])
               or not HASH.fullmatch(row["source_record_sha256"]) for key, row in reports.items()):
            raise ValueError("Invalid GAO report metadata")
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:
        raise document_archive.ArchiveUnavailable("The local GAO archive pack could not be verified.") from error
    return reports, build_index(reports)


def _votes(query, packed_index, reports):
    keys, locations, identities = packed_index
    votes = defaultdict(list)
    cached = {}
    for start in range(len(query) - WINDOW + 1):
        key = _window(query, start)
        slot = key & SLOT_MASK
        while keys[slot] and keys[slot] != key:
            slot = (slot + 1) & SLOT_MASK
        item = locations[slot]
        if not item:
            continue
        ref_index, position = item >> 16, item & 0xffff
        key = identities[ref_index - 1]
        if key not in cached:
            cached[key] = token_values(token_stream(reports[key]["text"]))
        if query[start:start + WINDOW] == cached[key][position:position + WINDOW]:
            votes[key].append((start, position))
            if len(votes) > 1:
                break
    return votes, cached


def _span(stream, start, end):
    point = stream[start][1] if start < len(stream) else stream[-1][2]
    return [point, stream[end - 1][2] if start < end else point]


def _excerpt(text, span, changed):
    start, end = span
    return {"start": start, "end": end, "text": text[start:end], "changed_tokens": [span] if changed and start < end else []}


def _context_ranges(doc, changes):
    from app.detectors.text.pipeline import segment_within

    ranges = []
    previous = None
    for unit in segment_within(doc):
        if not any(max(lo, unit.start) < min(hi, unit.end) or (lo == hi and unit.start < lo < unit.end)
                   for lo, hi in changes):
            continue
        if previous is not None and previous.idx + 1 == unit.idx and previous.paragraph == unit.paragraph:
            ranges[-1][1] = unit.end
        else:
            ranges.append([unit.start, unit.end])
        previous = unit
    return ranges


def compare(doc: Document, upload_sha256: str, pack_sha256: str, reports, packed_index):
    text = doc.text
    output = {
        "schema": SCHEMA, "status": "out_of_scope", "scope": SCOPE, "interpretation": INTERPRETATION,
        "coordinate_space": "extracted_text_unicode_codepoint", "layout": document_archive.SUBMITTED_PARAGRAPHS,
        "authorship_confidence_contribution": 0, "submitted_sha256": upload_sha256,
        "extracted_text_sha256": text_hash(text),
        "library": {"title": "GAO report highlights in GovReport", "publisher": "U.S. Government Accountability Office",
                    "url": "https://gov-report-data.github.io/", "license": "CC BY 4.0",
                    "license_url": LICENSE, "revision": REVISION, "sha256": pack_sha256,
                    "indexed_excerpts": len(reports)},
        "coverage": {"body_paragraphs": len(doc.paragraphs), "body_words": 0, "matched_words": 0},
        "reference": None, "aligned_paragraphs": [], "changes": [], "submitted_sentence_ranges": [],
    }
    if len(text) > 20_000 or any(line.lstrip().startswith(">") for line in text.splitlines()):
        return output
    stream = token_stream(text)
    if not 64 <= len(stream) <= 1400:
        return output
    words = token_values(stream)
    output["coverage"]["body_words"] = len(words)
    output["status"] = "no_qualified_reference"
    votes, cached = _votes(words, packed_index, reports)
    if len(votes) > 1:
        output["status"] = "ambiguous_reference"
        return output
    if not votes:
        return output
    source_id, anchors = next(iter(votes.items()))
    support = len({i for start, _ in anchors for i in range(start, start + WINDOW)})
    if support < 40 or support * 100 < len(words) * 55 or anchors[-1][0] - anchors[0][0] < 24:
        return output
    opcodes = SequenceMatcher(None, cached[source_id], words, autojunk=False).get_opcodes()
    aligned = {(query, ref) for query, ref in anchors if any(
        tag == "equal" and c <= query <= d - WINDOW and ref == a + query - c for tag, a, _, c, d in opcodes)}
    if len(aligned) != len(anchors):
        return output
    source = reports[source_id]
    reference = source["text"]
    source_stream = token_stream(reference)
    first, last = anchors[0][0] + WINDOW, anchors[-1][0]
    changes = [(a, b, c, d) for tag, a, b, c, d in opcodes if tag != "equal" and first <= c <= d <= last
               and max(b - a, d - c) <= 96]
    if not changes and cached[source_id] != words:
        return output
    if len(changes) > 80:
        return output
    for a, b, c, d in changes:
        submitted, earlier = _span(stream, c, d), _span(source_stream, a, b)
        output["changes"].append({"submitted_paragraph": 1 + next(
            (i for i, (start, stop) in enumerate(doc.paragraphs) if start <= submitted[0] <= stop), 0),
            "archive_paragraph": None, "alignment": "bounded_pair",
            "submitted": _excerpt(text, submitted, c < d), "archive": _excerpt(reference, earlier, a < b)})
    output["reference"] = {
        "id": source_id, "url": f"{GAO}/{source_id}", "version_url": f"{PUBLISHER}/gao_{source['split']}.jsonl",
        "volume": "GAO highlights", "document": source_id, "published_date": source["published_date"],
        "source_record_sha256": source["source_record_sha256"], "transcript_sha256": text_hash(reference),
        "submitted_range": _span(stream, anchors[0][0], anchors[-1][0] + WINDOW),
        "transcript_range": _span(source_stream, anchors[0][1], anchors[-1][1] + WINDOW),
    }
    output["coverage"]["matched_words"] = support
    if output["changes"]:
        output["submitted_sentence_ranges"] = _context_ranges(doc, [
            (item["submitted"]["start"], item["submitted"]["end"]) for item in output["changes"]])
    output["status"] = "reference_bound"
    return output
