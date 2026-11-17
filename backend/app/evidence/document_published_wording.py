"""Earlier published wording and its positions in a single uploaded report."""
from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher

from app.core.canonical import canonical_hash
from app.evidence import document_interval_locations as native
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import PASSAGE_INTERVALS_KEY
from app.evidence.document_scope import PUBLISHED_WORDING_COMMITMENT as COMMITMENT
from app.evidence.document_scope import PUBLISHED_WORDING_KEY as KEY

SCHEMA = "document-published-wording-review/1"
TITLE = "Earlier published wording"
SCOPE = ("This check compares the marked text with the cited U.S. Government Accountability Office highlight. "
         "Wording differences can result from ordinary editing, transcription or AI assistance. "
         "The comparison contributes zero weight to the overall AI assessment.")
ARCHIVE_LICENSE = "https://gov-report-data.github.io/#license"
FORMATS = {".txt": "plain", ".docx": "python-docx", ".pdf": "pymupdf"}
REPORT_ID = re.compile(r"[A-Z0-9][A-Z0-9-]{2,39}")
HASH = re.compile(r"[0-9a-f]{64}")


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def result_commitment(result):
    record = from_result(result)
    return canonical_hash(record) if record is not None else None


def archive_url(report_id):
    return f"https://www.gao.gov/products/{report_id}"


def eligible(extension, text, mapping, metadata, word_review):
    method = FORMATS.get(extension)
    if (method is None or method != metadata.get("method") or not 0 < len(text) <= 20_000 or not mapping
            or "\ufffd" in text or "\x00" in text):
        return False
    if method == "plain":
        return True
    if native.checked_end(method, text, mapping, metadata, word_review) != len(text):
        return False
    if method == "pymupdf":
        from app.evidence.document_revision_pdf import NATIVE_COMPARISON_MAX_PAGES, ordered_text

        return bool(ordered_text(text, mapping, page_limit=NATIVE_COMPARISON_MAX_PAGES, single_column=True))
    return True


def _places(method, mapping, start, end):
    # A deletion has no submitted characters. Locate the next surviving character.
    stop = max(end, start + 1)
    if method != "plain":
        return native.locate(method, mapping, start, stop)
    places = []
    for block in mapping:
        a, b = max(start, block["start"]), min(stop, block["end"])
        if a < b:
            ranges = clip_source_ranges(block["source_ranges"], a, b)
            if not ranges:
                return []
            places.append({"start": a, "end": b, "ranges": ranges})
    return places


def _words(text, span):
    start, end = span
    return [(match.group(), start + match.start(), start + match.end()) for match in re.finditer(r"\S+", text[start:end])]


def _exact_matches(text, record, intervals):
    ranges = (intervals or {}).get("ranges") or []
    if not ranges:
        return []
    earlier = _words(record["reference"]["text"], record["archive_range"])
    later = _words(text, record["submitted_range"])
    matcher = SequenceMatcher(None, [word for word, _, _ in later], [word for word, _, _ in earlier], autojunk=False)
    matches = []
    for i, j, count in matcher.get_matching_blocks():
        if count < 16:
            continue
        starts = {start: offset for offset, (_, start, _) in enumerate(later[i:i + count])}
        ends = {end: offset for offset, (_, _, end) in enumerate(later[i:i + count])}
        for row in ranges:
            first, last = starts.get(row["start"]), ends.get(row["end"])
            if first is None or last is None or last < first:
                continue
            # Require the complete highlight to appear in the dated transcript.
            if any(change["start"] < row["end"] and change["end"] > row["start"] for change in record["changes"]):
                continue
            matches.append({"start": row["start"], "end": row["end"],
                            "archive_range": [earlier[j + first][1], earlier[j + last][2]]})
    return matches


def _mapped_changes(method, text, mapping, changes):
    rows = []
    for (start, end), earlier in changes:
        places = _places(method, mapping, start, end)
        if places:
            rows.append({"start": start, "end": end, "submitted": text[start:end],
                         "archive_range": earlier, "locations": places})
    return rows


def capture(text, paragraphs, mapping, metadata, intervals, pack_hash, reports, index):
    from app.detectors.text.extract import Document
    from app.evidence import document_gao_archive

    doc = Document(text, paragraphs, metadata, mapping)
    result = document_gao_archive.compare(doc, metadata["original_sha256"], pack_hash, reports, index,
                                          resolve_competition=True)
    if result["status"] != "reference_bound":
        return None
    source = result["reference"]
    transcript = reports[source["id"]]["text"]
    record = {"schema": SCHEMA, "archive_sha256": pack_hash, "reference": {
                  "id": source["id"], "published_date": source["published_date"], "version_url": source["version_url"],
                  "source_record_sha256": source["source_record_sha256"], "transcript_sha256": source["transcript_sha256"],
                  "text": transcript},
              "submitted_range": source["submitted_range"], "archive_range": source["transcript_range"],
              "changes": [], "published_style_matches": [], "authorship_confidence_contribution": 0}
    record["changes"] = _mapped_changes(metadata["method"], text, mapping, (
        ([row["submitted"]["start"], row["submitted"]["end"]], [row["archive"]["start"], row["archive"]["end"]])
        for row in result["changes"]
    ))
    record["published_style_matches"] = _exact_matches(text, record, intervals)
    return record


def _bounded(value, limit, *, empty=False):
    return (isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value)
            and 0 <= value[0] <= value[1] <= limit and (empty or value[0] < value[1]))


def validate(document, expected):
    coverage = document.get("coverage") or {}
    source = document["source"]
    record = coverage.get(KEY)
    if record is None and expected is None and source.get(COMMITMENT) is None:
        return
    if (not isinstance(record, dict) or set(record) != {"schema", "archive_sha256", "reference", "submitted_range",
            "archive_range", "changes", "published_style_matches", "authorship_confidence_contribution"}
            or record["schema"] != SCHEMA or type(record["authorship_confidence_contribution"]) is not int
            or record["authorship_confidence_contribution"] != 0 or source.get("method") not in FORMATS.values()
            or source.get(COMMITMENT) != expected or canonical_hash(record) != expected
            or not HASH.fullmatch(record["archive_sha256"]) or not _bounded(record["submitted_range"], source["chars"])):
        raise ValueError("The published wording comparison differs from its recorded commitment or scope")
    ref = record["reference"]
    if (not isinstance(ref, dict) or set(ref) != {"id", "published_date", "version_url", "source_record_sha256",
                                                "transcript_sha256", "text"}
            or not REPORT_ID.fullmatch(ref["id"]) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", ref["published_date"])
            or not re.fullmatch(r"https://huggingface\.co/datasets/launch/gov_report/tree/[a-f0-9]{40}/data/gao_(train|valid|test)\.jsonl",
                                ref["version_url"])
            or not HASH.fullmatch(ref["source_record_sha256"]) or not 0 < len(ref["text"]) <= 20_000
            or hashlib.sha256(ref["text"].encode()).hexdigest() != ref["transcript_sha256"]
            or not _bounded(record["archive_range"], len(ref["text"]))):
        raise ValueError("The archived transcript differs from its recorded identity")
    start, end = record["submitted_range"]
    earlier_start, earlier_end = record["archive_range"]
    intervals = coverage.get(PASSAGE_INTERVALS_KEY) or {}
    proposed = {(row["start"], row["end"]) for row in intervals.get("ranges", [])}
    seen = set()
    for row in record["published_style_matches"]:
        pair = row["start"], row["end"]
        if (set(row) != {"start", "end", "archive_range"} or pair not in proposed or pair in seen
                or not _bounded(row["archive_range"], earlier_end) or row["archive_range"][0] < earlier_start
                or not start <= pair[0] < pair[1] <= end):
            raise ValueError("An archived match has an invalid writing highlight location")
        seen.add(pair)
    for row in record["changes"]:
        if (set(row) != {"start", "end", "submitted", "archive_range", "locations"}
                or not _bounded([row["start"], row["end"]], end, empty=True) or row["start"] < start
                or not _bounded(row["archive_range"], earlier_end, empty=True) or row["archive_range"][0] < earlier_start
                or not isinstance(row["submitted"], str) or len(row["submitted"]) != row["end"] - row["start"]
                or not row["locations"]):
            raise ValueError("An earlier wording difference has an invalid position")


def verify_source(document, text, mapping, metadata, word_review):
    record = (document.get("coverage") or {}).get(KEY)
    if record is None:
        return
    from app.evidence import document_gao_archive

    validate(document, document["source"].get(COMMITMENT))
    method = document["source"]["method"]
    extension = next(ext for ext, name in FORMATS.items() if method == name)
    intervals = document["coverage"].get(PASSAGE_INTERVALS_KEY)
    if (not eligible(extension, text, mapping, metadata, word_review)
            or _exact_matches(text, record, intervals) != record["published_style_matches"]):
        raise ValueError("The archived matches differ from the original wording")
    expected = document_gao_archive.replay_wording(text, record["reference"]["text"], record["submitted_range"],
                                                  record["archive_range"])
    if _mapped_changes(method, text, mapping, expected) != record["changes"]:
        raise ValueError("An earlier wording difference differs from the retained transcript or original locations")


def display_intervals(intervals, wording):
    excluded = {(row["start"], row["end"]) for row in wording["published_style_matches"]} if wording else set()
    return [row for row in intervals["ranges"] if (row["start"], row["end"]) not in excluded]
