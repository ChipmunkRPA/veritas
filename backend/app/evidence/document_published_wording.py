"""Earlier published wording and its positions in a single uploaded report."""
from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from functools import partial
from itertools import pairwise
from unicodedata import normalize

from app.core.canonical import canonical_hash
from app.evidence import document_interval_locations as native
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import PASSAGE_INTERVALS_KEY
from app.evidence.document_scope import PUBLISHED_WORDING_COMMITMENT as COMMITMENT
from app.evidence.document_scope import PUBLISHED_WORDING_KEY as KEY

SCHEMA = "document-published-wording-review/1"
PDF_COLUMNS_SCHEMA = "document-published-wording-review/2"
SENTENCES_SCHEMA = "document-published-wording-review/3"
TITLE = "Earlier published wording"
SCOPE = ("This check compares the marked text with the cited U.S. Government Accountability Office highlight. "
         "Wording differences can result from ordinary editing, transcription or AI assistance. "
         "The comparison contributes zero weight to the overall AI assessment.")
PDF_COLUMNS_SCOPE = ("Only the cited spans in the listed columns were compared. "
                     "The check sets aside bullets, matching wrapped words, and standalone headings without an earlier counterpart.")
ARCHIVE_LICENSE = "https://gov-report-data.github.io/#license"
FORMATS = {".txt": "plain", ".docx": "python-docx", ".pdf": "pymupdf"}
REPORT_ID = re.compile(r"[A-Z0-9][A-Z0-9-]{2,39}")
HASH = re.compile(r"[0-9a-f]{64}")
_TYPOGRAPHY = str.maketrans({**{c: bytes([c]).decode("cp1252", errors="replace") for c in range(0x80, 0xA0)},
                             "’": "'", "‘": "'", "“": '"', "”": '"', "•": None, "◦": None, "▪": None})


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def result_commitment(result):
    record = from_result(result)
    return canonical_hash(record) if record is not None else None


def archive_url(report_id):
    return f"https://www.gao.gov/products/{report_id}"


def report_view(record, method):
    if record is None:
        return None
    columns = record["schema"] == PDF_COLUMNS_SCHEMA
    sections = record["sections"] if columns else [record]
    return {"title": TITLE, "scope": SCOPE + (" " + PDF_COLUMNS_SCOPE if columns else ""),
            "license_url": ARCHIVE_LICENSE, "method": method, "published_style_matches": record["published_style_matches"],
            "sections": [{**section, "source_url": archive_url(section["reference"]["id"])} for section in sections]}


def _pdf_native_order(text, mapping):
    from app.evidence.document_revision_pdf import NATIVE_COMPARISON_MAX_PAGES, ordered_text

    return bool(ordered_text(text, mapping, page_limit=NATIVE_COMPARISON_MAX_PAGES, single_column=True))


def _pdf_columns(text, mapping):
    from app.evidence.document_pdf_columns import views
    from app.evidence.document_revision_pdf import NATIVE_COMPARISON_MAX_PAGES

    return views(text, mapping, NATIVE_COMPARISON_MAX_PAGES)


def _mapped_source(extension, text, mapping, metadata, word_review):
    method = FORMATS.get(extension)
    if (method is None or method != metadata.get("method") or not 0 < len(text) <= 20_000 or not mapping
            or "\ufffd" in text or "\x00" in text):
        return False
    return method == "plain" or native.checked_end(method, text, mapping, metadata, word_review) == len(text)


def eligible(extension, text, mapping, metadata, word_review, *, pdf_columns=False):
    if not _mapped_source(extension, text, mapping, metadata, word_review):
        return False
    return (extension != ".pdf" or _pdf_native_order(text, mapping)
            or (pdf_columns and any(_pdf_columns(text, mapping))))


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


def _mapped_changes(text, changes, places_at):
    rows = []
    for (start, end), earlier in changes:
        places = places_at(start, end)
        if places:
            rows.append({"start": start, "end": end, "submitted": text[start:end],
                         "archive_range": earlier, "locations": places})
    return rows


def _sentence_rows(doc, changes, places_at):
    from app.evidence.document_gao_archive import sentence_support

    spans = [(change["start"], change["end"]) for change in changes]
    return [{"sentence": unit.idx + 1, "start": unit.start, "end": unit.end,
             "text": doc.text[unit.start:unit.end], "locations": places, "differences": ids}
            for unit, ids in sentence_support(doc, spans) if (places := places_at(unit.start, unit.end))]


def _match(result, reports):
    source = result["reference"]
    transcript = reports[source["id"]]["text"]
    return {"reference": {"id": source["id"], "published_date": source["published_date"],
                          "version_url": source["version_url"], "source_record_sha256": source["source_record_sha256"],
                          "transcript_sha256": source["transcript_sha256"], "text": transcript},
            "submitted_range": source["submitted_range"], "archive_range": source["transcript_range"]}


def _edits(result):
    return (([row["submitted"]["start"], row["submitted"]["end"]], [row["archive"]["start"], row["archive"]["end"]])
            for row in result["changes"])


def _typography(text):
    clean = " ".join(normalize("NFKC", text.translate(_TYPOGRAPHY).translate(_TYPOGRAPHY)).split())
    return "—" if clean == "--" else clean


def _comparable_column(view, section):
    start, end = section["submitted_range"]
    first, last = section["archive_range"]
    sizes = [len(re.findall(r"\w+", text)) for text in (view.text[start:end], section["reference"]["text"][first:last])]
    # A sidebar must not claim a large intervening section in the archive's reading order.
    return min(sizes) * 5 >= max(sizes) * 3


def _column_layout_change(view, change, transcript):
    a, b = change["archive_range"]
    original, submitted = map(_typography, (transcript[a:b], change["submitted"]))
    if original == submitted:
        return True
    if not original and view.standalone_label(change["start"], change["end"]):
        return True
    if not re.fullmatch(r"\w+(?:[-‐]\s*\w+)+", original) or re.sub(r"[-‐]\s*", "", original) != submitted:
        return False
    # The original PDF word boxes must independently demonstrate a line wrap.
    boxes = [box for place in change["locations"] for box in place["boxes"]]
    return any(bottom - top > 2 for top, bottom in pairwise(sorted({box[1] for box in boxes})))


def _column_changes(view, mapping, transcript, edits):
    edits = list(edits)
    changes = _mapped_changes(view.text, edits, partial(view.places, mapping))
    if len(changes) != len(edits):
        return None
    return [change for change in changes if not _column_layout_change(view, change, transcript)]


def _capture_columns(text, mapping, metadata, pack_hash, reports, index):
    from app.detectors.text.extract import Document
    from app.evidence import document_gao_archive as archive

    sections = []
    for views in _pdf_columns(text, mapping):
        for view in views:
            doc = Document(view.text, view.paragraphs, metadata)
            result = archive.compare(doc, metadata["original_sha256"], pack_hash, reports, index,
                                     resolve_competition=True, allow_unmatched_flanks=True, include_sentence_context=False)
            if result["status"] != "reference_bound":
                continue
            section = view.scope() | _match(result, reports)
            if not _comparable_column(view, section):
                continue
            if view.side == "both":
                boundary = len(views[1].text)
                start, end = section["submitted_range"]
                if not start < boundary < end or min(boundary - start, end - boundary) < 80:
                    continue
            changes = _column_changes(view, mapping, section["reference"]["text"], _edits(result))
            if changes is None:
                continue
            sections.append(section | {"changes": changes})
            if view.side == "both":
                break
    if not sections or sum(len(section["changes"]) for section in sections) > 80:
        return None
    return {"schema": PDF_COLUMNS_SCHEMA, "archive_sha256": pack_hash, "sections": sections,
            "published_style_matches": [], "authorship_confidence_contribution": 0}


def capture(text, paragraphs, mapping, metadata, intervals, pack_hash, reports, index, *, pdf_columns=False):
    from app.detectors.text.extract import Document
    from app.evidence import document_gao_archive

    method = metadata["method"]
    if pdf_columns and method == "pymupdf" and not _pdf_native_order(text, mapping):
        return _capture_columns(text, mapping, metadata, pack_hash, reports, index)
    doc = Document(text, paragraphs, metadata, mapping)
    result = document_gao_archive.compare(doc, metadata["original_sha256"], pack_hash, reports, index,
                                          resolve_competition=True, include_sentence_context=False)
    if result["status"] != "reference_bound":
        return None
    places_at = partial(_places, method, mapping)
    changes = _mapped_changes(text, _edits(result), places_at)
    record = {"schema": SENTENCES_SCHEMA, "archive_sha256": pack_hash, **_match(result, reports),
              "changes": changes, "changed_sentences": _sentence_rows(doc, changes, places_at),
              "published_style_matches": [], "authorship_confidence_contribution": 0}
    record["published_style_matches"] = _exact_matches(text, record, intervals)
    return record


def _bounded(value, limit, *, empty=False):
    return (isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value)
            and 0 <= value[0] <= value[1] <= limit and (empty or value[0] < value[1]))


def _validate_reference(ref, archive_range):
    if (not isinstance(ref, dict) or set(ref) != {"id", "published_date", "version_url", "source_record_sha256",
                                                "transcript_sha256", "text"}
            or not REPORT_ID.fullmatch(ref["id"]) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", ref["published_date"])
            or not re.fullmatch(r"https://huggingface\.co/datasets/launch/gov_report/tree/[a-f0-9]{40}/data/gao_(train|valid|test)\.jsonl",
                                ref["version_url"])
            or not HASH.fullmatch(ref["source_record_sha256"]) or not 0 < len(ref["text"]) <= 20_000
            or hashlib.sha256(ref["text"].encode()).hexdigest() != ref["transcript_sha256"]
            or not _bounded(archive_range, len(ref["text"]))):
        raise ValueError("The archived transcript differs from its recorded identity")


def _validate_change(row, start, end, earlier_start, earlier_end):
    if (set(row) != {"start", "end", "submitted", "archive_range", "locations"}
            or not _bounded([row["start"], row["end"]], end, empty=True) or row["start"] < start
            or not _bounded(row["archive_range"], earlier_end, empty=True) or row["archive_range"][0] < earlier_start
            or not isinstance(row["submitted"], str) or len(row["submitted"]) != row["end"] - row["start"]
            or not row["locations"]):
        raise ValueError("An earlier wording difference has an invalid position")


def _validate_sentences(rows, changes, chars):
    if not isinstance(rows, list):
        raise ValueError("The changed sentences must record their source positions")
    previous, boundary = 0, 0
    for row in rows:
        index, start, end = row["sentence"], row["start"], row["end"]
        if (set(row) != {"sentence", "start", "end", "text", "locations", "differences"}
                or type(index) is not int or index <= previous or start < boundary or not _bounded([start, end], chars)
                or not isinstance(row["text"], str) or len(row["text"]) != end - start
                or not isinstance(row["locations"], list) or not row["locations"]
                or not isinstance(row["differences"], list) or not row["differences"]):
            raise ValueError("An earlier wording sentence has an invalid source location")
        expected = [i for i, change in enumerate(changes, start=1)
                    if max(start, change["start"]) < min(end, change["end"])
                    or (change["start"] == change["end"] and start < change["start"] < end)]
        if row["differences"] != expected:
            raise ValueError("A changed sentence links to a different wording difference")
        previous, boundary = index, end


def _validate_columns(record, source):
    sections = record.get("sections")
    if (set(record) != {"schema", "archive_sha256", "sections", "published_style_matches", "authorship_confidence_contribution"}
            or source.get("method") != "pymupdf" or record["published_style_matches"] != []
            or not isinstance(sections, list) or not 1 <= len(sections) <= 24):
        raise ValueError("An earlier wording comparison has an invalid PDF column scope")
    seen, used, count = {}, [], 0
    for section in sections:
        if set(section) != {"page", "column", "source_blocks", "reference", "submitted_range", "archive_range", "changes"}:
            raise ValueError("An earlier wording comparison has an invalid PDF column")
        page, side, ranges = section["page"], section["column"], section["source_blocks"]
        if (type(page) is not int or not 1 <= page <= 12 or side not in {"left", "right", "both"}
                or not isinstance(ranges, list) or not 2 <= len(ranges) <= 96
                or not all(_bounded(span, source["chars"]) for span in ranges)
                or not isinstance(section["changes"], list)):
            raise ValueError("An earlier wording comparison has an invalid PDF column location")
        lanes = seen.setdefault(page, [])
        if side in lanes or (lanes and "both" in [*lanes, side]):
            raise ValueError("A PDF column was compared more than once")
        lanes.append(side)
        used.extend(ranges)
        spans = []
        cursor = 0
        for a, b in ranges:
            end = cursor + b - a
            spans.append((cursor, end, a))
            cursor = end + 2
        if not _bounded(section["submitted_range"], cursor - 2):
            raise ValueError("The compared text falls outside the confirmed column")
        _validate_reference(section["reference"], section["archive_range"])
        start, end = section["submitted_range"]
        for change in section["changes"]:
            _validate_change(change, start, end, *section["archive_range"])
            lo, hi = change["start"], max(change["end"], change["start"] + 1)
            expected = [[source_start + max(a, lo) - a, source_start + min(b, hi) - a]
                        for a, b, source_start in spans if a < hi and lo < b]
            places = change["locations"]
            if (not isinstance(places, list) or [[place["start"], place["end"]] for place in places] != expected
                    or any(set(place) != {"start", "end", "page", "coordinate_space", "boxes"} or place["page"] != page
                           or place["coordinate_space"] != native.PDF_POINTS or not isinstance(place["boxes"], list)
                           or not place["boxes"] or not all(native._valid_box(box) for box in place["boxes"]) for place in places)):
                raise ValueError("A wording difference falls outside its original PDF column")
        count += len(section["changes"])
    used.sort()
    if count > 80 or any(previous[1] > current[0] for previous, current in zip(used, used[1:])):
        raise ValueError("Earlier wording comparisons overlap or exceed the report limit")


def validate(document, expected):
    coverage = document.get("coverage") or {}
    source = document["source"]
    record = coverage.get(KEY)
    if record is None and expected is None and source.get(COMMITMENT) is None:
        return
    if (not isinstance(record, dict) or type(record["authorship_confidence_contribution"]) is not int
            or record["authorship_confidence_contribution"] != 0 or source.get("method") not in FORMATS.values()
            or source.get(COMMITMENT) != expected or canonical_hash(record) != expected
            or not HASH.fullmatch(record["archive_sha256"])):
        raise ValueError("The published wording comparison differs from its recorded commitment or scope")
    if record.get("schema") == PDF_COLUMNS_SCHEMA:
        _validate_columns(record, source)
        return
    schema = record.get("schema")
    required = {"schema", "archive_sha256", "reference", "submitted_range", "archive_range", "changes",
                "published_style_matches", "authorship_confidence_contribution"}
    if schema == SENTENCES_SCHEMA:
        required.add("changed_sentences")
    if (schema not in {SCHEMA, SENTENCES_SCHEMA} or set(record) != required
            or not _bounded(record["submitted_range"], source["chars"])):
        raise ValueError("The published wording comparison differs from its recorded commitment or scope")
    _validate_reference(record["reference"], record["archive_range"])
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
        _validate_change(row, start, end, earlier_start, earlier_end)
    if schema == SENTENCES_SCHEMA:
        _validate_sentences(record["changed_sentences"], record["changes"], source["chars"])


def _verify_columns(record, text, mapping, metadata, word_review):
    from app.evidence import document_gao_archive

    if not _mapped_source(".pdf", text, mapping, metadata, word_review):
        raise ValueError("Earlier wording needs visible selectable text with original PDF positions")
    views = {(view.page, view.side): view for page in _pdf_columns(text, mapping) for view in page}
    for section in record["sections"]:
        view = views.get((section["page"], section["column"]))
        if view is None or section["source_blocks"] != view.scope()["source_blocks"]:
            raise ValueError("Earlier wording differs from the original PDF column geometry")
        if not _comparable_column(view, section):
            raise ValueError("The original PDF column does not cover enough of the archived wording")
        changes = document_gao_archive.replay_wording(view.text, section["reference"]["text"], section["submitted_range"],
                                                     section["archive_range"])
        if _column_changes(view, mapping, section["reference"]["text"], changes) != section["changes"]:
            raise ValueError("An earlier wording difference differs from the retained transcript or original locations")


def verify_source(document, text, mapping, metadata, word_review, *, paragraphs=None):
    record = (document.get("coverage") or {}).get(KEY)
    if record is None:
        return
    from app.evidence import document_gao_archive

    validate(document, document["source"].get(COMMITMENT))
    if record["schema"] == PDF_COLUMNS_SCHEMA:
        _verify_columns(record, text, mapping, metadata, word_review)
        return
    method = document["source"]["method"]
    extension = next(ext for ext, name in FORMATS.items() if method == name)
    intervals = document["coverage"].get(PASSAGE_INTERVALS_KEY)
    if (not eligible(extension, text, mapping, metadata, word_review)
            or _exact_matches(text, record, intervals) != record["published_style_matches"]):
        raise ValueError("The archived matches differ from the original wording")
    expected = document_gao_archive.replay_wording(text, record["reference"]["text"], record["submitted_range"],
                                                  record["archive_range"])
    places_at = partial(_places, method, mapping)
    if _mapped_changes(text, expected, places_at) != record["changes"]:
        raise ValueError("An earlier wording difference differs from the retained transcript or original locations")
    if record["schema"] == SENTENCES_SCHEMA:
        from app.detectors.text.extract import Document

        if paragraphs is None:
            raise ValueError("Original paragraph positions are required to check changed sentences")
        doc = Document(text, paragraphs, metadata, mapping)
        if _sentence_rows(doc, record["changes"], places_at) != record["changed_sentences"]:
            raise ValueError("The changed sentences differ from their original text, links or locations")


def display_intervals(intervals, wording):
    excluded = {(row["start"], row["end"]) for row in wording["published_style_matches"]} if wording else set()
    return [row for row in intervals["ranges"] if (row["start"], row["end"]) not in excluded]
