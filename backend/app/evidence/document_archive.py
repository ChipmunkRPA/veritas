"""Compare a retained plain-text body with the pinned FRUS excerpt library."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from app.detectors.text.extract import Document, text_hash
from app.evidence import document_archive_layout
from app.evidence.document_revision import MAX_CHARS, MAX_TOKENS, differences, tokens

SCHEMA = "chipmunk.document-archive-comparison/1"
ARCHIVE_COMMIT = "216cdb7649a8e29504377d4f2858351dc46331a8"
ARCHIVE_SHA256 = "29c3e843f74bcaa244ee15af6f5cebd572091051614d554823bebab46b7ca24e"
ARCHIVE_SIZE = 430369
ARCHIVE_PATH = Path(__file__).with_name("data") / "frus-archive.json"
ARCHIVE_TITLE = "Foreign Relations of the United States"
ARCHIVE_PUBLISHER = "Office of the Historian, U.S. Department of State"
ARCHIVE_NOTICE = f"https://github.com/HistoryAtState/frus/blob/{ARCHIVE_COMMIT}/LICENSE.md"
ARCHIVE_LIBRARY = "https://history.state.gov/historicaldocuments"
SUBMITTED_PARAGRAPHS = "submitted_paragraphs"
RECOVERED_BOUNDARIES = "recovered_archive_boundaries"
SCOPE = ("Compares wording and punctuation in aligned passages from 118 selected English-language archive excerpts. "
         "Unaligned passages are outside the comparison.")
INTERPRETATION = ("A match identifies wording shared with a cited archive version. Differences may reflect ordinary edits, "
                  "transcription or other changes. This comparison carries zero AI-assessment weight.")
STATUS = {
    "reference_bound": "Archive passage matched",
    "no_qualified_reference": "No matching passage in this archive library",
    "ambiguous_reference": "More than one archive passage could match",
    "out_of_scope": "Text is outside the comparison scope",
}
WORD = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*", re.UNICODE)
NGRAM = 8


class ArchiveUnavailable(RuntimeError):
    """The installed archive differs from the audited release."""


def grams(value):
    terms = [unicodedata.normalize("NFKC", match[0]).casefold().replace("’", "'") for match in WORD.finditer(value)]
    return [tuple(terms[i:i + NGRAM]) for i in range(len(terms) - NGRAM + 1)], len(terms)


def build_index(catalog):
    hits = defaultdict(set)
    for source, row in catalog.items():
        for paragraph, part in enumerate(row["paragraphs"]):
            for key in grams(part["text"])[0]:
                hits[key].add((source, paragraph))
    return {key: sorted(matches) for key, matches in hits.items() if len({source for source, _ in matches}) <= 4}


@lru_cache(maxsize=1)
def library():
    try:
        with ARCHIVE_PATH.open("rb") as stream:
            raw = stream.read(ARCHIVE_SIZE + 1)
        if len(raw) != ARCHIVE_SIZE or hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
            raise ArchiveUnavailable("The local archive pack does not match its pinned hash.")
        catalog = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ArchiveUnavailable("The local archive pack is unavailable.") from error
    return catalog, build_index(catalog)


@lru_cache(maxsize=1)
def _library_edges():
    return document_archive_layout.index_edges(library()[0])


def _anchors(text, body, index):
    evidence = defaultdict(dict)
    denominator = 0
    for i, (a, b) in enumerate(body):
        fingerprints, n = grams(text[a:b])
        denominator += n
        counts = defaultdict(set)
        for position, key in enumerate(fingerprints):
            for entry in index.get(key, ()):
                counts[entry].update(range(position, position + NGRAM))
        best_by_source = defaultdict(list)
        for (source, j), matches in counts.items():
            if len(matches) >= 20 and len(matches) * 100 >= 55 * n:
                best_by_source[source].append((len(matches), j))
        for source, choices in best_by_source.items():
            choices.sort(reverse=True)
            if len(choices) == 1 or choices[0][0] > choices[1][0]:
                evidence[source][i] = choices[0]
    ranked = []
    for source, items in evidence.items():
        ordered = sorted(items.items())
        if len(ordered) < 3 or any(a[1][1] >= b[1][1] for a, b in zip(ordered, ordered[1:])):
            continue
        coverage = sum(item[1][0] for item in ordered)
        if coverage >= 60:
            ranked.append((coverage, source, ordered))
    ranked.sort(reverse=True)
    if not ranked or ranked[0][0] * 100 < 65 * denominator:
        return "no_qualified_reference", 0, denominator, None, []
    if len(ranked) > 1 and (ranked[0][0] - ranked[1][0]) * 100 < 10 * denominator:
        return "ambiguous_reference", 0, denominator, None, []
    coverage, source, ordered = ranked[0]
    return "reference_bound", coverage, denominator, source, [(i, value[1]) for i, value in ordered]


def _alignment(pairs):
    mapped = [(i, j, "fingerprint_anchor") for i, j in pairs]
    for (a, b), (c, d) in zip(pairs, pairs[1:]):
        if c - a == 2 and d - b == 2:
            mapped.append((a + 1, b + 1, "bounded_pair"))
        elif c - a == 2 and d - b == 1:
            mapped.append((a + 1, None, "bounded_insert"))
        elif c - a == 1 and d - b == 2:
            mapped.append((c, b + 1, "bounded_delete"))
    return sorted(mapped, key=lambda item: (item[0], item[2], item[1] or 0))


def _excerpt(text, span, changed_tokens):
    if span is None:
        return None
    start, end = span
    return {"start": start, "end": end, "text": text[start:end], "changed_tokens": changed_tokens}


def _changes(text, body, paragraphs, alignment):
    aligned, changes = [], []
    for i, j, relation in alignment:
        number, (lo, hi) = body[i]
        part = paragraphs[j] if j is not None else None
        reference = part["text"] if part is not None else ""
        ordinal = part["ordinal"] + 1 if part is not None else None
        if relation == "bounded_delete":
            submission = [lo, lo]
            old_tokens = tokens(reference, [(0, len(reference))])
            records = [{"earlier": [0, len(reference)], "later": submission,
                        "earlier_changed_tokens": [[a, b] for _, a, b in old_tokens], "later_changed_tokens": []}]
        elif part is None:
            submission = [lo, hi]
            new_tokens = tokens(text, [(lo, hi)])
            records = [{"earlier": None, "later": submission, "earlier_changed_tokens": [],
                        "later_changed_tokens": [[a, b] for _, a, b in new_tokens]}]
        else:
            submission = [lo, hi]
            records = differences(tokens(reference, [(0, len(reference))]), tokens(text, [(lo, hi)]))
        aligned.append({"submitted_paragraph": number, "submitted_range": submission,
                        "archive_paragraph": ordinal, "alignment": relation})
        for record in records:
            changes.append({"submitted_paragraph": number, "archive_paragraph": ordinal, "alignment": relation,
                            "submitted": _excerpt(text, record["later"], record["later_changed_tokens"]),
                            "archive": _excerpt(reference, record["earlier"], record["earlier_changed_tokens"])})
    return aligned, changes


def compare(doc: Document, upload_sha256: str, catalog=None, index=None):
    """Return exact source evidence on the extracted text, without an authorship inference."""
    body = [(number, span) for number, (span, row) in enumerate(zip(doc.paragraphs, doc.source_map, strict=True), 1)
            if row.get("quote_role", "body") == "body" and not doc.text[slice(*span)].lstrip().startswith(">")]
    output = {
        "schema": SCHEMA, "status": "out_of_scope", "scope": SCOPE, "interpretation": INTERPRETATION,
        "coordinate_space": "extracted_text_unicode_codepoint", "layout": SUBMITTED_PARAGRAPHS,
        "authorship_confidence_contribution": 0, "submitted_sha256": upload_sha256,
        "extracted_text_sha256": text_hash(doc.text),
        "library": {"title": ARCHIVE_TITLE, "publisher": ARCHIVE_PUBLISHER, "url": ARCHIVE_LIBRARY,
                    "license": "CC0-1.0", "license_url": ARCHIVE_NOTICE, "revision": ARCHIVE_COMMIT,
                    "sha256": ARCHIVE_SHA256, "indexed_excerpts": 118},
        "coverage": {"body_paragraphs": len(body), "body_words": 0, "matched_words": 0},
        "reference": None, "aligned_paragraphs": [], "changes": [],
    }
    spans = [span for _, span in body]
    joined = len(body) == len(doc.paragraphs) == 1 and doc.source_map[0].get("kind") == "text" and ">" not in doc.text
    if len(doc.text) > MAX_CHARS or not (joined or 3 <= len(body) <= 24) or len(tokens(doc.text, spans)) > MAX_TOKENS:
        return output
    installed_library = catalog is None or index is None
    if installed_library:
        catalog, index = library()
    if joined:
        edges = _library_edges() if installed_library else document_archive_layout.index_edges(catalog)
        recovered = document_archive_layout.joined_spans(doc.text, spans[0], edges)
        if recovered is None:
            return output
        candidate_source, first, spans = recovered
        body = [(1, span) for span in spans]
    status, coverage, total, source, pairs = _anchors(doc.text, spans, index)
    if joined:
        if source != candidate_source or pairs != [(i, first + i) for i in range(len(spans))]:
            return output
        output["layout"] = RECOVERED_BOUNDARIES
    output["status"] = status
    output["coverage"].update(body_words=total, matched_words=coverage)
    if source is None:
        return output
    row = catalog[source]
    volume, document = row["volume"], row["doc_id"]
    output["reference"] = {"id": source, "url": f"{ARCHIVE_LIBRARY}/{volume}/{document}",
                           "version_url": f"https://github.com/HistoryAtState/frus/blob/{ARCHIVE_COMMIT}/volumes/{volume}.xml",
                           "volume": volume, "document": document}
    output["aligned_paragraphs"], output["changes"] = _changes(doc.text, body, row["paragraphs"], _alignment(pairs))
    return output
