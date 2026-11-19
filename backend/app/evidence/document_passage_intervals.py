"""Source-bound review of distinct writing passages."""
from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from app.core.canonical import canonical_hash
from app.evidence import document_interval_locations as native
from app.evidence import document_tracked_changes, report_passage_calibration
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import PASSAGE_INTERVALS_COMMITMENT as COMMITMENT
from app.evidence.document_scope import PASSAGE_INTERVALS_KEY as KEY
from app.evidence.passage_interval_selection import (
    add_disjoint_ranges,
    center_on_document,
    combine_responses,
    flank_adjusted_candidates,
    interval_candidates,
    length_bucket,
    normalize,
    probability_log_odds,
    select_intervals,
)

SCHEMA = "document-writing-passage-intervals/1"
NATIVE_SCHEMA = "document-writing-passage-intervals/2"
DOCUMENT_BASELINE_SCHEMA = "document-writing-passage-intervals/3"
REPORT_SCHEMA = "document-writing-passage-intervals/4"
SHORT_SCOPE_SCHEMA = "document-writing-passage-intervals/5"
SHORT_SCOPE_REASON = "too_few_sentences"
BASELINE_SCHEMAS = (DOCUMENT_BASELINE_SCHEMA, REPORT_SCHEMA)
SHORT_SCOPE_MIN_SENTENCES = 12
MIN_SENTENCES = SHORT_SCOPE_MIN_SENTENCES
MAX_SENTENCES = 128
PREVIEW_CHARS = 600
TITLE = "Possible AI-like writing passages"
SHORT_SCOPE_TITLE = "Optional writing passage check"
CHECK_SUMMARY = "This optional check combines readings of individual sentences and their surrounding text. "
ASSESSMENT_WEIGHT_NOTE = "This check carries zero weight in the overall assessment."
LIMITATION = "Ordinary writing can produce similar patterns. " + ASSESSMENT_WEIGHT_NOTE
EXPLANATION = CHECK_SUMMARY + "Sustained AI-like patterns can highlight separate passages for review. " + LIMITATION
BASELINE_EXPLANATION = CHECK_SUMMARY + "It also looks for passages that stand out within the document. " + LIMITATION
REPORT_EXPLANATION = (CHECK_SUMMARY + "This file matched the writing patterns of the economic-report reference set. "
                      "Each displayed passage also stood out against that reference. " + LIMITATION)
BASELINE_REASON = "Selected by comparing this passage with the writing patterns elsewhere in this file."
STATUS = {
    "flagged": "Inspect the highlighted passages in context.",
    "no_ranges": "No passage was selected. Authorship remains undetermined by this check.",
    "unassessed": "Complete readings or original passage positions were unavailable.",
}

# Measured on ordinary validation excerpts pooled across both source genres.
# See docs/writing-multi-passage-transfer-2026-11-16.md.
CONTEXT = (
    (-3.253234888587576, 3.5111942840046964),
    (-2.8778478483582566, 5.030326709587857),
    (-3.5523261390723144, 6.5736244126440315),
)
ISOLATED = (
    (-1.5621250334119368, 2.7888982913218143),
    (-1.3827710854493522, 4.661869464394859),
    (-4.495148951533944, 4.482536664350449),
)
POINT = 4.065364287484746
# References and operating point fixed on ordinary validation works. See
# docs/writing-local-baseline-transfer-2026-11-16.md for the separate transfer test.
DOCUMENT_BASELINE_ISOLATED = -0.4632072194951884
DOCUMENT_BASELINE_CONTEXT = -0.26678345428832334
DOCUMENT_BASELINE_POINT = 3.6112710098095553
REPORT_FIT_PATH = Path(__file__).resolve().parents[1] / "detectors/text/calibrations/report-passage-profile-v1.json"
REPORT_FIT_SHA256 = "2d42757ff0ec8429691ab7014a75318f9ae1367089534452af1774ec6103e118"


@lru_cache(maxsize=1)
def _report_fit():
    data = REPORT_FIT_PATH.read_bytes()
    if hashlib.sha256(data).hexdigest() != REPORT_FIT_SHA256:
        raise ValueError("The economic report calibration differs from its pinned edition")
    return json.loads(data)


def explanation(schema):
    return {DOCUMENT_BASELINE_SCHEMA: BASELINE_EXPLANATION, REPORT_SCHEMA: REPORT_EXPLANATION}.get(schema, EXPLANATION)


def status_note(record):
    if record["schema"] == SHORT_SCOPE_SCHEMA:
        return (f"This optional check requires at least {record['min_sentences']} eligible sentences. "
                f"The writing input contained {record['eligible_sentences']}, so the check was skipped.")
    return STATUS[record["status"]]


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def result_commitment(result):
    record = from_result(result)
    return canonical_hash(record) if record is not None else None


def _choices(sentences, readings, *, document_baseline):
    if readings is None:
        return ()
    words = [row[2] for row in sentences]
    iso = [normalize(probability_log_odds(value), *ISOLATED[length_bucket(count)])
           for value, count in zip(readings["isolated"], words, strict=True)]
    ctx = [normalize(probability_log_odds(value), *CONTEXT[length_bucket(sum(words[max(0, i - 1):i + 2]))])
           for i, value in enumerate(readings["context"])]
    count = len(sentences)
    pooled = select_intervals(interval_candidates(combine_responses(iso, ctx)), POINT, count)
    local = ()
    if document_baseline:
        centered = combine_responses(center_on_document(iso, DOCUMENT_BASELINE_ISOLATED),
                                     center_on_document(ctx, DOCUMENT_BASELINE_CONTEXT))
        local = select_intervals(flank_adjusted_candidates(centered), DOCUMENT_BASELINE_POINT, count)
    return add_disjoint_ranges(pooled, local)


def _reviewed_choices(sentences, readings, *, document_baseline, report_review):
    current = _choices(sentences, readings, document_baseline=document_baseline)
    if not (document_baseline and report_review and readings is not None):
        return current, False
    reference = report_passage_calibration.review(sentences, readings, {"isolated": ISOLATED, "context": CONTEXT}, _report_fit())
    return report_passage_calibration.confirm(current, reference), reference is not None


def _original_position(mapping, offset):
    for block in mapping:
        if block.get("kind") == "text" and block["start"] <= offset < block["end"]:
            positions = clip_source_ranges(block.get("source_ranges", ()), offset, offset + 1)
            return positions[0] if len(positions) == 1 else None
    return None


def short_scope(method, checked_end, sentence_count):
    result = {"schema": SHORT_SCOPE_SCHEMA, "status": "skipped", "reason": SHORT_SCOPE_REASON,
              "eligible_sentences": sentence_count, "min_sentences": SHORT_SCOPE_MIN_SENTENCES,
              "ranges": [], "authorship_confidence_contribution": 0}
    if method != "plain":
        result["checked_end"] = checked_end
    return result


def capture(text, mapping, sentences, readings, *, method="plain", checked_end=None, document_baseline=False,
            report_review=False):
    rows = []
    choices, applied = _reviewed_choices(sentences, readings, document_baseline=document_baseline, report_review=report_review)
    for first, stop, _, basis in choices:
        start, end = sentences[first][0], sentences[stop - 1][1]
        if method == "plain":
            left, right = _original_position(mapping, start), _original_position(mapping, end - 1)
            place = {"source_start": left[0], "source_end": right[1]} if left and right and left[0] < right[1] else {}
        else:
            locations = native.locate(method, mapping, start, end)
            place = {"locations": locations} if locations else {}
        if not place:
            readings, rows, applied = None, [], False
            break
        row = {"start": start, "end": end, "preview": text[start:min(end, start + PREVIEW_CHARS)], **place}
        if document_baseline:
            row["basis"] = basis
        rows.append(row)
    status = "unassessed" if readings is None else "flagged" if rows else "no_ranges"
    schema = DOCUMENT_BASELINE_SCHEMA if document_baseline else (SCHEMA if method == "plain" else NATIVE_SCHEMA)
    if applied:
        schema = REPORT_SCHEMA
    result = {"schema": schema, "sentences": sentences, "readings": readings,
              "ranges": rows, "status": status, "authorship_confidence_contribution": 0}
    if method != "plain":
        result["checked_end"] = checked_end
    return result


def validate(document, expected):
    metadata = document.get("coverage") or {}
    record = metadata.get(KEY)
    if record is None and expected is None:
        return
    method, chars = document["source"]["method"], document["source"]["chars"]
    is_native = method in native.KINDS
    if isinstance(record, dict) and record.get("schema") == SHORT_SCOPE_SCHEMA:
        count = record.get("eligible_sentences")
        checked = record.get("checked_end") if is_native else chars
        if (type(count) is not int or not 0 <= count < SHORT_SCOPE_MIN_SENTENCES
                or type(record.get("min_sentences")) is not int
                or type(record.get("authorship_confidence_contribution")) is not int
                or record != short_scope(method, checked, count) or canonical_hash(record) != expected
                or type(checked) is not int or not 0 < checked <= chars
                or not is_native and method != "plain"
                or is_native and (not native.eligible_source(method, metadata, metadata.get(document_tracked_changes.KEY))
                                  or method != "email" and checked != chars)):
            raise ValueError("Writing passage coverage has an invalid short-input record")
        return
    fields = {"schema", "sentences", "readings", "ranges", "status", "authorship_confidence_contribution"}
    if is_native:
        fields.add("checked_end")
    original_schema = NATIVE_SCHEMA if is_native else SCHEMA
    if (not isinstance(record, dict) or set(record) != fields or canonical_hash(record) != expected
            or record["schema"] not in (original_schema, *BASELINE_SCHEMAS) or (not is_native and method != "plain")
            or type(record["authorship_confidence_contribution"]) is not int
            or record["authorship_confidence_contribution"] != 0):
        raise ValueError("Writing passage interval evidence has an invalid commitment or scope")
    checked = record["checked_end"] if is_native else chars
    if is_native and (not native.eligible_source(method, metadata, metadata.get(document_tracked_changes.KEY))
                      or type(checked) is not int or not 0 < checked <= chars or (method != "email" and checked != chars)):
        raise ValueError("Incomplete native writing passage coverage")

    sentences, readings, rows = (record[name] for name in ("sentences", "readings", "ranges"))
    if not isinstance(sentences, list) or not MIN_SENTENCES <= len(sentences) <= MAX_SENTENCES:
        raise ValueError("Writing passage intervals need valid sentence boundaries")
    previous = 0
    for span in sentences:
        if (not isinstance(span, list) or len(span) != 3 or any(type(value) is not int for value in span)
                or not previous <= span[0] < span[1] <= checked or not 0 <= span[2] <= span[1] - span[0]):
            raise ValueError("Writing passage intervals have invalid sentence boundaries")
        previous = span[1]
    if readings is not None and (
            not isinstance(readings, dict) or set(readings) != {"isolated", "context"} or
            any(not isinstance(values, list) or len(values) != len(sentences)
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                       for value in values) for values in readings.values())):
        raise ValueError("Writing passage intervals have incomplete or invalid readings")
    document_baseline = record["schema"] in BASELINE_SCHEMAS
    report_review = record["schema"] == REPORT_SCHEMA
    choices, applied = _reviewed_choices(sentences, readings, document_baseline=document_baseline, report_review=report_review)
    status = "unassessed" if readings is None else "flagged" if choices else "no_ranges"
    if applied != report_review or record["status"] != status or not isinstance(rows, list) or len(rows) != len(choices):
        raise ValueError("Writing passage intervals differ from the recorded readings")
    source_previous = 0
    range_fields = {"start", "end", "preview"} | ({"locations"} if is_native else {"source_start", "source_end"})
    if document_baseline:
        range_fields.add("basis")
    for row, (first, stop, _, basis) in zip(rows, choices, strict=True):
        a, b = sentences[first][0], sentences[stop - 1][1]
        if (not isinstance(row, dict) or set(row) != range_fields
                or (row["start"], row["end"]) != (a, b)
                or not isinstance(row["preview"], str) or len(row["preview"]) != min(b - a, PREVIEW_CHARS)
                or (document_baseline and row["basis"] != basis)):
            raise ValueError("Writing passage intervals have invalid highlighted ranges")
        if is_native:
            native.validate(method, metadata, row["locations"], a, b)
        else:
            if (type(row["source_start"]) is not int or type(row["source_end"]) is not int
                    or not source_previous <= row["source_start"] < row["source_end"]):
                raise ValueError("Writing passage intervals have invalid original text positions")
            source_previous = row["source_end"]


def verify_source(document, text, mapping, sentences, *, metadata=None, word_review=None):
    record = (document.get("coverage") or {}).get(KEY)
    validate(document, document["source"].get(COMMITMENT))
    if record is not None:
        method = document["source"]["method"]
        if method != "plain" and metadata is None:
            raise ValueError("Native writing passage replay requires the mapped source coverage")
        end = len(text) if method == "plain" else native.checked_end(method, text, mapping, metadata, word_review)
        if end is None or (method != "plain" and end != record["checked_end"]):
            raise ValueError("Writing passage intervals differ from the mapped text scope")
        geometry = [[row["start"], row["end"], row["n_words"]] for row in sentences if row["end"] <= end]
        if record["schema"] == SHORT_SCOPE_SCHEMA:
            if len(geometry) != record["eligible_sentences"]:
                raise ValueError("Writing passage coverage sentence count differs from the source text")
            return
        if capture(text, mapping, geometry, record["readings"], method=method,
                   checked_end=end, document_baseline=record["schema"] in BASELINE_SCHEMAS,
                   report_review=record["schema"] == REPORT_SCHEMA) != record:
            raise ValueError("Writing passage intervals differ from the source text")
