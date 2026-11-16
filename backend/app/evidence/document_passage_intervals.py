"""Source-bound review of distinct passages in plain English text."""
from __future__ import annotations

import math

from app.core.canonical import canonical_hash
from app.evidence.document_native_locations import clip_source_ranges
from app.evidence.document_scope import PASSAGE_INTERVALS_COMMITMENT as COMMITMENT
from app.evidence.document_scope import PASSAGE_INTERVALS_KEY as KEY
from app.evidence.passage_interval_selection import (
    combine_responses,
    interval_candidates,
    length_bucket,
    normalize,
    probability_log_odds,
    select_intervals,
)

SCHEMA = "document-writing-passage-intervals/1"
MIN_SENTENCES = 12
MAX_SENTENCES = 128
PREVIEW_CHARS = 600
TITLE = "Possible AI-like writing passages"
EXPLANATION = (
    "This optional check combines readings of individual sentences and their surrounding text. "
    "Sustained AI-like patterns can highlight separate passages for review. "
    "Ordinary writing can produce similar patterns. This check carries zero weight in the overall assessment."
)
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


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def result_commitment(result):
    record = from_result(result)
    return canonical_hash(record) if record is not None else None


def _choices(sentences, readings):
    if readings is None:
        return ()
    words = [row[2] for row in sentences]
    iso = [normalize(probability_log_odds(value), *ISOLATED[length_bucket(count)])
           for value, count in zip(readings["isolated"], words, strict=True)]
    ctx = [normalize(probability_log_odds(value), *CONTEXT[length_bucket(sum(words[max(0, i - 1):i + 2]))])
           for i, value in enumerate(readings["context"])]
    scores = combine_responses(iso, ctx)
    return select_intervals(interval_candidates(scores), POINT, len(sentences))


def _original_position(mapping, offset):
    for block in mapping:
        if block.get("kind") == "text" and block["start"] <= offset < block["end"]:
            positions = clip_source_ranges(block.get("source_ranges", ()), offset, offset + 1)
            return positions[0] if len(positions) == 1 else None
    return None


def capture(text, mapping, sentences, readings):
    rows = []
    for first, stop, _ in _choices(sentences, readings):
        start, end = sentences[first][0], sentences[stop - 1][1]
        left, right = _original_position(mapping, start), _original_position(mapping, end - 1)
        if left is None or right is None or left[0] >= right[1]:
            readings, rows = None, []
            break
        rows.append({
            "start": start,
            "end": end,
            "source_start": left[0],
            "source_end": right[1],
            "preview": text[start:min(end, start + PREVIEW_CHARS)],
        })
    status = "unassessed" if readings is None else "flagged" if rows else "no_ranges"
    return {"schema": SCHEMA, "sentences": sentences, "readings": readings, "ranges": rows, "status": status,
            "authorship_confidence_contribution": 0}


def validate(document, expected):
    record = (document.get("coverage") or {}).get(KEY)
    if record is None and expected is None:
        return
    if (not isinstance(record, dict) or set(record) != {
            "schema", "sentences", "readings", "ranges", "status", "authorship_confidence_contribution"}
            or record["schema"] != SCHEMA or canonical_hash(record) != expected
            or document["source"]["method"] != "plain"
            or type(record["authorship_confidence_contribution"]) is not int
            or record["authorship_confidence_contribution"] != 0):
        raise ValueError("Writing passage interval evidence has an invalid commitment or scope")

    sentences, readings, rows = (record[name] for name in ("sentences", "readings", "ranges"))
    chars = document["source"]["chars"]
    if not isinstance(sentences, list) or not MIN_SENTENCES <= len(sentences) <= MAX_SENTENCES:
        raise ValueError("Writing passage intervals need valid sentence boundaries")
    previous = 0
    for span in sentences:
        if (not isinstance(span, list) or len(span) != 3 or any(type(value) is not int for value in span)
                or not previous <= span[0] < span[1] <= chars or not 0 <= span[2] <= span[1] - span[0]):
            raise ValueError("Writing passage intervals have invalid sentence boundaries")
        previous = span[1]
    if readings is not None and (
            not isinstance(readings, dict) or set(readings) != {"isolated", "context"} or
            any(not isinstance(values, list) or len(values) != len(sentences)
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
                       for value in values) for values in readings.values())):
        raise ValueError("Writing passage intervals have incomplete or invalid readings")
    choices = _choices(sentences, readings)
    status = "unassessed" if readings is None else "flagged" if choices else "no_ranges"
    if record["status"] != status or not isinstance(rows, list) or len(rows) != len(choices):
        raise ValueError("Writing passage intervals differ from the recorded readings")
    source_previous = 0
    for row, (first, stop, _) in zip(rows, choices, strict=True):
        a, b = sentences[first][0], sentences[stop - 1][1]
        if (not isinstance(row, dict) or set(row) != {"start", "end", "source_start", "source_end", "preview"}
                or (row["start"], row["end"]) != (a, b)
                or type(row["source_start"]) is not int or type(row["source_end"]) is not int
                or not source_previous <= row["source_start"] < row["source_end"]
                or not isinstance(row["preview"], str) or len(row["preview"]) != min(b - a, PREVIEW_CHARS)):
            raise ValueError("Writing passage intervals have invalid highlighted ranges")
        source_previous = row["source_end"]


def verify_source(document, text, mapping, sentences):
    record = (document.get("coverage") or {}).get(KEY)
    validate(document, document["source"].get(COMMITMENT))
    if record is not None:
        geometry = [[row["start"], row["end"], row["n_words"]] for row in sentences]
        if capture(text, mapping, geometry, record["readings"]) != record:
            raise ValueError("Writing passage intervals differ from the source text")
