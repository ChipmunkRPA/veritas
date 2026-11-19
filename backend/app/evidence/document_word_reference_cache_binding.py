# SPDX-License-Identifier: Apache-2.0
"""Bind saved literal Word cross-reference comparisons to the submitted source."""

from __future__ import annotations

from app.core.canonical import canonical_hash

KEY = "word_reference_cache"
SCHEMA = "document-word-reference-cache/1"
POLICY = "word_reference_cache_reader"
COMMITMENT = "word_reference_cache_sha256"
MAX_BODY_BLOCKS = 512
MAX_CHILDREN = 192
MAX_VALUE_CHARS = 256
MAX_PARAGRAPH_CHARS = 32_768
MAX_FINDINGS = 12
TITLE = "Saved Word cross-references"
LIMITATION = ("The two saved texts differ. Editing a bookmarked passage without refreshing its cross-reference can cause "
              "this difference. The check reads the submitted file without updating fields. It carries zero weight in the AI assessment.")


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def plain_text(value):
    """Compare words without attributing differences in spacing to a changed value."""
    return " ".join(value.split())


def _position(position):
    if not isinstance(position, dict) or position.keys() != {"body_index", "child_index", "range", "text"}:
        raise ValueError("Invalid Word cross-reference location")
    body_index, child_index, offsets, value = (position[k] for k in ("body_index", "child_index", "range", "text"))
    if (type(body_index) is not int or not 0 <= body_index < MAX_BODY_BLOCKS
            or type(child_index) is not int or not 0 <= child_index < MAX_CHILDREN
            or not isinstance(offsets, list) or len(offsets) != 2 or any(type(i) is not int for i in offsets)
            or not 0 <= offsets[0] < offsets[1] <= MAX_PARAGRAPH_CHARS
            or not isinstance(value, str) or not 0 < len(value) <= MAX_VALUE_CHARS
            or offsets[1] - offsets[0] != len(value) or value.strip() != value or not value.isprintable()):
        raise ValueError("Invalid Word cross-reference characters or position")
    return body_index, child_index


def validate(document):
    source = document["source"]
    if KEY not in document and COMMITMENT not in source:
        return
    record = document.get(KEY)
    fields = {"schema", "checked", "mismatches", "findings", "authorship_confidence_contribution"}
    if (source.get("method") != "python-docx" or not isinstance(record, dict) or record.keys() != fields
            or record["schema"] != SCHEMA or type(record["authorship_confidence_contribution"]) is not int
            or record["authorship_confidence_contribution"] != 0 or canonical_hash(record) != source.get(COMMITMENT)):
        raise ValueError("Word cross-reference evidence differs from its source commitment")
    checked, mismatches, findings = (record[k] for k in ("checked", "mismatches", "findings"))
    if (type(checked) is not int or type(mismatches) is not int or not 0 < mismatches <= checked <= MAX_BODY_BLOCKS
            or not isinstance(findings, list) or len(findings) != min(mismatches, MAX_FINDINGS)):
        raise ValueError("Word cross-reference counts differ from their saved findings")
    previous = (-1, -1)
    for row in findings:
        if (not isinstance(row, dict) or row.keys() != {"field", "target", "record_shape"}
                or row["record_shape"] not in ("simple", "complex")):
            raise ValueError("Invalid Word cross-reference finding")
        field = _position(row["field"])
        target = _position(row["target"])
        if (previous >= field or field[0] == target[0]
                or plain_text(row["field"]["text"]) == plain_text(row["target"]["text"])):
            raise ValueError("Invalid Word cross-reference comparison")
        previous = field


def verify_record(document, record, *, commitment):
    validate(document)
    if document["source"].get(COMMITMENT) != commitment or document.get(KEY) != record:
        raise ValueError("Word cross-reference evidence differs from the original inspection")
