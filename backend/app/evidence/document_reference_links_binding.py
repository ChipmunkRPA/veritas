# SPDX-License-Identifier: Apache-2.0
"""Validate source commitments for saved internal bibliography links."""

from __future__ import annotations

from app.core.canonical import canonical_hash
from app.evidence.document_scope import HTML_STATIC_KEY, HTML_STATIC_POLICY

KEY = "reference_link_discrepancies"
SCHEMA = "document-reference-link-discrepancies/1"
POLICY = "reference_link_reader"
COMMITMENT = "reference_link_discrepancies_sha256"
EXTENSIONS = frozenset({".html", ".htm"})
MAX_BYTES = 1024 * 1024
MAX_CHARS = 512 * 1024
MIN_REFERENCES = 8
MAX_REFERENCES = 512
MAX_LINKS = 2048
MAX_FINDINGS = 12


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def _span(span, value, characters):
    return (isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span)
            and 0 <= span[0] < span[1] <= characters and span[1] - span[0] == len(str(value)))


def validate(document):
    source = document["source"]
    if KEY not in document and COMMITMENT not in source:
        return
    coverage = document.get("coverage") or {}
    record = document.get(KEY)
    fields = {"schema", "decoded_characters", "references", "checked_links", "mismatches", "findings",
              "authorship_confidence_contribution", "destinations_contacted"}
    if (source.get("method") != "html" or coverage.get(HTML_STATIC_KEY) != HTML_STATIC_POLICY
            or not isinstance(record, dict) or record.keys() != fields or record["schema"] != SCHEMA
            or canonical_hash(record) != source.get(COMMITMENT)
            or any(type(record[field]) is not int or record[field] != 0
                   for field in ("authorship_confidence_contribution", "destinations_contacted"))):
        raise ValueError("Reference link evidence differs from the committed HTML inspection")
    chars, references, checked, mismatches = (record[k] for k in ("decoded_characters", "references",
                                                                  "checked_links", "mismatches"))
    findings = record["findings"]
    if (any(type(x) is not int for x in (chars, references, checked, mismatches))
            or not 0 < chars <= MAX_CHARS or not MIN_REFERENCES <= references <= MAX_REFERENCES
            or not 0 < mismatches <= checked <= MAX_LINKS or not isinstance(findings, list)
            or len(findings) != min(mismatches, MAX_FINDINGS)):
        raise ValueError("Reference link counts differ from their saved positions")
    previous = 0
    for row in findings:
        if not isinstance(row, dict) or row.keys() != {"label_number", "target_number", "label_range", "target_range"}:
            raise ValueError("Invalid reference link positions")
        label, target = row["label_number"], row["target_number"]
        marker, destination = row["label_range"], row["target_range"]
        if (type(label) is not int or type(target) is not int or not 1 <= label <= 999
                or not 1 <= target <= references or label == target or not _span(marker, label, chars)
                or not _span(destination, target, chars) or not previous <= marker[0]
                or not (marker[1] <= destination[0] or destination[1] <= marker[0])):
            raise ValueError("Invalid reference link numbers or positions")
        previous = marker[1]


def verify_record(document, record, *, commitment):
    validate(document)
    if document["source"].get(COMMITMENT) != commitment or document.get(KEY) != record:
        raise ValueError("Reference link evidence differs from the frozen original-file inspection")
