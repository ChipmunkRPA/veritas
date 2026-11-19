# SPDX-License-Identifier: Apache-2.0
"""Verify saved-link evidence and its commitments without parsing the original."""

from __future__ import annotations

import re

from app.core.canonical import canonical_hash

SCHEMA = "document-saved-link-hosts/1"
KEY = "saved_link_hosts"
COMMITMENT = "saved_link_hosts_sha256"
POLICY = "saved_link_hosts_reader"
LOCATIONS = {"html": "decoded_html", "email": "decoded_email_html", "python-docx": "word_paragraph"}
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_HTML_CHARS = 250_000
MAX_LINKS = 512
MAX_FINDINGS = 12
MAX_URL = 4096
COUNT_KEYS = {"unsupported_label", "unsupported_target", "same_host", "registered_wrapper", "related_hostname", "different_host"}
DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_CONVENTIONAL = frozenset(
    {"www", "mail", "docs", "help", "support", "portal", "login", "account", "accounts"}
)


def related_root(host: str) -> str:
    prefix, _, rest = host.partition(".")
    return rest if prefix in _CONVENTIONAL and "." in rest else host


def from_result(result):
    return result.get("manifest", {}).get("sampling", {}).get(COMMITMENT)


def _canonical_host(value):
    if not isinstance(value, str) or not 0 < len(value) <= 253:
        return False
    pieces = value.split(".")
    return (len(pieces) >= 2 and all(DNS_LABEL.fullmatch(piece) for piece in pieces)
            and any(ch.isalpha() for ch in pieces[-1]) and pieces[-1] not in {"local", "localhost", "internal"})


def _valid_location(location, method):
    kind = LOCATIONS[method]
    fields = {"kind", "ranges"}
    if kind == "word_paragraph":
        fields.add("body_path")
        path = location.get("body_path")
        if (not isinstance(path, list) or len(path) != 1 or type(path[0]) is not int
                or not 0 <= path[0] < MAX_FILE_BYTES):
            return False
    elif kind == "decoded_email_html":
        fields.add("mime_part")
        part = location.get("mime_part")
        if not isinstance(part, str) or re.fullmatch(r"1(?:\.[1-9][0-9]{0,6}){0,24}", part) is None:
            return False
    if location.keys() != fields or location["kind"] != kind:
        return False
    ranges = location["ranges"]
    if not isinstance(ranges, list) or not 0 < len(ranges) <= MAX_URL or (kind == "word_paragraph" and len(ranges) != 1):
        return False
    limit = MAX_FILE_BYTES if kind == "word_paragraph" else MAX_HTML_CHARS
    edge = -1
    for item in ranges:
        if (not isinstance(item, list) or len(item) != 2 or any(type(i) is not int for i in item)
                or not edge < item[0] < item[1] <= limit):
            return False
        edge = item[1]
    return True


def validate(document):
    source = document["source"]
    if KEY not in document and COMMITMENT not in source:
        return
    record = document.get(KEY)
    method = source.get("method")
    if (not isinstance(record, dict)
            or record.keys() != {"schema", "status", "method", "offset_unit", "counts", "findings", "omitted",
                                 "authorship_confidence_contribution", "destinations_contacted"}
            or record["schema"] != SCHEMA or record["status"] != "checked" or record["offset_unit"] != "unicode_codepoint"
            or record["method"] != method or method not in LOCATIONS or canonical_hash(record) != source.get(COMMITMENT)
            or any(type(record[key]) is not int or record[key] != 0
                   for key in ("authorship_confidence_contribution", "destinations_contacted"))):
        raise ValueError("Saved link evidence differs from the committed source inspection")
    counts, findings = record["counts"], record["findings"]
    if (not isinstance(counts, dict) or counts.keys() - COUNT_KEYS
            or any(type(count) is not int or not 0 < count <= MAX_LINKS for count in counts.values())
            or sum(counts.values()) > MAX_LINKS or not isinstance(findings, list) or not findings
            or len(findings) != min(MAX_FINDINGS, counts.get("different_host", 0))
            or type(record["omitted"]) is not int or record["omitted"] != counts["different_host"] - len(findings)):
        raise ValueError("Saved link counts differ from their retained locations")
    seen = set()
    selected_email = (document.get("coverage") or {}).get("email") if method == "email" else None
    for row in findings:
        if (not isinstance(row, dict) or row.keys() != {"label_host", "saved_host", "location"}
                or not all(_canonical_host(row[key]) for key in ("label_host", "saved_host"))
                or related_root(row["label_host"]) == related_root(row["saved_host"])
                or not isinstance(row["location"], dict) or not _valid_location(row["location"], method)):
            raise ValueError("Saved link evidence has an invalid hostname or source position")
        if selected_email and row["location"]["mime_part"] != selected_email.get("body_part"):
            raise ValueError("Saved link evidence comes from outside the selected email body")
        loc = canonical_hash(row["location"])
        if loc in seen:
            raise ValueError("Saved link evidence repeats a source position")
        seen.add(loc)


def verify_record(document, record, *, commitment):
    validate(document)
    if document["source"].get(COMMITMENT) != commitment or document.get(KEY) != record:
        raise ValueError("Saved link evidence differs from the frozen original-file inspection")
