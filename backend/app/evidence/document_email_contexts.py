"""Source roles represented in retained complete email classifier inputs."""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter

from app.core.canonical import canonical_hash
from app.detectors.text.extract import text_hash
from app.evidence.document_scope import check_range, mapped_email_html

KEY = "email_classifier_input_scope"
SCHEMA = "document-email-classifier-input-scope/1"
COMMITMENT = "email_classifier_input_scope_sha256"
MAX_WINDOWS = 120
MAX_SPANS = 9
ROLES = ("body", "quoted", "signature", "mixed", "unmapped")
SOURCE_ROLES = frozenset(ROLES[:3])
SHA256 = re.compile(r"[0-9a-f]{64}")


def _email_scope(email, mapping):
    if email.get("body_content_type") != "text/plain" and not mapped_email_html(email):
        return "unmapped"
    if not mapping or any(
        block.get("kind") != "email"
        or block.get("mime_part") != email.get("body_part")
        or block.get("quote_role") not in SOURCE_ROLES
        or not block.get("source_ranges")
        for block in mapping
    ):
        return "unmapped"
    return "mapped"


def _role(spans, mapping, starts, status):
    if status == "unmapped":
        return "unmapped"
    found = set()
    for start, end in spans:
        idx = bisect_right(starts, start) - 1
        if idx < 0 or end > mapping[idx]["end"]:
            return "unmapped"
        found.add(mapping[idx]["quote_role"])
    return next(iter(found)) if len(found) == 1 else "mixed"


def _complete(window):
    coverage = window.get("token_coverage") or {}
    tokens = coverage.get("tokens")
    return (
        type(tokens) is int
        and tokens > 0
        and coverage.get("covered_tokens", tokens) == tokens
        and not coverage.get("omitted_tokens", 0)
    )


def build(result, text, mapping, email):
    from app.api.public_projection import alias_signal

    if not email or not email.get("body_part"):
        return None
    status = _email_scope(email, mapping)
    starts = [block["start"] for block in mapping]
    rows, seen = [], set()
    for signal in result.get("signals", []):
        evidence = (signal.get("details") or {}).get("text_evidence") or {}
        label = signal.get("display_id")
        if (
            signal.get("status") != "ok"
            or not isinstance(label, str)
            or not re.fullmatch(r"S[0-9]{2,6}", label)
            or evidence.get("version") not in {"scored-context/1", "scored-context/2"}
        ):
            continue
        identity = (signal.get("detector_id", signal["id"]), canonical_hash(evidence))
        if identity in seen:
            continue
        seen.add(identity)
        counts = Counter(dict.fromkeys(ROLES, 0))
        windows = []
        incomplete = 0
        for window in evidence.get("windows", []):
            if window.get("score_kind", "classifier_probability") != "classifier_probability":
                continue
            if not _complete(window) or len(window["spans"]) > MAX_SPANS:
                incomplete += 1
                continue
            spans = [list(check_range(span, len(text))) for span in window["spans"]]
            pin = window["input_sha256"]
            if (
                not spans
                or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
                or (text_hash(" ".join(text[a:b] for a, b in spans)) != pin)
            ):
                raise ValueError("Email input context differs from the frozen text")
            if len(windows) < MAX_WINDOWS:
                role = _role(spans, mapping, starts, status)
                windows.append({"spans": spans, "input_sha256": pin, "role": role})
                counts[role] += 1
            else:
                incomplete += 1
        if windows or incomplete:
            rows.append(
                {
                    "signal_ref": {"id": alias_signal(signal["id"]), "display_id": label},
                    "counts": dict(counts),
                    "windows": windows,
                    "unlisted": incomplete,
                }
            )
    return {"schema": SCHEMA, "mime_part": email["body_part"], "mapping": status, "checks": rows}


def _range(value, size):
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(n) is not int for n in value)
        or not 0 <= value[0] < value[1] <= size
    ):
        raise ValueError("Invalid email classifier input range")


def validate(document, *, published_checks=None):
    coverage = document.get("coverage") or {}
    record = coverage.get(KEY)
    if record is None:
        return None
    email = coverage["email"]
    if (
        set(record) != {"schema", "mime_part", "mapping", "checks"}
        or record["schema"] != SCHEMA
        or document["source"]["method"] != "email"
        or not isinstance(record["mime_part"], str)
        or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", record["mime_part"])
        or record["mime_part"] != email["body_part"]
        or not isinstance(record["checks"], list)
        or record["mapping"] not in {"mapped", "unmapped"}
        or (
            record["mapping"] == "mapped"
            and email.get("body_content_type") != "text/plain"
            and not mapped_email_html(email)
        )
    ):
        raise ValueError("Email input scope does not match the selected MIME body")
    size = document["source"]["chars"]
    checks = {check["signal_ref"]["id"]: check for check in coverage.get("checks", [])}
    seen = set()
    for row in record["checks"]:
        ref, windows, unlisted = (row[k] for k in ("signal_ref", "windows", "unlisted"))
        check = checks.get(ref["id"], {})
        if (
            set(row) != {"signal_ref", "counts", "windows", "unlisted"}
            or check.get("signal_ref") != ref
            or published_checks is not None
            and ref["id"] not in published_checks
            or ref["id"] in seen
            or type(unlisted) is not int
            or unlisted < 0
            or not 0 <= len(windows) <= MAX_WINDOWS
            or len(windows) + unlisted > check["scored_windows"]
            or not (windows or unlisted)
        ):
            raise ValueError("Invalid email classifier scope")
        seen.add(ref["id"])
        counts = Counter(dict.fromkeys(ROLES, 0))
        for window in windows:
            if (
                set(window) != {"spans", "input_sha256", "role"}
                or not SHA256.fullmatch(window["input_sha256"])
                or window["role"] not in ROLES
                or not 0 < len(window["spans"]) <= MAX_SPANS
                or (record["mapping"] == "unmapped" and window["role"] != "unmapped")
            ):
                raise ValueError("Invalid email classifier input")
            for span in window["spans"]:
                _range(span, size)
            if any(a[1] > b[0] for a, b in zip(window["spans"], window["spans"][1:])):
                raise ValueError("Overlapping email classifier input ranges")
            counts[window["role"]] += 1
        if any(type(n) is not int for n in row["counts"].values()) or row["counts"] != dict(counts):
            raise ValueError("Email classifier input counts differ from the listed roles")
    return record


def verify_source(document, text, mapping, *, result=None):
    record = validate(document)
    if record is None:
        return 0
    email = document["coverage"]["email"]
    status = _email_scope(email, mapping)
    if record["mapping"] != status:
        raise ValueError("Email role mapping differs from the original source")
    starts = [block["start"] for block in mapping]
    amount = 0
    for row in record["checks"]:
        for window in row["windows"]:
            spans = window["spans"]
            if (
                _role(spans, mapping, starts, status) != window["role"]
                or text_hash(" ".join(text[a:b] for a, b in spans)) != window["input_sha256"]
            ):
                raise ValueError("Email classifier role or input differs from the original source")
            amount += 1
    if result is not None and record != build(result, text, mapping, email):
        raise ValueError("Email classifier input scope differs from the recorded analysis")
    return amount


def validate_commitment(document, expected, *, published_checks):
    record = validate(document, published_checks=published_checks)
    if expected != (canonical_hash(record) if record is not None else None):
        raise ValueError("Email classifier input scope differs from its published result commitment")
