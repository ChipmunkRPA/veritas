"""Persist the completion of core writing checks without publishing their identities."""

from app.core.canonical import canonical_hash

KEY = "writing_check_coverage"
COMMITMENT = "writing_check_coverage_sha256"
SCHEMA = "document-writing-check-coverage/1"
REASONS_SCHEMA = "document-writing-check-coverage/2"
STATES = frozenset({"completed", "not_applicable", "skipped", "failed", "timed_out"})
LABELS = {
    "completed": "Completed",
    "not_applicable": "Outside this check’s scope",
    "skipped": "Skipped",
    "failed": "Failed",
    "timed_out": "Timed out",
}
STATUS = {"ok": "completed", "not_applicable": "not_applicable", "skipped": "skipped",
          "failed": "failed", "timeout": "timed_out"}
REASONS = {
    "memory_budget": ("skipped", "Available analysis memory was too low to run this check."),
    "unavailable": ("skipped", "This check was unavailable on the analysis server."),
    "verified_credentials": ("skipped", "Verified Content Credentials established AI involvement."),
    "unsupported_language": ("not_applicable", "The extracted writing was outside English coverage."),
    "mixed_language": ("not_applicable", "The extracted text contained English and another language."),
    "language_unclear": ("not_applicable", "The check could not establish the language of the extracted writing."),
    "no_text": ("not_applicable", "There was no extracted text for this check to read."),
}


def capture(entries, classifier_ids, ratio_id, *, language=None, verified_credentials=False):
    from app.detectors.text.language_scope import MESSAGES

    runs, reasons = {}, {}
    for entry in entries:
        detector_id = entry["detector_id"]
        if detector_id not in classifier_ids and detector_id != ratio_id:
            continue
        name = "passage_classifier" if detector_id in classifier_ids else "writing_pattern_ratio"
        state = STATUS.get(entry["status"])
        if state is None:
            raise ValueError("Unknown writing check outcome")
        if name not in runs or state == "completed":
            runs[name] = state
            reason = entry.get("reason")
            code = None
            if state == "skipped":
                code = {"memory budget": "memory_budget", "model not installed": "unavailable"}.get(reason)
                if verified_credentials and (entry.get("details") or {}).get("provenance_gate"):
                    code = "verified_credentials"
            elif state == "not_applicable":
                if reason == "no extracted text":
                    code = "no_text"
                elif language in MESSAGES and reason == MESSAGES[language][1]:
                    code = {"unsupported": "unsupported_language", "mixed": "mixed_language",
                            "unknown": "language_unclear"}[language]
            if code is not None:
                reasons[name] = code
            else:
                reasons.pop(name, None)
    if not runs:
        return None
    return {"schema": REASONS_SCHEMA, "checks": runs, "reasons": reasons} if reasons else {"schema": SCHEMA, "checks": runs}


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def validate(record):
    if (not isinstance(record, dict) or record.get("schema") not in (SCHEMA, REASONS_SCHEMA)
            or set(record) != ({"schema", "checks", "reasons"} if record["schema"] == REASONS_SCHEMA else {"schema", "checks"})
            or not isinstance(record["checks"], dict) or not record["checks"]
            or set(record["checks"]) - {"passage_classifier", "writing_pattern_ratio"}
            or any(not isinstance(value, str) or value not in STATES for value in record["checks"].values())):
        raise ValueError("Invalid writing check coverage")
    if record["schema"] == REASONS_SCHEMA:
        reasons = record["reasons"]
        if (not isinstance(reasons, dict) or not reasons or not reasons.keys() <= record["checks"].keys()
                or any(not isinstance(reason, str) or reason not in REASONS or REASONS[reason][0] != record["checks"][key]
                       for key, reason in reasons.items())):
            raise ValueError("Invalid writing check coverage reason")
    return record


def commitment(record):
    return canonical_hash(validate(record)) if record is not None else None


def verify(evidence, expected):
    if commitment((evidence.get("coverage") or {}).get(KEY)) != expected:
        raise ValueError("Writing check coverage differs from the frozen analysis")
