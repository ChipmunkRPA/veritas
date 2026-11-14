"""Persist the completion of core writing checks without publishing their identities."""

from app.core.canonical import canonical_hash

KEY = "writing_check_coverage"
COMMITMENT = "writing_check_coverage_sha256"
SCHEMA = "document-writing-check-coverage/1"
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


def capture(entries, classifier_ids, ratio_id):
    runs = {}
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
    return {"schema": SCHEMA, "checks": runs} if runs else None


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def validate(record):
    if (not isinstance(record, dict) or set(record) != {"schema", "checks"} or record["schema"] != SCHEMA
            or not isinstance(record["checks"], dict) or not record["checks"]
            or set(record["checks"]) - {"passage_classifier", "writing_pattern_ratio"}
            or any(not isinstance(value, str) or value not in STATES for value in record["checks"].values())):
        raise ValueError("Invalid writing check coverage")
    return record


def commitment(record):
    return canonical_hash(validate(record)) if record is not None else None


def verify(evidence, expected):
    if commitment((evidence.get("coverage") or {}).get(KEY)) != expected:
        raise ValueError("Writing check coverage differs from the frozen analysis")
