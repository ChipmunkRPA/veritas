"""Persist the completion of core writing checks without publishing their identities."""

import re

from app.core.canonical import canonical_hash

KEY = "writing_check_coverage"
COMMITMENT = "writing_check_coverage_sha256"
SCHEMA = "document-writing-check-coverage/1"
REASONS_SCHEMA = "document-writing-check-coverage/2"
CLAUSE_SCHEMA = "document-writing-check-coverage/3"
SHA256 = re.compile(r"[a-f0-9]{64}")
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
    "too_little_after_cover": ("not_applicable", "Too little text remained after pages with covered text were excluded."),
    "unsupported_word_revisions": ("not_applicable", "Some Word revisions could change which text is visible."),
    "too_little_after_word_revisions": ("not_applicable", "Too little text remained after applying the supported Word revisions."),
    "no_text": ("not_applicable", "There was no extracted text for this check to read."),
}


def capture(entries, classifier_ids, ratio_id, *, language=None, verified_credentials=False, language_witness=None):
    from app.detectors.text.language_scope import (
        MESSAGES,
        TOO_LITTLE_AFTER_COVER,
        TOO_LITTLE_AFTER_WORD_REVISIONS,
        UNSUPPORTED_WORD_REVISIONS,
    )

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
                            "unknown": "language_unclear", TOO_LITTLE_AFTER_COVER: "too_little_after_cover",
                            UNSUPPORTED_WORD_REVISIONS: "unsupported_word_revisions",
                            TOO_LITTLE_AFTER_WORD_REVISIONS: "too_little_after_word_revisions"}[language]
            if code is not None:
                reasons[name] = code
            else:
                reasons.pop(name, None)
    if not runs:
        return None
    if language_witness is not None:
        return validate({"schema": CLAUSE_SCHEMA, "checks": runs, "reasons": reasons,
                         "language_witness": language_witness})
    return {"schema": REASONS_SCHEMA, "checks": runs, "reasons": reasons} if reasons else {"schema": SCHEMA, "checks": runs}


def from_result(result):
    return (result.get("manifest", {}).get("sampling") or {}).get(KEY)


def validate(record):
    fields = {SCHEMA: {"schema", "checks"}, REASONS_SCHEMA: {"schema", "checks", "reasons"},
              CLAUSE_SCHEMA: {"schema", "checks", "reasons", "language_witness"}}
    if (not isinstance(record, dict) or record.get("schema") not in fields
            or set(record) != fields[record["schema"]]
            or not isinstance(record["checks"], dict) or not record["checks"]
            or set(record["checks"]) - {"passage_classifier", "writing_pattern_ratio"}
            or any(not isinstance(value, str) or value not in STATES for value in record["checks"].values())):
        raise ValueError("Invalid writing check coverage")
    if record["schema"] in (REASONS_SCHEMA, CLAUSE_SCHEMA):
        reasons = record["reasons"]
        if (not isinstance(reasons, dict) or not reasons or not reasons.keys() <= record["checks"].keys()
                or any(not isinstance(reason, str) or reason not in REASONS or REASONS[reason][0] != record["checks"][key]
                       for key, reason in reasons.items())):
            raise ValueError("Invalid writing check coverage reason")
    if record["schema"] == CLAUSE_SCHEMA:
        if set(record["reasons"]) != set(record["checks"]) or set(record["reasons"].values()) != {"mixed_language"}:
            raise ValueError("Language evidence requires skipped English writing checks")
        validate_witness(record["language_witness"])
    return record


def commitment(record):
    return canonical_hash(validate(record)) if record is not None else None


def verify(evidence, expected):
    record = (evidence.get("coverage") or {}).get(KEY)
    if commitment(record) != expected:
        raise ValueError("Writing check coverage differs from the frozen analysis")
    if record and (witness := record.get("language_witness")) is not None:
        chars = evidence.get("source", {}).get("chars")
        if type(chars) is not int or witness["context"]["end"] > chars:
            raise ValueError("Language evidence falls outside the extracted document")


def freeze_witness(text, match):
    if match is None:
        return None
    from app.detectors.text.extract import text_hash

    def span(start, end):
        return {"start": start, "end": end, "sha256": text_hash(text[start:end])}

    return {"sentence": match["sentence"] + 1, "range": span(match["start"], match["end"]),
            "context": span(match["context_start"], match["context_end"])}


def validate_witness(witness):
    if (not isinstance(witness, dict) or set(witness) != {"sentence", "range", "context"}
            or type(witness["sentence"]) is not int or witness["sentence"] < 1):
        raise ValueError("Invalid language evidence")
    for span in (witness["range"], witness["context"]):
        if (not isinstance(span, dict) or set(span) != {"start", "end", "sha256"}
                or type(span["start"]) is not int or type(span["end"]) is not int
                or not 0 <= span["start"] < span["end"]
                or not isinstance(span["sha256"], str) or not SHA256.fullmatch(span["sha256"])):
            raise ValueError("Invalid language evidence range")
    narrow, context = witness["range"], witness["context"]
    if (not context["start"] <= narrow["start"] < narrow["end"] <= context["end"]
            or (context["start"], context["end"]) == (narrow["start"], narrow["end"])):
        raise ValueError("The two language readings do not cover nested samples")


def verify_source(evidence, text, sentences):
    record = (evidence.get("coverage") or {}).get(KEY) or {}
    if (witness := record.get("language_witness")) is None:
        return

    from app.detectors.text.extract import text_hash
    from app.detectors.text.language_clauses import NARROW_WORDS, WIDE_WORDS
    from app.detectors.text.language_scope import WORDS

    validate_witness(witness)
    index = witness["sentence"] - 1
    context = witness["context"]
    if index >= len(sentences) or not sentences[index]["start"] <= context["start"] < context["end"] <= sentences[index]["end"]:
        raise ValueError("Language evidence differs from the recorded sentence")
    for span, width in ((witness["range"], NARROW_WORDS), (context, WIDE_WORDS)):
        excerpt = text[span["start"]:span["end"]]
        if text_hash(excerpt) != span["sha256"] or len(WORDS.findall(excerpt)) != width:
            raise ValueError("Language evidence differs from the recorded original text")


def witness_description(witness):
    span, context = witness["range"], witness["context"]
    return (f"Two overlapping samples in extracted sentence {witness['sentence']} indicated another language. "
            f"Marked characters [{span['start']}, {span['end']}) and supporting context "
            f"[{context['start']}, {context['end']}) use zero-based extracted-text offsets.")
