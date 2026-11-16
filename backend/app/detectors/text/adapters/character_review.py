"""Optional second reading for exact contexts with selected Unicode characters."""

from __future__ import annotations

from app.core.config import settings
from app.detectors.text.adapters import agreement
from app.evidence import document_character_review as proof
from app.evidence import document_matched_context

STAGED = "character_adjusted_primary"


def enabled():
    return settings.text_character_adjusted_review and agreement.enabled()


def record_primary(classifier, doc, windows, ctx):
    inputs, records = [], []
    remaining = ctx.deadline.remaining()
    budget = 8 if remaining is None else min(8, remaining - 2)
    if budget <= 0:
        return []
    for window in windows:
        source_coverage = window["token_coverage"]
        if not document_matched_context.one_complete_window(source_coverage):
            continue
        material = proof.geometry(doc.text, window["spans"], doc.source_map, doc.meta)
        if material is None:
            continue
        row, alternate = material
        if row["source_input_sha256"] != window["input_sha256"] or window["focus"] != row["spans"][1]:
            raise ValueError("Character view differs from the recorded writing context")
        inputs.append(alternate)
        records.append(row | {"as_submitted": {"response_exact_hex": float(window["score"]).hex(), "tokens": source_coverage["tokens"]}})
        if len(inputs) >= proof.MAX_RANGES:
            break
    if not inputs:
        return []
    optional = ctx.deadline.nested(budget)
    responses = classifier.predict(inputs, batch=agreement.BATCH, check=optional.check)
    return [row | {"adjusted_primary": {"response_exact_hex": float(value).hex(), "tokens": coverage["tokens"]}}
            for row, value, coverage in zip(records, responses, classifier.last_coverage, strict=True)
            if document_matched_context.one_complete_window(coverage)]


def finalize(ctx, detected, decoded):
    """Read the same adjusted bytes through the existing independent checkpoint."""
    rows = []
    for entry in detected.get("detectors", ()):
        details = entry.get("details", {})
        rows.extend(details.pop(STAGED, ()))
    if not rows or not enabled():
        return None
    inputs, windows, pos = [], [], 0
    for row in rows:
        rebuilt = proof.geometry(decoded["text"], row["spans"], decoded["source_map"], decoded["meta"])
        if rebuilt is None or any(row[key] != value for key, value in rebuilt[0].items()):
            raise ValueError("Character view changed before the second reading")
        alternate = rebuilt[1]
        inputs.append(alternate)
        windows.append({"unit_ref": f"character:{pos}", "spans": [{"start": pos, "end": pos + len(alternate)}],
                        "input_sha256": row["adjusted_input_sha256"]})
        pos += len(alternate)
    readings = agreement._review(ctx, "".join(inputs), windows)
    verified = []
    for row, window in zip(rows, windows, strict=True):
        second = readings.get(window["unit_ref"])
        if second is None or second["input_sha256"] != row["adjusted_input_sha256"]:
            continue
        second = {"response_exact_hex": second["response_exact_hex"], "tokens": second["tokens"]}
        verified.append(row | {"adjusted_second": second,
                               "state": proof.response_state(row["as_submitted"], row["adjusted_primary"], second)})
    return {"schema": proof.SCHEMA, "input_view": proof.INPUT_VIEW, "ranges": verified,
            "authorship_confidence_contribution": 0} if verified else None
