"""Source-bound local sentence selection using the frozen matched full-window fit."""
from __future__ import annotations

import math
import re

from app.core.canonical import canonical_hash
from app.evidence import document_scope

SCHEMA = "document-matched-context-review/1"
KEY = "matched_context_review"
WINDOW_KEY = "matched_context_selected"
DETECTOR = "txt.four_class"
MODEL = "hf.wasitaigenerated.text_small"
REVISION = "f1795c86806e6838d4afa33d0b1427f8430c9615"
MIN_WORDS = 68
MAX_WORDS = 134
MIN_FOCUS_WORDS = 12

# Frozen matched_full_window arm in the published NIST evaluation. The USGS
# transfer reused these five aggregate scalars without refitting.
FIT_SHA256 = "7ecab534f10c7701810a237828f00079e81861282ade9034909ea64477f84054"
INTERCEPT = 0.0009003075900099412
COEFFICIENT = 0.16764671212110518
MEAN = -5.2738908547691015
SCALE = 2.4031803482622403
POINT = 0.27759856955436335


def matches(detector_id: str, version: str, revisions: dict) -> bool:
    return detector_id == DETECTOR and version == "1.0.0" and revisions.get(MODEL) == REVISION


def selected(response: float) -> bool:
    if not math.isfinite(response) or not 0 <= response <= 1:
        raise ValueError("Matched writing review has an invalid recorded response")
    p = min(max(response, 1e-6), 1 - 1e-6)
    log_odds = math.log(p / (1 - p))
    return INTERCEPT + COEFFICIENT * ((log_odds - MEAN) / SCALE) > POINT


def _scope(text: str, mapping: list[dict], method: str) -> bool:
    if method != "plain" or len(mapping) != 1 or mapping[0].get("kind") != "text":
        return False
    if (mapping[0].get("start"), mapping[0].get("end")) != (0, len(text)):
        return False
    if not MIN_WORDS <= len(text.split(None, MAX_WORDS)) <= MAX_WORDS:
        return False
    from app.detectors.text.extract import _ocr_language

    language = _ocr_language(text)
    return language.get("code") == "en" and language.get("status") == "supported"


def _geometry(sentences) -> list[tuple[dict, list[dict]]]:
    if len(sentences) != 4 or any(sentence.paragraph != 0 for sentence in sentences):
        return []
    spans = [{"start": sentence.start, "end": sentence.end} for sentence in sentences]
    return [(spans[i], spans[i - 1:i + 2]) for i in (1, 2) if sentences[i].n_words >= MIN_FOCUS_WORDS]


def one_complete_window(coverage: dict) -> bool:
    count = coverage.get("tokens")
    return (
        type(count) is int and 0 < count <= 512
        and type(coverage.get("token_windows")) is int and coverage["token_windows"] == 1
        and coverage.get("omitted_tokens", 0) == 0 and coverage.get("covered_tokens", count) == count
    )


def capture(doc, sentences, windows: list[dict]) -> dict | None:
    """Use existing scores only. Incomplete assessment leaves ordinary review in place."""
    geometry = _geometry(sentences)
    if not geometry:
        return None
    contexts = []
    for focus, spans in geometry:
        window = next((entry for entry in windows if entry["focus"] == focus), None)
        if window is None or window["spans"] != spans or not one_complete_window(window["token_coverage"]):
            return None
        raw = window["score"]
        contexts.append({"focus": focus, "spans": spans, "input_sha256": window["input_sha256"],
                         "score_exact_hex": float(raw).hex(), "token_coverage": window["token_coverage"],
                         "selected": selected(raw)})
    if not _scope(doc.text, doc.source_map, doc.meta.get("method")):
        return None
    return {"schema": SCHEMA, "contexts": contexts}


def _public_ref(ref: dict) -> dict:
    return {"schema": SCHEMA, "signal_ref": ref, "authorship_confidence_contribution": 0}


def from_signal(signal: dict, ref: dict) -> dict | None:
    record = (signal.get("details", {}).get("text_evidence") or {}).get(KEY)
    if record is None:
        return None
    if (record.get("schema") != SCHEMA or not matches(
        signal.get("detector_id"), signal.get("detector_version"), signal.get("model_revisions") or {}
    )):
        raise ValueError("Matched writing review does not belong to the pinned check")
    contexts = record["contexts"]
    return {**_public_ref(ref), "contexts": contexts, "eligible_sentences": len(contexts),
            "selected_sentences": sum(row["selected"] for row in contexts)}


def context_for(coverage: dict, focus: dict) -> dict | None:
    return next((row for row in coverage["contexts"] if row["focus"] == focus), None)


def apply_to_window(window: dict, focus: dict, coverage: dict) -> None:
    row = context_for(coverage, focus)
    window[WINDOW_KEY] = row is not None and row["selected"]
    if row is not None:
        window["score_exact_hex"] = row["score_exact_hex"]
        if any(window[key] != row[key] for key in ("spans", "input_sha256", "token_coverage")):
            raise ValueError("Matched writing review differs from the recorded inference context")


def packet_record(windows: list[dict]) -> dict | None:
    match = next((window for window in windows if window.get(WINDOW_KEY) is True), None)
    return _public_ref(match["signal_ref"]) if match is not None else None


def _validate_context(row: dict, size: int) -> None:
    if set(row) != {"focus", "spans", "input_sha256", "score_exact_hex", "token_coverage", "selected"}:
        raise ValueError("Matched writing review has incomplete context fields")
    spans = [document_scope.check_range(span, size) for span in row["spans"]]
    try:
        raw = float.fromhex(row["score_exact_hex"])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Matched writing review has an invalid exact response") from error
    if (len(spans) != 3 or row["focus"] != row["spans"][1]
            or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
            or any(set(span) != {"start", "end"} for span in row["spans"])
            or not isinstance(row["input_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", row["input_sha256"])
            or raw.hex() != row["score_exact_hex"] or row["selected"] is not selected(raw)
            or not one_complete_window(row["token_coverage"])):
        raise ValueError("Matched writing review has invalid bounds, coverage or decision")


def validate(document: dict, *, published_checks: set[str] | None = None) -> None:
    coverage = document.get("coverage", {})
    packets = document["packets"]
    if document["version"] != document_scope.MATCHED_VERSION:
        if KEY in coverage or any(KEY in packet or any(WINDOW_KEY in w for w in packet["supporting_windows"])
                                  for packet in packets):
            raise ValueError("Historical document evidence cannot contain matched writing review")
        return
    review = coverage[KEY]
    contexts, ref = review["contexts"], review["signal_ref"]
    if (set(review) != {*_public_ref(ref), "contexts", "eligible_sentences", "selected_sentences"}
            or review["schema"] != SCHEMA or type(review["authorship_confidence_contribution"]) is not int
            or review["authorship_confidence_contribution"] != 0
            or type(review["eligible_sentences"]) is not int or not 1 <= len(contexts) == review["eligible_sentences"] <= 2
            or document["source"].get("method") != "plain" or set(ref) != {"id", "display_id"}
            or not isinstance(ref["id"], str) or not ref["id"]
            or published_checks is not None and ref["id"] not in published_checks):
        raise ValueError("Matched writing review has invalid scope or check reference")
    size = document["source"]["chars"]
    for row in contexts:
        _validate_context(row, size)
    if len(contexts) == 2 and contexts[0]["spans"][1:] != contexts[1]["spans"][:2]:
        raise ValueError("Matched writing review contexts do not share their actual neighbours")
    expected = {(row["focus"]["start"], row["focus"]["end"]) for row in contexts if row["selected"]}
    if type(review["selected_sentences"]) is not int or review["selected_sentences"] != len(expected):
        raise ValueError("Matched writing review selection count differs from the recorded responses")
    displayed = set()
    for packet in packets:
        windows = packet["supporting_windows"]
        focus = {key: packet["focus"][key] for key in ("start", "end")}
        for window in windows:
            same_check = window["signal_ref"]["id"] == ref["id"]
            if not same_check:
                if WINDOW_KEY in window:
                    raise ValueError("Matched writing selection is on an unrelated check")
                continue
            row = context_for(review, focus)
            if (window["signal_ref"] != ref or window.get(WINDOW_KEY) is not (row is not None and row["selected"])
                    or (row is not None and any(window.get(key) != row[key] for key in
                                               ("spans", "input_sha256", "score_exact_hex", "token_coverage")))):
                raise ValueError("Matched writing review differs from its supporting window")
            if row is not None:
                raw, display = float.fromhex(row["score_exact_hex"]), window["score"]
                if type(display) not in {float, int} or not 0 <= display <= 1 or abs(raw - display) > 0.00000051:
                    raise ValueError("Matched writing review differs from the displayed raw response")
        marker = packet_record(windows)
        if packet.get(KEY) != marker or (marker is None and KEY in packet):
            raise ValueError("Matched writing packet has an inconsistent review marker")
        if marker is not None:
            position = (focus["start"], focus["end"])
            if position in displayed:
                raise ValueError("Matched writing review repeated a selected sentence")
            displayed.add(position)
    if displayed != expected:
        raise ValueError("Matched writing review is missing a selected sentence")


def verify_source(document: dict, text: str, mapping: list[dict], *, result: dict | None = None) -> None:
    validate(document)
    if document["version"] != document_scope.MATCHED_VERSION:
        return
    from app.detectors.text.extract import Document, text_hash
    from app.detectors.text.segment import segment

    review = document["coverage"][KEY]
    method = document["source"]["method"]
    if not _scope(text, mapping, method):
        raise ValueError("Matched writing review is outside its verified source scope")
    geometry = _geometry(segment(Document(text, [(0, len(text))], {"method": method}, mapping)))
    contexts = review["contexts"]
    if [(row["focus"], row["spans"]) for row in contexts] != geometry or any(
        text_hash(" ".join(text[span["start"]:span["end"]] for span in row["spans"])) != row["input_sha256"]
        for row in contexts
    ):
        raise ValueError("Matched writing review differs from the actual source sentences")
    if result is not None:
        from app.api.public_projection import alias_signal

        signal = next((s for s in result["signals"] if alias_signal(s["id"]) == review["signal_ref"]["id"]), None)
        if signal is None or signal.get("display_id") != review["signal_ref"]["display_id"] or (
            canonical_hash(from_signal(signal, review["signal_ref"])) != canonical_hash(review)
        ):
            raise ValueError("Matched writing review differs from the frozen analysis responses")
