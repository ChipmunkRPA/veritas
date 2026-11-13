"""Source-bound, optional paired writing responses for an authenticated draft comparison."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from app.detectors.text.extract import _normalise, _ocr_language, from_string, text_hash
from app.detectors.text.segment import segment

SCHEMA = "chipmunk.paired-writing-review/1"
WIDE_SCHEMA = "chipmunk.paired-writing-review/2"
MAX_WORDS = 420
SCOPE = ("This experimental check compares one changed sentence and its two unchanged neighbors in the same paragraph "
         "of a short English plain-text draft. Responses are uncalibrated and carry zero authorship confidence weight.")
WIDE_SCOPE = ("This experimental check compares two adjacent changed sentences with an unchanged sentence on each side "
              "in the same paragraph of a short English plain-text draft. "
              "Responses are uncalibrated and carry zero authorship confidence weight.")
PLAIN_SCOPE = ("This experimental check compares one changed sentence or two adjacent changed sentences with an unchanged "
               "sentence on each side in the same paragraph of a short English plain-text draft. "
               "Responses are uncalibrated and carry zero authorship confidence weight.")
WORD_SCOPE = ("This experimental check compares one changed sentence and its two unchanged neighbors in the same "
              "top-level body paragraph of a short English Word draft. Responses are uncalibrated and carry zero "
              "authorship confidence weight.")
EMAIL_SCOPE = ("This experimental check compares one changed sentence and its two unchanged neighbors within one "
               "paragraph of a short English email body. The check covers only the reply body. Responses are "
               "uncalibrated and carry zero authorship confidence weight.")
PDF_SCOPE = ("This experimental check compares one changed sentence and two unchanged neighboring sentences in one "
             "text block on a single upright PDF page with selectable text and a verified reading order. Responses are "
             "uncalibrated and carry zero authorship confidence weight.")
STATUS_TEXT = {
    "outside_scope": "This draft pair falls outside the exact single-sentence comparison scope.",
    "unavailable": "The local writing checks could not complete the paired comparison.",
    "no_joint_increase": "No shared increase reached the paired check’s review level.",
    "joint_increase": "Both writing checks gave a stronger AI-like response in the analyzed version’s changed context.",
}
REVIEW_SECONDS = 30.0


@dataclass(frozen=True)
class PairedInput:
    earlier: str
    later: str
    geometry: dict


def _source(text: str):
    document = from_string(text)
    if not document.paragraphs or document.meta["words"] > MAX_WORDS:
        return None
    language = _ocr_language(document.text)
    if language.get("code") != "en" or language.get("status") != "supported":
        return None
    return document, segment(document)


def paired_input(comparison: dict, earlier: str, later: str, *, allow_wide: bool = True) -> PairedInput | None:
    """Pair an interior change. For EML, use the verified decoded body prefix from the draft comparison."""
    changes = comparison["changes"]
    if comparison["format"] == "email":
        changes = [change for change in changes if change["role"] == "body"]
    if comparison["format"] not in ("plain_text", "docx", "email", "pdf") or not changes:
        return None
    old, new = _source(earlier), _source(later)
    if old is None or new is None:
        return None
    old_doc, old_units = old
    new_doc, new_units = new
    if [unit.paragraph for unit in old_units] != [unit.paragraph for unit in new_units]:
        return None
    changed = [i for i, (a, b) in enumerate(zip(old_units, new_units, strict=True)) if a.text != b.text]
    wide = (allow_wide and comparison["format"] == "plain_text" and len(changed) == 2
            and changed[1] == changed[0] + 1)
    if len(changed) != 1 and not wide:
        return None
    if changed[0] == 0 or changed[-1] == len(old_units) - 1:
        return None
    index = changed[0]
    selected = slice(index - 1, changed[-1] + 2)
    paragraph = old_units[index].paragraph
    if any(unit.paragraph != paragraph for unit in old_units[selected]):
        return None

    def context(document, units, raw_text):
        block = document.source_map[paragraph]
        ranges = block["source_ranges"]
        offset, stop = ranges[0][2], ranges[-1][3]
        normalized, origins = _normalise(raw_text[offset:stop])
        if normalized != document.text[block["start"]:block["end"]]:
            raise ValueError("Writing comparison paragraph differs from its decoded source")
        spans = [[offset + origins[unit.start - block["start"]][0], offset + origins[unit.end - block["start"] - 1][1]]
                 for unit in units]
        if [_normalise(raw_text[start:end])[0] for start, end in spans] != [unit.text for unit in units]:
            raise ValueError("Writing comparison context differs from its decoded source")
        start, end = spans[1][0], spans[-2][1]
        model_input = " ".join(unit.text for unit in units)
        return {"start": start, "end": end, "text": raw_text[start:end],
                "context_spans": spans, "input_sha256": text_hash(model_input)}, model_input

    before, old_input = context(old_doc, old_units[selected], earlier)
    after, new_input = context(new_doc, new_units[selected], later)

    def contained(change, side, expected):
        item = change[side]
        return item is not None and expected["start"] <= item["start"] <= item["end"] <= expected["end"]

    if (any(change["role"] != "body" or not contained(change, "earlier", before)
            or not contained(change, "later", after) for change in changes)
            or not all(any(change[side]["changed_tokens"] for change in changes) for side in ("earlier", "later"))):
        return None
    if comparison["format"] == "docx":
        # Word body paragraphs are joined with blank lines, which the source mapper always splits.
        # Check that each linked edit also resolves to the same Word paragraph in its version.
        for side, focus in (("earlier", before), ("later", after)):
            anchors = set()
            for change in changes:
                item = change[side]
                paragraphs = item["paragraphs"]
                if len(paragraphs) != 1:
                    return None
                anchor = paragraphs[0]
                start = item["start"] - anchor["start"]
                if focus["context_spans"][0][0] < start:
                    return None
                anchors.add((anchor["body_index"], start))
            if len(anchors) != 1:
                return None
    if comparison["format"] == "pdf":
        for side in ("earlier", "later"):
            for change in changes:
                item = change[side]
                pages = item.get("pdf_pages") or []
                if (len(pages) != 1 or pages[0]["page"] != 1 or pages[0]["page_rotation"] != 0
                        or (item["changed_tokens"] and not pages[0]["word_regions"])):
                    return None
    return PairedInput(old_input, new_input, {"earlier": before, "later": after,
                                             "change_ids": [change["id"] for change in changes]})


def _direction(responses: list[dict]) -> str:
    if (not isinstance(responses, list) or len(responses) != 2
            or any(not isinstance(pair, dict) or set(pair) != {"earlier", "later"} for pair in responses)):
        raise ValueError("Two paired writing responses are required")

    def odds(p):
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid paired writing response")
        value = min(1 - 1e-7, max(1e-7, p))
        return value / (1 - value)

    increases = [math.log(odds(row["later"]) / odds(row["earlier"])) for row in responses]
    return "joint_increase" if all(increase >= math.log(2) for increase in increases) else "no_joint_increase"


def result(candidate: PairedInput | None, responses: list[dict] | None = None, *, draft_format: str = "plain_text",
           allow_wide: bool = True) -> dict:
    status = "outside_scope" if candidate is None else _direction(responses) if responses is not None else "unavailable"
    wide = candidate is not None and len(candidate.geometry["later"]["context_spans"]) == 4
    plain_outside = candidate is None and draft_format == "plain_text" and allow_wide
    scope = WIDE_SCOPE if wide else {"docx": WORD_SCOPE, "email": EMAIL_SCOPE, "pdf": PDF_SCOPE}.get(draft_format, SCOPE)
    value = {"schema": WIDE_SCHEMA if wide or plain_outside else SCHEMA, "status": status,
             "scope": PLAIN_SCOPE if plain_outside else scope,
             "summary": "This draft pair falls outside the exact paired sentence scope." if plain_outside else STATUS_TEXT[status],
             "authorship_confidence_contribution": 0}
    if candidate is not None:
        value.update(candidate.geometry)
    if responses is not None:
        value["responses"] = responses
    return value


def verify(comparison: dict, earlier: str, later: str, recorded: dict) -> str:
    """Rebuild the source scope and check saved response arithmetic without inference."""
    allow_wide = recorded.get("schema") == WIDE_SCHEMA
    candidate = paired_input(comparison, earlier, later, allow_wide=allow_wide)
    if candidate is None and "responses" in recorded:
        raise ValueError("An out-of-scope draft cannot carry paired responses")
    expected = result(candidate, recorded.get("responses"), draft_format=comparison["format"], allow_wide=allow_wide)
    if json.dumps(recorded, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("The paired writing check does not replay from these source positions and saved responses")
    return expected["status"]


def score(candidate: PairedInput) -> list[dict] | None:
    """Use the two pinned local checks, one resident lease at a time."""
    from app.detectors.text.adapters import agreement
    from app.detectors.text.classifier import FourClassTextClassifier
    from app.engine.boundary import Deadline
    from app.engine.models import get_model_manager
    from app.engine.registry import get_registry
    from app.evidence.document_matched_context import DETECTOR, MODEL, REVISION, one_complete_window

    models = get_model_manager()
    registry = get_registry()
    inputs = [candidate.earlier, candidate.later]
    deadline = Deadline(REVIEW_SECONDS)
    measured = []
    for detector_id, model_id, revision in (
        (DETECTOR, MODEL, REVISION),
        (agreement.ID, agreement.MODEL, "a6efd4bf6ecd852f332b2569ae0f98cae13b7985"),
    ):
        if not registry.decision(detector_id).enabled:
            return None
        deadline.check("paired writing check")
        activation = models.activation_bytes_for_items({model_id: len(inputs)})
        with models.lease([model_id], prefer="cpu", activation_bytes=activation, wait_s=0.25) as lease:
            loaded = lease.get(model_id)
            if loaded.revision != revision:
                return None
            if model_id == MODEL:
                classifier = FourClassTextClassifier(*loaded.obj, lease.device)
                values = classifier.predict(inputs, batch=len(inputs), check=deadline.check)
                coverage = classifier.last_coverage
                del classifier
            else:
                values, coverage = agreement.score_inputs(inputs, loaded.obj, lease.device, check=deadline.check)
            deadline.check("paired writing check")
            del loaded
        if len(values) != 2 or len(coverage) != 2 or not all(one_complete_window(proof) for proof in coverage):
            return None
        measured.append({"earlier": float(values[0]), "later": float(values[1])})
    return measured
