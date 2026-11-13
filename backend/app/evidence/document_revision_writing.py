"""Source-bound, optional paired writing responses for an authenticated draft comparison."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from app.detectors.text.extract import _normalise, _ocr_language, from_string, text_hash
from app.detectors.text.segment import segment
from app.evidence.document_revision import WordLayouts, WordParagraphs

SCHEMA = "chipmunk.paired-writing-review/1"
WIDE_SCHEMA = "chipmunk.paired-writing-review/2"
BOUNDARY_SCHEMA = "chipmunk.paired-writing-review/3"
WORD_BOUNDARY_SCHEMA = "chipmunk.paired-writing-review/4"
# Second-highest genuine work response in the frozen paired center transfer.
BOUNDARY_POINT = 0.12844311183234092
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
BOUNDARY_SCOPE = ("This experimental check compares one changed sentence beside a paragraph break in a short English "
                  "plain-text draft. Both versions keep the same paragraph layout and the same adjacent sentences. "
                  "Each check reads the changed sentence alone and with its neighbors. "
                  "Responses are uncalibrated and carry zero authorship confidence weight.")
WORD_BOUNDARY_SCOPE = ("This experimental check compares one changed sentence beside a paragraph break in a short "
                       "English Word draft. Both versions keep the same adjacent body paragraphs and neighboring "
                       "sentences. Each check reads the changed sentence alone and with its neighbors. "
                       "Responses are uncalibrated and carry zero authorship confidence weight.")
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
    across_paragraphs: bool = False


def _source(text: str):
    document = from_string(text)
    if not document.paragraphs or document.meta["words"] > MAX_WORDS:
        return None
    language = _ocr_language(document.text)
    if language.get("code") != "en" or language.get("status") != "supported":
        return None
    return document, segment(document)


def _word_context_locations(spans: list[list[int]], layout: WordParagraphs) -> list[dict] | None:
    locations = []
    for start, end in spans:
        matches = [(paragraph, index, start - lo, end - lo) for paragraph, index, lo, hi in layout
                   if lo <= start < end <= hi]
        if len(matches) != 1:
            return None
        paragraph, index, local_start, local_end = matches[0]
        locations.append({"paragraph": paragraph, "body_index": index, "start": local_start, "end": local_end})
    first, last = locations[0], locations[-1]
    if last["paragraph"] != first["paragraph"] + 1 or last["body_index"] != first["body_index"] + 1:
        return None
    return locations


def paired_input(comparison: dict, earlier: str, later: str, *, allow_wide: bool = True,
                 allow_boundary: bool = False, word_layouts: WordLayouts | None = None) -> PairedInput | None:
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
    paragraphs = [unit.paragraph for unit in old_units[selected]]
    across_paragraphs = paragraphs[0] != paragraphs[-1]
    word_boundary = comparison["format"] == "docx" and word_layouts is not None
    if across_paragraphs and (not allow_boundary or (comparison["format"] != "plain_text" and not word_boundary) or wide
                              or paragraphs[-1] - paragraphs[0] != 1):
        return None

    def context(document, units, raw_text):
        origins_by_paragraph = {}
        spans = []
        for unit in units:
            block = document.source_map[unit.paragraph]
            if unit.paragraph not in origins_by_paragraph:
                ranges = block["source_ranges"]
                offset, stop = ranges[0][2], ranges[-1][3]
                normalized, origins = _normalise(raw_text[offset:stop])
                if normalized != document.text[block["start"]:block["end"]]:
                    raise ValueError("Writing comparison paragraph differs from its decoded source")
                origins_by_paragraph[unit.paragraph] = offset, origins
            offset, origins = origins_by_paragraph[unit.paragraph]
            spans.append([offset + origins[unit.start - block["start"]][0],
                          offset + origins[unit.end - block["start"] - 1][1]])
        if [_normalise(raw_text[start:end])[0] for start, end in spans] != [unit.text for unit in units]:
            raise ValueError("Writing comparison context differs from its decoded source")
        start, end = spans[1][0], spans[-2][1]
        model_input = " ".join(unit.text for unit in units)
        focus = {"start": start, "end": end, "text": raw_text[start:end],
                 "context_spans": spans, "input_sha256": text_hash(model_input)}
        if across_paragraphs:
            focus["center_input_sha256"] = text_hash(_normalise(focus["text"])[0])
        return focus, model_input

    before, old_input = context(old_doc, old_units[selected], earlier)
    after, new_input = context(new_doc, new_units[selected], later)

    def contained(change, side, expected):
        item = change[side]
        return item is not None and expected["start"] <= item["start"] <= item["end"] <= expected["end"]

    if (any(change["role"] != "body" or not contained(change, "earlier", before)
            or not contained(change, "later", after) for change in changes)
            or not all(any(change[side]["changed_tokens"] for change in changes) for side in ("earlier", "later"))):
        return None
    if across_paragraphs and word_boundary:
        assert word_layouts is not None
        before_locations = _word_context_locations(before["context_spans"], word_layouts[0])
        after_locations = _word_context_locations(after["context_spans"], word_layouts[1])
        if before_locations is None or after_locations is None:
            return None

        def paragraph_identity(locations):
            return [(item["paragraph"], item["body_index"]) for item in locations]

        if paragraph_identity(before_locations) != paragraph_identity(after_locations):
            return None
        for side, focus, locations in (("earlier", before, before_locations), ("later", after, after_locations)):
            center = locations[1]
            base = focus["start"] - center["start"]
            for change in changes:
                item = change[side]
                expected = center | {"start": item["start"] - base, "end": item["end"] - base}
                if item["paragraphs"] != [expected]:
                    return None
            focus["context_paragraphs"] = locations
    elif comparison["format"] == "docx":
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
                                             "change_ids": [change["id"] for change in changes]}, across_paragraphs)


def _increases(responses: list[dict]) -> list[float]:
    if (not isinstance(responses, list) or len(responses) != 2
            or any(not isinstance(pair, dict) or set(pair) != {"earlier", "later"} for pair in responses)):
        raise ValueError("Two paired writing responses are required")

    def odds(p):
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid paired writing response")
        value = min(1 - 1e-7, max(1e-7, p))
        return value / (1 - value)

    return [math.log(odds(row["later"]) / odds(row["earlier"])) for row in responses]


def result(candidate: PairedInput | None, responses: list[dict] | None = None, *, draft_format: str = "plain_text",
           allow_wide: bool = True, center_responses: list[dict] | None = None) -> dict:
    across_paragraphs = candidate is not None and candidate.across_paragraphs
    status = "outside_scope" if candidate is None else "unavailable"
    if responses is not None:
        if across_paragraphs:
            if center_responses is None:
                raise ValueError("The cross-paragraph check needs exact sentence responses")
            positive = min([*_increases(responses), *_increases(center_responses)]) > BOUNDARY_POINT
        else:
            if center_responses is not None:
                raise ValueError("The same-paragraph check has no separate sentence responses")
            positive = min(_increases(responses)) >= math.log(2)
        status = "joint_increase" if positive else "no_joint_increase"
    elif center_responses is not None:
        raise ValueError("Context responses are required with sentence responses")
    wide = candidate is not None and len(candidate.geometry["later"]["context_spans"]) == 4
    plain_outside = candidate is None and draft_format == "plain_text" and allow_wide
    scope = WIDE_SCOPE if wide else {"docx": WORD_SCOPE, "email": EMAIL_SCOPE, "pdf": PDF_SCOPE}.get(draft_format, SCOPE)
    value = {"schema": WIDE_SCHEMA if wide or plain_outside else SCHEMA, "status": status,
             "scope": PLAIN_SCOPE if plain_outside else scope,
             "summary": "This draft pair falls outside the exact paired sentence scope." if plain_outside else STATUS_TEXT[status],
             "authorship_confidence_contribution": 0}
    if across_paragraphs:
        if draft_format == "docx":
            value.update(schema=WORD_BOUNDARY_SCHEMA, scope=WORD_BOUNDARY_SCOPE)
        else:
            value.update(schema=BOUNDARY_SCHEMA, scope=BOUNDARY_SCOPE)
        if status == "joint_increase":
            value["summary"] = "Both writing checks gave a stronger AI-like response for the changed sentence alone and with its neighbors."
    if candidate is not None:
        value.update(candidate.geometry)
    if responses is not None:
        value["responses"] = responses
    if center_responses is not None:
        value["center_responses"] = center_responses
    return value


def verify(comparison: dict, earlier: str, later: str, recorded: dict, *,
           word_layouts: WordLayouts | None = None) -> str:
    """Rebuild the source scope and check saved response arithmetic without inference."""
    schema = recorded.get("schema")
    allow_wide = schema in (WIDE_SCHEMA, BOUNDARY_SCHEMA)
    candidate = paired_input(comparison, earlier, later, allow_wide=allow_wide,
                             allow_boundary=schema in (BOUNDARY_SCHEMA, WORD_BOUNDARY_SCHEMA),
                             word_layouts=word_layouts if schema == WORD_BOUNDARY_SCHEMA else None)
    if candidate is None and "responses" in recorded:
        raise ValueError("An out-of-scope draft cannot carry paired responses")
    expected = result(candidate, recorded.get("responses"), draft_format=comparison["format"], allow_wide=allow_wide,
                      center_responses=recorded.get("center_responses"))
    if json.dumps(recorded, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("The paired writing check does not replay from these source positions and saved responses")
    return expected["status"]


def score(candidate: PairedInput) -> list[dict] | None:
    """Use the two pinned local checks, one resident lease at a time."""
    measured = _score([candidate.earlier, candidate.later])
    return [{"earlier": row[0], "later": row[1]} for row in measured] if measured is not None else None


def score_boundary(candidate: PairedInput) -> tuple[list[dict] | None, list[dict] | None]:
    centers = [_normalise(candidate.geometry[side]["text"])[0] for side in ("earlier", "later")]
    if any(text_hash(center) != candidate.geometry[side]["center_input_sha256"]
           for side, center in zip(("earlier", "later"), centers, strict=True)):
        raise ValueError("The exact sentence input differs from its source range")
    inputs = [candidate.earlier, candidate.later, *centers]
    measured = _score(inputs)
    if measured is None:
        return None, None
    return ([{"earlier": row[0], "later": row[1]} for row in measured],
            [{"earlier": row[2], "later": row[3]} for row in measured])


def _score(inputs):
    from app.detectors.text.adapters import agreement
    from app.detectors.text.classifier import FourClassTextClassifier
    from app.engine.boundary import Deadline
    from app.engine.models import get_model_manager
    from app.engine.registry import get_registry
    from app.evidence.document_matched_context import DETECTOR, MODEL, REVISION, one_complete_window

    models = get_model_manager()
    registry = get_registry()
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
        if len(values) != len(inputs) or len(coverage) != len(inputs) or not all(one_complete_window(proof) for proof in coverage):
            return None
        measured.append([float(value) for value in values])
    return measured
