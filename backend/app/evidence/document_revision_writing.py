"""Source-bound, optional paired writing responses for an authenticated draft comparison."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

from app.detectors.text.extract import _normalise, _ocr_language, from_string, text_hash
from app.detectors.text.segment import segment
from app.evidence.document_revision import WordLayouts, WordParagraphs, differences, tokens

SCHEMA = "chipmunk.paired-writing-review/1"
WIDE_SCHEMA = "chipmunk.paired-writing-review/2"
BOUNDARY_SCHEMA = "chipmunk.paired-writing-review/3"
WORD_BOUNDARY_SCHEMA = "chipmunk.paired-writing-review/4"
INTERIOR_SCHEMA = "chipmunk.paired-writing-review/5"
MULTIPLE_SCHEMA = "chipmunk.paired-writing-review/6"
ANCHORED_DRAFT_SCHEMA = "chipmunk.document-revision-comparison/7"
MAX_REVISIONS = 4
MULTIPLE_SCOPE = ("Checks separate changed sentences in short English text drafts. Each comparison requires its own "
                  "unchanged neighbors. The responses are uncalibrated and carry zero authorship confidence weight.")
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
INTERIOR_SCOPE = ("This experimental check compares an exact changed sentence in available three-sentence contexts "
                  "within the same paragraph of a short English plain-text draft. Responses are uncalibrated and "
                  "carry zero authorship confidence weight.")
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
    neighboring_contexts: tuple[PairedInput, ...] = ()


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
                 allow_boundary: bool = False, allow_interior: bool = False,
                 word_layouts: WordLayouts | None = None) -> PairedInput | None:
    """Pair an interior change. For EML, use the verified decoded body prefix from the draft comparison."""
    changes = comparison["changes"]
    if comparison["format"] == "email":
        changes = [change for change in changes if change["role"] == "body"]
    if comparison["format"] not in ("plain_text", "docx", "email", "pdf") or not changes:
        return None
    old, new = _source(earlier), _source(later)
    if old is None or new is None:
        return None
    old_units, new_units = old[1], new[1]
    if [unit.paragraph for unit in old_units] != [unit.paragraph for unit in new_units]:
        return None
    changed = [i for i, (a, b) in enumerate(zip(old_units, new_units, strict=True)) if a.text != b.text]
    wide = (allow_wide and comparison["format"] == "plain_text" and len(changed) == 2
            and changed[1] == changed[0] + 1)
    if len(changed) != 1 and not wide:
        return None
    return _pair(comparison, changes, earlier, later, old, new, changed, wide=wide,
                 allow_boundary=allow_boundary, allow_interior=allow_interior, word_layouts=word_layouts)


def _context(document, units, raw_text, *, full_range=False):
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
    start, end = (spans[0][0], spans[-1][1]) if full_range else (spans[1][0], spans[-2][1])
    model_input = " ".join(unit.text for unit in units)
    return {"start": start, "end": end, "text": raw_text[start:end],
            "context_spans": spans, "input_sha256": text_hash(model_input)}, model_input


def _pair(comparison, changes, earlier, later, old, new, changed, *, wide=False,
          allow_boundary=False, allow_interior=False, word_layouts=None, assign_changes=False):
    old_doc, old_units = old
    new_doc, new_units = new
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

    before, old_input = _context(old_doc, old_units[selected], earlier)
    after, new_input = _context(new_doc, new_units[selected], later)
    if across_paragraphs:
        for focus in (before, after):
            focus["center_input_sha256"] = text_hash(_normalise(focus["text"])[0])

    def contained(change, side, expected):
        item = change[side]
        return item is not None and expected["start"] <= item["start"] <= item["end"] <= expected["end"]

    linked = [change for change in changes if change["role"] == "body"
              and contained(change, "earlier", before) and contained(change, "later", after)]
    if ((not assign_changes and len(linked) != len(changes))
            or not all(any(change[side]["changed_tokens"] for change in linked) for side in ("earlier", "later"))):
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
    neighboring_contexts = []
    if allow_interior and comparison["format"] == "plain_text" and not wide and not across_paragraphs:
        for center in (index - 1, index + 1):
            if not 0 < center < len(old_units) - 1:
                continue
            window = slice(center - 1, center + 2)
            if any(unit.paragraph != paragraphs[0] for unit in old_units[window]):
                continue
            old_geometry, old_window = _context(old_doc, old_units[window], earlier, full_range=True)
            new_geometry, new_window = _context(new_doc, new_units[window], later, full_range=True)
            neighboring_contexts.append(PairedInput(old_window, new_window, {"earlier": old_geometry, "later": new_geometry}))
    return PairedInput(old_input, new_input, {"earlier": before, "later": after,
                                             "change_ids": [change["id"] for change in linked]}, across_paragraphs,
                       tuple(neighboring_contexts))


def _separated_centers(comparison, earlier, later):
    if comparison["format"] != "plain_text" or comparison.get("relocated_passages"):
        return None
    old, new = _source(earlier), _source(later)
    if old is None or new is None:
        return None
    old_units, new_units = old[1], new[1]
    if [unit.paragraph for unit in old_units] != [unit.paragraph for unit in new_units]:
        return None
    changed = [i for i, (left, right) in enumerate(zip(old_units, new_units, strict=True)) if left.text != right.text]
    if not 2 <= len(changed) <= MAX_REVISIONS or any(b - a < 3 for a, b in zip(changed, changed[1:])):
        return None
    return old, new, changed


def separate_revisions(comparison: dict, earlier: str, later: str) -> list[PairedInput]:
    centers = _separated_centers(comparison, earlier, later)
    if centers is None:
        return []
    old, new, changed = centers
    candidates = [_pair(comparison, comparison["changes"], earlier, later, old, new, [i], assign_changes=True)
                  for i in changed]
    if any(item is None for item in candidates):
        return []
    linked = [change_id for candidate in candidates for change_id in candidate.geometry["change_ids"]]
    if sorted(linked) != sorted(change["id"] for change in comparison["changes"]):
        return []
    return candidates


def anchor_revisions(comparison: dict, earlier: str, later: str) -> tuple[dict, list[PairedInput]] | None:
    """Confine wording differences to exact sentence ranges between unchanged neighbors."""
    centers = _separated_centers(comparison, earlier, later)
    if centers is None:
        return None
    old, new, changed = centers
    old_units, new_units = old[1], new[1]
    if not all(0 < index < len(old_units) - 1 for index in changed):
        return None

    def same_paragraph(units, index):
        return units[index - 1].paragraph == units[index].paragraph == units[index + 1].paragraph

    def neighbors(units, index):
        return units[index - 1].text, units[index + 1].text

    for units in (old_units, new_units):
        counts = Counter(neighbors(units, i) for i in range(1, len(units) - 1) if same_paragraph(units, i))
        if any(not same_paragraph(units, i) or counts[neighbors(units, i)] != 1 for i in changed):
            return None
    if {old_units[i].text for i in changed} & {new_units[i].text for i in changed}:
        return None

    candidates, edits = [], []
    old_end = new_end = 0
    for i in changed:
        old_focus, old_input = _context(old[0], old_units[i - 1:i + 2], earlier)
        new_focus, new_input = _context(new[0], new_units[i - 1:i + 2], later)
        if _word_sequence(earlier, old_end, old_focus["start"]) != _word_sequence(later, new_end, new_focus["start"]):
            return None
        rows = differences(tokens(earlier, [(old_focus["start"], old_focus["end"])]),
                           tokens(later, [(new_focus["start"], new_focus["end"])]))
        if not rows:
            return None
        change_ids = []
        for row in rows:
            edit = {"id": f"C{len(edits) + 1:02d}", "role": "body"}
            for side, text in (("earlier", earlier), ("later", later)):
                start, end = row[side]
                edit[side] = {"start": start, "end": end, "text": text[start:end], "changed_tokens": row[f"{side}_changed_tokens"]}
            change_ids.append(edit["id"])
            edits.append(edit)
        candidates.append(PairedInput(old_input, new_input, {"earlier": old_focus, "later": new_focus,
                                                               "change_ids": change_ids}))
        old_end, new_end = old_focus["end"], new_focus["end"]
    if _word_sequence(earlier, old_end, len(earlier)) != _word_sequence(later, new_end, len(later)):
        return None
    return comparison | {
        "schema": ANCHORED_DRAFT_SCHEMA,
        "scope": comparison["scope"] + " Unchanged neighboring sentences anchor these wording changes.",
        "changes": edits,
    }, candidates


def _word_sequence(text, start, end):
    return [value for value, _, _ in tokens(text, [(start, end)])]


def refine_wording(comparison: dict, earlier: str, later: str) -> dict:
    """Improve exact wording links where the prior diff could not map separate changes."""
    try:
        anchored = anchor_revisions(comparison, earlier, later)
        if (anchored is not None and anchored[0]["changes"] != comparison["changes"]
                and not separate_revisions(comparison, earlier, later)):
            return anchored[0]
    except ValueError:
        return comparison
    return comparison


def multiple_result(candidates: list[PairedInput], measured: list[list[dict] | None] | None = None) -> dict:
    if not candidates or (measured is not None and len(measured) != len(candidates)):
        raise ValueError("Expected individually mapped paired revisions")
    reviews = []
    for candidate, responses in zip(candidates, measured if measured is not None else [None] * len(candidates), strict=True):
        status = result(candidate, responses)["status"]
        review = {"status": status, **candidate.geometry}
        if responses is not None:
            review["responses"] = responses
        reviews.append(review)
    increases = sum(row["status"] == "joint_increase" for row in reviews)
    checked = sum(row["status"] != "unavailable" for row in reviews)
    return {"schema": MULTIPLE_SCHEMA, "status": "joint_increase" if increases else
            "no_joint_increase" if checked == len(candidates) else "unavailable",
            "summary": f"Checked {checked} of {len(candidates)} mapped revisions. Both writing responses increased at {increases}.",
            "scope": MULTIPLE_SCOPE, "authorship_confidence_contribution": 0, "revisions": reviews}


def _increases(responses: list[dict], *, floor: float = 1e-7) -> list[float]:
    if (not isinstance(responses, list) or len(responses) != 2
            or any(not isinstance(pair, dict) or set(pair) != {"earlier", "later"} for pair in responses)):
        raise ValueError("Two paired writing responses are required")

    def odds(p):
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid paired writing response")
        value = min(1 - floor, max(floor, p))
        return value / (1 - value)

    return [math.log(odds(row["later"]) / odds(row["earlier"])) for row in responses]


def result(candidate: PairedInput | None, responses: list[dict] | None = None, *, draft_format: str = "plain_text",
           allow_wide: bool = True, center_responses: list[dict] | None = None,
           neighboring_responses: list[list[dict]] | None = None) -> dict:
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
    if neighboring_responses is not None:
        if (draft_format != "plain_text" or candidate is None or responses is None or not candidate.neighboring_contexts
                or not isinstance(neighboring_responses, list)
                or len(neighboring_responses) != len(candidate.neighboring_contexts)):
            raise ValueError("The paired review needs all exact neighboring responses")
        neighbors = [
            context.geometry | {"responses": scores, "status": "joint_increase"
                                if min(_increases(scores, floor=1e-6)) >= math.log(2) else "no_joint_increase"}
            for context, scores in zip(candidate.neighboring_contexts, neighboring_responses, strict=True)
        ]
        if status != "joint_increase" and any(row["status"] == "joint_increase" for row in neighbors):
            value.update(status="joint_increase", summary="Both writing checks gave a stronger AI-like response in "
                         "another three-sentence context containing the changed sentence.")
        value.update(schema=INTERIOR_SCHEMA, scope=INTERIOR_SCOPE, neighboring_contexts=neighbors)
    return value


def verify(comparison: dict, earlier: str, later: str, recorded: dict, *,
           word_layouts: WordLayouts | None = None) -> str:
    """Rebuild the source scope and check saved response arithmetic without inference."""
    schema = recorded.get("schema")
    if schema == MULTIPLE_SCHEMA:
        candidates = separate_revisions(comparison, earlier, later)
        revisions = recorded.get("revisions")
        if (not isinstance(revisions, list) or len(revisions) != len(candidates) or not candidates
                or any(not isinstance(row, dict) for row in revisions)):
            raise ValueError("The draft comparison does not contain all mapped revisions")
        expected = multiple_result(candidates, [row.get("responses") for row in revisions])
        if json.dumps(recorded, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError("Mapped revisions differ from the source positions or saved responses")
        return expected["status"]
    allow_wide = schema in (WIDE_SCHEMA, BOUNDARY_SCHEMA, INTERIOR_SCHEMA)
    candidate = paired_input(comparison, earlier, later, allow_wide=allow_wide,
                             allow_boundary=schema in (BOUNDARY_SCHEMA, WORD_BOUNDARY_SCHEMA),
                             allow_interior=schema == INTERIOR_SCHEMA,
                             word_layouts=word_layouts if schema == WORD_BOUNDARY_SCHEMA else None)
    if candidate is None and "responses" in recorded:
        raise ValueError("An out-of-scope draft cannot carry paired responses")
    neighboring = None
    if schema == INTERIOR_SCHEMA:
        records = recorded.get("neighboring_contexts")
        if not isinstance(records, list) or any(not isinstance(row, dict) or "responses" not in row for row in records):
            raise ValueError("The paired writing check does not replay without exact neighboring responses")
        neighboring = [row["responses"] for row in records]
    expected = result(candidate, recorded.get("responses"), draft_format=comparison["format"], allow_wide=allow_wide,
                      center_responses=recorded.get("center_responses"), neighboring_responses=neighboring)
    if json.dumps(recorded, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("The paired writing check does not replay from these source positions and saved responses")
    return expected["status"]


def score(candidate: PairedInput) -> list[dict] | None:
    """Use the two pinned local checks, one resident lease at a time."""
    measured = _score([candidate.earlier, candidate.later])
    return [{"earlier": row[0], "later": row[1]} for row in measured] if measured is not None else None


def score_separate(candidates: list[PairedInput]) -> list[list[dict] | None] | None:
    texts = [text for candidate in candidates for text in (candidate.earlier, candidate.later)]
    measured = _score(texts, allow_partial=True)
    if measured is None:
        return None
    readings = []
    for i in range(0, len(texts), 2):
        pairs = [{"earlier": row[i], "later": row[i + 1]} for row in measured]
        readings.append(pairs if all(None not in pair.values() for pair in pairs) else None)
    return readings


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


def score_interior(candidate: PairedInput) -> tuple[list[dict] | None, list[list[dict]] | None]:
    from app.engine.boundary import Deadline
    from app.engine.models import MemoryBudgetExceeded

    deadline = Deadline(REVIEW_SECONDS)
    contexts = (candidate, *candidate.neighboring_contexts)
    try:
        measured = _score([text for context in contexts for text in (context.earlier, context.later)],
                          allow_partial=True, deadline=deadline)
    except MemoryBudgetExceeded:
        measured = _score([candidate.earlier, candidate.later], deadline=deadline)
        return ([{"earlier": row[0], "later": row[1]} for row in measured] if measured is not None else None), None
    if measured is None or any(value is None for row in measured for value in row[:2]):
        return None, None
    primary = [{"earlier": row[0], "later": row[1]} for row in measured]
    if any(value is None for row in measured for value in row[2:]):
        return primary, None
    return primary, [[{"earlier": row[i], "later": row[i + 1]} for row in measured]
                     for i in range(2, 2 * len(contexts), 2)]


def _score(inputs, *, allow_partial=False, deadline=None):
    from app.detectors.text.adapters import agreement
    from app.detectors.text.classifier import FourClassTextClassifier
    from app.engine.boundary import Deadline
    from app.engine.models import get_model_manager
    from app.engine.registry import get_registry
    from app.evidence.document_matched_context import DETECTOR, MODEL, REVISION, one_complete_window

    models = get_model_manager()
    registry = get_registry()
    deadline = deadline if deadline is not None else Deadline(REVIEW_SECONDS)
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
        if len(values) != len(inputs) or len(coverage) != len(inputs):
            return None
        complete = [one_complete_window(proof) for proof in coverage]
        if not allow_partial and not all(complete):
            return None
        measured.append([float(value) if passed else None for value, passed in zip(values, complete, strict=True)])
    return measured
