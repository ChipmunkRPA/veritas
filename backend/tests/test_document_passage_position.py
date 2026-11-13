from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.core.canonical import canonical_hash
from app.core.config import settings
from app.detectors.text import passage_position
from app.detectors.text.extract import extract, text_hash
from app.detectors.text.segment import segment
from app.engine.boundary import Deadline, DetectorTimeout
from app.evidence import document_long_passage_position, document_matched_context
from app.evidence import document_passage_position as review

TEXT = ("The committee distributed its original recommendation and reviewed the supporting material. "
        "The published instructions include a detailed record of who authorized every major change in the final draft. "
        "The editor preserved those details for the next meeting and for later review.")


def setup_source(tmp_path, text=TEXT):
    original = tmp_path / "original.txt"
    original.write_text(text)
    doc = extract(original, ".txt")
    return doc, segment(doc)


def document(doc, row):
    ref = {"id": "check.test", "display_id": "S01"}
    return {"source": {"method": "plain", "chars": len(doc.text)},
            "coverage": {"checks": [{"signal_ref": ref}], review.KEY: {**row, "signal_ref": ref}}}


def test_position_is_bound_to_exact_source_and_public_result(tmp_path):
    doc, sentences = setup_source(tmp_path, "\n" + TEXT + "\n")
    spans = review.geometry(doc.text, doc.source_map, "plain", sentences)
    assert spans is not None and spans[0]["source_start"] == 1
    readings = [{"input_sha256": text_hash(value), "response_hex": float(score).hex(),
                 "focus_tokens": 12, "neighbor_tokens": 20}
                for (value, _, _), score in zip(review.inputs(doc.text, spans), (1, 5, 2), strict=True)]
    row = review.record(spans, readings, sentences[1].text)
    package = document(doc, row)
    expected = canonical_hash(package["coverage"][review.KEY])
    review.validate_commitment(package, expected)
    assert review.verify_source(package, doc.text, doc.source_map)
    assert review.validate(package)["status"] == "flagged"
    mutated = deepcopy(package)
    mutated["coverage"][review.KEY]["readings"][1]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the original"):
        review.verify_source(mutated, doc.text, doc.source_map)
    del mutated["coverage"][review.KEY]
    with pytest.raises(ValueError, match="published result commitment"):
        review.validate_commitment(mutated, expected)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_position_timeout_keeps_source_scope_and_no_responses(tmp_path, monkeypatch, device):
    doc, sentences = setup_source(tmp_path)
    monkeypatch.setattr(settings, "text_passage_position_review", True)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    spec = SimpleNamespace(id=document_matched_context.DETECTOR, version="1.0.0")
    revisions = {document_matched_context.MODEL: document_matched_context.REVISION}
    ctx = SimpleNamespace(device=device, deadline=Deadline(30))

    def timed_out(*args):
        raise DetectorTimeout("test")

    monkeypatch.setattr(passage_position, "_scores", timed_out)
    result = passage_position.capture(None, doc, sentences, ctx, spec, revisions)
    assert result["status"] == "unassessed" and "readings" not in result
    package = document(doc, result)
    assert not review.verify_source(package, doc.text, doc.source_map)
    assert review.validate(package)["authorship_confidence_contribution"] == 0


@pytest.mark.parametrize("text", [TEXT.replace("The published", "\n\nThe published"),
                                   TEXT.replace("major change", "major  change"),
                                   TEXT + " A fourth sentence is present in this draft."])
def test_position_scope_rejects_ambiguous_mapping(tmp_path, text):
    doc, sentences = setup_source(tmp_path, text)
    assert review.geometry(doc.text, doc.source_map, "plain", sentences) is None


LONG_PARAGRAPH = TEXT.replace("supporting material", "several archived copies of the supporting material circulated the previous week")
LONG_PARAGRAPH = LONG_PARAGRAPH.replace("for later review.", "for later review by anyone with access to the public archive.")
LONG_TEXT = "\n\n\n".join((LONG_PARAGRAPH, LONG_PARAGRAPH.replace("committee", "council"),
                            LONG_PARAGRAPH.replace("committee", "panel")))


def test_longer_writing_position_preserves_source_scope(tmp_path, monkeypatch):
    doc, sentences = setup_source(tmp_path, "\n" + LONG_TEXT + "\n")
    monkeypatch.setattr(settings, "text_long_passage_position_review", True)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    spec = SimpleNamespace(id=document_matched_context.DETECTOR, version="1.0.0")
    revisions = {document_matched_context.MODEL: document_matched_context.REVISION}
    ctx = SimpleNamespace(device="mps", deadline=Deadline(30))
    responses = iter((0., 1., 0., 0., 5., 1., 0., 2., 0.))

    def measured(_, values, deadline):
        assert len(values) <= 6
        return [{"input_sha256": text_hash(value), "response_hex": next(responses).hex(),
                 "focus_tokens": 12, "neighbor_tokens": 20} for value, _, _ in values]

    monkeypatch.setattr(passage_position, "_scores", measured)
    row = passage_position.capture(None, doc, sentences, ctx, spec, revisions)
    assert row["schema"] == document_long_passage_position.SCHEMA and row["status"] == "flagged"
    assert row["selected"] == 1 and row["focus_text"] == sentences[4].text
    focus = row["contexts"][1]["sentences"][1]
    assert focus["source_start"] - focus["start"] == 2
    package = document(doc, row)
    assert review.verify_source(package, doc.text, doc.source_map)
    tampered = deepcopy(package)
    tampered["coverage"][review.KEY]["contexts"][0]["readings"][0]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from the original"):
        review.verify_source(tampered, doc.text, doc.source_map)
    coverage = deepcopy(package)
    coverage["coverage"][review.KEY]["contexts"].pop()
    with pytest.raises(ValueError, match="positions differ from the original"):
        review.verify_source(coverage, doc.text, doc.source_map)


def test_longer_writing_unassessed_without_complete_responses(tmp_path, monkeypatch):
    doc, sentences = setup_source(tmp_path, LONG_TEXT)
    monkeypatch.setattr(settings, "text_long_passage_position_review", True)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    spec = SimpleNamespace(id=document_matched_context.DETECTOR, version="1.0.0")
    revisions = {document_matched_context.MODEL: document_matched_context.REVISION}

    def timed_out(*args):
        raise DetectorTimeout("test")

    monkeypatch.setattr(passage_position, "_scores", timed_out)
    row = passage_position.capture(None, doc, sentences, SimpleNamespace(device="cpu", deadline=Deadline(30)), spec, revisions)
    assert row["status"] == "unassessed" and row["selected"] is None
    assert "focus_text" not in row and all("readings" not in context for context in row["contexts"])
    assert not review.verify_source(document(doc, row), doc.text, doc.source_map)
    doc, sentences = setup_source(tmp_path, LONG_TEXT.replace("major change", "major  change", 1))
    assert document_long_passage_position.geometry(doc.text, doc.source_map, "plain", sentences) is None
