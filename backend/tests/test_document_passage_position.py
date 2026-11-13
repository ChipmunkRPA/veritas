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
from app.evidence import document_matched_context
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
