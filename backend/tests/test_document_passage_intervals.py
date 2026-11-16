from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from app.core.canonical import canonical_hash
from app.detectors.text import language_scope
from app.detectors.text.adapters import interval_review
from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from app.evidence import document_interval_locations as native
from app.evidence import document_passage_intervals as proof
from scripts.writing_native_passage_panel import fixture

BODY = "\n\n".join(
    " ".join(f"The writer described the original conditions and the records of the meeting for item {item}."
             for item in range(start, start + 6)) for start in (10, 20, 30)
)
QUOTE = "The previous author wrote a separate passage. Its text belongs to a different email."


@pytest.mark.parametrize("kind", ["email_plain", "email_html"])
def test_optional_passage_reading_excludes_the_quoted_context(tmp_path, monkeypatch, kind):
    path = tmp_path / "reply.eml"
    path.write_bytes(fixture(kind, BODY, None, quote=QUOTE))
    doc = extract(path, ".eml")
    all_sentences = segment(doc)
    fit_end = native.checked_end("email", doc.text, doc.source_map, doc.meta)
    assert fit_end == len(BODY)
    body = [row for row in all_sentences if row.end <= fit_end]
    assert len(body) == 18 < len(all_sentences)
    monkeypatch.setattr(interval_review, "enabled", lambda: True)
    monkeypatch.setattr(interval_review.document_matched_context, "matches", lambda *_: True)
    expected_final_window = " ".join(row.text for row in body[-2:])

    def predict(inputs, **_):
        assert inputs == [row.text for row in body] + [expected_final_window]
        return [.999] * len(body) + [.99]

    deadline = SimpleNamespace(remaining=lambda: None, nested=lambda _: SimpleNamespace(check=lambda _: None))
    ctx = SimpleNamespace(profile="full", outputs=SimpleNamespace(document={language_scope.KEY: language_scope.ENGLISH}), deadline=deadline)
    proof_value = interval_review.measure(SimpleNamespace(predict=predict), doc, all_sentences, [.999] * len(all_sentences),
                                          ctx, SimpleNamespace(id="fixture", version="1"), {})
    assert proof_value is not None and proof_value["status"] == "flagged" and proof_value["checked_end"] == fit_end
    assert proof_value["readings"]["context"][-1] == .99
    assert all(piece["end"] <= fit_end for row in proof_value["ranges"] for piece in row["locations"])
    evidence = {"source": {"method": "email", "chars": len(doc.text), proof.COMMITMENT: canonical_hash(proof_value)},
                "coverage": doc.meta | {proof.KEY: proof_value}}
    positions = [asdict(row) for row in all_sentences]
    proof.verify_source(evidence, doc.text, doc.source_map, positions, metadata=doc.meta)
    forged = deepcopy(evidence)
    forged_value = forged["coverage"][proof.KEY]
    forged_value["ranges"][0]["locations"][0]["ranges"][0][1] += 1
    forged["source"][proof.COMMITMENT] = canonical_hash(forged_value)
    with pytest.raises(ValueError):
        proof.verify_source(forged, doc.text, doc.source_map, positions, metadata=doc.meta)


def test_word_locations_follow_separate_native_paragraphs(tmp_path):
    path = tmp_path / "ordinary.docx"
    path.write_bytes(fixture("word", BODY, None))
    doc = extract(path, ".docx")
    assert native.checked_end("python-docx", doc.text, doc.source_map, doc.meta, doc.word_review) == len(BODY)
    first, second, _ = doc.source_map
    actual = native.locate("python-docx", doc.source_map, first["end"] - 5, second["start"] + 8)
    assert [row["paragraph"] for row in actual] == [1, 2]
    assert [row["ranges"] for row in actual] == [[[first["end"] - 5, first["end"]]], [[0, 8]]]
    revised = {"status": "found"}
    assert native.checked_end("python-docx", doc.text, doc.source_map, doc.meta, revised) is None


def test_pdf_locations_use_the_same_precision_before_and_after_export():
    import json

    from app.core.canonical import canonical_json

    mapping = [{"start": 0, "end": 5, "page": 1, "fragments": [
        {"start": 0, "end": 5, "bbox": [157.40997314453125, 64.40234375, 229.83998107910156, 74.40234375]}
    ]}]
    from_source = native.locate("pymupdf", mapping, 0, 5)
    from_frozen_mapping = native.locate("pymupdf", json.loads(canonical_json(mapping)), 0, 5)
    assert from_source == from_frozen_mapping == json.loads(canonical_json(from_source))
