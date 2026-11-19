from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from test_document_packets import packet_factory as packet_factory
from test_odt_section_visibility import odt

from app.billing.preflight import _document
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text import language_scope
from app.detectors.text.adapters import interval_review
from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from app.evidence import document_interval_locations as native
from app.evidence import document_passage_intervals as proof
from app.evidence.document_packets import build_packet_set
from app.reports import public_verify, verify_document_source
from scripts.writing_native_passage_panel import fixture

BODY = "\n\n".join(
    " ".join(f"The writer described the original conditions and the records of the meeting for item {item}."
             for item in range(start, start + 6)) for start in (10, 20, 30)
)
QUOTE = "The previous author wrote a separate passage. Its text belongs to a different email."


@pytest.mark.parametrize(("carrier", "suffix"), [("plain", ".txt"), ("word", ".docx"), ("email_html", ".eml")])
def test_short_passage_skip_replays_only_eligible_sentences(tmp_path, monkeypatch, packet_factory, carrier, suffix):
    body = " ".join(f"The writer recorded useful facts for document item {item}." for item in range(10, 21))
    path = tmp_path / f"short passages{suffix}"
    path.write_bytes(body.encode() if carrier == "plain" else fixture(carrier, body, None, quote=QUOTE))
    _document(path, suffix)
    doc = extract(path, suffix)
    sentences = segment(doc)
    assert len(sentences) == (13 if carrier == "email_html" else 11)
    monkeypatch.setattr(interval_review, "enabled", lambda: True)
    monkeypatch.setattr(interval_review.document_matched_context, "matches", lambda *_: True)
    ctx = SimpleNamespace(profile="full", outputs=SimpleNamespace(document={language_scope.KEY: language_scope.ENGLISH}))
    spec = SimpleNamespace(id="fixture", version="1")
    review = interval_review.measure(None, doc, sentences, [], ctx, spec, {})
    expected = {"schema": proof.SHORT_SCOPE_SCHEMA, "status": "skipped", "reason": "too_few_sentences",
                "eligible_sentences": 11, "min_sentences": 12, "ranges": [], "authorship_confidence_contribution": 0}
    if carrier != "plain":
        expected["checked_end"] = len(body)
    assert review == expected

    db, store, analysis, result, _, _ = packet_factory(suffix, source_pdf=path, include_word_review=carrier == "word")
    historical = build_packet_set(db, analysis, store, result)
    assert proof.KEY not in historical["coverage"]
    assert proof.COMMITMENT not in historical["source"]
    result["manifest"]["sampling"][proof.KEY] = review
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["coverage"][proof.KEY] == expected
    assert evidence["source"][proof.COMMITMENT] == canonical_hash(expected)
    assert historical["packets"] == evidence["packets"]

    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    archive_path = tmp_path / "proof.zip"
    for record in (historical, evidence):
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("evidence.json", canonical_json(record))
        with ZipFile(archive_path) as archive:
            note = verify_document_source.check(archive, bundle, path)
        assert ("Verified the eligible sentence count" in note) is (record is evidence)

    forged = evidence["coverage"][proof.KEY] | {"eligible_sentences": 10}
    evidence["coverage"][proof.KEY] = forged
    evidence["source"][proof.COMMITMENT] = canonical_hash(forged)
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(evidence))
    with ZipFile(archive_path) as archive, pytest.raises(ValueError, match="sentence count differs from the source text"):
        verify_document_source.check(archive, bundle, path)


@pytest.mark.parametrize("changes", [
    {"status": "no_ranges"}, {"reason": "model_unavailable"}, {"eligible_sentences": 12},
    {"min_sentences": 12.0}, {"authorship_confidence_contribution": False}, {"ranges": [{"start": 0, "end": 3}]},
])
def test_short_passage_skip_has_a_fixed_neutral_shape(changes):
    record = proof.short_scope("plain", 200, 11) | changes
    evidence = {"source": {"method": "plain", "chars": 200}, "coverage": {proof.KEY: record}}
    with pytest.raises(ValueError, match="invalid short-input record"):
        proof.validate(evidence, canonical_hash(record))


def test_twelve_eligible_sentences_run_the_existing_interval_check(tmp_path, monkeypatch):
    path = tmp_path / "twelve.txt"
    path.write_text("\n\n".join(BODY.split("\n\n")[:2]))
    doc = extract(path, ".txt")
    sentences = segment(doc)
    assert len(sentences) == 12
    monkeypatch.setattr(interval_review, "enabled", lambda: True)
    monkeypatch.setattr(interval_review.document_matched_context, "matches", lambda *_: True)
    deadline = SimpleNamespace(remaining=lambda: None, nested=lambda _: SimpleNamespace(check=lambda _: None))
    ctx = SimpleNamespace(profile="full", outputs=SimpleNamespace(document={language_scope.KEY: language_scope.ENGLISH}), deadline=deadline)
    calls = []

    def predict(inputs, **_):
        calls.extend(inputs)
        return [.005] * len(inputs)

    review = interval_review.measure(SimpleNamespace(predict=predict), doc, sentences, [.005] * 12,
                                     ctx, SimpleNamespace(id="fixture", version="1"), {})
    assert calls == [sentence.text for sentence in sentences]
    assert review["status"] == "no_ranges" and review["schema"] != proof.SHORT_SCOPE_SCHEMA
    assert "eligible_sentences" not in review


@pytest.mark.parametrize("gate", ["configuration", "profile", "language", "source", "checkpoint"])
def test_short_count_does_not_replace_other_passage_eligibility_gates(tmp_path, monkeypatch, gate):
    suffix = ".md" if gate == "source" else ".txt"
    path = tmp_path / f"input{suffix}"
    path.write_text(BODY.split("\n\n")[0])
    doc = extract(path, suffix)
    monkeypatch.setattr(interval_review, "enabled", lambda: gate != "configuration")
    monkeypatch.setattr(interval_review.document_matched_context, "matches", lambda *_: gate != "checkpoint")
    ctx = SimpleNamespace(profile="quick" if gate == "profile" else "full",
                          outputs=SimpleNamespace(document={language_scope.KEY: None if gate == "language" else language_scope.ENGLISH}))
    assert interval_review.measure(None, doc, segment(doc), [], ctx, SimpleNamespace(id="fixture", version="1"), {}) is None


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


@pytest.mark.parametrize("suffix", [".txt", ".docx", ".odt", ".rtf"])
def test_document_baseline_preserves_prior_ranges_and_replays_original_locations(tmp_path, suffix):
    path = tmp_path / f"passages{suffix}"
    if suffix == ".odt":
        odt(path, visible=BODY.split("\n\n"))
    elif suffix == ".rtf":
        first, second, third = BODY.split("\n\n")
        path.write_text(r"{\rtf1\ansi " + r"\par ".join((first, r"{\v " + QUOTE + "}", second, third)) + "}")
    else:
        path.write_bytes(BODY.encode() if suffix == ".txt" else fixture("word", BODY, None))
    doc = extract(path, suffix)
    assert doc.text == BODY
    sentences = segment(doc)
    geometry = [[row.start, row.end, row.n_words] for row in sentences]
    assert len(geometry) == 18
    response = [.005] * len(geometry)
    response[4:6] = [.999, .999]
    response[11:13] = [.85, .85]
    readings = {"context": response, "isolated": response}
    method = doc.meta["method"]
    if suffix in {".odt", ".rtf"}:
        assert native.checked_end(method, doc.text, doc.source_map, doc.meta) == len(BODY)
        key, _ = native.DECODED_BODY_POLICIES[method]
        historical_metadata = {name: value for name, value in doc.meta.items() if name != key}
        assert native.checked_end(method, doc.text, doc.source_map, historical_metadata) is None
    options = {"method": method, "checked_end": len(doc.text)}
    original = proof.capture(doc.text, doc.source_map, geometry, readings, **options)
    expanded = proof.capture(doc.text, doc.source_map, geometry, readings, document_baseline=True, **options)
    assert original["schema"] == (proof.SCHEMA if suffix == ".txt" else proof.NATIVE_SCHEMA)
    assert expanded["schema"] == proof.DOCUMENT_BASELINE_SCHEMA
    retained, additional = expanded["ranges"]
    assert retained == original["ranges"][0] | {"basis": "pooled"}
    assert (additional["start"], additional["end"], additional["basis"]) == (
        sentences[11].start, sentences[12].end, "document_baseline",
    )
    if suffix != ".txt":
        assert [place["paragraph"] for place in additional["locations"]] == ([3, 4] if suffix == ".rtf" else [2, 3])
    assert expanded["authorship_confidence_contribution"] == 0

    def evidence(record):
        return {"source": {"method": method, "chars": len(doc.text), proof.COMMITMENT: canonical_hash(record)},
                "coverage": doc.meta | {proof.KEY: record}}

    positions = [asdict(row) for row in sentences]
    for record in (original, expanded):
        proof.verify_source(evidence(record), doc.text, doc.source_map, positions, metadata=doc.meta)
    forged = deepcopy(expanded)
    forged["ranges"][1]["basis"] = "pooled"
    with pytest.raises(ValueError, match="invalid highlighted ranges"):
        proof.verify_source(evidence(forged), doc.text, doc.source_map, positions, metadata=doc.meta)
    if suffix in {".odt", ".rtf"}:
        forged = deepcopy(expanded)
        forged["ranges"][1]["locations"][1]["ranges"][0][1] += 1
        with pytest.raises(ValueError, match="differ from the source text"):
            proof.verify_source(evidence(forged), doc.text, doc.source_map, positions, metadata=doc.meta)
