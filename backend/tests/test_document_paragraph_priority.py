from __future__ import annotations

import zipfile
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_document_matched_context import _fuse
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.text.adapters.learned import FourClassTextDetector
from app.engine.boundary import Deadline
from app.evidence import document_matched_context
from app.evidence import document_paragraph_priority as priority
from app.evidence.document_packets import (
    DocumentPacketIntegrityError,
    build_packet_set,
    load_packet_set,
    packet_hash,
    validate_packet_set,
)
from app.reports import engine, multimedia_evidence, public_package, public_verify, verify_document_source
from app.reports.specimens import specimen

BODY = "\n\n".join((
    "The archive team catalogued instrument records from the northern watershed during spring. The staff checked each "
    "item against a handwritten ledger for several previous seasons. The notes describe the field crew's route through "
    "the valley and their return across the bridge.",
    "The survey started at a low point in the stream bed before sunrise. Each station supplied a series of measurements "
    "along the full length of the bank. Staff compared those readings and kept a separate copy of every sheet.",
    "The later report gives the names of the staff who inspected the stream. Several older maps show the places where "
    "they worked throughout the winter months. The field notes say that the team used the same method near the lake.",
    "The office stored the signed reports and the working notes in one shared room. The crew left copies of their "
    "drawings with the people who maintain the lake road. Staff can review the instruments and the route again after "
    "the next collection period.",
))
SCORES = (.1, .8, .1, .8, .75, .55, .95000911773, .91, .8, .1, .1, .1)


def make_packets(packet_factory, monkeypatch, *, environment="dev", prioritize=True):
    db, store, analysis, result, decoded, original = packet_factory(scores=SCORES, text=BODY)
    monkeypatch.setattr(settings, "environment", environment)
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_matched_context_review", False)
    monkeypatch.setattr(settings, "text_isolated_paragraph_review", True)
    monkeypatch.setattr(settings, "text_paragraph_evidence_priority", prioritize)

    class Classifier:
        calls = 0

        def tok(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split()) + 2))}

        def predict(self, inputs, **kwargs):
            self.calls += 1
            self.last_coverage = [{"tokens": 35, "covered_tokens": 35, "token_windows": 1} for _ in inputs]
            return list(SCORES) if len(inputs) == len(SCORES) else [.81] * len(inputs)

    classifier = Classifier()
    detector = FourClassTextDetector()
    monkeypatch.setattr(FourClassTextDetector, "backend", lambda self, ctx: classifier)
    ctx = SimpleNamespace(outputs=SimpleNamespace(document=decoded), deadline=Deadline(),
                          lease={detector.MODEL_ID: SimpleNamespace(revision=document_matched_context.REVISION)})
    ordinary = detector.run(ctx)
    decision = _fuse(detector, ordinary, decoded)
    assert classifier.calls == 3
    assert (priority.KEY in ordinary.details["text_evidence"]) is (environment == "dev" and prioritize)
    primary = decision["signals"][0]
    opposing = deepcopy(primary)
    opposing.update(id="authored-second-reading", detector_id="authored.secondary", display_id="S02")
    proof = opposing["details"]["text_evidence"]
    proof.pop("isolated_paragraph_review")
    proof.pop(priority.KEY, None)
    for row in proof["windows"]:
        row["score"] = .2
        row.pop("score_exact_hex", None)
    result["signals"] = [primary, opposing]
    result["findings"] = [{"id": "F1", "evidence_ids": [primary["id"]]}]
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = build_packet_set(db, analysis, store, result)
    return db, store, analysis, result, decoded, original, evidence, decision


def test_priority_moves_one_retained_passage_preserving_its_id_conflicts_and_confidence(packet_factory, monkeypatch):
    db, store, analysis, result, decoded, _, enabled, decision = make_packets(packet_factory, monkeypatch)
    ref = enabled["coverage"][priority.KEY]
    assert enabled["version"] == "document-evidence-7" and ref["authorship_confidence_contribution"] == 0
    assert enabled["packets"][0]["focus"]["text"] == decoded["sentences"][6]["text"]
    assert [w["direction"] for w in enabled["packets"][0]["supporting_windows"]] == ["supports_ai", "opposes_ai"]
    assert load_packet_set(db, analysis, store) == enabled
    assert multimedia_evidence.verify_assets(enabled, store, result=result) == {}

    result["signals"][0]["details"]["text_evidence"].pop(priority.KEY)
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    baseline = build_packet_set(db, analysis, store, result)
    assert baseline["version"] == "document-evidence-5"
    assert baseline["packets"][0]["id"] != ref["packet_id"]
    assert {p["id"]: p for p in enabled["packets"]} == {p["id"]: p for p in baseline["packets"]}
    for key in ("opposing_windows_recorded", "opposing_windows_displayed", "opposing_windows_omitted"):
        assert enabled["coverage"][key] == baseline["coverage"][key]
    assert decision["decision"]["probability_ai_calibrated"] is None


@pytest.mark.parametrize("environment,prioritize", [("prod", True), ("dev", False)])
def test_source_order_remains_when_priority_is_unavailable_or_off(packet_factory, monkeypatch, environment, prioritize):
    _, _, _, _, _, _, evidence, _ = make_packets(packet_factory, monkeypatch, environment=environment, prioritize=prioritize)
    assert evidence["version"] == "document-evidence-5"
    assert priority.KEY not in evidence["coverage"]
    assert evidence["packets"][0]["id"] == "T01"


def test_priority_geometry_and_saved_order_are_verified(packet_factory, monkeypatch):
    _, _, analysis, _, decoded, _, evidence, _ = make_packets(packet_factory, monkeypatch)
    assert not priority.eligible(decoded["text"], decoded["source_map"], "pymupdf")
    assert not priority.eligible(decoded["text"], decoded["source_map"][1:], "plain")
    for mutation in ("reorder", "wrong_id", "omit"):
        changed = deepcopy(evidence)
        if mutation == "reorder":
            changed["packets"].reverse()
        elif mutation == "wrong_id":
            changed["coverage"][priority.KEY]["packet_id"] = "T01"
        else:
            del changed["coverage"][priority.KEY]
        changed["sha256"] = packet_hash(changed)
        with pytest.raises(DocumentPacketIntegrityError, match="invalid source commitments or positions"):
            validate_packet_set(changed, analysis_id=analysis.id, revision=analysis.revision,
                                original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


def test_priority_agrees_in_pdf_json_and_source_replay(packet_factory, monkeypatch, tmp_path):
    _, store, analysis, result, _, original, evidence, _ = make_packets(packet_factory, monkeypatch)
    snapshot = specimen("document", "standard")
    report_result = {**deepcopy(snapshot.result), "signals": result["signals"]}
    report_result["input"].update(sha256=analysis.sha256, filename=analysis.filename,
                                 mime_sniffed=analysis.content_type, size_bytes=analysis.size_bytes)
    report_result["manifest"]["sampling"] = deepcopy(result["manifest"]["sampling"])
    evidence.update(analysis_id=snapshot.analysis.id, revision=snapshot.analysis.revision,
                    original_result_hash=canonical_hash(report_result))
    evidence["sha256"] = packet_hash(evidence)
    snapshot = snapshot.model_copy(update={"result": report_result, "evidence": evidence,
        "media": snapshot.media.model_copy(update={"sha256": analysis.sha256, "filename": analysis.filename,
            "content_type": analysis.content_type, "size_bytes": analysis.size_bytes})})
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    exported = public_package._evidence(snapshot)
    assert exported["coverage"][priority.KEY] == evidence["coverage"][priority.KEY]
    assert document_matched_context.MODEL.encode() not in canonical_json(exported)
    assert multimedia_evidence.view(snapshot, store=store)["packets"][0]["paragraph_priority_note"] == priority.NOTE
    rendered = engine.render_pdf(snapshot)
    pdf_text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert not rendered.refused and priority.NOTE in pdf_text
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    path = tmp_path / "package.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(exported))
    with zipfile.ZipFile(path) as archive:
        assert "extracted text and source mapping matched" in verify_document_source.check(archive, bundle, original)
