from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import Settings, settings
from app.detectors.text.adapters.learned import FourClassTextDetector
from app.engine.boundary import Deadline
from app.engine.stages import common, document
from app.evidence import document_matched_context as review
from app.evidence import document_scope
from app.evidence.document_packets import (
    DocumentPacketIntegrityError,
    build_packet_set,
    load_packet_set,
    packet_hash,
    validate_packet_set,
)
from app.reports.multimedia_evidence import verify_assets
from app.storage.objects import ObjectIntegrityError

BODY = (
    "The archive team catalogued a new set of instrument records from the northern watershed and compared them with "
    "notes from the previous season. The measurements cover several closely spaced intervals and the table identifies "
    "which readings were collected while the gauge was fully submerged. A second copy of the report contains several "
    "amended descriptions of field conditions, including updates from the crew that visited in autumn. The local office "
    "will keep the original map and the notes so staff can answer questions about the record in future reviews."
)
SCORES = (0.91, 0.31, 0.12, 0.89)


def test_runtime_rule_matches_published_frozen_fit():
    root = Path(__file__).resolve().parents[2]
    panel = json.loads((root / "docs/evaluation-panels/writing-two-sided-context-2026-11-11.json").read_bytes())
    fitted = panel["fitted"]
    raw = (json.dumps(fitted, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    assert hashlib.sha256(raw).hexdigest() == review.FIT_SHA256
    fit = fitted["arms"]["fitted_window"]
    for score in (0, 0.001, 0.1, 0.2129089236259463, 0.2129089236259466, 0.3, 0.9, 1):
        p = min(max(score, 1e-6), 1 - 1e-6)
        log_odds = math.log(p / (1 - p))
        expected = fit["intercept"] + fit["coefficient"][0] * ((log_odds - fit["mean"][0]) / fit["scale"][0])
        assert review.selected(score) is (expected > fit["point"])


def _run_adapter(monkeypatch, decoded, *, revision=review.REVISION, scores=SCORES, overflow=False):
    calls = []

    class Classifier:
        def predict(self, inputs, **kwargs):
            calls.append(inputs)
            self.last_coverage = [{"tokens": 50, "token_windows": 1} for _ in inputs]
            if overflow and len(inputs) == 4:
                self.last_coverage[1]["token_windows"] = 2
            return list(scores) if len(inputs) == 4 else [0.81]

    classifier = Classifier()
    detector = FourClassTextDetector()
    monkeypatch.setattr(FourClassTextDetector, "backend", lambda self, ctx: classifier)
    ctx = SimpleNamespace(outputs=SimpleNamespace(document=decoded), deadline=Deadline(),
                          lease={detector.MODEL_ID: SimpleNamespace(revision=revision)})
    return detector, detector.run(ctx), calls


def _fuse(detector, run, decoded):
    entry = common.detector_entry(detector.spec, run, None)
    ctx = SimpleNamespace(
        analysis_id="fixture",
        state={},
        outputs={"decode": decoded, "detect": {"detectors": [entry]}},
        exhibits=SimpleNamespace(add=lambda *args, **kwargs: "fixture", exhibits=[], artefacts=[]),
    )
    return document.fuse(ctx)


def _settings(monkeypatch, *, enabled):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "text_detector_mode", "candidate")
    monkeypatch.setattr(settings, "text_window_sentences", 3)
    monkeypatch.setattr(settings, "text_isolated_paragraph_review", False)
    monkeypatch.setattr(settings, "text_passage_position_review", False)
    monkeypatch.setattr(settings, "text_long_passage_position_review", False)
    monkeypatch.setattr(settings, "text_matched_context_review", enabled)


def _fixture(packet_factory, monkeypatch, *, enabled=True, scores=SCORES):
    _settings(monkeypatch, enabled=enabled)
    db, store, analysis, result, decoded, path = packet_factory(scores=scores, text=BODY)
    detector, run, calls = _run_adapter(monkeypatch, decoded, scores=scores)
    fused = _fuse(detector, run, decoded)
    result["signals"] = fused["signals"]
    result["findings"] = [{"id": "F1", "evidence_ids": [fused["signals"][0]["id"]]}]
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    return db, store, analysis, result, decoded, path, run, calls, fused


def test_opt_in_reuses_scores_and_fusion_and_changes_only_passage_selection(packet_factory, monkeypatch):
    assert Settings.model_fields["text_matched_context_review"].default is False
    db, store, analysis, result, decoded, _, enabled, calls, fused = _fixture(packet_factory, monkeypatch)
    evidence = build_packet_set(db, analysis, store, result)
    packet, = evidence["packets"]
    assert evidence["version"] == "document-evidence-6"
    assert evidence["coverage"][review.KEY]["eligible_sentences"] == 2
    assert evidence["coverage"][review.KEY]["selected_sentences"] == 1
    assert packet[review.KEY]["authorship_confidence_contribution"] == 0
    assert packet["focus"]["text"] == decoded["sentences"][1]["text"]
    window, = packet["supporting_windows"]
    assert packet["inspection_scope"] == {"basis": "scored_context", "spans": window["spans"]}
    assert window[review.WINDOW_KEY] is True and window["direction"] == "opposes_ai"
    assert window["score_exact_hex"] == float(SCORES[1]).hex()
    assert window["text"] == calls[0][1]
    assert len(calls) == 2
    assert review.MODEL.encode() not in canonical_json(evidence)
    assert review.REVISION.encode() not in canonical_json(evidence)
    assert str(review.POINT).encode() not in canonical_json(evidence)
    assert load_packet_set(db, analysis, store) == evidence
    assert verify_assets(evidence, store, result=result) == {}

    monkeypatch.setattr(settings, "text_matched_context_review", False)
    detector, ordinary, baseline_calls = _run_adapter(monkeypatch, decoded)
    baseline = _fuse(detector, ordinary, decoded)
    assert ordinary.unit_scores == enabled.unit_scores and ordinary.aggregate == enabled.aggregate
    assert baseline_calls == calls
    assert baseline["decision"] == fused["decision"]
    assert baseline["v1"]["verdict"] == fused["v1"]["verdict"]
    assert review.KEY not in ordinary.details["text_evidence"]
    result["signals"] = baseline["signals"]
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    legacy = build_packet_set(db, analysis, store, result)
    assert legacy["version"] == "document-evidence-5"
    assert review.KEY not in legacy["coverage"]
    assert [p["focus"]["text"] for p in legacy["packets"]] == [
        decoded["sentences"][i]["text"] for i in (0, 3)
    ]


@pytest.mark.parametrize("mismatch", ["checkpoint", "overflow", "language", "source", "configuration", "production"])
def test_opt_in_leaves_out_of_scope_documents_on_existing_selector(packet_factory, monkeypatch, mismatch):
    _settings(monkeypatch, enabled=True)
    _, _, _, _, decoded, _ = packet_factory(scores=SCORES, text=BODY)
    if mismatch == "language":
        monkeypatch.setattr("app.detectors.text.extract._ocr_language",
                            lambda _: {"code": "es", "status": "unknown_or_unsupported"})
    if mismatch == "source":
        decoded["meta"]["method"] = "email"
    if mismatch == "configuration":
        monkeypatch.setattr(settings, "text_window_sentences", 5)
    if mismatch == "production":
        monkeypatch.setattr(settings, "environment", "prod")
    _, run, calls = _run_adapter(
        monkeypatch, decoded, revision="other" if mismatch == "checkpoint" else review.REVISION,
        overflow=mismatch == "overflow",
    )
    assert len(calls) == 2 and review.KEY not in run.details["text_evidence"]


def test_absence_of_selected_sentence_is_recorded_without_authorship_direction(packet_factory, monkeypatch):
    db, store, analysis, result, _, _, _, _, fused = _fixture(packet_factory, monkeypatch, scores=(0.94, 0.1, 0.11, 0.9))
    evidence = build_packet_set(db, analysis, store, result)
    assert evidence["packets"] == []
    assert evidence["coverage"][review.KEY]["selected_sentences"] == 0
    assert fused["decision"]["code"] == "inconclusive"
    assert fused["decision"]["probability_ai_calibrated"] is None


def test_matched_scope_is_exact_even_when_another_check_reduces_over_the_whole_context():
    spans = [{"start": 2, "end": 24}, {"start": 25, "end": 43}, {"start": 45, "end": 65}]
    matched = {review.WINDOW_KEY: True, "direction": "opposes_ai", "spans": spans}
    other = {"direction": "supports_ai", "score_kind": "perplexity_ratio",
             "scored_spans": [{"start": 2, "end": 65}], "neighbor_expansion": True}
    assert document_scope.build(spans[1], [matched, other], 70) == {"basis": "scored_context", "spans": spans}


def test_tampered_response_geometry_and_legacy_marker_are_refused(packet_factory, monkeypatch):
    db, store, analysis, result, _, _, _, _, _ = _fixture(packet_factory, monkeypatch)
    evidence = build_packet_set(db, analysis, store, result)
    for mutation in ("geometry", "response", "legacy", "display", "overflow"):
        changed = deepcopy(evidence)
        packet = changed["packets"][0]
        context = changed["coverage"][review.KEY]["contexts"][0]
        if mutation == "geometry":
            context["spans"][0]["start"] += 1
            packet["supporting_windows"][0]["spans"][0]["start"] += 1
            packet["inspection_scope"]["spans"][0]["start"] += 1
        elif mutation == "response":
            exact = 0.32.hex()
            context["score_exact_hex"] = packet["supporting_windows"][0]["score_exact_hex"] = exact
            packet["supporting_windows"][0]["score"] = 0.32
        elif mutation == "legacy":
            changed["version"] = "document-evidence-5"
        elif mutation == "overflow":
            context["score_exact_hex"] = "0x1p+1024"
        else:
            packet["supporting_windows"][0]["score"] = 0.32
        changed["sha256"] = packet_hash(changed)
        if mutation in {"geometry", "response"}:
            with pytest.raises(ObjectIntegrityError, match="recorded source extraction"):
                verify_assets(changed, store, result=result)
        else:
            with pytest.raises(DocumentPacketIntegrityError, match="invalid source commitments"):
                validate_packet_set(changed, analysis_id=analysis.id, revision=analysis.revision,
                                    original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
