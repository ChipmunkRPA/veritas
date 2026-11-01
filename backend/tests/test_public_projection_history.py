"""Private historical identities still join and route after registry removal."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_image_evidence_access import evidence_http as evidence_http

from app.api import public_projection, v2_annotations, v2_evidence
from app.api.public_projection import alias_id, alias_signal, original_id, project
from app.api.v2 import install_problem_handlers
from app.audit import chain as audit
from app.core.canonical import canonical_hash
from app.evidence.image_packets import _public_signal
from app.storage.db import Analysis, AnnotationRow, CaseMemberRow, PrincipalRow, SignalRow


def test_validation_coverage_hides_generator_names_without_rewriting_source():
    wording = '2025–26 (one frontier model) plus Qwen2.5-3B and ChatGPT-2022 outputs'
    raw = {'manifest': {'detectors': [{'coverage_date': wording}]},
           'document': {'text': wording}, 'metadata': {'Software': 'Qwen2.5-3B'}}
    shown = project(raw)
    coverage = shown['manifest']['detectors'][0]['coverage_date']
    assert 'Qwen' not in coverage and 'ChatGPT' not in coverage
    assert '2025–26' in coverage
    assert shown['document']['text'] == wording
    assert shown['metadata'] == raw['metadata']


@pytest.mark.parametrize("root", ["vid.swap.retired", "retired_pipeline"])
def test_unknown_identity_has_digest_alias_and_quotes_are_preserved(root):
    signal_id = root + ":track:0"
    name = "Retired Private Architecture 2099"
    raw = {"signals": [{"id": signal_id, "display_id": "S01", "detector_id": root, "name": name,
                        "status": "ok", "observed": f'{name} returned 0.75; source says “{name}”.',
                        "details": {"model_name": name}}],
           "findings": [{"id": "F1", "family": "F3_whole_image_generation", "evidence_ids": [signal_id],
                         "statement": f'{name} returned 0.75; source quote "{name}".'},
                        {"id": "F2", "family": "F1_provenance", "statement": f"Source author wrote {name}"}],
           "metadata": {"Software": name}, "annotations": [{"body": name}],
           "manifest": {"models": ["unknown_private_weights_2099"]}}
    before = copy.deepcopy(raw)
    digest = canonical_hash(raw)
    shown = project(raw)
    alias = "check." + hashlib.sha256(root.encode()).hexdigest()[:16] + ":track:0"
    assert alias_signal(signal_id) == shown["signals"][0]["id"] == alias
    assert shown["findings"][0]["evidence_ids"] == [alias]
    assert name not in shown["signals"][0]["name"]
    assert shown["signals"][0]["observed"] == f'a detection check returned 0.75; source says “{name}”.'
    assert shown["findings"][0]["statement"] == f'a detection check returned 0.75; source quote "{name}".'
    assert shown["findings"][1] == raw["findings"][1]
    assert shown["metadata"] == raw["metadata"] and shown["annotations"] == raw["annotations"]
    assert shown["signals"][0]["details"] == {}
    assert "unknown_private_weights_2099" not in json.dumps(shown)
    assert raw == before and canonical_hash(raw) == digest
    assert original_id(alias, candidates=[signal_id, "other_old_pipeline:track:0"]) == signal_id
    assert original_id(alias, candidates=["other_old_pipeline:track:0"]) == alias
    assert _public_signal(raw["signals"][0])["id"] == alias


def test_known_aliases_and_public_image_references_remain_byte_identical():
    expected = "check." + hashlib.sha256(b"img.dda").hexdigest()[:16] + ":global"
    assert alias_signal("img.dda:global") == alias_id("img.dda:global") == expected
    assert original_id(expected) == "img.dda:global"
    assert alias_signal("img.forensic.ela:global") == "img.forensic.ela:global"
    assert alias_signal("img.meta.exif:global") == "img.meta.exif:global"
    assert _public_signal({"id": "img.forensic.ela:global", "display_id": "S01"}) == {
        "id": "img.forensic.ela:global", "display_id": "S01"}
    # An unknown descendant must not expose its private suffix after replacement
    # of a shorter registered root (img.dda).
    assert "retired" not in alias_signal("img.dda.retired:global")


def test_alias_mapping_refresh_does_not_leave_stale_values_or_match_patterns(monkeypatch):
    original = alias_signal("img.dda:global")
    with monkeypatch.context() as patch:
        patch.setattr(public_projection, "_identities", lambda: ({"img.dda": "check.temporary"}, ("DDA",)))
        assert alias_signal("img.dda:global") == "check.temporary:global"
        assert project({"signals": [{"id": "img.dda:global", "detector_id": "img.dda"}],
                        "findings": [{"evidence_ids": ["img.dda:global"]}]})["findings"][0]["evidence_ids"] == ["check.temporary:global"]
        # Same candidates, changed mapping keys: the compiled pattern must follow
        # the actual identity keys too, not just the candidate tuple.
        patch.setattr(public_projection, "_identities", lambda: ({"retired_plain": "check.changed"}, ()))
        assert alias_id("retired_plain") == "check.changed"
        assert alias_signal("img.dda:global") == "img.dda:global"
        patch.setattr(public_projection, "_identities", lambda: ({}, ()))
        assert alias_id("F1") == "F1"
    assert alias_signal("img.dda:global") == alias_id("img.dda:global") == original
    assert original_id(original) == "img.dda:global"


def test_ordinary_date_uuid_and_sort_cursor_values_are_unchanged():
    from app.api.public_projection import public_cursor
    from app.api.v2 import decode_cursor, encode_cursor, paginate

    keys = [("2026-10-31T12:00:00Z", "62778d30-f2e6-46f6-8f88-5f0889d2c38f"),
            ("2026-10-31T12:00:01Z", "940d641c-e6e2-4d39-95b4-c5beae1d86a8")]
    first, cursor = paginate(keys, lambda item: item, None, 1)
    assert first == keys[:1] and public_cursor(cursor) == encode_cursor(keys[0])
    assert decode_cursor(cursor, 2) == list(keys[0])
    assert paginate(keys, lambda item: item, public_cursor(cursor), 1) == (keys[1:], None)


def test_document_v1_shape_hides_method_names_but_retains_scores_ocr_and_verbatim_text():
    # Same field structure as the real rotated OCR document export. Source text
    # intentionally contains model names so broad recursive text replacement fails.
    text = "Source writer mentions DeBERTa, Binoculars and Qwen2.5-1.5B."
    ocr = {"pages": [{"page": 1, "engine_version": "tesseract 5.4.1",
                       "language_data_sha256": {"eng.traineddata": "a" * 64}, "languages_requested": "eng",
                       "mean_recognition_confidence": 96.4, "language": {"code": "en", "confidence": 1.0},
                       "recognized_text": text, "words": [{"text": "DeBERTa", "bbox": [1, 2, 3, 4]}]}]}
    raw = {"result": {"text": text, "paragraphs": [[0, len(text)]], "document": {"ocr": ocr},
                       "source_map": [{"ocr": copy.deepcopy(ocr)}],
                       "sentences": [{"text": text, "start": 0, "end": len(text),
                                      "reasons": ["DeBERTa AI-text classifier: 91% on this sentence with its neighbours",
                                                  "Binoculars score 1.557 vs threshold 0.9505"],
                                      "scores": {"classifier": .9146, "classifier_secondary": .8,
                                                 "binoculars": 1.557, "binoculars_p": 0.0}}],
                       "stats": {"words": 222, "binoculars_document": 1.1053},
                       "signals": [{"id": "classifier_document", "category": "model", "name": "DeBERTa",
                                    "reference": "DeBERTa-v3-large fine-tuned on human vs LLM text; 0.5 boundary"},
                                   {"id": "binoculars_document", "category": "statistical", "name": "Binoculars",
                                    "reference": "Calibrated threshold 0.9505 (Qwen2.5-1.5B pair)"}],
                       "verdict": {"caveats": ["Text classifier that scored this document: desklib-ft-v2 (Veritas fine-tune, DeBERTa-v3-large)."]}}}
    original = copy.deepcopy(raw)
    shown = project(raw)["result"]
    assert shown["text"] == text and shown["sentences"][0]["text"] == text
    assert shown["sentences"][0]["scores"] == {"classification_response": .9146, "classification_response_secondary": .8,
                                              "pattern_ratio": 1.557, "pattern_response": 0.0}
    assert shown["stats"] == {"words": 222, "document_pattern_ratio": 1.1053}
    generated = json.dumps([shown["sentences"][0]["reasons"], shown["signals"], shown["verdict"]])
    assert all(name not in generated for name in ("DeBERTa", "Binoculars", "Qwen2.5", "desklib", "Veritas fine-tune"))
    assert "91%" in generated and "1.557" in generated and "0.9505" in generated and "0.5 boundary" in generated
    page = shown["document"]["ocr"]["pages"][0]
    assert page["recognized_text"] == text and page["words"] == ocr["pages"][0]["words"]
    assert page["mean_recognition_confidence"] == 96.4 and page["language"]["code"] == "en"
    assert not {"engine_version", "language_data_sha256", "languages_requested"}.intersection(page)
    assert shown["source_map"][0]["ocr"] == shown["document"]["ocr"]
    assert raw == original
    assert project({"result": shown})["result"]["sentences"][0]["scores"] == shown["sentences"][0]["scores"]


def test_pdf_generated_references_and_validation_names_are_neutral_but_source_values_stay_exact():
    from app.reports.public_view import public_view
    from app.reports.specimens import specimen

    frozen = specimen("document", "standard")
    shown = {"detectors": {"ledger": [{"id": "S01", "name": "Internal classifier",
               "reference": 'DeBERTa-v3-large fine-tuned; threshold 0.5; source says “DeBERTa”.',
               "observed": "Source metadata contains DeBERTa"}],
               "detectors": [{"testing": {"text": "Qwen2.5-3B and ChatGPT-2022; source says 'Qwen2.5-3B'."}}]},
             "limitations": [{"text": "Text classifier that scored this document: desklib-ft-v2 (Veritas fine-tune, DeBERTa-v3-large)."}]}
    before = copy.deepcopy(shown)
    projected = public_view(frozen, shown)
    reference = projected["detectors"]["ledger"][0]["reference"]
    assert "DeBERTa-v3-large" not in reference and "threshold 0.5" in reference
    assert 'source says “DeBERTa”' in reference
    assert projected["detectors"]["ledger"][0]["observed"] == "Source metadata contains DeBERTa"
    validation = projected["detectors"]["detectors"][0]["testing"]["text"]
    assert validation == "A detection check and a detection check; source says 'Qwen2.5-3B'."
    assert "desklib" not in projected["limitations"][0]["text"]
    assert "DeBERTa" not in projected["limitations"][0]["text"]
    assert shown == before


@pytest.fixture
def historical_http(evidence_http, monkeypatch):
    f = evidence_http

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    for module in (v2_annotations, audit):
        monkeypatch.setattr(module, "get_session", sessions)
    ids = ["retired_pipeline:global", "vid.aaa.retired:global", "vid.zzz.retired:track:3"]
    with Session(f.engine) as db:
        db.get(PrincipalRow, f.viewer_id).role = "examiner"
        db.get(CaseMemberRow, ("visible-case", f.viewer_id)).role = "examiner"
        for index, sid in enumerate(ids):
            db.add(SignalRow(analysis_id="image-a", signal_id=sid, status="ok", payload={
                "id": sid, "detector_id": sid.split(":", 1)[0], "display_id": f"S{index + 1:02}", "status": "ok",
                "name": f"Retired Private {index}"}))
        db.add(SignalRow(analysis_id="image-b", signal_id="vid.other.private:global", status="ok",
                         payload={"id": "vid.other.private:global", "detector_id": "vid.other.private", "status": "ok"}))
        frozen_hash = db.get(Analysis, "image-a").result_hash
        frozen_json = db.get(Analysis, "image-a").result_json
        db.commit()
    app = FastAPI()
    install_problem_handlers(app)
    app.include_router(v2_evidence.router)
    app.include_router(v2_annotations.router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, engine=f.engine, headers=f.viewer, ids=ids,
                              frozen_hash=frozen_hash, frozen_json=frozen_json)


def test_real_signal_pages_reverse_historical_aliases_without_leaks_or_repeats(historical_http):
    f = historical_http
    cursor, seen = None, []
    for _ in range(len(f.ids) + 1):
        response = f.client.get("/api/v2/analyses/image-a/signals", headers=f.headers,
                                params={"limit": 1, **({"cursor": cursor} if cursor else {})})
        assert response.status_code == 200, response.text
        page = response.json()
        assert not any(sid in response.text for sid in f.ids)
        seen.extend(row["id"] for row in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
        decoded = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        assert not any(sid in decoded for sid in f.ids)
    assert seen == [alias_signal(sid) for sid in sorted(f.ids)]
    assert len(seen) == len(set(seen))


def test_real_annotations_resolve_only_own_analysis_and_preserve_user_text(historical_http):
    f = historical_http
    note = "The source mentions Retired Private 0 and vid.aaa.retired."
    for sid in f.ids:
        alias = alias_signal(sid)
        response = f.client.post("/api/v2/analyses/image-a/annotations", headers=f.headers,
                                 json={"target_type": "signal", "target_id": alias, "body": note})
        assert response.status_code == 201, response.text
        annotation = response.json()
        assert annotation["target_id"] == alias and annotation["body"] == note
        listed = f.client.get("/api/v2/analyses/image-a/annotations", headers=f.headers,
                              params={"target_type": "signal", "target_id": alias})
        assert listed.status_code == 200 and [r["id"] for r in listed.json()["items"]] == [annotation["id"]]
        with Session(f.engine) as db:
            row = db.get(AnnotationRow, annotation["id"])
            assert row.target_id == sid and row.body == note
    foreign = alias_signal("vid.other.private:global")
    refused = f.client.post("/api/v2/analyses/image-a/annotations", headers=f.headers,
                            json={"target_type": "signal", "target_id": foreign, "body": note})
    assert refused.status_code == 422 and "vid.other.private" not in refused.text
    with Session(f.engine) as db:
        assert list(db.scalars(select(SignalRow.signal_id).where(SignalRow.analysis_id == "image-a").order_by(SignalRow.signal_id))) == sorted(f.ids)
        original = db.get(Analysis, "image-a")
        assert original.result_hash == f.frozen_hash and original.result_json == f.frozen_json


def test_generated_assessment_notes_redact_names_but_user_notes_stay_verbatim():
    name = 'DFDC EfficientNet-B7 on face frames (selimsef, two-checkpoint ensemble)'
    raw = {'assessments': [{'note': 'no estimate: ' + name}], 'note': name,
           'annotations': [{'note': name}], 'intake_notes': name}
    shown = project(raw)
    assert 'DFDC' not in shown['assessments'][0]['note']
    assert shown['note'] == raw['note']
    assert shown['annotations'] == raw['annotations']
    assert shown['intake_notes'] == name


def test_retired_fusion_slot_uses_public_alias():
    value = {'fusion': {'intentionally_missing': ['aud.gustking_xlsr']}}
    shown = project(value)
    assert shown['fusion']['intentionally_missing'] == [alias_signal('aud.gustking_xlsr')]


def test_quoted_skipped_check_diagnostic_is_not_a_source_quotation():
    raw = {'findings': [{'id': 'F1', 'family': 'coverage',
                        'statement': 'Not run: “YuNet face detector skipped: memory budget”.',
                        'sentences': [{'text': 'Not run: “YuNet skipped”.', 'evidence_ids': ['aud.pella_v2:global']}]}]}
    shown = project(raw)
    assert 'YuNet' not in str(shown)
    assert shown['findings'][0]['sentences'][0]['evidence_ids'] == [alias_signal('aud.pella_v2:global')]
