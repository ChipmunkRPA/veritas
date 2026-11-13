from __future__ import annotations

import base64
import hashlib
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from io import BytesIO

import pymupdf
from docx import Document as WordDocument
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_stripe_billing import billing as billing
from test_stripe_billing import counts, event_for, signed

from app.api import v2_document_archive
from app.billing import purchase_notice
from app.core.config import settings
from app.detectors.text.extract import from_string
from app.evidence import document_archive, document_gao_archive
from app.main import app
from app.storage.db import Analysis, Artefact, BillingOrderRow, CaseMemberRow, CaseRow, Job

SOURCE = "frus1964-68v24/d585"


def _body():
    catalog, _ = document_archive.library()
    return "\n\n".join(row["text"] for row in catalog[SOURCE]["paragraphs"])


@contextmanager
def _install(fixture, monkeypatch):
    monkeypatch.setattr(v2_document_archive, "get_session", fixture.sessions)
    monkeypatch.setattr(v2_document_archive, "default_store", lambda: fixture.store)
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "document_archive_comparison_enabled", True)
    client = TestClient(app)
    try:
        yield client
    finally:
        client.close()


def _seed(fixture, identifier, text, *, owner=None, filename="original.txt"):
    retained = fixture.store.put_bytes(text.encode() if isinstance(text, str) else text)
    owner = owner or fixture.owner.id
    with fixture.sessions() as db:
        case_id = "archive-fixture"
        if db.get(CaseRow, case_id) is None:
            db.add(CaseRow(id=case_id, reference="archive-test", title="Archive test", classification="internal",
                           status="open", created_by=fixture.owner.id))
            db.flush()
            for principal in (fixture.owner, fixture.outsider):
                db.add(CaseMemberRow(case_id=case_id, principal_id=principal.id, role="owner" if principal == fixture.owner else "viewer",
                                     added_by=fixture.owner.id))
        db.add(Analysis(id=identifier, filename=filename, kind="document", content_type="text/plain",
                        status="done", submitted_by=owner, case_id=case_id, sha256=retained.sha256,
                        size_bytes=retained.size_bytes, result_hash="f" * 64))
        db.commit()
    return retained


def _url(aid):
    return f"/api/v2/analyses/{aid}/document-archive-comparison"


def test_local_archive_uses_unchanged_and_no_match_without_exposing_a_reference(billing, monkeypatch):
    fixture = billing
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    unchanged = _body()
    unrelated = "A community library opens at nine.\n\nThe manager posted a notice.\n\nA gardener watered the new trees."
    _seed(fixture, "unchanged", unchanged)
    _seed(fixture, "unknown", unrelated)
    _seed(fixture, "foreign-owner", unchanged, owner=fixture.outsider.id)
    _seed(fixture, "not-plain", unchanged, filename="sample.eml")
    with _install(fixture, monkeypatch) as client:
        known = client.post(_url("unchanged"), headers=fixture.headers)
        assert known.status_code == 200, known.text
        result = known.json()["comparison"]
        assert result["status"] == "reference_bound" and result["changes"] == []
        assert result["layout"] == "submitted_paragraphs"
        assert result["reference"]["id"] == SOURCE
        assert len(result["aligned_paragraphs"]) == 3
        other = client.post(_url("unknown"), headers=fixture.headers).json()["comparison"]
        assert other["status"] == "no_qualified_reference"
        assert other["reference"] is None and other["changes"] == [] and other["aligned_paragraphs"] == []
        assert unrelated not in str(other)
        assert other["authorship_confidence_contribution"] == 0
        for aid in ("foreign-owner", "not-plain"):
            assert client.get(_url(aid), headers=fixture.headers).json() == {"available": False}
            assert client.post(_url(aid), headers=fixture.headers).status_code == 404
        assert client.post(_url("unchanged")).status_code == 401
        monkeypatch.setattr(settings, "document_archive_comparison_enabled", False)
        assert client.get(_url("unchanged"), headers=fixture.headers).json() == {"available": False}
        assert client.post(_url("unchanged"), headers=fixture.headers).status_code == 404


def test_gao_archive_local_transcript_matches_api_and_pdf(billing, monkeypatch, tmp_path):
    import json

    from scripts.review_gao_archive_export import review

    monkeypatch.setattr(settings, "billing_mode", "disabled")
    transcript = ("A previous inquiry concerned a different financial period. "
                  "The regional audit office gathered records from several different departments and invited all "
                  "program administrators to comment on the analysis. The board reviewed the accounting process "
                  "in detail and approved its updated policy on Monday. The final review confirmed that the "
                  "field officers had recorded complete findings and resolved every outstanding concern in the "
                  "original worksheet. Several departments retained copies of that worksheet for their own records. "
                  "A separate committee will examine implementation during the following year.")
    changed = transcript.replace("Monday", "Thursday")
    crop_start = transcript.index("The regional audit office")
    crop_end = transcript.index(" A separate committee")
    crop = transcript[crop_start:crop_end]
    pack_path = tmp_path / "gao-pack.json"
    pack = {"schema": document_gao_archive.PACK_SCHEMA, "revision": document_gao_archive.REVISION,
            "reports": {"GAO-16-442": {"text": transcript, "published_date": "2016-04-18", "split": "train",
                                      "source_record_sha256": "1" * 64}}}
    payload = json.dumps(pack).encode()
    pack_path.write_bytes(payload)
    pack_sha = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(document_gao_archive, "REPORT_COUNT", 1)
    monkeypatch.setattr(settings, "document_gao_archive_path", pack_path)
    monkeypatch.setattr(settings, "document_gao_archive_sha256", pack_sha)
    _seed(billing, "gao-changed", changed)
    _seed(billing, "gao-unchanged", transcript)
    _seed(billing, "gao-exact-excerpt", crop)
    _seed(billing, "gao-altered-excerpt-edge", "An unsourced addition. " + crop)
    _seed(billing, "gao-unmatched", "> " + changed)
    with _install(billing, monkeypatch) as client:
        url = _url("gao-changed")
        assert client.get(url, headers=billing.headers).json() == {"available": True, "gao_available": True}
        assert client.post(url, headers=billing.headers).json()["comparison"]["schema"] == document_archive.SCHEMA
        selection = {"collection": "gao"}
        response = client.post(url, headers=billing.headers, params={"include_pdf": True, **selection})
        assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
        body = response.json()
        result = body["comparison"]
        assert result["schema"] == document_gao_archive.SCHEMA and result["authorship_confidence_contribution"] == 0
        assert result["reference"]["url"].endswith("/GAO-16-442")
        assert result["reference"]["transcript_range"] == [0, len(transcript)]
        assert result["reference"]["submitted_range"] == [0, len(changed)]
        change, = result["changes"]
        for side, text, word in (("submitted", changed, "Thursday"), ("archive", transcript, "Monday")):
            excerpt = change[side]
            assert text[excerpt["start"]:excerpt["end"]] == excerpt["text"] == word
            assert excerpt["changed_tokens"] == [[excerpt["start"], excerpt["end"]]]
        first, last = result["submitted_sentence_ranges"][0]
        assert changed[first:last] == "The board reviewed the accounting process in detail and approved its updated policy on Thursday."
        with pymupdf.open(stream=base64.b64decode(body["pdf"]), filetype="pdf") as pdf:
            text = " ".join(" ".join(page.get_text().split()) for page in pdf)
            assert "Published highlights transcript" in text and "Thursday" in text and "Monday" in text
            assert f"Submitted sentences containing differences [{first}, {last})" in text
            assert any(link.get("uri", "").endswith("/GAO-16-442") for page in pdf for link in page.get_links())
        submitted = tmp_path / "later.txt"
        submitted.write_text(changed)
        export = tmp_path / "comparison.json"
        export.write_text(json.dumps(result))
        assert review(export, submitted, pack_path, pack_sha)["exact_changes_verified"] == 1
        change["submitted"]["changed_tokens"][0][0] -= 1
        export.write_text(json.dumps(result))
        import pytest

        with pytest.raises(ValueError, match="does not replay"):
            review(export, submitted, pack_path, pack_sha)
        unchanged = client.post(_url("gao-unchanged"), headers=billing.headers, params=selection).json()["comparison"]
        assert unchanged["status"] == "reference_bound" and unchanged["changes"] == []
        exact_body = client.post(_url("gao-exact-excerpt"), headers=billing.headers,
                                 params={"include_pdf": True, **selection}).json()
        exact = exact_body["comparison"]
        assert exact["status"] == "reference_bound" and exact["changes"] == []
        assert exact["coverage"]["body_words"] == exact["coverage"]["matched_words"]
        assert exact["reference"]["submitted_range"] == [0, len(crop)]
        assert exact["reference"]["transcript_range"] == [crop_start, crop_end]
        with pymupdf.open(stream=base64.b64decode(exact_body["pdf"]), filetype="pdf") as pdf:
            exact_pdf = " ".join(" ".join(page.get_text().split()) for page in pdf)
            assert "No normalized wording differences in the compared passage." in exact_pdf
            assert f"Transcript range [{crop_start}, {crop_end})" in exact_pdf
        submitted.write_text(crop)
        export.write_text(json.dumps(exact))
        assert review(export, submitted, pack_path, pack_sha)["exact_changes_verified"] == 0
        reports, packed_index = document_gao_archive.library(pack_path, pack_sha)
        legacy = document_gao_archive.compare(from_string(crop), exact["submitted_sha256"], pack_sha, reports,
                                              packed_index, accept_exact_excerpt=False)
        assert legacy["status"] == "no_qualified_reference"
        export.write_text(json.dumps(legacy))
        assert review(export, submitted, pack_path, pack_sha)["status"] == "no_qualified_reference"
        legacy["coverage"]["matched_words"] = 1
        export.write_text(json.dumps(legacy))
        with pytest.raises(ValueError, match="does not replay"):
            review(export, submitted, pack_path, pack_sha)
        edge = client.post(_url("gao-altered-excerpt-edge"), headers=billing.headers, params=selection).json()["comparison"]
        assert edge["status"] == "no_qualified_reference" and edge["reference"] is None and edge["changes"] == []
        unknown = client.post(_url("gao-unmatched"), headers=billing.headers, params=selection).json()["comparison"]
        assert unknown["status"] == "out_of_scope" and unknown["reference"] is None and unknown["changes"] == []
        monkeypatch.setattr(settings, "document_gao_archive_sha256", "0" * 64)
        assert client.post(url, headers=billing.headers, params=selection).status_code == 503


def test_word_archive_change_keeps_native_paragraph_and_run_positions(billing, monkeypatch):
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    word = WordDocument()
    word.add_paragraph()
    lead, *tail = _body().split("\n\n")
    before, after = lead.split("opposing", 1)
    paragraph = word.add_paragraph()
    paragraph.add_run(f"  {before}alter")
    paragraph.add_run(f"ed{after}")
    for text in tail:
        word.add_paragraph(text)
    stream = BytesIO()
    word.save(stream)
    retained = _seed(billing, "word-source", stream.getvalue(), filename="source.DOCX")
    _seed(billing, "bad-word", b"not a Word package", filename="damaged.docx")
    with _install(billing, monkeypatch) as client:
        assert client.get(_url("word-source"), headers=billing.headers).json() == {"available": True}
        response = client.post(_url("word-source") + "?include_pdf=true", headers=billing.headers)
        assert response.status_code == 200, response.text
        result = response.json()
        comparison = result["comparison"]
        assert comparison["reference"]["id"] == SOURCE
        assert comparison["submitted_sha256"] == retained.sha256
        assert comparison["aligned_paragraphs"][0]["docx_paragraph"] == 2
        (change,) = comparison["changes"]
        assert change["archive"]["text"] == "opposing" and "docx_location" not in change["archive"]
        assert change["submitted"]["text"] == "altered"
        location = change["submitted"]["docx_location"]
        assert (location["paragraph"], location["body_index"], location["runs_complete"]) == (2, 1, True)
        assert location["paragraph_range"] == [len(before) + 2, len(before) + 9]
        assert location["changed_tokens"][0]["runs"] == [
            {"run": 1, "range": [len(before) + 2, len(before) + 7]}, {"run": 2, "range": [0, 2]},
        ]
        with pymupdf.open(stream=base64.b64decode(result["pdf"], validate=True), filetype="pdf") as pdf:
            assert "Word body paragraph 2" in " ".join(" ".join(page.get_text().split()) for page in pdf)
        assert client.post(_url("bad-word"), headers=billing.headers).status_code == 422


def test_real_paid_checkout_and_retained_archive_evidence(billing, monkeypatch):
    fixture = billing
    monkeypatch.setattr(settings, "document_gao_archive_path", None)
    text = _body().replace("opposing government", "op- posing government", 1)
    _seed(fixture, "unpaid-completed", text)
    compare = document_archive.compare
    calls = []

    def checked(doc, source):
        calls.append(source)
        return compare(doc, source)

    monkeypatch.setattr(document_archive, "compare", checked)
    with _install(fixture, monkeypatch) as client:
        assert client.get(_url("unpaid-completed"), headers=fixture.headers).json() == {"available": False}
        assert client.post(_url("unpaid-completed"), headers=fixture.headers).status_code == 402
        assert calls == []
        created = fixture.client.post("/api/v2/billing/checkout",
            headers={**fixture.headers, "Idempotency-Key": str(uuid.uuid4())},
            files={"file": ("source.txt", text.encode(), "text/plain")},
            data={"kind": "document", "profile": "full", "risk_accepted": "true",
                  "risk_notice_version": purchase_notice.VERSION, "risk_notice_sha256": purchase_notice.SHA256})
        assert created.status_code == 201, created.text
        oid = created.json()["order_id"]
        assert counts(fixture) == (1, 0, 0)
        assert signed(fixture, event_for(fixture, oid, paid=False)).status_code == 200
        assert counts(fixture) == (1, 0, 0)
        assert signed(fixture, event_for(fixture, oid, event_id="evt_archive_paid")).status_code == 200
        aid = fixture.client.get(f"/api/v2/billing/orders/{oid}", headers=fixture.headers).json()["analysis_id"]
        assert aid
        assert client.get(_url(aid), headers=fixture.headers).json() == {"available": False}
        assert client.post(_url(aid), headers=fixture.headers).status_code == 404
        assert calls == []
        with fixture.sessions() as db:
            analysis = db.get(Analysis, aid)
            analysis.status, analysis.result_hash = "done", "a" * 64
            db.commit()

        def state():
            with fixture.sessions() as db:
                return (db.execute(select(Analysis.id, Analysis.result_hash).order_by(Analysis.id)).all(),
                        db.execute(select(Artefact.id)).all(), db.execute(select(BillingOrderRow.id)).all(),
                        db.execute(select(Job.id)).all(), len(fixture.created))

        before = state()
        assert client.get(_url(aid), headers=fixture.headers).json() == {"available": True}
        assert client.post(_url(aid), headers=fixture.other_headers).status_code == 404
        assert client.post(_url(aid), headers={"X-API-Key": settings.api_key}).status_code == 404
        with fixture.sessions() as db:
            db.add(CaseMemberRow(case_id=db.get(Analysis, aid).case_id, principal_id=fixture.outsider.id, role="viewer",
                                 added_by=fixture.owner.id))
            db.commit()
        assert client.get(_url(aid), headers=fixture.other_headers).json() == {"available": False}
        assert client.post(_url(aid), headers=fixture.other_headers).status_code == 404
        assert calls == []
        response = client.post(_url(aid) + "?include_pdf=true", headers=fixture.headers)
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        result = body["comparison"]
        assert result["status"] == "reference_bound" and result["reference"]["id"] == SOURCE
        assert result["authorship_confidence_contribution"] == 0
        assert result["submitted_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert result["coordinate_space"] == "extracted_text_unicode_codepoint"
        assert result["library"]["sha256"] == document_archive.ARCHIVE_SHA256
        (change,) = result["changes"]
        assert change["archive_paragraph"] == 4 and change["submitted_paragraph"] == 1
        for key, expected in (("submitted", "op- posing"), ("archive", "opposing")):
            assert change[key]["text"] == expected
            assert change[key]["changed_tokens"]
        with pymupdf.open(stream=base64.b64decode(body["pdf"], validate=True), filetype="pdf") as pdf:
            assert pdf.metadata["author"] == "Chipmunk Verify"
            rendered = " ".join(" ".join(page.get_text().split()) for page in pdf)
            assert "Public archive comparison" in rendered and "Archive body paragraph 4" in rendered
            assert "op- posing" in rendered and "opposing" in rendered and result["submitted_sha256"] in rendered
        assert state() == before

        def withdraw(doc, source):
            result = compare(doc, source)
            with fixture.sessions() as db:
                db.get(Analysis, aid).withdrawn_at = datetime.now(UTC)
                db.commit()
            return result

        monkeypatch.setattr(document_archive, "compare", withdraw)
        refused = client.post(_url(aid), headers=fixture.headers)
        assert refused.status_code == 409
        assert "op- posing" not in refused.text and SOURCE not in refused.text


def test_archive_rejects_changed_original_bytes_and_invalid_catalog(billing, monkeypatch):
    fixture = billing
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    retained = _seed(fixture, "damaged", _body())
    _seed(fixture, "valid", _body())
    path = fixture.store.path_for(retained.sha256)
    raw = path.read_bytes()
    with _install(fixture, monkeypatch) as client:
        path.chmod(0o600)
        path.write_bytes(b"X" + raw[1:])
        assert client.post(_url("damaged"), headers=fixture.headers).status_code == 409
        path.write_bytes(raw)
        monkeypatch.setattr(document_archive, "library", lambda: (_ for _ in ()).throw(document_archive.ArchiveUnavailable()))
        assert client.post(_url("valid"), headers=fixture.headers).status_code == 503


def test_archive_preserves_boundaries_for_partial_or_reordered_excerpts():
    catalog, index = document_archive.library()
    first = next(row["paragraphs"] for row in catalog.values() if len(row["paragraphs"]) == 6)
    original = [row["text"] for row in first]
    for selected, status in ((original[:3], "reference_bound"), (original[::-1], "no_qualified_reference")):
        result = document_archive.compare(from_string("\n\n".join(selected)), "a" * 64, catalog, index)
        assert result["status"] == status and result["changes"] == []


def test_archive_recovers_collapsed_boundaries_and_exact_export_positions(billing, monkeypatch):
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    source = _body()
    edited = source.replace("October 11 resolution", "October 12 resolution", 1).replace("\n\n", " ")
    offset = edited.index("October 12 resolution") + len("October ")
    _seed(billing, "collapsed", edited)
    _seed(billing, "collapsed-intro", "Unrelated introduction. " + edited)
    with _install(billing, monkeypatch) as client:
        response = client.post(_url("collapsed") + "?include_pdf=true", headers=billing.headers)
        assert response.status_code == 200, response.text
        body = response.json()
        report = body["comparison"]
        assert report["status"] == "reference_bound" and report["layout"] == "recovered_archive_boundaries"
        assert report["reference"]["id"] == SOURCE
        assert report["coverage"]["body_paragraphs"] == 1
        assert [row["submitted_paragraph"] for row in report["aligned_paragraphs"]] == [1, 1, 1]
        (change,) = report["changes"]
        assert change["submitted"]["changed_tokens"] == [[offset, offset + 2]]
        assert change["submitted"]["text"] == edited[offset:offset + 2] == "12"
        assert change["archive"]["text"] == "11"
        with pymupdf.open(stream=base64.b64decode(body["pdf"], validate=True), filetype="pdf") as pdf:
            text = " ".join(" ".join(page.get_text().split()) for page in pdf)
            assert "recovered their boundaries from the archive" in text
            assert "Passage in submitted paragraph 1" in text and f"[{offset}, {offset + 2})" in text
        other = client.post(_url("collapsed-intro"), headers=billing.headers).json()["comparison"]
        assert other["reference"] is None and other["changes"] == []


def test_archive_recovers_single_edge_change_in_a_joined_copy(billing, monkeypatch):
    monkeypatch.setattr(settings, "billing_mode", "disabled")
    original = _body().replace("\n\n", " ")
    edited = original.replace("their pleas for clemency", "their requests for clemency", 1)
    offset = edited.index("their requests for clemency") + len("their ")
    _seed(billing, "single-edge", edited)
    _seed(billing, "two-edges", edited.replace("Strong reaction", "Marked reaction", 1))
    with _install(billing, monkeypatch) as client:
        response = client.post(_url("single-edge") + "?include_pdf=true", headers=billing.headers)
        assert response.status_code == 200
        body = response.json()
        report = body["comparison"]
        assert report["status"] == "reference_bound" and report["layout"] == "recovered_archive_boundaries"
        assert report["reference"]["id"] == SOURCE and report["authorship_confidence_contribution"] == 0
        (change,) = report["changes"]
        assert change["submitted"]["changed_tokens"] == [[offset, offset + 8]]
        assert change["archive"]["text"] == "pleas" and change["archive_paragraph"] == 4
        with pymupdf.open(stream=base64.b64decode(body["pdf"], validate=True), filetype="pdf") as pdf:
            text = " ".join(" ".join(page.get_text().split()) for page in pdf)
            assert f"[{offset}, {offset + 8})" in text and "requests" in text and "pleas" in text
        rejected = client.post(_url("two-edges"), headers=billing.headers).json()["comparison"]
        assert rejected["reference"] is None and rejected["changes"] == []
