from __future__ import annotations

import base64
import hashlib
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import select
from test_stripe_billing import billing as billing
from test_stripe_billing import counts, event_for, signed

from app.api import v2_document_archive
from app.billing import purchase_notice
from app.core.config import settings
from app.detectors.text.extract import from_string
from app.evidence import document_archive
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
    retained = fixture.store.put_bytes(text.encode())
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


def test_real_paid_checkout_and_retained_archive_evidence(billing, monkeypatch):
    fixture = billing
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
