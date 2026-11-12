from __future__ import annotations

import base64
import hashlib
from contextlib import contextmanager
from datetime import UTC, datetime
from email.message import EmailMessage
from tempfile import TemporaryDirectory

import pymupdf
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_image_reference_access import reference_http as reference_http

from app.api import auth, v2_document_revision
from app.core.config import settings
from app.evidence import document_revision
from app.reports.document_revision_pdf import _side
from app.storage.db import Analysis, Artefact, BillingOrderRow


def _email(text):
    message = EmailMessage()
    message["From"] = "fixture@example.invalid"
    message.set_content(text, cte="quoted-printable")
    message.add_alternative("<p>Excluded HTML body</p>", subtype="html")
    message.add_attachment(b"Excluded attachment body", maintype="application", subtype="octet-stream", filename="fixture.bin")
    return message.as_bytes()


def _install(fixture, monkeypatch):
    @contextmanager
    def sessions():
        with Session(fixture.engine) as db:
            yield db

    monkeypatch.setattr(v2_document_revision, "get_session", sessions)
    monkeypatch.setattr(v2_document_revision, "default_store", lambda: fixture.store)
    monkeypatch.setattr(settings, "document_revision_comparison_enabled", True)
    return fixture


def _saved(f, aid, extension, data, *, status="done"):
    retained = f.store.put_bytes(data)
    with Session(f.engine) as db:
        db.add(Analysis(id=aid, kind="document", filename="later" + extension, content_type="text/plain", case_id="reference-case",
                        status=status, sha256=retained.sha256, size_bytes=retained.size_bytes, result_hash="f" * 64))
        db.commit()
    return retained


def _url(aid):
    return f"/api/v2/analyses/{aid}/document-revision-comparison"


def _upload(data, extension=".eml"):
    return {"earlier": ("earlier" + extension, data, "application/octet-stream")}


def test_email_comparison_preserves_roles_and_exact_original_excerpts_in_json_and_pdf(reference_http, monkeypatch, tmp_path):
    f = _install(reference_http, monkeypatch)
    previous = "📎 Send the crème  brûlée on Tuesday.\n\n-- \nOld signature\n\n> Review the earlier draft."
    latest = "📎 Send the café  order on Thursday.\n\n-- \nNew signature\n\n> Review the final draft."
    earlier = _email(previous)
    retained = _saved(f, "email-draft", ".eml", _email(latest))
    staged = tmp_path / "staged"
    staged.mkdir()
    monkeypatch.setattr(document_revision, "TemporaryDirectory", lambda **kw: TemporaryDirectory(dir=staged, **kw))

    def state():
        with Session(f.engine) as db:
            return (db.execute(select(Analysis.id, Analysis.result_hash).order_by(Analysis.id)).all(),
                    db.execute(select(Artefact.id)).all(), db.execute(select(BillingOrderRow.id)).all(),
                    sorted(path for path in f.store.root.rglob("*") if path.is_file()))

    before = state()
    assert f.client.get(_url("email-draft"), headers=f.viewer).json() == {"available": True, "extension": ".eml"}
    response = f.client.post(_url("email-draft"), headers=f.viewer, files=_upload(earlier), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    result = response.json()
    comparison = result["comparison"]
    assert comparison["coordinate_space"] == "decoded_source_unicode_codepoint"
    assert comparison["earlier"]["sha256"] == hashlib.sha256(earlier).hexdigest()
    assert comparison["later"]["sha256"] == retained.sha256
    roles = {change["role"] for change in comparison["changes"]}
    assert roles == {"body", "quoted", "signature"}
    assert "Excluded" not in response.text
    for change in comparison["changes"]:
        for side, text in (("earlier", previous), ("later", latest)):
            excerpt = change[side]
            assert excerpt["text"] == text[excerpt["start"]:excerpt["end"]]
            assert all(excerpt["start"] <= a < b <= excerpt["end"] for a, b in excerpt["changed_tokens"])
    assert any("crème  brûlée" in change["earlier"]["text"] for change in comparison["changes"])
    with pymupdf.open(stream=base64.b64decode(result["pdf"], validate=True), filetype="pdf") as pdf:
        body = " ".join(" ".join(page.get_text().split()) for page in pdf)
        for part in ("crème brûlée", "café order", "Quoted text", "Signature", retained.sha256):
            assert part in body
        assert pdf.metadata["author"] == "Chipmunk Verify"
    assert not list(staged.iterdir()) and state() == before

    mislabeled = f.client.post(_url("email-draft"), headers=f.viewer, files=_upload(b"This is plain text with no email headers."))
    assert mislabeled.status_code == 422 and not list(staged.iterdir())


def test_auth_scope_format_mismatch_and_retained_later_integrity(reference_http, monkeypatch):
    f = _install(reference_http, monkeypatch)
    retained = _saved(f, "text-draft", ".txt", "🙂 Crème brûlée.".encode())
    _saved(f, "pending-draft", ".txt", b"Queued draft", status="queued")
    _saved(f, "pdf-draft", ".pdf", b"Outside the draft comparison")
    url = _url("text-draft")
    earlier = _upload("🚀 Déjà vu.".encode(), ".txt")
    assert f.client.post(url, files=earlier).status_code == 401
    assert f.client.post(url, headers=f.outsider, files=earlier).status_code == 404
    assert f.client.post(_url("pending-draft"), headers=f.viewer, files=earlier).status_code == 404
    assert f.client.get(_url("pdf-draft"), headers=f.viewer).json()["available"] is False
    assert f.client.post(url, headers=f.viewer, files=_upload(_email("A different file type"))).status_code == 422
    assert f.client.post(url, headers=f.viewer, files=_upload(b"x" * (document_revision.MAX_BYTES + 1), ".txt")).status_code == 413
    for setting, value in (("environment", "prod"), ("document_revision_comparison_enabled", False)):
        with monkeypatch.context() as patch:
            patch.setattr(settings, setting, value)
            assert not f.client.get(url, headers=f.viewer).json()["available"]
            assert f.client.post(url, headers=f.viewer, files=earlier).status_code == 404

    f.client.cookies.set(auth.SESSION_COOKIE, f.session)
    assert f.client.post(url, files=earlier).status_code == 403
    named = f.client.post(url, headers={auth.CSRF_HEADER: f.csrf}, files=earlier)
    assert named.status_code == 200, named.text
    assert named.json()["comparison"]["changes"][0]["earlier"]["text"] == "🚀 Déjà vu"

    retained.path.chmod(0o644)
    retained.path.write_bytes(retained.path.read_bytes()[:-1] + b"!")
    assert f.client.post(url, headers=f.viewer, files=earlier).status_code == 409
    retained.path.unlink()
    assert not f.client.get(url, headers=f.viewer).json()["available"]
    assert f.client.post(url, headers=f.viewer, files=earlier).status_code == 404


def test_refuses_an_attachment_selected_as_the_email_body(reference_http, monkeypatch):
    f = _install(reference_http, monkeypatch)
    ambiguous = EmailMessage()
    ambiguous["From"] = "fixture@example.invalid"
    ambiguous.make_mixed()
    named = EmailMessage()
    named.set_content("Private attachment contents")
    named.set_param("name", "private.txt", header="Content-Type")
    body = EmailMessage()
    body.set_content("This is the actual message body.")
    ambiguous.attach(named)
    ambiguous.attach(body)
    assert ambiguous.get_body(preferencelist=("plain",)).get_filename() == "private.txt"
    _saved(f, "ambiguous-later", ".eml", ambiguous.as_bytes())
    _saved(f, "plain-later", ".eml", _email("The later message body."))
    for aid, content in (("ambiguous-later", _email("An earlier message body.")), ("plain-later", ambiguous.as_bytes())):
        response = f.client.post(_url(aid), headers=f.viewer, files=_upload(content), data={"include_pdf": "true"})
        assert response.status_code == 422 and "is an attachment" in response.text
        assert "Private attachment contents" not in response.text


def test_inactive_revisions_are_hidden_and_refused(reference_http, monkeypatch):
    f = _install(reference_http, monkeypatch)
    _saved(f, "withdrawn-draft", ".txt", b"This was withdrawn")
    _saved(f, "old-draft", ".txt", b"This was superseded")
    _saved(f, "new-draft", ".txt", b"This is the active draft")
    with Session(f.engine) as db:
        db.get(Analysis, "withdrawn-draft").withdrawn_at = datetime.now(UTC)
        db.get(Analysis, "new-draft").supersedes_id = "old-draft"
        db.commit()
    for aid, reason in (("withdrawn-draft", "Revision withdrawn"), ("old-draft", "Revision superseded")):
        assert not f.client.get(_url(aid), headers=f.viewer).json()["available"]
        response = f.client.post(_url(aid), headers=f.viewer, files=_upload(b"The earlier wording", ".txt"))
        assert response.status_code == 409 and reason in response.text
    assert f.client.get(_url("new-draft"), headers=f.viewer).json()["available"]


def test_pdf_marks_only_changed_code_points_and_escapes_excerpt():
    result = document_revision.compare("🚀 Déjà <end>".encode(), "🙂 Déjà <début>".encode(), ".txt")
    earlier, later = result["changes"][0]["earlier"], result["changes"][0]["later"]
    assert "<mark>🚀</mark> Déjà &lt;<mark>end</mark>" in _side("Earlier draft", earlier)
    assert "<mark>🙂</mark> Déjà &lt;<mark>début</mark>" in _side("Analyzed version", later)
