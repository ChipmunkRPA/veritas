from __future__ import annotations

import io
from contextlib import contextmanager
from email.message import EmailMessage
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api import routes
from app.audit import chain
from app.batches.service import detect_kind
from app.cases import intake
from app.core.config import settings
from app.detectors.text.extract import extract
from app.storage import files
from app.storage.db import Analysis, Base, IntakeRecordRow, MediaObject
from app.storage.objects import ObjectStore


def message_bytes():
    mail = EmailMessage()
    mail["From"] = "unverified@example.test"
    mail["Subject"] = "Unicode email 中文 😀"
    mail["Authentication-Results"] = "untrusted.example; dkim=pass"
    mail.set_content("The selected email body has enough words for local analysis and retains its original MIME part.\n\n> Earlier quoted words belong to an earlier message.")
    mail.add_alternative("<p>Alternative HTML must not duplicate the message body.</p>", subtype="html")
    return mail.as_bytes()


@pytest.fixture
def eml_storage(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'intake.db'}")
    Base.metadata.create_all(engine)
    store = ObjectStore(tmp_path / "objects")

    @contextmanager
    def sessions():
        with Session(engine, expire_on_commit=False) as db:
            yield db

    monkeypatch.setattr(intake, "get_session", sessions)
    monkeypatch.setattr(chain, "get_session", sessions)
    monkeypatch.setattr(intake, "default_store", lambda: store)
    monkeypatch.setattr(intake.space, "low", lambda: None)
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(routes.jobs, "submit", lambda aid: None)
    return engine, store


def test_eml_quick_upload_and_case_intake_preserve_extension_mime_and_original(eml_storage):
    engine, store = eml_storage
    raw = message_bytes()
    assert files.validate_kind("document", raw[:8192], "mail.EML", len(raw)) == ("message/rfc822", ".eml")
    assert detect_kind(raw[:8192], "mail.eml") == "document"
    request = SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id="local-examiner")))
    analysis = routes._quick(request, io.BytesIO(raw), "mail.EML", "document")
    assert analysis.filename == "mail.EML" and analysis.content_type == "message/rfc822"
    assert store.read_bytes(analysis.sha256) == raw
    source = settings.uploads_dir / analysis.id / files.safe_filename(analysis.filename)
    extracted = extract(source, source.suffix)
    assert "selected email body" in extracted.text and "Alternative HTML" not in extracted.text
    assert extracted.meta["email"]["headers_authenticated"] is False
    again = intake.take_in(io.BytesIO(raw), "case-mail.eml", "document", principal_id="local-examiner",
                           case_id=analysis.case_id, analyze=False, store=store)
    assert again.case.id == analysis.case_id and not again.new_media
    with Session(engine) as db:
        assert len(db.scalars(select(Analysis)).all()) == 1
        assert len(db.scalars(select(MediaObject)).all()) == 1
        assert len(db.scalars(select(IntakeRecordRow)).all()) == 2


def test_eml_obeys_existing_document_upload_limit(eml_storage, monkeypatch):
    monkeypatch.setitem(intake.KIND_LIMITS, "document", 50)
    with pytest.raises(intake.IntakeError) as caught:
        intake.take_in(io.BytesIO(message_bytes()), "large.eml", "document", principal_id="local-examiner")
    assert caught.value.status == 413


def test_binary_disguised_as_eml_is_not_treated_as_email():
    with pytest.raises(ValueError, match="Unsupported document"):
        files.validate_kind("document", b"\x89PNG\r\n\x1a\n" + b"\x00" * 32, "image.eml")
