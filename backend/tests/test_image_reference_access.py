"""Exercise the earlier-copy endpoint with production authentication, upload limits and storage checks."""
from __future__ import annotations

import base64
import hashlib
import io
from contextlib import contextmanager
from types import SimpleNamespace

import cv2
import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api import auth, security, v2_image_reference
from app.core.config import settings
from app.main import app
from app.storage.db import Analysis, Artefact, Base, CaseMemberRow, CaseRow
from app.storage.objects import ObjectStore


def _png(pixels):
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def reference_http(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'reference.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    store = ObjectStore(tmp_path / "objects")
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_reference_comparison_enabled", True)
    monkeypatch.setattr(settings, "single_user", False)
    monkeypatch.setattr(settings, "api_key", "test-bootstrap-not-used")
    monkeypatch.setattr(settings, "api_key_pepper", "isolated-test-pepper")
    monkeypatch.setattr(settings, "billing_mode", "test")

    @contextmanager
    def sessions():
        with Session(engine) as db:
            yield db

    for module in (auth, security, v2_image_reference):
        monkeypatch.setattr(module, "get_session", sessions)
    monkeypatch.setattr(v2_image_reference, "default_store", lambda: store)
    monkeypatch.setattr(security, "limiter", security.RateLimiter(settings.rate_limit_per_minute))
    monkeypatch.setattr(security, "credential_limiter", security.RateLimiter(settings.credential_failures_per_minute))

    rng = np.random.default_rng(741)
    detail = rng.integers(65, 175, (18, 26, 3), dtype=np.uint8)
    base = cv2.resize(detail, (384, 256), interpolation=cv2.INTER_CUBIC)
    texture = rng.integers(-12, 13, base.shape, dtype=np.int16)
    earlier = np.clip(base.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    later = earlier[11:241, 16:372].copy()
    later[65:142, 118:207] = (210, 52, 119)
    original = store.put_bytes(_png(later))
    with Session(engine) as db:
        viewer = auth.new_principal("reference-viewer", "viewer")
        outsider = auth.new_principal("reference-outsider", "examiner")
        db.add_all([viewer, outsider])
        db.flush()
        viewer_key, _ = auth.issue_api_key(db, viewer)
        outsider_key, _ = auth.issue_api_key(db, outsider)
        session_value, session = auth.create_session(db, viewer, None)
        csrf = session.csrf_secret
        db.add(CaseRow(id="reference-case", reference="ref", title="reference", classification="internal",
                       status="open", created_by=outsider.id))
        db.flush()
        db.add(CaseMemberRow(case_id="reference-case", principal_id=viewer.id, role="viewer", added_by=outsider.id))
        db.add_all(Analysis(id=aid, kind=kind, filename="submitted.png", content_type="image/png",
                            case_id="reference-case", status=status, sha256=original.sha256,
                            size_bytes=original.size_bytes, result_hash="a" * 64)
                   for aid, kind, status in [("image-a", "image", "done"), ("pending", "image", "queued"),
                                             ("other-kind", "video", "done")])
        db.commit()

    # The process lifespan is unnecessary because all touched infrastructure is isolated above.
    client = TestClient(app)
    yield SimpleNamespace(client=client, viewer={"X-API-Key": viewer_key}, outsider={"X-API-Key": outsider_key},
                          engine=engine, store=store, original=original, reference=_png(earlier), reference_pixels=earlier,
                          submitted_pixels=later, session=session_value, csrf=csrf)
    client.close()
    engine.dispose()


def _url(aid="image-a"):
    return f"/api/v2/analyses/{aid}/image-reference-comparison"


def _upload(reference):
    return {"reference": ("earlier.png", reference, "image/png")}


def test_gate_requires_opt_in_completed_image_and_retained_original(reference_http, monkeypatch):
    f = reference_http
    response = f.client.get(_url(), headers=f.viewer)
    assert response.json() == {"available": True}
    assert response.headers["cache-control"] == "no-store"
    for aid in ("pending", "other-kind"):
        assert f.client.get(_url(aid), headers=f.viewer).json() == {"available": False}
        assert f.client.post(_url(aid), headers=f.viewer, files=_upload(f.reference)).status_code == 404

    for setting, value in [("image_reference_comparison_enabled", False), ("environment", "prod")]:
        with monkeypatch.context() as m:
            m.setattr(settings, setting, value)
            assert f.client.get(_url(), headers=f.viewer).json() == {"available": False}
            assert f.client.post(_url(), headers=f.viewer, files=_upload(f.reference)).status_code == 404

    f.original.path.unlink()
    assert f.client.get(_url(), headers=f.viewer).json() == {"available": False}
    assert f.client.post(_url(), headers=f.viewer, files=_upload(f.reference)).status_code == 404


def test_case_access_and_cookie_csrf_are_required(reference_http):
    f = reference_http
    assert f.client.get(_url()).status_code == 401
    assert f.client.post(_url(), files=_upload(f.reference)).status_code == 401
    for response in (f.client.get(_url(), headers=f.outsider),
                     f.client.post(_url(), headers=f.outsider, files=_upload(f.reference))):
        assert response.status_code == 404
        assert response.json()["title"] == "Analysis not found"

    f.client.cookies.set(auth.SESSION_COOKIE, f.session)
    assert f.client.get(_url()).status_code == 200
    assert f.client.post(_url(), files=_upload(f.reference)).status_code == 403
    assert f.client.post(_url(), headers={auth.CSRF_HEADER: f.csrf}, files=_upload(f.reference)).status_code == 200


def test_comparison_binds_hashes_and_native_crops_without_persistence(reference_http):
    f = reference_http

    def snapshot():
        with Session(f.engine) as db:
            return (db.execute(select(Analysis.id, Analysis.result_hash).order_by(Analysis.id)).all(),
                    db.execute(select(Artefact.id)).all(), sorted(path for path in f.store.root.rglob("*") if path.is_file()))

    before = snapshot()
    response = f.client.post(_url(), headers=f.viewer, files=_upload(f.reference))
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert set(payload) == {"comparison", "pngs"}
    comparison = payload["comparison"]
    assert comparison["reference"]["sha256"] == hashlib.sha256(f.reference).hexdigest()
    assert comparison["submitted"]["sha256"] == f.original.sha256
    assert comparison["coordinate_space"] == "exif-oriented-submitted"
    assert comparison["alignment"]["reference_minus_submitted"] == {"x": 16, "y": 11}
    region, = comparison["regions"]
    assert set(payload["pngs"]) == {comparison["mask"]["file"], region["reference_crop"]["file"], region["submitted_crop"]["file"]}
    for label, pixels, bounds in (("reference", f.reference_pixels, region["reference_region"]),
                                   ("submitted", f.submitted_pixels, region)):
        info = region[f"{label}_crop"]
        data = base64.b64decode(payload["pngs"][info["file"]], validate=True)
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
        with Image.open(io.BytesIO(data)) as crop:
            assert crop.size == (w, h)
            np.testing.assert_array_equal(np.asarray(crop), pixels[y:y+h, x:x+w])
    assert snapshot() == before


def test_tampered_original_and_unsupported_uploads_are_refused(reference_http, monkeypatch):
    f = reference_http
    invalid = f.client.post(_url(), headers=f.viewer, files=_upload(b"not an image"))
    assert invalid.status_code == 422
    assert invalid.headers["cache-control"] == "no-store"
    oversized = _png(np.zeros((2001, 2000), dtype=np.uint8))
    assert f.client.post(_url(), headers=f.viewer, files=_upload(oversized)).status_code == 413

    with monkeypatch.context() as m:
        m.setattr(settings, "max_image_bytes", f.original.size_bytes)
        for data in (f.reference, bytes(f.original.size_bytes + 1024 * 1024)):
            too_large = f.client.post(_url(), headers=f.viewer, files=_upload(data))
            assert too_large.status_code == 413
            assert too_large.headers["cache-control"] == "no-store"
            assert too_large.headers["content-type"] == "application/problem+json"

    f.original.path.chmod(0o644)
    original = f.original.path.read_bytes()
    f.original.path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    tampered = f.client.post(_url(), headers=f.viewer, files=_upload(f.reference))
    assert tampered.status_code == 409
    assert tampered.json()["title"] == "Original integrity check failed"
    assert str(f.original.path) not in tampered.text


def test_optional_pdf_uses_the_same_manifest_and_pngs(reference_http, monkeypatch):
    f = reference_http
    response = f.client.post(_url(), headers=f.viewer, files=_upload(f.reference), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    comparison = payload["comparison"]
    region, = comparison["regions"]
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as pdf:
        assert len(pdf) == 2
        assert pdf.metadata["author"] == "Chipmunk Verify"
        assert comparison["reference"]["sha256"] in pdf[0].get_text()
        assert comparison["submitted"]["sha256"] in pdf[0].get_text()
        assert comparison["mask"]["sha256"] in pdf[0].get_text()
        for index, asset_names in enumerate(((comparison["mask"],), (region["reference_crop"], region["submitted_crop"]))):
            exhibits = pdf[index].get_images()
            assert len(exhibits) == len(asset_names)
            for exhibit, asset in zip(exhibits, asset_names, strict=True):
                source = base64.b64decode(payload["pngs"][asset["file"]], validate=True)
                with Image.open(io.BytesIO(source)) as png:
                    assert pymupdf.Pixmap(pdf, exhibit[0]).samples == png.tobytes()
                if index == 1:
                    assert asset["sha256"] in pdf[index].get_text()

    from app.reports import image_reference_pdf

    def unavailable(*_args):
        raise RuntimeError("rendering unavailable")

    monkeypatch.setattr(image_reference_pdf, "render", unavailable)
    fallback = f.client.post(_url(), headers=f.viewer, files=_upload(f.reference), data={"include_pdf": "true"})
    assert fallback.status_code == 200
    assert fallback.json() == {**payload, "pdf": None}
