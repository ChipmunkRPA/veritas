"""Exercise the earlier-copy endpoint with production authentication, upload limits and storage checks."""
from __future__ import annotations

import base64
import hashlib
import io
import json
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
from app.detectors.image import reference_png_orientation
from app.main import app
from app.storage.db import Analysis, Artefact, Base, CaseMemberRow, CaseRow
from app.storage.objects import ObjectStore
from scripts.review_image_reference_export import review


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


def _asset_bytes(payload, asset):
    return base64.b64decode(payload["pngs"][asset["file"]], validate=True)


def _assert_pdf_assets(pdf, page, assets, payload):
    exhibits = list(dict.fromkeys((item[0], item[1]) for item in page.get_images()))
    for (xref, alpha_xref), asset in zip(exhibits, assets, strict=True):
        with Image.open(io.BytesIO(_asset_bytes(payload, asset))) as png:
            pixmap = pymupdf.Pixmap(pdf, xref)
            if png.mode == "RGBA":
                assert alpha_xref
                assert pymupdf.Pixmap(pdf, alpha_xref).samples == png.getchannel("A").tobytes()
                pixels = np.asarray(png)
                rgba = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(png.height, png.width, 3)
                selected = pixels[:, :, 3] > 0
                np.testing.assert_allclose(rgba[selected], pixels[:, :, :3][selected], atol=1)
                assert page.get_image_rects(xref)[0] in page.get_image_rects(exhibits[1][0])
            else:
                assert pixmap.samples == png.tobytes()
        assert asset["sha256"] in page.get_text()


def test_rotated_png_keeps_exact_native_pixels_in_both_exports(reference_http, tmp_path):
    f = reference_http
    earlier = f.reference_pixels
    projected = np.rot90(earlier, -1).copy()
    submitted = projected.copy()
    submitted[77:137, 84:124] ^= 63
    content = _png(submitted)
    stored = f.store.put_bytes(content)
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-a")
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        db.commit()

    reply = f.client.post(_url(), files=_upload(f.reference), headers=f.viewer, data={"include_pdf": "true"})
    assert reply.status_code == 200 and reply.headers["Cache-Control"] == "no-store"
    payload = reply.json()
    comparison = payload["comparison"]
    assert comparison["schema"] == "chipmunk.image-reference-comparison/6"
    proof = comparison["exact_orientation"]
    assert proof["transform"] == "rotate_90_clockwise"
    assert proof["matched_pixels"] == submitted.shape[0] * submitted.shape[1] - 60 * 40
    assert proof["rgb_sha256"] == hashlib.sha256(projected.tobytes()).hexdigest()
    [region] = comparison["regions"]
    assert region["reference_region"] == {"x": 77, "y": 132, "w": 60, "h": 40}
    with Image.open(io.BytesIO(_asset_bytes(payload, region["reference_crop"]))) as image:
        np.testing.assert_array_equal(np.asarray(image), earlier[132:172, 77:137])
    with Image.open(io.BytesIO(_asset_bytes(payload, region["submitted_overlay"]))) as image:
        assert image.getchannel("A").histogram()[144] == 60 * 40

    with pymupdf.open(stream=base64.b64decode(payload.pop("pdf")), filetype="pdf") as pdf:
        assert "90-degree clockwise rotation" in pdf[0].get_text()
        _assert_pdf_assets(pdf, pdf[1], [region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")], payload)

    export, original, analyzed = (tmp_path / name for name in ("export.json", "original.png", "submitted.png"))
    export.write_text(json.dumps(payload))
    original.write_bytes(f.reference)
    analyzed.write_bytes(content)
    assert review(export, original, analyzed)["mask_pixels_counted"] == 60 * 40
    proof["transform"] = "mirror_horizontal"
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Exact orientation or textured match differs"):
        review(export, original, analyzed)


def test_rotated_png_with_two_exact_alignments_abstains():
    rng = np.random.default_rng(63)
    half = rng.integers(0, 256, (240, 180, 3), dtype=np.uint8)
    symmetric = np.concatenate((half, half[:, ::-1]), axis=1)
    rotated = np.rot90(symmetric, -1)
    with Image.open(io.BytesIO(_png(rotated))) as carrier:
        assert reference_png_orientation.compare(symmetric, rotated, carrier) == (None, None)


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
    assert set(payload["pngs"]) == {comparison["mask"]["file"], region["reference_crop"]["file"],
                                    region["submitted_crop"]["file"], region["submitted_overlay"]["file"]}
    for label, pixels, bounds in (("reference", f.reference_pixels, region["reference_region"]),
                                   ("submitted", f.submitted_pixels, region)):
        info = region[f"{label}_crop"]
        data = base64.b64decode(payload["pngs"][info["file"]], validate=True)
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
        with Image.open(io.BytesIO(data)) as crop:
            assert crop.size == (w, h)
            np.testing.assert_array_equal(np.asarray(crop), pixels[y:y+h, x:x+w])
    x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
    with Image.open(io.BytesIO(_asset_bytes(payload, comparison["mask"]))) as mask, \
            Image.open(io.BytesIO(_asset_bytes(payload, region["submitted_overlay"]))) as overlay:
        np.testing.assert_array_equal(np.asarray(overlay.getchannel("A")) == 144, np.asarray(mask)[y:y+h, x:x+w] == 255)
        assert hashlib.sha256(_asset_bytes(payload, region["submitted_overlay"])).hexdigest() == region["submitted_overlay"]["sha256"]
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
        assert "/regions/0/submitted_overlay" in pdf[1].get_text()
        _assert_pdf_assets(pdf, pdf[0], (comparison["mask"],), payload)
        _assert_pdf_assets(pdf, pdf[1], (region["reference_crop"], region["submitted_crop"], region["submitted_overlay"]), payload)

    from app.reports import image_reference_pdf

    def unavailable(*_args):
        raise RuntimeError("rendering unavailable")

    monkeypatch.setattr(image_reference_pdf, "render", unavailable)
    fallback = f.client.post(_url(), headers=f.viewer, files=_upload(f.reference), data={"include_pdf": "true"})
    assert fallback.status_code == 200
    assert fallback.json() == {**payload, "pdf": None}


@pytest.mark.parametrize("format,scale", [("PNG", .82), ("JPEG", 1.17)])
def test_resized_route_pdf_and_export_bind_both_native_grids(reference_http, tmp_path, format, scale):
    f = reference_http
    edited = f.reference_pixels.copy()
    edited[70:142, 142:222] = (210, 52, 119)
    size = tuple(round(side * scale) for side in (384, 256))
    data = io.BytesIO()
    image = Image.fromarray(edited).resize(size, Image.Resampling.BICUBIC)
    if format == "JPEG":
        exif = Image.Exif()
        exif[274] = 6
        image.transpose(Image.Transpose.ROTATE_90).save(data, format, quality=75, subsampling=2, exif=exif)
    else:
        image.save(data, format)
    submitted_bytes = data.getvalue()
    source = f.store.put_bytes(submitted_bytes)
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-a")
        analysis.sha256, analysis.size_bytes = source.sha256, source.size_bytes
        db.commit()
    response = f.client.post(_url(), headers=f.viewer, files=_upload(f.reference), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    comparison = payload["comparison"]
    assert comparison["schema"] == "chipmunk.image-reference-comparison/3" and comparison["status"] == "comparable"
    assert comparison["submitted_over_reference_xy"] == [[size[0], 384], [size[1], 256]]
    assert comparison["framing_check"]["reference_minus_submitted"] == {"x": 0, "y": 0}
    assert comparison["mask"]["width"] == size[0] and comparison["mask"]["height"] == size[1]
    assert len(comparison["supported_resize_kernels"]) >= 2
    region, = comparison["regions"]
    assert region["reference_crop"]["width"] != region["submitted_crop"]["width"]
    paths = [tmp_path / filename for filename in ("export.json", "earlier.png", "submitted")]
    export = {"comparison": comparison, "pngs": payload["pngs"]}
    paths[0].write_text(json.dumps(export))
    paths[1].write_bytes(f.reference)
    paths[2].write_bytes(submitted_bytes)
    assert review(*paths) == {"reported_status": "comparable", "source_hashes_verified": 2,
                              "crops_verified": 2, "mask_pixels_counted": region["selected_pixels"]}
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as pdf:
        assert len(pdf) == 2 and "Same framing at different pixel dimensions" in pdf[0].get_text()
        _assert_pdf_assets(pdf, pdf[0], (comparison["mask"],), payload)
        _assert_pdf_assets(pdf, pdf[1], (region["reference_crop"], region["submitted_crop"], region["submitted_overlay"]), payload)

    mask = comparison["mask"]
    with Image.open(io.BytesIO(base64.b64decode(payload["pngs"][mask["file"]]))) as image:
        image.putpixel((0, 0), 255)
        changed = io.BytesIO()
        image.save(changed, format="PNG")
    mask["sha256"] = hashlib.sha256(changed.getvalue()).hexdigest()
    payload["pngs"][mask["file"]] = base64.b64encode(changed.getvalue()).decode("ascii")
    paths[0].write_text(json.dumps(export))
    with pytest.raises(ValueError, match="mask pixels differ from the recomputed resize comparison"):
        review(*paths)


@pytest.mark.parametrize("format,quality,schema", [
    ("JPEG", 75, "chipmunk.image-reference-comparison/4"),
    ("JPEG", 90, "chipmunk.image-reference-comparison/3"),
    ("PNG", 75, "chipmunk.image-reference-comparison/5"),
])
def test_crop_and_resize_fallback_requires_supported_delivery(reference_http, tmp_path, format, quality, schema):
    f = reference_http
    edited = f.reference_pixels.copy()
    edited[68:146, 130:220] = (210, 52, 119)
    crop = (24, 14, 352, 241)
    size = tuple(round(side * .86) for side in (crop[2] - crop[0], crop[3] - crop[1]))
    data = io.BytesIO()
    image = Image.fromarray(edited).crop(crop).resize(size, Image.Resampling.BICUBIC)
    image.save(data, format, quality=quality, subsampling=2)
    submitted = data.getvalue()
    source = f.store.put_bytes(submitted)
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-a")
        analysis.sha256, analysis.size_bytes = source.sha256, source.size_bytes
        db.commit()
    response = f.client.post(_url(), headers=f.viewer, files=_upload(f.reference), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    comparison = payload["comparison"]
    assert comparison["schema"] == schema
    if format == "JPEG" and quality != 75:
        assert comparison["status"] != "comparable" and payload["pngs"] == {}
        return
    assert comparison["status"] == "comparable"
    bounds = comparison["reference_crop_bounds"]
    assert [bounds[key] for key in ("x", "y", "w", "h")] == [crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]]
    region, = comparison["regions"]
    paths = [tmp_path / name for name in ("export.json", "earlier.png", "submitted." + format.lower())]
    export = {"comparison": comparison, "pngs": payload["pngs"]}
    paths[0].write_text(json.dumps(export))
    paths[1].write_bytes(f.reference)
    paths[2].write_bytes(submitted)
    assert review(*paths)["crops_verified"] == 2
    if format == "PNG":
        projection = np.asarray(Image.fromarray(f.reference_pixels).crop(crop).resize(size, Image.Resampling.BICUBIC))
        exact = comparison["exact_projection"]
        assert exact["rgb_sha256"] == hashlib.sha256(projection.tobytes()).hexdigest()
        assert exact["matched_pixels"] == int(np.count_nonzero(np.all(projection == np.asarray(image), axis=2)))
        assert comparison["supported_resize_kernels"] == ["bicubic"] and len(exact["matched_tiles"]) >= 12
        exact["matched_pixels"] -= 1
        paths[0].write_text(json.dumps(export))
        with pytest.raises(ValueError, match="Resize comparison differs from recomputed"):
            review(*paths)
        exact["matched_pixels"] += 1
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as pdf:
        summary = pdf[0].get_text()
        assert "Earlier copy area used: x 24, y 14, 328 x 227 px" in summary
        if format == "PNG":
            assert "Exact match on" in summary and "textured tiles (bicubic)" in summary
            assert "at least two tested resizes" not in summary
        _assert_pdf_assets(pdf, pdf[1], (region["reference_crop"], region["submitted_crop"], region["submitted_overlay"]), payload)
    bounds["x"] += 1
    paths[0].write_text(json.dumps(export))
    with pytest.raises(ValueError, match="Resize comparison differs from recomputed"):
        review(*paths)
