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
from app.detectors.image import (
    reference_exact_jpeg,
    reference_png_orientation,
    reference_same_jpeg,
    reference_shifted_jpeg,
    reference_structure_jpeg,
)
from app.main import app
from app.storage.db import Analysis, Artefact, Base, CaseMemberRow, CaseRow
from app.storage.objects import ObjectStore
from scripts.review_image_reference_export import review
from scripts.review_image_reference_oriented_crop_export import review as review_oriented_crop
from scripts.review_image_reference_oriented_resize_export import review as review_oriented_resize


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


def test_rotated_native_crop_survives_both_exports(reference_http, tmp_path):
    f = reference_http
    earlier = f.reference_pixels
    projected = np.rot90(earlier, -1)[29:341, 17:235].copy()
    submitted = projected.copy()
    submitted[80:126, 73:122] ^= 63
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
    assert comparison["schema"] == "chipmunk.image-reference-comparison/7"
    proof = comparison["exact_oriented_crop"]
    assert proof["transform"] == "rotate_90_clockwise"
    assert proof["projected_crop"] == {"x": 17, "y": 29, "w": 218, "h": 312}
    assert proof["native_crop"] == {"x": 29, "y": 21, "w": 312, "h": 218}
    assert proof["matched_pixels"] == 218 * 312 - 49 * 46
    [region] = comparison["regions"]
    assert region["reference_region"] == {"x": 109, "y": 117, "w": 46, "h": 49}
    with Image.open(io.BytesIO(_asset_bytes(payload, region["reference_crop"]))) as image:
        np.testing.assert_array_equal(np.asarray(image), earlier[117:166, 109:155])
    with pymupdf.open(stream=base64.b64decode(payload.pop("pdf")), filetype="pdf") as pdf:
        assert "Earlier copy overlap: x 29, y 21, 312 x 218 px" in pdf[0].get_text()
        _assert_pdf_assets(pdf, pdf[1], [region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")], payload)

    export, original, analyzed = (tmp_path / name for name in ("crop.json", "earlier.png", "submitted.png"))
    export.write_text(json.dumps(payload))
    original.write_bytes(f.reference)
    analyzed.write_bytes(content)
    result = review_oriented_crop(export, original, analyzed)
    assert result["mask_pixels_counted"] == 49 * 46 and result["crops_verified"] == 2
    assert review_oriented_resize(export, original, analyzed) == result
    proof["projected_crop"]["x"] += 1
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Exact oriented crop measurements differ"):
        review_oriented_crop(export, original, analyzed)


def test_rotated_resized_crop_keeps_both_native_grids(reference_http, tmp_path):
    f = reference_http
    earlier = f.reference_pixels
    with Image.fromarray(earlier).transpose(Image.Transpose.ROTATE_270) as oriented:
        with oriented.crop((17, 29, 235, 341)) as crop:
            projected = np.asarray(crop.resize((183, 262), Image.Resampling.BICUBIC))
    submitted = projected.copy()
    submitted[89:126, 64:102] ^= 63
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
    assert comparison["schema"] == "chipmunk.image-reference-comparison/8"
    proof = comparison["exact_oriented_resize"]
    assert proof["transform"] == "rotate_90_clockwise" and proof["kernels"] == ["bicubic"]
    assert proof["projected_crop"] == {"x": 17, "y": 29, "w": 218, "h": 312}
    assert proof["native_crop"] == {"x": 29, "y": 21, "w": 312, "h": 218}
    assert proof["matched_pixels"] == 183 * 262 - 38 * 37
    assert proof["projected_rgb_sha256"] == hashlib.sha256(projected.tobytes()).hexdigest()
    [region] = comparison["regions"]
    assert region["reference_region"] == {"x": 134, "y": 117, "w": 46, "h": 46}
    with pymupdf.open(stream=base64.b64decode(payload.pop("pdf")), filetype="pdf") as pdf:
        assert "90-degree clockwise rotation and resizing" in pdf[0].get_text()
        _assert_pdf_assets(pdf, pdf[1], [region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")], payload)

    export, original, analyzed = (tmp_path / name for name in ("resize.json", "earlier.png", "submitted.png"))
    export.write_text(json.dumps(payload))
    original.write_bytes(f.reference)
    analyzed.write_bytes(content)
    assert review_oriented_resize(export, original, analyzed)["mask_pixels_counted"] == 38 * 37
    proof["projected_crop"]["x"] += 1
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Earlier overlap does not match"):
        review_oriented_resize(export, original, analyzed)


def test_screenshot_photo_comparison_keeps_both_native_areas_and_exports(reference_http, tmp_path):
    f = reference_http
    earlier = f.reference_pixels
    photo = earlier[21:251, 19:372].copy()
    photo[68:141, :57] ^= 63
    submitted = np.full((440, 600, 3), 245, dtype=np.uint8)
    submitted[95:325, 105:458] = photo
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
    assert comparison["schema"] == "chipmunk.image-reference-comparison/9"
    proof = comparison["embedded_native_photo"]
    assert proof["reference_bounds"] == {"x": 19, "y": 21, "w": 353, "h": 230}
    assert proof["submitted_bounds"] == {"x": 105, "y": 95, "w": 353, "h": 230}
    assert proof["exact_pixels"] == 353 * 230 - 57 * 73
    [region] = comparison["regions"]
    assert region["reference_region"] == {"x": 19, "y": 89, "w": 57, "h": 73}
    assert (region["x"], region["y"], region["w"], region["h"]) == (105, 163, 57, 73)
    with pymupdf.open(stream=base64.b64decode(payload.pop("pdf")), filetype="pdf") as pdf:
        summary = pdf[0].get_text()
        assert "Screenshot photo area: x 105, y 95, 353 x 230 px" in summary
        assert "AI origin remains unassessed" in summary
        _assert_pdf_assets(pdf, pdf[0], [comparison["mask"]], payload)
        _assert_pdf_assets(pdf, pdf[1], [region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")], payload)

    export, original, analyzed = (tmp_path / name for name in ("screenshot.json", "earlier.png", "submitted.png"))
    export.write_text(json.dumps(payload))
    original.write_bytes(f.reference)
    analyzed.write_bytes(content)
    receipt = review_oriented_resize(export, original, analyzed)
    assert receipt["source_hashes_verified"] == 2 and receipt["crops_verified"] == 2
    assert receipt["mask_pixels_counted"] == 57 * 73
    proof["native_difference_sha256"] = "0" * 64
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Native inset pixel differences do not match"):
        review_oriented_resize(export, original, analyzed)


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


@pytest.mark.parametrize("format,quality,earlier_quality,schema", [
    ("JPEG", 75, None, "chipmunk.image-reference-comparison/4"),
    ("JPEG", 90, None, "chipmunk.image-reference-comparison/3"),
    ("JPEG", 90, 82, "chipmunk.image-reference-comparison/4"),
    ("JPEG", 90, 95, "chipmunk.image-reference-comparison/4"),
    ("JPEG", 90, 88, "chipmunk.image-reference-comparison/3"),
    ("JPEG", 87, 82, "chipmunk.image-reference-comparison/3"),
    ("PNG", 75, None, "chipmunk.image-reference-comparison/5"),
])
def test_crop_and_resize_fallback_requires_supported_delivery(reference_http, tmp_path, format, quality, earlier_quality, schema):
    f = reference_http
    reference = f.reference
    if earlier_quality is not None:
        data = io.BytesIO()
        Image.fromarray(f.reference_pixels).save(data, "JPEG", quality=earlier_quality, subsampling=2)
        reference = data.getvalue()
    with Image.open(io.BytesIO(reference)) as reference_image:
        edited = np.array(reference_image.convert("RGB"))
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
    response = f.client.post(_url(), headers=f.viewer, files=_upload(reference), data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    comparison = payload["comparison"]
    assert comparison["schema"] == schema
    if schema == "chipmunk.image-reference-comparison/3":
        assert comparison["status"] != "comparable" and payload["pngs"] == {}
        return
    assert comparison["status"] == "comparable"
    bounds = comparison["reference_crop_bounds"]
    assert [bounds[key] for key in ("x", "y", "w", "h")] == [crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]]
    region, = comparison["regions"]
    paths = [tmp_path / name for name in ("export.json", "earlier.png", "submitted." + format.lower())]
    export = {"comparison": comparison, "pngs": payload["pngs"]}
    paths[0].write_text(json.dumps(export))
    paths[1].write_bytes(reference)
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


def test_jpeg_png_envelope_is_opt_in_and_replays_native_evidence(reference_http, tmp_path, monkeypatch):
    f = reference_http
    earlier = io.BytesIO()
    Image.fromarray(f.reference_pixels).save(earlier, "JPEG", quality=95, subsampling=2)
    native = f.reference_pixels.copy()
    native[68:146, 130:220] = (210, 52, 119)
    crop = (24, 14, 352, 241)
    size = (282, 195)
    questioned = Image.fromarray(native).crop(crop).resize(size, Image.Resampling.BICUBIC)
    submitted = _png(np.asarray(questioned))
    source = f.store.put_bytes(submitted)
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-a")
        analysis.sha256, analysis.size_bytes = source.sha256, source.size_bytes
        db.commit()

    def compare():
        response = f.client.post(_url(), headers=f.viewer, files=_upload(earlier.getvalue()), data={"include_pdf": "true"})
        assert response.status_code == 200, response.text
        return response.json()

    monkeypatch.setattr(settings, "image_reference_jpeg_png_envelope", False)
    assert compare()["comparison"]["schema"] != "chipmunk.image-reference-comparison/10"
    monkeypatch.setattr(settings, "image_reference_jpeg_png_envelope", True)
    payload = compare()
    measured = payload["comparison"]
    assert measured["schema"] == "chipmunk.image-reference-comparison/10" and measured["status"] == "comparable"
    assert {key: measured["reference_crop_bounds"][key] for key in ("x", "y", "w", "h")} == {"x": 24, "y": 14, "w": 328, "h": 227}
    region, = measured["regions"]
    paths = [tmp_path / name for name in ("export.json", "earlier.jpg", "submitted.png")]
    export = {"comparison": measured, "pngs": payload["pngs"]}
    paths[0].write_text(json.dumps(export))
    paths[1].write_bytes(earlier.getvalue())
    paths[2].write_bytes(submitted)
    assert review(*paths)["crops_verified"] == 2
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as pdf:
        text = " ".join(pdf[0].get_text().split())
        assert "expected range for resize, tone and compression" in text
        assert "Ordinary processing can still produce differences." in text
        _assert_pdf_assets(pdf, pdf[1], tuple(region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")), payload)
    region["reference_region"]["x"] += 1
    paths[0].write_text(json.dumps(export))
    with pytest.raises(ValueError, match="Resize comparison differs from recomputed"):
        review(*paths)


def test_exact_jpeg_identity_replays_only_when_fallback_is_needed(reference_http, tmp_path, monkeypatch):
    f = reference_http
    # Smooth broadly distributed detail gives exact JPEG blocks but no distinctive ORB correspondence.
    y, x = np.indices((320, 384))
    red = 124 + 40 * np.sin((x + .34*y) / 30) + 30 * np.cos((y + .11*x) / 24)
    green = 116 + 36 * np.sin((x - .23*y) / 37) + 29 * np.sin((y + .13*x) / 28)
    blue = 119 + 40 * np.cos((y + .17*x) / 29) + 31 * np.sin((x + .21*y) / 30)
    original = np.rint(np.stack((red, green, blue), axis=-1)).astype(np.uint8)
    earlier = _png(original)
    altered = original.copy()
    patch = altered[164:196, 134:166]
    patch[:] = np.clip(patch.astype(np.int16) + [15, -8, 15], 0, 255)

    def question(pixels):
        with io.BytesIO() as output:
            Image.fromarray(pixels).save(output, "JPEG", quality=75, subsampling=2)
            body = output.getvalue()
        source = f.store.put_bytes(body)
        with Session(f.engine) as db:
            analysis = db.get(Analysis, "image-a")
            analysis.sha256, analysis.size_bytes = source.sha256, source.size_bytes
            db.commit()
        return body

    def compare():
        response = f.client.post(_url(), headers=f.viewer, files=_upload(earlier), data={"include_pdf": "true"})
        assert response.status_code == 200, response.text
        return response.json()

    submitted = question(altered)
    monkeypatch.setattr(settings, "image_reference_exact_jpeg_identity", False)
    off = compare()
    assert off["comparison"]["status"] == "integer_alignment_unresolved" and off["pngs"] == {}
    monkeypatch.setattr(settings, "image_reference_exact_jpeg_identity", True)
    payload = compare()
    comparison = payload["comparison"]
    assert comparison["schema"] == reference_exact_jpeg.SCHEMA and comparison["status"] == "comparable"
    proof = comparison["alignment"]
    assert proof["method"] == reference_exact_jpeg.METHOD and proof["reference_minus_submitted"] == {"x": 0, "y": 0}
    assert proof["anchor_count"] >= 24 and len(proof["anchor_sha256"]) == 64 and "anchors" not in proof
    paths = [tmp_path / name for name in ("export.json", "earlier.png", "submitted.jpg")]
    export = {"comparison": comparison, "pngs": payload["pngs"]}
    paths[0].write_text(json.dumps(export))
    paths[1].write_bytes(earlier)
    paths[2].write_bytes(submitted)
    assert review_oriented_resize(*paths)["crops_verified"] >= 2
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as pdf:
        text = pdf[0].get_text()
        assert "Same framing supported by exact JPEG block matches." in text
        assert f"{proof['exact_tiles']:,} of {proof['full_tiles']:,} full blocks matched" in text
        _assert_pdf_assets(pdf, pdf[0], (comparison["mask"],), payload)
        region = comparison["regions"][0]
        _assert_pdf_assets(pdf, pdf[1], tuple(region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")), payload)

    proof["anchor_sha256"] = "b" * 64
    paths[0].write_text(json.dumps(export))
    with pytest.raises(ValueError, match="Exact JPEG source proof"):
        review_oriented_resize(*paths)

    clean_bytes = question(original)
    clean = compare()
    assert clean["comparison"]["schema"] == reference_exact_jpeg.SCHEMA
    assert clean["comparison"]["status"] == "comparable" and clean["comparison"]["regions"] == [] and clean["pngs"] == {}
    assert clean["comparison"]["background_fraction"] == 1.
    clean["comparison"]["background_fraction"] = 1  # Browser JSON.stringify serializes integral measurements as integers.
    paths[0].write_text(json.dumps({key: clean[key] for key in ("comparison", "pngs")}))
    paths[2].write_bytes(clean_bytes)
    assert review_oriented_resize(*paths)["mask_pixels_counted"] == 0
    clean["comparison"]["background_fraction"] = True
    paths[0].write_text(json.dumps({key: clean[key] for key in ("comparison", "pngs")}))
    with pytest.raises(ValueError, match="Exact JPEG source proof"):
        review_oriented_resize(*paths)


def test_shifted_jpeg_evidence_keeps_only_the_certified_interior(reference_http, tmp_path, monkeypatch):
    f = reference_http
    original = f.reference_pixels
    before = f.reference
    changed = original.copy()
    changed[96:152, 116:181] = (213, 64, 149)
    dx, dy = 7, -11
    transform = np.array(((1, 0, dx), (0, 1, dy)), np.float32)

    def submit(pixels, *, border_value=245, interpolation=cv2.INTER_NEAREST, matrix=transform):
        moved = cv2.warpAffine(pixels, matrix, original.shape[1::-1], flags=interpolation,
                               borderValue=(border_value, border_value, border_value))
        with io.BytesIO() as output:
            Image.fromarray(moved).save(output, "JPEG", quality=75, subsampling=2)
            body = output.getvalue()
        item = f.store.put_bytes(body)
        with Session(f.engine) as db:
            analysis = db.get(Analysis, "image-a")
            analysis.sha256, analysis.size_bytes = item.sha256, item.size_bytes
            db.commit()
        return body

    def compare():
        response = f.client.post(_url(), headers=f.viewer, files=_upload(before), data={"include_pdf": "true"})
        assert response.status_code == 200, response.text
        return response.json()

    submitted = submit(changed)
    monkeypatch.setattr(settings, "image_reference_shifted_jpeg", False)
    assert compare()["comparison"]["status"] == "partial_jpeg_delivery_unassessed"
    monkeypatch.setattr(settings, "image_reference_shifted_jpeg", True)
    payload = compare()
    comparison = payload["comparison"]
    assert comparison["schema"] == reference_shifted_jpeg.SCHEMA and comparison["status"] == "comparable"
    assert comparison["affects_overall_confidence"] is False
    proof = comparison["alignment"]
    assert proof["method"] == reference_shifted_jpeg.METHOD
    assert proof["reference_minus_submitted"] == {"x": -dx, "y": -dy}
    assert proof["exact_tiles"] < proof["full_tiles"] and proof["anchor_count"] >= 24
    for region in comparison["regions"]:
        assert region["reference_region"] == {"x": region["x"]-dx, "y": region["y"]-dy, "w": region["w"], "h": region["h"]}
    paths = [tmp_path / name for name in ("export.json", "earlier.png", "submitted.jpg")]
    export = {"comparison": comparison, "pngs": payload["pngs"]}
    paths[0].write_text(json.dumps(export))
    paths[1].write_bytes(before)
    paths[2].write_bytes(submitted)
    assert review_oriented_resize(*paths)["crops_verified"] >= 2
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as pdf:
        words = " ".join(pdf[0].get_text().split())
        assert "Margins outside these bounds were unassessed." in words
        assert "complete JPEG blocks matched after whole-pixel alignment" in words
        _assert_pdf_assets(pdf, pdf[0], (comparison["mask"],), payload)
        region = comparison["regions"][0]
        _assert_pdf_assets(pdf, pdf[1], tuple(region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")), payload)
    proof["reference_minus_submitted"]["x"] += 1
    paths[0].write_text(json.dumps(export))
    with pytest.raises(ValueError, match="Exact JPEG source proof"):
        review_oriented_resize(*paths)

    submit(original, border_value=71)
    clean = compare()
    assert clean["comparison"]["schema"] == reference_shifted_jpeg.SCHEMA
    assert clean["comparison"]["regions"] == [] and clean["pngs"] == {}
    fractional = transform.copy()
    fractional[:, 2] += .5
    submit(original, interpolation=cv2.INTER_LINEAR, matrix=fractional)
    assert compare()["comparison"]["schema"] != reference_shifted_jpeg.SCHEMA


def test_tone_adjusted_jpeg_keeps_source_structure_evidence(reference_http, tmp_path, monkeypatch):
    f = reference_http
    tiles = np.random.default_rng(84).integers(10, 242, (55, 70, 3), dtype=np.uint8)
    original = cv2.resize(tiles, (768, 512), interpolation=cv2.INTER_CUBIC)
    changed = original.copy()
    changed[155:223, 219:306] = original[85:153, 355:442]

    def jpeg(pixels):
        with io.BytesIO() as output:
            Image.fromarray(pixels).save(output, "JPEG", quality=75, subsampling=2)
            return output.getvalue()

    def compare(pixels, *, grade=True):
        if grade:
            pixels = np.rint(255 * (.03 + .94 * (pixels.astype(np.float32) / 255)**.73)).astype(np.uint8)
        content = jpeg(pixels)
        item = f.store.put_bytes(content)
        with Session(f.engine) as db:
            analysis = db.get(Analysis, "image-a")
            analysis.sha256, analysis.size_bytes = item.sha256, item.size_bytes
            db.commit()
        response = f.client.post(_url(), headers=f.viewer, files={"reference": ("earlier.jpg", earlier, "image/jpeg")},
                                 data={"include_pdf": "true"})
        assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
        return response.json(), content

    earlier = jpeg(original)
    monkeypatch.setattr(settings, "image_reference_structure_jpeg", False)
    baseline = compare(changed)[0]["comparison"]
    assert baseline["status"] == "insufficient_common_background"
    monkeypatch.setattr(settings, "image_reference_structure_jpeg", True)
    payload, submitted = compare(changed)
    comparison = payload["comparison"]
    assert comparison["schema"] == reference_structure_jpeg.SCHEMA and comparison["status"] == "comparable"
    assert comparison["affects_overall_confidence"] is False
    [region] = comparison["regions"]
    assert region["reference_region"] == {key: region[key] for key in ("x", "y", "w", "h")}
    with Image.open(io.BytesIO(_asset_bytes(payload, comparison["mask"]))) as mask:
        assert np.mean(np.asarray(mask)[155:223, 219:306] > 0) > .95
        assert np.mean(np.asarray(mask) > 0) < .025
    with pymupdf.open(stream=base64.b64decode(payload.pop("pdf")), filetype="pdf") as pdf:
        assert "Highlights require local color and structure changes" in pdf[0].get_text()
        _assert_pdf_assets(pdf, pdf[0], [comparison["mask"]], payload)
        _assert_pdf_assets(pdf, pdf[1], [region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")], payload)

    paths = [tmp_path / name for name in ("comparison.json", "earlier.jpg", "submitted.jpg")]
    paths[0].write_text(json.dumps(payload))
    paths[1].write_bytes(earlier)
    paths[2].write_bytes(submitted)
    assert review_oriented_resize(*paths)["mask_pixels_counted"] == region["selected_pixels"]
    comparison["same_framing_structure"]["background_support"][0] = 1.
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="JPEG structure proof"):
        review_oriented_resize(*paths)
    clean, _ = compare(original)
    assert clean["comparison"]["regions"] == [] and clean["pngs"] == {}
    shift = cv2.warpAffine(changed, np.float32([[1, 0, 1.65], [0, 1, -1.65]]), (768, 512), borderMode=cv2.BORDER_REFLECT_101)
    assert compare(shift)[0]["comparison"]["schema"] != reference_structure_jpeg.SCHEMA
    with Session(f.engine) as db:
        assert db.get(Analysis, "image-a").result_hash == "a" * 64


@pytest.mark.parametrize("earlier_progressive, submitted_progressive", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_exact_jpeg_block_fallback_preserves_completed_analysis_and_exports(
    reference_http, tmp_path, monkeypatch, earlier_progressive, submitted_progressive,
):
    f = reference_http
    tiles = np.random.default_rng(661).integers(30, 220, (32, 48, 3), dtype=np.uint8)
    original = cv2.resize(tiles, (768, 512), interpolation=cv2.INTER_CUBIC)
    original[:176] = np.rint(115 + .09 * (original[:176].astype(float) - 115)).astype(np.uint8)
    changed = original.copy()
    changed[210:280, 349:404] = [204, 108, 77]

    def jpeg(pixels, quality=90, *, progressive=submitted_progressive):
        with io.BytesIO() as output:
            Image.fromarray(pixels).save(output, "JPEG", quality=quality, subsampling=2, progressive=progressive)
            return output.getvalue()

    def compare(content):
        stored = f.store.put_bytes(content)
        with Session(f.engine) as db:
            analysis = db.get(Analysis, "image-a")
            analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
            db.commit()
        response = f.client.post(_url(), headers=f.viewer, files={"reference": ("earlier.jpg", earlier, "image/jpeg")},
                                 data={"include_pdf": "true"})
        assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
        return response.json()

    earlier, submitted = jpeg(original, progressive=earlier_progressive), jpeg(changed)
    for name in ("image_reference_exact_jpeg_identity", "image_reference_shifted_jpeg", "image_reference_structure_jpeg"):
        monkeypatch.setattr(settings, name, True)
    monkeypatch.setattr(settings, "image_reference_same_jpeg", False)
    baseline = compare(submitted)["comparison"]
    assert baseline["status"] == "integer_alignment_unresolved" and baseline["regions"] == []
    monkeypatch.setattr(settings, "image_reference_same_jpeg", True)
    payload = compare(submitted)
    comparison = payload["comparison"]
    assert comparison["schema"] == reference_same_jpeg.SCHEMA and comparison["status"] == "comparable"
    assert comparison["affects_overall_confidence"] is False
    assert comparison["comparison_options"] == {"exact_jpeg_identity": True, "shifted_jpeg": True, "structure_jpeg": True}
    assert comparison["alignment"]["exact_tiles"] == 1490
    [region] = comparison["regions"]
    assert region["reference_region"] == {key: region[key] for key in ("x", "y", "w", "h")}
    with Image.open(io.BytesIO(_asset_bytes(payload, comparison["mask"]))) as mask:
        assert np.mean(np.asarray(mask)[210:280, 349:404] > 0) > .95
    with pymupdf.open(stream=base64.b64decode(payload.pop("pdf")), filetype="pdf") as pdf:
        assert "complete decoded blocks match exactly" in pdf[0].get_text()
        _assert_pdf_assets(pdf, pdf[0], [comparison["mask"]], payload)
        _assert_pdf_assets(pdf, pdf[1], [region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")], payload)
    exported, reference_path, submitted_path = [tmp_path / name for name in ("comparison.json", "earlier.jpg", "submitted.jpg")]
    reference_path.write_bytes(earlier)
    submitted_path.write_bytes(submitted)
    exported.write_text(json.dumps(payload))
    assert review(exported, reference_path, submitted_path)["mask_pixels_counted"] == region["selected_pixels"]
    comparison["alignment"]["exact_tiles"] += 1
    exported.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Exact JPEG source proof"):
        review(exported, reference_path, submitted_path)
    comparison["alignment"]["exact_tiles"] -= 1
    comparison["comparison_options"]["structure_jpeg"] = "false"
    exported.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid recorded comparison options"):
        review(exported, reference_path, submitted_path)
    assert compare(earlier)["comparison"]["regions"] == []
    assert compare(jpeg(changed, 85))["comparison"]["schema"] != reference_same_jpeg.SCHEMA
    assert compare(jpeg(np.roll(changed, 16, axis=1)))["comparison"]["schema"] != reference_same_jpeg.SCHEMA
    if submitted_progressive:
        assert compare(submitted + b"trailing data")["comparison"]["schema"] != reference_same_jpeg.SCHEMA
    monkeypatch.setattr(settings, "image_reference_same_jpeg", False)
    assert compare(submitted)["comparison"] == baseline
    with Session(f.engine) as db:
        assert db.get(Analysis, "image-a").result_hash == "a" * 64


@pytest.mark.parametrize("quality", (75, 85, 90))
def test_jpeg_detail_loss_keeps_earlier_copy_crops_and_replays(reference_http, tmp_path, monkeypatch, quality):
    rng = np.random.default_rng(560)
    width, height = 624, 528
    y, x = np.indices((height, width))
    background = cv2.resize(rng.integers(0, 255, (18, 26)).astype(np.float32), (width, height),
                            interpolation=cv2.INTER_CUBIC)
    detail = 7 * np.sin(1.55 * x + .20 * y) + 5 * np.cos(1.5 * y) + 6 * np.sin(1.3 * (x + y))
    detail += 3 * rng.normal(size=detail.shape)
    rgb = np.stack([89 + background * .25, 83 + background * .32, 113 + background * .30], axis=-1)
    source = np.rint(rgb + detail[..., None]).astype(np.uint8)
    softened = cv2.GaussianBlur(source, (0, 0), 1.35)
    modified = source.copy()
    modified[224:304, 288:384] = softened[224:304, 288:384]

    def encode(pixels, quality=90):
        with io.BytesIO() as buffer:
            Image.fromarray(pixels).save(buffer, "JPEG", quality=quality, subsampling=2)
            return buffer.getvalue()

    before = encode(source, quality)
    after = encode(modified, quality)
    f = reference_http

    def compare(payload):
        stored = f.store.put_bytes(payload)
        with Session(f.engine) as db:
            analysis = db.get(Analysis, "image-a")
            analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
            db.commit()
        response = f.client.post(_url(), headers=f.viewer, files={"reference": ("before.jpg", before, "image/jpeg")},
                                 data={"include_pdf": "true"})
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        return response.json()

    for flag in ("image_reference_exact_jpeg_identity", "image_reference_shifted_jpeg",
                 "image_reference_structure_jpeg", "image_reference_same_jpeg"):
        monkeypatch.setattr(settings, flag, True)
    monkeypatch.setattr(settings, "image_reference_jpeg_detail_loss", False)
    baseline = compare(after)["comparison"]
    assert baseline["regions"] == []
    monkeypatch.setattr(settings, "image_reference_jpeg_detail_loss", True)
    extra_quality = quality != 90
    monkeypatch.setattr(settings, "image_reference_jpeg_detail_loss_extra_qualities", False)
    if extra_quality:
        assert compare(after)["comparison"] == baseline
        monkeypatch.setattr(settings, "image_reference_jpeg_detail_loss_extra_qualities", True)
    payload = compare(after)
    result = payload["comparison"]
    expected_schema = "chipmunk.image-reference-comparison/" + ("16" if extra_quality else "15")
    assert result["schema"] == expected_schema
    assert result["alignment"].get("jpeg_quality", 90) == quality
    assert result["affects_overall_confidence"] is False
    [region] = result["regions"]
    bounds = region["reference_region"]
    assert 288 <= bounds["x"] < bounds["x"] + bounds["w"] <= 384
    assert 224 <= bounds["y"] < bounds["y"] + bounds["h"] <= 304
    assert 0 < region["directly_measured_tiles"] <= region["selected_pixels"] // 256
    assert result["comparison_options"]["same_jpeg"] is True
    with pymupdf.open(stream=base64.b64decode(payload.pop("pdf")), filetype="pdf") as pdf:
        assert "Fine luminance detail is weaker" in pdf[0].get_text()
        assert "directly support the finding" in pdf[1].get_text()
        _assert_pdf_assets(pdf, pdf[0], [result["mask"]], payload)
        _assert_pdf_assets(pdf, pdf[1], [region[key] for key in ("reference_crop", "submitted_crop", "submitted_overlay")], payload)
    manifest, reference_path, submitted_path = (tmp_path / name for name in ("comparison.json", "earlier.jpg", "submitted.jpg"))
    reference_path.write_bytes(before)
    submitted_path.write_bytes(after)
    manifest.write_text(json.dumps(payload))
    assert review(manifest, reference_path, submitted_path)["mask_pixels_counted"] == region["selected_pixels"]
    region["directly_measured_tiles"] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differ from recomputed evidence"):
        review(manifest, reference_path, submitted_path)
    assert compare(encode(softened, quality))["comparison"]["schema"] != expected_schema
    mismatched_quality = (85 if quality == 75 else 75)
    assert compare(encode(modified, mismatched_quality))["comparison"]["schema"] not in {
        "chipmunk.image-reference-comparison/15", "chipmunk.image-reference-comparison/16",
    }
    with Session(f.engine) as db:
        assert db.get(Analysis, "image-a").result_hash == "a" * 64
