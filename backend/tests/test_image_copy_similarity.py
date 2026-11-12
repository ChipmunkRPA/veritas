"""Exercise automatic mirrored pairing through the normal frozen evidence path."""

from __future__ import annotations

import copy
import io

import cv2
import numpy as np
import pymupdf
import pytest
from PIL import Image
from sqlalchemy.orm import Session
from test_image_evidence_access import evidence_http as evidence_http
from test_image_packets import _bytes
from test_image_packets import packet_fixture as packet_fixture

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.image import copy_move, copy_similarity
from app.evidence import image_packets_v5, image_packets_v7, service
from app.evidence.image_copy_similarity_binding import V7, V8, V9, V10, V11, VERSION, supported_size
from app.reports import engine, image_evidence, snapshot, verify_image_source
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis


def authored_source(*, reflected: bool) -> Image.Image:
    rng = np.random.default_rng(812)
    pixels = rng.integers(0, 256, (512, 768, 3), dtype=np.uint8)
    pixels = cv2.GaussianBlur(pixels, (3, 3), 0.65)
    if reflected:
        pixels[61:189, 31:159] = pixels[301:429, 527:655][:, ::-1]
    return Image.fromarray(pixels)


def low_texture_source(*, reflected: bool) -> Image.Image:
    rng = np.random.default_rng(917)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (400, 600)).astype(np.float32), (15, 15), 1.3)
    gray = np.rint(128 + noise * 8 / noise.std()).astype(np.uint8)
    if reflected:
        gray[80:176, :96] = gray[271:367, 491:587][:, ::-1]
    return Image.fromarray(gray).convert("RGB")


def large_source(*, reflected: bool) -> Image.Image:
    rng = np.random.default_rng(8812)
    pixels = rng.integers(0, 256, (896, 1408, 3), dtype=np.uint8)
    pixels = cv2.GaussianBlur(pixels, (3, 3), 0.65)
    if reflected:
        pixels[117:245, :128] = pixels[525:653, 1093:1221][:, ::-1]
    return Image.fromarray(pixels)


def reflected_result(packet_fixture, *, reflected=True, low_texture=False, large=False, translated=False):
    make_source = low_texture_source if low_texture else authored_source
    if large:
        make_source = large_source
    source = make_source(reflected=reflected)
    if translated:
        source.paste(source.crop((527, 301, 655, 429)), (31, 61))
    return result_from_source(packet_fixture, source)


def affine_result(packet_fixture):
    rng = np.random.default_rng(142)
    scene = cv2.resize(rng.integers(0, 256, (192, 192, 3), dtype=np.uint8), (768, 768), interpolation=cv2.INTER_CUBIC)
    matrix = cv2.getRotationMatrix2D((168, 200), 14, .83)
    matrix[:, 2] += [408, 380]
    alpha = np.zeros(scene.shape[:2], np.uint8)
    cv2.ellipse(alpha, (168, 200), (76, 62), 0, 0, 360, 1, thickness=-1)
    target = cv2.warpAffine(alpha, matrix, (768, 768), flags=cv2.INTER_NEAREST).astype(bool)
    warped = cv2.warpAffine(scene, matrix, (768, 768), flags=cv2.INTER_CUBIC)
    scene[target] = warped[target]
    return result_from_source(packet_fixture, Image.fromarray(scene))


def result_from_source(packet_fixture, source):
    db, store, analysis, _, _ = packet_fixture
    obj = store.put_bytes(_bytes(source, format="PNG"))
    analysis.sha256, analysis.size_bytes = obj.sha256, obj.size_bytes
    result = copy.deepcopy(specimen("image", "standard").result)
    result.update(exhibits=[], signals=[], findings=[])
    result["input"]["sha256"] = obj.sha256
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    return db, store, analysis, result, source


@pytest.mark.parametrize(("width", "height", "old", "latest"), [
    (512, 512, False, False),
    (513, 384, True, True),
    (1024, 384, True, True),
    (1025, 767, False, False),
    (1025, 768, False, True),
    (768, 2048, False, True),
    (2048, 2048, False, True),
    (2049, 2048, False, False),
])
def test_large_geometry_gate_preserves_old_scope(width, height, old, latest):
    for version in (V7, V8):
        assert supported_size(width, height, version=version) is old
    assert supported_size(width, height) is latest


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_lower_texture_pass_finds_an_edge_pair_after_the_first_pass(image_format):
    def roundtrip(reflected):
        data = _bytes(low_texture_source(reflected=reflected), format=image_format, quality=90)
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image)

    rgb = roundtrip(True)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    assert copy_similarity._search(rgb, gray, copy_similarity.PLAN) == ("no_pair", [])
    status, pairs = copy_similarity.analyse(rgb)
    assert status == "supported_pair"
    (first, second), = pairs
    assert copy_move.iou(first, (0, 80, 96, 96)) >= 0.8
    assert copy_move.iou(second, (491, 271, 96, 96)) >= 0.8
    assert copy_similarity.analyse(roundtrip(False)) == ("no_pair", [])


def test_automatic_pair_persists_both_original_crops_without_manual_highlight(packet_fixture, monkeypatch):
    db, store, analysis, result, source = reflected_result(packet_fixture)
    old = image_packets_v5.persist_packet_set(db, analysis, store, result=result)
    document = service.persist_packet_set(db, analysis, store, result=result)
    assert document["version"] == image_packets_v7.VERSION
    assert document["prior_image_evidence"] == {k: old[k] for k in ("id", "version", "sha256")}
    assert document["packets"] == old["packets"] == []
    checked = document["copy_similarity"]
    assert checked["status"] == "supported_pair"
    assert checked["authorship_confidence_contribution"] == 0
    (pair,) = checked["pairs"]
    assert pair["transform"] == "horizontal_reflection"
    assert pair["regions"] == {
        "area_a": {"x": 31, "y": 61, "w": 128, "h": 128},
        "area_b": {"x": 527, "y": 301, "w": 128, "h": 128},
    }
    for role, asset in pair["artifacts"].items():
        assert service.asset_refs(document)[asset["sha256"]] == asset
        with Image.open(io.BytesIO(store.read_bytes(asset["sha256"]))) as crop:
            box = image_packets_v7.box_bounds(pair["regions"][role], *source.size)
            assert crop.tobytes() == source.crop(box).tobytes()

    monkeypatch.setattr(
        copy_similarity, "analyse", lambda _: pytest.fail("A read must not rerun the detector")
    )
    assert service.load_packet_set(db, analysis, store) == document
    assert service.persist_packet_set(db, analysis, store, result=result) == document
    assert service.load_packet_set(db, analysis, store, set_id=old["id"]) == old

    wrong = copy.deepcopy(document)
    wrong["copy_similarity"]["pairs"][0]["regions"]["area_b"]["x"] += 1
    wrong["sha256"] = image_packets_v5.packet_hash(wrong)
    with pytest.raises(image_packets_v7.PacketIntegrityError, match="crop pixels differ"):
        image_packets_v7.validate_assets(wrong, store)

    wrong = copy.deepcopy(document)
    wrong["limitations"].append("Altered prior evidence")
    wrong["sha256"] = image_packets_v5.packet_hash(wrong)
    with pytest.raises(image_packets_v7.PacketIntegrityError, match="changes the frozen prior"):
        image_packets_v7.validate_packet_set(wrong, **image_packets_v5._binding(analysis))


@pytest.mark.parametrize(("version", "creator", "large"), [
    (V7, "image-evidence-packets/5", False),
    (V8, "image-evidence-packets/6", True),
    (V9, "image-evidence-packets/7", True),
    (V10, "image-evidence-packets/8", False),
    (V11, "image-evidence-packets/9", False),
])
def test_historical_reflection_remains_readable_after_writing_latest(packet_fixture, monkeypatch, version, creator, large):
    db, store, analysis, result, _ = reflected_result(packet_fixture, large=large)
    historical = service.build_packet_set(db, analysis, store, result)
    if version == V10:
        historical["copy_translation"].pop("basis")
    elif version != V11:
        historical.pop("copy_translation")
    historical["version"] = version
    historical["id"] = image_packets_v5.current_set_id(analysis, version)
    if large and version == V8:
        historical["copy_similarity"].update(status="unsupported_geometry", pairs=[])
    for pair in historical["copy_similarity"]["pairs"]:
        for asset in pair["artifacts"].values():
            asset["url"] = image_packets_v7.url(historical, asset["sha256"])
    historical["sha256"] = image_packets_v5.packet_hash(historical)
    historical_row = register_artefact(
        db, store, canonical_json(historical), media_type="application/json", role="exhibit_payload",
        created_by=creator, analysis_id=analysis.id, artefact_id=historical["id"],
    )
    newest = db.get(type(historical_row), service.current_set_id(analysis))
    db.delete(newest)
    db.flush()
    with monkeypatch.context() as context:
        context.setattr(copy_similarity, "analyse", lambda _: pytest.fail("A historical read must not rerun the detector"))
        assert service.load_packet_set(db, analysis, store) == historical
    latest = service.persist_packet_set(db, analysis, store, result=result)
    assert latest["version"] == VERSION and latest["id"] != historical["id"]
    assert latest["copy_similarity"]["status"] == "supported_pair"
    assert service.load_packet_set(db, analysis, store) == latest
    assert service.load_packet_set(db, analysis, store, set_id=historical["id"]) == historical
    assert store.read_bytes(historical_row.sha256) == canonical_json(historical)


def test_no_pair_keeps_explicit_coverage_and_no_assets(packet_fixture):
    db, store, analysis, result, _ = reflected_result(packet_fixture, reflected=False)
    document = service.persist_packet_set(db, analysis, store, result=result)
    assert document["copy_similarity"]["status"] == "no_pair"
    assert document["copy_similarity"]["pairs"] == []
    assert document["copy_translation"]["status"] == "no_pair"
    assert service.asset_refs(document) == {}


def test_translated_pair_budget_is_disclosed_as_unassessed(packet_fixture, monkeypatch):
    db, store, analysis, result, _ = reflected_result(packet_fixture, reflected=False, translated=True)
    monkeypatch.setattr(copy_similarity, "analyse_translation", lambda _: ("no_pair", []))
    monkeypatch.setattr(image_packets_v7, "MAX_COPY_CANDIDATES", 1)
    document = service.persist_packet_set(db, analysis, store, result=result)
    assert document["copy_translation"]["status"] == "unassessed_match_budget"
    assert document["copy_translation"]["basis"] is None
    assert document["copy_translation"]["pairs"] == []
    assert service.asset_refs(document) == {}


@pytest.mark.parametrize("large", [False, True])
def test_authenticated_image_api_serves_both_pair_crops(evidence_http, large):
    fixture = evidence_http
    analysis_id = "image-unprepared"
    with Session(fixture.engine) as db:
        analysis = db.get(Analysis, analysis_id)
        db, store, analysis, result, source = reflected_result(
            (db, fixture.store, analysis, None, None), low_texture=not large, large=large,
        )
        expected = service.persist_packet_set(db, analysis, store, result=result)
        db.commit()
    response = fixture.client.get(f"/api/v2/analyses/{analysis_id}/evidence-packets", headers=fixture.viewer)
    assert response.status_code == 200 and response.json() == expected
    (pair,) = expected["copy_similarity"]["pairs"]
    assert pair["regions"]["area_a"]["x"] == 0
    for role, asset in pair["artifacts"].items():
        assert fixture.client.get(asset["url"], headers=fixture.outsider).status_code == 404
        plate = fixture.client.get(asset["url"], headers=fixture.viewer)
        assert plate.status_code == 200
        assert plate.content == fixture.store.read_bytes(asset["sha256"])
        with Image.open(io.BytesIO(plate.content)) as crop:
            bounds = image_packets_v7.box_bounds(pair["regions"][role], *source.size)
            assert crop.tobytes() == source.crop(bounds).tobytes()


def test_translated_pair_is_saved_without_changing_assessment_and_serves_both_source_crops(evidence_http):
    fixture = evidence_http
    analysis_id = "image-unprepared"
    with Session(fixture.engine) as db:
        analysis = db.get(Analysis, analysis_id)
        db, store, analysis, result, source = reflected_result((db, fixture.store, analysis, None, None),
                                                               reflected=False, translated=True)
        previous = store.read_bytes(analysis.result_hash)
        document = service.persist_packet_set(db, analysis, store, result=result)
        assert store.read_bytes(analysis.result_hash) == previous
        assert document["original_result_hash"] == analysis.result_hash
        check = document["copy_translation"]
        assert check["status"] == "supported_pair" and check["authorship_confidence_contribution"] == 0
        assert check["basis"] == "native_patch_alignment"
        assert document["copy_similarity"]["status"] == "no_pair"
        pair, = check["pairs"]
        assert pair["id"] == "T01" and pair["transform"] == "same_orientation"
        wrong = copy.deepcopy(document)
        wrong["copy_translation"]["pairs"][0]["transform"] = "horizontal_reflection"
        wrong["sha256"] = image_packets_v5.packet_hash(wrong)
        with pytest.raises(image_packets_v7.PacketIntegrityError, match="malformed paired areas"):
            image_packets_v7.validate_packet_set(wrong, **image_packets_v5._binding(analysis))
        wrong = copy.deepcopy(document)
        wrong["copy_translation"]["basis"] = None
        wrong["sha256"] = image_packets_v5.packet_hash(wrong)
        with pytest.raises(image_packets_v7.PacketIntegrityError, match="invalid matching basis"):
            image_packets_v7.validate_packet_set(wrong, **image_packets_v5._binding(analysis))
        db.commit()
    response = fixture.client.get(f"/api/v2/analyses/{analysis_id}/evidence-packets", headers=fixture.viewer)
    assert response.status_code == 200 and response.json() == document
    for (role, asset), truth in zip(pair["artifacts"].items(), ((31, 61, 128, 128), (527, 301, 128, 128)), strict=True):
        region = pair["regions"][role]
        assert copy_move.iou(tuple(region[k] for k in "xywh"), truth) > 0.8
        assert fixture.client.get(asset["url"], headers=fixture.outsider).status_code == 404
        plate = fixture.client.get(asset["url"], headers=fixture.viewer)
        assert plate.status_code == 200
        with Image.open(io.BytesIO(plate.content)) as crop:
            assert crop.tobytes() == source.crop(image_packets_v7.box_bounds(region, *source.size)).tobytes()


def test_affine_pair_api_and_pdf_preserve_the_two_native_crops_and_binary_masks(evidence_http, monkeypatch):
    fixture = evidence_http
    analysis_id = "image-unprepared"
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_affine_copy_review", True)
    with Session(fixture.engine) as db:
        analysis = db.get(Analysis, analysis_id)
        db, store, analysis, result, source = affine_result((db, fixture.store, analysis, None, None))
        document = service.persist_packet_set(db, analysis, store, result=result)
        assert document["version"] == "image-evidence-13"
        assert document["original_result_hash"] == canonical_hash(result)
        check = document["copy_affine"]
        assert check["status"] == "supported_pair" and check["authorship_confidence_contribution"] == 0
        pair, = check["pairs"]
        report = specimen("image", "standard")
        frozen = report.model_copy(update={"result": result, "image_evidence": document,
            "analysis": report.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
            "media": report.media.model_copy(update={"sha256": analysis.sha256})})
        db.commit()
    response = fixture.client.get(f"/api/v2/analyses/{analysis_id}/evidence-packets", headers=fixture.viewer)
    assert response.status_code == 200 and response.json() == document
    oversized = copy.deepcopy(document)
    oversized["copy_affine"]["pairs"][0]["regions"]["area_a"] = {"x": 0, "y": 0, "w": source.width, "h": source.height}
    with pytest.raises(ValueError, match="exact source bounds"):
        verify_image_source._verify_geometric_masks(None, oversized, source)
    expected_rgb = set()
    for role, asset in pair["artifacts"].items():
        assert fixture.client.get(asset["url"], headers=fixture.outsider).status_code == 404
        response = fixture.client.get(asset["url"], headers=fixture.viewer)
        assert response.status_code == 200 and response.content == store.read_bytes(asset["sha256"])
        with Image.open(io.BytesIO(response.content)) as plate:
            area = "area_" + role[-1]
            bounds = image_packets_v7.box_bounds(pair["regions"][area], *source.size)
            if role.startswith("mask_"):
                values = plate.histogram()
                assert plate.mode == "L" and values[255] == pair["selected_pixels"][area]
                assert values[0] + values[255] == plate.width * plate.height
            else:
                assert plate.tobytes() == source.crop(bounds).tobytes()
            expected_rgb.add(plate.convert("RGB").tobytes())
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)
    rendered = engine.render_pdf(frozen).pdf
    assert "Geometric copy comparison" in engine.pdf_text(rendered)
    assert "Area A selected mask" in engine.pdf_text(rendered)
    assert "Area B mapped mask" in engine.pdf_text(rendered)
    with pymupdf.open(stream=rendered, filetype="pdf") as pdf:
        images = {pymupdf.Pixmap(pdf, row[0]).samples for page in pdf for row in page.get_images()}
        # PDF can store a binary mask in DeviceGray while retaining all native values.
        for page in pdf:
            for row in page.get_images():
                pixmap = pymupdf.Pixmap(pdf, row[0])
                if pixmap.n == 1:
                    images.add(pymupdf.Pixmap(pymupdf.csRGB, pixmap).samples)
        assert expected_rgb <= images


@pytest.mark.parametrize("large", [False, True])
def test_report_and_json_share_source_bound_pair_and_redaction(packet_fixture, monkeypatch, large):
    db, store, analysis, result, _ = reflected_result(packet_fixture, large=large)
    document = service.persist_packet_set(db, analysis, store, result=result)
    original = specimen("image", "standard")
    frozen = original.model_copy(
        update={
            "result": result,
            "image_evidence": document,
            "analysis": original.analysis.model_copy(
                update={"id": analysis.id, "revision": analysis.revision}
            ),
            "media": original.media.model_copy(update={"sha256": analysis.sha256}),
        }
    )
    assert snapshot.load(snapshot.dump(frozen)).image_evidence == document
    monkeypatch.setattr(image_evidence, "default_store", lambda: store)
    view = image_evidence.view(frozen)
    (rendered,) = view["copy_similarity"]["pairs"]
    assert [image["label"] for image in rendered["images"]] == ["Area A", "Area B"]
    assert all(image["uri"].startswith("data:image/png;base64,") for image in rendered["images"])
    manifest = image_evidence.manifest(frozen)
    assert manifest["copy_similarity"] == document["copy_similarity"]
    assert {a["role"] for a in manifest["artifacts"]} == {"area_a", "area_b"}
    html = engine.render_html(frozen, embed_fonts=False)
    x, y = (1093, 525) if large else (527, 301)
    assert "Same-image copy similarity" in html and f"Area B · x {x}, y {y}, 128 × 128 px" in html
    assert "which area came first" in html
    if large:
        rendered_pdf = engine.render_pdf(frozen)
        pdf_text = " ".join(engine.pdf_text(rendered_pdf.pdf).split())
        assert f"Area B · x {x}, y {y}, 128 × 128 px" in pdf_text
        expected_pixels = set()
        for asset in document["copy_similarity"]["pairs"][0]["artifacts"].values():
            with Image.open(io.BytesIO(store.read_bytes(asset["sha256"]))) as crop:
                expected_pixels.add(crop.convert("RGB").tobytes())
        with pymupdf.open(stream=rendered_pdf.pdf, filetype="pdf") as pdf:
            actual_pixels = set()
            for page in pdf:
                for xref, _, width, height, *_ in page.get_images():
                    if (width, height) == (128, 128):
                        pixmap = pymupdf.Pixmap(pdf, xref)
                        actual_pixels.add(pixmap.samples)
            assert expected_pixels <= actual_pixels
    redacted = frozen.model_copy(update={"report": frozen.report.model_copy(update={"redacted": True})})
    assert all(
        image["uri"] is None
        for image in image_evidence.view(redacted)["copy_similarity"]["pairs"][0]["images"]
    )
