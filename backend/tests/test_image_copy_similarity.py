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
from app.evidence import (
    image_jpeg_copy_blocks_source,
    image_jpeg_diagonal_source,
    image_jpeg_offgrid_source,
    image_packets_v5,
    image_packets_v7,
    service,
)
from app.evidence.image_copy_similarity_binding import (
    JPEG_DIAGONAL_Q90_VERSION,
    JPEG_DIAGONAL_Q92_VERSION,
    JPEG_DIAGONAL_VERSION,
    V7,
    V8,
    V9,
    V10,
    V11,
    VERSION,
    supported_size,
)
from app.evidence.image_embedded_pair_binding import base_document
from app.reports import engine, image_evidence, snapshot, verify_image_source
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis


def test_native_rotation_can_keep_two_disjoint_pairs_and_declines_shared_or_third_pairs():
    from app.detectors.image.jpeg_native_turned_copy import select
    from app.evidence.image_copy_similarity_binding import _validate_turned_jpeg

    witness = {"interior_mean": 1., "interior_p90": 2., "perimeter_means": [10., 11., 12., 13.]}
    a = ((32, 32, 48, 48), (160, 32, 48, 48))
    b = ((32, 160, 48, 48), (160, 160, 48, 48))
    match_a, match_b = (5, 100, a, 1, witness), (4, 90, b, 2, witness)
    assert select([match_a, match_b]) == ("ambiguous", [])
    status, pairs = select([match_a, match_b], allow_disjoint=True)
    assert status == "supported_pair" and [item["turns"] for item in pairs] == [1, 2]
    shared = (4, 90, (a[0], b[1]), 2, witness)
    near = (4, 90, ((95, 32, 48, 48), b[1]), 2, witness)
    third = (3, 80, ((320, 32, 48, 48), (480, 32, 48, 48)), 1, witness)
    assert select([match_a, shared], allow_disjoint=True) == ("ambiguous", [])
    assert select([match_a, near], allow_disjoint=True) == ("ambiguous", [])
    assert select([match_a, match_b, third], allow_disjoint=True) == ("ambiguous", [])
    nested = {"status": "supported_pair", "measurement": None, "measurements": [witness, witness]}
    assert _validate_turned_jpeg(nested, None, expanded=True, native=True, allow_disjoint=True) == 2
    with pytest.raises(ValueError, match="two measured pairs"):
        _validate_turned_jpeg(nested, None, expanded=True, native=True, allow_disjoint=False)


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


def affine_result(packet_fixture, *, large=False):
    rng = np.random.default_rng(142)
    scale = 2 if large else 1
    size = 768 * scale
    center = (168 * scale, 200 * scale)
    scene = cv2.resize(rng.integers(0, 256, (192, 192, 3), dtype=np.uint8), (size, size), interpolation=cv2.INTER_CUBIC)
    matrix = cv2.getRotationMatrix2D(center, 14, .83)
    matrix[:, 2] += [408 * scale, 380 * scale]
    alpha = np.zeros(scene.shape[:2], np.uint8)
    cv2.ellipse(alpha, center, (76 * scale, 62 * scale), 0, 0, 360, 1, thickness=-1)
    target = cv2.warpAffine(alpha, matrix, (size, size), flags=cv2.INTER_NEAREST).astype(bool)
    warped = cv2.warpAffine(scene, matrix, (size, size), flags=cv2.INTER_CUBIC)
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
    assert image_packets_v7.affine.supported(width, height) is old
    assert image_packets_v7.affine.supported(width, height, version=image_packets_v7.affine.LARGE_VERSION) is latest


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
    # Build the unwrapped base packet before reconstructing the known historical shape.
    monkeypatch.setattr(settings, "environment", "prod")
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


@pytest.mark.parametrize("large", [False, True])
def test_affine_pair_api_and_pdf_preserve_the_two_native_crops_and_binary_masks(evidence_http, monkeypatch, large):
    fixture = evidence_http
    analysis_id = "image-unprepared"
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_affine_copy_review", True)
    with Session(fixture.engine) as db:
        analysis = db.get(Analysis, analysis_id)
        db, store, analysis, result, source = affine_result((db, fixture.store, analysis, None, None), large=large)
        document = service.persist_packet_set(db, analysis, store, result=result)
        assert document["version"] == "image-evidence-14"
        assert document["original_result_hash"] == canonical_hash(result)
        check = document["copy_affine"]
        assert check["status"] == "supported_pair" and check["authorship_confidence_contribution"] == 0
        assert check["proposal_search"] == ({"basis": "reduced", "width": 1024, "height": 1024} if large else
                                            {"basis": "source", "width": 768, "height": 768})
        changed_grid = copy.deepcopy(document)
        changed_grid["copy_affine"]["proposal_search"]["width"] -= 1
        with pytest.raises(image_packets_v7.PacketIntegrityError, match="different working grid"):
            image_packets_v7.validate_packet_set(changed_grid, analysis_id=analysis.id, revision=analysis.revision,
                                                 original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
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
    assert ("1024 × 1024 pixel working image" in engine.pdf_text(rendered)) is large
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


@pytest.mark.parametrize("wrapped", [False, True])
def test_jpeg_copy_evidence_proves_equal_encoded_blocks_and_rejects_altered_coordinates(evidence_http, monkeypatch, wrapped):
    f = evidence_http
    aid = "image-unprepared"
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_jpeg_block_copy_review", True)
    monkeypatch.setattr(settings, "image_affine_copy_review", wrapped)
    monkeypatch.setattr(settings, "image_embedded_pair_review", wrapped)
    rng = np.random.default_rng(822)
    pixels = rng.integers(0, 256, (512, 768, 3), dtype=np.uint8)
    pixels[41:89, 49:97] = pixels[233:281, 241:289]
    raw = io.BytesIO()
    Image.fromarray(pixels).save(raw, "JPEG", quality=75, subsampling=2)
    original = raw.getvalue()
    with Image.open(io.BytesIO(original)) as source:
        rgb = source.convert("RGB")
    assert image_packets_v7.selected_translation_pairs(np.asarray(rgb)) == ("no_pair", [], None)
    with Session(f.engine) as db:
        analysis = db.get(Analysis, aid)
        stored = f.store.put_bytes(original)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        evidence = service.persist_packet_set(db, analysis, f.store, result=result)
        assert evidence["version"] == ("image-evidence-15" if wrapped else "image-evidence-16")
        base = base_document(evidence)
        assert base["version"] == ("image-evidence-17" if wrapped else "image-evidence-16")
        assert evidence["original_result_hash"] == analysis.result_hash
        check = evidence["copy_translation"]
        assert check["basis"] == "jpeg_brightness_blocks" and check["jpeg_block_review"] == "supported_pair"
        assert check["authorship_confidence_contribution"] == 0
        pair, = check["pairs"]
        assert image_jpeg_copy_blocks_source.verify(original, [pair]) == 25
        wrong = copy.deepcopy(base)
        wrong["copy_translation"]["pairs"][0]["regions"]["area_a"]["x"] += 1
        wrong["sha256"] = image_packets_v5.packet_hash(wrong)
        with pytest.raises(image_packets_v7.PacketIntegrityError, match="encoded block grid"):
            image_packets_v7.validate_packet_set(wrong, **image_packets_v5._binding(analysis))
        wrong_pair = copy.deepcopy(pair)
        wrong_pair["regions"]["area_b"]["x"] += 8
        with pytest.raises(ValueError, match="different encoded brightness blocks"):
            image_jpeg_copy_blocks_source.verify(original, [wrong_pair])
        db.commit()
    monkeypatch.setattr(settings, "image_jpeg_block_copy_review", False)
    reply = f.client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=f.viewer)
    assert reply.status_code == 200 and reply.json() == evidence
    for role, asset in pair["artifacts"].items():
        response = f.client.get(asset["url"], headers=f.viewer)
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.content)) as plate:
            assert plate.tobytes() == rgb.crop(image_packets_v7.box_bounds(pair["regions"][role], *rgb.size)).tobytes()


@pytest.mark.parametrize("quality", [85, 90])
def test_offgrid_jpeg_evidence_replays_native_color_measurements(evidence_http, monkeypatch, quality):
    f = evidence_http
    aid = "image-unprepared"
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_jpeg_offgrid_copy_review", True)
    monkeypatch.setattr(settings, "image_affine_copy_review", True)
    monkeypatch.setattr(settings, "image_embedded_pair_review", True)
    seed, grid = (1, 75) if quality == 85 else (4, 110)
    rng = np.random.default_rng(seed)
    pixels = cv2.resize(rng.integers(0, 256, (grid, grid, 3), dtype=np.uint8), (640, 640), interpolation=cv2.INTER_CUBIC)
    if quality == 85:
        pixels[52:116, 83:147] = pixels[527:591, 524:588]
    else:
        pixels[123:171, 137:177] = pixels[442:490, 434:474]
    raw = io.BytesIO()
    Image.fromarray(pixels).save(raw, "JPEG", quality=quality, subsampling=2)
    original = raw.getvalue()
    with Image.open(io.BytesIO(original)) as source:
        rgb = source.convert("RGB")
    raw.seek(0)
    raw.truncate()
    Image.fromarray(pixels).save(raw, "JPEG", quality=82, subsampling=2)
    assert not image_jpeg_offgrid_source.supported(raw.getvalue())
    with Session(f.engine) as db:
        analysis = db.get(Analysis, aid)
        stored = f.store.put_bytes(original)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        evidence = service.persist_packet_set(db, analysis, f.store, result=result)
        base = base_document(evidence)
        assert base["version"] == "image-evidence-19"
        check = base["copy_translation"]
        assert check["basis"] == "jpeg_offgrid_texture" and check["jpeg_block_review"] == "no_pair"
        assert check["authorship_confidence_contribution"] == 0 and check["jpeg_offgrid_review"]["status"] == "supported_pair"
        pair, = check["pairs"]
        measurement = check["jpeg_offgrid_review"]["measurement"]
        image_jpeg_offgrid_source.verify(original, np.asarray(rgb), pair, measurement)
        with pytest.raises(ValueError, match="differs from the original crop"):
            image_jpeg_offgrid_source.verify(original, np.asarray(rgb), pair,
                                            {**measurement, "interior_mean": measurement["interior_mean"] + .01})
        if quality == 85:
            cropped = copy.deepcopy(pair)
            for region in cropped["regions"].values():
                region["w"] = 55
            with pytest.raises(ValueError, match="smaller than the supported footprint"):
                image_jpeg_offgrid_source.verify(original, np.asarray(rgb), cropped, measurement)
            report = specimen("image", "standard")
            frozen = report.model_copy(update={"result": result, "image_evidence": evidence,
                "analysis": report.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
                "media": report.media.model_copy(update={"sha256": analysis.sha256})})
            monkeypatch.setattr(image_evidence, "default_store", lambda: f.store)
            assert image_evidence.manifest(frozen)["copy_translation"] == check
            rendered_pdf = engine.render_pdf(frozen).pdf
            pdf_text = " ".join(engine.pdf_text(rendered_pdf).split())
            assert "Mean RGB difference inside" in pdf_text
            with pymupdf.open(stream=rendered_pdf, filetype="pdf") as pdf:
                actual = set()
                for page in pdf:
                    for xref, _, width, height, *_ in page.get_images():
                        if (width, height) == (pair["regions"]["area_a"]["w"], pair["regions"]["area_a"]["h"]):
                            actual.add(pymupdf.Pixmap(pdf, xref).samples)
                assert {rgb.crop(image_packets_v7.box_bounds(region, *rgb.size)).tobytes()
                        for region in pair["regions"].values()} <= actual
        wrong = copy.deepcopy(base)
        for role in ("area_a", "area_b"):
            wrong["copy_translation"]["pairs"][0]["regions"][role]["w"] = 31
            wrong["copy_translation"]["pairs"][0]["artifacts"][role]["width"] = 31
        wrong["sha256"] = image_packets_v5.packet_hash(wrong)
        with pytest.raises(image_packets_v7.PacketIntegrityError, match="outside its measured geometry"):
            image_packets_v7.validate_packet_set(wrong, **image_packets_v5._binding(analysis))
        db.commit()
    monkeypatch.setattr(settings, "image_jpeg_offgrid_copy_review", False)
    reply = f.client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=f.viewer)
    assert reply.status_code == 200 and reply.json() == evidence
    for role, asset in pair["artifacts"].items():
        response = f.client.get(asset["url"], headers=f.viewer)
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.content)) as plate:
            assert plate.tobytes() == rgb.crop(image_packets_v7.box_bounds(pair["regions"][role], *rgb.size)).tobytes()


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


@pytest.mark.parametrize("quality,offgrid", [(85, "no_pair"), (90, "no_pair"), (95, "unsupported_profile")])
def test_rotated_jpeg_replays_both_native_areas_and_rejects_changed_measurements(monkeypatch, quality, offgrid):
    from app.detectors.image import jpeg_turned_copy
    from app.evidence import image_jpeg_turned_copy_source as rotated

    rng = np.random.default_rng(58)
    pixels = cv2.resize(rng.integers(0, 256, (86, 113, 3), dtype=np.uint8), (640, 512), interpolation=cv2.INTER_CUBIC)
    pixels[72:144, 81:137] = np.rot90(pixels[313:369, 405:477])
    image = Image.fromarray(pixels)
    raw = _bytes(image, format="JPEG", quality=quality, subsampling=2)
    with Image.open(io.BytesIO(raw)) as source:
        rgb = np.array(source)
    options = {"expanded": quality in {85, 95}, "offgrid_status": offgrid}
    assert image_jpeg_offgrid_source.supported(raw) is (quality != 95)
    review, pairs, turns = rotated.review(raw, rgb, **options)
    assert review["status"] == "supported_pair" and turns == 1
    pair = {"transform": "rotate_90_ccw", "regions": {
        area: dict(zip("xywh", box, strict=True)) for area, box in zip(("area_a", "area_b"), pairs[0], strict=True)}}
    rotated.verify(raw, rgb, pair, review["measurement"], **options)
    changed = {**review["measurement"], "interior_mean": 0.0}
    with pytest.raises(ValueError, match="differ from the submitted image"):
        rotated.verify(raw, rgb, pair, changed, **options)
    with pytest.raises(ValueError, match="requires a quarter turn or half turn"):
        rotated.verify(raw, rgb, {**pair, "transform": "same_orientation"}, review["measurement"], **options)
    if quality in {85, 95}:
        with pytest.raises(ValueError, match="supported encoded JPEG"):
            rotated.verify(raw, rgb, pair, review["measurement"], expanded=False, offgrid_status=offgrid)
        with pytest.raises(ValueError, match="supported encoded JPEG"):
            rotated.verify(raw, rgb, pair, review["measurement"], expanded=True,
                           offgrid_status="no_pair" if quality == 95 else "unsupported_profile")

    def unexpected_check(_):
        pytest.fail("An unsupported JPEG must not run the rotation search")

    monkeypatch.setattr(jpeg_turned_copy, "analyse", unexpected_check)
    unsupported = _bytes(image, format="JPEG", quality=85, subsampling=2)
    assert rotated.review(unsupported, rgb) == ({"status": "unsupported_profile", "measurement": None}, [], 0)
    unsupported = _bytes(image, format="JPEG", quality=92, subsampling=2)
    assert rotated.review(unsupported, rgb, expanded=True, offgrid_status="unsupported_profile") == (
        {"status": "unsupported_profile", "measurement": None}, [], 0)


@pytest.mark.parametrize(("size", "quality", "subsampling", "orientation", "supported"), [
    ((1920, 1280), 90, 2, 1, True),
    ((2048, 3072), 95, 2, 1, True),
    ((3072, 2048), 85, 2, 1, False),
    ((3072, 2048), 95, 0, 1, False),
    ((3072, 2048), 95, 2, 6, False),
    ((3073, 2048), 90, 2, 1, False),
    ((1280, 1023), 90, 2, 1, False),
    ((1024, 1024), 90, 2, 1, False),
])
def test_native_rotation_scope_requires_a_supported_encoded_jpeg(
    monkeypatch, size, quality, subsampling, orientation, supported,
):
    from app.detectors.image import jpeg_native_turned_copy
    from app.evidence import image_jpeg_turned_copy_source as rotated

    image = Image.new("RGB", size)
    exif = image.getexif()
    exif[274] = orientation
    raw = _bytes(image, format="JPEG", quality=quality, subsampling=subsampling, exif=exif)
    calls = []

    def search(rgb):
        calls.append(rgb)
        return "no_pair", []

    monkeypatch.setattr(jpeg_native_turned_copy, "analyse", search)
    status = "no_pair" if supported else "unsupported_profile"
    assert rotated.review(raw, image, native=True, offgrid_status=None) == (
        {"status": status, "measurement": None}, [], 0)
    assert len(calls) == int(supported)


@pytest.mark.parametrize(("quality", "progressive", "sampling", "orientation", "legacy", "extended", "browser"), [
    (75, False, 2, 1, True, True, False),
    (75, True, 2, 1, True, True, False),
    (90, False, 2, 1, False, True, False),
    (90, True, 2, 1, False, False, False),
    (90, False, 0, 1, False, False, False),
    (90, False, 2, 6, False, False, False),
    (92, False, 2, 1, False, False, True),
    (92, True, 2, 1, False, False, False),
    (92, False, 0, 1, False, False, False),
    (92, False, 2, 6, False, False, False),
    (93, False, 2, 1, False, False, False),
])
def test_diagonal_jpeg_profile_uses_the_recorded_version(quality, progressive, sampling, orientation, legacy, extended, browser):
    image = Image.new("RGB", (640, 512))
    exif = image.getexif()
    exif[274] = orientation
    data = _bytes(image, format="JPEG", quality=quality, progressive=progressive, subsampling=sampling, exif=exif)
    rgb = np.asarray(image)
    assert image_jpeg_diagonal_source.supported(data, rgb, version=JPEG_DIAGONAL_VERSION) is legacy
    assert image_jpeg_diagonal_source.supported(data, rgb, version=JPEG_DIAGONAL_Q90_VERSION) is extended
    assert image_jpeg_diagonal_source.supported(data, rgb, version=JPEG_DIAGONAL_Q92_VERSION) is extended
    assert image_jpeg_diagonal_source.supported(data, rgb, version=JPEG_DIAGONAL_Q92_VERSION,
                                                browser_predecessors_unavailable=True) is browser
    if not legacy:
        with pytest.raises(ValueError, match="supported upright JPEG"):
            image_jpeg_diagonal_source.verify(data, rgb, {"transform": "top_left_to_bottom_right"}, None,
                                              version=JPEG_DIAGONAL_VERSION)


def test_diagonal_jpeg_quality_90_requires_local_diagonal_review(monkeypatch):
    image = Analysis(kind="image", content_type="image/jpeg")
    for option in ("image_affine_copy_review", "image_turned_jpeg_copy_review",
                   "image_jpeg_native_rotation_review", "image_jpeg_disjoint_rotation_review",
                   "image_jpeg_diagonal_copy_review"):
        monkeypatch.setattr(settings, option, True)
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_jpeg_diagonal_q92_review", False)
    monkeypatch.setattr(settings, "image_jpeg_diagonal_q90_review", False)
    assert image_packets_v7._analysis_version(image) == JPEG_DIAGONAL_VERSION
    monkeypatch.setattr(settings, "image_jpeg_diagonal_q90_review", True)
    assert image_packets_v7._analysis_version(image) == JPEG_DIAGONAL_Q90_VERSION
    monkeypatch.setattr(settings, "image_jpeg_diagonal_q92_review", True)
    assert image_packets_v7._analysis_version(image) == JPEG_DIAGONAL_Q92_VERSION
    monkeypatch.setattr(settings, "image_jpeg_diagonal_copy_review", False)
    assert image_packets_v7._analysis_version(image) != JPEG_DIAGONAL_Q90_VERSION
    assert image_packets_v7._analysis_version(image) != JPEG_DIAGONAL_Q92_VERSION
    monkeypatch.setattr(settings, "image_jpeg_diagonal_copy_review", True)
    monkeypatch.setattr(settings, "environment", "production")
    assert image_packets_v7._analysis_version(image) != JPEG_DIAGONAL_Q90_VERSION
    assert image_packets_v7._analysis_version(image) != JPEG_DIAGONAL_Q92_VERSION


@pytest.mark.parametrize("quality", [90, 92])
def test_diagonal_jpeg_high_quality_requires_one_complete_encoded_image(quality):
    image = Image.new("RGB", (640, 512))
    data = _bytes(image, format="JPEG", quality=quality, subsampling=2)
    multi_picture = _bytes(image, format="MPO", save_all=True, append_images=[image], quality=quality, subsampling=2)
    rgb = np.asarray(image)
    policy = {"version": JPEG_DIAGONAL_Q90_VERSION if quality == 90 else JPEG_DIAGONAL_Q92_VERSION,
              "browser_predecessors_unavailable": quality == 92}
    assert image_jpeg_diagonal_source.supported(data, rgb, **policy)
    for compound in (data + data, data[:-2], multi_picture):
        assert not image_jpeg_diagonal_source.supported(compound, rgb, **policy)


def test_small_png_copy_preserves_exact_crops_and_rejects_unsupported_sources(evidence_http, monkeypatch):
    from app.evidence import image_exact_small_copy_source
    from app.evidence.image_copy_similarity_binding import SMALL_PNG_BASIS
    from scripts.image_png_exif_preview_transfer import corrupt_crc

    f = evidence_http
    aid = "image-unprepared"
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_small_png_copy_review", True)
    monkeypatch.setattr(settings, "image_turned_png_copy_review", False)
    monkeypatch.setattr(settings, "image_affine_copy_review", True)
    monkeypatch.setattr(settings, "image_embedded_pair_review", True)
    rng = np.random.default_rng(1532)
    rgb = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    rgb[74:108, 85:119] = rgb[344:378, 335:369]
    image = Image.fromarray(rgb)
    encoded = _bytes(image, format="PNG")
    for wrong in (_bytes(image, format="JPEG"), _bytes(image.convert("RGBA"), format="PNG"), corrupt_crc(encoded, b"IDAT"),
                  corrupt_crc(encoded, b"IEND"), _bytes(image.resize((800, 800)), format="PNG")):
        assert image_exact_small_copy_source.review(wrong)[0]["status"] == "unsupported_profile"
    with Session(f.engine) as db:
        analysis = db.get(Analysis, aid)
        analysis.content_type = "image/png"
        stored = f.store.put_bytes(encoded)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        evidence = service.persist_packet_set(db, analysis, f.store, result=result)
        base = base_document(evidence)
        assert base["version"] == "image-evidence-21"
        check = base["copy_translation"]
        assert check["basis"] == SMALL_PNG_BASIS and check["authorship_confidence_contribution"] == 0
        pair, = check["pairs"]
        assert pair["regions"] == {"area_a": {"x": 85, "y": 74, "w": 34, "h": 34},
                                   "area_b": {"x": 335, "y": 344, "w": 34, "h": 34}}
        measurement = check["small_png_copy_review"]["measurement"]
        assert measurement["matching_pixels"] == 34 ** 2
        image_exact_small_copy_source.verify(encoded, pair, measurement)
        changed = copy.deepcopy(pair)
        changed["regions"]["area_a"]["x"] += 1
        with pytest.raises(ValueError, match="measurement differs"):
            image_exact_small_copy_source.verify(encoded, changed, measurement)
        changed = copy.deepcopy(base)
        changed["copy_translation"]["small_png_copy_review"]["measurement"]["matching_pixels"] += 1
        changed["sha256"] = image_packets_v5.packet_hash(changed)
        with pytest.raises(image_packets_v7.PacketIntegrityError, match="exact crop geometry"):
            image_packets_v7.validate_packet_set(changed, **image_packets_v5._binding(analysis))
        db.commit()
    monkeypatch.setattr(settings, "image_small_png_copy_review", False)
    reply = f.client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=f.viewer)
    assert reply.status_code == 200 and reply.json() == evidence
    for role, crop in pair["artifacts"].items():
        bounds = image_packets_v7.box_bounds(pair["regions"][role], *image.size)
        response = f.client.get(crop["url"], headers=f.viewer)
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.content)) as output:
            assert output.tobytes() == image.crop(bounds).tobytes()


@pytest.mark.parametrize("axis, transform", [(1, "horizontal_reflection"), (0, "vertical_reflection")])
@pytest.mark.parametrize("affine", [False, True], ids=["exact-profile", "exact-and-geometric-profile"])
@pytest.mark.parametrize("webp", [False, True], ids=["png", "webp"])
def test_reflected_pair_binds_original_crops_and_exact_axis(evidence_http, monkeypatch, axis, transform, affine, webp):
    from app.evidence import image_exact_small_copy_source
    from app.evidence.image_copy_similarity_binding import (
        PNG_EXACT_REFLECTION_BASIS,
        PNG_EXACT_REFLECTION_KEY,
        WEBP_EXACT_COPY_KEY,
        WEBP_EXACT_REFLECTION_BASIS,
        WEBP_EXACT_REFLECTION_KEY,
    )

    f = evidence_http
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_lossless_webp_reflection_review", webp)
    monkeypatch.setattr(settings, "image_png_reflection_review", not webp)
    monkeypatch.setattr(settings, "image_affine_copy_review", affine)
    monkeypatch.setattr(settings, "image_embedded_pair_review", True)
    rgb = np.random.default_rng(4206).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    rgb[72:100, 81:117] = np.flip(rgb[246:274, 351:387], axis=axis)
    image = Image.fromarray(rgb)
    encoded = _bytes(image, format="WEBP", lossless=True) if webp else _bytes(image, format="PNG")
    assert image_exact_small_copy_source.review(encoded, webp=webp, include_rotation=True)[0]["status"] == "no_pair"
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-unprepared")
        analysis.content_type = "image/webp" if webp else "image/png"
        stored = f.store.put_bytes(encoded)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        document = service.persist_packet_set(db, analysis, f.store, result=result)
        base = base_document(document)
        assert base["version"] == f"image-evidence-{(34 if webp else 36) + affine}"
        check = base["copy_translation"]
        assert check["basis"] == (WEBP_EXACT_REFLECTION_BASIS if webp else PNG_EXACT_REFLECTION_BASIS)
        assert check["authorship_confidence_contribution"] == 0
        assert check[WEBP_EXACT_COPY_KEY if webp else "small_png_copy_review"] == {"status": "no_pair", "measurement": None}
        measured = check[WEBP_EXACT_REFLECTION_KEY if webp else PNG_EXACT_REFLECTION_KEY]["measurement"]
        assert measured["matching_pixels"] == 36 * 28
        pair, = check["pairs"]
        assert pair["transform"] == transform
        assert pair["regions"] == {"area_a": {"x": 81, "y": 72, "w": 36, "h": 28},
                                   "area_b": {"x": 351, "y": 246, "w": 36, "h": 28}}
        flipped = {**pair, "transform": "vertical_reflection" if axis == 1 else "horizontal_reflection"}
        with pytest.raises(ValueError, match="measurement differs"):
            image_exact_small_copy_source.verify(encoded, flipped, measured, webp=webp, reflected=True)
        for role, crop in pair["artifacts"].items():
            bounds = image_packets_v7.box_bounds(pair["regions"][role], *image.size)
            with Image.open(io.BytesIO(f.store.read_bytes(crop["sha256"]))) as actual:
                assert actual.tobytes() == image.crop(bounds).tobytes()


@pytest.mark.parametrize("webp", [False, True], ids=["png", "webp"])
def test_reflection_review_withholds_symmetry_and_competing_exact_pairs(webp):
    from app.evidence.image_exact_small_copy_source import review_reflection

    rgb = np.random.default_rng(1206).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    symmetrical = rgb.copy()
    symmetrical[:, 256:] = symmetrical[:, :256][:, ::-1]
    carrier = "WEBP" if webp else "PNG"
    encoded = _bytes(Image.fromarray(symmetrical), format=carrier, lossless=True)
    assert review_reflection(encoded, webp=webp)[0] == {"status": "unassessed_match_budget", "measurement": None}
    rgb[72:100, 81:117] = rgb[246:274, 351:387][:, ::-1]
    rgb[114:134, 383:407] = rgb[289:309, 119:143][::-1]
    encoded = _bytes(Image.fromarray(rgb), format=carrier, lossless=True)
    assert review_reflection(encoded, webp=webp)[0] == {"status": "ambiguous_pairs", "measurement": None}
    outside = (_bytes(Image.fromarray(rgb), format=carrier, lossless=False) if webp else
               _bytes(Image.fromarray(rgb).convert("RGBA"), format="PNG"))
    assert review_reflection(outside, webp=webp)[0] == {"status": "unsupported_profile", "measurement": None}


@pytest.mark.parametrize("webp", [False, True], ids=["png", "webp"])
def test_diagonal_reflection_uses_new_packets_and_replays_exact_swapped_crops(evidence_http, monkeypatch, webp):
    from app.evidence.image_copy_similarity_binding import (
        EXACT_DIAGONAL_NOTE,
        PNG_DIAGONAL_EXACT_VERSION,
        PNG_EXACT_REFLECTION_KEY,
        WEBP_DIAGONAL_EXACT_VERSION,
        WEBP_EXACT_REFLECTION_KEY,
    )
    from app.evidence.image_exact_small_copy_source import review_reflection, verify

    f = evidence_http
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_disjoint_exact_copy_review", True)
    monkeypatch.setattr(settings, "image_affine_copy_review", True)
    monkeypatch.setattr(settings, "image_three_exact_copy_review", True)
    monkeypatch.setattr(settings, "image_diagonal_exact_copy_review", False)
    rgb = np.random.default_rng(1297).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    aligned = rgb[72:100, 81:117].transpose(1, 0, 2)
    rgb[246:282, 351:379] = aligned[::-1, ::-1] if webp else aligned
    image = Image.fromarray(rgb)
    encoded = _bytes(image, format="WEBP" if webp else "PNG", lossless=True)
    assert review_reflection(encoded, webp=webp)[0]["status"] == "no_pair"
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-unprepared")
        analysis.content_type = "image/webp" if webp else "image/png"
        stored = f.store.put_bytes(encoded)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        old = service.persist_packet_set(db, analysis, f.store, result=result)
        assert base_document(old)["copy_translation"]["pairs"] == []

        monkeypatch.setattr(settings, "image_diagonal_exact_copy_review", True)
        new_analysis = Analysis(id="image-diagonal-copy", kind=analysis.kind, filename=analysis.filename,
                                content_type=analysis.content_type, size_bytes=analysis.size_bytes,
                                sha256=analysis.sha256, status="done", revision=analysis.revision,
                                result_schema_version=analysis.result_schema_version, result_hash=analysis.result_hash)
        db.add(new_analysis)
        db.flush()
        document = service.persist_packet_set(db, new_analysis, f.store, result=result)
        base = base_document(document)
        assert base["version"] == (WEBP_DIAGONAL_EXACT_VERSION if webp else PNG_DIAGONAL_EXACT_VERSION)
        assert document["id"] != old["id"]
        check = base["copy_translation"]
        assert check["interpretation"] == EXACT_DIAGONAL_NOTE and check["authorship_confidence_contribution"] == 0
        pair, = check["pairs"]
        assert pair["transform"] == ("top_right_to_bottom_left" if webp else "top_left_to_bottom_right")
        assert pair["regions"] == {"area_a": {"x": 81, "y": 72, "w": 36, "h": 28},
                                   "area_b": {"x": 351, "y": 246, "w": 28, "h": 36}}
        measured = check[WEBP_EXACT_REFLECTION_KEY if webp else PNG_EXACT_REFLECTION_KEY]["measurement"]
        verify(encoded, pair, measured, webp=webp, reflected=True, include_diagonal=True)
        with pytest.raises(ValueError, match="measurement differs"):
            verify(encoded, pair, measured, webp=webp, reflected=True)
        for role, crop in pair["artifacts"].items():
            bounds = image_packets_v7.box_bounds(pair["regions"][role], *image.size)
            with Image.open(io.BytesIO(f.store.read_bytes(crop["sha256"]))) as actual:
                assert actual.tobytes() == image.crop(bounds).tobytes()
        assert service.load_packet_set(db, analysis, f.store, set_id=old["id"]) == old


@pytest.mark.parametrize("webp", [False, True], ids=["png", "webp"])
@pytest.mark.parametrize("affine", [False, True])
def test_two_exact_pairs_bind_each_transform_and_both_source_witnesses(evidence_http, monkeypatch, webp, affine):
    from app.evidence.image_copy_similarity_binding import MULTI_EXACT_BASIS, MULTI_EXACT_KEY
    from app.evidence.image_exact_small_copy_source import verify_multiple

    f = evidence_http
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_disjoint_exact_copy_review", True)
    monkeypatch.setattr(settings, "image_affine_copy_review", affine)
    monkeypatch.setattr(settings, "image_embedded_pair_review", True)
    rgb = np.random.default_rng(4221).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    rgb[330:358, 360:384] = rgb[60:88, 40:64]
    rgb[380:408, 90:126] = rgb[60:88, 260:296][::-1]
    encoded = _bytes(Image.fromarray(rgb), format="WEBP" if webp else "PNG", lossless=True)
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-unprepared")
        analysis.content_type = "image/webp" if webp else "image/png"
        stored = f.store.put_bytes(encoded)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        document = service.persist_packet_set(db, analysis, f.store, result=result)
        base = base_document(document)
        assert base["version"] == f"image-evidence-{(40 if webp else 38) + affine}"
        check = base["copy_translation"]
        assert check["basis"] == MULTI_EXACT_BASIS
        assert check["authorship_confidence_contribution"] == 0
        assert [pair["transform"] for pair in check["pairs"]] == ["same_orientation", "vertical_reflection"]
        assert [item["matching_pixels"] for item in check[MULTI_EXACT_KEY]["measurements"]] == [672, 1008]
        verify_multiple(encoded, check, webp=webp)
        incorrect = copy.deepcopy(check)
        incorrect["pairs"][1]["transform"] = "horizontal_reflection"
        with pytest.raises(ValueError, match="differs from the submitted RGB pixels"):
            verify_multiple(encoded, incorrect, webp=webp)


@pytest.mark.parametrize("webp", [False, True], ids=["png", "webp"])
@pytest.mark.parametrize("affine", [False, True])
def test_three_exact_pairs_preserve_legacy_packages_and_bind_each_native_crop(evidence_http, monkeypatch, webp, affine):
    from app.evidence.image_copy_similarity_binding import MULTI_EXACT_BASIS, MULTI_EXACT_KEY
    from app.evidence.image_exact_small_copy_source import review_multiple, verify_multiple

    f = evidence_http
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_disjoint_exact_copy_review", True)
    monkeypatch.setattr(settings, "image_affine_copy_review", affine)
    monkeypatch.setattr(settings, "image_embedded_pair_review", True)
    monkeypatch.setattr(settings, "image_three_exact_copy_review", False)
    rgb = np.random.default_rng(4222).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    rgb[84:117, 371:414] = rgb[57:90, 56:99]
    rgb[297:331, 362:399] = rgb[305:339, 64:101][::-1, ::-1]
    rgb[170:207, 372:415] = rgb[166:203, 158:201][::-1]
    carrier = "WEBP" if webp else "PNG"
    encoded = _bytes(Image.fromarray(rgb), format=carrier, lossless=True)
    assert review_multiple(encoded, webp=webp)[0]["status"] == "ambiguous_pairs"
    outside = _bytes(Image.fromarray(rgb[:500]), format=carrier, lossless=True)
    assert not review_multiple(outside, webp=webp, allow_three=True)[1]
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "image-unprepared")
        analysis.content_type = "image/webp" if webp else "image/png"
        stored = f.store.put_bytes(encoded)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        old = service.persist_packet_set(db, analysis, f.store, result=result)
        assert base_document(old)["version"] == f"image-evidence-{(40 if webp else 38) + affine}"
        assert base_document(old)["copy_translation"]["basis"] != MULTI_EXACT_BASIS

        monkeypatch.setattr(settings, "image_three_exact_copy_review", True)
        new_analysis = Analysis(id="image-three-pairs", kind=analysis.kind, filename=analysis.filename,
                                content_type=analysis.content_type, size_bytes=analysis.size_bytes,
                                sha256=analysis.sha256, status="done", revision=analysis.revision,
                                result_schema_version=analysis.result_schema_version, result_hash=analysis.result_hash)
        db.add(new_analysis)
        db.flush()
        document = service.persist_packet_set(db, new_analysis, f.store, result=result)
        base = base_document(document)
        assert base["version"] == f"image-evidence-{(44 if webp else 42) + affine}"
        check = base["copy_translation"]
        assert check["basis"] == MULTI_EXACT_BASIS and check["authorship_confidence_contribution"] == 0
        assert [pair["id"] for pair in check["pairs"]] == ["T01", "T02", "T03"]
        assert [pair["transform"] for pair in check["pairs"]] == ["same_orientation", "rotate_180", "vertical_reflection"]
        assert [m["matching_pixels"] for m in check[MULTI_EXACT_KEY]["measurements"]] == [1419, 1258, 1591]
        verify_multiple(encoded, check, webp=webp)
        for pair in check["pairs"]:
            for role, artifact in pair["artifacts"].items():
                x, y, w, h = (pair["regions"][role][field] for field in "xywh")
                with Image.open(io.BytesIO(f.store.read_bytes(artifact["sha256"]))) as crop:
                    assert crop.tobytes() == rgb[y:y+h, x:x+w].tobytes()
        assert service.load_packet_set(db, analysis, f.store, set_id=old["id"]) == old
        altered = copy.deepcopy(check)
        altered["pairs"][2]["transform"] = "horizontal_reflection"
        with pytest.raises(ValueError, match="differs from the submitted RGB pixels"):
            verify_multiple(encoded, altered, webp=webp)


@pytest.mark.parametrize("webp", [False, True], ids=["png", "webp"])
def test_rotated_exact_copy_binds_direction_and_untouched_rectangular_crops(evidence_http, monkeypatch, webp):
    from app.evidence import image_exact_small_copy_source
    from app.evidence.image_copy_similarity_binding import (
        TURNED_PNG_BASIS,
        WEBP_EXACT_COPY_BASIS,
        WEBP_EXACT_COPY_KEY,
    )

    f = evidence_http
    aid = "image-unprepared"
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_small_png_copy_review", True)
    monkeypatch.setattr(settings, "image_turned_png_copy_review", True)
    monkeypatch.setattr(settings, "image_lossless_webp_copy_review", webp)
    monkeypatch.setattr(settings, "image_affine_copy_review", webp)
    monkeypatch.setattr(settings, "image_embedded_pair_review", webp)
    rng = np.random.default_rng(1592)
    rgb = rng.integers(0, 256, (512 if webp else 384, 512, 3), dtype=np.uint8)
    rgb[74:104, 85:109] = np.rot90(rgb[240:264, 335:365])
    image = Image.fromarray(rgb)
    encoded = _bytes(image, format="WEBP", lossless=True) if webp else _bytes(image, format="PNG")
    if webp:
        lossy = _bytes(image, format="WEBP", lossless=False)
        assert image_exact_small_copy_source.review(lossy, webp=True)[0]["status"] == "unsupported_profile"
    assert image_exact_small_copy_source.review(encoded, webp=webp)[0]["status"] == "no_pair"
    with Session(f.engine) as db:
        analysis = db.get(Analysis, aid)
        analysis.content_type = "image/webp" if webp else "image/png"
        stored = f.store.put_bytes(encoded)
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        result = copy.deepcopy(specimen("image", "standard").result)
        result.update(exhibits=[], signals=[], findings=[])
        result["input"]["sha256"] = stored.sha256
        analysis.result_hash = canonical_hash(result)
        f.store.put_bytes(canonical_json(result))
        document = service.persist_packet_set(db, analysis, f.store, result=result)
        base = base_document(document)
        assert base["version"] == ("image-evidence-33" if webp else "image-evidence-22")
        check = base["copy_translation"]
        assert check["basis"] == (WEBP_EXACT_COPY_BASIS if webp else TURNED_PNG_BASIS)
        assert check["authorship_confidence_contribution"] == 0
        pair, = check["pairs"]
        assert pair["transform"] == "rotate_90_ccw"
        assert pair["regions"] == {"area_a": {"x": 85, "y": 74, "w": 24, "h": 30},
                                   "area_b": {"x": 335, "y": 240, "w": 30, "h": 24}}
        measurement = check[WEBP_EXACT_COPY_KEY if webp else "small_png_copy_review"]["measurement"]
        assert measurement["matching_pixels"] == 24 * 30
        changed = copy.deepcopy(base)
        changed["copy_translation"]["pairs"][0]["transform"] = "rotate_180"
        with pytest.raises(image_packets_v7.PacketIntegrityError, match="share a native size"):
            image_packets_v7.validate_packet_set(changed, **image_packets_v5._binding(analysis))
        wrong = copy.deepcopy(pair)
        wrong["regions"]["area_b"]["y"] += 1
        with pytest.raises(ValueError, match="measurement differs"):
            image_exact_small_copy_source.verify(encoded, wrong, measurement, webp=webp)
        db.commit()
    evidence = f.client.get(f"/api/v2/analyses/{aid}/evidence-packets", headers=f.viewer)
    assert evidence.status_code == 200 and evidence.json() == document
    for role, crop in pair["artifacts"].items():
        bounds = image_packets_v7.box_bounds(pair["regions"][role], *image.size)
        response = f.client.get(crop["url"], headers=f.viewer)
        assert response.status_code == 200
        with Image.open(io.BytesIO(response.content)) as output:
            assert output.tobytes() == image.crop(bounds).tobytes()
