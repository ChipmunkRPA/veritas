"""Exercise automatic mirrored pairing through the normal frozen evidence path."""

from __future__ import annotations

import copy
import io

import cv2
import numpy as np
import pytest
from PIL import Image
from sqlalchemy.orm import Session
from test_image_evidence_access import evidence_http as evidence_http
from test_image_packets import _bytes
from test_image_packets import packet_fixture as packet_fixture

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.image import copy_move, copy_similarity
from app.evidence import image_packets_v5, image_packets_v7, service
from app.reports import engine, image_evidence, snapshot
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


def reflected_result(packet_fixture, *, reflected=True, low_texture=False):
    db, store, analysis, _, _ = packet_fixture
    make_source = low_texture_source if low_texture else authored_source
    source = make_source(reflected=reflected)
    obj = store.put_bytes(_bytes(source, format="PNG"))
    analysis.sha256, analysis.size_bytes = obj.sha256, obj.size_bytes
    result = copy.deepcopy(specimen("image", "standard").result)
    result.update(exhibits=[], signals=[], findings=[])
    result["input"]["sha256"] = obj.sha256
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    return db, store, analysis, result, source


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


def test_historical_v7_remains_readable_after_writing_v8(packet_fixture, monkeypatch):
    db, store, analysis, result, _ = reflected_result(packet_fixture)
    historical = service.build_packet_set(db, analysis, store, result)
    historical["version"] = image_packets_v7.V7
    historical["id"] = image_packets_v5.current_set_id(analysis, image_packets_v7.V7)
    for pair in historical["copy_similarity"]["pairs"]:
        for asset in pair["artifacts"].values():
            asset["url"] = image_packets_v7.url(historical, asset["sha256"])
    historical["sha256"] = image_packets_v5.packet_hash(historical)
    historical_row = register_artefact(
        db, store, canonical_json(historical), media_type="application/json", role="exhibit_payload",
        created_by="image-evidence-packets/5", analysis_id=analysis.id, artefact_id=historical["id"],
    )
    newest = db.get(type(historical_row), service.current_set_id(analysis))
    db.delete(newest)
    db.flush()
    with monkeypatch.context() as context:
        context.setattr(copy_similarity, "analyse", lambda _: pytest.fail("A historical read must not rerun the detector"))
        assert service.load_packet_set(db, analysis, store) == historical
    latest = service.persist_packet_set(db, analysis, store, result=result)
    assert latest["version"] == "image-evidence-8" and latest["id"] != historical["id"]
    assert service.load_packet_set(db, analysis, store) == latest
    assert service.load_packet_set(db, analysis, store, set_id=historical["id"]) == historical
    assert store.read_bytes(historical_row.sha256) == canonical_json(historical)


def test_no_pair_keeps_explicit_coverage_and_no_assets(packet_fixture):
    db, store, analysis, result, _ = reflected_result(packet_fixture, reflected=False)
    document = service.persist_packet_set(db, analysis, store, result=result)
    assert document["copy_similarity"]["status"] == "no_pair"
    assert document["copy_similarity"]["pairs"] == []
    assert service.asset_refs(document) == {}


def test_authenticated_image_api_serves_both_pair_crops(evidence_http):
    fixture = evidence_http
    analysis_id = "image-unprepared"
    with Session(fixture.engine) as db:
        analysis = db.get(Analysis, analysis_id)
        db, store, analysis, result, _ = reflected_result((db, fixture.store, analysis, None, None), low_texture=True)
        expected = service.persist_packet_set(db, analysis, store, result=result)
        db.commit()
    response = fixture.client.get(f"/api/v2/analyses/{analysis_id}/evidence-packets", headers=fixture.viewer)
    assert response.status_code == 200 and response.json() == expected
    (pair,) = expected["copy_similarity"]["pairs"]
    assert pair["regions"]["area_a"]["x"] == 0
    for asset in pair["artifacts"].values():
        assert fixture.client.get(asset["url"], headers=fixture.outsider).status_code == 404
        plate = fixture.client.get(asset["url"], headers=fixture.viewer)
        assert plate.status_code == 200
        assert plate.content == fixture.store.read_bytes(asset["sha256"])


def test_report_and_json_share_source_bound_pair_and_redaction(packet_fixture, monkeypatch):
    db, store, analysis, result, _ = reflected_result(packet_fixture)
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
    assert "Same-image copy similarity" in html and "Area B · x 527, y 301, 128 × 128 px" in html
    assert "which area came first" in html
    redacted = frozen.model_copy(update={"report": frozen.report.model_copy(update={"redacted": True})})
    assert all(
        image["uri"] is None
        for image in image_evidence.view(redacted)["copy_similarity"]["pairs"][0]["images"]
    )
