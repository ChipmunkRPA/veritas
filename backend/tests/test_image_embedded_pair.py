"""Check exact single-image evidence without involving model inference."""
from __future__ import annotations

import copy
import io

import numpy as np
import pytest
from PIL import Image
from test_image_copy_similarity import result_from_source
from test_image_packets import packet_fixture as packet_fixture

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.detectors.image import embedded_pair
from app.evidence import image_embedded_pair_binding as binding
from app.evidence import image_embedded_pair_pixels as pixels
from app.evidence import image_packets_v8, service
from app.evidence.image_copy_similarity_binding import SKIPPED_CREDENTIALS, credential_skip


def paired_image(mode: str = "difference") -> Image.Image:
    rng = np.random.default_rng(29177)
    scene = rng.integers(0, 256, (640, 768, 3), dtype=np.uint8)
    photo = rng.integers(0, 256, (220, 224, 3), dtype=np.uint8)
    scene[31:251, 25:249] = photo
    scene[39:259, 465:689] = photo
    if mode != "duplicate":
        scene[120:139, 553:576] ^= 71
    if mode == "third":
        scene[365:585, 273:497] = photo
    return Image.fromarray(scene)


def embedded_pair_result(packet_fixture, monkeypatch):
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "image_affine_copy_review", False)
    monkeypatch.setattr(settings, "image_embedded_pair_review", True)
    return result_from_source(packet_fixture, paired_image())


@pytest.mark.parametrize(("mode", "status"), [
    ("difference", "local_difference"),
    ("duplicate", "matching_without_difference"),
    ("third", "unassessed_ambiguous_match"),
])
def test_native_check_preserves_localized_pixel_bounds_or_withholds(mode, status):
    actual, pair = embedded_pair.inspect(np.asarray(paired_image(mode)))
    assert actual == status
    if mode == "difference":
        assert pair is not None
        assert np.count_nonzero(pair.difference) == 437
        x, y, right, bottom = pair.difference_bounds
        assert (pair.area_a[0] + x, pair.area_a[1] + y, right - x, bottom - y) == (113, 112, 23, 19)
        assert (pair.area_b[0] + x, pair.area_b[1] + y, right - x, bottom - y) == (553, 120, 23, 19)
    else:
        assert pair is None


def test_latest_packet_keeps_base_and_rejects_recommitted_wrong_pixels(packet_fixture, monkeypatch):
    db, store, analysis, result, source = embedded_pair_result(packet_fixture, monkeypatch)
    document = service.persist_packet_set(db, analysis, store, result=result)
    assert document["version"] == binding.VERSION
    record = document[binding.KEY]
    assert record["status"] == "local_difference"
    assert record["authorship_confidence_contribution"] == 0
    assert record["pair"]["different_pixels"] == 437
    assert record["pair"]["difference_regions"]["area_b"] == {"x": 553, "y": 120, "w": 23, "h": 19}
    base = binding.base_document(document)
    assert service.load_packet_set(db, analysis, store, set_id=base["id"]) == base
    assert service.load_packet_set(db, analysis, store, set_id=document["id"]) == document
    assert analysis.result_hash == canonical_hash(result)
    mask = record["pair"]["artifacts"]["difference_mask"]
    with Image.open(io.BytesIO(store.read_bytes(mask["sha256"]))) as image:
        assert image.mode == "L"
        assert np.count_nonzero(np.asarray(image)) == 437
    assert set(binding.artifacts(document)) == set(binding.ASSETS)
    assert all(asset["sha256"] in service.asset_refs(document) for asset in binding.artifacts(document).values())

    wrong = store.put_bytes(pixels.encode(Image.new("L", (mask["width"], mask["height"]))))
    changed = copy.deepcopy(document)
    replacement = changed[binding.KEY]["pair"]["artifacts"]["difference_mask"]
    replacement.update(sha256=wrong.sha256, url=mask["url"].replace(mask["sha256"], wrong.sha256))
    changed["sha256"] = service.packet_hash(changed)
    service.validate_packet_set(changed, analysis_id=analysis.id, revision=analysis.revision,
                                original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)
    with pytest.raises(image_packets_v8.PacketIntegrityError, match="difference_mask image differs"):
        image_packets_v8.validate_assets(changed, store)
    assert pixels.source_image(pixels.encode(source.convert("RGBA"))) is None
    assert pixels.source_image(pixels.encode(source.resize((512, 512)))) is None

    monkeypatch.setattr(settings, "image_embedded_pair_review", False)
    assert service.persist_packet_set(db, analysis, store, result=result) == base
    assert service.load_packet_set(db, analysis, store, set_id=document["id"]) == document


def test_frozen_signed_result_skips_exact_comparison(packet_fixture, monkeypatch):
    db, store, analysis, result, _ = embedded_pair_result(packet_fixture, monkeypatch)
    result["manifest"] = {"sampling": {"image_copy_coverage": credential_skip(analysis.sha256)}}
    result["decision"]["provenance_record"] = {"kind": "decisive_ai"}
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))

    def fail_if_scanned(_image):
        raise AssertionError("The exact comparison must follow the frozen Content Credentials decision")

    monkeypatch.setattr(embedded_pair, "inspect", fail_if_scanned)
    document = service.persist_packet_set(db, analysis, store, result=result)
    assert document[binding.KEY]["status"] == SKIPPED_CREDENTIALS
    assert document[binding.KEY]["pair"] is None
    assert not binding.artifacts(document)
    with pytest.raises(image_packets_v8.PacketIntegrityError, match="frozen signed provenance"):
        image_packets_v8.validate_packet_source_mapping(document, {**result, "manifest": {}})
