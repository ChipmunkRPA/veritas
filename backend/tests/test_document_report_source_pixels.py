from __future__ import annotations

import io
from copy import deepcopy

import pytest
from PIL import Image
from test_document_packets import packet_factory  # noqa: F401 - source-bound database fixture

from app.detectors.text.extract import _ocr_runtime
from app.evidence.document_packets import build_packet_set, packet_hash
from app.reports.multimedia_evidence import verify_assets
from app.storage.objects import ObjectIntegrityError


@pytest.mark.parametrize("tamper", ["metadata", "pixels"])
def test_resigned_ocr_crop_fails_source_pixel_verification(packet_factory, tamper):  # noqa: F811
    if not _ocr_runtime().get("available"):
        pytest.skip("Verified local OCR fixture data are not provisioned")
    db, store, analysis, result, _, _ = packet_factory(".pdf", ocr=True)
    original = build_packet_set(db, analysis, store, result)
    verify_assets(original, store)
    forged = deepcopy(original)
    crop = forged["packets"][0]["artifacts"]["page1_crop"]
    if tamper == "metadata":
        crop["source_crop_pixels"][0] += 1
        crop["source_crop_pixels"][2] += 1
    else:
        stream = io.BytesIO()
        Image.new("RGB", (crop["width"], crop["height"]), "white").save(stream, format="PNG")
        content = store.put_bytes(stream.getvalue())
        old = crop["sha256"]
        crop["sha256"] = content.sha256
        crop["url"] = crop["url"].replace(old, content.sha256)
    forged["sha256"] = packet_hash(forged)  # integrity hash alone no longer suffices
    with pytest.raises(ObjectIntegrityError, match="OCR (crop metadata|presentation pixels)"):
        verify_assets(forged, store)


def test_historical_ocr_set_without_integer_bounds_keeps_compatible_read_behavior(packet_factory):  # noqa: F811
    if not _ocr_runtime().get("available"):
        pytest.skip("Verified local OCR fixture data are not provisioned")
    db, store, analysis, result, _, _ = packet_factory(".pdf", ocr=True)
    historical = build_packet_set(db, analysis, store, result)
    for packet in historical["packets"]:
        for asset in packet["artifacts"].values():
            for key in ("source_crop_pixels", "source_raster_dimensions", "source_crop_coordinate_space"):
                asset.pop(key, None)
    historical["sha256"] = packet_hash(historical)
    before = deepcopy(historical)
    verify_assets(historical, store)
    assert historical == before
