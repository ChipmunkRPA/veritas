from __future__ import annotations

import copy
import hashlib
import io

import pytest
from PIL import Image

from app.core.canonical import canonical_hash
from app.evidence import image_printed_ids as reader
from app.evidence import image_printed_ids_binding as binding


@pytest.mark.parametrize("other", [None, "duplicate", "uncertain", "conflicting_read"])
def test_only_a_unique_twice_read_pair_can_generate_a_mismatch(monkeypatch, other):
    def recognized(binary, image, mode, deadline):
        entry = ("reference id", False, "AB-123456", [25, 90, 270, 115], True)
        confirmation = ("reference id", True, "AB-123450", [25, 150, 315, 175], other != "uncertain")
        if mode == 11:
            yield entry
            yield confirmation
            if other == "duplicate":
                yield ("reference id", False, "AB-867530", [25, 195, 285, 220], False)
        elif image.width < 580:
            yield entry
        elif other != "conflicting_read":
            yield confirmation

    monkeypatch.setattr(reader, "_ocr", recognized)
    with Image.new("RGB", (600, 320)) as image:
        record = reader._find("test", image, 0.)
    if other is None:
        assert record == {"field": "Reference ID", "values": {
            "entry": {"value": "AB-123456", "region": [25, 90, 270, 115]},
            "confirmation": {"value": "AB-123450", "region": [25, 150, 315, 175]},
        }}
    else:
        assert record is None


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_still_image_scope_supports_both_carriers(tmp_path, monkeypatch, format):
    monkeypatch.setattr(reader.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(reader, "_find", lambda *_: {"field": "Reference ID", "values": {}})
    weights = tmp_path / "eng.traineddata"
    digest = hashlib.sha256(b"local test weights").hexdigest()
    weights.write_bytes(b"local test weights")
    monkeypatch.setattr(reader.image_png_layout, "TESSDATA", tmp_path)
    monkeypatch.setattr(reader.image_png_layout, "ENGLISH_SHA256", digest)
    path = tmp_path / "image"
    with Image.new("RGB", (600, 320)) as image:
        image.save(path, format)
    assert reader.inspect(path, "0" * 64, 10.)["source_sha256"] == "0" * 64


def test_signed_position_fields_and_exact_crops_are_checked():
    region1, region2 = [25, 90, 270, 115], [25, 150, 315, 175]
    record = {"schema": binding.SCHEMA, "source_sha256": "1" * 64, "affects_overall_confidence": False,
              "field": "Reference ID", "values": {"entry": {"value": "AB-123456", "region": region1},
                                                   "confirmation": {"value": "AB-123450", "region": region2}}}
    commitment = canonical_hash(record)
    data = {"schema": "chipmunk.image-evidence/1", "source": {"sha256": record["source_sha256"],
            "exif_orientation": 1, "width": 600, "height": 320, binding.COMMITMENT: commitment},
            binding.KEY: {"record": record, "artifacts": {}}}
    assets = {}
    with Image.new("RGB", (600, 320), (224, 244, 250)) as source:
        for role, item in record["values"].items():
            with source.crop(tuple(item["region"])) as crop:
                encoded = io.BytesIO()
                crop.save(encoded, "PNG")
                png = encoded.getvalue()
                digest = hashlib.sha256(png).hexdigest()
                assets[digest] = png
                data[binding.KEY]["artifacts"][role] = {"media_type": "image/png", "width": crop.width,
                                                       "height": crop.height, "sha256": digest}
        binding.validate(data, commitment)
        reader.verify_crops(source, data[binding.KEY], assets.__getitem__)
        with pytest.raises(ValueError, match="Invalid printed identifier evidence"):
            binding.validate(data, commitment, stopped=True)
        data_without_output = copy.deepcopy(data)
        del data_without_output[binding.KEY]
        with pytest.raises(ValueError, match="Invalid printed identifier evidence"):
            binding.validate(data_without_output, commitment)
        source.putpixel((30, 92), (0, 0, 0))
        with pytest.raises(ValueError, match="crop differs from the original"):
            reader.verify_crops(source, data[binding.KEY], assets.__getitem__)
