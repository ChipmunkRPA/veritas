from __future__ import annotations

import copy
import hashlib
import io

import pytest
from PIL import Image

from app.core.canonical import canonical_hash
from app.evidence import image_printed_totals as reader
from app.evidence import image_printed_totals_binding as binding


def _word(text, bounds):
    return reader.Word(text, bounds, 99.0)


@pytest.mark.parametrize("other", [None, "additional row", "different currency", "balanced"])
def test_only_a_reproduced_mismatch_in_one_currency_is_flagged(monkeypatch, other):
    def recognized(binary, image, mode, end):
        total = "$110.00" if other == "balanced" else "$111.00"
        tax = "£10.00" if other == "different currency" else "$10.00"
        if mode != 11:
            return [_word({224: "$100.00", 204: tax}.get(image.width, total), (2, 4, 30, 19))]
        result = [
            _word("Subtotal", (50, 90, 120, 108)),
            _word("$100.00", (365, 90, 465, 108)),
            _word("Tax", (50, 127, 85, 145)),
            _word(tax, (375, 127, 465, 145)),
            _word("Total", (50, 164, 101, 182)),
            _word(total, (370, 164, 465, 182)),
        ]
        if other == "additional row":
            result.extend([_word("Service", (50, 200, 104, 220)), _word("§1.00", (399, 200, 465, 220))])
        return result

    monkeypatch.setattr(reader, "ocr", recognized)
    with Image.new("RGB", (600, 320)) as source:
        status, record = reader.find("ocr", source, 0.0)
    if other is None:
        assert status == "mismatch"
        assert record["calculated_cents"] == 11000 and record["delta_cents"] == 100
        assert record["fields"]["total"]["amount_region"] == [370, 164, 465, 182]
    else:
        assert status == ("match" if other == "balanced" else "not_supported")


def test_evidence_checks_amounts_geometry_and_source_pixels():
    record = {
        "schema": binding.SCHEMA,
        "source_sha256": "1" * 64,
        "affects_overall_confidence": False,
        "currency": "$",
        "calculated_cents": 11000,
        "delta_cents": 100,
        "fields": {
            role: {
                "printed": value,
                "source_region": [15, top, 255, top + 23],
                "amount_region": [150, top, 255, top + 23],
            }
            for role, top, value in zip(
                binding.ROLES, (90, 125, 160), ("100.00", "10.00", "111.00"), strict=True
            )
        },
    }
    commitment = canonical_hash(record)
    packet = {"record": record, "artifacts": {}}
    document = {
        "schema": "chipmunk.image-evidence/1",
        binding.KEY: packet,
        "source": {
            "sha256": record["source_sha256"],
            "exif_orientation": 1,
            "width": 600,
            "height": 320,
            binding.COMMITMENT: commitment,
        },
    }
    crops = {}
    with Image.new("RGB", (600, 320), "white") as source:
        for role, field in record["fields"].items():
            with source.crop(tuple(field["source_region"])) as crop:
                output = io.BytesIO()
                crop.save(output, "PNG")
                data = output.getvalue()
                digest = hashlib.sha256(data).hexdigest()
                crops[digest] = data
                packet["artifacts"][role] = {
                    "media_type": "image/png",
                    "width": crop.width,
                    "height": crop.height,
                    "sha256": digest,
                }
        binding.validate(document, commitment)
        reader.verify_crops(source, packet, crops.__getitem__)
        with pytest.raises(ValueError, match="Invalid printed total evidence"):
            binding.validate(document, commitment, stopped=True)
        forged = copy.deepcopy(document)
        forged[binding.KEY]["record"]["calculated_cents"] += 1
        forged[binding.KEY]["record"]["delta_cents"] -= 1
        forged_hash = canonical_hash(forged[binding.KEY]["record"])
        forged["source"][binding.COMMITMENT] = forged_hash
        with pytest.raises(ValueError, match="do not reproduce"):
            binding.validate(forged, forged_hash)
        source.putpixel((156, 91), (0, 0, 0))
        with pytest.raises(ValueError, match="differs from the original"):
            reader.verify_crops(source, packet, crops.__getitem__)
