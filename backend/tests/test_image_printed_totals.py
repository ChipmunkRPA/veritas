from __future__ import annotations

import copy
import hashlib
import io

import pytest
from PIL import Image

from app.core.canonical import canonical_hash
from app.evidence import image_printed_euro_totals as euro
from app.evidence import image_printed_totals as reader
from app.evidence import image_printed_totals_binding as binding


def _word(text, bounds):
    return reader.Word(text, bounds, 99.0)


@pytest.mark.parametrize("included", [True, False])
@pytest.mark.parametrize("extra_amount", [False, True])
def test_euro_summary_checks_the_declared_basis_and_linked_crop(monkeypatch, included, extra_amount):
    if included:
        statements = ["Subtotal (incl. VAT): 2'000.00 EUR", "Less: rebate per agreement: -EUR 5.00", "TOTAL: 2.100,00 EUR"]
        basis, expected = "tax_included", 199500
    else:
        statements = ["Plus 10% VAT on the subtotal.", "Subtotal: EUR 1,000.00", "VAT (10%): EUR 100.00",
                      "Less: rebate per agreement: -EUR 5.00", "TOTAL: EUR 1,200.00"]
        basis, expected = "tax_added", 109500
    if extra_amount:
        statements.append("Shipping later: EUR 10.00")
    by_row = []
    for i, statement in enumerate(statements):
        x, y = 20, 20 + 30 * i
        words = []
        for token in statement.split():
            right = x + 6 * len(token)
            words.append(_word(token, (x, y, right, y + 16)))
            x = right + 6
        by_row.append(words)
    monkeypatch.setattr(euro, "_confirm", lambda *_: True)
    with Image.new("RGB", (600, 320), "white") as original:
        measurement = euro.find("ocr", original, by_row, 0.0)
    if extra_amount:
        assert measurement is None
        return
    assert measurement["basis"] == basis and measurement["calculated_cents"] == expected
    assert measurement["delta_cents"] == 10500
    record = {**measurement, "source_sha256": "a" * 64, "affects_overall_confidence": False}
    assert binding.roles(record) == (("subtotal", "discount", "total") if included else ("subtotal", "tax", "discount", "total"))
    assert binding.amount_display(record, record["fields"]["discount"]["printed"]) == "-€5.00"
    assert binding.calculated_display(record) == ("€1.995,00" if included else "€1,095.00")
    evidence = {
        "schema": "chipmunk.image-evidence/1",
        "source": {"sha256": record["source_sha256"], "exif_orientation": 1, "width": 600, "height": 320},
        binding.KEY: {"record": record, "artifacts": {}},
    }
    for role, field in record["fields"].items():
        x, y, right, bottom = field["source_region"]
        evidence[binding.KEY]["artifacts"][role] = {
            "sha256": "b" * 64, "media_type": "image/png", "width": right - x, "height": bottom - y,
        }
    commitment = canonical_hash(record)
    evidence["source"][binding.COMMITMENT] = commitment
    binding.validate(evidence, commitment)
    record["basis"] = "tax_added" if included else "tax_included"
    evidence["source"][binding.COMMITMENT] = commitment = canonical_hash(record)
    with pytest.raises(ValueError, match="calculation term"):
        binding.validate(evidence, commitment)


@pytest.mark.parametrize(("printed", "calculated"), [("2'000.05", "€1'995.00"), ("2000.05", "€1995.00")])
def test_calculated_euro_total_preserves_the_printed_format(printed, calculated):
    assert binding.calculated_display({
        "schema": binding.EURO_SCHEMA, "calculated_cents": 199500, "currency": "€",
        "fields": {"total": {"printed": printed}},
    }) == calculated


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
