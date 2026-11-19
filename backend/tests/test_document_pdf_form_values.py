# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
from hashlib import sha256
from zipfile import ZipFile

import pymupdf
import pytest

from app.core.canonical import canonical_hash
from app.evidence import document_pdf_form_binding as binding
from app.evidence import document_pdf_form_values as reader

BOX = [80, 174, 358, 203]
DISPLAY = "North Review Group"
OTHER = "South Records Group"


def make_pdf(path, *, variant="different", parent=False):
    with pymupdf.open() as file:
        page = file.new_page(width=612, height=792)
        page.insert_text((50, 62), "The local archive tracks the review schedule for ordinary inspection records.")
        page.insert_text((50, 93), "The administrator updates the assignment and verifies the published worksheet each month.")
        page.insert_text((50, 152), "Saved group for the next monthly review")
        field = pymupdf.Widget()
        field.field_name = "review_group"
        field.field_type = pymupdf.PDF_WIDGET_TYPE_TEXT
        field.rect = pymupdf.Rect(BOX)
        field.field_value = DISPLAY
        field.text_fontsize = 10
        page.add_widget(field)
        file.save(path)

    with pymupdf.open(stream=path.read_bytes(), filetype="pdf") as file:
        page = file[0]
        (field,) = page.widgets()
        xref = field.xref
        values = {"same": DISPLAY, "equivalent": DISPLAY.lower().replace(" ", "  "), "formatted": "AUD 1,000"}
        saved = values.get(variant, OTHER)
        root = file.pdf_catalog()
        if parent:
            value_xref = file.get_new_xref()
            file.update_object(value_xref, f"<</FT /Tx /T (review_group) /Kids [{xref} 0 R] /V {pymupdf.get_pdf_str(saved)}>>")
            file.xref_set_key(xref, "Parent", f"{value_xref} 0 R")
            file.xref_set_key(xref, "V", "null")
            file.xref_set_key(root, "AcroForm/Fields", f"[{value_xref} 0 R]")
        else:
            file.xref_set_key(xref, "V", pymupdf.get_pdf_str(saved))
        if variant == "hidden":
            file.xref_set_key(xref, "F", "6")
        if variant == "needs_appearance":
            file.xref_set_key(root, "AcroForm/NeedAppearances", "true")
        if variant == "scripted":
            file.xref_set_key(xref, "AA", "<</F <</S /JavaScript /JS (event.value = event.value;)>>>>")
        if variant == "invisible":
            appearance = int(file.xref_get_key(xref, "AP/N")[1].split()[0])
            saved_paint = file.xref_stream(appearance)
            assert b"0 0 0 rg" in saved_paint
            file.update_stream(appearance, saved_paint.replace(b"0 0 0 rg", b"1 1 1 rg"))
        if variant == "large_appearance":
            appearance = int(file.xref_get_key(xref, "AP/N")[1].split()[0])
            file.update_stream(appearance, file.xref_stream(appearance) + b"\n" + b" " * 65536, compress=True)
            assert file.xref_get_key(appearance, "Filter") == ("name", "/FlateDecode")
        if variant == "flattened":
            file.bake(annots=False, widgets=True)
        file.save(path, no_new_id=True)
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("parent", [False, True])
def test_pdf_form_keeps_two_values_and_verifiable_crop(tmp_path, parent):
    source = tmp_path / "static.pdf"
    digest = make_pdf(source, parent=parent)
    record, crops = reader.inspect_within(source, source_sha256=digest, timeout=10)
    assert record["affects_overall_confidence"] is False
    [row] = record["rows"]
    assert row["page"] == 1 and row["box"] == BOX
    assert (row["saved_value"], row["display_value"]) == (OTHER, DISPLAY)
    crop = row["crop"]
    payload = crops["field_1"]
    reader.verify_crop(crop, payload)
    asset = {key: crop[key] for key in ("sha256", "width", "height")} | {"size_bytes": len(payload), "media_type": "image/png"}
    expected = canonical_hash(record)
    document = {"source": {"sha256": digest, "method": "pymupdf", binding.COMMITMENT: expected},
                binding.KEY: {"inspection": record, "artifacts": {"field_1": asset}}}
    binding.validate(document, expected)
    bundle = tmp_path / "source.zip"
    with ZipFile(bundle, "w") as archive:
        archive.writestr(f"assets/{crop['sha256']}.png", payload)
    with ZipFile(bundle) as archive:
        reader.verify_original(document, archive, source)
        altered = copy.deepcopy(document)
        altered[binding.KEY]["inspection"]["rows"][0]["saved_value"] = "West Parcel Group"
        with pytest.raises(ValueError, match="original raster coordinates differ"):
            reader.verify_original(altered, archive, source)
    with pytest.raises(ValueError, match="require their raster evidence"):
        binding.validate({**document, binding.KEY: {"inspection": record, "artifacts": {}}}, expected)
    binding.validate({"source": {"method": "pymupdf"}}, None)


@pytest.mark.parametrize("variant", ["same", "equivalent", "formatted", "hidden", "needs_appearance", "scripted", "invisible", "flattened",
                                     "large_appearance"])
def test_pdf_form_does_not_report_formatted_or_unsupported_values(tmp_path, variant):
    source = tmp_path / "static.pdf"
    digest = make_pdf(source, variant=variant)
    assert reader.capture(source, source_sha256=digest) == (None, {})
