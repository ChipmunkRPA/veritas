from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from test_billing_preflight import WORDS, verify
from test_document_native_locations import _snapshot
from test_document_packets import packet_factory as packet_factory

from app.billing.preflight import PreflightError
from app.core.canonical import canonical_json
from app.detectors.text.extract import extract
from app.evidence import document_native_locations as locations
from app.evidence.document_packets import build_packet_set
from app.reports import engine, multimedia_evidence, public_package, public_verify, verify_document_source

PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"
SHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
LINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WORKBOOK = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"


def workbook(path, rows, *, hidden=False, strict=False, external=False, workbook_type=WORKBOOK, style_codes=()):
    ns = "http://purl.oclc.org/ooxml/spreadsheetml/main" if strict else SHEET
    link = "http://purl.oclc.org/ooxml/officeDocument/relationships" if strict else LINK
    state = "hidden" if hidden else "visible"
    target = 'Target="https://example.invalid/cells.xml" TargetMode="External"' if external else 'Target="cells.xml"'
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("_rels/.rels", f'<Relationships xmlns="{PKG}"><Relationship Id="book" Type="{link}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("[Content_Types].xml", f'<Types xmlns="{CONTENT}"><Override PartName="/xl/workbook.xml" ContentType="{workbook_type}"/></Types>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="{link}"><sheets><sheet name="Notes" state="{state}" sheetId="1" r:id="sheet"/></sheets></workbook>')
        styles = f'<Relationship Id="styles" Type="{link}/styles" Target="formats.xml"/>' if style_codes else ""
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{PKG}"><Relationship Id="sheet" Type="{link}/worksheet" {target}/><Relationship Id="strings" Type="{link}/sharedStrings" Target="strings.xml"/>{styles}</Relationships>')
        archive.writestr("xl/cells.xml", f'<worksheet xmlns="{ns}"><cols><col min="3" max="3" hidden="true"/></cols><sheetData>{rows}</sheetData></worksheet>')
        archive.writestr("xl/strings.xml", f'<sst xmlns="{ns}"><si><r><t xml:space="preserve">  {WORDS} </t></r><r><t>_xD83D__xDE00_</t></r><rPh><t>Excluded pronunciation guide</t></rPh></si></sst>')
        if style_codes:
            formats = "".join(f'<numFmt numFmtId="{index}" formatCode="{code}"/>'
                              for index, code in enumerate(style_codes, 164))
            styles = '<xf numFmtId="0"/>' + "".join(f'<xf numFmtId="{index}" applyNumberFormat="1"/>'
                                                    for index in range(164, 164 + len(style_codes)))
            archive.writestr("xl/formats.xml", f'<styleSheet xmlns="{ns}"><numFmts>{formats}</numFmts><cellXfs>{styles}</cellXfs></styleSheet>')
    return path


@pytest.mark.parametrize("strict", [False, True])
def test_excel_extracts_literal_cells_with_native_positions(tmp_path, strict):
    source = workbook(tmp_path / "notes.XLSX", f'''
        <row r="3"><c r="B3" t="s"><v>0</v></c>
            <c r="C3" t="inlineStr"><is><t>{WORDS}</t></is></c>
            <c r="D3" t="s"><f>"formula"</f><v>0</v></c><c r="E3"><v>999</v></c></row>
        <row r="4" hidden="1"><c r="B4" t="s"><v>0</v></c></row>
        <row><c r="E6" t="inlineStr"><is><t>Later  entry</t></is></c></row>
        ''', strict=strict)
    verify(source.read_bytes(), source.name, "document")
    document = extract(source, ".xlsx")
    assert document.text == WORDS + " 😀\n\nLater entry"
    assert [(block["part_index"], block["row"], block["column"]) for block in document.source_map] == [(0, 3, 2), (0, 6, 5)]
    first = document.source_map[0]
    assert locations.clip_source_ranges(first["source_ranges"], first["start"], first["end"]) == [[2, len(WORDS) + 4]]


def test_excel_checkout_rejects_hidden_only_and_invalid_workbooks(tmp_path):
    row = '<row r="2"><c r="B2" t="s"><v>0</v></c></row>'
    source = tmp_path / "empty.xlsx"
    with pytest.raises(PreflightError, match="at least 15 readable words in supported text cells"):
        verify(workbook(source, row, hidden=True).read_bytes(), source.name, "document")
    with pytest.raises(PreflightError, match="internal relationship"):
        verify(workbook(source, row, external=True).read_bytes(), source.name, "document")
    with pytest.raises(PreflightError, match="macro-free Excel XLSX"):
        unsafe = "application/vnd.ms-excel.sheet.macroEnabled.main+xml"
        verify(workbook(source, row, workbook_type=unsafe).read_bytes(), source.name, "document")
    row = row.replace("<v>0", "<v>90")
    with pytest.raises(PreflightError, match="outside its range"):
        verify(workbook(source, row).read_bytes(), source.name, "document")


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("cell", ['t="s"><v>0</v>', f't="inlineStr"><is><t>{WORDS}</t></is>'])
def test_excel_checkout_excludes_cells_with_direct_hide_value_format(tmp_path, strict, cell):
    row = f'<row r="2"><c r="B2" s="1" {cell}</c></row>'
    path = tmp_path / "display.xlsx"
    workbook(path, row, strict=strict, style_codes=(";;;",))
    assert extract(path, ".xlsx").text == ""
    assert WORDS in extract(path, ".xlsx", skip_hidden_xlsx_cell_formats=False).text
    with pytest.raises(PreflightError, match="at least 15 readable words.*No payment was started"):
        verify(path.read_bytes(), path.name, "document")

    workbook(path, row, strict=strict, style_codes=("0;-0;0;@",))
    verify(path.read_bytes(), path.name, "document")
    assert [(part["row"], part["column"]) for part in extract(path, ".xlsx").source_map] == [(2, 2)]


@pytest.mark.parametrize("current", [False, True])
def test_excel_report_and_package_replay_their_saved_cell_visibility_policy(tmp_path, packet_factory, monkeypatch, current):
    visible = ("First sentence has enough visible text to read. The next sentence stays in the same Excel cell. "
               "The last sentence confirms the location on the worksheet.")
    source = workbook(tmp_path / "visibility.xlsx", f'''
        <row r="2"><c r="B2" s="1" t="s"><v>0</v></c></row>
        <row r="5"><c r="B5" s="2" t="inlineStr"><is><t>{visible}</t></is></c></row>''',
        style_codes=(";;;", "0;-0;0;@"))
    db, store, analysis, result, decoded, _ = packet_factory(
        ".xlsx", source_pdf=source, skip_hidden_xlsx_cell_formats=current,
    )
    evidence = build_packet_set(db, analysis, store, result)
    key = locations.XLSX_VISIBILITY_KEY
    assert (key in evidence["coverage"]) is current
    assert (WORDS not in decoded["text"]) is current
    snapshot = _snapshot(evidence, analysis, result)
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    export = public_package._evidence(snapshot)
    assert (key in export["coverage"]) is current
    if current:
        assert WORDS not in str(export)
        (packet,) = export["packets"]
        native = export["coverage"][locations.KEY]["packets"][packet["id"]]
        assert native
        for row in native:
            anchor = packet["anchors"][row["anchor_index"]]
            assert row["coordinate"] == "xlsx_cell" and (anchor["row"], anchor["column"]) == (5, 2)
            excerpt = decoded["text"][anchor["selected_start"]:anchor["selected_end"]]
            assert "".join(visible[start:end] for start, end in row["ranges"]) == excerpt
        rendered = engine.render_pdf(snapshot)
        pdf = " ".join(engine.pdf_text(rendered.pdf).split())
        assert not rendered.refused and WORDS not in pdf and "Saved Excel cell characters" in pdf
        assert "directly applied hide-value format" in pdf

    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    archive_path = tmp_path / "proof.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(export))
    with ZipFile(archive_path) as archive:
        assert "Verified decoded original text locations" in verify_document_source.check(archive, bundle, source)
    if current:
        del export["coverage"][key]
    else:
        export["coverage"][key] = locations.XLSX_VISIBILITY
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(export))
    with ZipFile(archive_path) as archive, pytest.raises(ValueError, match="source mapping differs"):
        verify_document_source.check(archive, bundle, source)
