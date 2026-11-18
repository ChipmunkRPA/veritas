from __future__ import annotations

from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from test_billing_preflight import WORDS, verify

from app.billing.preflight import PreflightError
from app.detectors.text.extract import extract
from app.evidence.document_native_locations import clip_source_ranges

PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"
SHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
LINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WORKBOOK = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"


def workbook(path, rows, *, hidden=False, strict=False, external=False, workbook_type=WORKBOOK):
    ns = "http://purl.oclc.org/ooxml/spreadsheetml/main" if strict else SHEET
    link = "http://purl.oclc.org/ooxml/officeDocument/relationships" if strict else LINK
    state = "hidden" if hidden else "visible"
    target = 'Target="https://example.invalid/cells.xml" TargetMode="External"' if external else 'Target="cells.xml"'
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("_rels/.rels", f'<Relationships xmlns="{PKG}"><Relationship Id="book" Type="{link}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("[Content_Types].xml", f'<Types xmlns="{CONTENT}"><Override PartName="/xl/workbook.xml" ContentType="{workbook_type}"/></Types>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{ns}" xmlns:r="{link}"><sheets><sheet name="Notes" state="{state}" sheetId="1" r:id="sheet"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{PKG}"><Relationship Id="sheet" Type="{link}/worksheet" {target}/><Relationship Id="strings" Type="{link}/sharedStrings" Target="strings.xml"/></Relationships>')
        archive.writestr("xl/cells.xml", f'<worksheet xmlns="{ns}"><cols><col min="3" max="3" hidden="true"/></cols><sheetData>{rows}</sheetData></worksheet>')
        archive.writestr("xl/strings.xml", f'<sst xmlns="{ns}"><si><r><t xml:space="preserve">  {WORDS} </t></r><r><t>_xD83D__xDE00_</t></r><rPh><t>Excluded pronunciation guide</t></rPh></si></sst>')
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
    assert clip_source_ranges(first["source_ranges"], first["start"], first["end"]) == [[2, len(WORDS) + 4]]


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
