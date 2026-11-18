from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest
from test_billing_preflight import WORDS, verify

from app.billing.preflight import PreflightError
from app.detectors.text.extract import extract
from app.evidence.document_native_locations import clip_source_ranges
from app.storage.files import ODS_MIME


def ods(path, sheets, *, mime=ODS_MIME):
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", mime, compress_type=ZIP_STORED)
        archive.writestr("content.xml", f'''
            <office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
                 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
                 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
                 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
              <office:automatic-styles>
                <style:style style:family="table" style:name="hidden">
                  <style:table-properties table:display="false"/>
                </style:style>
                <style:style style:family="table" style:name="derived" style:parent-style-name="hidden"/>
              </office:automatic-styles>
              <office:body><office:spreadsheet>{sheets}</office:spreadsheet></office:body>
            </office:document-content>''')
    return path


def test_ods_uses_saved_sheet_row_and_column_without_formulas_or_hidden_content(tmp_path):
    ignored = '<table:table-cell office:value-type="string" office:string-value="Ignored words"/>'
    source = ods(tmp_path / "book.ODS", f'''
        <table:table table:style-name="derived"><table:table-row>{ignored}</table:table-row></table:table>
        <table:table>
          <table:table-column table:number-columns-repeated="2"/>
          <table:table-column-group table:display="false"><table:table-column/></table:table-column-group>
          <table:table-header-rows>
            <table:table-row table:number-rows-repeated="2" table:visibility="filter">{ignored}</table:table-row>
          </table:table-header-rows>
          <table:table-row-group>
            <table:table-row table:number-rows-repeated="2">
              <table:table-cell/>
              <table:table-cell office:value-type="string" table:number-columns-repeated="2">
                <text:p>  {WORDS}<text:s text:c="3"/><text:span>😀</text:span></text:p>
                <office:annotation><text:p>Ignored annotation</text:p></office:annotation>
              </table:table-cell>
              <table:covered-table-cell/>
              <table:table-cell office:value-type="string" table:formula="of:=1" office:string-value="Ignored formula"/>
              <table:table-cell office:value-type="string" office:string-value="Other entry"/>
            </table:table-row>
          </table:table-row-group>
        </table:table>''')
    verify(source.read_bytes(), source.name, "document")
    document = extract(source, ".ods")
    assert document.text == "\n\n".join((WORDS + " 😀", "Other entry") * 2)
    assert [(block["part_index"], block["row"], block["column"]) for block in document.source_map] == [
        (1, 3, 2), (1, 3, 6), (1, 4, 2), (1, 4, 6),
    ]
    first = document.source_map[0]
    assert clip_source_ranges(first["source_ranges"], first["start"], first["end"]) == [[2, len(WORDS) + 6]]


def test_ods_rejects_hidden_only_wrong_kind_and_excessive_repetition_before_checkout(tmp_path):
    data = f'<table:table-cell office:value-type="string" office:string-value="{WORDS}"/>'
    source = tmp_path / "book.ods"
    with pytest.raises(PreflightError, match="at least 15 readable words in supported text cells"):
        verify(ods(source, f'<table:table table:style-name="hidden"><table:table-row>{data}</table:table-row></table:table>').read_bytes(),
               source.name, "document")
    sheet = f'<table:table><table:table-row table:number-rows-repeated="1048576">{data}</table:table-row></table:table>'
    with pytest.raises(PreflightError, match="exceeds the writing limit"):
        verify(ods(source, sheet).read_bytes(), source.name, "document")
    with pytest.raises(PreflightError, match="invalid repeated rows range"):
        verify(ods(source, sheet.replace("1048576", "1048577")).read_bytes(), source.name, "document")
    with pytest.raises(PreflightError, match="not an OpenDocument Spreadsheet"):
        verify(ods(source, sheet, mime="application/zip").read_bytes(), source.name, "document")
