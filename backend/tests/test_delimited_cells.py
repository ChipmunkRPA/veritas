from __future__ import annotations

from zipfile import ZipFile

import pytest
from test_billing_preflight import verify
from test_document_native_locations import _snapshot
from test_document_packets import packet_factory as packet_factory

from app.billing.preflight import PreflightError
from app.core.canonical import canonical_json
from app.detectors.text import delimited_cells
from app.engine.stages import document
from app.evidence import document_native_locations as locations
from app.evidence.document_packets import build_packet_set
from app.reports import engine, multimedia_evidence, public_package, public_verify, verify_document_source


def parse(tmp_path, text, extension=".csv", *, encoding="utf-8", limit=2_000):
    target = tmp_path / ("input" + extension)
    target.write_bytes(text.encode(encoding) if isinstance(text, str) else text)
    parts, _ = delimited_cells.read(target, extension, max_chars=limit)
    return parts


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_maps_quoted_unicode_cells_and_counts_logical_records(tmp_path, encoding):
    parts = parse(tmp_path, 'ID,Summary\r\n1,"  Café, Kit said ""ready"".\r\nCall again.  "\r\n\r\n2,Next review.\r\n',
                  encoding=encoding)
    assert [value for value, _ in parts] == ["ID", "Summary", '  Café, Kit said "ready".\r\nCall again.  ', "Next review."]
    assert [anchor for _, anchor in parts] == [
        {"kind": "markup", "paragraph_index": i, "part_kind": "delimited_cell", "row": r, "column": c}
        for i, (r, c) in enumerate([(1, 1), (1, 2), (2, 2), (4, 2)])
    ]


def test_tsv_preserves_tabs_inside_a_cell_and_skips_formula_like_fields(tmp_path):
    parts = parse(tmp_path, '=SUM(A1)\t"Writing\twith an internal tab."\n+COUNT(A2)\t@TOTAL(A2)\t- This sentence remains.\n'
                  '-$C$2\t+1+2\tordinary@example.org\thttps://example.org/path\t1.25e3\t+ Labelled note.\n'
                  '"=SUM(A1:A3)"\t42\t@editor recorded this point.\n', ".tsv")
    assert [value for value, _ in parts] == ["Writing\twith an internal tab.", "- This sentence remains.",
                                             "+ Labelled note.", "@editor recorded this point."]
    assert [(anchor["row"], anchor["column"]) for _, anchor in parts] == [(1, 2), (2, 3), (3, 6), (4, 3)]


def test_consistent_excel_semicolon_table_with_decimal_commas(tmp_path):
    parts = parse(tmp_path, 'Title;Estimate;Notes\r\nFirst;1,25;"A comma, and a semicolon; stay here."\r\nSecond;2,5;Another note.\r\n')
    assert [(value, anchor["row"], anchor["column"]) for value, anchor in parts] == [
        ("Title", 1, 1), ("Estimate", 1, 2), ("Notes", 1, 3), ("First", 2, 1),
        ("A comma, and a semicolon; stay here.", 2, 3), ("Second", 3, 1), ("Another note.", 3, 3),
    ]


def test_ambiguous_separators_keep_comma_csv(tmp_path):
    for text, expected in [
        ("Series;Type,Notes\nProject;Current,Writing.\n", ["Series;Type", "Notes", "Project;Current", "Writing."]),
        ("Review the application; the request is ready.\nOpen the archive; confirm the filing date.\n",
         ["Review the application; the request is ready.", "Open the archive; confirm the filing date."]),
    ]:
        assert [value for value, _ in parse(tmp_path, text)] == expected


@pytest.mark.parametrize(("declaration", "separator"), [("sep=;", ";"), ('"sep=|"', "|"), ("SEP=\t", "\t"), ("sep=,", ",")])
def test_csv_separator_declaration_preserves_record_and_field_positions(tmp_path, declaration, separator):
    text = (f'{declaration}\r\nID{separator}Summary\r\n1{separator}"Café, before{separator}and after.\r\n'
            f'New line with ""quotes""."\r\n\r\n2{separator}Keep this final field.')
    path = tmp_path / "declared.csv"
    path.write_bytes(text.encode("utf-16"))
    parts, declared = delimited_cells.read(path, ".csv", max_chars=2_000)
    assert declared == separator
    assert [(value, anchor["row"], anchor["column"]) for value, anchor in parts] == [
        ("ID", 2, 1), ("Summary", 2, 2),
        (f'Café, before{separator}and after.\r\nNew line with "quotes".', 3, 2),
        ("Keep this final field.", 5, 2),
    ]


def test_separator_declaration_applies_only_to_the_first_csv_line(tmp_path):
    assert [value for value, _ in parse(tmp_path, 'Heading\n"sep=;"\nFinal entry.')] == [
        "Heading", "sep=;", "Final entry.",
    ]
    assert [value for value, _ in parse(tmp_path, 'sep=;\tSecond field\n', ".tsv")] == [
        "sep=;", "Second field",
    ]


def test_separator_declaration_does_not_count_toward_checkout_minimum():
    words = "One two three four five six seven eight nine ten eleven twelve thirteen fourteen."
    with pytest.raises(PreflightError, match="at least 15 readable words"):
        verify(f"sep=,\n{words}".encode(), "brief.csv", "document")


def test_csv_resume_reextracts_a_checkpoint_without_the_current_reader(tmp_path, monkeypatch):
    source = tmp_path / "brief.csv"
    source.write_text("Header,Other header\nStored writing,More writing", encoding="utf-8")
    monkeypatch.setattr(document.settings, "text_bibliography_scope_review", False)
    monkeypatch.setattr(document.settings, "text_doi_reference_review", False)
    checkpoint = {"meta": document.decode_document(source, ".csv")[0].meta}
    assert document.current_decode_output(checkpoint)
    del checkpoint["meta"][locations.CSV_SEPARATOR_READER_KEY]
    assert not document.current_decode_output(checkpoint)
    assert document.current_decode_output(checkpoint, require_current_csv_reader=False)


@pytest.mark.parametrize("current", [True, False])
def test_reports_replay_their_saved_csv_separator_policy(tmp_path, packet_factory, monkeypatch, current):
    source = tmp_path / "declared.csv"
    writing = ("First sentence has enough words to inspect. The next sentence stays in the same field. "
               "The final sentence confirms the decoded character locations.")
    source.write_text(f'"sep=;"\n1;{writing}\n', encoding="utf-8")
    verify(source.read_bytes(), source.name, "document")
    db, store, analysis, report, decoded, _ = packet_factory(
        ".csv", source_pdf=source, honor_csv_separator_declaration=current,
    )
    evidence = build_packet_set(db, analysis, store, report)
    snapshot = _snapshot(evidence, analysis, report)
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    export = public_package._evidence(snapshot)
    key = locations.CSV_SEPARATOR_KEY
    assert (key in export["coverage"]) is current
    if current:
        assert export["coverage"][key] == ";"
        assert decoded["text"] == writing
        (packet,) = export["packets"]
        ranges = export["coverage"][locations.KEY]["packets"][packet["id"]]
        assert ranges
        for row in ranges:
            anchor = packet["anchors"][row["anchor_index"]]
            assert row["coordinate"] == "csv_cell" and (anchor["row"], anchor["column"]) == (2, 2)
            assert "".join(writing[a:b] for a, b in row["ranges"]) == writing[anchor["selected_start"]:anchor["selected_end"]]
        rendered = engine.render_pdf(snapshot)
        pdf = " ".join(engine.pdf_text(rendered.pdf).split())
        assert not rendered.refused and "Row 2, column 2" in pdf
        assert "The separator declaration is counted as row 1" in pdf

    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "document"},
              "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes}}
    proof = tmp_path / "proof.zip"
    with ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(export))
    with ZipFile(proof) as archive:
        assert "Verified decoded original text locations" in verify_document_source.check(archive, bundle, source)
    export["coverage"][key] = ","
    with ZipFile(proof, "w") as archive:
        archive.writestr("evidence.json", canonical_json(export))
    with ZipFile(proof) as archive, pytest.raises(ValueError, match="declared CSV separator differs"):
        verify_document_source.check(archive, bundle, source)


@pytest.mark.parametrize("malformed", ['Reference,"unclosed\n', 'Reference,un"quoted\n', 'Reference,"closed"suffix\n',
                                       'Reference,"closed" \n'])
def test_malformed_quotes_are_rejected(tmp_path, malformed):
    with pytest.raises(ValueError, match="invalid quoting"):
        parse(tmp_path, malformed)


def test_bad_unicode_and_binary_tables_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="UTF-8 or UTF-16"):
        parse(tmp_path, b"a,\xff\n")
    with pytest.raises(ValueError, match="binary data"):
        parse(tmp_path, b"a\x00,b\n")


def test_bounds_are_checked_before_returning_partial_text(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Split it into smaller files"):
        parse(tmp_path, "A,B", limit=3)
    with pytest.raises(ValueError, match="Split it into smaller files"):
        parse(tmp_path, "A" * 150_000)
    monkeypatch.setattr(delimited_cells, "MAX_COLUMNS", 2)
    with pytest.raises(ValueError, match="Split it into smaller files"):
        parse(tmp_path, "A,B,C")
    monkeypatch.setattr(delimited_cells, "MAX_ROWS", 2)
    with pytest.raises(ValueError, match="Split it into smaller files"):
        parse(tmp_path, "A\n\nB")
