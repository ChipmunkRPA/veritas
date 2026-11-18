from __future__ import annotations

import pytest

from app.detectors.text import delimited_cells


def parse(tmp_path, text, extension=".csv", *, encoding="utf-8", limit=2_000):
    target = tmp_path / ("input" + extension)
    target.write_bytes(text.encode(encoding) if isinstance(text, str) else text)
    return delimited_cells.read(target, extension, max_chars=limit)


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
