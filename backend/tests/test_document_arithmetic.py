from copy import deepcopy

import pytest

from app.detectors.text.extract import extract, from_string
from app.evidence import document_arithmetic as check


def test_arithmetic_preserves_decimal_precision_and_exact_original_positions(tmp_path):
    raw = ("A budget entry claims ($1,250.50 + $375.25 = $1,700).\n\n"
           "0.1 + 0.2 = 0.3\n\n"
           "999999999999.999999 × 999999999999.999999 = 0\n\n"
           "−32 - 39 = -70\n")
    path = tmp_path / "equations.txt"
    path.write_text(raw, encoding="utf-8")
    doc = extract(path, ".txt")
    source = {"method": doc.meta["method"], "chars": len(doc.text)}
    record = check.build(source, doc.text, doc.source_map)
    assert record["checked_equations"] == 4 and record["disagreements"] == 3
    expected = ["$1,625.75", "999999999999999998000000.000000000001", "-71"]
    assert [row["calculated"] for row in record["locations"]] == expected
    for row in record["locations"]:
        assert doc.text[row["start"]:row["end"]] == row["text"]
        assert "".join(raw[lo:hi] for lo, hi in row["location"]["ranges"]) == row["text"]
    result = {"source": source, check.KEY: record}
    check.validate(result)
    check.verify_source(result, doc.text, doc.source_map)
    altered = deepcopy(result)
    # A plausible coordinate in the same block still fails against the actual source map.
    altered[check.KEY]["locations"][0]["location"]["ranges"][0][0] -= 1
    check.validate(altered)
    with pytest.raises(ValueError, match="differs from the verified source"):
        check.verify_source(altered, doc.text, doc.source_map)


def test_incomplete_approximate_and_quoted_equations_are_outside_scope(tmp_path):
    path = tmp_path / "reply.eml"
    path.write_text("Subject: Worksheet\nContent-Type: text/plain; charset=UTF-8\n\n"
                    "1 + 2 = 3 + 0\n\n1 + 2 ≈ 4\n\n€1 + $2 = $3\n\n"
                    "1 / 3 = 0.33\n\n(1 + 2 = 3)\n\n> (4 + 4 = 7)\n", encoding="utf-8")
    doc = extract(path, ".eml")
    record = check.build({"method": doc.meta["method"]}, doc.text, doc.source_map)
    assert record["checked_equations"] == 1 and record["disagreements"] == 0 and record["locations"] == []


def test_length_conversion_uses_result_precision_and_preserves_legacy_replay():
    raw = ("An aerospace document gives an incorrect span (12 in = 35 cm).\n\n"
           "1 metre equals 39.37 inches\n\n1 in = 3 cm\n\n1 ft = 0.3048 m\n\n"
           "The budget includes (2 + 2 = 6).")
    doc = from_string(raw)
    source = {"method": "plain", "chars": len(doc.text)}
    legacy = {"source": source, check.KEY: check.build(source, doc.text, doc.source_map)}
    current = {"source": source, check.KEY: check.build(source, doc.text, doc.source_map, include_lengths=True)}
    for record in (legacy, current):
        check.validate(record)
        check.verify_source(record, doc.text, doc.source_map)
    assert legacy[check.KEY]["schema"] == check.SCHEMA
    assert legacy[check.KEY]["checked_equations"] == 1
    reading = current[check.KEY]
    assert reading["schema"] == check.LENGTH_SCHEMA and reading["checked_equations"] == 5
    assert reading["length_conversions_checked"] == 4 and reading["length_conversion_disagreements"] == 1
    assert reading["affects_overall_confidence"] is False and reading["disagreements"] == 2
    first, second = reading["locations"]
    assert first["stated"] == "35 cm" and first["calculated"] == "30 cm"
    assert second["text"] == "2 + 2 = 6"
    assert "".join(raw[a:b] for a, b in first["location"]["ranges"]) == first["text"]
    altered = deepcopy(current)
    altered[check.KEY]["locations"][0]["calculated"] = "31 cm"
    with pytest.raises(ValueError, match="disagreement differs"):
        check.validate(altered)


def test_length_conversion_leaves_qualified_and_historical_units_outside_scope():
    doc = from_string("The notebook uses survey feet.\n\n"
                      "1 ft = 0.304800609601 m\n\n1 mile = 1852 m\n\n"
                      "A nominal length (12 in = 30 cm).\n\nThe length may be around (1 m = 1100 mm).\n\n"
                      "1 in ≈ 2.5 cm\n\n1 square inch = 6 cm\n\n1 inch = 1 in\n\n"
                      "1 metre = 1,000 mm")
    result = check.build({"method": "plain"}, doc.text, doc.source_map, include_lengths=True)
    assert result["length_conversions_checked"] == 1
    assert result["disagreements"] == 0
