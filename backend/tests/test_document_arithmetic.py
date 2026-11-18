from copy import deepcopy

import pytest

from app.detectors.text.extract import extract
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
