# SPDX-License-Identifier: Apache-2.0
"""Validate source positions and decimal arithmetic against held human-worked examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import re
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from app.detectors.text.extract import _join_mapped, _text_parts
from app.evidence import document_arithmetic

REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
DATA_SHA = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
LICENSE_SHA = "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b"
ANNOTATIONS = re.compile(r"<<([^<>]+)>>")
TERM = r"(-?[0-9]{1,8}(?:\.[0-9]{1,4})?)"
ELIGIBLE = re.compile(rf" *{TERM} *([+*-]) *{TERM} *= *{TERM} *")
OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
DISPLAY = {"+": "+", "-": "−", "*": "×"}


def readout(text):
    doc = _join_mapped(_text_parts(text))
    source = {"method": "plain", "chars": len(doc.text)}
    result = document_arithmetic.build(source, doc.text, doc.source_map)
    document_arithmetic.validate({"source": source, document_arithmetic.KEY: result})
    return result, doc.text


def compare(text, expression, expected, *, covered):
    result, extracted = readout(text)
    disagreement = expected is not None
    if (result["checked_equations"] != int(covered) or result["disagreements"] != int(disagreement)
            or result["affects_overall_confidence"]):
        raise ValueError("Arithmetic coverage or disagreement differs from the independent calculation")
    if disagreement:
        [row] = result["locations"]
        [span] = row["location"]["ranges"]
        native = text[span[0]:span[1]]
        if (native != expression or extracted[row["start"]:row["end"]] != expression
                or Fraction(row["calculated"]) != expected):
            raise ValueError("Arithmetic disagreement has a wrong result or location")


def measure(dataset, terms):
    content = dataset.read_bytes()
    if hashlib.sha256(content).hexdigest() != DATA_SHA or hashlib.sha256(terms.read_bytes()).hexdigest() != LICENSE_SHA:
        raise ValueError("The test split must have its pinned publisher bytes and MIT license")
    counts, operations = Counter(), Counter()
    for item in content.splitlines():
        counts["source_problems"] += 1
        for annotation in ANNOTATIONS.findall(json.loads(item)["answer"]):
            counts["annotations"] += 1
            matched = ELIGIBLE.fullmatch(annotation)
            if not matched:
                continue
            left, op, right, stated = matched.groups()
            expected = OPERATORS[op](Fraction(left), Fraction(right))
            flawed = Fraction(stated) != expected
            compare(annotation, annotation.strip(), expected if flawed else None, covered=True)
            counts["originals_checked"] += 1
            operations[op] += 1
            if flawed:
                counts["original_disagreements"] += 1
                continue
            varied = f"{left} {DISPLAY[op]} {right}"
            changed = f"{varied} = {Decimal(stated) + 1:f}"
            restyled = f"{varied} = {stated}"
            compare(f"Worked example ({changed}).", changed, expected, covered=True)
            compare(f"Worked example ({restyled}).", restyled, None, covered=True)
            compare(f"Worked example ({varied} ≈ {stated}).", None, None, covered=False)
            counts.update(("correctly_located_edits", "clean_restylings", "unassessed_approximations"))
    if counts["source_problems"] != 1319 or counts["originals_checked"] == 0:
        raise ValueError("The declared publisher test split did not reproduce")
    counts.setdefault("original_disagreements", 0)
    return {"schema": "chipmunk.document-arithmetic-held/1", "revision": REVISION, "data_sha256": DATA_SHA,
            "license_sha256": LICENSE_SHA, "implementation_sha256": hashlib.sha256(Path(document_arithmetic.__file__).read_bytes()).hexdigest(),
            "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "counts": counts,
            "operations": operations, "restyled_flags": 0, "authorship_weight": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("dataset", "terms", "output"):
        parser.add_argument(arg, type=Path)
    options = parser.parse_args()
    receipt = measure(options.dataset, options.terms)
    with options.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt["counts"], sort_keys=True))
