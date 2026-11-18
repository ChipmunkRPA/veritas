# SPDX-License-Identifier: Apache-2.0
"""Replay published length conversion factors and controlled digit changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

SOURCE = "https://www.nist.gov/system/files/documents/2025/12/23/2026_NIST_HB133_AppendixE-20251222.pdf"
SOURCE_SHA = "5887c575df2a35d16af0de1c40d901dcedadafa98e8a7ce11cc89cee60d02c58"
PAGE = 9
UNITS = ("in", "ft", "yd", "mi", "cm", "m")
WORDS = ("inches", "feet", "yards", "miles", "centimeters", "metres")
ROW_TOP = (506.47, 523.75, 541.03, 558.31, 579.93, 601.53)
COLUMNS = (144, 202, 270, 340, 426, 493, 550)
SCHEMA = "chipmunk.document-length-reference/1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def published_table(source):
    import pymupdf

    if digest(source) != SOURCE_SHA:
        raise ValueError("The official NIST appendix differs from the held source")
    with pymupdf.open(source) as doc:
        page = doc[PAGE - 1]
        if "Multiply by the Ending Unit Conversion Factor" not in page.get_text():
            raise ValueError("The selected page does not contain the published length table")
        words = page.get_text("words")
    for y in ROW_TOP:
        values = []
        for a, b in zip(COLUMNS, COLUMNS[1:]):
            group = sorted((x, value) for x, top, _, _, value, *_ in words if a <= x < b and abs(top - y) < .25)
            joined = re.sub(r"\s", "", "".join(value for _, value in group))
            if not re.fullmatch(r"\d+(?:\.\d+)?", joined):
                raise ValueError("A table cell did not yield one exact printed factor")
            values.append(joined)
        yield values


def prepare(source, output):
    cases = []
    for i, row in enumerate(published_table(source)):
        if row[i] != "1":
            raise ValueError("The published table axes do not agree")
        for j, value in enumerate(row):
            if i == j:
                continue
            # Move two least-significant units above the authoritative printed value.
            wrong = format(Decimal(value) + 2 * Decimal(1).scaleb(Decimal(value).as_tuple().exponent), "f")
            original = f"1 {UNITS[i]} = {value} {UNITS[j]}"
            changed = f"1 {UNITS[i]} = {wrong} {UNITS[j]}"
            restyled = f"One length relationship is (1 {UNITS[i]} equals {value} {WORDS[j]})."
            approximate = f"The drawing notes an approximate dimension ({changed})."
            cases.append({"table_cell": [i + 1, j + 1], "original": original, "restyled": restyled,
                          "changed": changed, "approximate": approximate, "printed": value, "perturbed": wrong})
    if len(cases) != 30:
        raise ValueError("Every off-diagonal factor must appear once")
    plan = {"source_url": SOURCE, "source_sha256": SOURCE_SHA, "page": PAGE, "schema": SCHEMA, "cases": cases}
    with output.open("x") as stream:
        json.dump(plan, stream, sort_keys=True, indent=2)
        stream.write("\n")


def measure(plan, output):
    from app.detectors.text.extract import from_string
    from app.evidence import document_arithmetic as check
    from app.evidence import document_length

    fixed = json.loads(plan.read_text())
    if (fixed["schema"], fixed["source_sha256"], fixed["page"]) != (SCHEMA, SOURCE_SHA, PAGE) or len(fixed["cases"]) != 30:
        raise ValueError("The conversion table plan is incomplete")
    # Independent inch definitions in centimetres, with all derived units as rational numbers.
    inch = Fraction(127, 50)
    cm_per_unit = (inch, 12 * inch, 36 * inch, 63360 * inch, 1, 100)
    summaries = []
    counts = Counter()
    for case in fixed["cases"]:
        i, j = (cell - 1 for cell in case["table_cell"])
        expected = cm_per_unit[i] / cm_per_unit[j]
        printed = Fraction(case["printed"])
        resolution = 10 ** len(case["printed"].partition(".")[2])
        if abs(expected - printed) * resolution > Fraction(1, 2):
            raise ValueError("The printed value differs from independent rational length arithmetic")
        observations = {}
        for role in ("original", "restyled", "changed", "approximate"):
            raw = case[role]
            doc = from_string(raw)
            source = {"method": "plain", "chars": len(doc.text)}
            result = check.build(source, doc.text, doc.source_map, include_lengths=True)
            evidence = {"source": source, check.KEY: result}
            check.validate(evidence)
            check.verify_source(evidence, doc.text, doc.source_map)
            found = result["locations"]
            checked = result.get("length_conversions_checked", 0)
            if checked != (role != "approximate") or len(found) != (role == "changed"):
                raise ValueError(f"An official conversion case failed the fixed {role} expectation")
            ranges = []
            if found:
                (row,) = found
                ranges = row["location"]["ranges"]
                if (row["text"] != raw or "".join(raw[a:b] for a, b in ranges) != raw
                        or abs(Fraction(row["calculated"].split()[0]) - expected) * resolution > Fraction(1, 2)):
                    raise ValueError("A disagreement or its exact source position did not reproduce")
            counts.update({f"{role}_checked": checked, f"{role}_disagreements": len(found)})
            observations[role] = {"text_sha256": hashlib.sha256(raw.encode()).hexdigest(), "checked": checked,
                                  "disagreements": len(found), "source_ranges": ranges}
        summaries.append({"cell": case["table_cell"], "printed": case["printed"], "perturbed": case["perturbed"],
                          "observations": observations})
    receipt = {"schema": SCHEMA, "source_url": SOURCE, "source_sha256": SOURCE_SHA, "page": PAGE,
               "input_plan_sha256": digest(plan), "checked_against_independent_fractions": len(fixed["cases"]),
               "implementation_sha256": {name: digest(Path(module.__file__)) for name, module in
                                         (("arithmetic", check), ("length", document_length))},
               "counts": dict(counts), "cells": summaries}
    with output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(dict(counts), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="phase", required=True)
    modes.add_parser("prepare").add_argument("source", type=Path)
    modes.add_parser("measure").add_argument("plan", type=Path)
    for mode in modes.choices.values():
        mode.add_argument("output", type=Path)
    options = vars(parser.parse_args())
    {"prepare": prepare, "measure": measure}[options.pop("phase")](**options)
