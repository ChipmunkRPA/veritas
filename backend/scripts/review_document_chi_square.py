# SPDX-License-Identifier: Apache-2.0
"""Recalculate the public chi-square document panel using standard Python."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path


def displayed_bounds(token):
    value = Decimal(token)
    step = Decimal(1).scaleb(value.as_tuple().exponent)
    return float(max(0, value - step)), float(value + step)


def right_tail(statistic, degrees):
    half = statistic / 2
    decay = math.exp(-half)
    if degrees % 2:
        first = 1
        probability = math.erfc(math.sqrt(half))
        term = 2 * math.sqrt(half / math.pi) * decay
    else:
        first = 2
        probability = decay
        term = half * decay
    for degree in range(first, degrees, 2):
        probability += term
        term *= 2 * half / (degree + 2)
    return min(1, probability)


def disagrees(comparison, printed, low, high):
    tolerance = 1e-10
    if comparison in {"<", "≤"}:
        return low > float(printed) + tolerance
    if comparison in {">", "≥"}:
        return high < float(printed) - tolerance
    if comparison != "=":
        raise ValueError("Unknown comparison")
    lower, upper = displayed_bounds(printed)
    return low > upper + tolerance or high < lower - tolerance


def review(panel):
    counts = Counter()
    admitted = {}
    works = set()
    for source in panel["sources"]:
        counts[source["status"]] += 1
        for record in source["rows"]:
            start, end = record["probability_start"], record["probability_end"]
            if not (record["start"] <= start < end <= record["end"]
                    and end - start == len(record["printed"]) and 1 <= record["page"] <= source["pages"]):
                raise ValueError("Inconsistent source character or page range")
            lower, upper = displayed_bounds(record["statistic"])
            low, high = right_tail(upper, record["df"]), right_tail(lower, record["df"])
            if max(abs(low - record["tail_min"]), abs(high - record["tail_max"])) > 2e-11:
                raise ValueError("Calculated probability differs from the receipt")
            issue = disagrees(record["relation"], record["printed"], low, high)
            for interpretation_key, disagreement_key in (("interpretation", "disagreement"),
                                                           ("locally_named_interpretation", "locally_named_disagreement")):
                included = record[interpretation_key] == "checked"
                if record[disagreement_key] != (included and issue):
                    raise ValueError("Decision differs from the receipt")
            counts["matched_notation"] += 1
            counts["locally_named"] += record["locally_named_interpretation"] == "checked"
            if record["interpretation"] == "checked":
                works.add(source["id"])
                counts["admitted"] += 1
                counts["disagreements"] += issue
                admitted[source["id"], record["start"]] = (record, low, high)
    counts["admitted_works"] = len(works)
    for row in panel["staged_controls"]:
        _, low, high = admitted[row["doi"], row["statement_source_start"]]
        if row["label"] == "altered_probability":
            printed = ".990" if (low + high) / 2 < .5 else ".010"
            expected = disagrees("=", printed, low, high)
        else:
            expected = False
        if expected != row["disagreement"]:
            raise ValueError("Staged decision differs from the receipt")
        counts[row["label"]] += 1
        counts[f"{row['label']}_disagreements"] += expected
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as source:
        receipt = json.load(source)
    if receipt["schema"] != "chipmunk.document-chi-square-evaluation/1":
        raise ValueError("Expected the chi-square evaluation receipt")
    print(json.dumps({name: review(panel) for name, panel in receipt["panels"].items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
