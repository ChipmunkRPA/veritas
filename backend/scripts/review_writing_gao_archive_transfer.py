"""Recount the text-free held GAO source-transfer study."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.writing-gao-archive-transfer/1"
FAMILIES = {"original", "gemini-2.5-flash", "gpt-5.4-nano"}
STATUSES = {"reference_bound", "no_qualified_reference", "ambiguous_reference", "out_of_scope"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _span(value, length, *, empty=False):
    return (isinstance(value, list) and len(value) == 2 and all(type(item) is int for item in value)
            and 0 <= value[0] <= value[1] <= length and (empty or value[0] < value[1]))


def review(path):
    ledger = json.loads(path.read_bytes())
    require(ledger["schema"] == SCHEMA and ledger["catalog_reports"] == 12228, "Unexpected GAO study version or catalog")
    groups = defaultdict(set)
    ids = set()
    summary = defaultdict(Counter)
    for row in ledger["rows"]:
        role = row["family"], row["version"]
        group, identifier = row["group"], row["record_id"]
        require(row["family"] in FAMILIES and row["status"] in STATUSES and role not in groups[group] and identifier not in ids,
                "Repeated or unknown GAO study row")
        ids.add(identifier)
        groups[group].add(role)
        original = row["family"] == "original"
        matched = row["status"] == "reference_bound"
        require(bool(row["reference"]) == matched and (not matched or row["reference"] == row["report"]),
                "Unexpected archive source")
        require(bool(row["changes"]) == (matched and not original) and len(row["sentence_ranges"]) == len(row["overlap"]),
                "Unexpected source coverage")
        length = row["submitted_characters"]
        for submitted, archive in row["changes"]:
            require(_span(submitted, length, empty=True) and _span(archive, row["transcript_characters"], empty=True),
                    "Exact difference falls outside its file")
        for span, (selected, found) in zip(row["sentence_ranges"], row["overlap"], strict=True):
            require(_span(span, length) and type(selected) is int and type(found) is int
                    and 0 <= found <= min(selected, row["truth_characters"]) and 0 < selected <= span[1] - span[0],
                    "Invalid marked sentence overlap")
        truth = row["truth_characters"]
        strict = any(found > 0 and 2 * found >= truth and 10 * found >= 3 * (truth + selected - found)
                     for selected, found in row["overlap"])
        require(row["strict"] == strict, "Strict overlap differs")
        summary[row["family"]].update(files=1, matched=matched, differences=bool(row["changes"]), strict=strict,
                                      abstained=not matched)
    roles = {("original", "v0"), *((family, version) for family in FAMILIES - {"original"} for version in ("v1", "v2", "v4"))}
    require(len(groups) == 20 and all(value == roles for value in groups.values()), "Incomplete held trajectory groups")
    require(ledger["summary"] == summary, "Held totals differ")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.ledger), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
