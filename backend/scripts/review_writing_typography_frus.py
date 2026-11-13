"""Replay the fresh diplomatic-writing transfer at frozen typography points."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from scripts import writing_typography_transfer as study

SCHEMA = "chipmunk.writing-typography-frus/1"
HASH = re.compile(r"[0-9a-f]{64}")


def verify(path, earlier):
    value = study.read(path)
    previous = study.read(earlier)
    study.verify(earlier)
    study.require(value["schema"] == SCHEMA and study.sha(study.serialize(previous["points"])) == value["lock"]["points_sha256"]
                  and previous["freeze"]["runner_sha256"] == value["lock"]["current_runner_sha256"],
                  "Fresh transfer does not bind the original frozen points")
    sources = {row["group"]: row for row in value["source_roster"]}
    study.require(len(sources) == len(value["source_roster"]) == 17
                  and all(row["group"] == f"{row['volume']}/{row['doc_id']}"
                          and HASH.fullmatch(row["source_xml_sha256"])
                          and len(row["coordinates"]) == 2 and all(len(position) == 2 and
                          all(type(i) is int and i >= 0 for i in position) for position in row["coordinates"])
                          for row in sources.values()), "Malformed or duplicated publisher source")
    expected = {"before": 17, "after": 17, "granite": 16, "olmo": 8}
    rows = value["cases"]
    study.require(Counter(row["family"] for row in rows) == expected
                  and len({(row["group"], row["family"]) for row in rows}) == len(rows), "Missing or duplicated holdout case")
    study.require({row["group"] for row in rows if row["family"] == "before"} ==
                  {row["group"] for row in rows if row["family"] == "after"} == sources.keys(),
                  "Genuine source pairing differs")
    for row in rows:
        length = row["characters"]
        positions = row["units"]
        measured = {tuple(unit["span"]) for unit in positions}
        study.require(row["cohort"] == "frus" and row["group"] in sources and len(positions) == 3
                      and all(0 <= start < end <= length for start, end in measured)
                      and (tuple(row["truth"]) in measured if row["family"] in {"granite", "olmo"} else row["truth"] is None),
                      "Missing exact native focus or generator span")
        study.require(all(len(row[view]) == len(positions) and all(HASH.fullmatch(digest) and 0 <= score <= 1
                          for digest, score in row[view]) for view in ("context", *study.VIEWS)), "Invalid local response")
    result = study.result(rows, previous["points"]["points"])
    study.require(result == value["result"], "Fresh decisions do not replay")
    return result["decisions"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("earlier_ledger", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.ledger, args.earlier_ledger), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
