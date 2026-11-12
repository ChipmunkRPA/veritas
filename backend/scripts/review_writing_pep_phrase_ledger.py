"""Verify the public writing replication's decision arithmetic without model assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("current", "one_context", "stable", "local_contrast", "current_plus_local_contrast")
GENERATED = frozenset({"qwen", "smol2"})
KINDS = GENERATED | {"original", "ordinary"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(public, previous):
    prior = previous.read_bytes()
    record = json.loads(public.read_bytes())
    require(record["schema"] == "chipmunk.writing-pep-phrase-replication/1-audit"
            and record["prior_receipt_sha256"] == hashlib.sha256(prior).hexdigest()
            and record["points"] == json.loads(prior)["operating_points"], "Public operating point binding differs")
    works = {row["group"] for row in record["source_witnesses"]}
    require(len(works) == len(record["source_witnesses"]) == record["held_works"], "Public source roster differs")
    result = record["result"]
    counts, groups = defaultdict(Counter), defaultdict(set)
    case_ids, added = set(), set()
    for row in result["cases"]:
        key, kind = row["group"], row["kind"]
        require(key in works and kind in KINDS and (key, kind) not in case_ids
                and re.fullmatch(r"[0-9a-f]{64}", row["fixture_sha256"]), "Public case binding differs")
        case_ids.add((key, kind))
        counts[kind]["cases"] += 1
        outcomes = row["outcomes"]
        require(set(outcomes) == set(ARMS) and outcomes[ARMS[-1]]["spans"] ==
                outcomes["current"]["spans"] + outcomes["local_contrast"]["spans"], "Public arm ranges differ")
        require(type(row["candidate_count"]) is int and row["candidate_count"] >= 1, "Missing native candidate count")
        for arm in ARMS:
            result_row = outcomes[arm]
            located, possible, spans = result_row["useful"], result_row["geometry_allows_useful"], result_row["spans"]
            require(type(located) is bool and type(possible) is bool and (kind in GENERATED or not (located or possible))
                    and (not located or possible and spans) and len(spans) <= (6 if arm == "current" else 7 if arm == ARMS[-1] else 1),
                    "Public outcome geometry or label differs")
            for region in spans:
                require(region and all(type(start) is type(end) is int and 0 <= start < end for start, end in region),
                        "Invalid selected source range")
            counts[kind].update({f"{arm}_flag": bool(spans), f"{arm}_useful": located, f"{arm}_possible": possible})
            if spans and kind not in GENERATED:
                groups[arm].add(key)
        if outcomes[ARMS[-1]]["useful"] and not outcomes["current"]["useful"]:
            added.add((key, kind))
    require(all((work, kind) in case_ids for work in works for kind in ("original", "ordinary")), "An ordinary control is missing")
    require({name: row["cases"] for name, row in counts.items()} == record["admission"]["holdout"], "Admitted case counts differ")
    extra = len(groups[ARMS[-1]] - groups["current"])
    advance = (len({group for group, _ in added}) >= 2 and {kind for _, kind in added} == GENERATED and extra <= 1
               and all(counts[family]["cases"] >= 6 for family in GENERATED))
    require(dict(counts) == result["counts"] and {arm: len(groups[arm]) for arm in ARMS} == result["false_groups"]
            and sorted(map(list, added)) == result["added_useful_groups"] and extra == result["new_false_groups"]
            and advance == result["advance"], "Public aggregate or advancement differs")
    return {"cases": len(case_ids), "decisions": len(case_ids) * len(ARMS), "works": len(works), "advance": advance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.public, args.previous), sort_keys=True))


if __name__ == "__main__":
    main()
