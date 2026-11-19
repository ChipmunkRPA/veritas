"""Recount the printed euro summary evaluation without documents or models."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-euro-total-evaluation/1"
SPLITS = {"development": 11, "held": 29, "other": 23}
OUTCOMES = {"match", "mismatch", "not_measured", "not_confirmed"}
CONDITIONS = {"correct", "subtotal_error", "total_error"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recount(receipt):
    require(receipt["schema"] == SCHEMA, "Unexpected evaluation schema")
    signatures = receipt["commitments"]
    require(
        set(signatures) == {"publisher_pin", "source_plan", "operating_rule", "local_reader_replay"}
        and re.fullmatch(r"[a-f0-9]{40}", signatures["publisher_pin"])
        and all(re.fullmatch(r"[a-f0-9]{64}", signatures[k]) for k in set(signatures) - {"publisher_pin"}),
        "Invalid evaluation commitments",
    )
    frames = receipt["cases"]
    require(len(frames) == sum(SPLITS.values()), "Incomplete number of scored files")
    seen = defaultdict(set)
    result = defaultdict(Counter)
    for row in frames:
        split, index = row["split"], row["case"]
        require(
            split in SPLITS and type(index) is int and index not in seen[split], "Invalid or repeated case"
        )
        seen[split].add(index)
        condition = row["condition"]
        candidate = row["candidate"]
        baseline = row["baseline"]
        require(
            condition in CONDITIONS
            and candidate in OUTCOMES
            and baseline in {"match", "mismatch", "not_supported"},
            "Unexpected method outcome",
        )
        examined = row["fields"]
        if candidate in {"match", "mismatch"}:
            roles = [field["role"] for field in examined]
            require(
                len(roles) >= 2
                and roles[0] == "subtotal"
                and roles[-1] == "total"
                and roles[1:-1] in ([], ["tax"], ["discount"], ["tax", "discount"]),
                "Invalid supported terms",
            )
        for field in examined:
            require(
                type(field["correct_amount"]) is bool
                and type(field["horizontal_bp"]) is int
                and 0 <= field["horizontal_bp"] <= 10000,
                "Invalid crop comparison",
            )
        cohort = result[(split, "ordinary" if condition == "correct" else "inconsistent")]
        cohort["examples"] += 1
        cohort["baseline_mismatches"] += baseline == "mismatch"
        cohort["candidate_mismatches"] += candidate == "mismatch"
        cohort["candidate_unassessed"] += candidate in {"not_measured", "not_confirmed"}
        cohort["exact_crop_mismatches"] += candidate == "mismatch" and all(
            field["correct_amount"] and field["horizontal_bp"] >= 9000 for field in examined
        )
    require(all(seen[split] == set(range(count)) for split, count in SPLITS.items()), "Incomplete split")
    return {split: {arm: result[(split, arm)] for arm in ("ordinary", "inconsistent")} for split in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    result = recount(receipt)
    require(result == receipt["summary"], "Recount differs from the published summary")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
