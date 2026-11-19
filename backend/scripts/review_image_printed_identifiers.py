"""Recount anonymous outcomes for the printed screenshot identifier comparison."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROLES = {
    "unaltered", "different_confirmation", "different_entry", "independent_labels", "multiple_records",
    "old_and_new", "local_blur", "confirmed_after_correction",
}
CASES = {"different_confirmation", "different_entry"}
COHORTS = {"development": 6, "held": 16}
CARRIERS = ("png", "jpeg75")


def review(receipt):
    if receipt["schema"] != "chipmunk.image-printed-identifiers-evaluation/1":
        raise ValueError("Unknown screenshot evaluation")
    expected = {(split, carrier, group, role) for split, groups in COHORTS.items() for group in range(groups)
                for carrier in CARRIERS for role in ROLES}
    counts = {f"{split}:{carrier}": Counter() for split in COHORTS for carrier in CARRIERS}
    seen = set()
    exact_groups = {carrier: Counter() for carrier in CARRIERS}
    for row in receipt["decisions"]:
        split, carrier, group, role, flag, values, exact = row
        key = (split, carrier, group, role)
        labeled = role in CASES
        if (key not in expected or key in seen or any(type(value) is not bool for value in (flag, values, exact))
                or (values and not (flag and labeled)) or (exact and not values)):
            raise ValueError("Invalid screenshot outcome or duplicate source role")
        seen.add(key)
        counts[f"{split}:{carrier}"].update({"mismatches": int(labeled), "controls": int(not labeled),
            "matching_values": int(values), "location_hits": int(exact), "ordinary_flags": int(flag and not labeled),
            "wrong_differing_values": int(flag and labeled and not values)})
        if split == "held" and exact:
            exact_groups[carrier][group] += 1
    if seen != expected or sum(row[4] for row in receipt["decisions"]) != receipt["actual_port"]["matched_flags"]:
        raise ValueError("Screenshot outcomes are incomplete or differ from the production replay")
    result = {key: dict(value) for key, value in counts.items()}
    if result != receipt["counts"]:
        raise ValueError("Recorded results differ from independent counts")
    png = counts["held:png"]
    passes = (sum(count == 2 for count in exact_groups["png"].values()) >= 12 and png["ordinary_flags"] <= 1
              and png["wrong_differing_values"] == 0)
    if passes is not receipt["frozen_png_criterion"]:
        raise ValueError("Published continuation criterion differs from the frozen decision")
    return {"cases": len(seen), "held_png": result["held:png"], "held_jpeg75": result["held:jpeg75"],
            "frozen_png_criterion": passes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
