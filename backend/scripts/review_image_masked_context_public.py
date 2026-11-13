# SPDX-License-Identifier: Apache-2.0
"""Recompute the published masked-context aggregate with only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from itertools import product
from pathlib import Path

ARMS = ("current_luma", "three_class", "masked_context", "joint_context")
FAMILIES = ("kandinsky", "sd15")
POSITIVE = ("sharp", "feathered")
CONTROLS = ("original", "local_processing", "global_processing", "local_blur", "local_copy", "local_tone")
LIMITS = {"extra_feathered_over_luma": 3, "editor_families": 2, "extra_ordinary_groups": 2, "misplaced": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(path):
    with path.open("rb") as stream:
        body = stream.read(100_001)
    require(0 < len(body) <= 100_000, "Expected a bounded public aggregate")
    result = json.loads(body)
    require(result["schema"] == "chipmunk.image-masked-context-transfer/1" and result["design"]["advance"] == LIMITS,
            "Unexpected masked-context protocol")
    expected = {f"{family}/{role}/{arm}" for family, role, arm in product(FAMILIES, POSITIVE + CONTROLS, ARMS)}
    require(set(result["strata"]) == expected, "Public strata coverage differs")
    totals = {arm: Counter() for arm in ARMS}
    flagged_group_bounds = {arm: [0, 0] for arm in ARMS}
    for family, arm in product(FAMILIES, ARMS):
        ordinary_flags = []
        for role in POSITIVE + CONTROLS:
            row = result["strata"][f"{family}/{role}/{arm}"]
            require(set(row) == {"files", "flags", "useful", "useful_before_point", "misplaced"}
                    and all(type(number) is int and 0 <= number <= 6 for number in row.values())
                    and row["files"] == 6 and row["useful"] <= min(row["flags"], row["useful_before_point"]),
                    "Unexpected stratum measurements")
            if role in POSITIVE:
                require(row["misplaced"] == row["flags"] - row["useful"], "Edited file flag accounting differs")
                totals[arm][f"useful_{role}"] += row["useful"]
                totals[arm]["misplaced"] += row["misplaced"]
                if role == "feathered":
                    totals[arm]["editor_families"] += row["useful"] > 0
            else:
                require(row["useful"] == row["useful_before_point"] == row["misplaced"] == 0, "Ordinary row contains edit truth")
                ordinary_flags.append(row["flags"])
        totals[arm]["ordinary_flags"] += sum(ordinary_flags)
        flagged_group_bounds[arm][0] += max(ordinary_flags)
        flagged_group_bounds[arm][1] += min(6, sum(ordinary_flags))
    require(totals["current_luma"] == Counter(), "Current luma reference changed")
    for arm in ARMS[2:]:
        observed = result["outcome"][arm]
        lower, upper = flagged_group_bounds[arm]
        groups = observed["ordinary_groups"]
        require(type(groups) is int and lower <= groups <= upper, "Flagged source count outside possible public bounds")
        recomputed = {"extra_feathered_over_luma": totals[arm]["useful_feathered"], "editor_families": totals[arm]["editor_families"],
                      "extra_ordinary_groups": groups, "misplaced": totals[arm]["misplaced"]}
        require(all(observed[key] == value for key, value in recomputed.items()), "Advancement inputs differ")
        require(observed["passes"] is (recomputed["extra_feathered_over_luma"] >= LIMITS["extra_feathered_over_luma"]
                                      and recomputed["editor_families"] >= LIMITS["editor_families"]
                                      and groups <= LIMITS["extra_ordinary_groups"] and recomputed["misplaced"] <= LIMITS["misplaced"]),
                "Advancement status differs")
        require(totals["three_class"]["useful_feathered"] + observed["extra_feathered_over_frozen_head"]
                - observed["lost_feathered_vs_frozen_head"] == totals[arm]["useful_feathered"],
                "Frozen-head additions and losses do not reconcile")
    calls = result["counts"]
    require(calls["deliveries"] == 136 and calls["masked_batches"] == 4*calls["visible_batches"]
            and 0 < calls["tiles"] <= 4*calls["visible_batches"], "Forward accounting differs")
    audit = result["audit"]
    require(audit["exact_inputs_reproduced"] == 136 and audit["native_overlap_and_selection_checks"] == 504
            and audit["neural_inference_repeated"] is False, "Private replay counters differ")
    return {"public_strata_checked": len(expected), "receipt_sha256": hashlib.sha256(body).hexdigest(),
            "private_source_and_model_recomputed": False, "totals": totals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
