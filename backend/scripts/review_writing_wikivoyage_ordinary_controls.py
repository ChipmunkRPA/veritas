# SPDX-License-Identifier: Apache-2.0
"""Recount the anonymous Wikivoyage held panel with the standard library."""

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

SCHEMA = "chipmunk.writing-wikivoyage-ordinary-controls/1"
CARRIERS = (".txt", ".docx")
ROLES = ("ordinary", "ordinary_edit")
MODES = ("pooled", "document", "report")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_counts(section, groups):
    rows = section["rows"]
    names = {row["group"] for row in rows}
    require(len(names) == groups, "Incorrect number of paired works")
    directions = tuple(product(CARRIERS, repeat=2))
    keys = Counter((row["group"], tuple(row["carriers"])) for row in rows)
    require(keys == Counter(product(names, directions)), "Incomplete or duplicate carrier comparison")

    counts = Counter(pairs=len(rows), held_groups=groups)
    for row in rows:
        changed, inside = row["has_changed_tokens"], row["all_changed_tokens_inside_target"]
        require(type(changed) is bool and type(inside) is bool, "Invalid paired match fields")
        counts["matched_saved_change"] += changed and inside
        counts["missed_saved_change"] += not changed
        counts["displaced_saved_change"] += not inside
        counts["exact_native_locations"] += row["docx_locations_verified"]
        counts["self_comparison_alerts"] += bool(row["unchanged_comparison_changes"])
    require(dict(counts) == section["summary"], "Paired comparison totals disagree")
    return dict(counts), names


def standalone_counts(section, names):
    rows = section["rows"]
    require(Counter((row["group"], row["role"]) for row in rows) == Counter(product(names, ROLES)),
            "Incomplete or duplicate standalone control")
    summary = {}
    for mode in MODES:
        counts = {role: {"assessed": 0, "alerts": 0} for role in ROLES}
        decisions = {group: set() for group in names}
        flagged = set()
        applied = 0
        for row in rows:
            reading = row["modes"][mode]
            status = reading["status"]
            count = reading["range_count"]
            require(type(count) is int and count >= 0 and status in {"flagged", "no_ranges", "unassessed"}
                    and (status == "flagged") == bool(count), "Invalid standalone range assessment")
            role, group = row["role"], row["group"]
            hit = bool(count)
            counts[role]["assessed"] += status != "unassessed"
            counts[role]["alerts"] += hit
            decisions[group].add(hit)
            if hit:
                flagged.add(group)
            applied += reading["report_applied"]
        summary[mode] = counts | {
            "alerted_work_groups": len(flagged),
            "report_scope_applied": applied,
            "earlier_late_alert_disagreements": sum(len(options) > 1 for options in decisions.values()),
        }
    require(summary == section["summary"], "Standalone control totals disagree")
    return summary


def recount(receipt):
    require(receipt["schema"] == SCHEMA, "Unsupported writing trial receipt")
    funnel = receipt["source_funnel"]
    groups = sum(funnel["eligible_group_strata"].values())
    rendered = funnel["different_editor_exact_replacements"] + funnel["same_editor_reserve_exact_replacements"]
    excluded = funnel["excluded_named_third_party_source"] + funnel["prior_overlap_exclusions"]
    require(rendered == groups + excluded, "Source eligibility totals disagree")
    paired, names = paired_counts(receipt["paired"], groups)
    standalone = standalone_counts(receipt["standalone"], names)
    return {"eligible_groups": groups, "paired": paired, "standalone": standalone}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(recount(json.loads(args.receipt.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
