"""Prospectively choose the first complete held worker batch and preserve later batches."""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts import writing_revision_trajectory_transfer as source

GROUPS_PER_BATCH = 4
SCHEMA = "chipmunk.writing-revision-worker-batches/1"


def plan(prior):
    frozen = source.read(prior / "freeze.json")
    roster = prior / "roster.json"
    source.require(frozen["schema"] == source.SCHEMA and frozen["roster_sha256"] == source.digest(roster), "Wrong source roster")
    groups = {}
    for row in source.read(roster):
        if row["role"] == "held":
            groups.setdefault(row["group"], []).append(row["record_id"])
    source.require(len(groups) == 20 and all(len(items) == 7 for items in groups.values()), "Complete held trajectories required")
    ordered = sorted(groups, key=lambda group: source.sha("writing-worker-held-priority/1/" + group))
    batches = [ordered[start:start + GROUPS_PER_BATCH] for start in range(0, len(ordered), GROUPS_PER_BATCH)]
    return {"schema": SCHEMA, "roster_sha256": frozen["roster_sha256"], "groups": batches,
            "batches": [[record for group in batch for record in groups[group]] for batch in batches]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source.seal(args.output, plan(args.prior))


if __name__ == "__main__":
    main()
