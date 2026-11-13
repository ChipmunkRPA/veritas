"""Recount JPEG90 transfer decisions from source-bound pixel totals."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-regional-jpeg90-carrier/1"
HASH = re.compile(r"[a-f0-9]{64}")
EDITED = {"sharp", "feathered"}


def require(value, message):
    if not value:
        raise ValueError(message)


def measure(item, extent, edited):
    source = item["reference_positive_pixels"]
    measured = item["predicted_positive_pixels"]
    both = item["intersection"]
    require(item["pixels"] == extent and all(type(n) is int for n in (source, measured, both))
            and bool(source) == edited and 0 <= both <= min(source, measured)
            and source + measured - both <= extent, "Impossible source overlap")
    hit = edited and both / source >= .5 and both / (source + measured - both) >= .3
    return bool(measured), hit


def review(ledger):
    value = json.loads(ledger.read_bytes())
    freeze = value["freeze"]
    require(freeze["schema"] == SCHEMA and all(HASH.fullmatch(freeze[key]) for key in (
        "runner_sha256", "checkpoint_sha256", "source_freeze_sha256", "source_journal_sha256")), "Invalid study freeze")
    seen = set()
    groups = defaultdict(set)
    old_ordinary, new_ordinary, known_families, kept_families = set(), set(), set(), set()
    counts = Counter()
    carriers = Counter()
    for row in value["rows"]:
        panel, group, family, role = (row[key] for key in ("panel", "group", "family", "role"))
        identity = panel, group, family, role
        require(identity not in seen and all(HASH.fullmatch(row[key]) for key in (
            "input_sha256", "q90_sha256", "q90_mask_sha256", "truth_sha256")), "Duplicate or invalid delivery")
        seen.add(identity)
        groups[panel].add(group)
        carriers[panel] += 1
        h, w = row["shape"]
        require(max(h, w) == 768 and min(h, w) >= 512 and row["input_sha256"] != row["q90_sha256"], "Invalid source geometry or delivery")
        edited = role in EDITED
        old, new = (measure(row[quality], h * w, edited) for quality in ("q75", "q90"))
        require(row["q75"]["reference_positive_pixels"] == row["q90"]["reference_positive_pixels"], "Changed edit truth")
        if edited:
            counts.update(edited=1, old_useful=old[1], new_useful=new[1], kept=old[1] and new[1], gained=new[1] and not old[1],
                          old_wrong=old[0] and not old[1], new_wrong=new[0] and not new[1])
            if old[1]:
                known_families.add(family)
                if new[1]:
                    kept_families.add(family)
        else:
            counts.update(ordinary=1, old_flags=old[0], new_flags=new[0], added_flags=new[0] and not old[0])
            if old[0]:
                old_ordinary.add(group)
            if new[0]:
                new_ordinary.add(group)
    require(carriers == {"retained_b": 136, "external": 64} and len(groups["retained_b"]) == 15
            and len(groups["external"]) == 7 and not groups["retained_b"] & groups["external"], "Incomplete or overlapping source banks")
    counts["ordinary_groups_added"] = len(new_ordinary - old_ordinary)
    passes = (counts["old_useful"] == 11 and counts["kept"] >= 8 and kept_families == known_families
              and counts["added_flags"] <= 5 and counts["ordinary_groups_added"] <= 2 and counts["new_wrong"] - counts["old_wrong"] <= 4)
    require(counts == value["counts"] and sorted(kept_families) == value["kept_families"] and passes is value["passes"],
            "Recorded counts or decision differ")
    return {"counts": dict(counts), "kept_families": sorted(kept_families), "passes": passes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, nargs="+")
    paths = parser.parse_args().ledger
    values = [review(path) for path in paths]
    print(json.dumps(values[0] if len(values) == 1 else values, sort_keys=True))
