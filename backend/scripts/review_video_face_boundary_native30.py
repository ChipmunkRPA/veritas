# SPDX-License-Identifier: Apache-2.0
"""Recount exact 30 fps face-transition outcomes without source media or models."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from scripts.review_audio_two_second_opus_transfer import require

SCHEMA = "chipmunk.video-face-boundary-native30-review/1"
CARRIERS = ("direct", "tandem")
EDITS = ("prefix", "suffix")
ORDINARY = ("original", "exposure", "focus", "same_recording_jump", "cross_recording_cut")
ROLES = (*EDITS, *ORDINARY)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def frame(value):
    return type(value) is int and 0 <= value < 120


def replay(rows):
    groups, measurements = defaultdict(set), {name: Counter() for name in CARRIERS}
    gains, ordinary_flags = defaultdict(set), defaultdict(set)
    paired = {}
    require(len(rows) == 98, "The comparison requires all sealed deliveries")
    for row in rows:
        require(set(row) == {"carrier", "group", "role", "renderer", "truth", "baseline_ranges", "shown", "coverage", "reason"},
                "Unexpected fields in the portable record")
        carrier, identity, role, declared = (row[key] for key in ("carrier", "group", "role", "truth"))
        positive = role in EDITS
        require(carrier in CARRIERS and type(identity) is int and identity in range(1, 8) and role in ROLES
                and ((frame(declared) and row["renderer"] in ("swap_a", "swap_b")) if positive else
                     declared is None and row["renderer"] is None), "A case has invalid source or edit metadata")
        require(role not in groups[carrier, identity], "Duplicate camera, carrier and edit role")
        groups[carrier, identity].add(role)
        label = (declared, row["renderer"])
        require(paired.setdefault((identity, role), label) == label, "The same edit changed across deliveries")
        require(all(isinstance(span, list) and len(span) == 2 and all(frame(i) for i in span) and span[0] < span[1]
                    for span in row["baseline_ranges"]), "Invalid native Full interval")
        shown = row["shown"]
        coverage = row["coverage"]
        require(coverage in ("localized", "no_match", "unavailable") and (shown is not None) == (coverage == "localized")
                and (row["reason"] is None) == (coverage != "unavailable"), "Recorded witness and coverage disagree")
        if shown is not None:
            require(all(frame(shown[key]) for key in ("before", "transition", "after"))
                    and shown["before"] + 1 == shown["transition"] < shown["after"]
                    and shown["pts"] == shown["transition"] * 1000 and shown["time_base"] == "1/30000",
                    "The shown frames differ from the verified source clock")
        counts = measurements[carrier]
        counts.update({"delivered": 1, f"coverage_{coverage}": 1})
        if not positive:
            counts.update(ordinary=1, current_ordinary_alerts=bool(row["baseline_ranges"]),
                          added_ordinary_alerts=shown is not None and not row["baseline_ranges"],
                          candidate_ordinary_alerts=shown is not None)
            if shown is not None:
                ordinary_flags[carrier].add(identity)
                counts[f"candidate_alert_{role}"] += 1
            continue
        comparison = any(abs(edge - declared) <= 1 for span in row["baseline_ranges"] for edge in span)
        error = None if shown is None else abs(shown["transition"] - declared)
        candidate = error is not None and error <= 1
        counts.update(edits=1, current_useful=comparison, candidate_useful=candidate, candidate_exact=error == 0,
                      candidate_wrong=shown is not None and not candidate, combined_useful=comparison or candidate,
                      added_useful=candidate and not comparison)
        if candidate and not comparison:
            gains[carrier].add(identity)
            counts[f"added_{role}"] += 1
            counts[f"added_{row['renderer']}"] += 1
    require(all(groups[carrier, group] == set(ROLES) for carrier in CARRIERS for group in range(1, 8)),
            "Every camera and carrier must include both edits and all ordinary controls")
    for carrier, counts in measurements.items():
        counts["gain_groups"] = len(gains[carrier])
        counts["ordinary_flag_groups"] = len(ordinary_flags[carrier])
    return {carrier: dict(counts) for carrier, counts in measurements.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as file:
        receipt = json.load(file)
    require(receipt["schema"] == SCHEMA and receipt["rows_sha256"] == sha256(canonical(receipt["rows"])).hexdigest(),
            "Published rows differ from their commitment")
    result = replay(receipt["rows"])
    require(all(result[carrier] == receipt["summary"][carrier] for carrier in CARRIERS),
            "The independent counts differ from the published summary")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
