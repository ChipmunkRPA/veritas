# SPDX-License-Identifier: Apache-2.0
"""Check public dense third-copy outcomes without images or OpenCV."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SIZES = ("768x768", "1024x768", "1280x720")
VALID = ("valid_pair", "unrelated_third")
CONTROLS = (*VALID, "duplicate_pair")
THIRDS = ("third_062", "third_071", "third_087", "third_113", "third_121")


def verify(path):
    report = json.loads(path.read_bytes())
    assert report["schema"] == "chipmunk.image-embedded-pair-dense-third/1"
    assert len(report["rows"]) == 3*12*(len(CONTROLS)+len(THIRDS)) and set(report["outcomes"]) == set(SIZES)
    assert report["rule"] == {"patch": 16, "grid_stride": 19, "minimum_sd": 8, "minimum_correlation": .97,
                              "minimum_patches": 16, "minimum_cells": 6, "minimum_source_fraction": .55,
                              "minimum_exact_masks_per_valid_role": 10, "maximum_new_withholds": 0,
                              "minimum_missed_cases_per_size": 3, "minimum_rescued_fraction": .80,
                              "maximum_candidate_seconds": 1.5}
    rows = report["rows"]
    audit = Counter()
    for size in SIZES:
        claim = report["outcomes"][size]
        entries = [row for row in rows if "x".join(map(str, row["size"])) == size]
        assert len(entries) == 96 and {(row["slot"], row["role"]) for row in entries} == {
            (slot, role) for slot in range(12) for role in (*THIRDS, *CONTROLS)
        }
        roles = {}
        for row in entries:
            assert row["seconds"] >= 0
            witnesses = row["native_witness_count"]
            assert witnesses == 0 if not row["extra_withheld"] else witnesses >= 16 and row["current"] == "local_difference"
            assert not row["exact_valid_mask"] or row["role"] in VALID and row["current"] == "local_difference"
            fraction = row["whole_projection_in_authored_third"]
            centers = row["witness_centers_inside_authored_third"]
            assert (fraction is not None) == (row["extra_withheld"] and row["role"] in THIRDS)
            assert fraction is None or 0 <= fraction <= 1
            assert 0 <= centers <= witnesses and (fraction is not None or centers == 0)
            audit["native_patch_witnesses"] += witnesses
            audit["source_projections"] += row["extra_withheld"]
            audit["base_exact_masks"] += row["exact_valid_mask"]
            audit["witness_centers_inside_authored_third"] += centers
            audit["failed_original_footprint_requirement"] += fraction is not None and fraction < .90
            counts = roles.setdefault(row["role"], Counter())
            counts["current_reports"] += row["current"] == "local_difference"
            counts["extra_withholds"] += row["extra_withheld"]
            counts["exact_valid_masks"] += row["exact_valid_mask"]
        assert roles == claim["roles"]
        original = sum(roles[role]["current_reports"] for role in THIRDS)
        rescued = sum(roles[role]["extra_withholds"] for role in THIRDS)
        largest = max(row["seconds"] for row in entries)
        assert original == claim["old_third_reports"] and original-rescued == claim["remaining_third_reports"]
        assert largest == claim["max_seconds"]
        passed = (not any(roles[role]["extra_withholds"] for role in CONTROLS)
                  and all(roles[role]["exact_valid_masks"] >= 10
                          and roles[role]["exact_valid_masks"] == roles[role]["current_reports"] for role in VALID)
                  and original >= 3 and rescued * 5 >= original * 4 and largest <= 1.5)
        assert passed == claim["passed"]
        print(size, original, rescued, passed)
    assert all(audit[key] == report["audit"].get(key, 0) for key in audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    verify(parser.parse_args().receipt)
