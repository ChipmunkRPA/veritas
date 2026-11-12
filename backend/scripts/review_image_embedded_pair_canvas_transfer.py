# SPDX-License-Identifier: Apache-2.0
"""Independently recompute the public size-transfer decisions with the standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SIZES = ("768x768", "1024x768", "1280x720", "1366x768", "1920x1080")
MEASURED = {"sharp", "feather", "ordinary_local"}
CONTROLS = {"duplicate", "unrelated", "screenshot", "standalone"}
THIRDS = {"third_full", "third_partial", "third_064", "third_080", "third_118"}
ROLES = dict.fromkeys(MEASURED | CONTROLS | THIRDS, 16) | {"standalone": 6, "screenshot": 8}
FAMILIES = Counter(flux=6, lama=4, authored_paste=6)


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def review(path):
    package = json.loads(path.read_bytes())
    require(package["schema"] == "chipmunk.image-embedded-pair-canvas-transfer/1", "Unexpected study schema")
    require(len(package["rows"]) == 870 and set(package["results"]) == set(SIZES), "Public panels are incomplete")
    require(package["rules"] == {"minimum_exact_sharp": 6, "minimum_exact_feather": 6, "minimum_exact_ordinary_local": 10,
                                 "retained_exact_group_fraction": .75, "incorrect_valid_masks": 0,
                                 "reports_on_ordinary_controls": 0, "reports_with_additional_third": 0,
                                 "minimum_unguarded_third_challenges": 3, "maximum_per_case_seconds": 5.0,
                                 "maximum_peak_rss_bytes": 512 * 1024**2}, "Frozen eligibility rule changed")
    reference = set()
    for size in SIZES:
        claim = package["results"][size]
        rows = [row for row in package["rows"] if "x".join(map(str, row["size"])) == size]
        require({(row["role"], row["slot"]) for row in rows} == {(role, index) for role, n in ROLES.items() for index in range(n)}
                and len(rows) == sum(ROLES.values()), "Duplicate or missing role")
        require(Counter(row["family"] for row in rows if row["role"] == "sharp") == FAMILIES, "Photo families differ")
        for row in rows:
            require(row["reported"] == (row["status"] == "local_difference") == (row["pixels"] is not None)
                    and (not row["exact"] or row["role"] in MEASURED and row["reported"])
                    and (not row["reported"] or row["unguarded"] and row["pixels"] >= 48)
                    and row["current_report"] == (size == SIZES[0] and row["reported"]), "Report outcome is inconsistent")
        exact = {(row["role"], row["slot"]) for row in rows if row["exact"]}
        if size == SIZES[0]:
            reference = exact
        roles = {}
        for role in ROLES:
            selected = [row for row in rows if row["role"] == role]
            roles[role] = {"total": len(selected), "reports": sum(row["reported"] for row in selected),
                           "current_reports": sum(row["current_report"] for row in selected),
                           "unguarded": sum(row["unguarded"] for row in selected), "exact": sum(row["exact"] for row in selected),
                           "ambiguous": sum(row["status"] == "unassessed_ambiguous_match" for row in selected)}
        require(roles == claim["roles"], "Role totals differ")
        families = {family: {f"{role}_{metric}": sum(row["family"] == family and row["role"] == role
                                                     and (metric == "total" or row["exact"]) for row in rows)
                              for role in MEASURED for metric in ("total", "exact")} for family in FAMILIES}
        require(families == claim["families"], "Source-family totals differ")
        require(claim["paired_exact_retained"] == len(reference & exact) and claim["paired_exact_reference"] == len(reference)
                and claim["paired_exact_gains"] == len(exact - reference), "Matched groups differ")
        timing = package["measurements"][size]
        require(claim["resources"] == timing and timing["peak_rss_bytes"] > 0 and timing["seconds"] > 0
                and 0 < timing["max_case_seconds"] <= timing["seconds"], "Runtime record differs")
        checks = {"minimum_valid": min(roles["sharp"]["exact"], roles["feather"]["exact"]) >= 6
                  and roles["ordinary_local"]["exact"] >= 10,
                  "retained_exact_groups": 4 * len(reference & exact) >= 3 * len(reference),
                  "no_inexact_valid": all(roles[r]["reports"] == roles[r]["exact"] for r in MEASURED),
                  "no_ordinary_reports": all(roles[r]["reports"] == 0 for r in CONTROLS),
                  "no_third_reports": all(roles[r]["reports"] == 0 for r in THIRDS),
                  "adequate_third_challenge": all(roles[r]["unguarded"] >= 3 for r in THIRDS),
                  "runtime": timing["max_case_seconds"] <= 5 and timing["peak_rss_bytes"] <= 512 * 1024**2}
        require(claim["checks"] == checks and claim["passed"] == all(checks.values()), "Size decision differs")
        released = sum(roles[r]["current_reports"] for r in MEASURED)
        print(f"{size}  exact {len(exact)}/48  served {released}  third reports "
              f"{sum(roles[r]['reports'] for r in THIRDS)}/80  passes {claim['passed']}")
    print(hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    review(parser.parse_args().receipt)
