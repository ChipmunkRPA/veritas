"""Recount the published carrier results without the detector or source images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path

POSITIVES = {"change_tax", "change_total"}
CONTROLS = {
    "discount",
    "discount_after",
    "duplicate_card",
    "duties",
    "included_tax",
    "intact",
    "mixed_currency",
    "one_cent",
    "only_amount_due",
    "other_addition",
    "service_fee",
    "synchronized",
    "unrelated_rows",
}
CARRIERS = ("png", "jpeg75")
UNASSESSED = {"not_supported", "not_available"}


def recount(receipt):
    if (
        receipt["schema"] != "chipmunk.printed-total-evaluation/1"
        or receipt["verified_exact_native_edits"] != 32
    ):
        raise ValueError("Unexpected evaluation design")
    expected = set(product(range(16), POSITIVES | CONTROLS, CARRIERS))
    tally = {carrier: Counter() for carrier in CARRIERS}
    control_alerts = {carrier: Counter() for carrier in CARRIERS}
    for row in receipt["records"]:
        group, variant, carrier = row["id"], row["variant"], row["carrier"]
        expected.remove((group, variant, carrier))
        counts = tally[carrier]
        counts["evaluated"] += 1
        status = row["status"]
        overlap = row["min_amount_overlap"]
        if status not in {"mismatch", "match", *UNASSESSED} or row["region_match"] != (
            overlap is not None and overlap >= 0.9
        ):
            raise ValueError("Inconsistent published decision or crop coverage")
        if variant in POSITIVES:
            counts["positive"] += 1
            counts["unassessed_positive"] += status in UNASSESSED
            if status == "mismatch":
                counts["correct" if row["exact"] and row["region_match"] else "misread_positive"] += 1
        else:
            counts["controls"] += 1
            counts["unassessed_controls"] += status in UNASSESSED
            counts["control_alerts"] += status == "mismatch"
            if status == "mismatch":
                control_alerts[carrier][variant] += 1
    if expected:
        raise ValueError("Missing tested layouts and controls")
    summary = {}
    for carrier, counts in tally.items():
        keys = (
            "evaluated",
            "positive",
            "correct",
            "misread_positive",
            "unassessed_positive",
            "controls",
            "control_alerts",
            "unassessed_controls",
        )
        summary[carrier] = {key: counts[key] for key in keys} | {
            "valid_control_alert_roles": dict(control_alerts[carrier]),
            "gate_pass": counts["correct"] >= 25
            and counts["misread_positive"] == counts["control_alerts"] == 0,
        }
    if summary != receipt["fresh_confirmation"]:
        raise ValueError("Reported carrier counts differ from the individual observations")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(recount(json.loads(args.receipt.read_text())), indent=2))


if __name__ == "__main__":
    main()
