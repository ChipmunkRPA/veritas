"""Check the portable frequency-balance receipt using only the standard library."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from review_audio_clean_reference_duration import useful

SCHEMA = "chipmunk.audio-clean-reference-duration-eq/1"
LEVELS = (0, -5, -10)
STYLES = ("other_lowpass", "high_shelf")
WRONG = ("wrong_reader", "clock_shifted", "partial_match")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def balance_qualifies(profiles):
    if profiles[0] is None or profiles[2] is None:
        return False
    bands = tuple(median(values) for values in zip(*(row for row in profiles if row is not None), strict=True))
    tilt = bands[-1] - bands[0]
    if abs(tilt) < .35:
        return False
    direction = 1 if tilt > 0 else -1
    if direction * (bands[-1] - bands[2]) < .18:
        return False
    for section, row in enumerate(profiles):
        if row is None:
            continue
        required, tolerance = (.15, .22) if section == 1 else (.25, .18)
        shape_error = median(abs(x - y) for x, y in zip(row, bands, strict=True))
        if direction * (row[-1] - row[0]) < required or shape_error > tolerance:
            return False
    return True


def review(receipt):
    require(receipt["schema"] == SCHEMA and len(receipt["rows"]) == 585, "Unexpected receipt")
    tally = {role: {str(level): Counter() for level in LEVELS} for role in ("development", "assessment")}
    for row in receipt["rows"]:
        profiles = row["profiles"]
        require(len(profiles) == 3 and all(part is None or len(part) == 5 for part in profiles), "Invalid spectral profile")
        explained = balance_qualifies(profiles)
        require(explained == row["eq_explains"], "Profile does not support the recorded frequency-balance decision")
        source, result = row["prior_range"], row["retained_range"]
        require(result == (None if explained else source), "Wrong retained interval")
        counts = tally[row["role"]][str(row["level"])]
        kind = row["kind"]
        counts[f"rows:{kind}"] += 1
        counts[f"explained:{kind}"] += int(explained)
        counts[f"admitted:{kind}"] += int(row["admitted"])
        if source:
            counts[f"prior:{kind}"] += 1
            if result:
                counts[f"retained:{kind}"] += 1
            elif row["truth"] and useful(source, row["truth"]):
                counts["useful_removed"] += 1
    held = tally["assessment"]["-5"]
    gate = (held["prior:global_eq"] - held["retained:global_eq"] >= 3 and not held["useful_removed"]
            and all(not counts[f"admitted:{kind}"] for counts in tally["assessment"].values() for kind in WRONG)
            and all(2 * counts[f"retained:{style}"] <= counts[f"prior:{style}"]
                    for counts in tally["assessment"].values() for style in STYLES))
    return tally | {"gate_pass": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        receipt = json.loads(args.receipt.read_bytes())
        require(receipt["summary"] == review(receipt), "Frequency-balance or localization accounting differs")
        print(f"Verified frequency profiles and retained intervals for {len(receipt['rows'])} pairs.")
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
