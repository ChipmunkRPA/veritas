# SPDX-License-Identifier: Apache-2.0
"""Replay held email-alternative outcome counts and decoded MIME commitments."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{64}")
PARTS = (("1.1", "1.2"), ("1.1.1", "1.1.2"))
STATES = ("difference", "equivalent", "unsupported")


def require(test, reason):
    if not test:
        raise ValueError(reason)


def exact(row):
    require(row["status"] in STATES and row["expected"] in STATES and SHA.fullmatch(row["sha256"]), "Invalid fixture commitment")
    parts, active = row["parts"], row["active"]
    if row["status"] == "unsupported":
        require(parts is None and not active, "Unsupported file carries text positions")
        return row["expected"] == row["status"]
    require(set(parts) == {"plain", "html"} and (parts["plain"]["id"], parts["html"]["id"]) in PARTS,
            "Unexpected source MIME positions")
    require(all(SHA.fullmatch(part["sha256"]) and type(part["characters"]) is int and 0 < part["characters"] <= 64_000
                for part in parts.values()), "Invalid decoded source commitment")
    require(len({a["side"] for a in active}) == len(active) and {a["side"] for a in active} <= set(parts),
            "Repeated or unsupported affected part")
    for side in active:
        predicted, expected, overlap = (side[key] for key in ("selected_tokens", "expected_tokens", "overlap"))
        require(all(type(value) is int for value in (predicted, expected, overlap))
                and 0 <= overlap <= min(predicted, expected) and len(side["ranges"]) >= predicted,
                "Invalid exact source token comparison")
        previous = 0
        for token in side["ranges"]:
            require(bool(token), "Missing source token span")
            for start, stop in token:
                require(type(start) is int and type(stop) is int
                        and previous <= start < stop <= parts[side["side"]]["characters"], "Invalid decoded character range")
                previous = stop
    require((row["status"] == "difference") == any(s["selected_tokens"] for s in active), "Status differs from selected source ranges")
    if row["expected"] != row["status"]:
        return False
    if row["expected"] == "equivalent":
        return not active
    return bool(active) and all(s["expected_tokens"] and 2 * s["overlap"] >= s["expected_tokens"]
                                and 10 * s["overlap"] >= 3 * (s["expected_tokens"] + s["selected_tokens"] - s["overlap"])
                                for s in active)


def replay(receipt):
    require(receipt["schema"] == "chipmunk.email-alternatives-transfer/1"
            and SHA.fullmatch(receipt["implementation_sha256"])
            and all(SHA.fullmatch(pin) for pin in receipt["commitments"].values()), "Missing frozen evidence commitments")
    panels = receipt["initial_and_confirmation"]
    require({key: len(value) for key, value in panels.items()} == {"initial": 90, "confirmation": 24, "second_after_correction": 24},
            "Unexpected held fixture roster")
    result = {}
    for panel, rows in panels.items():
        require(len({r["sha256"] for r in rows}) == len(rows), "Repeated held fixture")
        verified = [exact(row) for row in rows]
        result[panel] = dict(Counter(f"{row['expected']}_{'exact' if ok else 'incorrect'}" for row, ok in zip(rows, verified, strict=True)))
    earlier, repeated = receipt["second_before_correction"], panels["second_after_correction"]
    require(len(earlier) == len(repeated) and all((a["sha256"], a["expected"]) == (b["sha256"], b["expected"])
                                                 for a, b in zip(earlier, repeated, strict=True)), "Correction uses different inputs")
    require(all(row["status"] in STATES and type(row["exact"]) is bool for row in earlier), "Invalid earlier outcomes")
    result["second_before_correction"] = dict(Counter(f"{r['expected']}_{'exact' if r['exact'] else 'incorrect'}" for r in earlier))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    result = replay(receipt)
    require(result == receipt["result"], "Published counts differ from replay")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
