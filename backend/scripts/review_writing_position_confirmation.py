"""Replay the fixed interior-position writing study on versioned sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

POINT = 2.7579683180027486
SCHEMA = "chipmunk.writing-position-confirmation/1"
PRIOR_FILENAME = "writing-two-context-transfer-2026-11-12.json"
SHA256 = re.compile(r"[0-9a-f]{64}")
ROLES = ("human", "ordinary", "granite", "tiny")
PUBLISHERS = {"python-peps": 4, "peerj": 2, "copernicus": 2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure(case, *, generated_roles=("granite", "tiny")):
    units = case["units"]
    require(len(units) == 3 and len(case["responses"]) == 3 and len(case["nonspace"]) == 3,
            "Missing exact measurements")
    require(12 <= units[1][2] <= 65 and all(7 <= units[i][2] <= 80 for i in (0, 2))
            and units[0][0] == 0 and units[-1][1] == case["characters"]
            and all(0 <= a < b and 0 < n <= b - a for a, b, n in units)
            and all(a[1] < b[0] for a, b in zip(units, units[1:])),
            "Native sentence scope or positions differ")
    require(all(type(n) is int and 0 < n <= b - a for n, (a, b, _) in zip(case["nonspace"], units, strict=True)),
            "Invalid non-whitespace sizes")
    generated = case["role"] in generated_roles
    require(case["truth"] == (units[1][:2] if generated else None), "Known focus differs")
    for item in case["responses"]:
        require(math.isfinite(item["head"]) and math.isfinite(item["current"]) and 0 <= item["current"] <= 1
                and item["focus_tokens"] > 0 and item["neighbor_tokens"] > 0
                and item["focus_tokens"] + item["neighbor_tokens"] <= 512
                and all(SHA256.fullmatch(item[key]) for key in ("input_sha256", "input_ids_sha256")),
                "Invalid native response")
    left, center, right = case["responses"]
    score = center["head"] - max(left["head"], right["head"])
    picked = [i for i, item in enumerate(case["responses"]) if item["current"] >= 0.5]
    useful = generated and any(case["nonspace"][1] * 10 >= 3 * sum(case["nonspace"][max(i - 1, 0):i + 2])
                               for i in picked)
    require(score == case["contrast"] and (score > POINT) == case["contrast_selected"]
            and picked == case["current"] and useful == case["current_useful"]
            and (score > POINT and generated) == case["contrast_useful"], "Saved response or overlap arithmetic differs")


def review(receipt, prior):
    require(receipt["schema"] == SCHEMA and receipt["point"] == POINT, "Frozen point or receipt differs")
    require(hashlib.sha256(prior).hexdigest() == receipt["prior_receipt_sha256"]
            and json.loads(prior)["points"]["interior_position"] == POINT, "Published initial study does not bind this point")
    sources = receipt["sources"]
    groups = {(x["publisher"], x["group"]) for x in sources}
    require(len(groups) == len(sources) == 8 and Counter(x["publisher"] for x in sources) == PUBLISHERS,
            "Publisher roster differs")
    cases = receipt["cases"]
    counts = Counter(x["role"] for x in cases)
    require(counts == {"human": 8, "ordinary": 8, "granite": 8, "tiny": 4}
            and len({(x["publisher"], x["group"], x["role"]) for x in cases}) == len(cases)
            and all((x["publisher"], x["group"]) in groups for x in cases), "Case roster differs")
    for case in cases:
        measure(case)
    result = {}
    for publisher in PUBLISHERS:
        row = result[publisher] = {}
        for role in ROLES:
            subset = [x for x in cases if (x["publisher"], x["role"]) == (publisher, role)]
            row[role] = {"cases": len(subset), "incumbent_flags": sum(bool(x["current"]) for x in subset),
                         "incumbent_useful": sum(x["current_useful"] for x in subset),
                         "candidate_flags": sum(x["contrast_selected"] for x in subset),
                         "new_useful": sum(x["contrast_useful"] and not x["current_useful"] for x in subset)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    try:
        prior = args.prior or args.receipt.with_name(PRIOR_FILENAME)
        summary = review(json.loads(args.receipt.read_bytes()), prior.read_bytes())
    except (OSError, KeyError, IndexError, TypeError, ValueError) as error:
        raise SystemExit(f"Review failed. {error}") from None
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
