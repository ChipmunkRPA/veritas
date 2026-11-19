"""Recount the public JPEG residual-phase comparison without media or model libraries."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-jpeg-phase-consistency/1"
QUALITIES = ("75", "90")
ARMS = ("signed_phase_two", "magnitude_phase_eight")
ROLES = ("ordinary", "inpaint_A", "inpaint_B", "blur_A", "blur_B", "tone_A", "tone_B")
AREA = 64 * 64
DEFAULT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/image-jpeg-phase-consistency-2026-11-18.json.gz"


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError(detail)


def number(raw: str | None) -> float | None:
    if raw is None:
        return None
    require(isinstance(raw, str), "Scores must be in hexadecimal")
    value = float.fromhex(raw)
    require(math.isfinite(value) and 0 <= value <= 2.00000001 and value.hex() == raw, "Invalid phase reading")
    return value


def pixels(value: int) -> bool:
    return type(value) is int and 0 <= value <= AREA


def decode_case(case: dict, role: str) -> dict:
    require(set(case) == {"score", "proposals", "mask", "intersection", "ordinary_intersection", "sites"},
            "Unexpected case fields")
    score = number(case["score"])
    count, mask, common, ordinary = (case[key] for key in ("proposals", "mask", "intersection", "ordinary_intersection"))
    require(type(count) is int and 0 <= count <= 2500 and (count == 0) == (score is None), "Invalid proposal count")
    require(pixels(mask) and pixels(common) and common <= mask and pixels(ordinary), "Invalid overlap counts")
    require((mask > 0) == (role != "ordinary" and score is not None), "Unexpected changed-pixel mask")
    require(score is not None or (common == ordinary == 0), "Unexpected unsupported-case geometry")
    require(isinstance(case["sites"], list) and len(case["sites"]) == 2, "Expected both matched sites")
    places = tuple(number(value) for value in case["sites"])
    require(all(value is None or score is not None and value <= score for value in places),
            "A known-site result exceeds the measured maximum")
    return {"score": score, "useful": mask > 0 and common * 2 >= mask and common * 10 >= 3 * (AREA + mask - common),
            "top_at_ordinary": ordinary * 3 >= 2 * AREA, "sites": places}


def contrast(ordinary: dict, first: dict, second: dict) -> tuple[bool, bool]:
    ua, ub = ordinary["sites"]
    aa, ab = first["sites"]
    ba, bb = second["sites"]
    if None in (ua, ub, aa, ab, ba, bb):
        return False, False
    normal = ua - ub
    return True, aa - ab > max(0.0, normal) + 0.02 and bb - ba > max(0.0, -normal) + 0.02


def tally(groups: list[dict], quality: str, arm: str, cutoff: float) -> dict:
    totals = Counter()
    correct_pair_banks = set()
    for group in groups:
        cases = group["qualities"][quality][arm]
        require(set(cases) == set(ROLES), "A group is missing a comparison role")
        parsed = {role: decode_case(cases[role], role) for role in ROLES}
        flags = {role: case["score"] is not None and case["score"] > cutoff for role, case in parsed.items()}
        for role, case in parsed.items():
            kind = role.split("_")[0]
            totals[f"{kind}_total"] += 1
            totals[f"{kind}_unassessed"] += int(case["score"] is None)
            totals[f"{kind}_flags"] += int(flags[role])
            totals[f"{kind}_useful_before_point"] += int(case["useful"])
            totals[f"{kind}_useful"] += int(case["useful"] and flags[role])
            if kind == "inpaint":
                totals["inpaint_wrong"] += int(flags[role] and not case["useful"])
            if kind != "ordinary":
                totals[f"{kind}_top_at_ordinary"] += int(case["top_at_ordinary"])
        for role in ("inpaint", "blur", "tone"):
            a, b = role + "_A", role + "_B"
            measured, moved = contrast(parsed["ordinary"], parsed[a], parsed[b])
            totals[f"{role}_paired_assessed"] += int(measured)
            totals[f"{role}_follows_placement"] += int(moved)
            if flags[a] and flags[b] and parsed[a]["useful"] and parsed[b]["useful"]:
                totals[f"{role}_useful_pairs"] += 1
                if role == "inpaint":
                    correct_pair_banks.add(group["bank"])
    advances = (totals["inpaint_useful_pairs"] >= 3 and len(correct_pair_banks) == 2 and totals["ordinary_flags"] <= 2
                and totals["blur_flags"] + totals["tone_flags"] <= 2 and totals["inpaint_wrong"] <= 2)
    return {"cutoff_hex": cutoff.hex(), **dict(sorted(totals.items())), "advance": advances}


def review(document: dict) -> dict:
    require(set(document) == {"schema", "commitment", "calibration", "groups", "native_review"}
            and document["schema"] == SCHEMA and re.fullmatch(r"[a-f0-9]{64}", document["commitment"]) is not None,
            "Invalid phase comparison envelope")
    native = document["native_review"]
    require(set(native) == {"jpeg_files", "mask_deliveries", "score_checks", "missing_support_checks",
                            "maximum_absolute_score_difference"}
            and native["jpeg_files"] == 240 and native["mask_deliveries"] == 192
            and all(type(native[key]) is int and native[key] >= 0 for key in ("score_checks", "missing_support_checks"))
            and 0 <= native["maximum_absolute_score_difference"] <= 3e-5, "Invalid native replay summary")
    groups, calibration = document["groups"], document["calibration"]
    require(len(groups) == 16 and Counter(g["bank"] for g in groups) == {"camera": 8, "transfer": 8}
            and [group["group"] for group in groups] == list(range(1, 17)), "Unexpected held units")
    require(set(calibration) == set(QUALITIES), "Missing delivery qualities")
    for group in groups:
        require(set(group) == {"group", "bank", "qualities"} and set(group["qualities"]) == set(QUALITIES),
                "Unexpected held group fields")
        require(all(set(arms) == set(ARMS) for arms in group["qualities"].values()), "Unexpected held measurements")
    totals = {}
    for quality in QUALITIES:
        require(set(calibration[quality]) == set(ARMS), "Missing calibration arm")
        for arm in ARMS:
            calibration_scores = [number(raw) for raw in calibration[quality][arm]]
            require(len(calibration_scores) == 8 and all(value is not None for value in calibration_scores),
                    "Missing ordinary calibration maximum")
            cutoff = sorted(calibration_scores, reverse=True)[1]
            totals[f"{quality}/{arm}"] = tally(groups, quality, arm, cutoff)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as handle:
        results = review(json.load(handle))
    print(json.dumps(results, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
