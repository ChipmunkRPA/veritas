"""Recount the anonymous matched-site JPEG study with the standard library."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-jpeg-matched-site-transfer/1"
QUALITIES = ("75", "90")
ARMS = ("edge_continuation", "chroma_alignment", "joint")
ROLES = ("inpaint", "blur", "tone")
LOCATIONS = ("a", "b")
REGION = 64**2
DEFAULT = (
    Path(__file__).resolve().parents[2]
    / "docs/evaluation-panels/image-jpeg-matched-site-transfer-2026-11-18.json.gz"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value: str) -> float:
    require(isinstance(value, str), "Expected a hexadecimal reading")
    reading = float.fromhex(value)
    require(math.isfinite(reading) and reading >= 0 and reading.hex() == value, "Invalid reading")
    return reading


def count(value: int, minimum: int, maximum: int) -> int:
    require(type(value) is int and minimum <= value <= maximum, "Invalid pixel count")
    return value


def readings(payload: dict) -> dict[str, float | None]:
    values = {k: number(payload[k]) if payload[k] is not None else None for k in ("peak", *LOCATIONS)}
    peak = values["peak"]
    require(
        all(v is None or (peak is not None and v <= peak) for v in (values["a"], values["b"])),
        "Fixed-site reading exceeds the global peak",
    )
    return values


def flag(peak: float | None, point: float) -> bool:
    return peak is not None and peak > point


def variant(payload: dict, ordinary: dict, point: float) -> dict:
    require(
        set(payload)
        == {"peak", *LOCATIONS, "changed_pixels", "changed_pixels_in_top", "overlap_with_original"},
        "Invalid edited reading fields",
    )
    values = readings(payload)
    area = count(payload["changed_pixels"], math.ceil(REGION * 0.65), REGION)
    overlap = count(payload["changed_pixels_in_top"], 0, area)
    same = payload["overlap_with_original"]
    has_both_boxes = values["peak"] is not None and ordinary["peak"] is not None
    require((same is not None) == has_both_boxes, "Invalid original overlap")
    require(values["peak"] is not None or not overlap, "Abstention overlaps changed pixels")
    useful = 2 * overlap >= area and 10 * overlap >= 3 * (REGION + area - overlap)
    return {
        **values,
        "flag": flag(values["peak"], point),
        "useful": useful,
        "same": same is not None and 3 * count(same, 0, REGION) >= 2 * REGION,
    }


def pair_counts(edits: list, ordinary: dict, point: float, margin: float) -> dict:
    require(isinstance(edits, list) and len(edits) == 2, "Both planted sites are required")
    placed = [variant(row, ordinary, point) for row in edits]
    oa, ob, aa, ab, ba, bb = [row[site] for row in (ordinary, *placed) for site in LOCATIONS]
    measured = None not in (oa, ob, aa, ab, ba, bb)
    return {
        "pairs": 1,
        "measured_files": sum(r["peak"] is not None for r in placed),
        "useful_rank_files": sum(r["useful"] for r in placed),
        "useful_point_files": sum(r["useful"] and r["flag"] for r in placed),
        "after_point_flags": sum(r["flag"] for r in placed),
        "wrong_flags": sum(r["flag"] and not r["useful"] for r in placed),
        "useful_rank_pairs": int(all(r["useful"] for r in placed)),
        "useful_point_pairs": int(all(r["useful"] and r["flag"] for r in placed)),
        "same_ordinary_rank_files": sum(r["same"] for r in placed),
        "site_measured_pairs": int(measured),
        "site_switch_pairs": int(measured and aa > ab and bb > ba),
        "site_causal_pairs": int(
            measured and aa - ab > max(0, oa - ob) + margin and bb - ba > max(0, ob - oa) + margin
        ),
    }


def recount(receipt: dict) -> tuple[dict, dict]:
    require(
        set(receipt) == {"schema", "source_counts", "commitments", "clean_maxima", "groups", "reported"}
        and receipt["schema"] == SCHEMA
        and receipt["source_counts"] == {"calibration": 8, "held": 16, "deliveries": 240},
        "Unexpected study design",
    )
    commits = receipt["commitments"]
    require(
        set(commits) == {"input", "method", "native_recount", "predictions"}
        and all(re.fullmatch("[a-f0-9]{64}", v) for v in commits.values()),
        "Invalid study commitments",
    )
    maxima = receipt["clean_maxima"]
    require(set(maxima) == set(QUALITIES), "Missing quality for clean calibration")
    points = {}
    for quality, arms in maxima.items():
        require(set(arms) == set(ARMS), "Missing clean calibration arm")
        points[quality] = {}
        for arm, values in arms.items():
            require(isinstance(values, list) and len(values) == 8, "Expected eight calibration maxima")
            point = sorted(map(number, values))[-2]
            require(point > 0, "Expected a positive clean point")
            points[quality][arm] = point

    scores = defaultdict(Counter)
    bank_ids = set()
    for row in receipt["groups"]:
        bank = count(row["bank"], 1, 2)
        group = count(row["group"], 1, 8)
        require(
            set(row) == {"bank", "group", "features"} and (bank, group) not in bank_ids, "Invalid held group"
        )
        bank_ids.add((bank, group))
        cells = set()
        for cell in row["features"]:
            quality, arm = str(cell["quality"]), cell["arm"]
            require(
                set(cell) == {"quality", "arm", "original", "edited"}
                and quality in QUALITIES
                and arm in ARMS
                and (quality, arm) not in cells,
                "Unexpected quality or measurement arm",
            )
            cells.add((quality, arm))
            point = points[quality][arm]
            require(set(cell["original"]) == {"peak", *LOCATIONS}, "Invalid original reading fields")
            ordinary = readings(cell["original"])
            counters = scores[quality, arm, bank]
            counters["ordinary_groups"] += 1
            counters["ordinary_flags"] += flag(ordinary["peak"], point)
            counters["ordinary_unmeasured"] += ordinary["peak"] is None
            require(set(cell["edited"]) == set(ROLES), "Missing controlled edit family")
            for role, edits in cell["edited"].items():
                measures = pair_counts(edits, ordinary, point, 0.05 if arm == "joint" else 0.002)
                counters.update({f"{role}_{key}": value for key, value in measures.items()})
        require(len(cells) == len(QUALITIES) * len(ARMS), "Incomplete held group")
    require(len(bank_ids) == 16, "Expected two separate eight-source banks")

    outcome = {}
    for quality in QUALITIES:
        outcome[quality] = {}
        for arm in ARMS:
            banks = {str(b): dict(scores[quality, arm, b]) for b in (1, 2)}
            first, second = banks.values()
            total = {k: v + second[k] for k, v in first.items()}
            advanced = (
                total["inpaint_useful_point_pairs"] >= 3
                and min(first["inpaint_useful_point_pairs"], second["inpaint_useful_point_pairs"]) >= 1
                and total["ordinary_flags"] <= 2
                and total["blur_after_point_flags"] + total["tone_after_point_flags"] <= 2
                and total["inpaint_wrong_flags"] <= 2
                and total["inpaint_site_causal_pairs"]
                > max(total["blur_site_causal_pairs"], total["tone_site_causal_pairs"])
            )
            outcome[quality][arm] = {"banks": banks, "total": total, "advanced": advanced}
    require(receipt["reported"] == outcome, "Reported counts disagree with the per-source readings")
    return points, outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?", default=DEFAULT)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as payload:
        points, outcome = recount(json.load(payload))
    for quality, cells in outcome.items():
        for arm, reading in cells.items():
            counts = reading["total"]
            print(
                f"JPEG {quality} {arm} | point {points[quality][arm]:.6f} | "
                f"useful planted regions before point {counts['inpaint_useful_rank_files']}/32 | "
                f"thresholded pairs {counts['inpaint_useful_point_pairs']}/16 | "
                f"untouched flags {counts['ordinary_flags']}/16 | "
                f"advances {reading['advanced']}"
            )


if __name__ == "__main__":
    main()
