"""Recount a grouped single-JPEG localization comparison without reading images."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

SCHEMA = "chipmunk.image-jpeg-rank-smoothing-groups/1"
ARMS = ("plateau", "plateau_and_turns")
QUALITIES = ("75", "90")
BANKS = {"camera": 12, "transfer": 8}
COMMON = (
    "edits", "ordinary", "baseline_correct", "baseline_ordinary",
    "geometry_possible", "geometry_with_support",
)
READINGS = (
    "assessed", "top_useful", "selected_useful", "added_correct", "added_feather", "added_ordinary",
    "wrong_edit_location", "candidate_ordinary", "same_box_as_original", "gains_repeated_by_control",
    "added_lama", "added_migan",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def counts(record, fields):
    return (set(record) == set(fields)
            and all(type(value) is int and value >= 0 for value in record.values()))


def review_points(receipt):
    calibration = receipt["calibration"]
    require(set(calibration) == set(QUALITIES), "Expected both frozen JPEG qualities")
    points = {}
    for q, group in calibration.items():
        require(group["ordinary_deliveries"] == 120 and group["source_groups"] == 12
                and set(group["arms"]) == set(ARMS), "Unexpected ordinary-calibration roster")
        points[q] = {}
        for arm, record in group["arms"].items():
            values = [float.fromhex(value) for value in record["group_maxima_hex"]]
            require(len(values) == (11 if q == "75" else 12)
                    and sorted(values) == values and all(math.isfinite(value) and 0 <= value <= 1 for value in values),
                    "Unexpected comparator-clear ordinary maxima")
            expected = values[-2].hex()
            require(record["point_hex"] == expected, "The cutoff differs from the second-largest group maximum")
            points[q][arm] = expected
    return points


def grouped_readings(groups, quality, arm, ordinary_per_group):
    nominal = [group[quality] for group in groups]
    common = [row["counts"] for row in nominal]
    results = [row["arms"][arm] for row in nominal]
    for record, result in zip(common, results, strict=True):
        require(counts(record, COMMON) and counts(result, READINGS), "A held count is missing or outside its domain")
        require(record["edits"] == 4 and record["ordinary"] == ordinary_per_group
                and 0 <= record["geometry_with_support"] <= record["geometry_possible"] <= record["edits"]
                and record["baseline_correct"] <= record["edits"]
                and record["baseline_ordinary"] <= record["ordinary"],
                "Held source membership or geometry counts differ")
        require(result["assessed"] <= record["edits"] + record["ordinary"]
                and result["same_box_as_original"] <= record["edits"]
                and result["selected_useful"] <= result["top_useful"] <= record["geometry_with_support"]
                and result["added_correct"] <= result["selected_useful"]
                and result["added_feather"] <= result["added_correct"]
                and result["gains_repeated_by_control"] <= result["added_correct"]
                and result["added_lama"] + result["added_migan"] == result["added_correct"]
                and result["wrong_edit_location"] + result["selected_useful"] <= record["edits"]
                and result["added_ordinary"] <= min(result["candidate_ordinary"], record["ordinary"] - record["baseline_ordinary"])
                and result["candidate_ordinary"] <= record["ordinary"],
                "Held decisions fail basic overlap or control arithmetic")
    total = {key: sum(row[key] for row in common) for key in COMMON}
    total.update({key: sum(row[key] for row in results) for key in READINGS})
    return total | {
        "gained_groups": sum(row["added_correct"] > 0 for row in results),
        "gained_feathered_groups": sum(row["added_feather"] > 0 for row in results),
        "added_ordinary_groups": sum(row["added_ordinary"] > 0 for row in results),
    }


def recount(receipt):
    require(set(receipt) == {"schema", "calibration", "held", "authored_fixture", "native_custody", "summary"}
            and receipt["schema"] == SCHEMA, "Unexpected grouped JPEG receipt")
    points = review_points(receipt)
    held = receipt["held"]
    require(set(held) == set(BANKS)
            and all(len(held[bank]) == size for bank, size in BANKS.items())
            and all(set(row) == set(QUALITIES) and all(set(row[q]["arms"]) == set(ARMS) for q in QUALITIES)
                    for groups in held.values() for row in groups)
            and all(groups == sorted(groups, key=lambda row: json.dumps(row, sort_keys=True)) for groups in held.values()),
            "The two held banks must use anonymous grouped counts")
    fixture = receipt["authored_fixture"]
    require(set(fixture) == set(QUALITIES) and all(
        set(fixture[q]) == {"median", "gaussian", "unchanged"} and all(
            set(readings) == set(ARMS) and all(
                set(detail) == {"score_hex", "overlapped_treatment"} and type(detail["overlapped_treatment"]) is bool
                and 0 <= float.fromhex(detail["score_hex"]) <= 1 for detail in readings.values()
            ) for readings in fixture[q].values()
        ) for q in QUALITIES
    ), "Invalid authored fixture diagnostic")
    custody = receipt["native_custody"]
    require(custody["jpeg_mask_pairs_read"] == 768 and custody["support_checked"] == 160
            and len(custody["commitments"]) == 8
            and all(re.fullmatch(r"[a-f0-9]{64}", value) for value in custody["commitments"].values()),
            "Unexpected native intake or commitment summary")
    summary = {}
    for q in QUALITIES:
        for arm in ARMS:
            banks = {name: grouped_readings(groups, q, arm, 10 if name == "camera" else 8) for name, groups in held.items()}
            totals = {key: sum(row[key] for row in banks.values()) for key in banks["camera"]}
            advance = (
                totals["gained_groups"] >= 3 and totals["gained_feathered_groups"] >= 2
                and totals["added_lama"] > 0 and totals["added_migan"] > 0
                and totals["added_ordinary_groups"] <= 2 and totals["added_ordinary"] <= 4
                and totals["wrong_edit_location"] <= 3 and totals["gains_repeated_by_control"] == 0
                and all(row["added_ordinary_groups"] <= 1 and row["gained_groups"] > 0 for row in banks.values())
            )
            summary[f"{q}/{arm}"] = {"point_hex": points[q][arm], "banks": banks, "totals": totals, "advance": advance}
    require(summary == receipt["summary"], "The displayed results differ from the grouped counts")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        result = recount(json.loads(gzip.decompress(args.receipt.read_bytes())))
        print(json.dumps(result, sort_keys=True, indent=2))
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"JPEG rank-smoothing review failed. {error}\n")
