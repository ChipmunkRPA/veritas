# SPDX-License-Identifier: Apache-2.0
"""Replay the public FLAME JPEG comparison from anonymous pixel and region counts."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ARMS = ("current_luma", "native", "publisher_fixed", "ordinary_point", "native_plus_flame")
EDIT_ROLES = {"sharp", "feathered"}


def useful(pixels):
    target, selected, overlap = pixels
    return bool(target and 10 * overlap >= 3 * (target + selected - overlap) and 2 * overlap >= target)


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def verify(receipt):
    rows = receipt["anonymous_measurements"]
    main, hold = ([row for row in rows if row["panel"] == name] for name in ("retained_b", "external"))
    need([row["case"] for row in rows] == list(range(1, 201)) and rows == main + hold
         and [len(main), len(hold)] == [136, 64], "Incomplete JPEG roster")
    counts = [{row["group"] for row in bank} for bank in (main, hold)]
    need(counts == [set(range(1, 16)), set(range(16, 23))], "Source groups overlap or differ")

    scores = {}
    for row in main:
        if row["role"] not in EDIT_ROLES:
            group = row["group"]
            scores[group] = max(scores.get(group, .5), max(row["component_scores"], default=.5))
    point = sorted(scores.values())[-2]
    need(len(scores) == 15 and receipt["points"] == {"ordinary_component_mean": point, "pixel": .5},
         "Ordinary source maxima do not reproduce the frozen point")
    result = {}
    for name, bank in (("retained_b", main), ("external", hold)):
        allowed = ARMS[:3] if name == "retained_b" else ARMS
        totals = {arm: {} for arm in allowed}
        ordinary = {arm: set() for arm in allowed}
        extra_useful, extra_wrong = set(), 0
        for row in bank:
            edited = row["role"] in EDIT_ROLES
            pixels = row["pixels"]
            regions = row["component_scores"]
            need(set(pixels) == set(allowed) and all(.5 < score <= 1 for score in regions) and len(regions) <= 2,
                 "Invalid response arms or component measurements")
            need(bool(regions) == bool(pixels["publisher_fixed"][1]), "Publisher component display differs")
            if name == "external":
                selected = pixels["ordinary_point"]
                native = pixels["native"]
                blended = pixels["native_plus_flame"]
                need(bool(selected[1]) == any(score > point for score in regions)
                     and blended == (native if native[1] else selected), "Additive region policy differs")
                if edited and not native[1] and blended[1]:
                    extra_wrong += not useful(blended)
                    if row["family"] == "lama" and row["role"] == "feathered" and useful(blended):
                        extra_useful.add(row["group"])
            for arm, measurement in pixels.items():
                target, flag_pixels, overlap = measurement
                need(all(type(n) is int for n in measurement)
                     and 0 <= overlap <= min(target, flag_pixels) and max(target, flag_pixels) <= row["image_pixels"]
                     and bool(target) == edited and target == pixels["native"][0], "Impossible native pixel counts")
                flagged, found = bool(flag_pixels), useful(measurement)
                for family in ("all", row["family"]):
                    prefix = f"{family}/{row['role'] if edited else 'ordinary'}"
                    totals[arm].setdefault(prefix, Counter()).update(
                        files=1, flags=flagged, useful=found, misplaced=edited and flagged and not found)
                if flagged and not edited:
                    ordinary[arm].add(row["group"])
        result[name] = {arm: {"strata": rows, "ordinary_groups": len(ordinary[arm])}
                        for arm, rows in totals.items()}
        if name == "external":
            extra_groups = len(ordinary["native_plus_flame"] - ordinary["native"])
            outcome = {"additional_feathered_lama_groups": len(extra_useful),
                       "added_ordinary_groups": extra_groups, "added_misplaced_edit_deliveries": extra_wrong}
            rule = receipt["design"]["advancement"]
            result[name]["paired"] = outcome | {
                "passes": len(extra_useful) >= rule["additional_feathered_lama_groups"]
                and extra_groups <= rule["added_ordinary_groups"] and extra_wrong <= rule["added_misplaced_edit_deliveries"],
            }
    need(result == receipt["banks"], "Recorded comparison differs from anonymous measurements")
    return {"native_source_rows": len(rows), "banks": result, "points": receipt["points"]}


def public_measurements(rows):
    groups = {}
    for start, bank in ((1, "retained_b"), (16, "external")):
        private_ids = sorted({r["group"] for r in rows if r["panel"] == bank})
        need(not groups.keys() & set(private_ids), "Private source group crosses study banks")
        groups.update({group: n for n, group in enumerate(private_ids, start)})
    return [{key: row[key] for key in ("panel", "role", "family", "component_scores")} | {
        "case": index, "group": groups[row["group"]], "image_pixels": row["measures"]["native"]["pixels"],
        "pixels": {arm: [m["reference_positive_pixels"], m["predicted_positive_pixels"], m["intersection"]]
                   for arm, m in row["measures"].items()},
    } for index, row in enumerate(rows, 1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--attach-local-ledger", type=Path,
                        help="Bind anonymous measurements from the already verified private ledger")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_bytes())
    if args.attach_local_ledger:
        ledger = args.attach_local_ledger.read_bytes()
        need(hashlib.sha256(ledger).hexdigest() == receipt["rows_sha256"], "Verified local ledger digest differs")
        receipt["anonymous_measurements"] = public_measurements(json.loads(ledger))
        receipt["portable_reader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result = verify(receipt)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    else:
        result = verify(receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
