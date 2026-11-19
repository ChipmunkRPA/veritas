# SPDX-License-Identifier: Apache-2.0
"""Recount frozen rendered JPEG localization comparisons without media or model dependencies."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

SCHEMA = "veritas-jpeg-diagonal-panel/1"
KNOWN = ("diagonal", "anti_diagonal", "feathered_diagonal")
ORDINARY = ("unchanged", "tone", "blur")
CONTROL = (*ORDINARY, "ordinary_axis", "ordinary_turn", "unrelated_paste", "whole_diagonal")
SHA256 = re.compile(r"[0-9a-f]{64}")
GROUP_COLUMNS = ("group", "roles", "category")
ROW_COLUMNS = ("case", "size", "truth", "sha256", "candidate", "old", "added")
CHECKS = ("copy_similarity", "copy_translation", "copy_affine")
JPEG_STAGES = ("jpeg_block_review", "jpeg_offgrid_review", "jpeg_turned_review")
GATE_FIELDS = {
    "added_correct_per_encoding", "added_camera_sources_per_encoding", "extra_ordinary_source_groups_per_encoding",
    "extra_wrong_localizations_per_encoding", "extra_whole_symmetry_per_encoding",
}
CLEAR = {"no_pair", "unsupported_geometry"}


def fail_unless(valid, description):
    if not valid:
        raise ValueError(description)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def serial(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def iou(a, b):
    x, y, w, h = a
    u, v, p, q = b
    area = max(0, min(x+w, u+p)-max(x, u))*max(0, min(y+h, v+q)-max(y, v))
    return area/(w*h+p*q-area)


def locates(pairs, truth):
    if not truth:
        return False
    return any(min(iou(a, c), iou(b, d)) >= .5 for a, b in pairs for c, d in (truth, truth[::-1]))


def pair_valid(pair, size, *, disjoint=True):
    if not isinstance(pair, list) or len(pair) != 2:
        return False
    for box in pair:
        if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
            return False
        x, y, w, h = box
        if min(x, y) < 0 or min(w, h) <= 0 or x + w > size[0] or y + h > size[1]:
            return False
    return not disjoint or iou(*pair) == 0


def witnessed(witness, size):
    if set(witness) != {"pair", "direction", "measurement"} or witness["direction"] not in {1, 2}:
        return False
    if not pair_valid(witness["pair"], size):
        return False
    a, b = witness["pair"]
    if a[2:] != b[2:][::-1] or min(a[2:]) < 32:
        return False
    metrics = witness["measurement"]
    if set(metrics) != {"interior_mean", "interior_p90", "perimeter_means"}:
        return False
    mean, p90, perimeter = (metrics[key] for key in ("interior_mean", "interior_p90", "perimeter_means"))
    return (
        math.isfinite(mean) and 0 <= mean <= 3.2 and math.isfinite(p90) and 0 <= p90 <= 6
        and len(perimeter) == 4
        and all(value is None or math.isfinite(value) and 0 <= value <= 255 for value in perimeter)
        and sum(value is not None and value + 0.00005 >= max(7, 2.5 * mean) for value in perimeter) >= 3
    )


def count(groups, rows, gate):
    fail_unless(set(gate) == GATE_FIELDS and all(type(x) is int and x >= 0 for x in gate.values()),
                "JPEG panel decision rule differs")
    schedule = [[group["group"], role, q] for group in groups for role in group["roles"] for q in (75, 90)]
    fail_unless(len(schedule) == len({tuple(case) for case in schedule}) and [row["case"] for row in rows] == schedule
                and len({row["sha256"] for row in rows}) == len(schedule), "JPEG delivery roster differs")
    reports = {}
    for q in (75, 90):
        counts, roles, sources = Counter(), {}, {}
        for row in rows:
            source, role, quality = row["case"]
            if quality != q:
                continue
            group = sources.setdefault(source, Counter())
            role_count = roles.setdefault(role, Counter())
            eligible = role in KNOWN
            size, truth = row["size"], row["truth"]
            fail_unless(role in (*KNOWN, *CONTROL) and len(size) == 2 and 384 <= min(size) <= max(size) <= 1024
                        and SHA256.fullmatch(row["sha256"]) and (pair_valid(truth, size) if eligible else truth is None),
                        "JPEG source metrics are outside their expected scope")
            before = row["old"]
            fail_unless(set(before) == set(CHECKS) and all(
                isinstance(check["pairs"], list) and all(pair_valid(pair, size, disjoint=False) for pair in check["pairs"])
                and (check["status"] == "supported_pair") == bool(check["pairs"]) for check in before.values()
            ), "An earlier image check has mismatched geometry")
            candidate = row["candidate"]
            available = candidate["pairs"]
            fail_unless(
                candidate["status"] in {"supported_pair", "no_pair", "unassessed_match_budget", "ambiguous"}
                and len(available) == int(candidate["status"] == "supported_pair")
                and all(witnessed(w, size) for w in available), "JPEG candidate geometry or replay measures differ"
            )
            admitted = bool(available and all(check["status"] in CLEAR for check in before.values())
                            and all(before["copy_translation"][key] == "no_pair" for key in JPEG_STAGES))
            fail_unless(admitted == row["added"], "JPEG candidate did not follow its earlier-check gate")
            old_pairs = [pair for check in before.values() for pair in check["pairs"]]
            extra_pairs = [pair["pair"] for pair in available] if admitted else []
            old_found = locates(old_pairs, truth)
            new_found = locates(extra_pairs, truth)
            for counter in (counts, role_count, group):
                counter["deliveries"] += 1
                counter["planted_copies"] += eligible
                counter["original_useful_locations"] += old_found
                counter["combined_useful_locations"] += old_found or new_found
                counter["added_useful_locations"] += new_found and not old_found
                counter["added_wrong_locations"] += eligible and admitted and not new_found
                counter["added_control_alerts"] += (not eligible) and admitted
                counter["added_ordinary_alerts"] += role in ORDINARY and admitted
                counter["added_whole_symmetry_alerts"] += role == "whole_diagonal" and admitted
                counter["added_transform_control_alerts"] += role not in (*ORDINARY, *KNOWN, "whole_diagonal") and admitted
        counts["added_location_sources"] = sum(bool(src["added_useful_locations"]) for src in sources.values())
        counts["added_ordinary_sources"] = sum(bool(src["added_ordinary_alerts"]) for src in sources.values())
        accepted = (
            counts["added_useful_locations"] >= gate["added_correct_per_encoding"]
            and counts["added_location_sources"] >= gate["added_camera_sources_per_encoding"]
            and counts["added_ordinary_sources"] <= gate["extra_ordinary_source_groups_per_encoding"]
            and counts["added_wrong_locations"] <= gate["extra_wrong_localizations_per_encoding"]
            and counts["added_whole_symmetry_alerts"] <= gate["extra_whole_symmetry_per_encoding"]
            and not counts["added_transform_control_alerts"]
        )
        reports[str(q)] = {"counts": dict(counts), "roles": {role: dict(c) for role, c in roles.items()}, "passes": accepted}
    return reports


def make_receipt(raw_folder):
    freeze = (raw_folder / "frozen.json").read_bytes()
    journal = (raw_folder / "rows.jsonl").read_bytes()
    plan = json.loads(freeze)
    groups = [{key: group.get(key) for key in GROUP_COLUMNS} for group in plan["roster"]]
    rows = [{key: row[key] for key in ROW_COLUMNS} for line in journal.splitlines() if (row := json.loads(line))]
    gate = plan["gates"]
    return {"schema": SCHEMA, "groups": groups, "rule": gate, "rows": rows, "closure": count(groups, rows, gate),
            "private_freeze_sha256": digest(freeze), "private_ledger_sha256": digest(journal),
            "public_rows_sha256": digest(serial(rows)), "methodology": plan["implementation"]}


def review(packet):
    fail_unless(packet["schema"] == SCHEMA and digest(serial(packet["rows"])) == packet["public_rows_sha256"],
                "JPEG receipt commitment differs")
    result = count(packet["groups"], packet["rows"], packet["rule"])
    fail_unless(result == packet["closure"], "Reported JPEG result differs from decoded numeric evidence")
    return {q: {"counts": r["counts"], "passes": r["passes"]} for q, r in result.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path, metavar="PRIVATE_RUN")
    args = parser.parse_args()
    if args.assemble:
        packet = make_receipt(args.assemble)
        print(json.dumps(review(packet), indent=2, sort_keys=True))
        with args.receipt.open("xb") as out:
            out.write(gzip.compress(serial(packet), mtime=0))
    else:
        print(json.dumps(review(json.loads(gzip.decompress(args.receipt.read_bytes()))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
