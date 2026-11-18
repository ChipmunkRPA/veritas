# SPDX-License-Identifier: Apache-2.0
"""Recount matched square-video coverage with the Python standard library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def coverage(rows, point):
    outcomes = Counter()
    correct, sources = set(), set()
    identities = set()
    for row in rows:
        target, proposal = row["truth"], row["proposal"]
        key = row["group"], row["kind"]
        if key in identities:
            raise ValueError("A source treatment was counted twice")
        identities.add(key)
        scope = "edit" if target is not None else "ordinary"
        outcomes[f"{scope}_files"] += 1
        outcomes[f"{scope}_assessed"] += proposal is not None
        if proposal is None or float.fromhex(proposal["score"]) <= point:
            continue
        if proposal["cuts"]:
            outcomes["guard_vetoes"] += 1
            continue
        if target is None:
            outcomes["ordinary_flags"] += 1
            continue
        start, end = proposal["range"]
        common = max(0, min(end, target[1]) - max(start, target[0]))
        total = end - start + target[1] - target[0] - common
        if common * 2 >= total:
            outcomes["located"] += 1
            correct.add(key)
            sources.add(row["group"])
        else:
            outcomes["misplaced"] += 1
    return outcomes, correct, sources


def labeled(rows):
    return {(row["group"], row["kind"]): row["truth"] for row in rows}


def review(plan_path, portrait_path, result_path):
    plan, earlier, result = (read(path) for path in (plan_path, portrait_path, result_path))
    if (digest(plan_path) != result["plan_sha256"] or digest(portrait_path) != plan["held_sha256"]
            or plan["schema"] != result["schema"] or plan["minimum_temporal_iou"] != .5):
        raise ValueError("The evaluation inputs differ from their frozen commitments")
    point = float.fromhex(plan["point"])
    rows_by_arm = {name: [row for group in groups for row in group["rows"]] for name, groups in result["cohorts"].items()}
    rows_by_arm["portrait"] = [row for row in earlier["banks"]["640"]["rows"] if row["delivery"] == "direct"]
    if labeled(rows_by_arm["portrait"]) != labeled(rows_by_arm["paired"]):
        raise ValueError("The shape comparison has mismatched source cases")
    checks = {name: coverage(rows, point) for name, rows in rows_by_arm.items()}
    fields = ("edit_files", "edit_assessed", "ordinary_files", "ordinary_assessed", "ordinary_flags", "located", "misplaced", "guard_vetoes")
    compact = {name: {key: data[0][key] for key in fields} for name, data in checks.items()}
    for name, scores in compact.items():
        if any(value != result["summary"][name][key] for key, value in scores.items()):
            raise ValueError("Recomputed case counts differ")
    old, new = checks["portrait"][1], checks["paired"][1]
    paired = {"paired_retained": len(old & new), "paired_lost": len(old - new), "paired_gained": len(new - old)}
    if any(result["summary"][key] != value for key, value in paired.items()):
        raise ValueError("Recomputed paired counts differ")
    print(json.dumps(compact | paired, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("plan_path", "portrait_path", "result_path"):
        parser.add_argument(field, type=Path)
    review(**vars(parser.parse_args()))
