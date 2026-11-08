"""Additional independent count/gate/bootstrap audit; private scalar packet only.

Published before diagnostic image processing. It changes no frozen inference,
point or comparison rule and does not import the primary summary calculator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def require(value, reason):
    if not value:
        raise ValueError(reason)


def close(a, b):
    require(
        math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-12, "Independent aggregate scalar differs"
    )


def calculate(rows, arm):
    ai = [r for r in rows if r["kind"] == "ai"]
    return {
        "files": len(rows),
        "edited": len(ai),
        "assessed": len(rows),
        "any_overlap": len([r for r in ai if r["metrics"][arm]["intersection"] != 0]),
        "useful_iou_at_least_03": len([r for r in ai if r["metrics"][arm]["iou"] >= 0.3]),
        "mean_edited_iou": math.fsum(r["metrics"][arm]["iou"] for r in ai) / len(ai),
        "mean_edited_pixel_f1": math.fsum(r["metrics"][arm]["pixel_f1"] for r in ai) / len(ai),
        **{
            kind + "_highlights": len(
                [r for r in rows if r["kind"] == kind and r["metrics"][arm]["predicted_positive_pixels"]]
            )
            for kind in ("original", "ordinary")
        },
    }


def compare(actual, expected):
    require(set(actual) == set(expected), "Independent aggregate fields differ")
    for key, value in expected.items():
        if type(value) is float:
            close(actual[key], value)
        else:
            require(actual[key] == value, "Independent aggregate count differs: " + key)


def review(packet, result):
    require(len(packet) == len({r["id"] for r in packet}) == 288, "Every exact diagnostic case required")
    for row in packet:
        require(
            row["kind"] in ("original", "ordinary", "ai")
            and row["format"] in ("native", "jpeg75")
            and row["label"] == int(row["kind"] == "ai"),
            "Source operation/label differs",
        )
        for value in row["metrics"].values():
            tp, fp, fn = [
                value[k] for k in ("intersection", "false_positive_pixels", "false_negative_pixels")
            ]
            require(
                all(type(v) is int and v >= 0 for v in (tp, fp, fn))
                and tp + fn == value["reference_positive_pixels"]
                and tp + fp == value["predicted_positive_pixels"]
                and (tp + fn > 0) == bool(row["label"]),
                "Native pixel denominators differ",
            )
            close(value["iou"], tp / (tp + fp + fn) if tp + fp + fn else 1.0)
            close(value["pixel_f1"], 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0)
    for fmt in ("native", "jpeg75"):
        part = [r for r in packet if r["format"] == fmt]
        groups = {r["source_id"] for r in part}
        require(
            len(part) == 144
            and len(groups) == 48
            and all(
                len([r for r in part if r["source_id"] == g]) == 3
                and {r["kind"] for r in part if r["source_id"] == g} == {"original", "ordinary", "ai"}
                for g in groups
            ),
            "Complete correlated source triplets required",
        )
        actual = result["formats"][fmt]
        values = {arm: calculate(part, arm) for arm in ("current", "whole", "zoom")}
        for arm, expected in values.items():
            compare(actual["arms"][arm], expected)
        z = values["zoom"]
        gates = {}
        for baseline in ("current", "whole"):
            b = values[baseline]
            gates[baseline + "_more_useful"] = z["useful_iou_at_least_03"] > b["useful_iou_at_least_03"]
            gates[baseline + "_higher_iou"] = z["mean_edited_iou"] > b["mean_edited_iou"]
        lost = [
            r
            for r in part
            if r["kind"] == "ai"
            and r["metrics"]["current"]["iou"] >= 0.3
            and r["metrics"]["zoom"]["iou"] < 0.3
        ]
        gates["preserve_current_useful"] = not lost
        for kind in ("original", "ordinary"):
            key = kind + "_highlights"
            gates["current_" + kind + "_budget"] = z[key] - values["current"][key] <= 1
            gates["whole_" + kind + "_budget"] = z[key] - values["whole"][key] <= 0
        gates["complete_coverage"] = all(v["assessed"] == 144 for v in values.values())
        require(
            gates == actual["gates"] and actual["qualifies_fresh_confirmation"] == all(gates.values()),
            "Independent complete qualification rules differ",
        )
        for family in ("firefly", "sd2", "flux-fill"):
            local = [r for r in part if r["family"] == family]
            require(len({r["source_id"] for r in local}) == 16, "Complete generator stratum required")
            for arm in values:
                compare(actual["families"][family][arm], calculate(local, arm))
        ai = [r for r in part if r["kind"] == "ai"]
        for baseline in ("current", "whole"):
            delta = [r["metrics"]["zoom"]["iou"] - r["metrics"][baseline]["iou"] for r in ai]
            indices = np.random.default_rng(20261108).integers(0, 48, (5000, 48))
            samples = sorted(math.fsum(delta[int(i)] for i in draw) / 48 for draw in indices)
            interval = []
            for q in (0.025, 0.975):
                position = 4999 * q
                lo, hi = math.floor(position), math.ceil(position)
                interval.append(samples[lo] + (samples[hi] - samples[lo]) * (position - lo))
            other = actual["paired_source_intervals"][baseline]
            close(other["mean_iou_difference"], math.fsum(delta) / 48)
            for a, b in zip(other["source_bootstrap_95_percentile"], interval, strict=True):
                close(a, b)
    return {
        "schema": "chipmunk.image-iml-zoom-separate-count-review/16",
        "source_groups": 48,
        "file_cases": 288,
        "arm_metric_records": 864,
        "all_counts_families_gates_and_bootstraps_match": True,
        "new_neural_forwards": 0,
        "outside_human_reviewers": 0,
        "scope": "Independent scalar count/gate/bootstrap arithmetic; native pixels and grid reconstruction are checked separately.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    packet_path, result_path = args.trial / "private-outcomes.json", args.trial / "result.json"
    packet, result = json.loads(packet_path.read_text()), json.loads(result_path.read_text())
    require(
        hashlib.sha256(packet_path.read_bytes()).hexdigest() == result["private_outcomes_sha256"],
        "Original full metric packet changed",
    )
    receipt = review(packet, result)
    receipt.update(
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
