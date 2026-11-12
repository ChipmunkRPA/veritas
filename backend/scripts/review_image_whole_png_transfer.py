# SPDX-License-Identifier: Apache-2.0
"""Recompute the frozen PNG case counts and independently verify retained crop geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

EXPECTED = {"flux-schnell": 5, "sdxl": 5, "gemini": 6, "raw-camera": 4, "camera-jpeg-export": 8, "browser": 8}
RULE = {"minimum_total": 36, "minimum_generated_flags": 12, "minimum_per_generator": 3,
        "maximum_control_groups_flagged": 1, "maximum_raw_camera_flags": 0,
        "minimum_new_generated": 3, "minimum_new_generator_families": 2, "maximum_new_control_groups": 1}
GENERATED = {"flux-schnell", "sdxl", "gemini"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def journal(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {row["id"]: row for row in rows}
    require(len(indexed) == len(rows) == 36, "Expected one completed outcome per frozen PNG")
    return indexed


def check(freeze, cases):
    items = {row["id"]: row for row in freeze["items"]}
    require(Counter(x["family"] for x in items.values()) == EXPECTED and freeze["criteria"] == RULE
            and {row["id"] for row in cases} == items.keys() and len(cases) == len(items), "Missing or changed PNG roster")
    counts = {family: Counter(total=0, flags=0, misses=0, baseline_ai=0, baseline_record=0,
                              baseline_original=0, baseline_unavailable=0, baseline_inconclusive=0, added=0)
              for family in EXPECTED}
    source_flags, added_controls, added_families = set(), set(), set()
    complete = all(row.get("baseline_direction") is not None for row in cases)
    require(complete or not any(row.get("baseline_direction") is not None for row in cases), "Partial baseline cannot support admission")
    threshold = freeze["operating_point"]["raw_mean_logit"]
    for row in cases:
        item = items[row["id"]]
        require(row["sha256"] == item["sha256"] and item["format"] == "png", "Case is not bound to the exact PNG")
        logits, mean = row["crop_logits"], row["mean_logit"]
        require(len(logits) == 3 and all(math.isfinite(x) for x in [mean, *logits]) and abs(math.fsum(logits) / 3 - mean) <= 1e-4,
                "Invalid three-view reduction")
        flag = mean > threshold
        require(flag == row["flag"], "Recorded flag differs from the inherited point")
        family = item["family"]
        cell = counts[family]
        cell["total"] += 1
        cell["flags"] += flag
        cell["misses"] += not flag
        if family not in GENERATED and flag:
            source_flags.add(item["group"])
        if complete:
            direction = row["baseline_direction"]
            require(direction in {"ai", "record", "original", "unavailable", "inconclusive"}, "Unknown baseline outcome")
            cell[f"baseline_{direction}"] += 1
            if flag and direction not in {"ai", "record"}:
                cell["added"] += 1
                if family in GENERATED:
                    added_families.add(family)
                else:
                    added_controls.add(item["group"])
    first = (sum(counts[family]["flags"] for family in GENERATED) >= RULE["minimum_generated_flags"]
             and all(counts[family]["flags"] >= RULE["minimum_per_generator"] for family in GENERATED)
             and len(source_flags) <= RULE["maximum_control_groups_flagged"] and counts["raw-camera"]["flags"] == 0)
    final = (first and complete and sum(counts[family]["added"] for family in GENERATED) >= RULE["minimum_new_generated"]
             and len(added_families) >= RULE["minimum_new_generator_families"]
             and len(added_controls) <= RULE["maximum_new_control_groups"])
    return {"by_family": counts, "baseline_complete": complete, "candidate_gate_passed": first, "full_admission_passed": final,
            "control_groups_flagged": sorted(source_flags), "added_control_groups": sorted(added_controls),
            "added_generator_families": sorted(added_families)}


def native_audit(directory, freeze, response):
    from PIL import Image

    examined = 0
    for item in freeze["items"]:
        path = (directory / item["path"]).resolve()
        raw = path.read_bytes()
        require(path.is_relative_to(directory) and sha(raw) == item["sha256"], "Original PNG changed")
        record = response[item["id"]]
        with Image.open(path, formats=["PNG"]) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            require(list(rgb.size) == item["size"] and sha(str(rgb.size).encode() + rgb.tobytes()) == item["rgb_sha256"], "Native RGB changed")
            factor = 576 / min(width, height)
            rw, rh = round(width * factor), round(height * factor)
            view = rgb.resize((rw, rh), Image.Resampling.LANCZOS)
            cx, cy = (rw - 528) // 2, (rh - 528) // 2
            positions = ((0, cy), (cx, cy), (rw - 528, cy)) if rw >= rh else ((cx, 0), (cx, cy), (cx, rh - 528))
            for (x, y), trace, expected in zip(positions, record["traces"], record["crop_rgb_sha256"], strict=True):
                crop = (x, y, x + 528, y + 528)
                edges = [[crop[i] * (height if i % 2 else width), rh if i % 2 else rw] for i in range(4)]
                require(trace["resized_size"] == [rw, rh] and trace["resized_crop_xyxy"] == list(crop)
                        and trace["source_edge_bounds_rational"] == edges and trace["resampling"] == "LANCZOS"
                        and sha(view.crop(crop).tobytes()) == expected, "Inspected crop, source bounds or pixels changed")
                examined += 1
    require(examined == 108, "Incomplete native crop replay")
    return examined


def export(directory, output):
    frozen = read(directory / "freeze.json")
    candidate = journal(directory / "candidate.jsonl")
    completed = read(directory / "candidate-complete.json")
    input_sha = sha((directory / "freeze.json").read_bytes())
    require(completed["input_freeze_sha256"] == input_sha
            and completed["candidate_journal_sha256"] == sha((directory / "candidate.jsonl").read_bytes())
            and completed["point"] == frozen["operating_point"]["raw_mean_logit"], "Candidate result is not bound to the freeze")
    baseline = journal(directory / "baseline.jsonl") if (directory / "baseline.jsonl").exists() else None
    if baseline:
        lock = read(directory / "baseline-lock.json")
        require(lock["input_freeze_sha256"] == input_sha, "Baseline used a different freeze")
    cases = []
    for item in frozen["items"]:
        name = item["id"]
        row = candidate[name]
        require(row["input_sha256"] == item["sha256"] and row["status"] == "ok", "Candidate source or status differs")
        record = {"id": name, "sha256": item["sha256"], "crop_logits": row["crop_logits"], "mean_logit": row["mean_logit"], "flag": row["flagged"]}
        if baseline:
            base = baseline[name]
            packet = directory / "baseline-results" / f"{name}.json"
            require(base["input_sha256"] == item["sha256"] and base["status"] == "done"
                    and sha(packet.read_bytes()) == base["retained_packet_sha256"], "Baseline source or result changed")
            record["baseline_direction"] = base["interpretation"]["customer_direction"]
        cases.append(record)
    outcome = {"freeze_sha256": input_sha, "freeze": frozen, "cases": cases, "summary": check(frozen, cases),
               "native_crops_verified": native_audit(directory, frozen, candidate),
               "baseline_commit": lock["source_sha256"]["commit"] if baseline else None}
    with output.open("x") as stream:
        json.dump(outcome, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return outcome["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    if args.native:
        summary = export(args.native.resolve(), args.ledger)
    else:
        receipt = read(args.ledger)
        reconstructed = (json.dumps(receipt["freeze"], indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        require(sha(reconstructed) == receipt["freeze_sha256"], "Public freeze commitment changed")
        summary = check(receipt["freeze"], receipt["cases"])
        require(summary == receipt["summary"], "Public PNG totals differ from independent readback")
    print(json.dumps(summary, sort_keys=True), flush=True)
