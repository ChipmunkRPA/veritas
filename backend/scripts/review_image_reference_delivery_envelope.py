# SPDX-License-Identifier: Apache-2.0
"""Audit sealed native region counts without executing any image detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

SCOPE = "chipmunk.image-reference-delivery-envelope/1"
NUISANCE = {"untouched_crop", "prior_jpeg", "contrast", "matched_earlier", "rotation", "fractional",
            "nonproportional", "different_photo"}
VIEWS = {"development": 144, "held": 120}
GENERATOR = {"sharp", "feathered"}
HEX = re.compile(r"[0-9a-f]{64}")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def point_id(point):
    sigma, floor = point
    return f"sigma_{sigma:g}_floor_{floor:g}"


def metrics(row, record):
    need(set(record) == {"selected", "target", "overlap", "useful", "mask_sha256"}
         and HEX.fullmatch(record["mask_sha256"]), "Invalid native response fields")
    selected, target, overlap = (record[key] for key in ("selected", "target", "overlap"))
    need(all(type(value) is int for value in (selected, target, overlap))
         and 0 <= overlap <= min(selected, target), "Impossible native pixel counts")
    need(row.get("truth_pixels", target) == target and (row["role"] not in NUISANCE or target == 0),
         "Response changes the pinned pixel truth")
    useful = target > 0 and 10 * overlap >= 3 * (selected + target - overlap) and 2 * overlap >= target
    need(useful is record["useful"], "Incorrect native overlap assessment")
    return bool(selected), useful


def validate_rows(plan, rows, phase, points):
    expected = plan["rows"][phase]
    need(len(rows) == len(expected) == VIEWS[phase], "Frozen source pair count differs")
    seen = set()
    for item, frozen in zip(rows, expected, strict=True):
        need(all(item[key] == value for key, value in frozen.items()), "Submitted file, truth or earlier copy commitment differs")
        key = item["cohort"], item["slot"], item["role"]
        need(key not in seen and all(HEX.fullmatch(item[field]) for field in ("earlier_sha256", "submitted_sha256", "truth_sha256")),
             "Invalid or repeated source role")
        seen.add(key)
        response = item["responses"]
        need(set(response) == {"current", "raw", "fixed_ten", "envelope"} and set(response["envelope"]) == points,
             "Missing or extra method responses")
        records = [response[key] for key in ("current", "raw", "fixed_ten")] + list(response["envelope"].values())
        need(len({record["target"] for record in records}) == 1, "Methods measured different truth masks")
        for record in records:
            metrics(item, record)
        need(not any(record["selected"] for record in records[1:]) or item["measured_crop"] is not None,
             "Unexpected highlights on an unassessed crop")
        overlays = item["verified_overlays"]
        need(type(overlays) is int and 0 <= overlays <= 5, "Invalid source overlay count")
        if phase == "held":
            record = next(iter(response["envelope"].values()))
            need(bool(overlays) == bool(record["selected"]), "Retained candidate has no native crop readback")
        else:
            need(overlays == 0, "Development should have no exported overlays")


def evaluate(rows, point, gate):
    strata = defaultdict(Counter)
    total = Counter()
    for row in rows:
        role, group = row["role"], row["cohort"]
        candidate_crop, expected = row["measured_crop"], row["crop"]
        exact = expected is not None and candidate_crop == expected
        total["incompatible_or_wrong_crops"] += candidate_crop is not None and not exact
        bucket = strata[f"{group}/{role}"]
        bucket.update(views=1, exact=exact, current_comparable=row["current_status"] == "comparable")
        responses = {**row["responses"], "envelope": row["responses"]["envelope"][point]}
        for method, measured in responses.items():
            highlight, overlap_passed = metrics(row, measured)
            crop_matches = expected is not None and row["current_crop"] == expected if method == "current" else exact
            useful = crop_matches and overlap_passed
            for name, value in (("useful", useful), ("highlighted", highlight), ("selected_pixels", measured["selected"]),
                                ("truth_overlap_pixels", measured["overlap"]), ("truth_pixels", measured["target"])):
                bucket[f"{method}_{name}"] += value
            if method == "envelope":
                total["nuisance_highlights"] += role in NUISANCE and highlight
                total["mislocalized_generator_views"] += role in GENERATOR and highlight and not useful
                total[f"useful_{role}"] += useful
                total["useful_public_tone"] += role == "smooth_tone" and group == "ordinary" and useful
    checks = {}
    for key, limit in gate.items():
        direction, name = key.split("_", 1)
        checks[key] = total[name] <= limit if direction == "maximum" else total[name] >= limit
    overlays = sum(row["verified_overlays"] for row in rows)
    return {"views": len(rows), "native_source_crops_verified": 2 * overlays, "native_overlays_verified": overlays,
            "strata": dict(strata), "counts": dict(total), "conditions": checks, "passed": all(checks.values())}


def audit(directory):
    plan_path, development_path, held_path = (directory / name for name in ("freeze.json", "development.json", "held.json"))
    plan, development, held = (read(path) for path in (plan_path, development_path, held_path))
    need(plan["schema"] == SCOPE and sha(plan_path) == development["freeze_sha256"] == held["freeze_sha256"]
         and sha(development_path) == held["development_sha256"], "Freeze or stage receipt does not match")
    candidates = {point_id(point): point for point in product(plan["design"]["gaussian_sigma"],
                                                              plan["design"]["minimum_excess_rgb"])}
    validate_rows(plan, development["rows"], "development", set(candidates))
    results = {point: evaluate(development["rows"], point, plan["gate"])["counts"] for point in candidates}
    need(results == development["point_counts"], "Development point counts differ")
    eligible = [point for point, result in results.items() if result["incompatible_or_wrong_crops"] == 0
                and result["nuisance_highlights"] <= plan["gate"]["maximum_nuisance_highlights"]]
    def ranking(point):
        result = results[point]
        sigma, floor = candidates[point]
        return (-result["useful_feathered"], -result["useful_sharp"], result["nuisance_highlights"], -floor, sigma)

    chosen = min(eligible, key=ranking, default=None)
    need(chosen is not None and list(candidates[chosen]) == development["selected"] == held["selected"],
         "Held parameters were not selected solely on development")
    validate_rows(plan, held["rows"], "held", {chosen})
    summary = evaluate(held["rows"], chosen, plan["gate"])
    need(summary == held["summary"], "Held aggregate differs from its independent count")
    replay = read(directory / "review.json")
    need(replay == {"verified": True, "held_sha256": sha(held_path), "summary": summary}, "Native replay receipt differs")
    return {"verified": True, "schema": SCOPE, "freeze_sha256": sha(plan_path),
            "development_sha256": sha(development_path), "held_sha256": sha(held_path),
            "native_replay_sha256": sha(directory / "review.json"), "selected": held["selected"], "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
