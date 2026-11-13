# SPDX-License-Identifier: Apache-2.0
"""Verify the published PNG preview comparison without image libraries or media."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SHA = re.compile(r"[0-9a-f]{64}")
SCHEMA = "chipmunk.image-png-exif-preview-transfer/1"
GATE = {"minimum_useful_lama_feather": 6, "minimum_useful_flux_feather": 3,
        "maximum_ordinary_delivery_flags": 1, "maximum_unrelated_admitted": 0, "maximum_malformed_admitted": 0}
DELIVERY = {"ordinary", "global_tone", "resampled"}
PROFILES = DELIVERY | {"sharp", "feather", "local_blur", "local_tone", "missing", "foreign"}
BINDING = ("family", "source", "role", "sha256", "truth", "target")
STATUS = {"alignment_unresolved", "comparable", "no_retained_thumbnail", "unsupported_carrier", "unsupported_thumbnail"}


def require(valid, message):
    if not valid:
        raise ValueError(message)


def check_hash(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def counts(rows):
    totals, groups = defaultdict(Counter), defaultdict(set)
    hashes = set()
    crop_count = 0
    for row in rows:
        family, source, role, pin, mask, target = (row[key] for key in BINDING)
        status, regions, area = (row[key] for key in ("status", "regions", "area"))
        key = f"{family}/{role}"
        require(check_hash(pin) and check_hash(source) and pin not in hashes and source not in groups[key]
                and status in STATUS and type(target) is int and target >= 0
                and (check_hash(mask) if target else mask is None), "Invalid or repeated PNG input commitment")
        hashes.add(pin)
        groups[key].add(source)
        comparable = status == "comparable"
        require((check_hash(row["mask"]) if comparable else row["mask"] is None)
                and (comparable or not regions) and isinstance(regions, list), "Invalid preview assessment scope")
        for region in regions:
            require(set(region["crop_sha256"]) == {"submitted", "preview_bounds"}
                    and all(check_hash(crop) for crop in region["crop_sha256"].values()), "Missing exact crop commitments")
            for name in ("embedded_thumbnail", "submitted", "preview_bounds"):
                box = region[name]
                require(set(box) == {"x", "y", "w", "h"} and all(type(value) is int for value in box.values())
                        and min(box["x"], box["y"]) >= 0 and min(box["w"], box["h"]) > 0, "Invalid saved crop coordinates")
            crop_count += 2
        thumb, selected, overlap = (area[name] for name in ("selected_thumb", "selected_native", "native_overlap"))
        require(type(thumb) is type(overlap) is int and thumb >= 0 and 0 <= overlap <= target
                and (selected is None if not comparable or not target else type(selected) is int and selected >= overlap),
                "Invalid saved mask arithmetic")
        useful = bool(target and selected and 2 * overlap >= target and 10 * overlap >= 3 * (selected + target - overlap))
        require(useful is area["useful"], "Stored useful-mask decision differs")
        totals[key].update(total=1, comparable=comparable, flags=bool(regions), useful=useful)
    for family, size in (("lama", 12), ("flux", 6)):
        same = groups[f"{family}/ordinary"]
        require(len(same) == size and all(groups[f"{family}/{role}"] == same for role in PROFILES), "Matched photo groups differ")
    return {key: dict(tally) for key, tally in totals.items()}, crop_count


def review(panel, observations):
    rows = observations["rows"]
    require(panel["schema"] == SCHEMA and panel["gate"] == GATE and observations["freeze_sha256"] == panel["freeze_sha256"]
            and check_hash(panel["freeze_sha256"]) and len(rows) == panel["native_inputs"] == 179
            and panel["authorship_confidence_weight"] == 0, "Comparison schema, policy or freeze differs")
    totals, crop_count = counts(rows)
    malformed = [key for key in totals if key.startswith("malformed/")]
    require(len(malformed) == 15 and all(totals[key]["total"] == 1 for key in malformed)
            and sum(row["total"] for key, row in totals.items() if key.startswith("encoding/")) == 2
            and crop_count == panel["native_source_crops_verified"], "Control roster or crop count differs")
    actual = {"minimum_useful_lama_feather": totals["lama/feather"]["useful"],
              "minimum_useful_flux_feather": totals["flux/feather"]["useful"],
              "maximum_ordinary_delivery_flags": sum(row["flags"] for key, row in totals.items() if key.split("/")[1] in DELIVERY),
              "maximum_unrelated_admitted": sum(row["comparable"] for key, row in totals.items() if key.split("/")[1] in {"missing", "foreign"}),
              "maximum_malformed_admitted": sum(totals[key]["comparable"] for key in malformed)}
    passed = all(actual[key] >= n if key.startswith("minimum") else actual[key] <= n for key, n in GATE.items())
    expected = {"counts": totals, "gate_actual": actual, "passed": passed}
    require({key: panel[key] for key in expected} == expected == observations["summary"], "Published counts or frozen decision differ")
    return {"cases": len(rows), "saved_crop_hashes": crop_count, **expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("observations", type=Path)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_bytes())
    raw = args.observations.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == panel["local_result_sha256"], "Published observations do not match the original native run")
    result = review(panel, json.loads(raw))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
