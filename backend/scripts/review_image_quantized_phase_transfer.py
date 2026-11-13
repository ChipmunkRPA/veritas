"""Audit quantized phase trial decisions and native serving transfer counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

POSITIVE = {"generated_sharp", "generated_feathered", "generated_linear_promote"}
GRID = "image-native-generated-grid"
ROTATION = "image-native-rotated-stencil"
THIRD = "image-native-stencil-interpolation"
SUFFIX = "2026-11-13.json"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def read_reference(path, prefix):
    roster = path.with_name(f"{prefix}-cc0-roster-{SUFFIX}").read_bytes()
    measurements = path.with_name(f"{prefix}-measurements-{SUFFIX}").read_bytes()
    models = {entry["id"]: entry["model"].casefold() for entry in json.loads(roster) if entry["license"] == "CC0-1.0"}
    document = json.loads(measurements)
    rows = {(row["key"], row["role"]): row for row in document["rows"]}
    require(set(models) == {key for key, _ in rows} and len(set(models.values())) == len(models), "Camera roster differs")
    return models, rows, (sha(roster), sha(measurements), document["freeze_sha256"])


def review(path, serving_path):
    raw = path.read_bytes()
    ledger = json.loads(raw)
    freeze, development = ledger["freeze"], ledger["development"]
    expected_freeze = (json.dumps(freeze, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(freeze["schema"] == "chipmunk.image-quantized-phase-transfer/1"
            and development["freeze_sha256"] == sha(expected_freeze), "Frozen design differs")
    seen_models = set()
    readings = []
    counts = {}
    groups = set()
    for label, prefix, observed in (("development", GRID, development["rows"]), ("held", ROTATION, ledger["held"])):
        models, prior, commitments = read_reference(path, prefix)
        require(len(models) == 8 and not set(models.values()) & seen_models, "Camera families overlap")
        seen_models.update(models.values())
        require(commitments == tuple(freeze["sources"][label][key] for key in ("roster", "prior", "prior_lock")),
                "Original input ledger differs")
        for row in observed:
            role = row["role"]
            witness = prior.pop((row["camera"], role))
            require(all(row[k] == witness[k] for k in ("input_sha256", "truth_sha256")), "Native delivery or truth differs")
            status, lattice, score = (row[k] for k in ("status", "lattice", "score"))
            require(status in {"no_candidate", "rotation", "no_lattice", "measured"}
                    and (status == "measured") == (lattice is not None) == (score is not None), "Invalid measurement state")
            if lattice is not None:
                require(lattice[0] in (256, 257) and .95 <= lattice[1] <= 1 and math.isfinite(score) and 0 <= score <= 3,
                        "Invalid code step or phase contrast")
                if label == "development" and role in POSITIVE:
                    readings.append(score)
            if label == "held":
                flagged = score is not None and score > ledger["point"]
                counts.setdefault(role, Counter()).update(files=1, eligible=status in {"no_lattice", "measured"},
                                                          measured=score is not None, explained=flagged)
                if flagged and role == "quantized_shift":
                    groups.add(row["camera"])
        require(not prior, "Missing test delivery")
    require(readings and ledger["point"] == development["point"] == max(readings), "Frozen point does not match development")
    advances = len(groups) >= 2 and not any(counts[role]["explained"] for role in POSITIVE | {"original", "resized", "eight_bit"})
    require(counts == ledger["counts"] and len(groups) == ledger["explained_camera_groups"] and advances == ledger["advances"],
            "Held decisions differ")

    serving = json.loads(serving_path.read_bytes())
    require(serving["schema"] == "chipmunk.quantized-serving-transfer/1" and serving["base_sha256"] == sha(raw)
            and serving["native_replay"] is True, "Serving transfer binding differs")
    models, previous, third_commitments = read_reference(path, THIRD)
    require((serving.get("source_roster_sha256"), serving.get("source_measurements_sha256")) == third_commitments[:2]
            and len(serving.get("serving_sha256", {})) == 3, "Serving method or camera ledger not committed")
    require(len(models) == 13 and not set(models.values()) & seen_models, "Serving camera families overlap")
    roles = {"untouched", "quantized_bilinear", "rotated", "generated_sharp", "generated_feathered", "bilinear_diagonal",
             "bicubic_diagonal", "local_tone", "blur"}
    pending = {key for key in previous if key[1] in roles}
    serving_counts = {role: Counter() for role in roles}
    for row in serving["rows"]:
        key = row["key"], row["role"]
        pending.remove(key)
        original = previous[key]
        require(all(row[k] == original[k] for k in ("input_sha256", "truth_sha256", "truth_pixels")), "Source bytes differ")
        status, box, reading = (row[k] for k in ("status", "box", "reading"))
        require(status in {"higher_priority", "no_match", "rotation", "quantized_resampling"}
                and bool(box) == (status in {"rotation", "quantized_resampling"})
                and bool(reading) == (status == "quantized_resampling"), "Unexpected serving state")
        if box:
            x, y, x1, y1 = box
            require(all(type(n) is int and n % 16 == 0 for n in box) and 0 <= x < x1 <= 2048 and 0 <= y < y1 <= 1536
                    and (x1-x) in (64, 96, 128, 160, 192) and (y1-y) in (64, 96, 128, 160, 192)
                    and 0 <= row["intersection"] <= min((x1-x)*(y1-y), row["truth_pixels"]), "Invalid source crop")
        if reading:
            step, fraction, contrast = reading
            require(step in (256, 257) and .95 <= fraction <= 1 and ledger["point"] < contrast <= 3
                    and row["role"] == "quantized_bilinear" and row["intersection"] == row["truth_pixels"] == (x1-x)*(y1-y),
                    "Invalid lower-depth resampling localization")
        serving_counts[key[1]][status] += 1
    require(not pending and serving_counts == serving["counts"], "Serving transfer coverage or counts differ")
    return {"held_explained_cameras": len(groups), "advances": advances, "serving": serving_counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("serving", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.ledger, args.serving), sort_keys=True))
