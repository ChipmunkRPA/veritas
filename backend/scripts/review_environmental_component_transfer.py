"""Recount component flags and optionally verify every local source mixture."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.environmental-component-transfer/1"
ROLES = {"fit": 36, "selection": 20, "held": 24}
FIT = {"real", "audioldm1", "audioldm2"}
HELD = {"real", "audiogen", "audiolcm", "tangoflux"}
ANCHORS = {"fit": 4, "selection": 4, "held": 28}
SCENES = {"isolated", "genuine_6db", "genuine_balanced"}
EXTRA = {"genuine_opus_6db", "edited_6db"}
ARMS = ("mixture", "estimated_background", "both", "both_paired")
ARCHIVED = {"archived_magnitude", "archived_texture"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hash_record(record):
    raw = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return digest(raw + b"\n")


def load(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def validate_rosters(packet):
    lock, plan, acquisition, cases = (packet[key] for key in ("freeze", "source_plan", "acquisition", "cases"))
    require(packet["schema"] == lock["schema"] == SCHEMA and lock["method"]["samples"] == 48_000
            and lock["method"]["rate"] == 16_000, "Unknown audio procedure")
    require(hash_record(plan) == lock["sources"]["plan"] == acquisition["plan_sha256"]
            and hash_record(acquisition) == lock["sources"]["acquisition"]
            and hash_record(cases) == lock["cases_sha256"], "Frozen inputs differ")
    groups = {row["group"]: row for row in plan["groups"]}
    require(set(groups) == set(range(80)) and Counter(row["role"] for row in groups.values()) == ROLES
            and len({row["creator"].strip().casefold() for row in groups.values()}) == 80
            and len({row["source_id"] for row in groups.values()}) == 80, "Source groups or creator separation differ")
    expected = {(group, lane): tuple(pair) for group, row in groups.items() for lane, pair in row["parts"].items()}
    assets = {(row["group"], row["lane"]): row for row in acquisition["rows"]}
    received = {key: (row["index"], row["publisher_filename"]) for key, row in assets.items()}
    require(received == expected and len(assets) == len(acquisition["rows"]) == 264
            and len({row["pcm_sha256"] for row in assets.values()}) == 264, "Publisher source commitments differ")

    voice = {(role, item["source"], kind): item[kind] for role, rows in lock["voice_sources"].items()
             for item in rows for kind in ("genuine", "edited") if kind in item}
    observed, matches, recipe, controls = set(), set(), {}, Counter()
    for identifier, row in enumerate(cases):
        group, role, lane, scene = (row[key] for key in ("group", "role", "lane", "scene"))
        require(row["id"] == identifier, "The rendered case order differs")
        if group is None:
            key = role, row["voice"], scene.removesuffix("_only")
            require(lane == "voice_only" and key not in matches and row["pcm_sha256"] == voice[key], "Invalid voice-only control")
            matches.add(key)
            controls[role] += 1
            continue
        key = group, lane, scene
        require(role == groups[group]["role"] and key not in observed, "Duplicated or misplaced case")
        observed.add(key)
        require(row["environment_sha256"] == assets[group, lane]["pcm_sha256"], "An input references another background")
        if scene == "isolated":
            require(row["pcm_sha256"] == row["environment_sha256"], "An isolated control differs from its source")
            continue
        kind = "edited" if scene == "edited_6db" else "genuine"
        require(row["voice_sha256"] == voice[role, row["voice"], kind] and row["voice_kind"] == kind
                and row["environment_relative_db"] == (0 if scene == "genuine_balanced" else -6)
                and 0 < row["common_gain"] <= 1 and row["environment_scale"] > 0, "Unverified mixture recipe")
        entry = (row["voice_sha256"], row["common_gain"])
        require(recipe.setdefault((group, scene), entry) == entry, "The voice or final gain changes with the background")
    planned = {(group, lane, scene) for group, row in groups.items()
               for lane in (HELD if row["role"] == "held" else FIT)
               for scene in (SCENES | EXTRA if row["role"] == "held" else SCENES)}
    require(observed == planned and controls == ANCHORS and matches == set(voice)
            and len(cases) == 1020, "Coverage differs from the frozen study")
    return groups, assets


def rows_for(packet, role, rows, arms):
    planned = {row["id"]: row for row in packet["cases"] if row["role"] == role}
    seen = set()
    for row in rows:
        identifier = row["id"]
        require(identifier in planned and identifier not in seen
                and all(row.get(key) == value for key, value in planned[identifier].items()), "An outcome is missing its exact source case")
        require(set(row["scores"]) == arms and all(math.isfinite(x) for x in row["scores"].values()), "An outcome has invalid scalar readings")
        seen.add(identifier)
    require(seen == set(planned), "Incomplete comparison")


def review(packet):
    validate_rosters(packet)
    calibration, held = packet["calibration"], packet["held"]
    require(calibration["freeze_sha256"] == hash_record(packet["freeze"]), "The recorded calibration used a different freeze")
    selection, points = calibration["selection"], packet["points"]
    rows_for(packet, "selection", selection, set(ARMS))
    rows_for(packet, "held", held, set(ARMS) | ARCHIVED)
    rank = []
    for index, arm in enumerate(ARMS):
        genuine = [max(row["scores"][arm] for row in selection if row["group"] == group and row["lane"] == "real")
                   for group in {row["group"] for row in selection if row["group"] is not None}]
        sentinel = max(row["scores"][arm] for row in selection if row["lane"] == "voice_only")
        expected = {key: max(sentinel, sorted(genuine, reverse=True)[place])
                    for key, place in (("wider", 2), ("strict", 1))}
        require(points[arm] == calibration["points"][arm] == expected, "An operating point changed")
        positives = Counter(row["lane"] for row in selection if row["scene"] in ("genuine_6db", "genuine_balanced")
                            and row["scores"][arm] > expected["wider"])
        hits = [positives[lane] for lane in sorted(FIT - {"real"})]
        rank.append(((-min(hits), -sum(hits), positives["real"], index), arm))
    require(calibration["order"] == [arm for _, arm in sorted(rank)], "Selection ordering differs")
    tally = defaultdict(list)
    for row in held:
        for arm, cutoffs in points.items():
            for label, cutoff in cutoffs.items():
                tally[f"{arm}/{label}/{row['scene']}/{row['lane']}"].append(row["scores"][arm] > cutoff)
    counts = {key: {"total": len(flags), "flags": sum(flags)} for key, flags in tally.items()}
    require(packet["summary"] == counts, "Held stratum arithmetic differs")
    return {"cases": len(packet["cases"]), "selection": len(selection), "held": len(held),
            "selected": calibration["order"][0], "cells": len(counts)}


def review_native(packet, folder):
    import numpy as np

    _, assets = validate_rosters(packet)
    cases = packet["cases"]

    def pcm(path, sha256):
        raw = path.read_bytes()
        require(len(raw) == 96_000 and digest(raw) == sha256, "A local source or delivery changed")
        return np.frombuffer(raw, "<i2").astype(np.float64)

    environment = {(group, lane): pcm(folder / "environment" / f"{group:02d}-{lane}.pcm", row["pcm_sha256"])
                   for (group, lane), row in assets.items()}
    def delivery(row):
        return pcm(folder / "deliveries" / f"{row['id']:04d}.pcm", row["pcm_sha256"])

    foreground = {row["pcm_sha256"]: delivery(row) for row in cases if row["group"] is None}
    uncompressed = {(row["group"], row["lane"]): row["pcm_sha256"] for row in cases if row["scene"] == "genuine_6db"}
    worst_db, count = 0., 0
    for row in (item for item in cases if item["group"] is not None):
        mixed = delivery(row)
        scene = row["scene"]
        if scene == "isolated":
            continue
        key = row["group"], row["lane"]
        if scene == "genuine_opus_6db":
            require(uncompressed[key] == row["input_pcm_sha256"], "The Opus case has a different input")
            continue
        voice = foreground[row["voice_sha256"]] * row["common_gain"]
        added = environment[key] * row["environment_scale"] * row["common_gain"]
        require(np.max(np.abs(mixed - voice - added)) <= 0.50000001, "The delivered samples differ from the stated mixture")
        target = 10 * math.log10(float(added @ added) / float(voice @ voice))
        worst_db = max(worst_db, abs(target - row["environment_relative_db"]))
        count += 1
    require(worst_db < 1e-8 and count == 624, "Mixture-level verification is incomplete")
    return {"native_delivery_hashes": len(cases), "source_pcm_hashes": len(environment),
            "exact_mixture_checks": count, "maximum_ratio_error_db": worst_db}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    packet = load(args.receipt)
    result = review(packet)
    if args.native:
        result.update(review_native(packet, args.native))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
