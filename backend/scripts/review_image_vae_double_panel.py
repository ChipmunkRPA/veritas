# SPDX-License-Identifier: Apache-2.0
"""Pack or independently recount the numeric JPEG localization receipt."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-vae-double-jpeg/1"
STAGES = ("fit", "calibration", "held")
DELIVERIES = ("jpeg60", "jpeg90_then_jpeg60")
ORDINARY = ("pristine", "global_vae", "local_vae", "local_blur", "local_copy")
EDITS = ("original_outside", "vae_outside")
ARMS = ("single_binary", "single_three_class", "paired_binary", "paired_three_class")
FAMILIES = ("sdxl", "kandinsky", "sd15")
IDENTITIES = ("recipient", "photo", "author", "credited_author")
PROOF = ("original", "edited", "reconstructed", "truth", "paired", "shape", "edit_pixels", "changed_outside")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha(encode(value))


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pack(directory):
    source_freeze = directory / "freeze.json"
    freeze = read(source_freeze)
    stages = {phase: read(directory / f"{phase}.json") for phase in STAGES}
    fit, cal, held = (stages[phase] for phase in STAGES)
    lock = sha(source_freeze.read_bytes())
    require(all(record["phase"] == phase and record["freeze_sha256"] == lock for phase, record in stages.items()),
            "Measurement files do not share the same source and method freeze")
    require(fit["heads"] == cal["heads"] == held["heads"] and cal["points"] == held["points"],
            "Model heads or calibration points changed after the fit")
    roster, keys = {}, {}
    for key, group in freeze["cases"].items():
        identities = {field: digest([SCHEMA, field, group["info"][field]]) for field in IDENTITIES}
        alias = identities["recipient"]
        keys[key] = alias
        roster[alias] = {"role": group["phase"], "identities": identities, "family": group["info"]["family"],
                         "geometry": {field: group["info"][field] for field in PROOF},
                         "fit_sites": group["fit_sites"], "deliveries": group["deliveries"]}
    require(len(roster) == len(keys), "Group identity duplicated")
    receipts = {phase: [{**row, "source": keys[row["source"]]} for row in stages[phase]["rows"]]
                for phase in STAGES[1:]}
    result = {
        "schema": SCHEMA, "freeze_sha256": lock, "model": freeze["model"],
        "implementation": freeze["implementation"], "versions": freeze["versions"],
        "method": {field: freeze[field] for field in ("resize", "native_cells", "tile_px", "tile_step_px", "matching", "cutoff")},
        "sources": roster, "source_commitment": digest(roster), "head_commitments": {
            arm: digest(fit["heads"][arm]) for arm in ARMS},
        "fit_counts": {arm: fit[arm] for arm in ARMS}, "points": cal["points"], "observations": receipts,
        "resource_readings": {phase: {name: stage[name] for name in ("views_measured", "seconds", "maximum_rss_bytes")}
                              for phase, stage in stages.items()},
    }
    return {**result, "summary": recount(result)}


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def planned_deliveries(role):
    variants = {"fit": ("pristine", "global_vae", *EDITS), "calibration": ORDINARY, "held": (*ORDINARY, *EDITS)}
    return {f"{delivery}/{view}" for delivery in DELIVERIES for view in variants[role]}


def verify_sources(panel):
    roster = panel["sources"]
    require(panel["schema"] == SCHEMA and digest(roster) == panel["source_commitment"], "Source commitment mismatch")
    method = panel["method"]
    require(not method["resize"] and (method["native_cells"], method["tile_px"], method["tile_step_px"]) == (14, 224, 168)
            and method["matching"] == {"iou": .30, "recall": .50}, "Geometry or matching rule changed")
    require(panel["model"]["license"] == "Apache-2.0"
            and all(is_sha256(item["sha256"]) for item in panel["model"]["files"])
            and all(is_sha256(value) for value in panel["implementation"].values())
            and is_sha256(panel["freeze_sha256"]), "Invalid model or implementation commitment")
    expected = {"fit": 16, "calibration": 8, "held": 16}
    require(Counter(row["role"] for row in roster.values()) == expected, "Incomplete source roles")
    identities_by_role = {
        role: [row["identities"] | {"original": row["geometry"]["original"]}
               for row in roster.values() if row["role"] == role]
        for role in STAGES
    }
    for name in (*IDENTITIES, "original"):
        sets = [{identity[name] for identity in rows} for rows in identities_by_role.values()]
        require(sum(map(len, sets)) == len(set.union(*sets)) == len(roster), "Source identities overlap")
    family_counts = defaultdict(Counter)
    fit_sources = fit_sites = 0
    for alias, row in roster.items():
        role, family = row["role"], row["family"]
        require(alias == row["identities"]["recipient"] and family in FAMILIES
                and all(is_sha256(value) for value in row["identities"].values()), "Incorrect source alias or family")
        family_counts[role].update([family])
        require(set(row["deliveries"]) == planned_deliveries(role), "Incorrect planned delivery scope")
        for key, (final, previous) in row["deliveries"].items():
            require(is_sha256(final) and bool(previous) == key.startswith(DELIVERIES[1] + "/")
                    and (previous is None or is_sha256(previous)), "Incorrect JPEG history commitment")
        geom = row["geometry"]
        require(all(is_sha256(geom[key]) for key in ("original", "edited", "reconstructed", "truth", "paired")),
                "Invalid pre-JPEG pixel commitment")
        height, width = geom["shape"]
        target = geom["edit_pixels"]
        require(0 < target < height * width and 0 < geom["changed_outside"] <= height * width - target
                and (role == "fit" or geom["changed_outside"] >= 4 * target), "Source pixel counts differ")
        if row["fit_sites"] is None:
            continue
        inside, outside = row["fit_sites"]
        fit_sources += 1
        fit_sites += len(inside)
        require(role == "fit" and 1 <= len(inside) == len(outside) <= 6
                and len({tuple(point) for point in inside + outside}) == len(inside) + len(outside)
                and all(0 <= y < height // 14 and 0 <= x < width // 14 for y, x in inside + outside),
                "Invalid fitted cell positions")
    require(fit_sources == 15 and fit_sites == 90, "Training group or cell counts differ")
    return expected, family_counts, fit_sources, fit_sites


def bounds(geom, view, region):
    a, b, c = (region[name] for name in ("target", "reported", "overlap"))
    target = geom["edit_pixels"] if view in EDITS else 0
    height, width = geom["shape"]
    require(a == target and 0 <= c <= min(a, b), "Impossible target/region overlap")
    if region["box"] is None:
        require(b == 0, "Area reported without a box")
    else:
        x, y, r, s = region["box"]
        require(0 <= x < r <= width and 0 <= y < s <= height and (r - x) * (s - y) == b,
                "Region is off the native pixel clock")
    union = a + b - c
    require(region["iou"] == (c / union if a else None) and region["recall"] == (c / a if a else None),
            "Reported region geometry differs")
    return a > 0 and 10 * c >= 3 * union and 2 * c >= a


def recount(panel):
    groups, families, fit_sources, fit_sites = verify_sources(panel)
    expected_arms = set(ARMS)
    require(set(panel["head_commitments"]) == set(panel["fit_counts"]) == expected_arms
            and all(is_sha256(value) for value in panel["head_commitments"].values()), "Incorrect fitted heads")
    for arm in ARMS:
        factor = 1 if arm.startswith("single_") else 2
        use = panel["fit_counts"][arm]
        require(use["tokens"] == 8 * fit_sites * factor and use["classes"] == [n * fit_sites * factor for n in (3, 3, 2)],
                "Training labels differ from the matched fit schedule")
    rows = panel["observations"]
    learned = defaultdict(dict)
    for stage in STAGES[1:]:
        scope = {(key, delivery) for key, value in panel["sources"].items() if value["role"] == stage
                 for delivery in value["deliveries"]}
        for item in rows[stage]:
            source, carrier, view = (item[key] for key in ("source", "delivery", "view"))
            roster = panel["sources"][source]
            pair = f"{carrier}/{view}"
            require((source, pair) in scope and item["sha256"] == roster["deliveries"][pair][0]
                    and item["family"] == roster["family"], "Scored source differs from the exact plan")
            scope.remove((source, pair))
            required_arms = expected_arms if stage == "calibration" else expected_arms | {"current_luma"}
            require(set(item["model"]) == required_arms, "Incomplete comparator arms")
            for arm, region in item["model"].items():
                bounds(roster["geometry"], view, region)
                if arm == "current_luma":
                    require(region["flagged"] == bool(region["box"]), "Inconsistent incumbent flag")
                    continue
                score = region["score"]
                require(math.isfinite(score), "Invalid learned score")
                if stage == "calibration":
                    prior = learned[carrier, arm].get(source, -math.inf)
                    learned[carrier, arm][source] = max(prior, score)
                else:
                    require(region["flagged"] == (score > panel["points"][carrier][arm]), "Inconsistent learned flag")
        require(not scope, "Measurements are missing")
    for (delivery, arm), controls in learned.items():
        require(len(controls) == 8 and panel["points"][delivery][arm] == sorted(controls.values())[-2],
                "Operating point differs from the ordinary control maxima")
    require({key: val["views_measured"] for key, val in panel["resource_readings"].items()}
            == {"fit": fit_sources * 8, "calibration": len(rows["calibration"]), "held": len(rows["held"])},
            "Incomplete phase intake")

    stats, false_sources, pairs = defaultdict(Counter), defaultdict(set), defaultdict(dict)
    for item in rows["held"]:
        carrier, view, source = (item[key] for key in ("delivery", "view", "source"))
        for arm, region in item["model"].items():
            positive = view in EDITS
            possible = bounds(panel["sources"][source]["geometry"], view, region)
            flag = region["flagged"]
            correct = flag and possible
            stats[carrier, arm, view if positive else "ordinary"].update(
                cases=1, flagged=flag, located=correct, top_geometry=possible, misplaced=flag and positive and not possible)
            if flag and not positive:
                false_sources[carrier, arm].add(source)
            if positive:
                pairs[carrier, arm, source][view] = correct
    context_counts = defaultdict(Counter)
    for (carrier, arm, _), views in pairs.items():
        original, processed = (views[name] for name in EDITS)
        context_counts[carrier, arm].update(original_only=original and not processed,
                                            reconstruction_only=processed and not original,
                                            both=original and processed)
    return {
        "source_groups": groups, "families_by_phase": dict(families), "fit_sources": fit_sources,
        "counts": {"/".join(key): dict(value) for key, value in stats.items()},
        "ordinary_false_groups": {"/".join(key): len(value) for key, value in false_sources.items()},
        "matched_context_outcomes": {"/".join(key): dict(value) for key, value in context_counts.items()},
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="command", required=True)
    build = modes.add_parser("build")
    build.add_argument("directory", type=Path)
    build.add_argument("output", type=Path)
    check = modes.add_parser("verify")
    check.add_argument("receipt", type=Path)
    check.add_argument("--native", type=Path, help="Compare against the portable runner's local replay directory.")
    args = cli.parse_args()
    if args.command == "build":
        receipt = pack(args.directory)
        with args.output.open("xb") as file:
            file.write(gzip.compress(encode(receipt) + b"\n", mtime=0))
    else:
        receipt = read(args.receipt)
        require(recount(receipt) == receipt["summary"], "Stored summary differs from its numeric inputs")
        if args.native:
            result = pack(args.native)
            retained = {key: value for key, value in receipt.items() if key != "resource_readings"}
            replay = {key: value for key, value in result.items() if key != "resource_readings"}
            require(retained == replay, "Native replay differs from published case commitments or readings")
    print(json.dumps(receipt["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
