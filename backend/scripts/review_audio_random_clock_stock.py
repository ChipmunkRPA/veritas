"""Recompute the stock-voice placement outcomes from a scalar-only receipt."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median

from scripts import review_audio_stock_voice_transfer as prior

SCHEMA = "chipmunk.audio-random-clock-stock/1"
WINDOWS = prior.WINDOWS
POINT = prior.SHORT_POINT
EARLIER = prior.ROUTES["earlier_sparse"][1]
BOUNDARIES = (0, 10080, 20160, 30240, 40320, 50400, 60480, 70560, 80641)
EDIT_LENGTH = 15360
MIN_SUPPRESSION = 6
GTCRN_FRAMES = 1 + 96000 // 256
STOCK = prior.SYNTHESIS
ORDINARY = prior.CONTROLS
ROUTES = {**prior.ROUTES, "enhanced_rescue": (prior.GRID_FULL, POINT)}


def strongest(scores, slots):
    return prior.strongest({"scores": scores}, slots)


def selected(row, route):
    slots, point = ROUTES[route]
    choice = strongest(row["original_scores"], slots)
    source = "original"
    if route == "enhanced_rescue" and (choice is None or choice[0] <= point):
        enhanced = row["enhanced"]
        if enhanced is not None and enhanced["scores"] is not None:
            alternative = strongest(enhanced["scores"], slots)
            if alternative is not None and alternative[0] > point:
                choice, source = alternative, "enhanced"
    if choice is None:
        return None, source, False
    return WINDOWS[-choice[1]], source, choice[0] > point


def summary(rows, route):
    stats = Counter(dict.fromkeys(("positive", "ordinary", "isolated", "correct", "wrong_window", "no_flag_edit",
                                    "unassessed_edit", "unassessed_ordinary", "unassessed_isolated", "false_ordinary",
                                    "false_isolated", "geometry_eligible", "selected_original", "selected_enhanced"), 0))
    locations, ordinary_groups = [], set()
    for row in rows:
        truth = row["edit_clock"]
        kind = "positive" if truth else "isolated" if row["kind"] == "isolated" else "ordinary"
        stats[kind] += 1
        if truth:
            stats["geometry_eligible"] += any(prior.useful(WINDOWS[i], truth) for i in ROUTES[route][0])
        window, source, flagged = selected(row, route)
        if window is None:
            stats[f"unassessed_{'edit' if truth else kind}"] += 1
        elif flagged:
            stats[f"selected_{source}"] += 1
            if truth:
                if prior.useful(window, truth):
                    stats["correct"] += 1
                    locations.append({k: row[k] for k in ("group", "kind", "capture", "snr", "stratum", "edit_clock")}
                                     | {"window": list(window), "view": source})
                else:
                    stats["wrong_window"] += 1
            else:
                stats[f"false_{kind}"] += 1
                if kind == "ordinary":
                    ordinary_groups.add(row["group"])
        elif truth:
            stats["no_flag_edit"] += 1
    return dict(stats, ordinary_groups=len(ordinary_groups), distinct_edits=len({(x["group"], x["kind"]) for x in locations}),
                locations=locations)


def verify(receipt):
    require = prior.require
    require(receipt["schema"] == SCHEMA and prior.sha256(receipt["native_freeze_sha256"]), "Missing frozen source binding")
    require(receipt["sample_rate"] == prior.RATE and receipt["window_samples"] == [list(x) for x in WINDOWS]
            and receipt["stratum_sample_limits"] == list(BOUNDARIES), "Changed localization clock")
    require(receipt["readout_points"] == {kind: record[1] for kind, record in ROUTES.items()}
            and receipt["minimum_enhancement_suppression_db"] == MIN_SUPPRESSION, "Changed route commitments")
    rows = receipt["rows"]
    pairs = {(i, kind, snr) for i in range(8) for kind in (*ORDINARY, *STOCK) for snr in (0, -5)}
    pairs.update((f"isolated-{i}", "isolated", None) for i in range(3))
    require(len(rows) == len(pairs) == 83 and {(r["group"], r["kind"], r["snr"]) for r in rows} == pairs,
            "Missing planned source or delivery")
    require(len({r["pcm_sha256"] for r in rows}) == len(rows), "Duplicate mixtures")
    clocks, dry_commitments = {}, {}
    originals, enhanced = 0, 0
    reduced_only = []

    def scores(value):
        require(len(value) == len(WINDOWS), "Lost scheduled window")
        require(all(x is None or type(x) in (float, int) and math.isfinite(x) for x in value), "Invalid response")
        return sum(x is not None for x in value)
    for row in rows:
        require(set(row) == {"group", "capture", "kind", "snr", "stratum", "edit_clock", "ordinary_splice_clock",
                             "pcm_sha256", "dry_pcm_sha256", "original_scores", "enhanced"}, "Unexpected published field")
        require(prior.sha256(row["pcm_sha256"]) and row["capture"] in (0, 1, 2), "Invalid source hash or capture")
        kind, group, stratum = (row[k] for k in ("kind", "group", "stratum"))
        if kind == "isolated":
            require(row["dry_pcm_sha256"] is None and row["edit_clock"] is None and row["ordinary_splice_clock"] is None
                    and stratum is None and int(group[-1]) == row["capture"], "Isolated audio contains a target")
        else:
            require(prior.sha256(row["dry_pcm_sha256"]) and stratum in range(8) and row["capture"] == group % 3,
                    "Invalid foreground role")
            dry = dry_commitments.setdefault((group, kind), row["dry_pcm_sha256"])
            require(dry == row["dry_pcm_sha256"], "Matched source changed between noise levels")
            truth = row["edit_clock"] if kind in STOCK else row["ordinary_splice_clock"]
            require((row["edit_clock"] is not None) == (kind in STOCK)
                    and (row["ordinary_splice_clock"] is not None) == kind.endswith("splice"), "Case has incorrect truth role")
            if truth is not None:
                a, b = truth
                require(BOUNDARIES[stratum] <= a < BOUNDARIES[stratum + 1] and b - a == EDIT_LENGTH,
                        "Incorrect planted edit interval")
                old = clocks.setdefault(group, (stratum, a, b))
                require(old == (stratum, a, b), "A matched editor clock changed")
        originals += scores(row["original_scores"])
        candidate = row["enhanced"]
        considered = row["snr"] == -5 or kind == "isolated"
        require((candidate is not None) == considered, "Unexpected background-reduced view")
        if candidate is None:
            continue
        require(set(candidate) == {"scores", "sha256", "suppression_db", "si_sdr_gain_db"}
                and prior.sha256(candidate["sha256"]) and math.isfinite(candidate["suppression_db"]),
                "Unexpected enhancement record")
        strength = strongest(row["original_scores"], prior.GRID_FULL)
        enabled = (strength is None or strength[0] <= POINT) and candidate["suppression_db"] >= MIN_SUPPRESSION
        require((candidate["scores"] is not None) == enabled, "Wrong rescue admission")
        if enabled:
            enhanced += scores(candidate["scores"])
        if kind == "isolated":
            require(candidate["si_sdr_gain_db"] is None, "Isolated audio has a reference")
        else:
            require(math.isfinite(candidate["si_sdr_gain_db"]), "Invalid paired acoustic measurement")
            reduced_only.append(candidate)
    require(set(x[0] for x in clocks.values()) == set(range(8)), "Unbalanced true placement")
    require(receipt["classifier_forwards"] == originals + enhanced and receipt["enhancement_frame_forwards"] == 44 * GTCRN_FRAMES,
            "Incorrect inference counts")
    cohorts = {"all": rows, "0dB": [r for r in rows if r["snr"] == 0], "-5dB": [r for r in rows if r["snr"] == -5],
               "isolated": [r for r in rows if r["kind"] == "isolated"]}
    readout = {route: {label: summary(cohort, route) for label, cohort in cohorts.items()} for route in ROUTES}
    readout["geometry"] = {k: sum(any(prior.useful(WINDOWS[i], [a, b]) for i in slots) for _, a, b in clocks.values())
                           for k, (slots, _) in ROUTES.items()}
    readout["signal_audit"] = {"original_classifier_forwards": originals, "enhanced_classifier_forwards": enhanced,
                               "admitted_enhancement_cases": sum(r["enhanced"] is not None and r["enhanced"]["scores"] is not None for r in rows),
                               "median_si_sdr_gain_db": median(x["si_sdr_gain_db"] for x in reduced_only),
                               "median_suppression_db": median(x["suppression_db"] for x in reduced_only)}
    return readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    value = json.loads(args.receipt.read_text())
    readout = verify(value)
    prior.require(readout == value["summary"], "Published interpretation differs")
    print(json.dumps(readout, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
