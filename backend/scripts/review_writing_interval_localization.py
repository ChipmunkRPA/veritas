# SPDX-License-Identifier: MIT
"""Recount writing interval misses and a split ablation using numeric receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import document_passage_intervals as runtime
from app.evidence import report_passage_calibration
from app.evidence.passage_interval_selection import CAP, LONGEST, add_disjoint_ranges, select_intervals
from scripts.writing_local_baseline_transfer import proposals
from scripts.writing_multi_passage_selection import _match, _overlap
from scripts.writing_profile_calibration import recorded

SCHEMA = "chipmunk.writing-interval-localization-audit/1"
PANELS = {
    "retained": "writing-local-baseline-retained-2026-11-16.json.gz",
    "fresh": "writing-local-baseline-fresh-2026-11-16.json.gz",
    "profile_comparison": "writing-profile-calibration-2026-11-16.json.gz",
}
POINTS = {"pooled": runtime.POINT, "both": runtime.DOCUMENT_BASELINE_POINT}
MISS_CAUSES = (
    "no_sentence_range_fits",
    "no_supported_range_fits",
    "below_both_points",
    "within_lane_selection",
    "local_range_suppressed",
    "shares_selected_range_with_another_label",
)


def compact(variant, first, stop):
    rows = variant["sentences"]
    return rows[first][2], rows[stop - 1][3]


def matches(variant, ranges, targets):
    locations = [{"compact": compact(variant, first, stop)} for first, stop, *_ in ranges]
    return {target for _, target in _match(locations, targets)}


def lanes(variant, anchors):
    geometry, readings = recorded(variant)
    if readings is None:
        raise ValueError("The held comparison needs complete saved readings")
    candidates = proposals(variant | readings, anchors)
    supported = {lane: candidates[lane] for lane in POINTS}
    qualified = {lane: [row for row in choices if row[2] > POINTS[lane]]
                 for lane, choices in supported.items()}
    pooled, local = [select_intervals(choices, POINTS[lane], len(geometry))
                     for lane, choices in supported.items()]
    combined = add_disjoint_ranges(pooled, local)
    if combined != runtime._choices(geometry, readings, document_baseline=True):
        raise ValueError("Saved panels no longer replay the pooled and local selector")
    reference = report_passage_calibration.review(
        geometry, readings, {"isolated": runtime.ISOLATED, "context": runtime.CONTEXT}, runtime._report_fit(),
    )
    confirmed = report_passage_calibration.confirm(combined, reference)
    return supported, qualified, local, combined, reference, confirmed


def fits(variant, ranges, target):
    return any(_overlap(compact(variant, first, stop), target) for first, stop, *_ in ranges)


def explain_misses(variant, targets, hits, supported, qualified, local, combined):
    count = len(variant["sentences"])
    native = [(first, stop) for first in range(count)
              for stop in range(first + 1, min(first + LONGEST, count) + 1)]
    reasons = Counter(dict.fromkeys(MISS_CAUSES, 0))
    for index, target in enumerate(targets):
        if index in hits:
            continue
        if not fits(variant, native, target):
            reason = "no_sentence_range_fits"
        elif not any(fits(variant, spans, target) for spans in supported.values()):
            reason = "no_supported_range_fits"
        elif not any(fits(variant, spans, target) for spans in qualified.values()):
            reason = "below_both_points"
        elif fits(variant, combined, target):
            reason = "shares_selected_range_with_another_label"
        elif fits(variant, local, target):
            reason = "local_range_suppressed"
        else:
            reason = "within_lane_selection"
        reasons[reason] += 1
    return reasons


def _split_rank(left, right, point):
    excess_left, excess_right = left[2] - point, right[2] - point
    width = left[1] - left[0] + right[1] - right[0]
    return excess_left + excess_right, min(excess_left, excess_right), -width, -left[0], -right[0]


def split_once(qualified, confirmed, reference):
    if len(confirmed) >= CAP:
        return confirmed
    best = None
    for i, (first, stop, _, basis) in enumerate(confirmed):
        lane = "both" if basis == "document_baseline" else "pooled"
        inside = [(*span, basis) for span in qualified[lane] if first <= span[0] and span[1] <= stop]
        eligible = report_passage_calibration.confirm(inside, reference)
        for left in eligible:
            for right in eligible:
                if left[1] < right[0]:
                    proposal = _split_rank(left, right, POINTS[lane]), i, left, right
                    best = proposal if best is None else max(best, proposal)
    if best is None:
        return confirmed
    _, i, left, right = best
    return (*confirmed[:i], left, right, *confirmed[i + 1:])


def summarize(panel, anchors, *, use_saved_baseline):
    groups = defaultdict(Counter)
    misses = defaultdict(Counter)
    paired = "held" in panel
    works = panel["held"] if paired else panel["source"]["works"]
    for work in works:
        for variant in work["variants"]:
            key = work["genre"], variant["layout"]
            stats = groups[key]
            supported, qualified, local, combined, reference, confirmed = lanes(variant, anchors)
            split = split_once(qualified, confirmed, reference)
            kind = variant["version"]
            if use_saved_baseline:
                frozen = panel["decisions"][work["document"]]["addition"]["wider"][variant["layout"]][kind]
                expected = [(tuple(row["sentences"]), row["basis"]) for row in frozen]
                if [(span[:2], span[3]) for span in combined] != expected:
                    raise ValueError("Stored pooled and local ranges differ from their replay")
            if kind == "ordinary":
                stats.update({
                    "ordinary_files": 1,
                    "ordinary_alerts_pooled_and_local": bool(combined),
                    "ordinary_alerts_report_confirmation": bool(confirmed),
                    "ordinary_alerts_with_split": bool(split),
                    "ordinary_additional_ranges_with_split": len(split) - len(confirmed),
                })
                continue
            targets = [item["compact"] for item in variant["truth"]]
            base_hits = matches(variant, combined, targets)
            confirmed_hits = matches(variant, confirmed, targets)
            split_hits = matches(variant, split, targets)
            if paired:
                misses[key].update(explain_misses(variant, targets, base_hits, supported, qualified, local, combined))
            stats.update({
                "labeled_passages": len(targets),
                "localized_pooled_and_local": len(base_hits),
                "localized_report_confirmation": len(confirmed_hits),
                "localized_with_split": len(split_hits),
                "locations_gained_with_split": len(split_hits - confirmed_hits),
                "locations_lost_with_split": len(confirmed_hits - split_hits),
                "unmatched_ranges_report_confirmation": len(confirmed) - len(confirmed_hits),
                "unmatched_ranges_with_split": len(split) - len(split_hits),
            })
    return {
        "/".join(key): {"counts": dict(stats), **({"pooled_and_local_misses": dict(misses[key])} if key in misses else {})}
        for key, stats in sorted(groups.items())
    }


def recount(folder):
    panels, hashes = {}, {}
    for name, filename in PANELS.items():
        raw = (folder / filename).read_bytes()
        panels[name] = json.loads(gzip.decompress(raw))
        hashes[filename] = hashlib.sha256(raw).hexdigest()
    anchors = panels["retained"]["selection"]["anchors"]
    return {
        "schema": SCHEMA,
        "input_receipts_sha256": hashes,
        "results": {name: summarize(panel, anchors, use_saved_baseline=name != "profile_comparison")
                    for name, panel in panels.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    expected = json.loads(args.receipt.read_bytes())
    actual = recount(args.receipt.parent)
    if actual != expected:
        parser.exit(1, "Writing interval localization replay differs from its receipt.\n")
    print(json.dumps(actual, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
