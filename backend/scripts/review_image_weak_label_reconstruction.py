"""Recount the JPEG localization trial without models or source images."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("appearance", "joint", "reconstruction")
POSITIVE = {"original_outside", "vae_outside"}


def read_groups(rows, expected_groups, expected_views):
    groups = defaultdict(set)
    for row in rows:
        key = row["delivery"], row["group"]
        view = row["view"]
        if view in groups[key] or set(row["readings"]) != set(ARMS):
            raise ValueError("A trial row is duplicated or missing a measurement arm")
        groups[key].add(view)
    counts = Counter(delivery for delivery, _ in groups)
    if len(counts) != 2 or any(count != expected_groups for count in counts.values()):
        raise ValueError("The trial is missing an expected source group")
    views = {row["view"] for row in rows}
    if len(views) != expected_views or any(value != views for value in groups.values()):
        raise ValueError("One or more source groups lack a view")
    return counts


def review(receipt):
    if receipt["schema"] != "chipmunk.image-weak-label-jpeg/1" or receipt["fit_groups"] != 16:
        raise ValueError("Unexpected JPEG trial version")
    calibration, held = receipt["calibration"], receipt["held"]
    deliveries = read_groups(calibration, 8, 5)
    if deliveries.keys() != read_groups(held, 16, 7).keys():
        raise ValueError("The locked JPEG deliveries differ")
    maxima = defaultdict(dict)
    for row in calibration:
        if row["view"] in POSITIVE:
            raise ValueError("An edited image entered ordinary calibration")
        for arm, (score, _, _) in row["readings"].items():
            by_source = maxima[row["delivery"], arm]
            group = row["group"]
            by_source[group] = max(score, by_source.get(group, score))
    points = {delivery: {arm: sorted(maxima[delivery, arm].values())[-2] for arm in ARMS} for delivery in deliveries}
    if points != receipt["points"]:
        raise ValueError("The calibrated operating points differ")
    counts = defaultdict(Counter)
    prior_counts = defaultdict(Counter)
    added_locations, added_controls = defaultdict(set), defaultdict(set)
    for row in held:
        delivery = row["delivery"]
        positive = row["view"] in POSITIVE
        prior_flag, prior_iou, prior_recall = row["baseline"]
        prior = positive and prior_flag and prior_iou >= .3 and prior_recall >= .5
        prior_counts[delivery].update(positives=positive, controls=not positive, useful=prior,
                                      misplaced=positive and prior_flag and not prior, ordinary=not positive and prior_flag)
        for arm, (score, iou, recall) in row["readings"].items():
            good_rectangle = positive and iou >= .3 and recall >= .5
            flagged = score > points[delivery][arm]
            useful = flagged and good_rectangle
            extra_location = useful and not prior
            ordinary = flagged and not positive
            extra_control = ordinary and not prior_flag
            key = delivery, arm
            counts[key].update(positives=positive, controls=not positive, location_before_cutoff=good_rectangle,
                               useful=useful, additional=extra_location, ordinary=ordinary, new_ordinary=extra_control,
                               misplaced=positive and flagged and not good_rectangle)
            if extra_location:
                added_locations[key].add(row["group"])
            if extra_control:
                added_controls[key].add(row["group"])
    results = {f"{delivery}/{arm}": dict(value, source_groups=len(added_locations[delivery, arm]),
                                        added_control_groups=len(added_controls[delivery, arm]))
               for (delivery, arm), value in counts.items()}
    if results != receipt["counts"] or prior_counts != receipt["baseline_counts"]:
        raise ValueError("The reproduced JPEG counts differ from the receipt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    with gzip.open(parser.parse_args().receipt, "rt", encoding="utf-8") as stream:
        print(json.dumps(review(json.load(stream)), indent=2, sort_keys=True))
