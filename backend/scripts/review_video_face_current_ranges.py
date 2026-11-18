"""Recount exact native video ranges in the paired stage and split-screen replays."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-face-current-ranges/1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def span(values):
    a, b = values
    require(type(a) is int and type(b) is int and 0 <= a < b < 120, "Range is outside the measured native pictures")
    return a, b


def matches(ranges, labels):
    """Assign ranges one-to-one to labels at half temporal intersection over union."""
    edges = []
    for a, b in labels:
        choices = set()
        for p, q in ranges:
            intersection = max(0, min(b, q) - max(a, p))
            if 3 * intersection >= (b - a) + (q - p):
                choices.add((p, q))
        edges.append(choices)
    paired = {}

    def place(index, visited):
        for found in sorted(edges[index] - visited):
            visited.add(found)
            if found not in paired or place(paired[found], visited):
                paired[found] = index
                return True
        return False

    for index in range(len(labels)):
        place(index, set())
    return len(paired)


def stage(rows):
    counts, added_groups = Counter(), set()
    for row in rows:
        truth = list(map(span, row["truth"]))
        received = {span(value["span"]) for value in row["received"]}
        supplemented = {span(value["span"]) for value in row["supplemented"]}
        before, after = row["prior_scores"]
        require(all(item in row["supplemented"] for item in row["received"]) and before == after,
                "The second search changed earlier evidence or the appearance score")
        extra = [item for item in row["supplemented"] if item not in row["received"]]
        require(extra == row["additional"] and len(extra) <= 1, "Unexpected additional range")
        if extra:
            item, = extra
            a, b = span(item["span"])
            pins = item["anchors"]
            require(item["key"] == "face_return_secondary" and pins["before"] == a - 1
                    and a <= pins["during"] < b == pins["after"], "Additional range anchor clocks differ")
            counts["misplaced_added"] += bool(truth) and not matches([(a, b)], truth)
            if (a, b) in truth and (a, b) not in received:
                added_groups.add(row["group"])
        counts.update(files=1, edited=bool(truth), ordinary=not truth, labels=len(truth),
                      exact_before=len(received.intersection(truth)), exact_after=len(supplemented.intersection(truth)),
                      useful_before=matches(received, truth), useful_after=matches(supplemented, truth),
                      ordinary_alerts_before=not truth and bool(received), ordinary_alerts_after=not truth and bool(supplemented))
    return dict(counts, additional_exact_groups=len(added_groups))


def split(rows):
    counts, rank_groups = Counter(), set()
    for row in rows:
        frames, first, step, time_base = row["picture_clock"]
        require(frames == 120 and type(first) is int and step * Fraction(time_base) == Fraction(1, 30),
                "Unexpected split-screen clock")
        labels = list(map(span, row["truth"]))
        current = set(map(span, row["incumbent"]))
        brief = set(map(span, row["brief"]))
        require(brief <= current and row["routed"] == (not brief), "Incorrect supplemental route admission")
        measurements = row["rank"]
        for a, b in map(span, row["crossovers"]):
            expected = ((a - 2, "host"), ((a + b) // 2, "guest"), (b, "host"))
            if all(measurements[str(index)] in ([who, "guest"], [who, "host"])
                   and len(set(measurements[str(index)])) == 2 for index, who in expected):
                counts["proved_rank_crossovers"] += 1
                if row["role"] == "original":
                    rank_groups.add(row["group"])
        counts.update(files=1, edited=bool(labels), ordinary=not labels, labels=len(labels),
                      exact_incumbent=len(brief.intersection(labels)), ordinary_incumbent_flags=not labels and bool(current),
                      ordinary_brief_flags=not labels and bool(brief), routed_edited=row["routed"] and bool(labels),
                      routed_ordinary=row["routed"] and not labels)
        for arm in ("largest", "dominant"):
            proposal = set() if row[arm] is None else {span(row[arm])}
            require(not proposal or row["routed"], "Candidate ran after a brief finding")
            retained = proposal if not matches(current, list(proposal)) else set()
            counts[f"{arm}_additional_exact"] += len((retained - current).intersection(labels))
            counts[f"{arm}_additional_ordinary_flags"] += not labels and bool(retained) and not current
    return dict(counts, groups_with_real_rank_crossovers=len(rank_groups))


def review(data):
    require(data["schema"] == SCHEMA, "Unrecognized replay schema")
    retained = data["paired_serving_stage"]["rows"]
    layouts = data["split_screen_transfer"]["rows"]
    for rows, count in ((retained, 12), (layouts, 36)):
        require(len(rows) == len({row["input_sha256"] for row in rows}) == count,
                "Repeated or missing replay inputs")
    return {"paired_serving_stage": stage(retained), "split_screen_transfer": split(layouts)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), indent=2, sort_keys=True))
