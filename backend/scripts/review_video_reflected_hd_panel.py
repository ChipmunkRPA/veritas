"""Recount HD mirrored-footage results and independently check original frame anchors."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path

from scripts.review_video_picture_speed_panel import _reproduce_anchors
from scripts.review_video_reflected_sequence_panel import _paired_overlap

ROLES = ("later_mirrored", "earlier_mirrored", "later_mirrored_tandem", "original",
         "in_place_mirror", "held_mirror", "same_speed", "original_tandem")


def review(suite, media=None):
    label = suite["delivery_profile"]
    method = suite["method"]
    limit = method["frame_limit"]
    period = 1 / Fraction(method["parameters"]["frame_rate"])
    candidates = {(row["source"], row["role"]) for row in suite["cases"]}
    assert len(candidates) == len(suite["cases"])
    assert candidates == set(product((source["id"] for source in suite["sources"]), ROLES))
    totals = Counter()
    for row in suite["cases"]:
        pair, truth = row["reflected"], row["truth"]
        category = ("tandem" if row["role"].endswith("_tandem") else "direct") if truth else "controls"
        totals[category] += 1
        totals[category + "_unavailable"] += row["readout"] == "unavailable"
        assert row["readout"] in {"matched", "no_match", "unavailable"}
        assert (row["readout"] == "matched") == (pair is not None)
        correct = bool(truth and _paired_overlap(pair, truth))
        assert correct == row["useful"] and row["exact"] == bool(correct and pair == truth)
        totals[category + "_correct"] += correct
        totals[category + "_exact"] += row["exact"]
        totals[category + "_flags"] += pair is not None
        if not pair:
            assert not row["anchors"] and not row["support"]
            continue
        (a, b), (c, d) = pair
        assert 0 <= a < b < c < d < limit and b - a == d - c
        edges = row["support"]
        assert len(edges) >= 8 and len(edges) * 2 >= b - a and edges[0] == 0 and edges[-1] == b - a - 1
        assert all(0 < right - left <= 3 for left, right in zip(edges, edges[1:]))
        tb = Fraction(row["time_base"])
        assert 0 < tb <= Fraction(1, 1000)
        anchors = row["anchors"]
        landmarks = (("first", "start", a), ("first", "end", b), ("second", "start", c), ("second", "end", d))
        assert [tuple(anchor[:3]) for anchor in anchors] == list(landmarks)
        for _, _, frame, pts, _ in anchors:
            assert abs((pts - row["source_first_pts"]) * tb - frame * period) < tb
        if media is not None:
            assert _reproduce_anchors(media / label / row["delivery"], [anchor[2:] for anchor in anchors], count=limit) == row["sha256"]
            totals["independently_checked_endpoints"] += len(anchors)
    return dict(totals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    contents = gzip.decompress(args.receipt.read_bytes()) if args.receipt.suffix == ".gz" else args.receipt.read_bytes()
    receipt = json.loads(contents)
    suites = receipt.get("profiles", [receipt])
    print(json.dumps({suite["delivery_profile"]: review(suite, args.media) for suite in suites}, indent=2, sort_keys=True))
