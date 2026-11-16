"""Recount mirrored-video localization and optionally verify each reported frame independently."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path

from scripts.review_video_picture_speed_panel import _reproduce_anchors


def _paired_overlap(predicted, known):
    if not predicted:
        return False
    for (start, end), (left, right) in zip(predicted, known, strict=True):
        chosen = set(range(start, end + 1))
        target = set(range(left, right + 1))
        if 2 * len(chosen & target) < len(chosen | target):
            return False
    return True


def review(receipt, media=None):
    counts = {codec: Counter() for codec in ("h264", "vp9")}
    seen = set()
    sources = {row["id"] for row in receipt["sources"]}
    roles = {"later_mirrored", "earlier_mirrored", "later_mirrored_tandem", "original", "in_place_mirror",
             "held_mirror", "same_speed", "reverse", "original_tandem"}
    for row in receipt["cases"]:
        source, codec, role = row["source"], row["codec"], row["role"]
        identity = source, codec, role
        assert identity not in seen, identity
        seen.add(identity)
        tally = counts[codec]
        pair, known = row["reflected"], row["truth"]
        conditional = "tandem" if role.endswith("_tandem") else "direct"
        if known:
            got = _paired_overlap(pair, known)
            baseline = any(_paired_overlap(spans, known) for spans in row["baseline"].values())
            assert got == row["useful"] and (pair == known) == row["exact"], identity
            tally[f"{conditional}_edits"] += 1
            tally[f"{conditional}_located_by_baseline"] += baseline
            tally[f"{conditional}_located_by_reflection"] += got
            tally[f"{conditional}_located_with_both"] += got or baseline
            tally[f"{conditional}_exact_by_reflection"] += pair == known
        else:
            assert row["useful"] is row["exact"] is False, identity
            family = "other_repeat" if role in {"same_speed", "reverse"} else "ordinary"
            tally[f"{family}_controls"] += 1
            tally[f"{family}_baseline_flags"] += bool(row["baseline"])
            tally[f"{family}_additional_flags"] += pair is not None and not row["baseline"]
            tally[f"{family}_reflection_flags"] += pair is not None
        if pair:
            (a, b), (c, d) = pair
            assert 0 <= a < b < c < d < 225 and b - a == d - c >= 13, identity
            edges = row["supporting_edge_offsets"]
            assert len(edges) >= 8 and 2 * len(edges) >= b - a and edges[0] == 0 and edges[-1] == b - a - 1, identity
            assert all(0 < y - x <= 3 for x, y in zip(edges, edges[1:])), identity
            clock = Fraction(row["time_base"])
            anchor_roles = (("first", "start"), ("first", "end"), ("second", "start"), ("second", "end"))
            anchors = row["anchors"]
            assert [(side, bound, index) for side, bound, index, *_ in anchors] == [
                (*label, index) for label, index in zip(anchor_roles, (a, b, c, d), strict=True)], identity
            assert all(tick * clock == Fraction(index, 25) for _, _, index, tick, _ in anchors), identity
            if media is not None:
                assert _reproduce_anchors(media / row["delivery"], [entry[2:] for entry in anchors]) == row["sha256"]
                tally["independent_source_frames"] += len(anchors)
        else:
            assert not row["anchors"] and not row["supporting_edge_offsets"], identity
    assert seen == set(product(sources, counts, roles)), "The panel omitted a delivery"
    return {codec: dict(tally) for codec, tally in counts.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    contents = gzip.decompress(args.receipt.read_bytes()) if args.receipt.suffix == ".gz" else args.receipt.read_bytes()
    print(json.dumps(review(json.loads(contents), args.media), indent=2, sort_keys=True))
