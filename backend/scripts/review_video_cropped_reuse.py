"""Recount the crop comparison and independently read back source timestamps."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from scripts.review_video_reflected_sequence_panel import _paired_overlap


def source_clock(path, sha):
    with path.open("rb") as clip:
        if hashlib.file_digest(clip, "sha256").hexdigest() != sha:
            raise ValueError(f"Delivery differs from the source receipt: {path.name}")
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=time_base:frame=pts", "-of", "json", str(path),
    ], timeout=20))
    return data["streams"][0]["time_base"], [int(row["pts"]) for row in data["frames"]]


def recount(receipt, deliveries=None):
    by_role = defaultdict(Counter)
    ids = {source["id"] for source in receipt["sources"]}
    roles = {row["role"] for row in receipt["cases"]}
    assert len({(row["source"], row["role"]) for row in receipt["cases"]}) == len(ids) * len(roles) == len(receipt["cases"])
    assert receipt["minimum_iou"] == .5
    limit = 225 if receipt["dims"] == [640, 360] else 200
    for row in receipt["cases"]:
        group = by_role[row["role"]]
        group["deliveries"] += 1
        truth, candidate = row["truth"], row["candidate"]
        if truth:
            current = any(_paired_overlap(pair, truth["ranges"]) for pair in row["current"].values())
            added = bool(candidate and _paired_overlap(candidate["ranges"], truth["ranges"]))
            group["exact"] += bool(added and candidate["ranges"] == truth["ranges"])
            group["wider_range_correct"] += bool(added and candidate["wider_range"] == truth["wider_range"])
        else:
            current, added = bool(row["current"]), bool(candidate)
        group["existing"] += current
        group["crop"] += added
        group["either"] += current or added
        if candidate is None:
            assert row["candidate_anchors"] is None
            continue
        (a, b), (c, d) = candidate["ranges"]
        assert 0 <= a < b < c < d < limit and b - a == d - c
        supports = candidate["support"]
        assert len(supports) >= 8 and supports[0] == 0 and supports[-1] == b - a - 1 and 5 * len(supports) >= 3 * (b - a)
        assert all(0 < y - x <= 3 for x, y in zip(supports, supports[1:]))
        x, y, right, bottom = candidate["crop_box"]
        assert 0 <= x < right <= 192 and 0 <= y < bottom <= 108
        anchors = row["candidate_anchors"]
        assert all((stop - start) * Fraction(row["time_base"]) == Fraction(b - a, 25) for start, stop in anchors)
        if deliveries:
            base, points = source_clock(deliveries / row["delivery"], row["sha256"])
            assert row["time_base"] == base and anchors == [[points[a], points[b]], [points[c], points[d]]]
            group["read_back_endpoints"] += 4
    return dict(by_role)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--media", type=Path)
    options = parser.parse_args()
    raw = options.receipt.read_bytes()
    report = json.loads(gzip.decompress(raw) if options.receipt.suffix == ".gz" else raw)
    suites = report.get("profiles", [report])
    print(json.dumps({
        suite["profile"]: recount(suite, options.media / suite["run"] if options.media and "run" in suite else options.media)
        for suite in suites
    }, indent=2))
