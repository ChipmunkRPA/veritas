"""Recount the retimed video panel, with optional independent FFmpeg readback."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path


def _reproduce_anchors(path, anchors, *, count=225):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
         "frame=best_effort_timestamp", "-of", "json", str(path)], timeout=20,
    ))
    clock = [int(frame["best_effort_timestamp"]) for frame in probe["frames"]]
    hashes = subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0",
         "-filter_threads", "1", "-vf", "format=rgb24", "-fps_mode", "passthrough", "-f", "framehash",
         "-hash", "sha256", "-"], text=True, timeout=20,
    )
    observed = [line.rsplit(",", 1)[-1].strip() for line in hashes.splitlines() if not line.startswith("#")]
    assert len(clock) == len(observed) == count, path.name
    for index, tick, pixel_digest in anchors:
        assert (clock[index], observed[index]) == (tick, pixel_digest), (path.name, index)
    return digest


def review(receipt, run=None):
    counts = Counter()
    groups = set()
    examples = set()
    for row in receipt["cases"]:
        identity = row["source"], row["role"]
        assert identity not in examples
        examples.add(identity)
        pair = row["added"]
        assert pair == (None if row["current"] else row["candidate_alone"])
        truth = row["truth"]
        family = ("tandem" if row["role"].endswith("_tandem") else "primary") if truth else (
            "known_repeat" if row["role"] in {"same_speed", "reverse"} else "ordinary")
        counts[family] += 1
        counts[family + "_current_alerts"] += bool(row["current"])
        counts[family + "_added_alerts"] += pair is not None
        counts["held_picture_added_alerts"] += row["role"] == "held_picture" and pair is not None
        overlaps = []
        anchors = []
        if pair:
            assert len(pair) == 2 and sorted(interval[2] for interval in pair) == [1, 2]
            assert pair[0][1] < pair[1][0]
            assert all(0 <= start < stop < 225 for start, stop, _ in pair)
            (a, b, x), (c, d, y) = pair
            assert Fraction(b - a, x) == Fraction(d - c, y) >= 13
            edge_count = (b - a) // x
            support = row["support"]
            assert 2 * len(support) >= edge_count and len(support) >= 8
            assert support[0] == 0 and support[-1] == edge_count - 1
            assert all(1 <= j - i <= 3 for i, j in zip(support, support[1:]))
            for (start, end, _), (left, right) in zip(pair, row["anchors"], strict=True):
                assert (right[0] - left[0]) * Fraction(row["time_base"]) == Fraction(end - start, 25)
                anchors.extend(((start, *left), (end, *right)))
            if truth:
                for interval, target in zip(pair, truth, strict=True):
                    pictures = set(range(interval[0], interval[1] + 1))
                    expected = set(range(target[0], target[1] + 1))
                    overlaps.append(interval[2] == target[2] and 2 * len(pictures & expected) >= len(pictures | expected))
        correct = bool(overlaps) and all(overlaps)
        assert correct == row["useful"], identity
        counts[family + "_correct_ranges"] += correct
        counts[family + "_exact_ranges"] += correct and pair == truth
        if correct and family == "primary":
            groups.add(row["source"])
        if run is not None and anchors:
            assert _reproduce_anchors(run / row["delivery"], anchors) == row["sha256"]
            counts["independently_redecoded_pictures"] += 225
            counts["independently_verified_endpoints"] += len(anchors)
    return dict(counts) | {"primary_correct_sources": len(groups)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    print(json.dumps(review(receipt, args.media), sort_keys=True))


if __name__ == "__main__":
    main()
