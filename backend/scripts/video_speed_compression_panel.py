"""Compare a coarse retiming fallback with the current native repeat checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction
from pathlib import Path

from app.evidence import picture_sequence as binding
from scripts.video_face_temporal_fit import read, save, sha

EDITS = ("later_fast", "earlier_fast", "later_slow", "earlier_slow")
CONTROLS = ("original", "original_tandem", "in_place_speed", "jump_cut", "held_picture", "same_speed", "reverse")
CASES = ("original", *EDITS, *CONTROLS[2:], "original_tandem", *(name + "_tandem" for name in EDITS))
PARAMETERS = binding.motion_parameters(binding.MOTION_RETIMED_COMPRESSED_SCHEMA, 25)
SCHEMA = "chipmunk.retimed-compression-comparison/1"


def measure(clip, row):
    from app.detectors.video import picture_motion, picture_speed
    from app.detectors.video.decode import decode_frames, probe_frames
    from scripts.video_picture_speed_panel import LENGTH, RATE, _current, _predicted

    if sha(clip) != row["sha256"]:
        raise ValueError(f"Delivery changed for {clip.name}")
    table = probe_frames(clip)
    observations, grids = [], []
    with closing(decode_frames(clip, table, range(LENGTH))) as frames:
        for frame in frames:
            if frame.index != len(observations) or not frame.pts_verified:
                raise ValueError(f"Unverified source picture in {clip.name}")
            observations.append((frame.pts, hashlib.sha256(frame.rgb).hexdigest()))
            grids.append(picture_motion.descriptor(frame.rgb))
    if len(table) != len(observations) or len(table) != LENGTH:
        raise ValueError(f"Unexpected picture count in {clip.name}")
    other = _current(observations, grids, table)
    current = _predicted(picture_speed.find(grids, RATE)) if not other else None
    proposed = picture_speed.find(picture_speed.coarse_grid(grids), RATE, parameters=PARAMETERS) if not other and not current else None
    fallback = _predicted(proposed)
    accepted = current or fallback
    anchors = [[observations[index] for index in interval[:2]] for interval in accepted] if accepted else []
    return row | {"other_repeat": other, "current": [list(r) for r in current] if current else None,
                  "added": [list(r) for r in fallback] if fallback else None, "time_base": table.time_base,
                  "supporting_edges": proposed[2] if proposed else [], "anchors": anchors}


def useful(intervals, truth):
    def matches(a, b):
        shared = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
        occupied = max(a[1], b[1]) - min(a[0], b[0]) + 1
        return a[2] == b[2] and 2 * shared >= occupied

    return bool(intervals and truth and all(matches(a, b) for a, b in zip(intervals, truth, strict=True)))


def summary(rows):
    out = Counter()

    for row in rows:
        role = row["role"].removesuffix("_tandem")
        case = "recompressed" if row["role"].endswith("_tandem") else "direct"
        truth = row["truth"]
        section = "retimed" if truth else "ordinary" if role not in {"same_speed", "reverse"} else "other_repeat"
        original, fallback = row["current"], row["added"]
        out[f"{case}/{section}/total"] += 1
        if truth:
            out[f"{case}/{section}/baseline"] += useful(original, truth)
            out[f"{case}/{section}/combined"] += useful(original or fallback, truth)
            out[f"{case}/{section}/exact"] += (original or fallback) == truth
            out[f"{case}/{section}/added_wrong"] += fallback is not None and not useful(fallback, truth)
        else:
            out[f"{case}/{section}/baseline"] += bool(row["other_repeat"] or original)
            out[f"{case}/{section}/combined"] += bool(row["other_repeat"] or original or fallback)
    return dict(out)


def verify(receipt, media=None):
    from scripts.review_video_picture_speed_panel import _reproduce_anchors

    assert receipt["schema"] == SCHEMA and receipt["freeze"]["parameters"] == PARAMETERS
    sources = receipt["sources"]
    assert len({row["sha256"] for row in sources}) == len({row["id"] for row in sources}) == len(sources)
    planned = {(source["id"], role) for source in sources for role in CASES}
    rows = receipt["cases"]
    assert len(rows) == len(planned) and {(r["source"], r["role"]) for r in rows} == planned
    assert summary(rows) == receipt["summary"]
    checked = 0
    for row in rows:
        active = row["current"] or row["added"]
        assert not (row["added"] and (row["current"] or row["other_repeat"]))
        assert bool(row["truth"]) == (row["role"].removesuffix("_tandem") in EDITS)
        if active is None:
            assert not row["anchors"] and not row["supporting_edges"]
            continue
        first, second = active
        assert first[1] < second[0] and sorted([first[2], second[2]]) == [1, 2]
        count = Fraction(first[1] - first[0], first[2])
        assert count.denominator == 1 and count >= 13 and count == Fraction(second[1] - second[0], second[2])
        anchors = []
        for (start, end, _), (left, right) in zip(active, row["anchors"], strict=True):
            assert 0 <= start < end < 225
            assert (right[0] - left[0]) * Fraction(row["time_base"]) == Fraction(end - start, 25)
            anchors.extend(((start, *left), (end, *right)))
        if row["added"]:
            supports = row["supporting_edges"]
            assert len(supports) >= 8 and 2 * len(supports) >= count
            assert supports[0] == 0 and supports[-1] == count - 1
            assert all(1 <= b - a <= 3 for a, b in zip(supports, supports[1:]))
        if media is not None:
            assert _reproduce_anchors(media / row["delivery"], anchors) == row["sha256"]
            checked += len(anchors)
    return {"deliveries": len(rows), "sources": len(sources), "independently_reproduced_endpoints": checked,
            **receipt["summary"]}


def build(folder, roster):
    from scripts.video_picture_speed_panel import build as render

    sources = read(roster)
    if len(sources) < 8 or len({row["sha256"] for row in sources}) != len(sources):
        raise ValueError("The held recordings must have distinct acquired hashes")
    render(folder, sources, cases=CASES)


def score(folder, destination, freeze_path):
    from app.detectors.video import picture_speed

    plan = folder / "plan.json"
    frozen = read(freeze_path)
    if (frozen["plan_sha256"] != sha(plan) or frozen["code_sha256"] != sha(Path(__file__))
            or frozen["speed_sha256"] != sha(Path(picture_speed.__file__)) or frozen["parameters"] != PARAMETERS):
        raise ValueError("The locked held input or comparison changed")
    original = read(plan)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        for measured in workers.map(lambda row: measure(folder / row["delivery"], row), original["cases"]):
            rows.append(measured)
            if len(rows) % 15 == 0:
                print(f"Measured {len(rows)}/{len(original['cases'])} deliveries.", flush=True)
    values = summary(rows)
    print(json.dumps(values, sort_keys=True))
    save(destination, {"schema": SCHEMA, "freeze": frozen, "sources": original["sources"], "cases": rows, "summary": values})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    actions = cli.add_subparsers(dest="phase", required=True)
    builder = actions.add_parser("build")
    builder.add_argument("media", type=Path)
    builder.add_argument("roster", type=Path)
    scorer = actions.add_parser("score")
    scorer.add_argument("media", type=Path)
    scorer.add_argument("result", type=Path)
    scorer.add_argument("freeze", type=Path)
    verifier = actions.add_parser("verify")
    verifier.add_argument("result", type=Path)
    verifier.add_argument("--media", type=Path)
    args = cli.parse_args()
    if args.phase == "build":
        build(args.media, args.roster)
    elif args.phase == "score":
        score(args.media, args.result, args.freeze)
    else:
        print(json.dumps(verify(read(args.result), args.media), sort_keys=True))
