"""Compare native playback ratios on staged edits and ordinary camera controls."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from app.evidence import picture_sequence as binding
from scripts.video_face_temporal_fit import read, save, sha

SCHEMA = "chipmunk.video-fractional-retiming-comparison/1"
FRAME_RATE, FRAME_COUNT = 25, 225
RATIOS = ((2, 3), (3, 4))
ORDINARY = ("original", "in_place_3_2", "in_place_4_3", "jump_cut", "held_picture")
EARLIER_CHECKS = ("same_speed", "reverse", "later_fast_2_1")
EDITS = tuple(f"{position}_{speed}_{long}_{short}" for short, long in RATIOS
              for position in ("earlier", "later") for speed in ("fast", "slow"))
CASES = (*ORDINARY, *EARLIER_CHECKS, *EDITS, "original_tandem", *(name + "_tandem" for name in EDITS))


def altered(pictures, early, late, role):
    import numpy as np

    from scripts.video_picture_speed_panel import _delivery

    if role == "later_fast_2_1":
        return _delivery(pictures, early, late, "later_fast")[0], None
    if role.startswith("in_place_"):
        long, short = map(int, role.removeprefix("in_place_").split("_"))
        points = early + np.arange(18 * short + 1) * long // short
        return np.concatenate((pictures[:early], pictures[points], pictures[points[-1] + 1:]))[:FRAME_COUNT], None
    if role not in EDITS:
        return _delivery(pictures, early, late, role)
    placement, speed, long, short = role.split("_")
    long, short = int(long), int(short)
    source, target = (early, late) if placement == "later" else (late, early)
    source_step, target_step = (long, short) if speed == "fast" else (short, long)
    points = source + np.arange(18 * target_step + 1) * source_step // target_step
    output = pictures[:FRAME_COUNT].copy()
    output[target:target + len(points)] = pictures[points]
    return output, sorted(((source, int(points[-1]), source_step), (target, target + len(points) - 1, target_step)))


def build(folder, roster):
    from scripts.video_picture_speed_panel import _encode, _input_pictures

    sources = read(roster)
    if (len(sources) < 8 or len({row["id"] for row in sources}) != len(sources)
            or len({row["sha256"] for row in sources}) != len(sources)):
        raise ValueError("Expected distinct camera source recordings")
    folder.mkdir()

    def render(source):
        path = Path(source["path"])
        digest = sha(path)
        if digest != source["sha256"]:
            raise ValueError(f"Source bytes differ for {source['id']}")
        pictures = _input_pictures(path, source["offset_s"])
        early, late = 25 + int(digest[:2], 16) % 16, 132 + int(digest[2:4], 16) % 16
        emitted = {}
        for role in CASES:
            name = f"{source['id']}-{role}.mp4"
            target = folder / name
            if role.endswith("_tandem"):
                before = emitted[role.removesuffix("_tandem")]
                truth = before["truth"]
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2",
                                "-i", str(folder / before["delivery"]), "-an", "-c:v", "libx264", "-threads", "2",
                                "-g", "40", "-preset", "veryfast", "-crf", "33", str(target)],
                               capture_output=True, check=True, timeout=35)
            else:
                frames, truth = altered(pictures, early, late, role)
                _encode(target, frames)
            emitted[role] = {"source": source["id"], "role": role, "delivery": name, "sha256": sha(target), "truth": truth}
        print(f"Prepared {source['id']}.", flush=True)
        return list(emitted.values())

    with ThreadPoolExecutor(max_workers=2) as workers:
        cases = [row for rows in workers.map(render, sources) for row in rows]
    save(folder / "plan.json", {"sources": [{k: v for k, v in row.items() if k != "path"} for row in sources], "cases": cases})


def commitment(plan):
    from app.detectors.video import picture_hold, picture_motion, picture_speed

    modules = (picture_hold, picture_motion, picture_speed, binding)
    return {"plan_sha256": sha(plan), "comparison_sha256": sha(Path(__file__)),
            "serving_sha256": {Path(module.__file__).name: sha(Path(module.__file__)) for module in modules},
            "parameters": binding.motion_parameters(binding.MOTION_RETIMED_FRACTIONAL_SCHEMA, FRAME_RATE),
            "required_interval_iou": .5}


def measure(folder, row):
    from app.detectors.video import picture_hold
    from app.detectors.video.decode import probe_frames

    src = folder / row["delivery"]
    if sha(src) != row["sha256"]:
        raise ValueError(f"Delivery bytes differ for {src.name}")
    table = probe_frames(src)
    cover = picture_hold.inspect(src, table, origin=0., sequence_review=True, motion_encoding=("h264", "yuv420p"),
                                 retimed_review=True, fractional_retimed_review=True)["coverage"]
    prior = any(cover.get(key, {}).get("pair") for key in ("sequence_review", binding.REVERSE_KEY,
                 "motion_sequence_review", binding.MOTION_REVERSE_KEY))
    result = cover.get(binding.MOTION_RETIMED_KEY)
    if (not prior and (not result or result["status"] == "unavailable")) or cover["frames_scanned"] != FRAME_COUNT:
        raise ValueError(f"The expected native clock was unavailable for {src.name}")
    active = result["pair"] if result else None
    fractional = result is not None and result["schema"] == binding.MOTION_RETIMED_FRACTIONAL_SCHEMA
    spans, anchors = None, []
    if active:
        spans = [[active[side]["start_frame_index"], active[side]["end_frame_index"], active["frame_steps"][side]]
                 for side in ("first", "second")]
        anchors = [[index, pts, digest] for _, _, index, pts, digest in binding.endpoints(active)]
    return row | {"earlier_repeat": prior, "current": None if fractional else spans, "added": spans if fractional else None,
                  "anchors": anchors, "time_base": table.time_base,
                  "supporting_edges": active["supporting_edge_offsets"] if active else [],
                  "completed_fractional": fractional}


def useful(intervals, truth):
    if intervals is None or truth is None:
        return False
    return all(step == target_step and 2 * (min(end, target_end) - max(start, target_start) + 1)
               >= max(end, target_end) - min(start, target_start) + 1
               for (start, end, step), (target_start, target_end, target_step) in zip(intervals, truth, strict=True))


def summary(rows):
    results = Counter()
    for row in rows:
        name = row["role"].removesuffix("_tandem")
        delivery = "second_encode" if row["role"].endswith("_tandem") else "first_encode"
        kind = name.split("_", 2)[-1] if name in EDITS else "ordinary" if name in ORDINARY else "existing_repeat"
        active = row["current"] or row["added"]
        prefix = f"{delivery}/{kind}"
        results[f"{prefix}/total"] += 1
        if row["truth"]:
            located = useful(active, row["truth"])
            results[f"{prefix}/current_located"] += useful(row["current"], row["truth"])
            results[f"{prefix}/with_fractional_located"] += located
            results[f"{prefix}/exact"] += active == row["truth"]
            results[f"{prefix}/wrong_fractional"] += row["added"] is not None and not located
        else:
            results[f"{prefix}/current_alerts"] += bool(row["current"] or row["earlier_repeat"])
            results[f"{prefix}/with_fractional_alerts"] += bool(active or row["earlier_repeat"])
    return dict(results)


def verify(receipt, media=None):
    from scripts.review_video_picture_speed_panel import _reproduce_anchors

    assert receipt["schema"] == SCHEMA and receipt["freeze"]["required_interval_iou"] == .5
    assert receipt["freeze"]["parameters"] == binding.motion_parameters(binding.MOTION_RETIMED_FRACTIONAL_SCHEMA, FRAME_RATE)
    sources, rows = receipt["sources"], receipt["cases"]
    expected = {(source["id"], role) for source in sources for role in CASES}
    assert len(rows) == len(expected) and {(row["source"], row["role"]) for row in rows} == expected
    assert len({source["sha256"] for source in sources}) == len(sources)
    assert summary(rows) == receipt["summary"]
    endpoints = 0
    for row in rows:
        role = row["role"].removesuffix("_tandem")
        assert bool(row["truth"]) == (role in EDITS)
        active = row["current"] or row["added"]
        assert not (row["earlier_repeat"] and active) and not (row["current"] and row["added"])
        assert not row["added"] or row["completed_fractional"]
        if active is None:
            assert not row["anchors"] and not row["supporting_edges"]
            continue
        assert active[0][1] < active[1][0]
        steps = tuple(sorted(span[2] for span in active))
        assert steps in (RATIOS if row["added"] else ((1, 2),))
        ticks = Fraction(row["time_base"])
        edges = (active[0][1] - active[0][0]) // active[0][2]
        assert edges >= 13 and len(row["anchors"]) == 4
        for i, (start, end, step) in enumerate(active):
            left, right = row["anchors"][2 * i:2 * i + 2]
            assert 0 <= start < end < FRAME_COUNT and end - start == step * edges
            assert (left[0], right[0]) == (start, end) and (right[1] - left[1]) * ticks == Fraction(end - start, FRAME_RATE)
        support = row["supporting_edges"]
        assert len(support) >= 8 and len(support) * 2 >= edges and support[0] == 0 and support[-1] == edges - 1
        assert all(0 < b - a <= 3 for a, b in zip(support, support[1:]))
        if media is not None:
            assert _reproduce_anchors(media / row["delivery"], row["anchors"]) == row["sha256"]
            endpoints += len(row["anchors"])
    return {"deliveries": len(rows), "sources": len(sources), "independent_frame_replays": endpoints, **receipt["summary"]}


def score(folder, freeze, output):
    plan = folder / "plan.json"
    locked = read(freeze)
    if commitment(plan) != locked:
        raise ValueError("The frozen inputs or comparison changed")
    inputs = read(plan)
    rows = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        for row in workers.map(lambda row: measure(folder, row), inputs["cases"]):
            rows.append(row)
            if len(rows) % len(CASES) == 0:
                print(f"Measured {len(rows)}/{len(inputs['cases'])} deliveries.", flush=True)
    result = {"schema": SCHEMA, "freeze": locked, "sources": inputs["sources"], "cases": rows, "summary": summary(rows)}
    print(json.dumps(verify(result), sort_keys=True))
    save(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "freeze", "score", "verify"))
    parser.add_argument("media", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--verify-native", action="store_true")
    args = parser.parse_args()
    if args.phase == "build":
        build(args.media, args.receipt)
    elif args.phase == "freeze":
        save(args.receipt, commitment(args.media / "plan.json"))
    elif args.phase == "score":
        if args.freeze is None:
            parser.error("--freeze is required for scoring")
        score(args.media, args.freeze, args.receipt)
    else:
        print(json.dumps(verify(read(args.receipt), args.media if args.verify_native else None), sort_keys=True))
