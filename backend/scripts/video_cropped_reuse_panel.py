"""Evaluate a native-frame crop comparison on matched repeated and ordinary footage."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import picture_crop, picture_hold, picture_motion
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import picture_sequence as binding
from scripts.video_picture_speed_panel import _overlap, _sha, _write
from scripts.video_reflected_hd_panel import _encode, _prepared_frames

PROFILES = {"avc-360": (640, 360), "avc-720": (1280, 720)}
RATE = "25"
COUNT = 225
ROLES = (
    "center_reuse", "offset_reuse", "between_grid_reuse", "offset_reuse_tandem",
    "original", "in_place_crop", "moving_crop", "held_crop", "straight_repeat", "original_tandem",
)


def _planes(width, height):
    cursor = 0
    for columns, rows in ((width, height), (width // 2, height // 2), (width // 2, height // 2)):
        stop = cursor + columns * rows
        yield cursor, stop, columns, rows
        cursor = stop


def _transform(frames, dimensions, region, ending=None):
    transformed = np.empty_like(frames)
    for start, stop, columns, rows in _planes(*dimensions):
        block = frames[:, start:stop].reshape(-1, rows, columns)
        for i, picture in enumerate(block):
            box = (region if ending is None else
                   [a + (b - a) * i / max(len(frames) - 1, 1) for a, b in zip(region, ending, strict=True)])
            x, x2 = (round(box[j] * columns) for j in (0, 2))
            y, y2 = (round(box[j] * rows) for j in (1, 3))
            resized = cv2.resize(picture[y:y2, x:x2], (columns, rows), interpolation=cv2.INTER_CUBIC)
            transformed[i, start:stop] = resized.ravel()
    return transformed


def _region(box):
    left, top, right, bottom = box
    return (left / picture_crop.WORK_WIDTH, top / picture_crop.WORK_HEIGHT,
            right / picture_crop.WORK_WIDTH, bottom / picture_crop.WORK_HEIGHT)


def _deliveries(frames, case, geometry, dimensions, *, fractional=None):
    early, late, short, long, zoom = geometry
    if case == "original":
        return (frames,), None
    # Matching durations are fixed before scoring. Both chronological crop
    # directions and one geometry between the search grid points are tested.
    reference, target, count = early, late, short
    if case == "offset_reuse":
        reference, target, count = late, early, long
        region = zoom
    elif case == "between_grid_reuse":
        region = fractional[0] if fractional else (0.035, 0.121, 0.86, 0.946)
    elif case == "center_fractional_reuse":
        reference, target, count = late, early, long
        region = fractional[1]
    else:
        region = (0.075, 0.075, 0.925, 0.925)
    if case == "in_place_crop" or case == "moving_crop":
        reference = target
    donor = frames[reference:reference + count]
    if case == "straight_repeat":
        replacement = donor
    elif case == "held_crop":
        replacement = np.repeat(_transform(donor[:1], dimensions, region), count, axis=0)
    else:
        end_box = (0.14, 0.08, 0.92, 0.86) if case == "moving_crop" else None
        replacement = _transform(donor, dimensions, region, end_box)
    truth = ({"ranges": [[early, early + count - 1], [late, late + count - 1]],
              "wider_range": "first" if reference == early else "second", "crop_box": region}
             if case.endswith("_reuse") else None)
    return (frames[:target], replacement, frames[target + count:]), truth


def _fractional_crops(digest):
    size = (.7265, .7765, .8265, .8765, .9115)[int(digest[14:16], 16) % 5]
    x, y = [(1 - size) * (.14 + byte / 255 * .72) for byte in (int(digest[16:18], 16), int(digest[18:20], 16))]
    center = (1 - size) / 2
    return (x, y, x + size, y + size), (center, center, center + size, center + size)


def build(output, sources, profile, *, alignment_plan=False):
    dimensions = PROFILES[profile]
    output.mkdir(parents=True)
    assert len({row["sha256"] for row in sources}) == len(sources)
    cases = []
    roles = (ROLES + ("center_fractional_reuse", "center_fractional_reuse_tandem", "between_grid_reuse_tandem")
             if alignment_plan else ROLES)
    for source in sources:
        url_path = Path(source["path"])
        digest = _sha(url_path)
        if digest != source["sha256"]:
            raise ValueError(f"Source differs from ledger: {source['id']}")
        early = 24 + int(digest[:2], 16) % 18
        late = 134 + int(digest[2:4], 16) % 22
        off_center = [_region(box) for i, box in enumerate(picture_crop.crop_boxes()) if i % 9 != 4]
        zoom = off_center[int(digest[8:10], 16) % len(off_center)]
        short, long = 19 + int(digest[4:6], 16) % 5, 27 + int(digest[6:8], 16) % 6
        geometry = early, late, short, long, zoom
        with tempfile.TemporaryDirectory(prefix=".frames-", dir=output) as spool:
            frames = _prepared_frames(source, Path(spool) / "source.yuv", *dimensions, RATE, COUNT)
            priors = {}
            for case in roles:
                filename = f"{source['id']}-{case}.mp4"
                parent = case.removesuffix("_tandem")
                if parent == case:
                    segments, truth = _deliveries(frames, case, geometry, dimensions,
                                                   fractional=_fractional_crops(digest) if alignment_plan else None)
                    prior = None
                else:
                    original, truth = priors[parent]
                    prior, segments = output / original, ()
                _encode(output / filename, dimensions, RATE, "h264", segments, prior)
                priors[case] = filename, truth
                cases.append({"source": source["id"], "role": case, "delivery": filename,
                              "sha256": _sha(output / filename), "truth": truth})
            del frames
        print(f"{profile} prepared {source['id']}", flush=True)
    _write(output / "plan.json", {
        "profile": profile, "dims": list(dimensions), "minimum_iou": .5,
        "candidate_sha256": _sha(Path(picture_crop.__file__)),
        "sources": [{k: v for k, v in row.items() if k != "path"} for row in sources], "cases": cases,
        **({"alignment_plan": True} if alignment_plan else {}),
    })


def ranges(pair):
    return [[pair[side]["start_frame_index"], pair[side]["end_frame_index"]] for side in ("first", "second")]


def _correct(intervals, truth, minimum_iou):
    return bool(intervals and truth and all(_overlap(actual, known) >= minimum_iou
                                          for actual, known in zip(intervals, truth["ranges"], strict=True)))


def _measure(clip, table):
    profile = picture_motion.profile(table, ("h264", "yuv420p"), schema=binding.MOTION_REFLECTED_HD_SCHEMA)
    limit = profile[1] if profile else len(table)
    with closing(decode_frames(clip, table, range(limit), timeout_s=20)) as decoded:
        frames = [picture_crop.luminance(frame.rgb) for frame in decoded if frame.pts_verified]
    if len(frames) != limit:
        raise ValueError(f"Incomplete native frames for {clip.name}")
    return picture_crop.find(frames, int(RATE))


def score(output, result):
    plan = json.loads((output / "plan.json").read_text())
    if plan["candidate_sha256"] != _sha(Path(picture_crop.__file__)):
        raise ValueError("Candidate changed after delivery freeze")
    cases = []
    for row in plan["cases"]:
        clip = output / row["delivery"]
        table = probe_frames(clip)
        assert _sha(clip) == row["sha256"] and len(table) == COUNT and not table.interpolated
        assert [table.width, table.height] == plan["dims"]
        encoding = ("h264", "yuv420p")
        existing = picture_hold.inspect(clip, table, origin=0., sequence_review=True, motion_encoding=encoding,
                                       retimed_review=True, reflected_review=True)["coverage"]
        keys = ("sequence_review", binding.REVERSE_KEY, *(key for key, _, _ in binding.MOTION_REVIEWS))
        current = {key: ranges(existing[key]["pair"]) for key in keys if existing.get(key, {}).get("pair")}
        start = time.monotonic()
        candidate = _measure(clip, table)
        elapsed = round(time.monotonic() - start, 3)
        clock = [[table.pts[i], table.pts[j]] for i, j in candidate.ranges] if candidate else None
        cases.append(row | {"current": current, "candidate": candidate._asdict() if candidate else None,
                            "candidate_anchors": clock, "time_base": table.time_base, "seconds": elapsed})
        print(clip.name, [key for key, r in current.items() if _correct(r, row["truth"], plan["minimum_iou"])],
              candidate.ranges if candidate else None, _correct(candidate.ranges if candidate else None, row["truth"], plan["minimum_iou"]),
              elapsed, flush=True)
    _write(result, plan | {"cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("build")
    prepare.add_argument("output", type=Path)
    prepare.add_argument("sources", type=Path)
    prepare.add_argument("profile", choices=PROFILES)
    prepare.add_argument("--alignment-plan", action="store_true")
    measure = commands.add_parser("score")
    measure.add_argument("output", type=Path)
    measure.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output, json.loads(args.sources.read_text()), args.profile, alignment_plan=args.alignment_plan)
    else:
        score(args.output, args.result)
