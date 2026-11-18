# SPDX-License-Identifier: MIT
"""Freeze, compare and audit localized eye holds on verified source clocks."""
from __future__ import annotations

import argparse
import hashlib
import statistics
import subprocess
import tempfile
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.video_cropped_reverse_panel import load, save, sha

PROFILE, COUNT, RATE = (480, 360), 50, 25
SCHEMA = "chipmunk.eye-region-hold-evaluation/1"
EDITS = ("left_hold", "right_hold", "right_hold_tandem")
CONTROLS = ("original", "original_tandem", "whole_hold", "picture_jump", "eye_blur", "eye_gain")
PLACEMENT = {"left_hold": (0, 16, 27), "right_hold": (1, 28, 41), "right_hold_tandem": (1, 28, 41)}
OFFSETS = (.48, 1.48, 2.48)
GATE = {"min_added_useful": 15, "min_groups": 6, "max_new_controls": 2, "max_misplaced": 2,
        "min_native_coverage": .80, "max_new_global_holds_or_cuts": 0}


def ff(tool, *args, input_data=None):
    return subprocess.run([tool, *args], input=input_data, capture_output=True, check=True, timeout=40).stdout


def _locator():
    from app.detectors.image.faces import load_yunet
    from app.detectors.video.mouth_local_hold import local_detector_path

    path = local_detector_path()
    if path is None:
        raise ValueError("The pinned local face locator is missing")
    return load_yunet(path), sha(path)


def _prepare(source, offset):
    import numpy as np

    w, h = PROFILE
    raw = ff("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", "-ss", str(offset),
             "-i", str(source), "-map", "0:v:0", "-an", "-vf",
             f"fps={RATE},scale={w}:{h}:force_original_aspect_ratio=increase:force_divisible_by=2,crop={w}:{h},"
             "setsar=1,format=yuv420p", "-frames:v", str(COUNT), "-f", "rawvideo", "-threads:v", "1", "pipe:1")
    if len(raw) != COUNT * (w * h * 3 // 2):
        raise ValueError(f"Too few camera frames at offset {offset}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(COUNT, -1)


def _authoring_geometry(frames, detector):
    import cv2

    from scripts.video_eye_hold_compare import centered_box, geometry

    w, h = PROFILE
    images = [cv2.cvtColor(item.reshape(h * 3 // 2, w), cv2.COLOR_YUV2RGB_I420) for item in frames]
    stable = geometry(images, detector)
    if stable is None:
        return None
    eyes, scale = stable
    boxes = [centered_box(x, y, .60 * scale, .43 * scale, (h, w)) for x, y in eyes]
    return [list(box) for box in boxes] if all(boxes) and all(min(box[2:]) >= 12 for box in boxes) else None


def _pins():
    backend = Path(__file__).resolve().parents[1]
    paths = ("scripts/video_eye_hold_panel.py", "scripts/video_eye_hold_compare.py",
             "app/detectors/video/mouth_local_hold.py", "app/detectors/video/picture_hold.py")
    return {Path(name).name: sha(backend / name) for name in paths}


def _cases(groups):
    return [
        {"group": group["group"], "index": group["index"], "split": group["role"], "role": role,
         "delivery": f"{group['index']:02d}-{role}.mp4",
         "truth": {"eye": side, "frames": [start, stop], "box": group["eye_boxes"][side]}
         if role in EDITS and group["eye_boxes"] else None}
        for group in groups for role in (*CONTROLS, *EDITS)
        if (role not in EDITS and role not in {"eye_blur", "eye_gain"}) or group["eye_boxes"]
        for side, start, stop in (PLACEMENT.get(role, (0, 0, 0)),)
    ]


def freeze(roster, sources, output):
    import cv2

    cv2.setNumThreads(1)
    value = load(roster)
    detector, weight = _locator()
    groups = []
    for original in value["groups"]:
        path = sources / f"{original['index']:02d}-ordinary.mp4"
        if sha(path) != original["files"]["ordinary"]["sha256"]:
            raise ValueError("The original video differs from its pinned publisher copy")
        boxes = None
        for offset in OFFSETS:
            frames = _prepare(path, offset)
            boxes = _authoring_geometry(frames, detector)
            if boxes:
                break
        groups.append(original | {"offset": offset, "eye_boxes": boxes})
        print(original["index"], original["role"], bool(boxes), flush=True)
    prepared = Counter(row["role"] for row in groups if row["eye_boxes"])
    if prepared["development"] < 2 or prepared["held"] < 8:
        raise ValueError(f"Too few captures with authorable eye geometry: {prepared}")
    roster = value | {"groups": groups}
    cases = _cases(groups)
    save(output, {"schema": SCHEMA, "roster": roster, "cases": cases, "gate": GATE, "method_sha256": _pins(),
                  "geometry_weight_sha256": weight, "roles": {"edited": EDITS, "control": CONTROLS}})
    print("Frozen", len(cases), "deliveries with plan SHA-256", sha(output), flush=True)


def _vary(frames, box, role, start, stop):
    import cv2
    import numpy as np

    from scripts.video_cropped_reuse_panel import _planes

    updated = frames.copy()
    for row_start, row_stop, w, h in _planes(*PROFILE):
        factor = PROFILE[0] // w
        x, y, rw, rh = [int(round(v / factor)) for v in box]
        plane = updated[:, row_start:row_stop].reshape(-1, h, w)
        selected = plane[start:stop, y:y + rh, x:x + rw]
        raw = plane[start - 4, y:y + rh, x:x + rw]
        if role == "eye_blur":
            replacement = np.stack([cv2.GaussianBlur(image, (5, 5), 1.3) for image in selected])
        elif role == "eye_gain":
            replacement = np.uint8(np.minimum(255, selected.astype(np.int16) + 12)) if factor == 1 else selected
        else:
            replacement = raw
        edge_y = np.minimum(np.arange(rh), np.arange(rh)[::-1])[:, None]
        edge_x = np.minimum(np.arange(rw), np.arange(rw)[::-1])[None, :]
        alpha = np.minimum(np.minimum(edge_x, edge_y) / max(1, 4 // factor), 1.)
        selected[:] = np.uint8(np.rint(alpha * replacement + (1 - alpha) * selected))
    return updated


def _encode(frames, role, boxes, output):
    import numpy as np

    w, h = PROFILE
    if role.endswith("_tandem"):
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            prior = Path(temporary) / "base.mp4"
            _encode(frames, role.removesuffix("_tandem"), boxes, prior)
            _h264(output, ("-i", str(prior)), crf=34)
        return
    if role in EDITS or role in {"eye_blur", "eye_gain"}:
        side, start, stop = PLACEMENT.get(role, PLACEMENT["left_hold"])
        frames = _vary(frames, boxes[side], role, start, stop)
    elif role in {"whole_hold", "picture_jump"}:
        frames = frames.copy()
        if role == "whole_hold":
            frames[16:27] = np.repeat(frames[12:13], 11, axis=0)
        else:
            frames[16:27] = frames[38:49]
    _h264(output, ("-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", f"{w}x{h}",
                   "-framerate", str(RATE), "-i", "pipe:0"), crf=22, data=memoryview(frames))


def _h264(output, input_options, *, crf, data=None):
    ff("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", *input_options, "-an", "-c:v", "libx264",
       "-preset", "fast", "-threads", "1", "-crf", str(crf), "-pix_fmt", "yuv420p", "-g", "48",
       "-video_track_timescale", "25000", str(output), input_data=data)


def _anchors(path, table, rgb, match):
    from scripts.video_face_portrait_hevc_transfer import rgb_anchors

    values = {}
    x, y, w, h = match["box"]
    for role, index in zip(("before", "during", "after"), match["anchors"], strict=True):
        image = rgb[index]
        values[role] = [index, table.pts[index], hashlib.sha256(image).hexdigest(),
                        hashlib.sha256(image[y:y + h, x:x + w].tobytes()).hexdigest()]
    rgb_anchors(path, {"proposal": {"anchors": values}})
    select = "+".join(f"eq(n,{index})" for index in match["anchors"])
    native = ff("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
                "-i", str(path), "-map", "0:v:0", "-an", "-vf",
                f"select='{select}',format=rgb24,crop={w}:{h}:{x}:{y}:exact=1", "-fps_mode", "passthrough",
                "-threads:v", "1", "-c:v", "rawvideo", "-f", "framehash", "-hash", "sha256", "pipe:1")
    observed = [line.rsplit(",", 1)[1].strip() for line in native.decode().splitlines() if line and not line.startswith("#")]
    if observed != [item[3] for item in values.values()]:
        raise ValueError("The crop differs from its source in the independent FFmpeg decode")
    return values


def _measure(path, detector):
    from app.detectors.video import mouth_local_hold, picture_hold
    from scripts import video_eye_hold_compare

    table, frames = mouth_local_hold.native(path)
    if [point * table.tb for point in table.pts] != [Fraction(i, RATE) for i in range(COUNT)]:
        raise ValueError("The delivered native clock differs")
    baseline = picture_hold.inspect(path, table, origin=0., sequence_review=True, mouth_review=True,
                                    motion_encoding=("h264", "yuv420p"))
    coverage = baseline["coverage"]
    record = coverage["mouth_local_hold"]
    started = time.monotonic()
    status, match = video_eye_hold_compare.observe(frames, detector)
    elapsed = round(time.monotonic() - started, 4)
    return {"sha256": sha(path), "clock": table.time_base, "source_frames": len(frames), "baseline_mouth": record["match"],
            "baseline_picture": baseline["match"], "candidate": match, "status": status, "elapsed_s": elapsed,
            "verified_anchors": _anchors(path, table, frames, match) if match else {}}


def _matches(proposal, truth):
    if not proposal or not truth or proposal["eye"] != truth["eye"]:
        return False
    a, b = proposal["frames"]
    c, d = truth["frames"]
    x, y, w, h = proposal["box"]
    i, j, width, height = truth["box"]
    area = max(0, min(x + w, i + width) - max(x, i)) * max(0, min(y + h, j + height) - max(y, j))
    return 2 * (min(b, d) - max(a, c)) >= max(b, d) - min(a, c) and 4 * area >= 3 * w * h


def summary(rows):
    totals = Counter()
    arms, groups = {}, set()
    for row in rows:
        role, match, truth = (row[key] for key in ("role", "candidate", "truth"))
        located = _matches(match, truth)
        group = "edited" if truth else "control"
        numbers = {"files": 1, "assessed": row["status"] != "no_stable_face",
                   "baseline_picture": bool(row["baseline_picture"]), "baseline_mouth": bool(row["baseline_mouth"]),
                   "candidate_flags": bool(match), "useful": located,
                   "exact": bool(located and match["frames"] == truth["frames"])}
        arms.setdefault(role, Counter()).update(numbers)
        totals.update({f"{group}_{name}": value for name, value in numbers.items()})
        if located:
            groups.add(row["group"])
        totals["misplaced"] += bool(truth and match and not located)
    totals["located_groups"] = len(groups)
    totals["new_global_holds_or_cuts"] = sum(arms.get(role, {}).get("candidate_flags", 0) for role in ("whole_hold", "picture_jump"))
    passed = (totals["edited_useful"] >= GATE["min_added_useful"] and len(groups) >= GATE["min_groups"]
              and totals["control_candidate_flags"] <= GATE["max_new_controls"] and totals["misplaced"] <= GATE["max_misplaced"]
              and not totals["new_global_holds_or_cuts"] and all(totals[f"{name}_assessed"] >= GATE["min_native_coverage"]
                * totals[f"{name}_files"] for name in ("edited", "control")))
    return dict(totals) | {"passed": passed, "by_role": arms,
                           "median_elapsed_s": statistics.median(row["elapsed_s"] for row in rows)}


def score(plan_path, sources, split, output):
    import cv2

    cv2.setNumThreads(1)
    plan = load(plan_path)
    if plan["method_sha256"] != _pins() or plan["cases"] != _cases(plan["roster"]["groups"]):
        raise ValueError("The test definitions changed after freezing")
    detector, weight = _locator()
    if weight != plan["geometry_weight_sha256"]:
        raise ValueError("The local locator changed after freezing")
    expected = [row for row in plan["cases"] if row["split"] == split]
    rows = []
    for group in (item for item in plan["roster"]["groups"] if item["role"] == split):
        path = sources / f"{group['index']:02d}-ordinary.mp4"
        if sha(path) != group["files"]["ordinary"]["sha256"]:
            raise ValueError("The source differs from its public bytes")
        frames = _prepare(path, group["offset"])
        with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
            for case in (row for row in expected if row["index"] == group["index"]):
                target = Path(temporary) / case["delivery"]
                _encode(frames, case["role"], group["eye_boxes"], target)
                rows.append(case | _measure(target, detector))
                target.unlink()
        save(output, plan | {"split": split, "rows": rows, "plan_sha256": sha(plan_path),
                             "summary": summary(rows), "complete": len(rows) == len(expected)})
        print(split, len(rows), "/", len(expected), flush=True)


def verify(path):
    value = load(path)
    if value["schema"] != SCHEMA or not value["complete"] or value["cases"] != _cases(value["roster"]["groups"]):
        raise ValueError("The held receipt is incomplete")
    rows = value["rows"]
    for case, row in zip((r for r in value["cases"] if r["split"] == value["split"]), rows, strict=True):
        assert all(row[field] == datum for field, datum in case.items()) and row["source_frames"] == COUNT
        if proposal := row["candidate"]:
            start, stop = proposal["frames"]
            measured = row["verified_anchors"]
            for role, index in zip(("before", "during", "after"), (start - 1, (start + stop) // 2, stop), strict=True):
                frame, tick, rgb, crop = measured[role]
                assert frame == index and Fraction(row["clock"]) * tick == Fraction(index, RATE)
                assert len(bytes.fromhex(rgb)) == len(bytes.fromhex(crop)) == 32
    assert summary(rows) == value["summary"]
    print(value["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("freeze", "score"):
        runner = commands.add_parser(name)
        runner.add_argument("input", type=Path)
        runner.add_argument("sources", type=Path)
        runner.add_argument("output", type=Path)
        if name == "score":
            runner.add_argument("--split", choices=("development", "held"), required=True)
    read = commands.add_parser("verify")
    read.add_argument("input", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.input, args.sources, args.output)
    elif args.action == "score":
        score(args.input, args.sources, args.split, args.output)
    else:
        verify(args.input)


if __name__ == "__main__":
    main()
