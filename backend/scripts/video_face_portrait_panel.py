# SPDX-License-Identifier: Apache-2.0
"""Prepare and measure controlled face changes on a 30 fps portrait video clock."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from bisect import bisect_left
from contextlib import closing
from fractions import Fraction
from pathlib import Path

from app.evidence import face_local_return_binding as local
from scripts import video_common_sources_v89 as io
from scripts import video_face_local_measure as weights
from scripts import video_face_local_panel as earlier

SCHEMA = "chipmunk.video-face-portrait/1"
DELIVERIES = ("direct", "tandem")
GENERATORS = earlier.GENERATORS
DIMENSIONS = {640: (360, 640), 1280: (720, 1280)}
START = Fraction(2, 5)
FPS = 30
GRID = local.PORTRAIT


def source_table(path):
    from app.detectors.video.decode import probe_frames

    table = probe_frames(path, cap=1600)
    if len(table) < 2 or table.interpolated or any(a >= b for a, b in zip(table.pts, table.pts[1:])):
        raise ValueError("The source lacks a reliable original timeline")
    ticks = [table.pts[0] + (START + GRID.step * i) / table.tb for i in range(GRID.frame_count)]
    if table.pts[-1] < ticks[-1]:
        raise ValueError("The source ends within the four-second excerpt")
    indices = []
    for tick in ticks:
        right = bisect_left(table.pts, tick)
        nearest = min((right - 1, right), key=lambda i: (abs(table.pts[i] - tick), i))
        if abs(table.pts[nearest] - tick) * table.tb > Fraction(1, 24):
            raise ValueError("A source picture is too far from the delivery clock")
        indices.append(nearest)
    return table, indices


def crop_center(path, table, indices, locator):
    import cv2
    import numpy as np

    from app.detectors.image.faces import detect_faces
    from app.detectors.video.decode import decode_frames

    xs = []
    with closing(decode_frames(path, table, indices[:40:8], max_side=640, timeout_s=30)) as images:
        for frame in images:
            if not frame.pts_verified:
                raise ValueError("The crop anchor lacks its presented timestamp")
            pixels = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            faces = detect_faces(pixels, detector=locator, min_size=36)
            if len(faces) == 1:
                x, _, w, _ = faces[0].box()
                xs.append((x + w / 2) / pixels.shape[1])
    return round(float(np.median(xs)), 7) if xs else .5


def portrait(rgb, center, size):
    import cv2

    height, width = rgb.shape[:2]
    crop_width, crop_height = min(width, 9 * height // 16), min(height, 16 * width // 9)
    left = max(0, min(round(width * center - crop_width / 2), width - crop_width))
    top = (height - crop_height) // 2
    region = rgb[top:top + crop_height, left:left + crop_width]
    interpolation = cv2.INTER_AREA if size[1] < crop_height else cv2.INTER_LINEAR
    return cv2.resize(region, size, interpolation=interpolation)


def source_pictures(path, table, indices, center, size):
    from app.detectors.video.decode import decode_frames

    pictures = {}
    with closing(decode_frames(path, table, sorted(set(indices)), max_side=1280, timeout_s=55)) as frames:
        for frame in frames:
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("A decoded publisher picture has an unverified timestamp")
            pictures[frame.index] = portrait(frame.rgb, center, size)
    if set(pictures) != set(indices):
        raise ValueError("The native source scan missed a sampled picture")
    return [pictures[i] for i in indices]


def clock(table, indices):
    return [(index, table.pts[index], table.time_base) for index in indices]


def edit_interval(position, kind=0):
    start, stop = earlier.interval(position, kind)
    return [2 * round(value * FPS / 50) for value in (start, stop)]


def encode(pictures, direct, tandem, size):
    base = ["ffmpeg", "-nostdin", "-v", "error", "-xerror"]
    common = ["-map", "0:v:0", "-an", "-map_metadata", "-1", "-frames:v", str(GRID.frame_count),
              "-c:v", "libx264", "-preset", "fast", "-threads", "2", "-g", "60", "-pix_fmt", "yuv420p",
              "-video_track_timescale", "30000", "-movflags", "+faststart", "-fps_mode", "passthrough"]
    fmt = ["-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{size[0]}x{size[1]}",
           "-framerate", str(FPS), "-i", "pipe:0"]
    with subprocess.Popen([*base, *fmt, *common, "-crf", "23", str(direct)], stdin=subprocess.PIPE) as process:
        with process.stdin as stream:
            for frame in pictures:
                stream.write(memoryview(frame))
        if process.wait(timeout=120):
            raise ValueError("The direct portrait delivery could not be encoded")
    subprocess.run([*base, "-i", str(direct), *common, "-crf", "36", str(tandem)], check=True, timeout=120)


def prepare(roster, sources, output, *, role, height):
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet

    cv2.setNumThreads(1)
    locator = load_yunet(weights.LOCATOR, "cpu", "float32").obj
    bank = [group for group in io.read(roster)["groups"] if group["role"] == role]
    size = DIMENSIONS[height]
    output.mkdir(mode=0o700)
    rows, omissions, origins = [], [], {}
    for group in bank:
        group_id = group["group"]
        paths = {key: sources / f'{group["index"]:02d}-{key}.mp4' for key in group["files"]}
        if any(io.sha(path) != group["files"][key]["sha256"] for key, path in paths.items()):
            raise ValueError("A publisher clip differs from its source commitment")
        original = paths["ordinary"]
        try:
            table, indices = source_table(original)
            center = crop_center(original, table, indices, locator)
        except (OSError, ValueError) as error:
            omissions.append({"group": group_id, "kind": "original", "reason": str(error)})
            continue
        origins[group_id] = (group, original, table, indices, center)
    if len(origins) < 2:
        raise ValueError("The study needs at least two continuous source recordings per role")
    enrolled = list(origins)
    for position, group_id in enumerate(enrolled):
        group, original, table, indices, center = origins[group_id]
        parent = group["files"]["ordinary"]["sha256"]
        reference = source_pictures(original, table, indices, center, size)
        original_clock = clock(table, indices)
        other_group, other, other_table, other_ids, other_center = origins[enrolled[(position + 1) % len(enrolled)]]
        other_pictures = source_pictures(other, other_table, other_ids, other_center, size)
        a, b = edit_interval(position)
        cases = []
        for kind, frames in earlier.camera_treatments(reference, other_pictures, a, b, locator):
            pin = other_group["files"]["ordinary"]["sha256"] if kind == "different_shot" else parent
            cases.append((kind, [a, b] if frames else [], None, frames, pin, None))
        if not any(item[0] == "face_blur" for item in cases):
            omissions.append({"group": group_id, "kind": "face_blur", "reason": "No face was available for the staged blur"})
        del other_pictures
        for variant, kind in enumerate(GENERATORS):
            derived = sources / f'{group["index"]:02d}-{kind}.mp4'
            try:
                changed_table, changed_indices = source_table(derived)
                changed_clock = clock(changed_table, changed_indices)
                if not earlier.compatible_clocks(original_clock, changed_clock, table.pts[0] * table.tb,
                                                  changed_table.pts[0] * changed_table.tb):
                    raise ValueError("The swap clock differs from the original")
                if Fraction(changed_table.width, changed_table.height) != Fraction(table.width, table.height):
                    raise ValueError("The publisher swap and original have different framing")
                low, high = edit_interval(position, variant)
                frames = source_pictures(derived, changed_table, changed_indices[low:high], center, size)
            except (OSError, ValueError) as error:
                omissions.append({"group": group_id, "kind": kind, "reason": str(error)})
                continue
            cases.append((kind, [low, high], [low, high], frames, group["files"][kind]["sha256"], changed_clock[low:high]))
        for kind, bounds, truth, changed, pin, changed_clock in cases:
            pictures = reference[:]
            if bounds:
                low, high = bounds
                if len(changed) != high - low:
                    raise ValueError("The staged treatment changed the frame count")
                pictures[low:high] = changed
            tag = f'{group["index"]:02d}-{kind}'
            direct, tandem = (output / f"{tag}-{delivery}.mp4" for delivery in DELIVERIES)
            encode(pictures, direct, tandem, size)
            for delivery, path in zip(DELIVERIES, (direct, tandem), strict=True):
                rows.append({"file": path.name, "group": group_id, "source_index": group["index"], "kind": kind,
                    "delivery": delivery, "truth": truth, "treatment": bounds, "treated_sha256": pin,
                    "source_sha256": parent, "input_sha256": io.sha(path), "changed_clock": changed_clock})
        print(f"Prepared {position+1}/{len(enrolled)} {role} groups at {height} pixels", flush=True)
    public_sources = [{"group": key, "crop_center": item[4], "native_clock": clock(item[2], item[3]),
                       "display_dimensions": [item[2].width, item[2].height]} for key, item in origins.items()]
    io.write(output / "manifest.json", {"schema": SCHEMA, "role": role, "size": list(size),
        "roster_sha256": io.sha(roster), "rows": rows, "omissions": omissions, "sources": public_sources})


def measure_clip(path, locator, recognizer, dimensions):
    from app.detectors.video import face_local_detail as check
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(path, cap=GRID.frame_count + 1)
    if ((table.width, table.height) != dimensions or table.rotation or len(table) != GRID.frame_count
            or table.interpolated or table.truncated
            or any((pts - table.pts[0]) * table.tb != i * GRID.step for i, pts in enumerate(table.pts))):
        raise ValueError("A prepared input differs from the portrait clock")
    samples, detail, anchors = {}, {}, {}
    with closing(decode_frames(path, table, GRID.schedule, timeout_s=45)) as frames:
        for frame in frames:
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("A delivered picture lacks its verified portrait timestamp")
            sample, frequency, anchor, _ = check.observe(frame.rgb, locator, recognizer, with_texture=True)
            samples[frame.index] = sample
            if frequency is not None:
                detail[frame.index] = frequency
                anchors[frame.index] = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
    if set(samples) != set(GRID.schedule):
        raise ValueError("The local scan did not see every planned native picture")
    ranked = [(local.response([round(x, 9) for x in item.features[:2]]), item)
              for item in check.proposals(samples, detail, geometry=GRID)]
    reading = {"face_frames": len(detail), "proposal": None}
    if ranked:
        score, best = max(ranked, key=lambda pair: (pair[0], pair[1].end - pair[1].start, -pair[1].start))
        cuts = check.shot_changes(path, table, best.start, best.end, timeout_s=20, geometry=GRID)
        reading["proposal"] = {"range": [best.start, best.end], "score": score.hex(),
            "features": [round(x, 9).hex() for x in best.features[:2]], "cuts": list(cuts),
            "anchors": {key: [i, table.pts[i], anchors[i]] for key, i in best.anchors().items()}}
    return reading | {"first_pts": table.pts[0], "step_pts": table.pts[1] - table.pts[0], "time_base": table.time_base}


def measure(inputs, output):
    import json

    import cv2

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video import face_local_detail as check
    from app.detectors.video.adapters.loaders import load_face_appearance

    cv2.setNumThreads(1)
    assets = weights.assets()
    locator = load_yunet(weights.LOCATOR, "cpu", "float32").obj
    recognizer = load_face_appearance(weights.ENCODER, "cpu", "float32").obj
    manifest = inputs / "manifest.json"
    data = io.read(manifest)
    header = {"schema": SCHEMA, "manifest_sha256": io.sha(manifest), "role": data["role"], "size": data["size"], "weights": assets,
              "script_sha256": io.sha(Path(__file__)), "proposal_sha256": io.sha(Path(check.__file__)),
              "score_sha256": io.sha(Path(local.__file__))}
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        for index, row in enumerate(data["rows"], 1):
            path = inputs / row["file"]
            if io.sha(path) != row["input_sha256"]:
                raise ValueError("A staged file differs from the bound manifest")
            found = measure_clip(path, locator, recognizer, tuple(data["size"]))
            stream.write(json.dumps(row | found, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            if index % 14 == 0 or index == len(data["rows"]):
                print(f"Measured {index}/{len(data['rows'])} {data['role']} files at {data['size'][1]} pixels", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    for name in ("roster", "sources", "output"):
        build.add_argument(name, type=Path)
    build.add_argument("--role", choices=("development", "held"), required=True)
    build.add_argument("--height", choices=DIMENSIONS, type=int, required=True)
    score = commands.add_parser("measure")
    score.add_argument("inputs", type=Path)
    score.add_argument("output", type=Path)
    args = vars(parser.parse_args())
    {"prepare": prepare, "measure": measure}[args.pop("command")](**args)
