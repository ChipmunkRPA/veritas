"""Test whether frozen motion sequence criteria transfer to 640×360 at 25 fps."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import picture_sequence
from scripts.video_lossy_sequence_transfer import hash_file, load, overlap, pair_matrices, search, write

WIDTH, HEIGHT, FPS, FRAMES = 640, 360, 25, 225
POINT = (1.6, .4, .65)
SLOTS = (3, 5, 6, 9, 10, 12)
CODECS = {
    "h264": ("-c:v", "libx264", "-crf", "23", "-g", "47", "-preset", "fast"),
    "vp9": ("-c:v", "libvpx-vp9", "-crf", "31", "-b:v", "0", "-g", "61", "-deadline", "realtime", "-cpu-used", "5"),
}
CAMERA_KINDS = ("copy", "unchanged", "cut", "hold", "reverse", "spacing")
COMMONS = {
    "pushups": ("webm", 8),
    "golf": ("webm", 1),
    "rowing": ("webm", 2),
    "lunge": ("webm", 8),
    "wobble": ("ogv", 2),
    "tennis": ("webm", 1),
    "wind": ("webm", 2),
    "army": ("webm", 3),
}
GATE = {
    "minimum_camera_useful_per_codec": 4,
    "minimum_commons_useful_per_codec": 5,
    "maximum_new_control_flags_per_codec": 1,
    "maximum_new_hold_or_spacing_flags": 0,
    "minimum_iou_each_range": .5,
}


def source_plan(archive, commons, earlier):
    prior = set()
    for role in ("development", "held"):
        prior.update(row["original_source_sha256"] for row in load(earlier / f"{role}.json")["sources"])
    plan = []
    for slot in SLOTS:
        source = archive / f"slot-{slot:02d}-attempt-01"
        path = source / "source.mkv"
        digest = hash_file(path)
        if digest != load(source / "qualification.json")["sha256"] or digest in prior:
            raise ValueError(f"Camera slot {slot} differs from its qualification or reuses an earlier source")
        plan.append(("camera", f"camera-{slot:02d}", path, digest, 0))
    for name, (ext, start) in COMMONS.items():
        path = commons / f"{name}.{ext}"
        plan.append(("commons", name, path, hash_file(path), start))
    if len({p[3] for p in plan}) != len(plan) or {p[3] for p in plan} & prior:
        raise ValueError("Source overlap inside or across frozen panels")
    return plan


def source_raster(path, offset):
    return subprocess.check_output(
        ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-threads", "1", "-ss", str(offset),
         "-i", f"file:{path}", "-map", "0:v:0", "-an", "-filter_threads", "1", "-vf",
         f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase:flags=area,crop={WIDTH}:{HEIGHT},setsar=1",
         "-frames:v", str(FRAMES), "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1"], timeout=40)


def flip_yuv(frame):
    luma = WIDTH * HEIGHT
    chroma = luma // 4
    return b"".join(np.frombuffer(frame[a:b], dtype="u1").reshape(-1, stride)[:, ::-1].tobytes()
                    for a, b, stride in ((0, luma, WIDTH), (luma, luma + chroma, WIDTH // 2), (luma + chroma, luma + 2 * chroma, WIDTH // 2)))


def authored(raw, path, first, second, count, kind, codec):
    length = WIDTH * HEIGHT * 3 // 2
    if len(raw) != length * FRAMES:
        raise ValueError("The frozen source interval does not contain nine full seconds")
    view = memoryview(raw)
    frames = [view[length * i:length * (i + 1)] for i in range(FRAMES)]
    if kind in {"copy", "reverse", "spacing"}:
        segment = frames[first:first + count]
        frames[second:second + count] = segment[::-1] if kind == "reverse" else segment
    elif kind == "hold":
        frames[second:second + count] = [frames[second]] * count
    elif kind == "cut":
        frames[second:second + count] = [flip_yuv(frame) for frame in frames[second:second + count]]
    clock = "settb=expr=1/1000,setpts=N*40"
    if kind == "spacing":
        clock += rf"+8*mod(N\,2)*between(N\,{second}\,{second + count - 1})"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", f"{WIDTH}x{HEIGHT}",
         "-framerate", str(FPS), "-i", "pipe:0", "-filter_threads", "1", "-vf", clock, "-fps_mode", "passthrough",
         "-enc_time_base", "1:1000", "-threads", "1", "-pix_fmt", "yuv420p", *CODECS[codec], str(path)],
        input=b"".join(frames), check=True, timeout=35)


def inspect(path, spacing):
    table = probe_frames(path)
    if (table.width, table.height, len(table)) != (WIDTH, HEIGHT, FRAMES) or table.interpolated:
        raise ValueError("The delivery has unsupported dimensions, length, or inferred timestamps")
    values, digests = [], []
    for frame in decode_frames(path, table, range(FRAMES), timeout_s=10):
        if not frame.pts_verified:
            raise ValueError("The native delivery frame did not supply a verified timestamp")
        means = frame.rgb.reshape(30, 12, 40, 16, 3).mean(axis=(1, 3), dtype=np.float64)
        values.append((means @ np.array([77., 150., 29.]) / 256).reshape(-1))
        digests.append(hashlib.sha256(frame.rgb).hexdigest())
    if len(digests) != FRAMES:
        raise ValueError("Incomplete native frame decode")
    observations = list(zip(table.pts, digests, strict=True))
    exact = picture_sequence.scan(observations, table.tb)["pair"]
    indices = [row[2] for row in picture_sequence.endpoints(exact)] if exact else None
    verified_cfr = all((b - a) * table.tb * FPS == 1 for a, b in zip(table.pts, table.pts[1:]))
    if verified_cfr == spacing:
        raise ValueError("Native clock does not match the authored format")
    motion = None
    if verified_cfr:
        motion = search(np.asarray(table.pts), table.tb, pair_matrices(table, np.stack(values)), POINT)
    return {"exact": indices, "motion": motion, "combined": indices or motion, "frames": FRAMES,
            "pts": list(table.pts), "time_base": table.time_base, "rgb_sha256": digests, "verified_cfr": verified_cfr}


def summarize(rows):
    results = {}
    for codec in CODECS:
        selected = [row for row in rows if row["codec"] == codec]
        controls = [row for row in selected if row["truth"] is None]
        hits = {role: {arm: sum(overlap(row[arm], row["truth"]) >= GATE["minimum_iou_each_range"] for row in selected
                                if row["cohort"] == role)
                       for arm in ("exact", "motion", "combined")} for role in ("camera", "commons")}
        flags = {arm: dict(Counter(row["kind"] for row in controls if row[arm] is not None)) for arm in ("exact", "motion", "combined")}
        added = [row for row in controls if row["exact"] is None and row["combined"] is not None]
        results[codec] = {"useful": hits, "flags_by_kind": flags,
                          "copy_cases": {role: sum(row["cohort"] == role and row["truth"] is not None for row in selected)
                                         for role in ("camera", "commons")},
                          "controls": len(controls), "additional_controls": len(added),
                          "new_hold_or_spacing": sum(row["kind"] in {"hold", "spacing"} for row in added),
                          "lost_exact": sum(overlap(row["exact"], row["truth"]) >= .5 and overlap(row["combined"], row["truth"]) < .5
                                            for row in selected)}
    passes = all(row["useful"]["camera"]["combined"] >= GATE["minimum_camera_useful_per_codec"]
                 and row["useful"]["commons"]["combined"] >= GATE["minimum_commons_useful_per_codec"]
                 and row["additional_controls"] <= GATE["maximum_new_control_flags_per_codec"]
                 and row["new_hold_or_spacing"] <= GATE["maximum_new_hold_or_spacing_flags"]
                 and row["lost_exact"] == 0 for row in results.values())
    return results, passes


def run(output, archive, commons, earlier):
    plan = source_plan(archive, commons, earlier)
    output.mkdir(mode=0o700)
    write(output / "freeze.json", {"source_sha256": {name: digest for _, name, _, digest, _ in plan},
                                    "geometry": [WIDTH, HEIGHT, FPS, FRAMES], "point": POINT,
                                    "gate": GATE, "runner_sha256": hash_file(Path(__file__))})
    rows = []
    for number, (role, name, path, _, start) in enumerate(plan):
        raster = source_raster(path, start)
        first, second, count = 18 + number, 144 + number % 8, (23, 28, 34)[number % 3]
        kinds = CAMERA_KINDS if role == "camera" else ("copy", "published")
        for codec in CODECS:
            for kind in kinds:
                clip = output / f"{name}-{codec}-{kind}.mkv"
                authored(raster, clip, first, second, count, kind, codec)
                outcome = inspect(clip, kind == "spacing")
                rows.append({"cohort": role, "source": name, "codec": codec, "kind": kind, **outcome, "sha256": hash_file(clip),
                             "truth": [first, first + count - 1, second, second + count - 1] if kind == "copy" else None})
        print(name, len(rows), flush=True)
    metrics, passes = summarize(rows)
    write(output / "native.json", {"rows": rows, "metrics": metrics, "passes": passes})
    print(metrics, passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "archive", "commons", "earlier"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.output, args.archive, args.commons, args.earlier)
