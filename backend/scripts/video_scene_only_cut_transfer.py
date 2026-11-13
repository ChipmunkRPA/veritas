# SPDX-License-Identifier: Apache-2.0
"""Build a matched genuine scene-cut bank and measure existing local checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import face_spatial_cut
from app.detectors.video.decode import decode_frames, probe_frames
from app.detectors.video.shots import ShotDetector
from app.evidence import face_paired_edge_binding, face_spatial_cut_binding
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
WEIGHT = ROOT / "models/weights/face_detection_yunet_2023mar.onnx"
WEIGHT_SHA = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SOURCE_PLAN = DATA / "video-common-acquisition-v91/asset-plan.json"
NATIVE = DATA / "video-native-sources-v92"
EXCLUDED = (
    DATA / "video-face-near-head-transfer-2026-11-12-v2/freeze.json",
    DATA / "video-face-scale-context-holdout-2026-11-12-v2/freeze.json",
)
ROLES = ("unchanged", "speaker_only", "scene_only", "speaker_and_scene")
PLACEMENTS = ((44, 55), (52, 65), (61, 77), (73, 91))
W, H, RATE, N = 1280, 720, 30, 120
TILE = (432, 182, 416, 312)
SCHEMA = "chipmunk.video-scene-only-cut-transfer/1"
SPEC = {
    "selection": "First eight confirmation-role CREMA-D v91/v92 actors in acquisition order, at least 55 native pictures, with actor keys absent from both earlier CREMA-D near-head/scale scene-cut banks. Pair consecutively, each direction. Prior source-level project exposure is known.",
    "delivery": "Upright 1280 by 720 at exact 30 Hz, 120 pictures, exact RGB checks before encoding and HEVC hvc1 yuv420p CRF23 submitted files. Use the nearest real native source picture at each rational target time and reflect existing pictures at ends. No blended or generated speech pictures. Silent fixture.",
    "matrix": "Each direction supplies four exact input arrangements: unchanged; replace only the speaker rectangle; change only the surrounding layout with unchanged speaker pixels; replace both. The two ordinary layouts use deterministic geometric webcam background patterns. Across roles, placements alternate by pair; both directions use the same authored interval.",
    "scope": "Score whole-picture HSV shot detection and the existing off-face paired context calculation with its fixed 0.30 point. Give the latter the exact authored range, and the same range as a sham on the unchanged control. Test the context measurement, without claiming Full face routing or AI detection performance.",
    "matching": "Report exact entry and return separately for HSV, plus a paired exact-range count. For off-face context, report exact span and any flagged overlapping candidate, plus sham controls. Preserve paired direction dependence. No threshold or serving change.",
}


def plan():
    excluded = {item["source_group"] for file, field in zip(EXCLUDED, ("crema_originals", "sources"), strict=True)
                for item in read(file)[field]}
    require(len(excluded) == 20 and sha(WEIGHT) == WEIGHT_SHA, "Earlier actor banks or face asset differ")
    assets = read(SOURCE_PLAN)
    native = {}
    for file in sorted(NATIVE.glob("batch-*.jsonl")):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if row["source_family"] == "recorded_audiovisual_speech":
                native[row["index"]] = row
    chosen = []
    for asset in assets["assets"]:
        if (asset["source_family"] != "recorded_audiovisual_speech" or asset["provisional_role"] != "confirmation"
                or asset["group"] in excluded):
            continue
        row = native[asset["index"]]
        table = row["frame_table"]
        require(row["video_clock_qualified"] and asset["sha256"] == row["input_sha256"], "Archived source proof differs")
        if len(table["pts"]) >= 55:
            chosen.append({"file": str(Path(assets["media_root"]) / asset["file"]), "sha256": asset["sha256"],
                           "group": asset["group"], "frame_table": table, "rgb_sha256": row["rgb_sha256"]})
    chosen = chosen[:8]
    require(len({p["group"] for p in chosen}) == len({p["sha256"] for p in chosen}) == 8, "Need eight distinct actors")
    sources = [{**source, "ordinal": i} for i, source in enumerate(chosen)]
    return {"schema": SCHEMA, "protocol": SPEC, "roles": ROLES, "placements": PLACEMENTS, "sources": sources,
            "bindings": {p.name if p in (SOURCE_PLAN, Path(__file__)) else p.parent.name: sha(p)
                         for p in (SOURCE_PLAN, *EXCLUDED, Path(__file__))},
            "opencv": cv2.__version__, "weights": WEIGHT_SHA,
            "checks": {"shot": sha(ROOT / "backend/app/detectors/video/shots.py"),
                       "paired": sha(Path(face_spatial_cut.__file__)),
                       "binding": sha(Path(face_spatial_cut_binding.__file__))},
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]}


def source_pictures(source):
    path = Path(source["file"])
    require(sha(path) == source["sha256"], "Genuine source bytes differ")
    table = probe_frames(path, timeout_s=20)
    known = source["frame_table"]
    require(list(table.pts) == known["pts"] and table.time_base == known["time_base"]
            and (table.width, table.height) == (480, 360), "Exact input picture clock differs")
    native = []
    for frame in decode_frames(path, table, None, timeout_s=25):
        require(frame.pts_verified and frame.index == len(native) and
                hashlib.sha256(frame.rgb.tobytes()).hexdigest() == source["rgb_sha256"][frame.index],
                "Genuine input picture differs")
        native.append(frame.rgb)
    require(len(native) == len(table), "Incomplete genuine source")
    times = [p * table.tb for p in table.pts]
    interval = times[-1] - times[0]
    center = (times[0] + times[-1]) / 2
    selected = []
    for n in range(N):
        desired = center + Fraction(n - N // 2, RATE)
        t = (desired - times[0]) % (interval * 2)
        target = times[0] + min(t, 2 * interval - t)
        index = min(range(len(times)), key=lambda i: (abs(times[i] - target), i))
        selected.append(index)
    _, _, w, h = TILE
    return [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in native], selected


def layout(style):
    """Two ordinary webcam layouts with the same mean palette and different texture."""
    y, x = np.mgrid[:H, :W]
    if style == 0:
        seed = ((x // 15 + 2 * (y // 11)) % 7) + ((x // 79 + y // 83) % 3)
    else:
        seed = ((2 * (x // 11) + y // 15) % 7) + ((2 * (x // 83) + y // 79) % 3)
    base = 39 + 11 * seed
    return np.stack((base + 18, base + 22, base + 30), axis=-1).astype(np.uint8)


def compose(host, guest, span, role):
    x, y, w, h = TILE
    background = (layout(0), layout(1))
    for i in range(N):
        active = span[0] <= i < span[1]
        actor = guest if active and role in {"speaker_only", "speaker_and_scene"} else host
        canvas = background[active and role in {"scene_only", "speaker_and_scene"}].copy()
        picture = actor[0][actor[1][i]]
        canvas[y:y + h, x:x + w] = picture
        yield canvas


def commit_regions(frames, commitments):
    x, y, w, h = TILE
    for frame in frames:
        patch = frame[y:y + h, x:x + w]
        outside = (frame[:y], frame[y:y + h, :x], frame[y:y + h, x + w:], frame[y + h:])
        commitments.append((hashlib.sha256(patch.tobytes()).hexdigest(),
                            hashlib.sha256(b"".join(p.tobytes() for p in outside)).hexdigest()))
        yield frame


def write_video(path, pictures):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}", "-framerate", str(RATE), "-i", "-", "-an",
            "-c:v", "libx265", "-preset", "ultrafast", "-crf", "23", "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
            "-x265-params", "pools=1:frame-threads=1:log-level=error", "-threads:v", "1", str(path)]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as ffmpeg:
        try:
            for frame in pictures:
                ffmpeg.stdin.write(frame.tobytes())
            ffmpeg.stdin.close()
            require(ffmpeg.wait(timeout=45) == 0, "Could not create the fixed HEVC carrier")
        finally:
            if ffmpeg.poll() is None:
                ffmpeg.kill()


def prepare(work, lock):
    folder = work / "media"
    folder.mkdir(mode=0o700)
    rows = []
    for pair, span in enumerate(PLACEMENTS):
        sources = [source_pictures(source) for source in lock["sources"][pair * 2:pair * 2 + 2]]
        for direction in range(2):
            host, guest = sources[direction], sources[1 - direction]
            by_role = {}
            for role in ROLES:
                path = folder / f"{pair}-{direction}-{role}.mp4"
                exact = []
                write_video(path, commit_regions(compose(host, guest, span, role), exact))
                by_role[role] = exact
                rows.append({"pair": pair, "direction": direction, "role": role, "span": list(span),
                             "name": path.name, "sha256": sha(path)})
            expected = list(range(*span))
            first = by_role["unchanged"]
            for role in ROLES:
                for side, expected_change in enumerate((role in {"speaker_only", "speaker_and_scene"},
                                                       role in {"scene_only", "speaker_and_scene"})):
                    actual = [i for i, (base, changed) in enumerate(zip(first, by_role[role], strict=True))
                              if base[side] != changed[side]]
                    require(actual == (expected if expected_change else []), "Factorial pixels differ from recipe")
            print("prepared pair", pair + 1, "direction", direction + 1, flush=True)
    require(len(rows) == 32, "Need all eight matched sets")
    write(work / "media.json", {"freeze_sha256": sha(work / "freeze.json"), "rows": rows,
                               "raw_factorial_frames_verified": 32 * N, "speaker_tile": TILE})


def inspect(path, row, detector):
    table = probe_frames(path, timeout_s=15)
    profile = json_profile(path)
    require(face_paired_edge_binding.supported(table, profile, "mp4"), "Output carrier differs from serving profile")
    start, end = row["span"]
    schedule, face_schedule = face_spatial_cut_binding.schedule(start, end)
    shot = ShotDetector(table)
    frames, boxes, commits = {}, [], []
    for frame in decode_frames(path, table, None, timeout_s=30):
        i = frame.index
        require(i == len(commits) and frame.pts_verified and frame.pts == table.pts[i], "Decoded carrier clock differs")
        shot.update(i, frame.rgb)
        if i in schedule:
            frames[i] = face_spatial_cut.reduce_picture(frame.rgb)
        if i in face_schedule:
            faces = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
            if len(faces) == 1:
                boxes.append({"bbox": dict(zip(("x", "y", "w", "h"), faces[0].box(), strict=True))})
        commits.append({"frame": i, "pts": frame.pts, "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest()})
    require(len(commits) == N and set(frames) == set(schedule), "Incomplete carrier frames")
    result, tiles = face_spatial_cut.paired_edges(frames, face_spatial_cut.face_box(boxes), (start, end)) if len(boxes) >= 2 else (None, 0)
    after = result and result["score"] >= face_spatial_cut_binding.POINT
    take = {start - 1, (start + end) // 2, end}
    if after:
        a, b = result["interval"]
        take.update((a - 1, (a + b) // 2, b))
    return {"frame_count": len(table), "time_base": table.time_base, "first_pts": table.pts[0],
            "shot_frames": [x["frame_index"] for x in shot.boundaries], "face_observations": len(boxes), "tiles": tiles,
            "paired": result, "selected": [commits[i] for i in sorted(take)]}


def json_profile(path):
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                              "stream=codec_name,pix_fmt,codec_tag_string", "-of", "json", str(path)],
                                             timeout=15))
    stream, = data["streams"]
    return tuple(stream[name] for name in ("codec_name", "pix_fmt", "codec_tag_string"))


def score(work):
    media = read(work / "media.json")
    require(media["freeze_sha256"] == sha(work / "freeze.json"), "Prepared media is not bound to the freeze")
    detector = load_yunet(WEIGHT)
    rows = []
    for row in media["rows"]:
        path = work / "media" / row["name"]
        require(sha(path) == row["sha256"], "Prepared carrier differs")
        rows.append({**row, **inspect(path, row, detector)})
        if len(rows) % 8 == 0:
            print("measured", len(rows), flush=True)
    write(work / "results.json", {"media_sha256": sha(work / "media.json"), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score"))
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    lock = plan()
    # Normalize tuples to their persisted JSON form.
    lock["roles"], lock["placements"] = list(ROLES), list(map(list, PLACEMENTS))
    path = args.work / "freeze.json"
    if args.phase == "freeze":
        args.work.mkdir(mode=0o700)
        write(path, lock)
    else:
        require(read(path) == lock, "Frozen plan differs")
        cv2.setNumThreads(1)
        if args.phase == "prepare":
            prepare(args.work, lock)
        else:
            score(args.work)


if __name__ == "__main__":
    main()
