# SPDX-License-Identifier: Apache-2.0
"""Prepare a prospective H.264 360p trial from qualified matched source frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2

from app.detectors.video import face_return
from scripts import video_short_return_resolution_transfer as prior

common, trial, require = prior.common, prior.trial, prior.require
ARCHIVE = prior.ARCHIVE
DEFAULT_PRIOR = common.ROOT / "backend/data/independent-evaluation/video-short-return-resolution-2026-11-11/freeze.json"
DIMENSIONS = (640, 360)
RATE = 25
FRAMES = 150
SPANS = ((67, 76), (75, 84))
CLOCK = [Fraction(i, RATE) for i in range(FRAMES)]
CONDITIONS = ("original", "recompressed", "ordinary_light", "ordinary_blur", "ordinary_time_step", "swap_a", "swap_b")
GATE = "At least four added correct short intervals across three source groups; at least 80% assessed separately for swaps and ordinary controls; at most one new matched ordinary flag and zero new hard-cut flags."
PROFILES = ((1280, 720), (1920, 1080))
ENCODE = ("-an", "-sn", "-dn", "-map_metadata", "-1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
          "-threads", "2", "-x264-params", "keyint=25:min-keyint=25:scenecut=0", "-video_track_timescale", "25000",
          "-movflags", "+faststart")


def bindings():
    from scripts import video_short_return_encoded_app as app_trial
    from app.detectors.video.adapters import face_appearance
    from app.engine.stages import video
    from app.evidence import face_return_binding
    return {**common.bindings(), **{str(path.relative_to(common.ROOT)): common.digest(path) for path in
            map(lambda module: Path(module.__file__), (face_return, face_appearance, face_return_binding, video, app_trial))},
            "worker_sha256": common.digest(Path(__file__)), "opencv": cv2.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=5).splitlines()[0]}


def freeze(folder, archive, preceding):
    paths = [preceding, *(archive / name / "freeze.json" for name in prior.PRIOR)]
    exposed = set.union(*(trial.previous_rtf_sources(common.read(path)) for path in paths))
    require(len(exposed) == 18 and archive.resolve() == ARCHIVE, "Exact preceding source panels and archive are required")
    metadata = trial.prior.comparison.metadata()
    earlier = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in exposed]
    urls, donors = {row["url"] for row in earlier}, set.union(*(row["donors"] for row in earlier))
    candidates = {size: [] for size in PROFILES}
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        doc = common.read(path)
        index, row = doc["index"], metadata[doc["group"]]
        if index in exposed or row["url"] in urls or row["donors"] & donors or doc["role"] == "confirmation" or "child" in row["tags"]:
            continue
        original = next(case for case in doc["cases"] if case["condition"] == "original")
        clock = original["frame_table"]
        size = clock["width"], clock["height"]
        times = [(pts - clock["pts"][0]) * Fraction(clock["time_base"]) for pts in clock["pts"]]
        if size in candidates and times == CLOCK:
            candidates[size].append((sum(case["bytes"] for case in doc["cases"]), index, row))
    roster = []
    for size, options in candidates.items():
        chosen = 0
        for _, index, row in sorted(options):
            if row["url"] in urls or row["donors"] & donors:
                continue
            roster.append({**prior.archive_io.source_pin(index, archive), "native_size": size})
            urls.add(row["url"])
            donors.update(row["donors"])
            chosen += 1
            if chosen == 2:
                break
        require(chosen == 2, f"Need two distinct sources with native geometry {size}")
    folder.mkdir(mode=0o700, exist_ok=False)
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-short-return-encoded-freeze/1",
        "bindings": bindings(), "prior_freeze_sha256": list(map(common.digest, paths)), "roster": roster,
        "conditions": CONDITIONS, "source_cut_pairs": [[0, 1], [2, 3]], "edit_frame_spans": SPANS,
        "delivery_size": DIMENSIONS, "delivery_rate_hz": RATE, "native_frame_count": FRAMES,
        "review_point": face_return.POINT, "minimum_correct_iou": "1/2",
        "primary_crf": 23, "ordinary_recompression_crf": 36, "encoding_options": ENCODE, "gate": GATE})


def frozen(folder):
    plan = common.read(folder / "freeze.json")
    require(plan["bindings"] == bindings() and plan["gate"] == GATE and plan["conditions"] == list(CONDITIONS)
            and plan["edit_frame_spans"] == list(map(list, SPANS)), "Scoring or preparation differs from freeze")
    return plan


def clock_ok(table, width, height):
    return (table.width, table.height) == (width, height) and not (table.rotation or table.truncated or table.interpolated) and (
        common.relative_times(table) == CLOCK)


def native_frames(path, case, size, indices):
    require(common.digest(path) == case["sha256"], "Source media digest differs")
    table = common.probe_frames(path, timeout_s=30)
    expected = case["frame_table"]
    require(clock_ok(table, *size) and [pts * table.tb for pts in table.pts] ==
            [pts * Fraction(expected["time_base"]) for pts in expected["pts"]], "Source clock differs")
    found = {}
    for decoded in common.decode_frames(path, table, list(indices), timeout_s=120):
        index = decoded.index
        require(decoded.pts_verified and index not in found
                and hashlib.sha256(decoded.rgb).hexdigest() == case["expected_rgb_sha256"][index], "Source RGB or PTS differs")
        found[index] = cv2.resize(decoded.rgb, DIMENSIONS, interpolation=cv2.INTER_AREA)
    require(list(found) == list(indices), "Native source picture set differs")
    return found


def encode(path, *, frames=None, source=None, crf):
    inputs = ("-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "640x360", "-framerate", "25", "-i", "pipe:0") if frames is not None else (
        "-threads", "2", "-i", str(source))
    cmd = ("ffmpeg", "-nostdin", "-n", "-v", "error", *inputs, *ENCODE, "-crf", str(crf), str(path))
    if frames is None:
        subprocess.run(cmd, check=True, timeout=60)
        return
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ffmpeg:
        for image in frames:
            ffmpeg.stdin.write(image.tobytes())
        ffmpeg.stdin.close()
        require(ffmpeg.wait(timeout=60) == 0, "Encoded output failed")


def describe(path, group, condition, truth):
    table = common.probe_frames(path, timeout_s=30)
    require(clock_ok(table, *DIMENSIONS), "Encoded video clock or picture grid differs")
    pts = []
    hashes = []
    for frame in common.decode_frames(path, table, list(range(FRAMES)), timeout_s=60):
        require(frame.pts_verified and frame.index == len(hashes), "Encoded frame lacks its exact native timestamp")
        pts.append(frame.pts)
        hashes.append(hashlib.sha256(frame.rgb).hexdigest())
    require(tuple(pts) == table.pts and len(hashes) == FRAMES, "Encoded decoded-picture evidence is incomplete")
    return {"file": path.name, "sha256": common.digest(path), "bytes": path.stat().st_size, "group": group,
            "condition": condition, "truth_frames": truth, "frame_table": table.to_json(), "rgb_sha256": hashes}


def prepare(folder, archive):
    plan = frozen(folder)
    cv2.setNumThreads(2)
    output = folder / "media"
    output.mkdir(mode=0o700)
    manifest, original = [], {}
    for group, pin in enumerate(plan["roster"]):
        index = pin["index"]
        require(prior.archive_io.source_pin(index, archive) == {k: v for k, v in pin.items() if k != "native_size"},
                "Qualified source changed")
        native_root = archive / "video-native-inserts-v95" / f"group-{index:03d}"
        source = common.read(common.DATA / f"group-{index:03d}-complete.json")
        cases = {case["condition"]: case for case in source["cases"]}
        full = native_frames(native_root / cases["original"]["file"], cases["original"], tuple(pin["native_size"]), range(FRAMES))
        original[group] = full
        for condition in CONDITIONS:
            filename = output / f"source-{group + 1}-{condition}.mp4"
            truth = None
            if condition == "original":
                encode(filename, frames=full.values(), crf=23)
            elif condition == "recompressed":
                encode(filename, source=output / f"source-{group + 1}-original.mp4", crf=36)
            else:
                left, right = SPANS[int(condition in {"ordinary_blur", "swap_b"})]
                case = cases[condition]
                edited = native_frames(native_root / case["file"], case, tuple(pin["native_size"]), range(left, right))
                encode(filename, frames=(edited.get(i, full[i]) for i in range(FRAMES)), crf=23)
                truth = [left, right] if condition.startswith("swap_") else None
            manifest.append(describe(filename, group, condition, truth))
        print("encoded source group", group + 1, flush=True)
        if group % 2 == 1:
            a, b = original.pop(group - 1), original.pop(group)
            for direction, (base, inserted) in enumerate(((a, b), (b, a))):
                left, right = SPANS[direction]
                filename = output / f"cut-{group // 2 + 1}-{direction + 1}.mp4"
                encode(filename, frames=(inserted[i] if left <= i < right else base[i] for i in range(FRAMES)), crf=23)
                manifest.append(describe(filename, group // 2, "hard_cut", None))
            print("encoded cut pair", group // 2 + 1, flush=True)
    require(len(manifest) == 32 and len({row["sha256"] for row in manifest}) == 32, "Incomplete or duplicate output panel")
    common.write(folder / "media.json", {"freeze_sha256": common.digest(folder / "freeze.json"), "media": manifest})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--preceding", type=Path, default=DEFAULT_PRIOR)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.folder, args.archive, args.preceding)
    else:
        prepare(args.folder, args.archive)


if __name__ == "__main__":
    main()
