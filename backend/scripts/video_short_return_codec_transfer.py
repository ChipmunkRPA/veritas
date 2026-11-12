# SPDX-License-Identifier: Apache-2.0
"""Freeze 360p30 HEVC/AV1 short-swap and natural camera controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
from app.detectors.video import face_return
from scripts import video_short_return_encoded_transfer as previous
from scripts import video_short_return_resolution_transfer as earlier

common, trial, require = earlier.common, earlier.trial, earlier.require
ARCHIVE = earlier.ARCHIVE
DATA = common.ROOT / "backend/data/independent-evaluation"
CAMERA_PLAN = DATA / "av-sentence-support-v104/preparation/plan.json"
CAMERA = DATA / "native-mouth-clips-v120"
DIMENSIONS = (640, 360)
CAMERA_DIMENSIONS = (480, 360)
RATE = 30
FRAMES = 180
CAMERA_MAX_RETIME = Fraction(1, 250)
SPANS = {"swap_a": (84, 95), "swap_b": (91, 102)}
CONDITIONS = ("original", "ordinary_light", "ordinary_blur", *SPANS)
CODECS = {
    "hevc": ("-c:v", "libx265", "-preset", "ultrafast", "-crf", "25", "-tag:v", "hvc1", "-x265-params",
             "pools=2:frame-threads=1:keyint=30:min-keyint=30:scenecut=0:log-level=error"),
    "av1": ("-c:v", "libaom-av1", "-cpu-used", "8", "-row-mt", "1", "-crf", "30", "-b:v", "0", "-g", "30"),
}
GATE = ("For each codec, add at least three correct intervals across two swap source groups, assess at least 80% "
        "of swaps and ordinary controls, and add at most one false range on combined matched/camera controls and none on hard cuts.")


def bindings():
    return {**previous.bindings(), "worker_sha256": common.digest(Path(__file__))}


def rosters(archive):
    return [*(archive / name / "freeze.json" for name in earlier.PRIOR), previous.DEFAULT_PRIOR,
            DATA / "video-short-return-encoded-2026-11-11/freeze.json"]


def swaps(archive, paths):
    exposed = set.union(*(trial.previous_rtf_sources(common.read(path)) for path in paths))
    require(len(exposed) == 22, "All four preceding short-return source panels are required")
    metadata = trial.prior.comparison.metadata()
    used = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in exposed]
    urls, donors = {row["url"] for row in used}, set.union(*(row["donors"] for row in used))
    selected = []
    for size, limit in (((1280, 720), 2), ((1920, 1080), 1)):
        eligible = []
        for path in sorted(common.DATA.glob("group-*-complete.json")):
            doc = common.read(path)
            index, row = doc["index"], metadata[doc["group"]]
            if index in exposed or row["url"] in urls or row["donors"] & donors or doc["role"] == "confirmation" or "child" in row["tags"]:
                continue
            original = next(case for case in doc["cases"] if case["condition"] == "original")
            clock = original["frame_table"]
            times = [(pts - clock["pts"][0]) * Fraction(clock["time_base"]) for pts in clock["pts"]]
            if (clock["width"], clock["height"]) == size and times == [Fraction(i, RATE) for i in range(FRAMES)]:
                eligible.append((sum(case["bytes"] for case in doc["cases"]), index, row))
        n = 0
        for _, index, row in sorted(eligible):
            if row["url"] in urls or row["donors"] & donors:
                continue
            selected.append({**earlier.archive_io.source_pin(index, archive), "native_size": size})
            urls.add(row["url"])
            donors.update(row["donors"])
            n += 1
            if n == limit:
                break
        require(n == limit, "Need three disjoint source/donor groups with exact 30 Hz media")
    return selected


def camera_rows():
    source = {row["asset_index"]: row for row in common.read(CAMERA_PLAN)["records"]}
    options = []
    for path in sorted(CAMERA.glob("extract-*/output/cases.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["condition"] != "original" or row["status"] != "ready":
                continue
            pts = row["expected"]["picture_pts_ms"]
            if not 60 <= len(pts) <= 240:
                continue
            maximum = max(abs(Fraction(q - pts[0], 1000) - Fraction(i, RATE)) for i, q in enumerate(pts))
            if maximum <= CAMERA_MAX_RETIME:
                record = source[row["asset_index"]]
                options.append((maximum, row["asset_index"], record["actor"], row, path))
    chosen = []
    actors = set()
    for maximum, index, actor, row, path in sorted(options):
        if actor in actors:
            continue
        chosen.append({"index": index, "media_sha256": row["file_sha256"], "journal": str(path.relative_to(DATA)),
                       "journal_sha256": common.digest(path), "maximum_relative_retime_s": str(maximum), "pictures": len(row["expected"]["picture_pts_ms"])})
        actors.add(actor)
        if len(chosen) == 6:
            break
    require(len(chosen) == 6, "Six distinct low-resolution camera actor codes required")
    return chosen


def freeze(folder, archive):
    paths = rosters(archive)
    require(archive.resolve() == ARCHIVE, "Use the retained licensed source archive")
    rtfs, camera = swaps(archive, paths), camera_rows()
    folder.mkdir(mode=0o700, exist_ok=False)
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-short-return-codec-freeze/1", "bindings": bindings(),
        "prior_freeze_sha256": list(map(common.digest, paths)), "rtfs": rtfs, "camera": camera,
        "camera_plan_sha256": common.digest(CAMERA_PLAN), "gate": GATE, "delivery_size": DIMENSIONS, "delivery_rate_hz": RATE,
        "review_point": face_return.POINT, "codecs": CODECS, "edit_frame_spans": SPANS, "matched_conditions": CONDITIONS,
        "camera_max_retime_s": str(CAMERA_MAX_RETIME), "cut_pair": [0, 1], "cut_spans": list(SPANS.values())})


def frozen(folder):
    plan = common.read(folder / "freeze.json")
    require(plan["bindings"] == bindings() and plan["gate"] == GATE and plan["delivery_size"] == list(DIMENSIONS)
            and plan["delivery_rate_hz"] == RATE and plan["review_point"] == face_return.POINT
            and plan["prior_freeze_sha256"] == list(map(common.digest, rosters(ARCHIVE)))
            and plan["camera_plan_sha256"] == common.digest(CAMERA_PLAN) and plan["camera"] == camera_rows(),
            "Frozen methods, source selection or scope differ")
    return plan


def source_pictures(path, sha, expected_pts, hashes, indices, geometry, *, camera=False):
    require(common.digest(path) == sha, "Native file digest differs from its source receipt")
    table = common.probe_frames(path, timeout_s=30)
    require((table.width, table.height) == geometry and not (table.truncated or table.rotation or table.interpolated)
            and [p * table.tb for p in table.pts] == expected_pts, "Native presentation clock differs")
    result = {}
    for frame in common.decode_frames(path, table, list(indices), timeout_s=110):
        index = frame.index
        require(index not in result and frame.pts_verified and hashlib.sha256(frame.rgb).hexdigest() == hashes[index],
                "Source picture or timestamp differs")
        if camera:
            result[index] = cv2.copyMakeBorder(frame.rgb, 0, 0, 80, 80, cv2.BORDER_CONSTANT, value=0)
        else:
            result[index] = cv2.resize(frame.rgb, DIMENSIONS, interpolation=cv2.INTER_AREA)
    require(list(result) == list(indices), "Native picture roster differs")
    return result


def encode(path, pictures, codec):
    argv = ("ffmpeg", "-nostdin", "-n", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "640x360",
            "-framerate", str(RATE), "-i", "pipe:0", "-an", "-map_metadata", "-1", "-pix_fmt", "yuv420p", "-threads", "2",
            *CODECS[codec], "-video_track_timescale", "30000", "-movflags", "+faststart", str(path))
    with subprocess.Popen(argv, stdin=subprocess.PIPE) as writer:
        for rgb in pictures:
            writer.stdin.write(rgb.tobytes())
        writer.stdin.close()
        require(writer.wait(timeout=70) == 0, "Codec encoder failed")


def verify_delivery(path, codec, n):
    actual_codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name", "-of", "default=nw=1:nk=1", str(path)], text=True, timeout=15).strip()
    table = common.probe_frames(path, timeout_s=30)
    require(codec == actual_codec and (table.width, table.height) == DIMENSIONS
            and not (table.truncated or table.rotation or table.interpolated)
            and [p * table.tb for p in table.pts] == [Fraction(i, RATE) for i in range(n)],
            "Delivery codec, geometry or actual picture clock differs")
    hashes = []
    for frame in common.decode_frames(path, table, list(range(n)), timeout_s=70):
        require(frame.pts_verified and frame.index == len(hashes), "Delivered picture lacks native timestamp")
        hashes.append(hashlib.sha256(frame.rgb).hexdigest())
    require(len(hashes) == n, "Qualified delivery dropped pictures")
    return {"file": path.name, "sha256": common.digest(path), "bytes": path.stat().st_size,
            "codec": codec, "frame_table": table.to_json(), "rgb_sha256": hashes}


def render(output, filename, pictures, *, group, condition, truth=None):
    rows = []
    for codec in CODECS:
        path = output / f"{filename}-{codec}.mp4"
        encode(path, pictures, codec)
        rows.append({**verify_delivery(path, codec, len(pictures)), "group": group, "condition": condition,
                     "truth_frames": truth})
    return rows


def load_swap_case(source, case, indices, geometry):
    clock = case["frame_table"]
    pts = [p * Fraction(clock["time_base"]) for p in clock["pts"]]
    require([p - pts[0] for p in pts] == [Fraction(i, RATE) for i in range(FRAMES)], "Native condition has a different clock")
    return source_pictures(source / case["file"], case["sha256"], pts, case["expected_rgb_sha256"], indices, geometry)


def prepare(folder, archive):
    plan = frozen(folder)
    output = folder / "media"
    output.mkdir(mode=0o700)
    rows, originals = [], {}
    for group, pin in enumerate(plan["rtfs"]):
        i, geometry = pin["index"], tuple(pin["native_size"])
        require(earlier.archive_io.source_pin(i, archive) == {k: v for k, v in pin.items() if k != "native_size"},
                "Native swap receipt differs")
        cases = {row["condition"]: row for row in common.read(common.DATA / f"group-{i:03d}-complete.json")["cases"]}
        folder_for_source = archive / "video-native-inserts-v95" / f"group-{i:03d}"

        original = load_swap_case(folder_for_source, cases["original"], range(FRAMES), geometry)
        if group in plan["cut_pair"]:
            originals[group] = original
        for condition in CONDITIONS:
            if condition == "original":
                pictures, truth = list(original.values()), None
            else:
                a, b = SPANS["swap_b" if condition in {"swap_b", "ordinary_blur"} else "swap_a"]
                alternate = load_swap_case(folder_for_source, cases[condition], range(a, b), geometry)
                pictures = [alternate.get(index, rgb) for index, rgb in original.items()]
                truth = [a, b] if condition in SPANS else None
            rows.extend(render(output, f"matched-{group}-{condition}", pictures, group=group, condition=condition, truth=truth))
        print("encoded matched group", group + 1, flush=True)
    left, right = (originals[i] for i in plan["cut_pair"])
    for i, ((a, b), (base, insert)) in enumerate(zip(SPANS.values(), ((left, right), (right, left)), strict=True)):
        pictures = [insert[j] if a <= j < b else base[j] for j in range(FRAMES)]
        rows.extend(render(output, f"cut-{i}", pictures, group=i, condition="hard_cut"))
    originals.clear()
    for group, pin in enumerate(plan["camera"]):
        journal = DATA / pin["journal"]
        require(common.digest(journal) == pin["journal_sha256"], "Camera source receipt differs")
        [camera] = [row for row in map(json.loads, journal.read_text().splitlines())
                    if row["asset_index"] == pin["index"] and row["condition"] == "original"]
        media = archive / "native-mouth-clips-v120" / Path(*Path(camera["file"]).parts[-3:])
        clock = [Fraction(p, 1000) for p in camera["expected"]["picture_pts_ms"]]
        require(camera["file_sha256"] == pin["media_sha256"] and len(clock) == pin["pictures"], "Camera file differs")
        pictures = source_pictures(media, pin["media_sha256"], clock, camera["expected"]["rgb_sha256"],
                                   range(len(clock)), CAMERA_DIMENSIONS, camera=True)
        rows.extend(render(output, f"camera-{group}", list(pictures.values()), group=group, condition="camera"))
        print("encoded natural camera group", group + 1, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == 46, "Encoded panel is incomplete or duplicated")
    common.write(folder / "media.json", {"freeze_sha256": common.digest(folder / "freeze.json"), "media": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    {"freeze": freeze, "prepare": prepare}[args.stage](args.folder, args.archive)


if __name__ == "__main__":
    main()
