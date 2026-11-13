# SPDX-License-Identifier: Apache-2.0
"""Test the frozen single-upload picture change on timestamp-verified VP9 WebM."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import cv2

from app.detectors.video import scene_return
from app.detectors.video.decode import decode_frames, decode_stream, probe_frames
from scripts import video_scene_return_h264_transfer as avc
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-scene-return-webm-transfer/1"
TICKS = tuple((100 * i + 1) // 3 for i in range(120))
ORIGINAL = avc.PUBLISHED
ENCODER = ["-an", "-sn", "-dn", "-map_metadata", "-1", "-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0",
           "-crf", "32", "-deadline", "realtime", "-cpu-used", "6", "-threads", "1", "-lag-in-frames", "0", "-fps_mode", "passthrough"]
DESIGN = "Preserve all 14 exact held MP4 findings and flag none of 40 paired and 18 supplemental ordinary controls. No new fit."


def case_rows(hevc, h264):
    old = read(ORIGINAL)
    replay = read(hevc / "results.json")
    require(all(replay[key] == old[key] for key in ("schema", "bindings", "rows")),
            "Source panel differs from published HEVC decisions")
    source = {(row["pair"], row["direction"], row["role"]): row for row in old["rows"]}
    for item in read(hevc / "media.json")["rows"]:
        row = source[item["pair"], item["direction"], item["role"]]
        require(item["sha256"] == row["sha256"], "Unexpected source hash")
        yield {"name": Path(item["name"]).stem, "path": str(hevc / "media" / item["name"]), "source_sha256": row["sha256"],
               "role": row["role"], "expected": row["candidate"], "truth": row["truth"]}
    controls = [row for row in read(h264 / "results.json")["rows"] if row["condition"] != "paired"]
    require(len(controls) == 18 and not any(row["candidate"] for row in controls), "Supplemental controls differ")
    for row in controls:
        yield {"name": Path(row["name"]).stem, "path": str(h264 / "media" / row["name"]), "source_sha256": row["sha256"],
               "role": row["condition"], "expected": None, "truth": None}


def inspect(path):
    native = probe_frames(path, cap=130, timeout_s=15)
    require((native.width, native.height, native.coded_width, native.coded_height, native.tb) == (1280, 720, 1280, 720, Fraction(1, 1000))
            and not (native.rotation or native.truncated or native.interpolated) and native.pts == TICKS
            and avc.video_encoding(path) == ("vp9", "[0][0][0][0]", "yuv420p"), "Rounded WebM carrier differs")
    internal = replace(native, time_base="1/30000", pts=tuple(i * 1000 for i in range(120)))
    seen = []

    def verified():
        for frame in decode_stream(path, native, timeout_s=24):
            require(frame.index == len(seen) and frame.pts_verified and frame.pts == native.pts[frame.index], "Source clock differs")
            seen.append(frame.pts)
            yield replace(frame, pts=internal.pts[frame.index])

    match = scene_return.detect(verified(), internal)
    require(tuple(seen) == native.pts, "Incomplete picture inspection")
    anchors = []
    if match:
        for frame in decode_frames(path, native, match.anchors, timeout_s=24):
            require(frame.pts_verified and frame.pts == native.pts[frame.index]
                    and frame.rgb.shape == (720, 1280, 3), "Native evidence frame differs")
            anchors.append({"index": frame.index, "pts": frame.pts, "sha256": hashlib.sha256(frame.rgb).hexdigest()})
        require(tuple(a["index"] for a in anchors) == match.anchors, "Missing native evidence")
    return {"candidate": [match.start, match.stop] if match else None, "tiles": list(match.tiles) if match else [],
            "anchors": anchors, "webm_sha256": sha(path)}


def independently_verify(path, row):
    require(sha(path) == row["webm_sha256"], "WebM bytes changed before independent decoding")
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", "rgb24",
        "-threads", "1", "-f", "framehash", "-hash", "sha256", "-"], text=True, timeout=25).splitlines()
    require("#tb 0: 1/1000" in raw, "Independent time base differs")
    frames = [line.replace(" ", "").split(",") for line in raw if not line.startswith("#")]
    require(tuple(int(frame[2]) for frame in frames) == TICKS, "Independent WebM timestamps differ")
    for anchor in row["anchors"]:
        native = frames[anchor["index"]]
        require(int(native[2]) == anchor["pts"] and native[5] == anchor["sha256"], "Independent native frame differs")
    return len(frames), len(row["anchors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "audit"))
    parser.add_argument("hevc", type=Path)
    parser.add_argument("h264", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "audit":
        result = read(args.run / "results.json")
        verification = [independently_verify(args.run / "media" / (row["name"] + ".webm"), row) for row in result["rows"]]
        result["independent_replay"] = {"frames": sum(item[0] for item in verification), "anchors": sum(item[1] for item in verification)}
        if args.receipt:
            with args.receipt.open("x", encoding="utf-8") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
        print(json.dumps(result["independent_replay"], sort_keys=True))
        return
    lock = {"schema": SCHEMA, "design": DESIGN, "encoder": ENCODER, "opencv": cv2.__version__,
            "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
            "parent_ledger_sha256": sha(ORIGINAL), "detector_sha256": sha(scene_return.__file__),
            "runner_sha256": sha(__file__), "rows": list(case_rows(args.hevc, args.h264))}
    require(len(lock["rows"]) == 74, "Wrong transfer roster")
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        (args.run / "media").mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        return
    require(read(args.run / "freeze.json") == lock, "Frozen plan or source roster differs")
    results = []
    for row in lock.pop("rows"):
        path = Path(row["path"])
        require(sha(path) == row["source_sha256"], "Frozen input bytes differ")
        media = args.run / "media" / (row["name"] + ".webm")
        subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-threads", "1", "-i", str(path), "-map", "0:v:0",
                        *ENCODER, str(media)], check=True, timeout=35)
        results.append({key: value for key, value in row.items() if key != "path"} | inspect(media))
        if len(results) % 8 == 0:
            print("scored", len(results), flush=True)
    write(args.run / "results.json", lock | {"rows": results})


if __name__ == "__main__":
    main()
