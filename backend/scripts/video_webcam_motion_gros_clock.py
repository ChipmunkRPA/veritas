# SPDX-License-Identifier: Apache-2.0
"""Operational source-clock fallback for the large controlled motion originals."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from scripts import video_webcam_motion_worker as worker


def packet_clock(source, require):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,time_base:stream_side_data=rotation:packet=pts", "-show_packets",
        "-of", "json", str(source)], check=True, capture_output=True, timeout=25)
    doc = json.loads(result.stdout)
    [stream] = doc["streams"]
    pts = tuple(sorted(int(row["pts"]) for row in doc["packets"]))
    base = Fraction(stream["time_base"])
    require(stream["codec_name"] == "h264" and stream["width"] * 9 == stream["height"] * 16
                  and not any(row.get("rotation", 0) for row in stream.get("side_data_list", ()))
                  and 150 <= len(pts) <= 950 and len(pts) == len(set(pts)) and base > 0,
                  "Controlled original requires an upright MP4 with a complete unique H.264 packet PTS clock")
    return base, pts


def delivery(source, work, bound):
    base, points = packet_clock(source, bound.require)
    output = work / "source-delivery.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-filter_threads", "2",
                    "-i", str(source), "-map", "0:v:0", "-an", "-vf", "scale=1280:720:flags=area,format=yuv420p",
                    "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                    "-bf", "0", "-threads", "2", "-enc_time_base", str(base), "-video_track_timescale",
                    str(base.denominator), "-n", str(output)], check=True, timeout=110)
    table = bound.common.probe_frames(output, timeout_s=40)
    bound.require(not (table.rotation or table.truncated or table.interpolated)
                  and (table.width, table.height) == (1280, 720)
                  and tuple((p - table.pts[0]) * table.tb for p in table.pts) == tuple((p - points[0]) * base for p in points),
                  "Every decoded output picture must match one exact original source-relative packet timestamp")
    return output, table


def prepare(folder):
    from scripts import video_webcam_motion_transfer as study

    lock = study.frozen(folder)
    partial = folder / "gros.jsonl"
    prior_log = folder / "gros-worker-1.log"
    study.require(partial.stat().st_size == 0 and "frame probe timed out" in prior_log.read_text(),
                  "Source-clock amendment applies only after the initial pre-inference timeout")
    partial.rename(folder / "gros-empty-first-attempt.jsonl")
    study.common.write(folder / "gros-clock-amendment.json", {
        "original_freeze_sha256": study.common.digest(folder / "freeze.json"),
        "operational_source_clock_worker_sha256": study.common.digest(Path(__file__)),
        "initial_failure_sha256": study.common.digest(prior_log),
        "cases": len(lock["gros"]), "previous_scored": 0,
        "change": "Read exact presentation timestamps from H.264 MP4 packets instead of full source-picture probing; each decoded output picture must match the sorted original packet timestamp grid. Deliver the fixed sources at 720p CRF 16 H.264 after the lossless delivery exceeded the app limit.",
    })


def child(folder, ordinal):
    with tempfile.TemporaryDirectory(prefix="motion-worker-", dir=folder) as data:
        work = Path(data)
        worker.environment(Path(__file__).resolve().parents[2], work)
        from scripts import video_webcam_motion_transfer as study
        from scripts.video_appearance_corrected_v112 import scoped

        fn = scoped(worker.genuine_app, {"source_delivery": delivery})
        print(json.dumps(fn(folder, ordinal, work, study), allow_nan=False), flush=True)


def run(folder):
    from scripts import video_webcam_motion_transfer as study

    amendment = study.common.read(folder / "gros-clock-amendment.json")
    study.require(amendment["original_freeze_sha256"] == study.common.digest(folder / "freeze.json")
                  and amendment["operational_source_clock_worker_sha256"] == study.common.digest(Path(__file__)),
                  "Operational clock amendment changed")
    with (folder / "gros.jsonl").open("x") as stream:
        for ordinal in range(1, amendment["cases"] + 1):
            path = folder / f"gros-amended-worker-{ordinal}.log"
            with path.open("x") as log:
                process = subprocess.run([sys.executable, "-B", "-m", "scripts.video_webcam_motion_gros_clock",
                    "child", str(folder), str(ordinal)], check=False, stdout=subprocess.PIPE, stderr=log,
                    text=True, timeout=290, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                log.write(process.stdout)
            study.require(process.returncode == 0, f"Amended Gros worker failed before readout, case {ordinal}")
            result = json.loads(process.stdout.splitlines()[-1])
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            print("gros", ordinal, result["status"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "child"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("ordinal", type=int, nargs="?")
    args = parser.parse_args()
    if args.stage == "child":
        child(args.folder, args.ordinal)
    else:
        {"prepare": prepare, "run": run}[args.stage](args.folder)
