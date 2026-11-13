# SPDX-License-Identifier: Apache-2.0
"""Test the frozen standalone picture-change detector on clock-matched H.264."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2

from app.detectors.video import scene_return
from app.detectors.video.decode import decode_frames, decode_stream, probe_frames
from scripts import review_video_scene_return_transfer as audit
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
EXISTING = DATA / "video-face-codec-background-transfer-2026-11-12"
CAMERAS = DATA / "video-short-return-codec-2026-11-11"
PUBLISHED = ROOT / "docs/evaluation-panels/video-scene-return-transfer-2026-11-13.json"
AVC = ("h264", "avc1", "yuv420p")
RECIPES = {"veryfast21": ("veryfast", "21"), "medium29": ("medium", "29")}
SCHEMA = "chipmunk.video-scene-return-h264-transfer/1"
ENCODER = ["-map_metadata", "-1", "-an", "-sn", "-dn", "-c:v", "libx264",
           "-threads", "1", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-enc_time_base", "demux",
           "-video_track_timescale", "30000", "-tag:v", "avc1", "-movflags", "+faststart"]
N = 120
RATE = 30
CLOCK = [Fraction(i, RATE) for i in range(N)]


def freeze(folder, original_path, recipe):
    original_results = read(original_path / "results.json")
    require(read(PUBLISHED) == original_results, "Published HEVC decision ledger differs")
    require(sha(Path(scene_return.__file__)) == original_results["bindings"]["scene_return.py"], "Frozen detector changed")
    prior = {(r["pair"], r["direction"], r["role"]): r for r in original_results["rows"]}
    frozen = []
    for item in read(original_path / "media.json")["rows"]:
        old = prior[item["pair"], item["direction"], item["role"]]
        require(item["sha256"] == old["sha256"], "HEVC source does not match scored row")
        frozen.append({"source": str(original_path / "media" / item["name"]), "sha256": old["sha256"],
                       "prior": old, "condition": "paired", "name": item["name"]})
    if recipe == "veryfast21":
        frozen.extend(supplemental_cases())
    folder.mkdir(mode=0o700)
    (folder / "media").mkdir(mode=0o700)
    preset, crf = RECIPES[recipe]
    lock = {
        "schema": SCHEMA,
        "recipe": recipe,
        "candidate_sha256": sha(Path(scene_return.__file__)),
        "hevc_ledger_sha256": sha(PUBLISHED),
        "runner_sha256": sha(Path(__file__)),
        "codec_panel_sha256": sha(EXISTING / "media.json") if recipe == "veryfast21" else None,
        "camera_panel_sha256": sha(CAMERAS / "media.json") if recipe == "veryfast21" else None,
        "encoder": ["-preset", preset, "-crf", crf, *ENCODER],
        "admission": "Preserve all exact HEVC detections and withhold on all ordinary controls.",
        "rows": frozen,
    }
    write(folder / "freeze.json", lock)
    return lock


def supplemental_cases():
    frozen = []
    panel = read(EXISTING / "media.json")
    original = {r["group"]: r for r in panel["rows"] if r["codec"] == "hevc" and r["condition"] == "original"}
    require(len(original) == 6, "Six matched ordinary source groups required")
    for group in range(6):
        own = original[group]
        other = original[(group + 1) % len(original)]
        for cut in (39, 77):
            frozen.append({"source": str(EXISTING / "media" / own["file"]), "sha256": own["sha256"],
                           "source_receipt": own, "donor": str(EXISTING / "media" / other["file"]),
                           "donor_sha256": other["sha256"], "donor_receipt": other,
                           "name": f"hard-cut-g{group}-{cut}.mp4", "condition": "one_way_hard_cut", "cut": cut})
    camera = [r for r in read(CAMERAS / "media.json")["media"] if r["codec"] == "hevc" and r["condition"] == "camera"]
    require(len(camera) == 6, "Six archived ordinary retimed camera groups required")
    for row in camera:
        frozen.append({"source": str(CAMERAS / "media" / row["file"]), "sha256": row["sha256"], "source_receipt": row,
                       "name": f"retimed-{Path(row['file']).stem.removesuffix('-hevc')}.mp4", "condition": "reflected_camera_retime"})
    return frozen


def video_encoding(path):
    probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=codec_name,codec_tag_string,pix_fmt", "-of", "json", str(path)], timeout=12)
    [stream] = json.loads(probe)["streams"]
    return tuple(stream[k] for k in ("codec_name", "codec_tag_string", "pix_fmt"))


def native_small(path, table, witness):
    # Independent native RGB read before scaling or retiming any supplemental control.
    expected = witness.get("clock") or witness["frame_table"]
    require(table.to_json() == expected, "Archived ordinary control source clock changed")
    frames = []
    for f in decode_frames(path, table, None, timeout_s=20):
        require(f.index == len(frames) and f.pts_verified and f.pts == table.pts[f.index], "Unverified ordinary source clock")
        require(hashlib.sha256(f.rgb.tobytes()).hexdigest() == witness["rgb_sha256"][f.index],
                "Ordinary source pixels differ")
        frames.append(f.rgb)
    require(len(frames) == len(table), "Ordinary source frame count differs")
    return frames


def render(folder, row, encoder):
    path = Path(row["source"])
    require(sha(path) == row["sha256"], "Original source bytes differ")
    source = probe_frames(path, timeout_s=12)
    target = folder / "media" / row["name"]
    if row["condition"] == "paired":
        require(scene_return.supported(source) and video_encoding(path) == ("hevc", "hvc1", "yuv420p"),
                "HEVC source profile changed")
        require(source.time_base == row["prior"]["time_base"] and
                hashlib.sha256(json.dumps(source.pts).encode()).hexdigest() == row["prior"]["pts_sha256"],
                "HEVC source PTS commitment differs")
        subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-copyts", "-threads", "1", "-i", str(path),
                        "-map", "0:v:0", *encoder, str(target)], check=True, timeout=25)
    else:
        host = native_small(path, source, row["source_receipt"])
        if row["condition"] == "one_way_hard_cut":
            donor_path = Path(row["donor"])
            require(sha(donor_path) == row["donor_sha256"], "Ordinary donor differs")
            donor = native_small(donor_path, probe_frames(donor_path, timeout_s=12), row["donor_receipt"])
            require(len(host) == len(donor) == N, "Ordinary hard cut source length differs")
            cut = row["cut"]
            frames = host[:cut] + donor[cut:]
        else:
            # Reflect only native pictures around both ends to preserve a continuous camera recording.
            period = 2 * (len(host) - 1)
            frames = [host[min(i % period, period - i % period)] for i in range(N)]
        args = ["ffmpeg", "-nostdin", "-n", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", "1280x720", "-framerate", "30", "-i", "-", *encoder, str(target)]
        with subprocess.Popen(args, stdin=subprocess.PIPE) as proc:
            try:
                for frame in frames:
                    proc.stdin.write(cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR).tobytes())
                proc.stdin.close()
                require(proc.wait(timeout=25) == 0, "Control encode failed")
            finally:
                if proc.poll() is None:
                    proc.kill()
    table = probe_frames(target, timeout_s=12)
    require(scene_return.supported(table) and video_encoding(target) == AVC
            and [p * table.tb for p in table.pts] == CLOCK, "Unsupported or inexact H.264 profile")
    if row["condition"] == "paired":
        require([p * source.tb for p in source.pts] == [p * table.tb for p in table.pts], "Carrier conversion retimed a frame")
    verified = []

    def observe():
        for f in decode_stream(target, table, timeout_s=25):
            require(f.index == len(verified) and f.pts_verified and f.pts == table.pts[f.index], "Unverified measured frame")
            verified.append(f.pts)
            yield f

    found = scene_return.detect(observe(), table)
    require(verified == list(table.pts), "Incomplete measured clock")
    anchors = []
    if found:
        for f in decode_frames(target, table, found.anchors, timeout_s=25):
            require(f.pts_verified and f.pts == table.pts[f.index], "Selected H.264 anchor clock differs")
            require(f.rgb.shape == (720, 1280, 3), "Selected H.264 native frame was resized")
            anchors.append({"index": f.index, "pts": f.pts, "sha256": hashlib.sha256(f.rgb.tobytes()).hexdigest()})
        require([f["index"] for f in anchors] == list(found.anchors), "Selected anchor readback incomplete")
    return {"name": target.name, "sha256": sha(target), "condition": row["condition"],
            "role": row.get("prior", {}).get("role"), "truth": row.get("prior", {}).get("truth"),
            "prior_candidate": row.get("prior", {}).get("candidate"),
            "candidate": [found.start, found.stop] if found else None,
            "tiles": list(found.tiles) if found else [], "anchors": anchors,
            "time_base": table.time_base, "pts_sha256": hashlib.sha256(json.dumps(table.pts).encode()).hexdigest()}


def independently_verify(path, row):
    require(sha(path) == row["sha256"], "H.264 bytes changed before independent decoding")
    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-copyts", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-enc_time_base", "demux", "-pix_fmt", "rgb24",
        "-threads", "1", "-f", "framehash", "-hash", "sha256", "-"], text=True, timeout=25).splitlines()
    require(f"#tb 0: {row['time_base']}" in raw, "Independent native time base differs")
    frames = [line.replace(" ", "").split(",") for line in raw if not line.startswith("#")]
    ticks = [int(frame[2]) for frame in frames]
    require(len(ticks) == N and [i * Fraction(row["time_base"]) for i in ticks] == CLOCK
            and hashlib.sha256(json.dumps(ticks).encode()).hexdigest() == row["pts_sha256"],
            "Independent frame clock differs")
    for f in row["anchors"]:
        require(ticks[f["index"]] == f["pts"] and frames[f["index"]][5] == f["sha256"],
                "Independently decoded source anchor differs")
    return len(ticks), len(row["anchors"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("run", "audit"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--hevc", type=Path, required=True, help="Earlier scene-return run directory")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="veryfast21")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    folder = args.folder.resolve()
    if args.phase == "run":
        lock = freeze(folder, args.hevc.resolve(), args.recipe)
        results = []
        for case in lock["rows"]:
            row = render(folder, case, lock["encoder"])
            print(row["name"], row["candidate"], flush=True)
            results.append(row)
        write(folder / "results.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": results})
        return
    previous = read(PUBLISHED)
    audit.validate(previous)
    source_audit = audit.replay_sources(previous, args.hevc / "media")
    result = read(folder / "results.json")
    lock = read(folder / "freeze.json")
    require(result["freeze_sha256"] == sha(folder / "freeze.json") and lock["schema"] == SCHEMA
            and lock["runner_sha256"] == sha(Path(__file__)) and lock["candidate_sha256"] == sha(Path(scene_return.__file__))
            and lock["hevc_ledger_sha256"] == sha(PUBLISHED) and lock["recipe"] == args.recipe,
            "Results or code differ from frozen plan")
    timing, anchors = 0, 0
    counts = Counter()
    for case, row in zip(lock["rows"], result["rows"], strict=True):
        prior = case.get("prior", {})
        require((row["name"], row["condition"], row["role"], row["truth"], row["prior_candidate"])
                == (case["name"], case["condition"], prior.get("role"), prior.get("truth"), prior.get("candidate")),
                "Case provenance changed after scoring")
        if row["candidate"]:
            start, stop = row["candidate"]
            require([f["index"] for f in row["anchors"]] == [start - 1, (start + stop) // 2, stop],
                    "Selected anchor roles differ")
        key = row["role"] if row["condition"] == "paired" else row["condition"]
        counts[f"{key}_files"] += 1
        if row["candidate"] is not None:
            counts[f"{key}_ranges"] += 1
            if row["truth"] == row["candidate"]:
                counts[f"{key}_exact"] += 1
            else:
                counts[f"{key}_unexpected"] += 1
        if row["candidate"] != row["prior_candidate"] and row["condition"] == "paired":
            counts[f"{key}_carrier_change"] += 1
        n, m = independently_verify(folder / "media" / row["name"], row)
        timing += n
        anchors += m
    receipt = {
        "schema": SCHEMA, "recipe": lock["recipe"], "encoder": lock["encoder"],
        "detector_sha256": lock["candidate_sha256"], "hevc_ledger_sha256": lock["hevc_ledger_sha256"],
        "freeze_sha256": sha(folder / "freeze.json"), "results_sha256": sha(folder / "results.json"),
        "carrier_timestamps_verified": timing, "carrier_anchors_verified": anchors,
        "source_replay": source_audit, "counts": dict(counts),
        "admission_passed": all(row["candidate"] == row["prior_candidate"] for row in result["rows"]),
    }
    write(folder / "audit.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
