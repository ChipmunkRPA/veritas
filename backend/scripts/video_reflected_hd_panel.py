"""Create and compare source-matched HD and upright portrait video deliveries."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.detectors.video import picture_hold, picture_motion
from app.detectors.video.decode import probe_frames
from app.evidence import picture_sequence as binding
from scripts.video_picture_speed_panel import _overlap, _sha, _write
from scripts.video_reflected_sequence_panel import KEY, mirror

SCHEMA = binding.MOTION_REFLECTED_HD_SCHEMA
PROFILES = {
    "avc-720-25": (1280, 720, "25", "h264", "mp4", ("bike-descent", "hd-vcd-15", "street-cycling")),
    "avc-1080-2997": (1920, 1080, "30000/1001", "h264", "mp4", ("street-dance", "swiss-train", "open-streets")),
    "webm-720-2997": (1280, 720, "30000/1001", "vp9", "webm", ("hd-vcd-20", "swiss-train", "street-cycling")),
    "webm-1080-24": (1920, 1080, "24", "vp9", "webm", ("street-dance", "bike-descent", "open-streets")),
    "hevc-portrait-30": (1080, 1920, "30", "hevc", "mov", ("vertical-bee", "street-dance", "open-streets")),
    "avc-portrait-23976": (720, 1280, "24000/1001", "h264", "mp4", ("vertical-bee", "hd-vcd-22", "street-cycling")),
}
ROLES = ("later_mirrored", "earlier_mirrored", "later_mirrored_tandem", "original",
         "in_place_mirror", "held_mirror", "same_speed", "original_tandem")


def _prepared_frames(source, spool, width, height, frame_rate, count):
    # Fill the raster with a centered crop. This also exercises portrait deliveries
    # constructed from landscape originals while preserving the authentic content.
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-filter_threads", "1",
        "-ss", str(source["offset_s"]), "-i", source["path"], "-an",
        "-vf", f"fps={frame_rate},scale={width}:{height}:force_original_aspect_ratio=increase:"
               f"force_divisible_by=2,crop={width}:{height},setsar=1,format=yuv420p",
        "-frames:v", str(count), "-f", "rawvideo", str(spool),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)
    bytes_per_frame = width * height * 3 // 2
    if spool.stat().st_size != count * bytes_per_frame:
        raise ValueError(f"Too few decoded pictures in {source['id']}")
    return np.memmap(spool, dtype=np.uint8, mode="r", shape=(count, bytes_per_frame))


def _segments(frames, role, dims, early, late, count):
    if role == "original":
        return (frames,), None
    start, stop = late, early
    if role == "earlier_mirrored":
        start, stop = early, late
    if role == "in_place_mirror":
        stop = start
    source = frames[stop:stop + count]
    if role == "held_mirror":
        replacement = np.repeat(mirror(source[:1], dimensions=dims), count, axis=0)
    elif role == "same_speed":
        replacement = source
    else:
        replacement = mirror(source, dimensions=dims)
    truth = [[offset, offset + count - 1] for offset in (early, late)] if role.endswith("_mirrored") else None
    return (frames[:start], replacement, frames[start + count:]), truth


def _encoder(codec, tandem):
    if codec == "h264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "33" if tandem else "24"]
    if codec == "hevc":
        return ["-c:v", "libx265", "-preset", "ultrafast", "-tag:v", "hvc1", "-x265-params", "pools=2:frame-threads=2",
                "-crf", "35" if tandem else "26"]
    return ["-c:v", "libvpx-vp9", "-b:v", "0", "-deadline", "good", "-cpu-used", "6", "-row-mt", "1",
            "-crf", "37" if tandem else "28"]


def _encode(output, dims, rate, codec, segments, prior):
    input_options = ["-i", str(prior)] if prior else ["-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size",
                                                     f"{dims[0]}x{dims[1]}", "-framerate", rate, "-i", "pipe:0"]
    command = ["ffmpeg", "-nostdin", "-v", "error", *input_options, "-an", *_encoder(codec, bool(prior)),
               "-threads", "3", "-g", "48", "-pix_fmt", "yuv420p", str(output)]
    if prior:
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        return
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL) as encoder:
        assert encoder.stdin is not None
        for segment in segments:
            encoder.stdin.write(memoryview(segment))
        encoder.stdin.close()
        if encoder.wait(timeout=120):
            raise RuntimeError(f"Encoding failed for {output.name}")


def build(output, source_records, label):
    width, height, rate_text, codec, suffix, source_ids = PROFILES[label]
    rate = Fraction(rate_text)
    dims = width, height
    limit = math.ceil(rate * binding.HD_PREFIX_S)
    output.mkdir(parents=True)
    by_id = {source["id"]: source for source in source_records}
    sources = [by_id[source_id] for source_id in source_ids]
    cases = []
    for source in sources:
        digest = _sha(Path(source["path"]))
        if digest != source["sha256"]:
            raise ValueError(f"Source differs from the intake receipt: {source['id']}")
        early = round(rate * (Fraction(9, 10) + Fraction(int(digest[:2], 16) % 30, 100)))
        late = round(rate * (Fraction(51, 10) + Fraction(int(digest[2:4], 16) % 45, 100)))
        with tempfile.TemporaryDirectory(prefix=".frames-", dir=output) as workspace:
            frames = _prepared_frames(source, Path(workspace) / "source.yuv", width, height, rate_text, limit)
            rendered = {}
            for role in ROLES:
                filename = f"{source['id']}-{role}.{suffix}"
                parent = role.removesuffix("_tandem")
                if parent == role:
                    duration = Fraction(72, 100) if role == "later_mirrored" else Fraction(110, 100)
                    segments, truth = _segments(frames, role, dims, early, late, math.ceil(rate * duration))
                    prior = None
                else:
                    original, truth = rendered[parent]
                    segments, prior = (), output / original
                path = output / filename
                _encode(path, dims, rate_text, codec, segments, prior)
                rendered[role] = filename, truth
                cases.append({"source": source["id"], "role": role, "delivery": filename, "sha256": _sha(path), "truth": truth})
            del frames
        print(f"{label} prepared {source['id']}", flush=True)
    _write(output / "plan.json", {"delivery_profile": label,
        "method": {"dimensions": list(dims), "codec": codec, "frame_limit": limit, "schema": SCHEMA,
                   "parameters": binding.motion_parameters(SCHEMA, rate),
                   "detector_sha256": _sha(Path(picture_motion.__file__)), "clock_and_binding_sha256": _sha(Path(binding.__file__))},
        "sources": [{k: v for k, v in row.items() if k != "path"} for row in sources], "cases": cases})


def score(output, result):
    plan = json.loads((output / "plan.json").read_text())
    method = plan["method"]
    if (method["detector_sha256"], method["clock_and_binding_sha256"]) != (
        _sha(Path(picture_motion.__file__)), _sha(Path(binding.__file__)),
    ):
        raise ValueError("Method changed after the held deliveries were frozen")
    encoding = method["codec"], "yuv420p"
    cases = []
    for row in plan["cases"]:
        path = output / row["delivery"]
        table = probe_frames(path)
        assert _sha(path) == row["sha256"] and len(table) == method["frame_limit"]
        assert [table.width, table.height] == method["dimensions"] and not table.interpolated
        assert picture_motion.profile(table, encoding, schema=binding.MOTION_REFLECTED_SCHEMA) is None
        profile = picture_motion.profile(table, encoding, schema=SCHEMA)
        assert profile == (Fraction(method["parameters"]["frame_rate"]), len(table)), path.name
        begin = time.monotonic()
        check = picture_hold.inspect(path, table, origin=0.0, sequence_review=True, motion_encoding=encoding,
                                     reflected_review=True)["coverage"].get(KEY)
        pair = check["pair"] if check else None
        found = [[pair[side]["start_frame_index"], pair[side]["end_frame_index"]] for side in ("first", "second")] if pair else None
        useful = bool(found and row["truth"] and all(_overlap(a, b) >= .5 for a, b in zip(found, row["truth"], strict=True)))
        cases.append(row | {"reflected": found, "useful": useful, "exact": found is not None and found == row["truth"],
                            "readout": check["status"] if check else "unavailable", "time_base": table.time_base,
                            "source_first_pts": table.pts[0], "anchors": list(binding.endpoints(pair)) if pair else [],
                            "support": pair["supporting_edge_offsets"] if pair else [],
                            "scan_ms": round(1000 * (time.monotonic() - begin), 1)})
        print(path.name, cases[-1]["readout"], found, useful, flush=True)
    _write(result, plan | {"cases": cases})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    make.add_argument("output", type=Path)
    make.add_argument("sources", type=Path)
    make.add_argument("profile", choices=PROFILES)
    read = commands.add_parser("score")
    read.add_argument("output", type=Path)
    read.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.output, json.loads(args.sources.read_text()), args.profile)
    else:
        score(args.output, args.result)
