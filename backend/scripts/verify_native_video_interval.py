"""Verify native video interval evidence offline with Python and FFmpeg."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

SCHEMA = "chipmunk.native-video-interval/1"
COMPARISON_SCHEMA = "chipmunk.video-reference-comparison/2"
TRIM_REENCODE_SCHEMA = "chipmunk.video-reference-comparison/10"
LOCAL_AREA_BASIS = "paired_h264_local_area/1"
MEASURED_INTERVAL_BASIS = "paired_h264_measured_picture_interval/1"
ROLES = ("before", "during", "after")
SHA = re.compile(r"[0-9a-f]{64}")
TIMESTAMP = re.compile(rb"\bn:\s*(\d+)\s+pts:\s*(-?\d+)")
INPUT_CLOCK = re.compile(rb"config in time_base:\s*(\d+)/(\d+)")
TIMEOUT = 60
MAX_FRAME_PIXELS = 4096 * 2160
MAX_COMPARISON_BYTES = 48 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clock_sha256(pts):
    payload = json.dumps(pts, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def run(command):
    try:
        return subprocess.run(command, capture_output=True, check=True, timeout=TIMEOUT)
    except (subprocess.SubprocessError, OSError) as error:
        raise ValueError(f"Native replay could not complete ({type(error).__name__})") from None


def probe(path):
    report = run([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,time_base,codec_name:stream_side_data=rotation:frame=pts",
        "-show_frames", "-show_streams", "-of", "json", str(path.resolve()),
    ])
    document = json.loads(report.stdout)
    require(len(document.get("streams", [])) == 1, "One primary video stream is required")
    stream = document["streams"][0]
    require(not any(entry.get("rotation", 0) for entry in stream.get("side_data_list", [])),
            "The interval receipt requires an unrotated source stream")
    width, height = stream["width"], stream["height"]
    require(type(width) is int and type(height) is int and width > 0 and height > 0
            and width * height <= MAX_FRAME_PIXELS, "Native frame dimensions exceed replay support")
    pts = [entry.get("pts") for entry in document.get("frames", [])]
    require(len(pts) >= 3 and all(type(p) is int for p in pts) and all(a < b for a, b in zip(pts, pts[1:])),
            "Source pictures need explicit, strictly increasing presentation timestamps")
    base = Fraction(stream["time_base"])
    require(base > 0, "The native time base must be positive")
    return width, height, base, pts, stream["codec_name"]


def decode_rgb_frames(path, indices, pts, time_base, frame_bytes):
    selection = "+".join(f"eq(n,{index})" for index in indices)
    decoded = run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-protocol_whitelist", "file",
        "-copyts", "-i", str(path.resolve()), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-fps_mode", "passthrough", "-vf", f"select='{selection}',showinfo,format=rgb24",
        "-frames:v", str(len(indices)), "-f", "rawvideo", "pipe:1",
    ])
    matches = TIMESTAMP.findall(decoded.stderr)
    clocks = {Fraction(int(n), int(d)) for n, d in INPUT_CLOCK.findall(decoded.stderr)}
    require(len(clocks) == 1 and len(matches) == len(indices), "Decoder did not report every required timestamp")
    native_base = clocks.pop()
    require(all(int(n) == i and int(t) * native_base == pts[i] * time_base
                for i, (n, t) in enumerate(matches)), "Decoder timestamps differ from the committed pictures")
    require(len(decoded.stdout) == len(indices) * frame_bytes, "Decoder returned incomplete native RGB")
    return tuple(decoded.stdout[i * frame_bytes:(i + 1) * frame_bytes] for i in range(len(indices)))


def check_exhibits(directory, digests, width, height):
    for role, expected in zip(ROLES, digests, strict=True):
        path = directory / f"{role}.png"
        with path.open("rb") as stream:
            require(stream.read(8) == b"\x89PNG\r\n\x1a\n", "An exhibit must be a PNG")
        width_found, height_found = probe_image(path)
        require((width_found, height_found) == (width, height), "An exhibit has different native geometry")
        frame = run([
            "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(path.resolve()),
            "-map", "0:v:0", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]).stdout
        require(len(frame) == width * height * 3 and hashlib.sha256(frame).hexdigest() == expected,
                "An exhibit differs from the committed native picture")


def probe_image(path):
    report = run([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "json", str(path.resolve()),
    ])
    streams = json.loads(report.stdout).get("streams", [])
    require(len(streams) == 1 and streams[0].get("codec_name") == "png" and streams[0].get("nb_read_frames") == "1",
            "A PNG exhibit must contain exactly one still picture")
    return streams[0]["width"], streams[0]["height"]


def verify(source_path, receipt, exhibits=None):
    require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA, "Unsupported interval receipt")
    source = receipt["source"]
    require(isinstance(source, dict) and source.get("stream") == "v:0" and bool(SHA.fullmatch(str(source.get("sha256")))),
            "Invalid primary source commitment")
    require(type(source.get("bytes")) is int and source["bytes"] > 0
            and source_path.is_file() and source_path.stat().st_size == source["bytes"]
            and sha_file(source_path) == source["sha256"],
            "Source file does not match the receipt")
    width, height, time_base, pts, codec = probe(source_path)
    require((width, height, str(time_base), len(pts), clock_sha256(pts)) ==
            (source["width"], source["height"], source["time_base"], source["frames"], source["pts_sha256"]),
            "Native geometry, time base or presentation clock differs from the receipt")
    interval = receipt["interval_pts"]
    require(isinstance(interval, list) and len(interval) == 2 and all(type(p) is int for p in interval)
            and interval[0] < interval[1], "Interval bounds must be ordered integer native ticks")
    start, end = interval
    rows = receipt["anchors"]
    require(isinstance(rows, list) and len(rows) == 3 and all(isinstance(row, dict) for row in rows)
            and [row.get("role") for row in rows] == list(ROLES),
            "Three ordered before, during and after anchors are required")
    indices = [row["index"] for row in rows]
    require(all(type(i) is int and 0 <= i < len(pts) for i in indices), "Anchor index falls outside the source")
    native = [pts[i] for i in indices]
    require(all(type(row["pts"]) is int for row in rows) and native == [row["pts"] for row in rows],
            "Anchor presentation timestamp differs from its native picture")
    require(native[0] < start <= native[1] < end <= native[2], "An anchor falls outside its half-open interval role")
    expected = tuple(row["rgb_sha256"] for row in rows)
    require(all(isinstance(d, str) and bool(SHA.fullmatch(d)) for d in expected), "Invalid RGB picture digest")
    frames = decode_rgb_frames(source_path, indices, native, time_base, width * height * 3)
    require(tuple(hashlib.sha256(frame).hexdigest() for frame in frames) == expected,
            "Decoded RGB differs from the committed native picture")
    if exhibits is not None:
        check_exhibits(exhibits, expected, width, height)
    return {
        "status": "verified",
        "scope": "source bytes, presentation clock and three sampled RGB pictures",
        "codec": codec,
        "time_base": str(time_base),
        "frame_count": len(pts),
        "interval_seconds": [str((point - pts[0]) * time_base) for point in (start, end)],
        "anchor_seconds": {role: str((point - pts[0]) * time_base) for role, point in zip(ROLES, native, strict=True)},
        "exhibit_pngs_verified": len(ROLES) if exhibits is not None else 0,
    }


def comparison_source(path, source, *, local_area=False):
    require(0 < path.stat().st_size == source["bytes"] <= MAX_COMPARISON_BYTES
            and sha_file(path) == source["sha256"], "Comparison source bytes do not match")
    width, height, base, pts, codec = probe(path)
    shapes = {(720, 480)} if local_area else {(360, 288), (480, 360)}
    require((width, height) in shapes and len(pts) <= 100,
            "Source lies outside the local comparison dimensions or frame limit")
    if local_area:
        step = (pts[1] - pts[0]) * base
        require(codec == "h264" and len(pts) >= 40 and Fraction(1, 25) <= step <= Fraction(1, 23)
                and all((b - a) * base == step for a, b in pairwise(pts)), "Unsupported local area clock")
    require((width, height, str(base), len(pts), pts[0], pts[-1], clock_sha256(pts)) == (
        source["width"], source["height"], source["time_base"], source["frames"], source["first_pts"],
        source["last_pts"], source["frame_pts_sha256"],
    ), "Comparison geometry or absolute picture clock differs")
    return width, height, base, pts


def native_crop(frame, width, x, y, w, h):
    return b"".join(frame[3 * (row * width + x):3 * (row * width + x + w)] for row in range(y, y + h))


def verify_comparison(earlier, submitted, export):
    record, assets = export["comparison"], export["assets"]
    trimmed = record["schema"] == TRIM_REENCODE_SCHEMA
    require(record["schema"] in (COMPARISON_SCHEMA, TRIM_REENCODE_SCHEMA) and record["status"] == "measured"
            and record["affects_overall_confidence"] is False, "Expected a measured local video comparison")
    local = record["picture"]["local_comparison"]
    local_area = local.get("basis") == LOCAL_AREA_BASIS
    require(local["status"] == "local_difference", "The export has no measured local picture interval")
    result = _verify_comparison_range(earlier, submitted, record, assets, local["range"], local_area=local_area, trimmed=trimmed)
    if precise := local.get("measured_interval"):
        coarse = local["range"]
        require(local_area and not trimmed and precise.get("basis") == MEASURED_INTERVAL_BASIS
                and precise["id"] == coarse["id"] + "-M" and precise["tile"] == coarse["tile"],
                "Measured picture interval differs from the initial local area")
        changes = [abs(precise[key] - coarse[key]) for key in ("start_frame", "end_frame_exclusive")]
        require(0 < max(changes) <= 6, "Measured picture boundaries differ from the initial range")
        verified = _verify_comparison_range(earlier, submitted, record, assets, precise, local_area=True, measured=True)
        result["measured_interval_seconds"] = verified["interval_seconds"]
        result["native_crops_verified"] += verified["native_crops_verified"]
        result["scope"] = "two source files, two shared picture intervals and twelve native crops"
    return result


def _verify_comparison_range(earlier, submitted, record, assets, region, *, local_area, trimmed=False, measured=False):
    start, end = region["start_frame"], region["end_frame_exclusive"]
    require(type(start) is type(end) is int and 0 < start < end, "Invalid half-open comparison interval")
    previews = region["previews"]
    require(set(previews) == set(ROLES), "Expected one before, during and after picture")
    indices = [previews[role]["frame"] for role in ROLES]
    require(all(type(index) is int for index in indices) and indices[0] == start - 1
            and start <= indices[1] < end == indices[2], "Picture indices differ from the interval roles")
    if local_area:
        width_valid = end - start >= 4 if measured else end - start in {8, 12, 16, 20}
        require(width_valid and indices[1] == (start + end) // 2,
                "The local area requires the selected middle picture")
    x, y, w, h = (region["tile"][key] for key in ("x", "y", "w", "h"))
    require(all(type(value) is int for value in (x, y, w, h)) and min(x, y) >= 0 and min(w, h) > 0,
            "Invalid crop rectangle")

    offsets = {"earlier": 0, "submitted": 0}
    if trimmed:
        alignment = record["trim_alignment"]
        offset = alignment["earlier_frames"][0]
        length = record["submitted"]["frames"]
        require(local_area and alignment["basis"] == "paired_h264_trim_motion/1"
                and type(offset) is int and 0 <= offset <= record["earlier"]["frames"] - length
                and alignment["earlier_frames"] == [offset, offset + length]
                and alignment["submitted_frames"] == [0, length] and alignment["audio_checked"] is False,
                "Trim alignment lies outside the two source clocks")
        offsets["earlier"] = offset
    clocks = []
    for side, path in (("earlier", earlier), ("submitted", submitted)):
        width, height, base, pts = comparison_source(path, record[side], local_area=local_area)
        offset = offsets[side]
        begin, stop = start + offset, end + offset
        require(stop < len(pts) and x + w <= width and y + h <= height, "Interval or crop lies outside the source")
        if measured:
            require(begin >= 2 and stop + 2 <= len(pts), "Measured interval lacks both neighboring pictures")
        clock = [(tick - pts[0]) * base for tick in pts]
        length = record["submitted"]["frames"] if trimmed else len(pts)
        clocks.append(((width, height), [time - clock[offset] for time in clock[offset:offset + length]]))
        prefix = "earlier_" if trimmed and side == "earlier" else ""
        require((region[f"{prefix}start_s"], region[f"{prefix}end_s"]) == (float(clock[begin]), float(clock[stop])),
                "Displayed interval differs from the source clock")
        require(region["source_ranges"][side] == {
            "start_pts": pts[begin], "last_pts": pts[stop - 1], "end_exclusive_pts": pts[stop],
        }, "Interval ticks differ from the source clock")
        source_indices = [index + offset for index in indices]
        ticks = [pts[index] for index in source_indices]
        frames = decode_rgb_frames(path, source_indices, ticks, base, width * height * 3)
        with TemporaryDirectory() as folder:
            expected = []
            for role, index, tick, frame in zip(ROLES, source_indices, ticks, frames, strict=True):
                preview = previews[role]
                require(preview[f"{side}_pts"] == tick and preview[f"{prefix}time_s"] == float(clock[index])
                        and (not prefix or preview["earlier_frame"] == index),
                        "Preview time differs from its native picture")
                require(preview[f"{side}_rgb_sha256"] == hashlib.sha256(frame).hexdigest(),
                        "Native picture RGB differs from the export")
                asset = preview["crop_assets"][side]
                png = base64.b64decode(assets[asset["file"]], validate=True)
                require(asset["media_type"] == "image/png" and hashlib.sha256(png).hexdigest() == asset["sha256"],
                        "Embedded crop bytes differ from their recorded hash")
                (Path(folder) / f"{role}.png").write_bytes(png)
                expected.append(hashlib.sha256(native_crop(frame, width, x, y, w, h)).hexdigest())
            check_exhibits(Path(folder), expected, w, h)
    require(clocks[0] == clocks[1], "Relative comparison clocks or dimensions differ")
    clock = clocks[0][1]
    return {"status": "verified", "scope": "two source files, shared interval clock and six native picture crops",
            "interval_seconds": [str(clock[start]), str(clock[end])], "native_crops_verified": 6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--expect-receipt-sha256")
    parser.add_argument("--exhibits", type=Path)
    parser.add_argument("--earlier", type=Path, help="Earlier clip for a paired comparison JSON export")
    args = parser.parse_args()
    try:
        require(args.earlier is None or args.exhibits is None, "Paired comparison crops must come from the export")
        limit = MAX_COMPARISON_BYTES if args.earlier else 65536
        require(args.receipt.stat().st_size <= limit, "Interval receipt exceeds the metadata limit")
        payload = args.receipt.read_bytes()
        if args.expect_receipt_sha256 is not None:
            require(hashlib.sha256(payload).hexdigest() == args.expect_receipt_sha256,
                    "Receipt SHA-256 differs from the caller's commitment")
        receipt = json.loads(payload)
        result = (verify_comparison(args.earlier, args.source, receipt) if args.earlier
                  else verify(args.source, receipt, args.exhibits))
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(json.dumps({"status": "not_verified", "reason": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
