"""Verify a three-frame video interval receipt offline with Python and FFmpeg."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.native-video-interval/1"
ROLES = ("before", "during", "after")
SHA = re.compile(r"[0-9a-f]{64}")
TIMESTAMP = re.compile(rb"\bn:\s*(\d+)\s+pts:\s*(-?\d+)")
INPUT_CLOCK = re.compile(rb"config in time_base:\s*(\d+)/(\d+)")
TIMEOUT = 60
MAX_FRAME_PIXELS = 4096 * 2160


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


def rgb_frames(path, indices, pts, time_base, frame_bytes):
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
    return tuple(hashlib.sha256(decoded.stdout[i * frame_bytes:(i + 1) * frame_bytes]).hexdigest()
                 for i in range(len(indices)))


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
    require(rgb_frames(source_path, indices, native, time_base, width * height * 3) == expected,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--expect-receipt-sha256")
    parser.add_argument("--exhibits", type=Path)
    args = parser.parse_args()
    try:
        require(args.receipt.stat().st_size <= 65536, "Interval receipt exceeds the metadata limit")
        payload = args.receipt.read_bytes()
        if args.expect_receipt_sha256 is not None:
            require(hashlib.sha256(payload).hexdigest() == args.expect_receipt_sha256,
                    "Receipt SHA-256 differs from the caller's commitment")
        result = verify(args.source, json.loads(payload), args.exhibits)
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(json.dumps({"status": "not_verified", "reason": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
