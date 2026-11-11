"""Freeze VCD2 recording controls and qualify one official source without inference."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path, PurePosixPath

REVISION = "3c87cbdbe68505da6d590cc4ed8c2af73bcc5c68"
ROOT = Path(__file__).resolve().parents[2]
VOLUME = Path("/Volumes/SEAGATE")
ARCHIVE = VOLUME / "Codex-Development-Archive/2026-11-08-Chipmunk-Verify"
TRIAL = ARCHIVE / "video-vcd2-genuine-v195"
AZURE = "https://vcdpublic.blob.core.windows.net/vcd2/"
RAW_REPO = f"https://raw.githubusercontent.com/microsoft/VCD/{REVISION}/"
PINS = {
    "vcd2_metadata.csv": (
        AZURE + "metadata/vcd2_metadata.csv",
        "4d3043d4963c780ca2653e1a3c0703f3aa7923d8f3a43044eff439cd987e6d56",
    ),
    "benchmarking-set_metadata.csv": (
        AZURE + "metadata/benchmarking-set_metadata.csv",
        "ac2aa2c59399a7701d8120b5905545884a7cf6296eb2bdcb2db2eae3f9830452",
    ),
    "raw_mkv_list.txt": (
        RAW_REPO + "download/vcd2/raw_mkv_list.txt",
        "f2656d37685f91159478dbd569a9760164a394f1be31dd7a0424e93fbbfdb237",
    ),
    "benchmarking_mkv_list.txt": (
        RAW_REPO + "download/vcd2/benchmarking_mkv_list.txt",
        "d8920e6f0a9508bb8b3dec89475e03b6e4f6003089ca6ffe3fa00f3beea43b52",
    ),
    "LICENSE-DATA": (
        RAW_REPO + "LICENSE-DATA",
        "4531a67d443284d93ffed0803df5b10634aff21c3d77e381f2d48af01d875868",
    ),
}
STRATA = ("High SI - High TI", "High SI - Low TI", "Low SI - High TI", "Low SI - Low TI")
SEED = "chipmunk-vcd2-genuine-20261110"
CAMERA_CODE = re.compile(r"vid_[0-9a-f]{4}&pid_[0-9a-f]{4}", re.I)
RAW_PATH = re.compile(r"raw/mkv/[0-9a-f]{32}\.mkv")
BENCH_PATH = re.compile(r"benchmarking/mkv/(TH|TH-BB|TH-BR)/[0-9a-f]{32}\.mkv")
MAX_BYTES = 512 * 1024**2
MIN_ARCHIVE_FREE = 3 * 1024**3
WINDOW = (3, 6)


def require(condition, message: str):
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, record: dict):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked_index(directory: Path, csv_name: str, list_name: str, pattern, count: int) -> dict:
    with (directory / csv_name).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    paths = [row["relative_path"] for row in rows]
    published = (directory / list_name).read_text(encoding="utf-8").splitlines()
    require(
        len(paths) == len(published) == len(set(paths)) == count
        and set(paths) == set(published)
        and all(pattern.fullmatch(path) for path in paths),
        "Microsoft CSV and file list disagree",
    )
    by_stem = {PurePosixPath(row["relative_path"]).stem: row for row in rows}
    require(len(by_stem) == count, "Recording stems must be unique")
    return by_stem


def roster(directory: Path) -> list[dict]:
    for name, (_, digest) in PINS.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink() and sha(path) == digest, f"Pin mismatch for {name}")
    raw = checked_index(directory, "vcd2_metadata.csv", "raw_mkv_list.txt", RAW_PATH, 838)
    bench = checked_index(
        directory, "benchmarking-set_metadata.csv", "benchmarking_mkv_list.txt", BENCH_PATH, 120
    )
    require(
        bench.keys() <= raw.keys()
        and Counter(r["group"] for r in bench.values()) == {"TH": 40, "TH-BB": 40, "TH-BR": 40},
        "Unexpected recording or benchmark group",
    )
    require(
        all(
            r["relative_path"].split("/")[2] == r["group"] == raw[k]["vcd2_benchmarking"]
            for k, r in bench.items()
        ),
        "Microsoft raw and benchmark group labels disagree",
    )
    ranked = sorted(bench, key=lambda key: hashlib.sha256(f"{SEED}:{key}".encode()).digest())
    cameras, cases = set(), []
    for stratum in STRATA:
        selected = []
        for key in ranked:
            source, candidate = raw[key], bench[key]
            camera = source["log_recording_camera_code"].lower()
            if (
                candidate["group"] != "TH"
                or source["si_ti_group"] != stratum
                or source["video_resolution"] not in {"1280x720", "1920x1080"}
                or not CAMERA_CODE.fullmatch(camera)
                or camera in cameras
            ):
                continue
            require(
                candidate["video_resolution"] == source["video_resolution"]
                and candidate["log_recording_camera_code"].lower() == camera,
                "Microsoft source pairing disagrees",
            )
            cameras.add(camera)
            selected.append(
                {
                    "slot": len(cases) + len(selected) + 1,
                    "recording_stem": key,
                    "camera_type": camera,
                    "stratum": stratum,
                    "resolution": source["video_resolution"],
                    "scenario": source["log_scenario_id"],
                    "lighting_issue_fraction": float(source["token_light_percent"]),
                    "capture_format": source["log_recording_input_fmt"],
                    "native_path": source["relative_path"],
                    "benchmark_path": candidate["relative_path"],
                    "benchmark_reported_frames": int(candidate["video_frames"]),
                    "benchmark_reported_duration": candidate["video_duration"],
                }
            )
            if len(selected) == 3:
                break
        require(len(selected) == 3, "Insufficient distinct camera types within the four SI/TI strata")
        cases.extend(selected)
    return cases


def context(directory: Path) -> dict:
    cases = roster(directory)
    return {
        "schema": "chipmunk.video-vcd2-genuine/195",
        "metadata": {name: {"url": url, "sha256": digest} for name, (url, digest) in PINS.items()},
        "code_sha256": {
            path: sha(ROOT / path)
            for path in (
                "backend/scripts/video_vcd2_genuine_v195.py",
                "backend/app/detectors/video/decode.py",
                "docs/video-vcd2-genuine-v195-method.md",
            )
        },
        "seed": SEED,
        "sample_unit": "recording stem; participant identity is unavailable",
        "declared_license": "CDLA-Permissive-2.0",
        "source_origin": "genuine camera, Microsoft TH benchmark",
        "relative_ordinary_interval_s": list(WINDOW),
        "cases": cases,
    }


def archive_ready():
    require(
        VOLUME.is_mount()
        and ARCHIVE.is_dir()
        and ARCHIVE.resolve() == ARCHIVE
        and shutil.disk_usage(ARCHIVE).free >= MIN_ARCHIVE_FREE,
        "Mounted Seagate archive with 3 GiB free is required",
    )


def private_directory(path: Path):
    path.mkdir(mode=0o700, exist_ok=False)


def freeze(directory: Path):
    record = context(directory)
    archive_ready()
    private_directory(TRIAL)
    write(TRIAL / "freeze.json", record)
    print(json.dumps({"status": "frozen", "recording_components": len(record["cases"])}))


def download(url: str, output: Path) -> dict:
    import httpx

    deadline = time.monotonic() + 180
    with httpx.Client(follow_redirects=False, timeout=25, headers={"Accept-Encoding": "identity"}) as client:
        head = client.head(url)
        head.raise_for_status()
        etag, length = head.headers.get("ETag"), int(head.headers.get("Content-Length", "0"))
        require(
            head.status_code == 200
            and etag
            and etag.startswith('"')
            and etag.endswith('"')
            and 0 < length <= MAX_BYTES,
            "Missing or out-of-bounds official object headers",
        )
        require(
            head.headers.get("Content-Encoding", "identity") == "identity",
            "Expected exact uncompressed HTTP body",
        )
        md5_header = head.headers.get("Content-MD5")
        digests = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
        size = 0
        with client.stream("GET", url, headers={"If-Match": etag}) as response:
            response.raise_for_status()
            response_md5 = response.headers.get("Content-MD5")
            require(
                response.status_code == 200
                and response.headers.get("ETag") == etag
                and response.headers.get("Content-Length") == str(length)
                and response.headers.get("Content-Encoding", "identity") == "identity"
                and response.headers.get("Content-Type") == head.headers.get("Content-Type")
                and (not md5_header or not response_md5 or md5_header == response_md5),
                "Official GET did not match the HEAD object",
            )
            with output.open("xb") as stream:
                for chunk in response.iter_raw(1024 * 1024):
                    size += len(chunk)
                    require(
                        time.monotonic() < deadline and size <= length, "Source transfer exceeded its limits"
                    )
                    for digest in digests:
                        digest.update(chunk)
                    stream.write(chunk)
        require(size == length, "Official source length changed during transfer")
        actual_md5 = base64.b64encode(digests[1].digest()).decode("ascii")
        publisher_md5 = md5_header or response_md5
        require(
            not publisher_md5 or publisher_md5 == actual_md5,
            "Official Content-MD5 disagrees with received bytes",
        )
        return {
            "source_url": url,
            "bytes": size,
            "sha256": digests[0].hexdigest(),
            "etag": etag,
            "publisher_content_md5": publisher_md5,
            "last_modified": head.headers.get("Last-Modified"),
        }


def ffprobe(path: Path, *flags: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", *flags, "-of", "json", f"file:{path}"],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    require(not result.stderr.strip(), "FFprobe reported an error in the official video")
    return json.loads(result.stdout)


def clock_record(table, packets: list[dict]) -> dict:
    pts = list(table.pts)
    require(
        2 <= len(pts) <= 1200
        and not table.truncated
        and not table.interpolated
        and table.rotation == 0
        and all(a < b for a, b in zip(pts, pts[1:])),
        "Video requires a complete monotone picture clock without interpolated PTS",
    )
    require(
        [packet.get("pts") for packet in packets] == pts, "Source packet and decoded picture PTS disagree"
    )
    seconds = [(pts_value - pts[0]) * table.tb for pts_value in pts]
    require(Fraction(7) <= seconds[-1] <= Fraction(20), "Unexpected benchmark duration")
    during = [index for index, second in enumerate(seconds) if WINDOW[0] <= second < WINDOW[1]]
    require(
        during and during[0] > 0 and during[-1] + 1 < len(pts),
        "Ordinary intervention requires before, during and after pictures",
    )
    return {
        "time_base": table.time_base,
        "pts": pts,
        "packet_durations": [packet.get("duration") for packet in packets],
        "last_picture_s_relative": str(seconds[-1]),
        "frame_count": len(pts),
        "ordinary_frame_interval": [during[0], during[-1] + 1],
        "ordinary_picture_pts": [pts[during[0]], pts[during[-1] + 1]],
        "actual_picture_span_s_relative": [str(seconds[during[0]]), str(seconds[during[-1] + 1])],
    }


def media_record(path: Path, resolution: str) -> dict:
    from app.detectors.video.decode import probe_frames

    with path.open("rb") as stream:
        require(
            stream.read(4) == bytes.fromhex("1a45dfa3"), "Official source does not have a Matroska header"
        )
    header = ffprobe(
        path,
        "-show_entries",
        "format=format_name,duration:stream=codec_type,codec_name,width,height,"
        "pix_fmt,color_range,color_space,color_primaries,color_transfer,time_base",
    )
    video = [stream for stream in header["streams"] if stream.get("codec_type") == "video"]
    require(
        "matroska" in header["format"]["format_name"].split(",")
        and len(video) == 1
        and video[0]["codec_name"] == "ffv1",
        "Expected one official lossless FFV1 video stream",
    )
    table = probe_frames(path, cap=1201, timeout_s=120)
    require(
        f"{table.width}x{table.height}" == resolution and table.time_base == video[0]["time_base"],
        "Decoded geometry or source time base disagrees",
    )
    packet_data = ffprobe(path, "-select_streams", "v:0", "-show_entries", "packet=pts,duration")
    return {"header": header, "clock": clock_record(table, packet_data["packets"])}


def qualify(directory: Path, slot: int, attempt: int):
    record = context(directory)
    archive_ready()
    require(
        TRIAL.is_dir()
        and TRIAL.resolve() == TRIAL
        and not (TRIAL / "freeze.json").is_symlink()
        and json.loads((TRIAL / "freeze.json").read_text()) == record,
        "Exact private roster freeze required",
    )
    require(1 <= slot <= len(record["cases"]), "Select one roster slot from 1 to 12")
    require(1 <= attempt <= 9, "Select an attempt number from 1 to 9")
    source = record["cases"][slot - 1]
    folder = TRIAL / f"slot-{slot:02d}-attempt-{attempt:02d}"
    private_directory(folder)
    partial, file = folder / "source.mkv.partial", folder / "source.mkv"
    acquisition = download(AZURE + source["benchmark_path"], partial)
    require(sha(partial) == acquisition["sha256"], "Stored bytes differ from the received official object")
    partial.rename(file)
    write(folder / "acquisition.json", acquisition)
    verified = media_record(file, source["resolution"])
    require(sha(file) == acquisition["sha256"], "Source changed while the picture clock was verified")
    write(
        folder / "qualification.json",
        {"slot": slot, "attempt": attempt, "sha256": acquisition["sha256"], **verified},
    )
    print(json.dumps({"status": "qualified", "slot": slot, "frames": verified["clock"]["frame_count"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "qualify"))
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--attempt", type=int)
    args = parser.parse_args()
    if (args.command == "qualify") != (args.slot is not None):
        parser.error("--slot is required only for qualify")
    if args.command != "qualify" and args.attempt is not None:
        parser.error("--attempt applies only to qualify")
    if args.command == "preflight":
        cases = roster(args.metadata_dir)
        print(
            json.dumps(
                {
                    "status": "metadata_verified",
                    "recording_components": len(cases),
                    "camera_types": len({row["camera_type"] for row in cases}),
                    "by_resolution": dict(Counter(row["resolution"] for row in cases)),
                    "by_stratum": dict(Counter(row["stratum"] for row in cases)),
                }
            )
        )
    elif args.command == "freeze":
        freeze(args.metadata_dir)
    else:
        qualify(args.metadata_dir, args.slot, 1 if args.attempt is None else args.attempt)


if __name__ == "__main__":
    main()
