"""Build data-only, source-clock VCD2 ordinary-edit controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_vcd2_genuine_v197 as source

BASE = source.previous
TRIAL = BASE.ARCHIVE / "video-vcd2-controls-v200"
SOURCE_FREEZE = "6863b19519d9262485638e1ac6717de80b217b5bd1fc0594365bf262a0b813db"
PUBLIC_SOURCE = BASE.ROOT / "docs/evaluation-panels/video-vcd2-genuine-roster-freeze-v197.json"
CODE = (
    "backend/scripts/video_vcd2_controls_v200.py",
    "backend/tests/test_video_vcd2_controls_v200.py",
    "backend/app/detectors/video/decode.py",
    "docs/video-vcd2-controls-v200-method.md",
)
ROLES = ("unchanged", "chroma_conversion_only", "recompression_only")
INTERMEDIATE_MATRIX = "bt709"
MATRICES = {
    "bt709": "bt709",
    "bt470bg": "bt601",
    "smpte170m": "bt601",
    "bt2020nc": "bt2020",
    "smpte240m": "smpte240m",
    "fcc": "fcc",
}
YUV = {"yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p"}
ATTEMPT_NAME = re.compile(r"slot-(\d{2})-attempt-0([1-9])")
RGB_BYTES_PER_PIXEL = 3
require, sha, write = BASE.require, BASE.sha, BASE.write


@dataclass(frozen=True)
class Recipe:
    width: int
    height: int
    pts: tuple[int, ...]
    origin_pts: int
    tb: Fraction
    last_duration: int
    interval: tuple[int, int]
    source_range: str
    source_matrix: str
    assumed_range: bool
    assumed_matrix: bool
    sar: Fraction
    assumed_sar: bool


def read(path: Path):
    require(path.resolve() == path and path.is_file(), "Expected an owned regular receipt")
    return json.loads(path.read_text(encoding="utf-8"))


def source_plan(metadata: Path) -> tuple[dict, list[dict]]:
    locked, cases = source.context(metadata)
    prior = source.TRIAL / "freeze.json"
    require(
        read(prior) == read(PUBLIC_SOURCE) == locked and sha(prior) == sha(PUBLIC_SOURCE) == SOURCE_FREEZE,
        "Exact published and private v197 roster required",
    )
    return locked, cases


def inventory(cases: list[dict], unavailable: set[int]) -> list[dict]:
    by_slot: dict[int, list[Path]] = {case["slot"]: [] for case in cases}
    for folder in sorted(source.TRIAL.iterdir()):
        match = ATTEMPT_NAME.fullmatch(folder.name)
        if match:
            slot = int(match[1])
            require(
                slot in by_slot and folder.resolve() == folder and folder.is_dir(),
                "Unexpected qualification attempt",
            )
            by_slot[slot].append(folder)
    result = []
    for case in cases:
        slot = case["slot"]
        attempts = []
        successes = []
        for folder in by_slot[slot]:
            qualification = folder / "qualification.json"
            success = qualification.exists() or qualification.is_symlink()
            attempts.append({"attempt": folder.name, "status": "qualified" if success else "incomplete"})
            if success:
                receipt = read(qualification)
                acquisition = folder / "acquisition.json"
                official = read(acquisition)
                file = folder / "source.mkv"
                require(
                    file.resolve() == file
                    and file.is_file()
                    and file.stat().st_size == official["bytes"]
                    and receipt["slot"] == slot
                    and receipt["attempt"] == int(folder.name[-2:])
                    and receipt["sha256"] == official["sha256"]
                    and official["source_url"] == BASE.AZURE + case["benchmark_path"],
                    "Mismatched v197 source receipt",
                )
                width, height = (int(dimension) for dimension in case["resolution"].split("x"))
                frames = receipt["clock"]["frame_count"]
                require(
                    type(frames) is int and 3 <= frames <= 1200, "Unexpected qualified source frame count"
                )
                successes.append(
                    {
                        "attempt": folder.name,
                        "source_sha256": receipt["sha256"],
                        "qualification_sha256": sha(qualification),
                        "acquisition_sha256": sha(acquisition),
                        "rgb_picture_bytes": width * height * frames * RGB_BYTES_PER_PIXEL,
                    }
                )
        require(
            len({r["source_sha256"] for r in successes}) <= 1,
            "Successful retry changed the qualified source bytes",
        )
        require(
            bool(successes) != (slot in unavailable),
            "Each slot needs either a qualified receipt or explicit unavailability",
        )
        result.append(
            {
                "slot": slot,
                "status": "qualified" if successes else "unavailable",
                "attempts": attempts,
                "selected": successes[0] if successes else None,
            }
        )
    require(unavailable <= by_slot.keys(), "Unknown unavailable slot")
    return result


def storage_projection(slots: list[dict]) -> dict:
    source_sizes = [row["selected"]["rgb_picture_bytes"] for row in slots if row["selected"]]
    return {
        "retained_rgb_equivalent_bytes": sum(source_sizes) * len(ROLES),
        "largest_slot_workspace_estimate_bytes": max(source_sizes, default=0) * (len(ROLES) + 1),
    }


def tool(name: str) -> str:
    path = shutil.which(name)
    require(path is not None, f"Local {name} is required")
    return str(Path(path).resolve())


def build_identity() -> dict:
    return {
        "schema": "chipmunk.video-vcd2-controls/200",
        "source_freeze_sha256": SOURCE_FREEZE,
        "code_sha256": {name: sha(BASE.ROOT / name) for name in CODE},
        "tool_sha256": {name: sha(Path(tool(name))) for name in ("ffmpeg", "ffprobe")},
        "roles": list(ROLES),
        "relative_interval_s": list(BASE.WINDOW),
        "model_calls": 0,
    }


def recipe(qualification: dict, video: dict) -> Recipe:
    clock = qualification["clock"]
    raw_pts = tuple(clock["pts"])
    require(
        len(raw_pts) == clock["frame_count"]
        and len(raw_pts) >= 3
        and all(type(value) is int for value in raw_pts)
        and all(a < b for a, b in zip(raw_pts, raw_pts[1:])),
        "Qualified source clock is incomplete",
    )
    tb = Fraction(clock["time_base"])
    require(
        tb == Fraction(video["time_base"]) and tb > 0 and tb.denominator <= 1_000_000 and tb.numerator == 1,
        "Expected an exact source time base representable on the final MP4 clock",
    )
    pts = tuple(value - raw_pts[0] for value in raw_pts)
    during = [index for index, value in enumerate(pts) if BASE.WINDOW[0] <= value * tb < BASE.WINDOW[1]]
    require(
        during
        and during[0] > 0
        and during[-1] + 1 < len(pts)
        and [during[0], during[-1] + 1] == clock["ordinary_frame_interval"]
        and [raw_pts[during[0]], raw_pts[during[-1] + 1]] == clock["ordinary_picture_pts"],
        "Frozen frame interval differs from the qualified source clock",
    )
    durations = clock["packet_durations"]
    require(len(durations) == len(pts), "Qualified packet clock is incomplete")
    last_duration = durations[-1] if type(durations[-1]) is int and durations[-1] > 0 else pts[-1] - pts[-2]
    width, height = video["width"], video["height"]
    require(
        width % 2 == height % 2 == 0 and video.get("pix_fmt") in YUV,
        "Expected an even-sized 8-bit YUV source",
    )
    require(
        video.get("field_order") in {None, "unknown", "progressive"},
        "Interlaced webcam video is out of scope",
    )
    tag_range, tag_matrix, tag_sar = (
        video.get(key) for key in ("color_range", "color_space", "sample_aspect_ratio")
    )
    unspecified = {None, "unknown", "unspecified", "N/A"}
    assume_range, assume_matrix = tag_range in unspecified, tag_matrix in unspecified
    require(assume_range or tag_range in {"pc", "tv"}, "Unknown tagged color range")
    require(assume_matrix or tag_matrix in MATRICES, "Unknown tagged color matrix")
    assume_sar = tag_sar in unspecified or tag_sar == "0:1"
    sar = Fraction(1) if assume_sar else Fraction(tag_sar.replace(":", "/"))
    require(sar > 0 and max(sar.numerator, sar.denominator) <= 65535, "Unsupported sample aspect ratio")
    return Recipe(
        width,
        height,
        pts,
        raw_pts[0],
        tb,
        last_duration,
        (during[0], during[-1] + 1),
        ("pc" if video["pix_fmt"].startswith("yuvj") else "tv") if assume_range else tag_range,
        "bt709" if assume_matrix else MATRICES[tag_matrix],
        assume_range,
        assume_matrix,
        sar,
        assume_sar,
    )


def normalize(range_name: str, matrix: str) -> str:
    return f"scale=in_range={range_name}:out_range=pc:in_color_matrix={matrix}:flags=bilinear+accurate_rnd+bitexact,format=rgb24"


def ffmpeg(*args: str, timeout: int = 180) -> str:
    process = subprocess.run(
        [tool("ffmpeg"), "-nostdin", "-v", "error", "-n", "-protocol_whitelist", "file", "-copyts", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    require(not process.stderr.strip(), "FFmpeg reported an encoding or decoding error")
    return process.stdout


def clock_filter(plan: Recipe) -> str:
    return f"settb=expr={plan.tb},setpts=PTS-STARTPTS,setsar=ratio={plan.sar}:max=65535"


def input_args(path: Path) -> list[str]:
    return ["-threads", "1", "-i", f"file:{path}"]


def render(
    path: Path, target: Path, plan: Recipe, *, codec_variant: str | None = None, edit: Path | None = None
):
    src = f"{normalize(plan.source_range, plan.source_matrix)},{clock_filter(plan)}"
    inputs = input_args(path)
    graph = f"[0:v:0]{src}[out]"
    if edit is not None:
        require(codec_variant is None, "Final controls only can include an edited interval")
        inputs.extend(input_args(edit))
        start, stop = plan.interval
        choose = f"between(n,{start},{stop - 1})"
        graph = (
            f"[0:v:0]{src},select='not({choose})'[outer];"
            f"[1:v:0]{normalize('tv', INTERMEDIATE_MATRIX)},{clock_filter(plan)},select='{choose}'[inner];"
            f"[outer][inner]interleave=2,settb=expr={plan.tb},format=rgb24[out]"
        )
    if codec_variant is not None:
        require(codec_variant in {"chroma_conversion_only", "recompression_only"}, "Unknown edit condition")
        graph = (
            f"[0:v:0]{src},scale=in_range=pc:out_range=tv:out_color_matrix={INTERMEDIATE_MATRIX}:"
            "flags=bilinear+accurate_rnd+bitexact,format=yuv420p[out]"
        )
        quality = ["-qp", "0"] if codec_variant == "chroma_conversion_only" else ["-crf", "24"]
        codec = [
            "-c:v",
            "libx264",
            *quality,
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            INTERMEDIATE_MATRIX,
        ]
    else:
        codec = [
            "-c:v",
            "libx264rgb",
            "-qp",
            "0",
            "-pix_fmt",
            "rgb24",
            "-color_range",
            "pc",
            "-colorspace",
            "rgb",
        ]
    ffmpeg(
        *inputs,
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-fps_mode",
        "passthrough",
        *codec,
        "-preset",
        "veryfast",
        "-bf",
        "0",
        "-threads",
        "1",
        "-enc_time_base",
        str(plan.tb),
        "-bsf:v",
        f"setts=duration={plan.last_duration}",
        "-video_track_timescale",
        str(plan.tb.denominator),
        str(target),
    )


def normalized_hashes(path: Path, plan: Recipe, *, edited=False) -> list[str]:
    range_name, matrix = ("tv", INTERMEDIATE_MATRIX) if edited else (plan.source_range, plan.source_matrix)
    log = ffmpeg(
        *input_args(path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-filter_threads",
        "1",
        "-vf",
        f"{normalize(range_name, matrix)},{clock_filter(plan)}",
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        str(plan.tb),
        "-c:v",
        "rawvideo",
        "-threads",
        "1",
        "-f",
        "framehash",
        "-hash",
        "sha256",
        "pipe:1",
    )
    lines = log.splitlines()
    rows = [
        [field.strip() for field in line.split(",")] for line in lines if line and not line.startswith("#")
    ]
    require(
        f"#tb 0: {plan.tb}" in lines
        and len(rows) == len(plan.pts)
        and all(
            len(row) == 6
            and row[0] == "0"
            and int(row[4]) == plan.width * plan.height * 3
            and re.fullmatch(r"[0-9a-f]{64}", row[5])
            for row in rows
        )
        and tuple(int(row[2]) for row in rows) == plan.pts,
        "Independent RGB frame hashes differ from the source clock",
    )
    return [row[5] for row in rows]


def verify_clock(path: Path, plan: Recipe, *, edited=False):
    metadata = BASE.ffprobe(
        path,
        "-show_entries",
        "stream=codec_type,codec_name,width,height,pix_fmt,color_range,color_space,time_base,"
        "duration_ts,sample_aspect_ratio:packet=pts,dts,duration",
    )
    streams = metadata["streams"]
    require(len(streams) == 1, "Expected one silent output video stream")
    stream = streams[0]
    allowed_sar = stream.get("sample_aspect_ratio") or "1:1"
    require(
        stream["codec_type"] == "video"
        and stream["codec_name"] == "h264"
        and (stream["width"], stream["height"]) == (plan.width, plan.height)
        and Fraction(stream["time_base"]) == plan.tb
        and Fraction(allowed_sar.replace(":", "/")) == plan.sar
        and stream["duration_ts"] == plan.pts[-1] + plan.last_duration,
        "Output geometry or duration changed",
    )
    require(
        (stream["pix_fmt"], stream.get("color_range")) == (("yuv420p", "tv") if edited else ("gbrp", "pc")),
        "Unexpected output pixel format or range",
    )
    require(
        not edited or stream.get("color_space") == INTERMEDIATE_MATRIX, "Intermediate color matrix changed"
    )
    durations = [b - a for a, b in zip(plan.pts, plan.pts[1:])] + [plan.last_duration]
    require(
        metadata["packets"]
        == [
            {"pts": pts, "dts": pts, "duration": duration}
            for pts, duration in zip(plan.pts, durations, strict=True)
        ],
        "Output packet clock changed",
    )
    table = probe_frames(path, cap=1201, timeout_s=120)
    require(
        not table.truncated
        and not table.interpolated
        and table.rotation == 0
        and table.tb == plan.tb
        and table.pts == plan.pts,
        "Production decoder picture clock changed",
    )
    return table


def expected_edit(base: list[str], edit: list[str], interval: tuple[int, int]) -> list[str]:
    start, stop = interval
    require(len(base) == len(edit) and 0 < start < stop < len(base), "Invalid source-aligned replacement")
    return [*base[:start], *edit[start:stop], *base[stop:]]


def verify_pixels(
    path: Path, table, expected: list[str], *, baseline: Path | None = None, interval=None
) -> dict:
    differences = []
    seen = 0
    with closing(decode_frames(path, table, None, timeout_s=180)) as generated:
        reference = decode_frames(baseline, table, None, timeout_s=180) if baseline is not None else None
        try:
            for frame in generated:
                require(
                    frame.index == seen and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == expected[seen],
                    "Final decoded RGB differs from the intended frame",
                )
                seen += 1
                if reference is not None:
                    before = next(reference, None)
                    require(
                        before is not None and before.index == frame.index and before.pts == frame.pts,
                        "Paired decoded picture times differ",
                    )
                    if not np.array_equal(frame.rgb, before.rgb):
                        require(
                            interval[0] <= frame.index < interval[1],
                            "Pixel change outside the ordinary-edit interval",
                        )
                        delta = np.subtract(frame.rgb, before.rgb, dtype=np.int16)
                        np.abs(delta, out=delta)
                        differences.append(
                            {"index": frame.index, "pts": frame.pts, "rgb_mae": float(delta.mean())}
                        )
            require(
                reference is None or next(reference, None) is None, "The reference has additional pictures"
            )
        finally:
            if reference is not None:
                reference.close()
    require(seen == len(expected), "The final decoder omitted pictures")
    return {"changed_frames": differences, "verified_frames": seen}


def construct(path: Path, folder: Path, plan: Recipe) -> dict:
    baseline = folder / "unchanged.mp4"
    base_hashes = normalized_hashes(path, plan)
    render(path, baseline, plan)
    base_table = verify_clock(baseline, plan)
    cases = {
        "unchanged": {
            "file": baseline.name,
            "sha256": sha(baseline),
            "rgb_sha256": base_hashes,
            **verify_pixels(baseline, base_table, base_hashes),
        }
    }
    hashes = {"unchanged": base_hashes}
    for role in ROLES[1:]:
        intermediate = folder / f"{role}-intermediate.mp4"
        output = folder / f"{role}.mp4"
        render(path, intermediate, plan, codec_variant=role)
        verify_clock(intermediate, plan, edited=True)
        expected = expected_edit(
            base_hashes, normalized_hashes(intermediate, plan, edited=True), plan.interval
        )
        render(path, output, plan, edit=intermediate)
        table = verify_clock(output, plan)
        receipt = verify_pixels(output, table, expected, baseline=baseline, interval=plan.interval)
        cases[role] = {
            "file": output.name,
            "sha256": sha(output),
            "intermediate_sha256": sha(intermediate),
            "rgb_sha256": expected,
            **receipt,
        }
        hashes[role] = expected
        intermediate.unlink()
    lossy_changes = [
        i for i, (a, b) in enumerate(zip(hashes[ROLES[1]], hashes[ROLES[2]], strict=True)) if a != b
    ]
    require(
        all(plan.interval[0] <= i < plan.interval[1] for i in lossy_changes),
        "Unexpected out-of-interval codec difference",
    )
    require(
        cases["recompression_only"]["changed_frames"] and lossy_changes,
        "The lossy control produced no distinct codec changes",
    )
    return {"cases": cases, "lossy_codec_changed_indices": lossy_changes}


def prepare(frozen: dict, slot: int, attempt: int):
    require(1 <= slot <= len(frozen["slots"]) and 1 <= attempt <= 9, "Invalid source slot or attempt")
    entry = frozen["slots"][slot - 1]
    require(entry["slot"] == slot and entry["status"] == "qualified", "Selected slot is unavailable")
    selected = entry["selected"]
    folder = source.TRIAL / selected["attempt"]
    path = folder / "source.mkv"
    qualification, acquisition = folder / "qualification.json", folder / "acquisition.json"
    require(
        sha(qualification) == selected["qualification_sha256"]
        and sha(acquisition) == selected["acquisition_sha256"]
        and path.resolve() == path
        and sha(path) == selected["source_sha256"],
        "Qualified source changed since freeze",
    )
    receipt = read(qualification)
    video = BASE.ffprobe(
        path,
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,color_range,color_space,time_base,sample_aspect_ratio,field_order",
    )["streams"][0]
    plan = recipe(receipt, video)
    reserve = selected["rgb_picture_bytes"] * (len(ROLES) + 1) + BASE.MIN_ARCHIVE_FREE
    require(shutil.disk_usage(TRIAL).free >= reserve, "Insufficient archive space for a complete recording")
    output = TRIAL / f"slot-{slot:02d}-attempt-{attempt:02d}"
    BASE.private_directory(output)
    result = construct(path, output, plan)
    require(sha(path) == selected["source_sha256"], "Source changed while constructing the controls")
    write(
        output / "qualification.json",
        {
            "slot": slot,
            "source_sha256": selected["source_sha256"],
            "frame_count": len(plan.pts),
            "width": plan.width,
            "height": plan.height,
            "source_origin_pts": plan.origin_pts,
            "time_base": str(plan.tb),
            "relative_pts": list(plan.pts),
            "last_duration": plan.last_duration,
            "frame_interval": list(plan.interval),
            "source_color": {
                "range": plan.source_range,
                "matrix": plan.source_matrix,
                "assumed_range": plan.assumed_range,
                "assumed_matrix": plan.assumed_matrix,
                "sample_aspect_ratio": str(plan.sar),
                "assumed_sar": plan.assumed_sar,
            },
            "model_calls": 0,
            **result,
        },
    )
    print(json.dumps({"status": "prepared", "slot": slot, "cases": len(result["cases"]), "model_calls": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "prepare"))
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--unavailable", type=int, nargs="*", default=[])
    parser.add_argument("--slot", type=int)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if (args.command == "prepare") != (args.slot is not None):
        parser.error("--slot is required only for prepare")
    if args.command == "prepare" and args.unavailable:
        parser.error("unavailability is set at freeze")
    _, cases = source_plan(args.metadata_dir)
    identity = build_identity()
    if args.command == "prepare":
        BASE.archive_ready()
        frozen = read(TRIAL / "freeze.json")
        unavailable = {row["slot"] for row in frozen["slots"] if row["status"] == "unavailable"}
        require(
            frozen == {"identity": identity, "slots": inventory(cases, unavailable)},
            "Exact v200 method and source freeze required",
        )
        prepare(frozen, args.slot, args.attempt)
        return
    slots = inventory(cases, set(args.unavailable))
    if args.command == "freeze":
        BASE.archive_ready()
        BASE.private_directory(TRIAL)
        write(TRIAL / "freeze.json", {"identity": identity, "slots": slots})
    print(
        json.dumps(
            {
                "status": args.command,
                "source_outcomes": dict(Counter(row["status"] for row in slots)),
                "storage_projection": storage_projection(slots),
            }
        )
    )


if __name__ == "__main__":
    main()
