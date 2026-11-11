"""Prepare a matched synthetic swap interval and ordinary controls without model calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import cv2

from app.detectors.video.decode import probe_frames
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped
from scripts.video_native_inserts_v95 import verify_rgb

ROOT = runtime.ROOT
ARCHIVE = runtime.ARCHIVE
NAME = "video-faceswapai-matched-v186"
MODULE = "scripts.video_faceswapai_matched_v186"
TRIAL = ROOT / "backend/data/independent-evaluation" / NAME
PUBLIC = ROOT / "docs/evaluation-panels/video-faceswapai-matched-freeze-v186.json"
SOURCES = ARCHIVE / "video-faceswapai-continuity-v185"
MEDIA = ARCHIVE / NAME
STAGE = TRIAL / "bind-00"
STUDY_SHA = "e73ecc30c700b95246fa56977e6e596eb685035a16a0b19d72e9f14e76095e8c"
ASSETS = {
    "original": (
        "video-continuity-input-v1.mp4",
        956633,
        "3a8cefeb1e69d433ff171f2639c15c9d45e9d87932dba8c046c4105653af7f1f",
    ),
    "swap": (
        "video-continuity-output-v1.mp4",
        1329222,
        "8cd3b9ac4dfdc62c635d5bf0e00cb01b3c6ecd23cc08efdb0ec1dccd1592ad77",
    ),
    "donor": (
        "video-continuity-source-v1.jpg",
        157351,
        "5f6aa1a771c4940ed053d2dce3932f5c360840511e80893e72b507c5eaa05b59",
    ),
}
CODE = (
    "backend/scripts/video_faceswapai_matched_v186.py",
    "backend/tests/test_video_faceswapai_matched_v186.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/video_native_inserts_v95.py",
    "backend/scripts/video_native_sources_v92.py",
    "backend/app/detectors/video/decode.py",
    "docs/video-faceswapai-source-v185.md",
    "docs/video-faceswapai-matched-v186-method.md",
)
FRAME_COUNT, WIDTH, HEIGHT = 240, 1280, 720
TIME_BASE = "1/12288"
TICK, START, STOP = 512, 72, 144
PTS = list(range(0, FRAME_COUNT * TICK, TICK))
REPRESENTATIVES = {"before": START - 1, "during": (START + STOP - 1) // 2, "after": STOP}
ORIGINAL_COLOR = ("pc", "bt601")
SWAP_COLORS = {
    "swap_tv_bt601": ("tv", "bt601"),
    "swap_pc_bt601": ("pc", "bt601"),
    "swap_tv_bt709": ("tv", "bt709"),
    "swap_pc_bt709": ("pc", "bt709"),
}
RECOMPRESSION_COLOR = ("tv", "bt601")
ROLES = {
    "unchanged": "unswapped_synthetic_reference",
    "recompression_only": "ordinary_recompression_control",
    **dict.fromkeys(SWAP_COLORS, "publisher_swap_insertion"),
}
SCOPE = "One synthetic source component, controlled timing and codec evidence only."
read, write, sha, require = runtime.read, runtime.write, runtime.sha, runtime.require


def source_files():
    require(
        SOURCES.is_dir() and SOURCES.resolve() == SOURCES, "Exact local publisher source directory required"
    )
    manifest = SOURCES / "study.json"
    require(
        manifest.is_file() and not manifest.is_symlink() and sha(manifest) == STUDY_SHA,
        "Exact publisher JSON required",
    )
    study = read(manifest)
    require(
        study["version"] == "1.0.0"
        and study["dataLicense"] == "https://creativecommons.org/licenses/by/4.0/",
        "Pinned publisher study and data license required",
    )
    paths = {}
    for role, (name, size, digest) in ASSETS.items():
        path = SOURCES / name
        require(
            path.is_file() and not path.is_symlink() and path.stat().st_size == size and sha(path) == digest,
            f"Exact publisher {role} bytes required",
        )
        paths[role] = path
    return paths


def tool(name):
    path = shutil.which(name)
    require(path is not None, f"Local {name} required")
    return str(Path(path).resolve())


def identity():
    source_files()
    return {
        "schema": "chipmunk.video-faceswapai-matched-freeze/186",
        "publisher_manifest_sha256": STUDY_SHA,
        "publisher_asset_sha256": {key: value[2] for key, value in ASSETS.items()},
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "binary_sha256": {name: sha(Path(tool(name))) for name in ("ffmpeg", "ffprobe")},
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless", "psutil")
        },
        "conditions": ROLES,
        "source_color_assumptions": {key: list(color) for key, color in SWAP_COLORS.items()},
        "edit_frame_range": [START, STOP],
        "edit_pts_range": [PTS[START], PTS[STOP]],
        "native_clock": {"time_base": TIME_BASE, "step": TICK, "frames": FRAME_COUNT},
        "new_neural_calls": 0,
        "scope": SCOPE,
    }


def freeze():
    require(
        not any(path.exists() or path.is_symlink() for path in (TRIAL, PUBLIC, MEDIA)),
        "Preserve earlier or incomplete v186 attempts",
    )
    record = identity()
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", record)
    write(PUBLIC, record)
    return {"status": "frozen", "conditions": list(ROLES), "new_neural_calls": 0}


def verify():
    expected = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == expected, "Exact v186 freeze required")
    return expected


def stream_metadata(file, *, rendered=False):
    result = json.loads(
        subprocess.check_output(
            [
                tool("ffprobe"),
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_entries",
                "stream=codec_type,codec_name,width,height,pix_fmt,color_range,color_space,time_base,start_pts,duration_ts,nb_frames"
                + (":packet=pts,dts,duration" if rendered else ""),
                "-of",
                "json",
                str(file),
            ],
            timeout=60,
            text=True,
        )
    )
    rows = result["streams"]
    require(
        len(rows) == 1 and rows[0]["codec_type"] == "video", "Exactly one video stream and no audio required"
    )
    stream = rows[0]
    require(
        stream["codec_name"] == "h264"
        and (stream["width"], stream["height"]) == (WIDTH, HEIGHT)
        and stream["time_base"] == TIME_BASE
        and stream["start_pts"] == 0
        and stream["duration_ts"] == FRAME_COUNT * TICK
        and int(stream["nb_frames"]) == FRAME_COUNT,
        "Exact common HD container clock and geometry required",
    )
    if rendered:
        require(
            result["packets"] == [{"pts": pts, "dts": pts, "duration": TICK} for pts in PTS],
            "All rendered packets must have the exact native PTS, DTS and display duration",
        )
    return stream


def check_clock(table):
    require(
        table.time_base == TIME_BASE
        and list(table.pts) == PTS
        and not table.truncated
        and not table.interpolated
        and table.rotation == 0
        and (table.width, table.height) == (WIDTH, HEIGHT),
        "All original native PTS and display dimensions must match",
    )


def normalize(color):
    range_name, matrix = color
    require(
        range_name in {"pc", "tv"} and matrix in {"bt601", "bt709"}, "Explicit YUV range and matrix required"
    )
    return f"scale=in_range={range_name}:out_range=pc:in_color_matrix={matrix}:flags=bilinear+accurate_rnd+bitexact,format=rgb24"


def decoder_args(file):
    return ["-threads", "1", "-i", str(file)]


def clock_filter():
    return f"settb={TIME_BASE},setpts=N*{TICK},setsar=1"


def filter_graph(replacement_color=None, recompress=False):
    base = normalize(ORIGINAL_COLOR)
    out = clock_filter()
    if recompress:
        require(replacement_color is None, "Recompression input must be the unswapped source")
        return (
            f"[0:v]{base},{out},scale=in_range=pc:out_range=tv:out_color_matrix=bt601:"
            "flags=bilinear+accurate_rnd+bitexact,format=yuv420p[out]"
        )
    if replacement_color is None:
        return f"[0:v]{base},{out}[out]"
    return (
        f"[0:v]{base},setsar=1,split=2[before_source][after_source];"
        f"[before_source]trim=end_frame={START},setpts=PTS-STARTPTS[before];"
        f"[1:v]{normalize(replacement_color)},setsar=1,"
        f"trim=start_frame={START}:end_frame={STOP},setpts=PTS-STARTPTS[during];"
        f"[after_source]trim=start_frame={STOP},setpts=PTS-STARTPTS[after];"
        f"[before][during][after]concat=n=3:v=1:a=0,{out},format=rgb24[out]"
    )


def render(target, original, replacement=None, replacement_color=None, *, recompress=False):
    require(
        (replacement is None) == (replacement_color is None),
        "One matching replacement input and color arm required",
    )
    inputs = decoder_args(original)
    if replacement is not None:
        inputs += decoder_args(replacement)
    codec = (
        ["-c:v", "libx264", "-crf", "24", "-color_range", "tv", "-colorspace", "bt470bg"]
        if recompress
        else ["-c:v", "libx264rgb", "-qp", "0", "-color_range", "pc", "-colorspace", "rgb"]
    )
    subprocess.run(
        [
            tool("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-protocol_whitelist",
            "file",
            *inputs,
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            filter_graph(replacement_color, recompress),
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
            TIME_BASE,
            "-bsf:v",
            f"setts=duration={TICK}",
            "-video_track_timescale",
            TIME_BASE.split("/")[1],
            str(target),
        ],
        check=True,
        timeout=240,
    )


def frame_hashes(file, color):
    log = subprocess.check_output(
        [
            tool("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-copyts",
            *decoder_args(file),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-filter_threads",
            "1",
            "-vf",
            normalize(color),
            "-fps_mode",
            "passthrough",
            "-enc_time_base",
            TIME_BASE,
            "-threads",
            "1",
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "pipe:1",
        ],
        timeout=120,
        text=True,
    )
    lines = log.splitlines()
    rows = [
        [field.strip() for field in line.split(",")] for line in lines if line and not line.startswith("#")
    ]
    require(
        f"#tb 0: {TIME_BASE}" in lines
        and len(rows) == FRAME_COUNT
        and all(
            len(row) == 6
            and row[0] == "0"
            and int(row[4]) == WIDTH * HEIGHT * 3
            and re.fullmatch(r"[0-9a-f]{64}", row[5])
            for row in rows
        )
        and [int(row[2]) for row in rows] == PTS,
        "Independent normalized RGB decode must retain all exact PTS",
    )
    return [row[5] for row in rows]


def expected_case(original, replacement=None):
    require(
        len(original) == FRAME_COUNT and (replacement is None or len(replacement) == FRAME_COUNT),
        "Both normalized streams must have the complete native clock",
    )
    return (
        list(original)
        if replacement is None
        else [*original[:START], *replacement[START:STOP], *original[STOP:]]
    )


def changed_frames(base, actual, condition):
    indices = [index for index, pair in enumerate(zip(base, actual, strict=True)) if pair[0] != pair[1]]
    require(
        all(START <= index < STOP for index in indices) and bool(indices) == (condition != "unchanged"),
        "Every edit requires observed support confined to the frozen interval",
    )
    return indices


def picture_receipts(condition, pictures, hashes):
    reps = {}
    for label, index in REPRESENTATIVES.items():
        name = f"{condition}-{label}.png"
        file = MEDIA / name
        bgr = cv2.cvtColor(pictures[index], cv2.COLOR_RGB2BGR)
        success, png = cv2.imencode(".png", bgr)
        require(success, "Source-native PNG encoding failed")
        with file.open("xb") as out:
            out.write(png.tobytes())
        readback = cv2.imread(str(file), cv2.IMREAD_COLOR)
        require(
            readback is not None
            and hashlib.sha256(cv2.cvtColor(readback, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
            == hashes[index],
            "PNG readback must match its committed decoded RGB",
        )
        reps[label] = {
            "file": name,
            "sha256": sha(file),
            "frame_index": index,
            "pts": PTS[index],
            "time_base": TIME_BASE,
            "rgb_sha256": hashes[index],
        }
    return reps


def worker(output):
    verify()
    require(
        output == STAGE / "output"
        and (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and not (STAGE / "profile.json").exists()
        and ARCHIVE.resolve() == ARCHIVE
        and ARCHIVE.parent.parent.is_mount(),
        "The new media bank needs its own active local data guard and mounted archive",
    )
    paths = source_files()
    source_streams = {}
    for role in ("original", "swap"):
        source_streams[role] = stream_metadata(paths[role])
        check_clock(probe_frames(paths[role], timeout_s=120))
    require(
        source_streams["original"]["pix_fmt"] == "yuvj420p"
        and source_streams["original"]["color_range"] == "pc"
        and source_streams["original"]["color_space"] == "bt470bg"
        and source_streams["swap"]["pix_fmt"] == "yuv420p"
        and not source_streams["swap"].get("color_range")
        and not source_streams["swap"].get("color_space"),
        "Expected explicit source color and untagged publisher output",
    )
    base = frame_hashes(paths["original"], ORIGINAL_COLOR)
    MEDIA.mkdir(mode=0o700, exist_ok=False)
    output.mkdir(mode=0o700, exist_ok=False)
    compressed = MEDIA / "recompression_source_crf24.mp4"
    render(compressed, paths["original"], recompress=True)
    ordinary_stream = stream_metadata(compressed, rendered=True)
    require(
        ordinary_stream["pix_fmt"] == "yuv420p"
        and ordinary_stream["color_range"] == "tv"
        and ordinary_stream["color_space"] == "bt470bg",
        "Expected explicit limited-range Rec.601 intermediate",
    )
    check_clock(probe_frames(compressed, timeout_s=120))
    cases = []
    for condition, role in ROLES.items():
        color = SWAP_COLORS.get(condition)
        replacement = paths["swap"] if color is not None else None
        if condition == "recompression_only":
            color, replacement = RECOMPRESSION_COLOR, compressed
        alternative = frame_hashes(replacement, color) if replacement is not None else None
        expected = expected_case(base, alternative)
        actual = changed_frames(base, expected, condition)
        target = MEDIA / f"{condition}.mp4"
        render(target, paths["original"], replacement, color)
        stream = stream_metadata(target, rendered=True)
        require(
            stream["pix_fmt"] == "gbrp" and stream["color_range"] == "pc",
            "All final files must retain the common full-range RGB encoding",
        )
        table, pictures = verify_rgb(target, expected, Fraction(1, 24), REPRESENTATIVES.values())
        check_clock(table)
        cases.append(
            {
                "condition": condition,
                "role": role,
                "file": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha(target),
                "frame_table": table.to_json(),
                "rgb_sha256": expected,
                "actual_changed_frames": actual,
                "representatives": picture_receipts(condition, pictures, expected),
                "replacement_color_assumption": list(color) if color is not None else None,
                "stream": stream,
            }
        )
        print(
            json.dumps(
                {"condition": condition, "verified_frames": len(expected), "changed_frames": len(actual)}
            ),
            flush=True,
        )
    manifest = output / "private-manifest.json"
    write(
        manifest,
        {
            "schema": "chipmunk.video-faceswapai-matched-manifest/186",
            "scope": SCOPE,
            "edit_frame_range": [START, STOP],
            "edit_pts_range": [PTS[START], PTS[STOP]],
            "source_streams": source_streams,
            "cases": cases,
            "ordinary_intermediate": {
                "file": compressed.name,
                "sha256": sha(compressed),
                "bytes": compressed.stat().st_size,
                "stream": ordinary_stream,
            },
        },
    )
    write(
        output / "result.json",
        {
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_manifest_sha256": sha(manifest),
            "cases_verified": len(cases),
            "new_neural_calls": 0,
        },
    )


def run():
    def launch(argv, **kwargs):
        output = STAGE / "output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "bind", str(output), "0"],
            "Only the exact data-only worker may be launched",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(output)], **kwargs)

    require(not MEDIA.exists() and not MEDIA.is_symlink(), "Preserve any prior archive rendering")
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "ARCHIVE": ARCHIVE,
            "verify": verify,
            "prerequisites": lambda *_: True,
            "subprocess": SimpleNamespace(
                Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
            "__file__": __file__,
        },
    )("bind", 0)


def readout():
    verify()
    output = STAGE / "output"
    profile_file = STAGE / "profile.json"
    profile, result = read(profile_file), read(output / "result.json")
    manifest_file = output / "private-manifest.json"
    manifest = read(manifest_file)
    require(
        profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["child_started"]
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"]
        and profile["new_neural_calls"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and profile["worker_receipt_sha256"] == sha(output / "result.json")
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_manifest_sha256"] == sha(manifest_file)
        and result["cases_verified"] == len(ROLES)
        and result["new_neural_calls"] == 0
        and manifest["schema"] == "chipmunk.video-faceswapai-matched-manifest/186"
        and [case["condition"] for case in manifest["cases"]] == list(ROLES),
        "Complete bound data-only monitor and construction receipts required",
    )
    base = manifest["cases"][0]["rgb_sha256"]
    summaries = {}
    for case in manifest["cases"]:
        condition = case["condition"]
        require(
            case["file"] == f"{condition}.mp4" and case["role"] == ROLES[condition],
            "Exact planned file and label required",
        )
        target = MEDIA / case["file"]
        require(
            not target.is_symlink()
            and sha(target) == case["sha256"]
            and target.stat().st_size == case["bytes"]
            and case["frame_table"]["pts"] == PTS
            and case["frame_table"]["time_base"] == TIME_BASE
            and changed_frames(base, case["rgb_sha256"], condition) == case["actual_changed_frames"],
            "Saved matched construction changed",
        )
        for label, index in REPRESENTATIVES.items():
            row = case["representatives"][label]
            require(
                row["file"] == f"{condition}-{label}.png"
                and row["frame_index"] == index
                and row["pts"] == PTS[index]
                and row["time_base"] == TIME_BASE
                and row["rgb_sha256"] == case["rgb_sha256"][index],
                "Representative must retain its committed PTS and RGB",
            )
            path = MEDIA / row["file"]
            require(not path.is_symlink() and sha(path) == row["sha256"], "Representative PNG bytes changed")
        summaries[condition] = {
            "role": case["role"],
            "frames_verified": len(PTS),
            "changed_frames": len(case["actual_changed_frames"]),
            "bytes": case["bytes"],
        }
    intermediate = manifest["ordinary_intermediate"]
    intermediate_file = MEDIA / "recompression_source_crf24.mp4"
    require(
        intermediate["file"] == intermediate_file.name
        and not intermediate_file.is_symlink()
        and intermediate_file.stat().st_size == intermediate["bytes"]
        and sha(intermediate_file) == intermediate["sha256"],
        "Bound ordinary recompression intermediary changed",
    )
    return {
        "schema": "chipmunk.video-faceswapai-matched-readout/186",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "private_manifest_sha256": sha(manifest_file),
        "profile_sha256": sha(profile_file),
        "case_counts": summaries,
        "edit_frame_range": [START, STOP],
        "edit_pts_range": [PTS[START], PTS[STOP]],
        "time_base": TIME_BASE,
        "independent_source_components": 1,
        "new_neural_calls": 0,
        "scope": SCOPE,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("freeze", "run", "readout"):
        actions.add_parser(name)
    actions.add_parser("_worker").add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run()
    elif args.action == "_worker":
        worker(args.output)
    else:
        action = {"freeze": freeze, "readout": readout}[args.action]
        print(json.dumps(action(), allow_nan=False))


if __name__ == "__main__":
    main()
