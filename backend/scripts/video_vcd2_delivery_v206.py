"""Prepare one fixed upload delivery encode for each v200 webcam control."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from contextlib import ExitStack, closing
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.detectors.video.decode import FrameTable, decode_frames
from scripts import video_vcd2_controls_v200 as controls

BASE = controls.BASE
TRIAL = BASE.ARCHIVE / "video-vcd2-delivery-v206"
PUBLIC_V200 = BASE.ROOT / "docs/evaluation-panels/video-vcd2-controls-freeze-v200.json"
ROLES = controls.ROLES
SLOTS = range(1, 13)
CODE = (
    "backend/scripts/video_vcd2_delivery_v206.py",
    "backend/tests/test_video_vcd2_delivery_v206.py",
    "docs/video-vcd2-delivery-v206-method.md",
)
SHA = re.compile(r"[a-f0-9]{64}")
MAX_UPLOAD_BYTES = 500 * 1024**2
QP = 18
X264 = "open-gop=0:scenecut=0:rc-lookahead=0:sync-lookahead=0:mbtree=0:aq-mode=0:ipratio=1"
require, read, sha, write = controls.require, controls.read, controls.sha, controls.write


def direct_file(path: Path) -> Path:
    require(path.is_file() and path.resolve() == path, "Expected an owned regular file")
    return path


def identity() -> tuple[dict, list[dict]]:
    private = direct_file(controls.TRIAL / "freeze.json")
    public = read(PUBLIC_V200)
    frozen = read(private)
    previous = controls.build_identity()
    slots = frozen["slots"]
    require(
        frozen["identity"] == previous
        and all(public.get(key) == value for key, value in previous.items())
        and sha(private) == public["private_freeze_sha256"]
        and [row["slot"] for row in slots] == list(SLOTS),
        "Exact published v200 construction freeze and decoder required",
    )
    return {
        "schema": "chipmunk.video-vcd2-delivery/206",
        "v200_freeze_sha256": public["private_freeze_sha256"],
        "v200_publication_sha256": sha(PUBLIC_V200),
        "code_sha256": {name: sha(BASE.ROOT / name) for name in CODE},
        "tool_sha256": previous["tool_sha256"],
        "roles": list(ROLES),
        "relative_interval_s": list(BASE.WINDOW),
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "encode": {
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "range": "tv",
            "matrix": "bt709",
            "qp": QP,
            "preset": "veryfast",
            "gop_size": 1,
            "b_frames": 0,
            "threads": 1,
            "x264_params": X264,
        },
        "model_calls": 0,
    }, slots


def diff_indices(left: list[str], right: list[str]) -> list[int]:
    return [i for i, (a, b) in enumerate(zip(left, right, strict=True)) if a != b]


def receipt_plan(entry: dict, receipt: dict) -> controls.Recipe:
    pts = receipt["relative_pts"]
    count = receipt["frame_count"]
    width, height = receipt["width"], receipt["height"]
    origin, duration = receipt["source_origin_pts"], receipt["last_duration"]
    start, stop = receipt["frame_interval"]
    tb = Fraction(receipt["time_base"])
    sar = Fraction(receipt["source_color"]["sample_aspect_ratio"])
    require(
        receipt["slot"] == entry["slot"]
        and receipt["source_sha256"] == entry["selected"]["source_sha256"]
        and receipt["model_calls"] == 0
        and type(count) is int
        and 3 <= count <= 1200
        and len(pts) == count
        and all(type(value) is int for value in pts)
        and pts[0] == 0
        and all(a < b for a, b in zip(pts, pts[1:]))
        and tb.numerator == 1
        and 0 < tb.denominator <= 2_147_483_647
        and all(type(value) is int and value > 0 and value % 2 == 0 for value in (width, height))
        and entry["selected"]["rgb_picture_bytes"] == width * height * count * 3
        and type(origin) is int
        and type(duration) is int
        and duration > 0
        and sar > 0
        and type(start) is type(stop) is int
        and 0 < start < stop < count
        and [i for i, value in enumerate(pts) if BASE.WINDOW[0] <= value * tb < BASE.WINDOW[1]]
        == list(range(start, stop))
        and set(receipt["cases"]) == set(ROLES),
        "Exact v200 receipt clock, dimensions, and ordinary-edit interval required",
    )
    cases = receipt["cases"]
    baseline = cases["unchanged"]["rgb_sha256"]
    for role in ROLES:
        case = cases[role]
        hashes, changes = case["rgb_sha256"], case["changed_frames"]
        require(
            case["file"] == f"{role}.mp4"
            and SHA.fullmatch(case["sha256"])
            and len(hashes) == case["verified_frames"] == count
            and all(SHA.fullmatch(digest) for digest in hashes)
            and [row["index"] for row in changes] == diff_indices(baseline, hashes)
            and all(
                type(row["index"]) is int
                and start <= row["index"] < stop
                and row["pts"] == pts[row["index"]]
                and 0 < row["rgb_mae"] <= 255
                for row in changes
            ),
            "Exact v200 RGB commitments and paired ordinary changes required",
        )
    lossy = diff_indices(cases[ROLES[1]]["rgb_sha256"], cases[ROLES[2]]["rgb_sha256"])
    require(
        lossy and lossy == receipt["lossy_codec_changed_indices"],
        "Exact v200 compression contrast required",
    )
    return controls.Recipe(
        width, height, tuple(pts), origin, tb, duration, (start, stop), "pc", "rgb", False, False, sar, False
    )


def input_selection(entry: dict, folder: Path) -> dict:
    path = direct_file(folder / "qualification.json")
    receipt = read(path)
    receipt_plan(entry, receipt)
    cases = {}
    for role in ROLES:
        case = receipt["cases"][role]
        size = direct_file(folder / case["file"]).stat().st_size
        require(size > 0, "Empty v200 source file")
        cases[role] = {"sha256": case["sha256"], "bytes": size}
    return {"attempt": folder.name, "qualification_sha256": sha(path), "cases": cases}


def inventory(entries: list[dict], unavailable: set[int]) -> list[dict]:
    require(
        unavailable <= {row["slot"] for row in entries if row["status"] == "qualified"},
        "Unavailable override is for unprepared qualified slots",
    )
    attempts = {row["slot"]: [] for row in entries}
    for folder in sorted(controls.TRIAL.iterdir()):
        match = controls.ATTEMPT_NAME.fullmatch(folder.name)
        if match:
            slot = int(match[1])
            require(
                slot in attempts and folder.is_dir() and folder.resolve() == folder,
                "Unexpected v200 attempt directory",
            )
            attempts[slot].append(folder)

    result = []
    for entry in entries:
        slot = entry["slot"]
        history, completed = [], []
        for folder in attempts[slot]:
            receipt = folder / "qualification.json"
            ready = receipt.exists() or receipt.is_symlink()
            history.append({"attempt": folder.name, "status": "prepared" if ready else "incomplete"})
            if ready:
                completed.append(input_selection(entry, folder))
        require(not completed or slot not in unavailable, "A completed input cannot be closed as unavailable")
        if entry["status"] == "unavailable":
            require(not completed, "Unavailable source has unexpected prepared controls")
            status = "source_unavailable"
        elif completed:
            require(
                all(row["cases"] == completed[0]["cases"] for row in completed),
                "Completed v200 retry changed the input media",
            )
            status = "prepared"
        else:
            require(entry["status"] == "qualified", "Unknown v200 slot status")
            status = "preparation_unavailable" if slot in unavailable else "preparation_pending"
        result.append(
            {
                "slot": slot,
                "status": status,
                "attempts": history,
                "selected": completed[0] if completed else None,
            }
        )
    return result


def encode(source: Path, target: Path, plan: controls.Recipe) -> None:
    convert = (
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:"
        "flags=bilinear+accurate_rnd+bitexact,format=yuv420p"
    )
    controls.ffmpeg(
        *controls.input_args(source),
        "-filter_threads",
        "1",
        "-map",
        "0:v:0",
        "-vf",
        f"{convert},{controls.clock_filter(plan)}",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-qp",
        str(QP),
        "-g",
        "1",
        "-bf",
        "0",
        "-threads",
        "1",
        "-x264-params",
        X264,
        "-enc_time_base",
        str(plan.tb),
        "-bsf:v",
        f"setts=duration={plan.last_duration}",
        "-video_track_timescale",
        str(plan.tb.denominator),
        "-use_editlist",
        "0",
        "-movflags",
        "+faststart",
        str(target),
        timeout=600,
    )


def rgb_change(before, after, index: int, pts: int) -> dict:
    delta = np.subtract(before, after, dtype=np.int16)
    np.abs(delta, out=delta)
    return {"index": index, "pts": pts, "rgb_mae": float(delta.mean())}


def verify_pictures(
    source: Path,
    target: Path,
    plan: controls.Recipe,
    raw: dict,
    baseline: tuple[Path, FrameTable, list[str]] | None = None,
) -> tuple[dict, FrameTable]:
    raw_table = controls.verify_clock(source, plan)
    table = controls.verify_clock(target, plan, edited=True)
    require(
        table.key_frames == tuple(range(len(plan.pts))) and table.pict_types == "I" * len(plan.pts),
        "Delivery output contains non-intra pictures",
    )
    require(
        (table.width, table.height, table.coded_width, table.coded_height)
        == (plan.width, plan.height, plan.width, plan.height),
        "Delivery picture geometry differs from the v200 source",
    )
    hashes, delivery_changes, pair_changes = [], [], []
    with ExitStack() as stack:
        # Decode in the production RGB path. The baseline iterator is a third
        # bounded stream only for the edited variants.
        native = stack.enter_context(closing(decode_frames(source, raw_table, None, timeout_s=600)))
        delivered = stack.enter_context(closing(decode_frames(target, table, None, timeout_s=600)))
        reference = None
        if baseline is not None:
            reference = stack.enter_context(
                closing(decode_frames(baseline[0], baseline[1], None, timeout_s=600))
            )
        for index, (before, after) in enumerate(zip(native, delivered, strict=True)):
            pts = plan.pts[index]
            before_hash = hashlib.sha256(before.rgb.tobytes()).hexdigest()
            after_hash = hashlib.sha256(after.rgb.tobytes()).hexdigest()
            require(
                before.index == after.index == index
                and before.pts == after.pts == pts
                and before_hash == raw["rgb_sha256"][index],
                "Delivery input no longer matches the v200 picture receipt",
            )
            hashes.append(after_hash)
            if before_hash != after_hash:
                delivery_changes.append(rgb_change(before.rgb, after.rgb, index, pts))
            if reference is not None and baseline is not None:
                other = next(reference, None)
                require(
                    other is not None and other.index == index and other.pts == pts,
                    "Delivered roles have different picture clocks",
                )
                if after_hash != baseline[2][index]:
                    require(
                        plan.interval[0] <= index < plan.interval[1],
                        "Delivered roles differ outside the ordinary-edit interval",
                    )
                    pair_changes.append(rgb_change(other.rgb, after.rgb, index, pts))
        require(reference is None or next(reference, None) is None, "Delivered baseline has extra pictures")
    require(len(hashes) == len(plan.pts), "Delivery decoder omitted pictures")
    old_changes = {row["index"] for row in raw["changed_frames"]}
    new_changes = {row["index"] for row in pair_changes}
    require(new_changes <= old_changes, "Same input pictures produced different delivery output")
    return {
        "rgb_sha256": hashes,
        "verified_frames": len(hashes),
        "changed_frames": pair_changes,
        "delivery_encode_changed_frames": delivery_changes,
        "suppressed_pair_change_indices": sorted(old_changes - new_changes),
    }, table


def construct(source_folder: Path, output: Path, receipt: dict, plan: controls.Recipe) -> dict:
    cases = {}
    baseline = None
    for role in ROLES:
        raw = receipt["cases"][role]
        source, target = direct_file(source_folder / raw["file"]), output / raw["file"]
        require(sha(source) == raw["sha256"], "v200 input bytes differ from their receipt")
        encode(source, target, plan)
        measured, table = verify_pictures(source, target, plan, raw, baseline)
        require(sha(source) == raw["sha256"], "v200 input bytes changed during preparation")
        if role == "unchanged":
            baseline = (target, table, measured["rgb_sha256"])
        size = target.stat().st_size
        cases[role] = {
            "file": target.name,
            "sha256": sha(target),
            "bytes": size,
            "input_sha256": raw["sha256"],
            "status": "prepared" if size <= MAX_UPLOAD_BYTES else "unsupported_upload_size",
            **measured,
        }
    lossy = diff_indices(cases[ROLES[1]]["rgb_sha256"], cases[ROLES[2]]["rgb_sha256"])
    v200_lossy = set(receipt["lossy_codec_changed_indices"])
    require(set(lossy) <= v200_lossy, "Fixed delivery encode introduced a compression-control difference")
    return {
        "cases": cases,
        "lossy_codec_changed_indices": lossy,
        "suppressed_lossy_codec_change_indices": sorted(v200_lossy.difference(lossy)),
    }


def prepare(frozen: dict, entry: dict, slot: int, attempt: int) -> None:
    require(slot in SLOTS and 1 <= attempt <= 9, "Invalid slot or attempt")
    selected = frozen["slots"][slot - 1]["selected"]
    require(selected is not None, "Frozen slot has no prepared v200 inputs")
    source_folder = controls.TRIAL / selected["attempt"]
    require(input_selection(entry, source_folder) == selected, "v200 input receipt changed after freeze")
    receipt = read(source_folder / "qualification.json")
    plan = receipt_plan(entry, receipt)
    reserve = BASE.MIN_ARCHIVE_FREE + len(ROLES) * entry["selected"]["rgb_picture_bytes"]
    require(shutil.disk_usage(TRIAL).free >= reserve, "Insufficient Seagate space for this recording")
    output = TRIAL / f"slot-{slot:02d}-attempt-{attempt:02d}"
    BASE.private_directory(output)
    measurements = construct(source_folder, output, receipt, plan)
    require(
        sha(source_folder / "qualification.json") == selected["qualification_sha256"],
        "v200 input receipt changed during preparation",
    )
    record = {
        key: receipt[key]
        for key in (
            "slot",
            "source_sha256",
            "frame_count",
            "width",
            "height",
            "source_origin_pts",
            "time_base",
            "relative_pts",
            "last_duration",
            "frame_interval",
            "source_color",
            "model_calls",
        )
    }
    record.update(
        schema="chipmunk.video-vcd2-delivery-qualification/206",
        delivery_attempt=attempt,
        freeze_sha256=sha(TRIAL / "freeze.json"),
        v200_qualification_sha256=selected["qualification_sha256"],
        **measurements,
    )
    write(output / "qualification.json", record)
    print(
        json.dumps(
            {
                "slot": slot,
                "cases": dict(Counter(row["status"] for row in record["cases"].values())),
                "model_calls": 0,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "prepare"))
    parser.add_argument("--unavailable", type=int, nargs="*", default=[])
    parser.add_argument("--slot", type=int, choices=SLOTS)
    parser.add_argument("--attempt", type=int, choices=range(1, 10), default=1)
    args = parser.parse_args()
    if (args.command == "prepare") != (args.slot is not None):
        parser.error("--slot is required only for prepare")
    if args.command == "prepare" and args.unavailable:
        parser.error("unavailability is set before freeze")
    current, entries = identity()
    if args.command == "prepare":
        BASE.archive_ready()
        frozen = read(TRIAL / "freeze.json")
        require(
            frozen["identity"] == current and [row["slot"] for row in frozen["slots"]] == list(SLOTS),
            "Exact frozen v206 method and slot roster required",
        )
        prepare(frozen, entries[args.slot - 1], args.slot, args.attempt)
        return
    slots = inventory(entries, set(args.unavailable))
    if args.command == "freeze":
        require(
            all(row["status"] != "preparation_pending" for row in slots),
            "Each v200 preparation must finish or be explicitly closed before freeze",
        )
        BASE.archive_ready()
        BASE.private_directory(TRIAL)
        write(TRIAL / "freeze.json", {"identity": current, "slots": slots})
    print(
        json.dumps(
            {
                "status": args.command,
                "source_outcomes": dict(Counter(row["status"] for row in slots)),
                "storage_projection": controls.storage_projection(entries),
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
