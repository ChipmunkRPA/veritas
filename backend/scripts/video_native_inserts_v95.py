"""Native RGB/time-preserving source inserts and matched ordinary controls.

No detector or generator is loaded. Changed RGB support is not an AI-pixel mask.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import threading
from collections import defaultdict
from contextlib import suppress
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import FrameTable, decode_frames, probe_frames
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.video_native_sources_v92 import ARCHIVE, checked, probe_json
from scripts.video_native_sources_v92 import MEDIA as SOURCES
from scripts.video_native_sources_v92 import TRIAL as PARENT

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/video-native-inserts-v95"
MEDIA = ARCHIVE / "video-native-inserts-v95"
CONDITIONS = ("original", "swap_a", "swap_b", "ordinary_light", "ordinary_blur", "ordinary_time_step")
CODE = (
    "backend/scripts/video_native_inserts_v95.py",
    "backend/scripts/video_insert_runtime_v95.py",
    "backend/tests/test_video_native_inserts_v95.py",
    "docs/video-native-inserts-v95-method.md",
)


def clock(table):
    require(
        len(table) > 2 and not table.truncated and not table.interpolated,
        "Complete observed frame clock required",
    )
    times = tuple(p * table.tb for p in table.pts)
    steps = {b - a for a, b in zip(times, times[1:])}
    require(len(steps) == 1 and next(iter(steps)) > 0, "An exact constant observed clock is required")
    return times, next(iter(steps))


def recipe(tables):
    require(len(tables) == 3, "Original and two swap sources required")
    times, step = clock(tables[0])
    for table in tables[1:]:
        current, _ = clock(table)
        require(
            current == times and (table.width, table.height) == (tables[0].width, tables[0].height),
            "Paired native picture times/dimensions differ",
        )
    n = sum(t - times[0] < 6 for t in times)
    a, b = n // 3, 2 * n // 3
    require(n >= 9 and 0 < a < b < n - 2, "Before/during/after source support required")
    return {
        "frames": n,
        "source_origin_seconds": str(times[0]),
        "frame_step_seconds": str(step),
        "native_rate": str(1 / step),
        "edit_frames": [a, b],
        "edit_seconds": [str(a * step), str(b * step)],
        "representative_frames": {"before": a - 1, "during": (a + b - 1) // 2, "after": b},
    }


def transform(rgb, condition):
    if condition == "ordinary_light":
        return np.rint(rgb.astype(np.float32) * 1.08).clip(0, 255).astype(np.uint8)
    if condition == "ordinary_blur":
        return cv2.GaussianBlur(rgb, (0, 0), 2.0)
    return rgb


def encode_rgb(path, frames, width, height, step):
    """RGB lossless MP4 at the measured rational source rate; exclusive output."""
    require(not path.exists(), "Never overwrite a prior/partial rendering")
    step = Fraction(step)
    require(step > 0 and width > 0 and height > 0, "Positive observed clock and dimensions required")
    argv = [
        shutil.which("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(1 / step),
        "-i",
        "pipe:0",
        "-an",
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264rgb",
        "-qp",
        "0",
        "-preset",
        "veryfast",
        "-bf",
        "0",
        "-threads",
        "1",
        "-enc_time_base",
        f"1/{step.denominator}",
        "-video_track_timescale",
        str(step.denominator),
        str(path),
    ]
    proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    errors = []

    def drain():
        # Bound retained diagnostics; never block the encoder on a full error pipe.
        for line in iter(proc.stderr.readline, b""):
            errors.append(line[:1024])
            del errors[:-20]

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    timer = threading.Timer(360, proc.kill)
    timer.start()
    hashes = []
    try:
        for rgb in frames:
            require(
                rgb.dtype == np.uint8 and rgb.shape == (height, width, 3), "Exact native RGB picture required"
            )
            data = np.ascontiguousarray(rgb).tobytes()
            proc.stdin.write(data)
            hashes.append(hashlib.sha256(data).hexdigest())
        proc.stdin.close()
        require(proc.wait(timeout=30) == 0, "Lossless native RGB encoding failed")
        reader.join(timeout=5)
        require(not errors, "Encoder reported errors")
        return hashes
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if not proc.stdin.closed:
            with suppress(BrokenPipeError):
                proc.stdin.close()
        reader.join(timeout=5)
        proc.stderr.close()


def verify_rgb(path, expected, step, representatives=()):
    table = probe_frames(path, timeout_s=120)
    require(
        len(table) == len(expected) and not table.truncated and not table.interpolated,
        "Complete generated native frame table required",
    )
    require(
        all(p * table.tb == i * Fraction(step) for i, p in enumerate(table.pts)),
        "Generated native presentation times differ from the exact source-relative clock",
    )
    streams = probe_json(path, ["-select_streams", "v:0", "-show_entries", "stream=time_base,duration_ts"])[
        "streams"
    ]
    require(
        len(streams) == 1
        and int(streams[0]["duration_ts"]) * Fraction(streams[0]["time_base"])
        == len(expected) * Fraction(step),
        "Generated display duration differs from the constructed complete frame periods",
    )
    seen = []
    pictures = {}
    for frame in decode_frames(path, table, None, timeout_s=180):
        require(frame.index == len(seen), "Generated frame order differs")
        seen.append(hashlib.sha256(frame.rgb.tobytes()).hexdigest())
        if frame.index in representatives:
            pictures[frame.index] = frame.rgb.copy()
    require(seen == expected, "Lossless generated RGB does not match intended source pixels")
    return table, pictures


def source_frames(original, insert, construction, condition):
    n = construction["frames"]
    a, b = construction["edit_frames"]
    if condition in ("swap_a", "swap_b"):
        spans = [(original, range(a)), (insert, range(a, b)), (original, range(b, n))]
    else:
        spans = [(original, range(n))]
    for source, indices in spans:
        table = FrameTable.from_json(source["frame_table"])
        path = SOURCES / source["file"]
        require(sha(path) == source["input_sha256"], "Source bytes changed")
        wanted = list(indices)
        if condition == "ordinary_time_step":
            wanted = sorted({i + 2 if a <= i < b else i for i in indices})
        # Time-step controls can revisit a future source picture; retain only this
        # short two-frame lookahead, not a complete video in memory.
        iterator = iter(decode_frames(path, table, wanted, timeout_s=180))
        cache = {}
        try:
            for i in indices:
                target = i + 2 if condition == "ordinary_time_step" and a <= i < b else i
                while target not in cache:
                    frame = next(iterator)
                    digest = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
                    require(digest == source["rgb_sha256"][frame.index], "Recorded native source RGB changed")
                    cache[frame.index] = frame.rgb.copy()
                rgb = cache[target]
                if condition not in ("swap_a", "swap_b") and a <= i < b:
                    rgb = transform(rgb, condition)
                yield rgb
                # At the retime boundary, source b/b+1 are reused once before eviction.
                cache = {k: v for k, v in cache.items() if k >= i + 1}
        finally:
            iterator.close()


def identity():
    return {
        "schema": "chipmunk.native-video-insert-freeze/95",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "parent_freeze_sha256": sha(PARENT / "freeze.json"),
        "qualified_parent_plan_sha256": sha(PARENT / "qualified-plan.json"),
        "asset_plan_sha256": sha(PARENT.parent / "video-common-acquisition-v91/asset-plan.json"),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "opencv-python-headless", "psutil")},
        "binary_sha256": {p: sha(Path(shutil.which(p))) for p in ("ffmpeg", "ffprobe")},
        "conditions": list(CONDITIONS),
        "center_time_cap_seconds": 6,
        "new_neural_calls": 0,
        "scope": "Controlled source inserts, not a detector result or exact AI-pixel truth. All original native assets retained; incompatible groups stay explicit. No source-role change, fitting, serving change or customer input.",
    }


def prepare():
    checked()
    parent = read(PARENT / "qualified-plan.json")
    summary = read(PARENT / "summary.json")
    require(
        sha(PARENT / "qualified-plan.json") == summary["qualified_private_plan_sha256"],
        "Qualified source plan changed",
    )
    acquired = read(PARENT.parent / "video-common-acquisition-v91/asset-plan.json")
    names = {r["index"]: r["file"] for r in acquired["assets"]}
    grouped = defaultdict(list)
    for row in parent["assets"]:
        if row["source_family"] == "paired_face_swap":
            grouped[row["group"]].append({**row, "file": names[row["index"]]})
    groups = []
    excluded = []
    for group in sorted(
        grouped, key=lambda g: hashlib.sha256(("chipmunk.native-inserts/95\0" + g).encode()).digest()
    ):
        rows = grouped[group]
        by_kind = {r["kind"]: r for r in rows}
        require(
            set(by_kind) == {"original_videos", "inswapper", "uniface"} and len(rows) == 3,
            "Complete paired source group required",
        )
        require(
            not any(r["has_audio"] for r in rows)
            and len({r["qualification"]["role"] for r in rows}) == 1
            and len({r["qualification"]["component"] for r in rows}) == 1,
            "Exact shared silent source-component role required",
        )
        ordered = [by_kind[k] for k in ("original_videos", "inswapper", "uniface")]
        try:
            require(
                all(r["qualification"]["eligible"] for r in rows), "Parent source component is ineligible"
            )
            construction = recipe([FrameTable.from_json(r["frame_table"]) for r in ordered])
        except ValueError as error:
            excluded.append({"group": group, "reason": str(error)})
            continue
        groups.append(
            {
                "group": group,
                "role": rows[0]["qualification"]["role"],
                "component": rows[0]["qualification"]["component"],
                "sources": ordered,
                "construction": construction,
            }
        )
    require(
        len(groups) == 90 and len(excluded) == 6,
        "All known compatible/incompatible source groups must remain",
    )
    require(not MEDIA.exists(), "Preserve prior rendered media")
    MEDIA.mkdir()
    pilot = next(i for i, g in enumerate(groups) if g["role"] == "fusion_fit")
    body = {
        "schema": "chipmunk.native-video-insert-plan/95",
        "identity": identity(),
        "groups": groups,
        "excluded_groups": excluded,
        "planned_groups": 90,
        "planned_cases": 540,
        "pilot_group_index": pilot,
        "media_root": str(MEDIA),
        "detector_study_ready": False,
    }
    write(TRIAL / "plan.json", body)
    write(
        TRIAL / "freeze.json",
        {
            "schema": "chipmunk.native-video-insert-input-lock/95",
            "identity": identity(),
            "private_plan_sha256": sha(TRIAL / "plan.json"),
        },
    )
    print(
        json.dumps(
            {
                "planned_groups": 90,
                "planned_cases": 540,
                "incompatible_groups_retained": 6,
                "pilot_group_index": pilot,
                "new_neural_calls": 0,
            }
        ),
        flush=True,
    )


def build(index):
    checked()
    plan = read(TRIAL / "plan.json")
    frozen = read(TRIAL / "freeze.json")
    require(
        frozen["identity"] == identity() == plan["identity"]
        and frozen["private_plan_sha256"] == sha(TRIAL / "plan.json")
        and plan["media_root"] == str(MEDIA),
        "Frozen construction changed",
    )
    require(type(index) is int and 0 <= index < 90, "One explicit source-group rendering required")
    group = plan["groups"][index]
    construction = group["construction"]
    original, swap_a, swap_b = group["sources"]
    folder = MEDIA / f"group-{index:03d}"
    folder.mkdir(exist_ok=False)
    results = []
    a, b = construction["edit_frames"]
    n = construction["frames"]
    for condition in CONDITIONS:
        source = swap_a if condition == "swap_a" else swap_b if condition == "swap_b" else original
        target = folder / (condition + ".mp4")
        expected = encode_rgb(
            target,
            source_frames(original, source, construction, condition),
            original["frame_table"]["width"],
            original["frame_table"]["height"],
            construction["frame_step_seconds"],
        )
        require(len(expected) == n, "Complete planned native output required")
        actual = [i for i, h in enumerate(expected) if h != original["rgb_sha256"][i]]
        require(
            all(a <= i < b for i in actual) and (condition != "original" or not actual),
            "Change outside the declared edit interval",
        )
        table, pictures = verify_rgb(
            target,
            expected,
            construction["frame_step_seconds"],
            construction["representative_frames"].values(),
        )
        exhibits = {}
        for label, i in construction["representative_frames"].items():
            name = f"{condition}-{label}.png"
            path = folder / name
            require(
                cv2.imwrite(str(path), cv2.cvtColor(pictures[i], cv2.COLOR_RGB2BGR)),
                "Native representative capture failed",
            )
            exhibits[label] = {
                "frame_index": i,
                "pts": table.pts[i],
                "time_base": table.time_base,
                "file": name,
                "sha256": sha(path),
            }
        results.append(
            {
                "condition": condition,
                "label": int(condition in ("swap_a", "swap_b")),
                "file": target.name,
                "bytes": target.stat().st_size,
                "sha256": sha(target),
                "frame_table": table.to_json(),
                "encoded_duration_seconds": str(n * Fraction(construction["frame_step_seconds"])),
                "expected_rgb_sha256": expected,
                "actual_changed_frames": actual,
                "condition_changed_pixels": bool(actual),
                "construction": construction,
                "representatives": exhibits,
            }
        )
        print(
            json.dumps({"group_index": index, "condition_completed": condition, "verified_native_frames": n}),
            flush=True,
        )
    write(
        TRIAL / f"group-{index:03d}-complete.json",
        {
            "index": index,
            "group": group["group"],
            "role": group["role"],
            "freeze_sha256": sha(TRIAL / "freeze.json"),
            "cases": results,
            "new_neural_calls": 0,
            "scope": "Verified source construction/actual changed-frame support, not AI-pixel proof, model quality or outside human review.",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build"))
    parser.add_argument("index", nargs="?", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.index is None, "Preparation takes no group index")
        prepare()
    else:
        build(args.index)
