# SPDX-License-Identifier: Apache-2.0
"""Prospective unchanged-point H.264 comparison on source-disjoint GRID reserve clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts import review_video_reference_h264_earlier_transfer as replay
from scripts import video_reference_h264_earlier_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write

PLAN = study.previous.bank.DATA / "grid-transfer-plan-v143/bind-00/output/private-plan.json"
INITIAL = study.previous.bank.DATA / "video-reference-h264-earlier-transfer-2026-11-11"
SHAPE = (360, 288)
CLOCK = tuple(Fraction(i, 25) for i in range(75))
RANGE = (27, 47)
TILE = (136, 157, 224, 203)
ROLES = ("neutral_crf28", "neutral_crf33", "global_tone", "tandem_crf28", "tandem_crf33",
         "unrelated", "local_tone", "local_blur")
SPEC = {
    "schema": "chipmunk.video-reference-h264-grid-reserve/1",
    "cohort": "All 16 ready GRID reserve clips across eight known source components, separate from initial pilot.",
    "delivery": "Independent H.264 yuv420p MP4. Earlier CRF23 medium fixed GOP48. Delivery CRF28/33 veryfast.",
    "authorship": "Two ordinary authored edits in a fixed lower-central rectangle. No synthetic render or authorship labels.",
    "edit": {"frames": list(RANGE), "xyxy": list(TILE), "tone_plus": 46, "blur_sigma": 2.5, "global_tone_plus": 12},
    "point": study.local.POINT,
    "gate": {"tone_joint_min": 14, "tone_group_min": 6, "neutral_flags_max": 1,
             "tandem_flags_max": 0, "unrelated_flags_max": 0, "coverage_min": .80},
}


def inputs(plan, initial):
    data = read(plan)
    require(sha(plan) == read(plan.parent / "result.json")["private_plan_sha256"], "Original source plan differs")
    prior = read(initial / "aggregate.json")
    require(prior["panels"]["grid"]["advances"] and not prior["advances_both"], "Initial scope contrast is not preserved")
    rows = data["sources"]
    chosen = [row for row in rows if row["role"] == "reserve"]
    components = {row["component_index"] for row in chosen}
    require(len(chosen) == 16 and len(components) == 8
            and not (components & {row["component_index"] for row in rows if row["role"] == "pilot"}),
            "GRID reserve source grouping differs")
    for row in chosen:
        entry = row["input"]
        path = Path(entry["source_file"])
        native = entry["native_video_table"]
        require(row["plan_status"] == "ready" and entry["source_license"] == "CC-BY-4.0"
                and sha(path) == entry["source_sha256"], "Reserve source bytes or rights differ")
        require((native["width"], native["height"]) == SHAPE and not native["rotation"]
                and not native["truncated"] and not native["interpolated"]
                and tuple((p - native["pts"][0]) * Fraction(native["time_base"]) for p in native["pts"]) == CLOCK,
                "GRID native frame shape or clock differs")
    return chosen


def binding(plan, initial):
    return {"plan": sha(plan), "completed_plan": sha(plan.parent / "result.json"),
            "initial": sha(initial / "aggregate.json"), "initial_audit": sha(initial / "independent-audit.json"),
            "implementation": sha(Path(__file__)), "method": study.bindings(study.PRIOR)}


def donor_for(row, chosen):
    index = chosen.index(row)
    return next(x for x in [*chosen[index+1:], *chosen[:index]]
                if x["component_index"] != row["component_index"])


def freeze(run, plan, initial):
    chosen = inputs(plan, initial)
    roster = [{"ordinal": row["ordinal"], "component": row["component_index"],
               "input_sha256": row["input"]["source_sha256"], "unrelated_ordinal": donor_for(row, chosen)["ordinal"]}
              for row in chosen]
    run.mkdir(mode=0o700, parents=True)
    write(run / "freeze.json", {"spec": SPEC, "binding": binding(plan, initial), "roster": roster})


def pictures(row):
    records = row["input"]["pictures"]
    native = row["input"]["native_video_table"]
    require(len(records) == len(CLOCK), "Reserve native pictures are incomplete")
    output = []
    for i, record in enumerate(records):
        path = Path(record["file"])
        require(record["frame_index"] == i and record["pts"] == native["pts"][i]
                and sha(path) == record["file_sha256"], "Native reserve picture differs")
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
        require(hashlib.sha256(pixels.tobytes()).hexdigest() == record["rgb_sha256"], "Decoded reserve pixels differ")
        output.append(pixels)
    return np.stack(output)


def author(frames, role):
    if role == "global_tone":
        return np.clip(frames.astype(np.int16) + SPEC["edit"]["global_tone_plus"], 0, 255).astype(np.uint8)
    if role not in {"local_tone", "local_blur"}:
        return frames
    edited = frames.copy()
    x0, y0, x1, y1 = TILE
    first, end = RANGE
    target = edited[first:end, y0:y1, x0:x1]
    if role == "local_tone":
        target[:] = np.clip(target.astype(np.int16) + SPEC["edit"]["tone_plus"], 0, 255)
    else:
        for i in range(first, end):
            blurred = cv2.GaussianBlur(frames[i], (0, 0), SPEC["edit"]["blur_sigma"])
            edited[i, y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
    return edited


def encode(frames, destination, crf, *, reference=False):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-filter_threads", "1", "-threads", "1", "-f", "rawvideo",
            "-pixel_format", "rgb24", "-video_size", "360x288", "-framerate", "25", "-i", "pipe:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1", "-crf", str(crf), "-video_track_timescale", "1000"]
    args += ["-preset", "medium", "-g", "48", "-keyint_min", "48", "-sc_threshold", "0"] if reference else ["-preset", "veryfast"]
    subprocess.run([*args, str(destination)], input=frames.tobytes(), check=True, timeout=25)


def expected_clock(table):
    require((table.width, table.height) == SHAPE and replay.clocks(table) == list(CLOCK)
            and not (table.rotation or table.truncated or table.interpolated), "Delivery native picture clock differs")


def spatial(native, edited, best):
    index = best["samples"][1]
    x, y, w, h = best["tile_xywh"]
    mask = np.any(native[index] != edited[index], axis=-1)
    return [int(mask[y:y+h, x:x+w].sum()), int(mask.sum()), w*h]


def run_study(run, chosen):
    folder = run / "deliveries"
    folder.mkdir(mode=0o700)
    start = time.monotonic()
    with (run / "held.jsonl").open("x") as stream:
        for row in chosen:
            idx = row["ordinal"]
            source = pictures(row)
            ref = folder / f"{idx:02d}-earlier.mp4"
            encode(source, ref, 23, reference=True)
            earlier = study.table(ref)
            expected_clock(earlier)
            earlier_frames = list(study.previous.verified_frames(ref, earlier))
            for role in ROLES:
                path = folder / f"{idx:02d}-{role}.mp4"
                crf = 33 if role.endswith("33") else 28
                if role.startswith("tandem"):
                    study.encode_h264(ref, path, crf)
                    frames = None
                else:
                    frames = pictures(donor_for(row, chosen)) if role == "unrelated" else author(source, role)
                    encode(frames, path, crf)
                later = study.table(path)
                expected_clock(later)
                record = study.observe(earlier_frames, path, earlier, later)
                truth = list(RANGE) if role.startswith("local_") else None
                if truth:
                    changed = np.flatnonzero(np.any(frames != source, axis=(1, 2, 3))).tolist()
                    require(changed == list(range(*RANGE)), "Authored edit differs from the exact interval")
                masks = spatial(source, frames, record["best"]) if truth and record["best"] else None
                stream.write(json.dumps({"ordinal": idx, "component": row["component_index"], "condition": role,
                    "known_edit": truth, "spatial": masks, "earlier_sha256": sha(ref), "submitted_sha256": sha(path),
                    **record}, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print("source", idx, len(ROLES), "exact cases", flush=True)
    write(run / "runtime.json", {"elapsed_s": time.monotonic() - start,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          "freeze_sha256": sha(run / "freeze.json"), "held_sha256": sha(run / "held.jsonl")})


def report(run, chosen):
    runtime = read(run / "runtime.json")
    require(runtime["freeze_sha256"] == sha(run / "freeze.json") and runtime["held_sha256"] == sha(run / "held.jsonl"),
            "Held run or freeze commitment differs")
    cases = study.previous.rows(run, "held")
    require([(r["ordinal"], r["condition"]) for r in cases] == [(row["ordinal"], role) for row in chosen for role in ROLES],
            "Incomplete or duplicate case roster")
    counts = {role: Counter() for role in ROLES}
    useful_groups = set()
    for case in cases:
        found = replay.approved(case, CLOCK)
        counts[case["condition"]].update({"views": 1, **found})
        if case["condition"] == "local_tone" and found["joint_correct"]:
            useful_groups.add(case["component"])
    gate = SPEC["gate"]
    criteria = {
        "tone_locations": counts["local_tone"]["joint_correct"] >= gate["tone_joint_min"],
        "tone_groups": len(useful_groups) >= gate["tone_group_min"],
        "coverage": all(counts[role]["assessed"] >= gate["coverage_min"] * counts[role]["views"]
                        for role in ROLES if role != "unrelated"),
        "neutral": sum(counts[r]["flag"] for r in ROLES[:3]) <= gate["neutral_flags_max"],
        "tandem": sum(counts[r]["flag"] for r in ROLES[3:5]) <= gate["tandem_flags_max"],
        "unrelated": counts["unrelated"]["flag"] <= gate["unrelated_flags_max"],
    }
    write(run / "aggregate.json", {"spec": SPEC, "runtime": runtime, "counts": counts, "useful_tone_groups": len(useful_groups),
                                    "criteria": criteria, "advances": all(criteria.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "held", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        freeze(args.run, args.plan, args.initial)
        return
    saved = read(args.run / "freeze.json")
    require(saved["spec"] == SPEC and saved["binding"] == binding(args.plan, args.initial), "Reserve study freeze differs")
    chosen = inputs(args.plan, args.initial)
    {"held": run_study, "report": report}[args.phase](args.run, chosen)


if __name__ == "__main__":
    main()
