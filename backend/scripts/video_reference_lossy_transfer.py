# SPDX-License-Identifier: Apache-2.0
"""Measure bounded local changes between matched-clock earlier and delivered pictures."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from contextlib import closing
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference
from scripts import video_mouth_jaw_transfer as bank
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "backend/data/independent-evaluation/video-mouth-jaw-transfer-2026-11-11/freeze.json"
MAX_PIXELS = 640 * 480
MAX_FRAMES = 100
FLOOR = .8
SPEC = {
    "schema": "chipmunk.video-reference-lossy-transfer/1",
    "tile": "Native luma, Gaussian sigma 1.2, tile width 10% of short side and height 7%, half-overlap.",
    "comparison": "Whole-frame robust luma adjustment, tile error above the frame's 65th percentile; compare core to both flanks.",
    "admission": "At least 90% of picture pairs have luma support; the 25th and 65th percentiles of per-frame background residual are at most 2.5 and 3.0.",
    "clock": "Exact equal native relative PTS; at most 100 pictures, 640x480 pixels, continuous ~25 or 30 fps.",
    "baseline": "Current exact picture ranges (same byte-difference semantics).",
    "windows_ms": [480, 640, 800, 960],
    "flank_ms": 240,
    "min_temporal_iou": "1/2",
    "min_tile_changed_fraction": "1/8",
    "min_edit_pixels_covered": "1/12",
    "delivery": "H.264 libx264 veryfast, CRF 28 common deliveries; originals also CRF 33 and ordinary global tone at CRF 28.",
    "calibration": "Larger of 0.8 and maximum neutral/different-source development score plus 0.15. Strict exceedance.",
    "gate": "Both banks assess at least 80% of neural and neutral views, add useful locations on at least 3 source groups, "
            "add at most one neutral flag each and no different-source flags.",
}

def binding():
    return {
        "runner": sha(Path(__file__)),
        "source_plan": sha(PRIOR),
        "picture_compare": sha(Path(video_reference.__file__)),
        "decode": sha(ROOT / "backend/app/detectors/video/decode.py"),
        "versions": [cv2.__version__, np.__version__,
                     subprocess.check_output(["ffmpeg", "-version"], text=True, timeout=3).splitlines()[0]],
    }


def freeze(run):
    prior = read(PRIOR)
    archive = Path(prior["archive"])
    for split in ("development", "held"):
        require(len(prior["grid"][split]) == 8, "Need eight known GRID source recordings per role")
        for pin in prior["grid"][split]:
            bank.prior.inspect(prior, pin)
    native = bank.native_groups(archive)
    require(len(native) == 10, "Need the ten native CREMA-D transfer groups")
    run.mkdir(mode=0o700, parents=True)
    write(run / "freeze.json", {
        "spec": SPEC, "binding": binding(), "grid": prior["grid"], "archive": prior["archive"],
        "native": [{"index": n, "digests": [row["file_sha256"] for row in rows]} for n, rows in native.items()],
    })


def sealed(run):
    plan = read(run / "freeze.json")
    require(plan["spec"] == SPEC and plan["binding"] == binding(), "Frozen method or source roster differs")
    return plan


def earlier_table(path, row):
    require(sha(path) == row["file_sha256"], "Authored source file differs")
    table = probe_frames(path, cap=MAX_FRAMES, timeout_s=10, max_pixels=MAX_PIXELS)
    require(not (table.truncated or table.interpolated or table.rotation)
            and [p * table.tb for p in table.pts] == [Fraction(p, 1000) for p in row["expected"]["picture_pts_ms"]],
            "Authored native clock differs")
    return table


def encode(src, dest, crf=28, tone=False):
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-filter_threads", "1", "-threads", "1", "-copyts",
           "-i", str(src), "-map", "0:v:0", "-an"]
    if tone:
        cmd += ["-vf", "eq=brightness=0.055:contrast=0.92"]
    cmd += ["-fps_mode", "passthrough", "-enc_time_base", "1:1000", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-threads", "1", "-crf", str(crf), str(dest)]
    subprocess.run(cmd, check=True, timeout=25)


def _rectangles(height, width):
    side = min(height, width)
    w, h = round(.10 * side), round(.07 * side)
    ys, xs = np.meshgrid(np.arange(0, height - h + 1, max(1, h // 2)),
                         np.arange(0, width - w + 1, max(1, w // 2)), indexing="ij")
    return xs.ravel(), ys.ravel(), w, h


def _measure(first, second, rects):
    a, b = (cv2.GaussianBlur(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY).astype(np.float32), (7, 7), 1.2)
            for pixels in (first, second))
    a_small, b_small = [cv2.resize(value, (32, 32), interpolation=cv2.INTER_AREA) for value in (a, b)]
    a_q = np.percentile(a_small, (20, 50, 80))
    b_q = np.percentile(b_small, (20, 50, 80))
    if min(a_q[2] - a_q[0], b_q[2] - b_q[0]) < 10:
        return None, None
    scale = float(np.clip((a_q[2] - a_q[0]) / (b_q[2] - b_q[0]), .75, 1.35))
    residual = cv2.absdiff(a, np.clip((b - b_q[1]) * scale + a_q[1], 0, 255).astype(np.float32))
    xs, ys, w, h = rects
    total = cv2.integral(residual, sdepth=cv2.CV_64F)
    mean = (total[ys+h, xs+w] - total[ys, xs+w] - total[ys+h, xs] + total[ys, xs]) / (w * h)
    background = float(np.percentile(mean, 65))
    return np.maximum(0, mean - background), background


def _seconds(table):
    return [(p - table.pts[0]) * table.tb for p in table.pts]


def native_cadence(table):
    clock = _seconds(table)
    steps = [b - a for a, b in pairwise(clock)]
    if not steps:
        return None
    dt = float(np.median([float(step) for step in steps]))
    return dt if 1 / 32 <= dt <= 1 / 24 and all(abs(float(step) - dt) <= .002 for step in steps) else None


def _best(signals, dt, rects):
    xs, ys, w, h = rects
    flank = round(.24 / dt)
    best = None
    for width in sorted({round(ms / 1000 / dt) for ms in SPEC["windows_ms"]}):
        for start in range(flank, len(signals) - width - flank + 1):
            end = start + width
            if end >= len(signals):
                break
            block = signals[start-flank:end+flank]
            if not np.isfinite(block).all():
                continue
            before = np.median(block[:flank], axis=0)
            during = np.median(block[flank:flank+width], axis=0)
            after = np.median(block[flank+width:], axis=0)
            responses = during - np.maximum(before, after)
            index = int(np.argmax(responses))
            score = float(responses[index])
            if best is None or score > best["score"]:
                # Report the representative peak of the detected core in the selected native tile.
                peak = start + int(np.argmax(signals[start:end, index]))
                best = {"score": score, "frames": [start, end], "tile_xywh": [int(xs[index]), int(ys[index]), w, h],
                        "samples": [start - 1, peak, end], "tile_response": [float(before[index]), float(during[index]),
                                                                             float(after[index])]}
    return best


def verified_frames(path, table):
    with closing(decode_frames(path, table, None, timeout_s=40, max_pixels=MAX_PIXELS)) as decoded:
        count = 0
        for frame in decoded:
            require(frame.index == count and frame.pts_verified and frame.pts == table.pts[count],
                    "Missing or unverified decoded picture")
            yield frame.rgb
            count += 1
        require(count == len(table), "Incomplete native picture stream")


def observe(earlier_frames, submitted, early, later):
    issue = video_reference._comparison_issue((early, later))
    dt = native_cadence(early)
    if issue or not dt or later.rotation:
        return {"status": issue or "unsupported_cadence", "best": None, "exact": []}
    rects = _rectangles(early.height, early.width)
    signals, global_errors, exact = [], [], []
    with closing(verified_frames(submitted, later)) as delivered:
        for first, second in zip(earlier_frames, delivered, strict=True):
            tiles, error = _measure(first, second, rects)
            signals.append(np.full(len(rects[0]), np.nan) if tiles is None else tiles)
            global_errors.append(error)
            exact.append(not np.array_equal(first, second))
    require(len(signals) == len(early), "Incomplete matched picture comparison")
    base = {"exact": [[int(a), int(b)] for a, b in video_reference.marked_ranges(np.array(exact))], "best": None}
    usable = [value for value in global_errors if value is not None]
    if len(usable) < .90 * len(signals):
        return {**base, "status": "unverified_photometric_support"}
    if float(np.percentile(usable, 65)) > 3 or float(np.percentile(usable, 25)) > 2.5:
        return {**base, "status": "insufficient_shared_picture_content"}
    best = _best(np.asarray(signals), dt, rects)
    if best is not None:
        clock = _seconds(later)
        best["bounds_s"] = [str(clock[i]) for i in best["frames"]]
        best["sample_s"] = [str(clock[i]) for i in best["samples"]]
    return {**base, "status": "measured" if best is not None else "no_complete_flanks", "best": best}


def authored_spatial(earlier_frames, source, authored, expected, choice):
    if choice is None:
        return None
    index = choice["samples"][1]
    x, y, w, h = choice["tile_xywh"]
    [selected] = list(decode_frames(source, authored, [index], timeout_s=10, max_pixels=MAX_PIXELS))
    require(selected.pts_verified and selected.pts == authored.pts[index]
            and bank.raw_digest(selected.rgb) == expected["rgb_sha256"][index], "No verified authored picture")
    mask = np.any(earlier_frames[index] != selected.rgb, axis=-1)
    return [int(mask[y:y+h, x:x+w].sum()), int(mask.sum()), w*h]


def prepare_groups(plan, phase):
    archive = Path(plan["archive"])
    split = "development" if phase == "development" else "held"
    for pin in plan["grid"][split]:
        rows = bank.prior.inspect(plan, pin)
        originals = [bank.prior.inspect(plan, other)[0] for other in plan["grid"][split]
                     if other["component"] != pin["component"]]
        yield "grid", pin["ordinal"], pin["component"], rows, originals[0]
    if phase == "held":
        native = bank.native_groups(archive)
        for pin in plan["native"]:
            rows = native[pin["index"]]
            require(pin["digests"] == [row["file_sha256"] for row in rows], "Native source commitments differ")
            yield "cremad", pin["index"], pin["index"], [(archive / row["file"], row, None) for row in rows], None


def cases_for(rows, alternative):
    origin, original, _ = rows[0]
    yield "neutral", "reencode", origin, original, 28, False, None, False
    yield "neutral", "reencode_low", origin, original, 33, False, None, False
    yield "neutral", "global_tone", origin, original, 28, True, None, False
    if alternative is not None:
        source, row, _ = alternative
        yield "different_source", "different_source", source, row, 28, False, None, False
    for source, row, motion in rows[1:]:
        role = row["condition"]
        if role == "audio_delay_320ms":
            continue  # Visually identical to the matched original.
        positive = role in bank.prior.NEURAL or role.startswith("ai_")
        require(not positive or row.get("neural_provenance_verified") or row.get("known_ai_provenance"),
                "Positive must have verified authored provenance")
        yield ("neural" if positive else "ordinary_local"), role, source, row, 28, False, row["edit_frames_half_open"], (
            role == "liveportrait_source_time" and motion == bank.prior.SOURCE_CLOCK
        )


def run_phase(run, phase):
    plan = sealed(run)
    cv2.setNumThreads(1)
    if phase == "held":
        cal = read(run / "calibration.json")
        require(cal["freeze"] == sha(run / "freeze.json") and cal["development"] == sha(run / "development.jsonl"),
                "Calibrate before measuring held views")
    deliver = run / phase
    deliver.mkdir(mode=0o700)
    with (run / f"{phase}.jsonl").open("x") as journal:
        for bank_name, index, group, rows, alternative in prepare_groups(plan, phase):
            earlier, original, _ = rows[0]
            early = earlier_table(earlier, original)
            earlier_frames = list(verified_frames(earlier, early))
            require([bank.raw_digest(rgb) for rgb in earlier_frames] == original["expected"]["rgb_sha256"],
                    "Decoded earlier picture identities differ from their authored source")
            for kind, label, source, authored, crf, tone, truth, qualified in cases_for(rows, alternative):
                authored_table = earlier_table(source, authored)
                dest = deliver / f"{bank_name}-{index:03d}-{label}.mkv"
                encode(source, dest, crf, tone)
                later = probe_frames(dest, cap=MAX_FRAMES, timeout_s=10, max_pixels=MAX_PIXELS)
                require(_seconds(later) == _seconds(authored_table) and later.width == authored_table.width
                        and later.height == authored_table.height, "Delivery changed the native picture clock or dimensions")
                measure = observe(earlier_frames, dest, early, later)
                spatial = authored_spatial(earlier_frames, source, authored_table, authored["expected"],
                                            measure["best"]) if truth else None
                out = {"bank": bank_name, "slot": index, "source_group": group, "kind": kind, "condition": label,
                       "known_edit": truth, "visible_source_time": qualified, "spatial": spatial, **measure,
                       "authored_sha256": sha(source), "earlier_sha256": sha(earlier), "submitted_sha256": sha(dest)}
                journal.write(json.dumps(out, sort_keys=True) + "\n")
                journal.flush()
            print(f"{phase} {bank_name} {index}", flush=True)


def rows(run, phase):
    return bank.prior.jsonl(run / f"{phase}.jsonl")


def calibrate(run):
    sealed(run)
    ordinary = [row for row in rows(run, "development") if row["kind"] in {"neutral", "different_source"}]
    values = [row["best"]["score"] for row in ordinary if row["best"] is not None]
    require(len(ordinary) == 32 and len(values) >= 22, "Development comparison coverage differs")
    write(run / "calibration.json", {"freeze": sha(run / "freeze.json"), "development": sha(run / "development.jsonl"),
                                    "point": max(FLOOR, max(values) + .15), "development_controls_measured": len(values)})


def iou(a, b):
    return Fraction(max(0, min(a[1], b[1]) - max(a[0], b[0])), max(a[1], b[1]) - min(a[0], b[0]))


def decide(row, point):
    best = row["best"]
    flag = best is not None and best["score"] > point
    truth = row["known_edit"]
    spatial = row["spatial"]
    return {
        "assessed": row["status"] == "measured",
        "flag": flag,
        "exact_correct": bool(truth and any(iou(r, truth) >= Fraction(1, 2) for r in row["exact"])),
        "time_correct": bool(flag and truth and iou(best["frames"], truth) >= Fraction(1, 2)),
        "spatial_correct": bool(flag and spatial and spatial[0] * 8 >= spatial[2] and spatial[0] * 12 >= spatial[1]),
    }


def report(run):
    sealed(run)
    cal = read(run / "calibration.json")
    held = rows(run, "held")
    require(len(held) == 8 * 13 + 10 * 8, "Held roster incomplete")
    panels = {}
    for name in ("grid", "cremad"):
        counts, new_groups = {}, set()
        for kind in ("neutral", "different_source", "ordinary_local", "neural"):
            total = Counter()
            for row in (row for row in held if (row["bank"], row["kind"]) == (name, kind)):
                metrics = decide(row, cal["point"])
                joint = metrics["time_correct"] and metrics["spatial_correct"]
                total.update({"views": 1, **metrics, "joint_correct": joint,
                              "visible_source_time": row["visible_source_time"],
                              "visible_source_time_joint": row["visible_source_time"] and joint})
                if kind == "neural" and joint and not metrics["exact_correct"]:
                    new_groups.add(row["source_group"])
            counts[kind] = dict(total)
        qualifying = all(counts[k]["assessed"] >= .8 * counts[k]["views"] for k in ("neural", "neutral"))
        panels[name] = {"counts": counts, "added_correct_source_groups": len(new_groups),
                        "advances": qualifying and len(new_groups) >= 3 and counts["neutral"]["flag"] <= 1
                        and not counts["different_source"].get("flag", 0)}
    write(run / "aggregate.json", {"spec": SPEC, "point": cal["point"], "freeze_sha256": sha(run / "freeze.json"),
                                    "held_sha256": sha(run / "held.jsonl"), "panels": panels,
                                    "advances_both": all(panel["advances"] for panel in panels.values())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "calibrate", "held", "report"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.stage in ("development", "held"):
        run_phase(args.run, args.stage)
    else:
        {"freeze": freeze, "calibrate": calibrate, "report": report}[args.stage](args.run)


if __name__ == "__main__":
    main()
