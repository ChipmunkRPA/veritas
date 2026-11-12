# SPDX-License-Identifier: Apache-2.0
"""Frozen matched local timing comparison on separate native actor groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio, source_ranges
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import lipsync, sampling, xcorr
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import video_face_texture_return as common
from scripts import video_lipsync_short_bracket as method

CONDITIONS = ("genuine", "gain", "global_late", "global_early", "local_late", "local_early")
DELTA_TICKS = 5
EDIT_TICKS = 16
FADE_SAMPLES = 80
MIN_IOU = Fraction(2, 5)
GATE = "Gain four correct signed local delay ranges across at least three held actors, preserve all correct incumbent ranges, and add no more than one false range across all matched controls."


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def bindings():
    sources = (Path(__file__), Path(method.__file__), Path(lipsync.__file__), Path(xcorr.__file__),
               common.ROOT / "backend/app/detectors/audio/decode.py",
               common.ROOT / "backend/app/detectors/video/adapters/lipsync_xcorr.py")
    return {**common.bindings(), **{str(p.relative_to(common.ROOT)): common.digest(p) for p in sources}}


def freeze(folder):
    intake, receipt = (common.read(folder / f"intake-{part}.json") for part in ("freeze", "receipt"))
    require(receipt["roster_sha256"] == common.digest(folder / "intake-freeze.json")
            and [row["actor"] for row in intake["sources"]] == [row["actor"] for row in receipt["rows"]]
            and len(intake["sources"]) == len({row["actor"] for row in intake["sources"]}) == 15,
            "Exact source roster and separate known actor codes required")
    common.write(folder / "freeze.json", {"schema": "chipmunk.video-lipsync-local-delay-freeze/1",
        "bindings": bindings(), "rows": receipt["rows"], "intake_sha256": common.digest(folder / "intake-receipt.json"),
        "conditions": CONDITIONS, "delta_ticks": DELTA_TICKS, "edit_ticks": EDIT_TICKS, "fade_samples": FADE_SAMPLES,
        "minimum_exact_iou": str(MIN_IOU), "gate": GATE,
        "calibration": "For each arm use the larger of the fixed 0.10 floor and maximum negative development score plus 0.025.",
        "audio": "Map signed mono16k exactly onto its native source timeline; retain earlier samples on bounded tiny overlaps and require every sample covered.",
        "comparator": "Use the same current AV window routing, speaker attribution, crop aperture, lag and range arithmetic on the same source-bound single face inputs.",
        "candidate": "Short three-window lag brackets, aperture/energy alone and equal-weight aperture/energy plus mouth-motion/audio-spectral-change profiles. Timing evidence only."})


def frozen(folder):
    doc = common.read(folder / "freeze.json")
    require(doc["bindings"] == bindings() and doc["intake_sha256"] == common.digest(folder / "intake-receipt.json")
            and tuple(doc["conditions"]) == CONDITIONS and doc["gate"] == GATE, "Frozen comparison differs")
    return doc


def resample_clock(decoded):
    segments = source_ranges(decoded.mapping, 0, len(decoded.pcm))
    ticks = [(r["source_start_pts"], r["source_end_pts"]) for r in segments]
    first, last = ticks[0][0], ticks[-1][1]
    require(all(right[0] >= left[1] - 4 and right[0] <= left[1] for left, right in zip(ticks, ticks[1:], strict=False)),
            "Source audio contains a gap or more than four overlapping samples at a decoder boundary")
    integer = np.rint(decoded.pcm * 32768).astype("<i2")
    sample = np.zeros(last-first, dtype="<i2")
    filled = np.zeros(last-first, dtype=bool)
    for span in segments:
        lo, hi = span["source_start_pts"] - first, span["source_end_pts"] - first
        source = integer[span["start_sample"]:span["end_sample"]]
        open_positions = ~filled[lo:hi]
        require(len(source) == hi-lo, "Decoded source run does not map one-to-one")
        sample[lo:hi][open_positions] = source[open_positions]
        filled[lo:hi] = True
    require(filled.all(), "Native source audio gaps were not filled or interpolated")
    return first, sample, len(integer) - len(sample)


def sha(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def inputs(folder, row, yunet):
    path = folder / row["file"]
    require(common.digest(path) == row["sha256"], "Licensed native source bytes differ")
    table = probe_frames(path, timeout_s=15)
    require(not (table.rotation or table.truncated or table.interpolated) and (table.width, table.height) == (480, 360),
            "Expected verified complete upright pictures from the qualified source family")
    audio = decode_audio(path, source_sha256=row["sha256"], max_seconds=8)
    audio_start, sample, removed = resample_clock(audio)
    origin = Fraction(audio_start, lipsync.SR)
    grid = lipsync.tick_grid(table.times(), sampling.duration_s(table), len(sample), audio_start_s=float(origin))
    wanted = set(grid.frames) - {-1}
    faces, native_patches, commitments = {}, {}, {}
    for frame in decode_frames(path, table, sorted(wanted), timeout_s=20):
        require(frame.pts_verified and frame.pts == table.pts[frame.index], "Decoder returned an unverified native frame")
        commitments[frame.index] = {"frame_index": frame.index, "pts": frame.pts, "decoded_rgb_sha256": sha(frame.rgb)}
        detected = detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=yunet, min_size=48)
        if len(detected) != 1:
            continue
        face = detected[0]
        record = {"frame_index": frame.index, "pts": frame.pts, "landmarks": face.landmarks,
                  "bbox": dict(zip(("x", "y", "w", "h"), face.box(), strict=True))}
        x, y, w, h = lipsync.mouth_box(record)
        bounds = (int(np.clip(np.floor(x), 0, table.width-1)), int(np.clip(np.floor(y), 0, table.height-1)),
                  int(np.clip(np.ceil(x+w), 1, table.width)), int(np.clip(np.ceil(y+h), 1, table.height)))
        patch = lipsync.mouth_patch(frame.rgb, record)
        native_patches[frame.index], faces[frame.index] = patch, record
        commitments[frame.index].update(mouth_box_xyxy=bounds, mouth_patch_sha256=sha(patch))
    require(wanted == set(commitments), "Exact decoded picture support is incomplete")
    patches = {i: native_patches[frame] for i, frame in enumerate(grid.frames) if frame in native_patches}
    return grid, table, origin, sample, removed, faces, patches, commitments


def shift(pcm, delta):
    out = np.zeros_like(pcm)
    if delta > 0:
        out[delta:] = pcm[:-delta]
    else:
        out[:delta] = pcm[-delta:]
    return out


def variants(pcm, ticks):
    start = (ticks - EDIT_TICKS) // 2
    end = start + EDIT_TICKS
    yield "genuine", pcm, None, None
    ramp = np.ones(len(pcm), dtype=np.float64)
    ramp[start*lipsync.HOP:end*lipsync.HOP] = .70
    yield "gain", np.rint(pcm * ramp).astype("<i2"), None, None
    for role, delta in (("late", DELTA_TICKS), ("early", -DELTA_TICKS)):
        moved = shift(pcm, delta*lipsync.HOP)
        yield "global_"+role, moved, None, None
        a, b = (tick * lipsync.HOP for tick in (start, end))
        weight = np.ones(b-a, dtype=np.float64)
        fade = np.arange(1, FADE_SAMPLES+1, dtype=np.float64) / FADE_SAMPLES
        weight[:FADE_SAMPLES], weight[-FADE_SAMPLES:] = fade, fade[::-1]
        changed = pcm.copy()
        changed[a:b] = np.rint((1-weight)*pcm[a:b] + weight*moved[a:b]).astype("<i2")
        require(np.array_equal(changed[:a], pcm[:a]) and np.array_equal(changed[b:], pcm[b:])
                and np.count_nonzero(changed[a:b] != pcm[a:b]) >= lipsync.HOP,
                "The local intervention has no verified source sample support")
        yield "local_"+role, changed, [start, end], delta


def evidence(candidate, grid, origin, table, commitments, pcm):
    if candidate is None:
        return None
    records = {}
    for role, (start, end) in candidate["roles"].items():
        ticks = list(range(start, end))
        entries = [{"tick": tick, **commitments[grid.frames[tick]]} for tick in ticks]
        audio = {}
        for name in ("flank", "changed"):
            lag = candidate[f"{name}_lag_ticks"]
            a, b = (tick * lipsync.HOP for tick in (start+lag, end+lag))
            audio[name] = {"samples": [a,b], "native_time": [str(origin + Fraction(i,lipsync.SR)) for i in (a,b)],
                           "pcm_sha256": sha(pcm[a:b])}
        records[role] = {"native_video_time_base": table.time_base,
                         "video_ticks_native_time": [str(origin + Fraction(i,25)) for i in (start,end)],
                         "frames": entries, "audio": audio}
    return records


def run(folder, split):
    doc = frozen(folder)
    if split == "held":
        require(common.read(folder / "calibration.json")["freeze_sha256"] == common.digest(folder / "freeze.json"),
                "Seal the independent development points before held scores")
    cv2.setNumThreads(2)
    yu = load_yunet(common.FACE_FILE)
    start = time.monotonic()
    with (folder / f"{split}.jsonl").open("x") as stream:
        for row in (r for r in doc["rows"] if r["split"] == split):
            grid, table, origin, pcm, removed, faces, patches, committed = inputs(folder, row, yu)
            cases = []
            for label, wave, truth, delta in variants(pcm, grid.ticks):
                signal = wave.astype(np.float32) / 32768.
                reference = method.current(faces, patches, signal, grid)
                arms = method.candidate(patches, signal, grid.ticks)
                for trial in arms.values():
                    trial["evidence"] = evidence(trial["best"], grid, origin, table, committed, wave)
                cases.append({"case": label, "edit_extent_ticks": truth, "known_relative_lag_ticks": delta,
                              "input_pcm_sha256": sha(wave), "current": reference, "arms": arms})
            item = {"group": row["ordinal"], "source_sha256": row["sha256"], "origin_seconds": str(origin),
                    "removed_overlapping_samples": removed, "verified_native_pictures": len(committed),
                    "available_mouth_ticks": len(patches), "cases": cases}
            stream.write(json.dumps(item, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            print(split, row["ordinal"], grid.ticks, len(patches), "committed", flush=True)
    print("elapsed_s", round(time.monotonic()-start,3), flush=True)


def calibrate(folder):
    doc = frozen(folder)
    rows = list(map(json.loads, (folder / "development.jsonl").open()))
    require(len(rows) == sum(row["split"] == "development" for row in doc["rows"]), "Complete development groups required")
    points = {}
    for arm in method.ARMS:
        negative = [case["arms"][arm]["best"]["score"] for row in rows for case in row["cases"]
                    if not case["edit_extent_ticks"] and case["arms"][arm]["best"] is not None]
        points[arm] = max(method.POINT_FLOOR, max(negative, default=0) + method.NEGATIVE_MARGIN)
    common.write(folder / "calibration.json", {"freeze_sha256": common.digest(folder / "freeze.json"),
        "development_sha256": common.digest(folder / "development.jsonl"), "points": points})
    print(points, flush=True)


def overlap(first, second):
    a,b=map(lambda val: Fraction(str(val)), first)
    c,d=map(lambda val: Fraction(str(val)), second)
    return max(0,min(b,d)-max(a,c))/(max(b,d)-min(a,c))


def correct(row, truth, lag):
    relative = row["relative_lag_ticks"]
    return overlap(row["extent_ticks"], truth) >= MIN_IOU and (relative is None or relative*lag > 0)


def readout(folder):
    doc = frozen(folder)
    calibration = common.read(folder / "calibration.json")
    points = calibration["points"]
    rows = list(map(json.loads,(folder / "held.jsonl").open()))
    require([row["group"] for row in rows] == [row["ordinal"] for row in doc["rows"] if row["split"] == "held"]
            and calibration["development_sha256"] == common.digest(folder / "development.jsonl"),
            "Held group or development scoring receipt differs")
    totals = {arm: Counter() for arm in ("current", *method.ARMS)}
    added, losses = {key: set() for key in method.ARMS}, Counter()
    by_condition = {label: {arm: Counter() for arm in totals} for label in CONDITIONS}
    for group in rows:
        require(set(case["case"] for case in group["cases"]) == set(CONDITIONS), "Incomplete matched group")
        for case in group["cases"]:
            truth, lag = case["edit_extent_ticks"], case["known_relative_lag_ticks"]
            current = case["current"]
            reference = any(correct(row, truth, lag) for row in current["ranges"]) if truth else False
            for arm in totals:
                candidate = None if arm == "current" else case["arms"][arm]["best"]
                selected = bool(candidate is not None and candidate["score"] > points[arm]) if arm != "current" else bool(current["ranges"])
                measured = bool(current["measured_windows"] if arm == "current" else case["arms"][arm]["compared_brackets"])
                located = (reference if arm=="current" else selected and correct(candidate, truth,lag)) if truth else False
                key = ("correct_range" if located else "wrong_range" if selected else "measured_no_range" if measured else "unassessed") if truth else (
                      "false_range" if selected else "measured_no_range" if measured else "unassessed")
                totals[arm][key] += 1
                by_condition[case["case"]][arm][key] += 1
                if arm != "current" and located and not reference:
                    added[arm].add(group["group"])
                if arm != "current" and reference and not located:
                    losses[arm] += 1
    result = {"schema": "chipmunk.video-lipsync-local-delay-transfer-result/1",
        "freeze_sha256": common.digest(folder / "freeze.json"), "held_journal_sha256": common.digest(folder / "held.jsonl"),
        "calibration_sha256": common.digest(folder / "calibration.json"), "held_groups": len(rows),
        "verified_native_pictures": sum(r["verified_native_pictures"] for r in rows), "totals": totals,
        "by_condition": by_condition, "added_localization_groups": {key: len(value) for key,value in added.items()},
        "lost_incumbent_correct_ranges": dict(losses),
        "pilot_advances": {arm: totals[arm]["correct_range"] >= totals["current"]["correct_range"] + 4 and len(added[arm])>=3
                           and not losses[arm] and totals[arm]["false_range"] <= totals["current"]["false_range"] + 1
                           for arm in method.ARMS}}
    common.write(folder / "aggregate.json", result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "development", "calibrate", "held", "readout"))
    parser.add_argument("folder", type=Path)
    args=parser.parse_args()
    if args.stage in {"development","held"}:
        run(args.folder,args.stage)
    else:
        {"freeze":freeze,"calibrate":calibrate,"readout":readout}[args.stage](args.folder)


if __name__ == "__main__":
    main()
