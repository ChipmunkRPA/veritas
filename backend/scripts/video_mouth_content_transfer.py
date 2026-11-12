# SPDX-License-Identifier: Apache-2.0
"""Test local speech-content agreement against same-speaker, energy-matched dubs."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio
from app.detectors.image.faces import load_yunet
from app.detectors.video import xcorr
from scripts import video_lipsync_local_delay_transfer as prior
from scripts import video_lipsync_short_bracket as incumbent
from scripts.video_common_sources_v89 import read, require, sha, write

HOP, RATE, FPS = 640, 16000, 25
FADE, EDIT_TICKS = 160, 24
FIT_RMS, READOUT_RMS = .009, .004
ROLE_ORDER = ("genuine", "gain", "filter", "same_words", "different_words")
ARMS = ("current", "audio_only", "paired")
METHOD = "Frozen linear speech-spectrum prediction from ±1 native mouth samples with separate fit and control-only calibration actors."
GATE = "Locate at least 6/12 different-speech ranges across six held speakers, at least three more than either comparator, with no more than two added ordinary ranges relative to current timing."
BANDS = (180, 300, 460, 680, 980, 1350, 1850, 2500, 3400, 4600, 6500)


def bindings():
    paths = (Path(__file__), Path(prior.__file__), Path(incumbent.__file__))
    return {**prior.bindings(), **{str(path.relative_to(prior.common.ROOT)): sha(path) for path in paths}}


def freeze(folder):
    roster, receipt = (read(folder / f"intake-{part}.json") for part in ("freeze", "receipt"))
    require(receipt["roster_sha256"] == sha(folder / "intake-freeze.json"), "Publisher receipt differs from the freeze")
    rows = receipt["rows"]
    require([{key: row[key] for key in pin} for row, pin in zip(rows, roster["sources"], strict=True)] == roster["sources"],
            "Exact predeclared source list required")
    actors = {row["actor"]: row["split"] for row in rows}
    require(len(rows) == 72 and Counter(actors.values()) == {"fit": 8, "calibration": 4, "held": 12},
            "Expected 24 disjoint actor groups and all publisher recordings")
    write(folder / "freeze.json", {"schema": "chipmunk.video-mouth-content-freeze/1", "bindings": bindings(),
          "receipt_sha256": sha(folder / "intake-receipt.json"), "method": METHOD, "gate": GATE,
          "edit_ticks": EDIT_TICKS, "fade_samples": FADE, "conditions": ROLE_ORDER, "rows": rows})


def frozen(folder):
    data = read(folder / "freeze.json")
    require(data["bindings"] == bindings() and data["receipt_sha256"] == sha(folder / "intake-receipt.json")
            and data["method"] == METHOD and data["gate"] == GATE, "The comparison differs from its frozen code or sources")
    return data


def groups(data, split):
    grouped = {}
    for row in data["rows"]:
        if row["split"] == split:
            grouped.setdefault(row["group"], {})[row["role"]] = row
    return grouped.items()


def acoustic(pcm, ticks):
    source = pcm[:ticks * HOP].reshape(ticks, HOP).astype(np.float64) / 32768.0
    rms = np.sqrt(np.mean(source**2, axis=1))
    power = np.abs(np.fft.rfft(source * np.hanning(HOP), axis=1))**2
    frequencies = np.fft.rfftfreq(HOP, 1 / RATE)
    bands = np.column_stack([power[:, (frequencies >= low) & (frequencies < high)].sum(axis=1)
                             for low, high in zip(BANDS, BANDS[1:], strict=False)])
    logs = np.log(np.maximum(bands, 1e-8))
    return logs - logs.mean(axis=1, keepdims=True), rms


def visual(patches, ticks):
    rows = {}
    for tick, patch in patches.items():
        intensity = patch.astype(np.float64)
        detail = (intensity - intensity.mean()) / max(intensity.std(), 8)
        cells = detail.reshape(4, 6, 4, 8).mean(axis=(1, 3)).ravel()
        rows[tick] = np.r_[cells, xcorr.aperture(patch)]
    available = np.zeros(ticks, dtype=bool)
    matrix = np.zeros((ticks, 51), dtype=np.float64)
    for tick in range(1, ticks - 1):
        if all(i in rows for i in (tick - 1, tick, tick + 1)):
            matrix[tick] = np.concatenate([rows[i] for i in (tick - 1, tick, tick + 1)])
            available[tick] = True
    return matrix, available


def normalizer(data):
    return data.mean(axis=0), np.maximum(data.std(axis=0), .01)


def fit(folder):
    data = frozen(folder)
    yunet = load_yunet(prior.common.FACE_FILE)
    visual_samples, audio_samples = [], []
    for group, rows in groups(data, "fit"):
        for role in ("target", "different_words"):
            grid, _, _, pcm, _, _, patches, _ = prior.inputs(folder, rows[role], yunet)
            a, rms = acoustic(pcm, grid.ticks)
            v, available = visual(patches, grid.ticks)
            use = available & (rms >= FIT_RMS)
            visual_samples.append(v[use])
            audio_samples.append(a[use])
        print("fit", group, flush=True)
    x, y = np.vstack(visual_samples), np.vstack(audio_samples)
    require(len(x) >= 200, "Insufficient speaker-separated active fit examples")
    x_mean, x_std = normalizer(x)
    y_mean, y_std = normalizer(y)
    xx, yy = (x - x_mean) / x_std, (y - y_mean) / y_std
    gram = xx.T @ xx + np.eye(xx.shape[1]) * len(xx) * .10
    weights = np.linalg.solve(gram, xx.T @ yy)
    write(folder / "fit.json", {"freeze_sha256": sha(folder / "freeze.json"), "active_ticks": len(x),
          "x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "y_mean": y_mean.tolist(), "y_std": y_std.tolist(),
          "weights": weights.tolist()})


def donor_audio(folder, row):
    path = folder / row["file"]
    require(sha(path) == row["sha256"], "Same-speaker publisher audio differs from the frozen source")
    decoded = decode_audio(path, source_sha256=row["sha256"], max_seconds=8)
    return prior.resample_clock(decoded)[1]


def dub(donor, target, lo, hi):
    """Match overall duration without shifting pitch, then match the 40 ms energy envelope."""
    tempo = len(donor) / len(target)
    require(.5 <= tempo <= 2, "Speaker re-recording falls outside the fixed time-stretch range")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", "pipe:0",
               "-af", f"atempo={tempo:.12f}", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"]
    output = subprocess.run(command, input=donor.tobytes(), capture_output=True, check=True, timeout=12)
    stretched = np.frombuffer(output.stdout, dtype="<i2")
    require(len(stretched) >= hi, "Duration-matched speech failed to cover the authored edit")
    new = stretched[lo:hi].astype(np.float64)
    original = target[lo:hi].astype(np.float64)
    powers = [np.sqrt(np.mean(a.reshape(-1, HOP)**2, axis=1)) for a in (original, new)]
    factors = np.clip(powers[0] / np.maximum(powers[1], 16), .05, 12)
    return new * np.interp(np.arange(len(new)), (np.arange(len(factors)) + .5) * HOP, factors)


def cases(folder, rows, pcm, ticks):
    start = (ticks - EDIT_TICKS) // 2
    end = start + EDIT_TICKS
    require(start >= 8 and end + 8 <= ticks, "The exact center intervention needs two eight-tick flanks")
    lo, hi = start * HOP, end * HOP
    source = pcm[lo:hi].astype(np.float64)
    alpha = np.ones(hi - lo, dtype=np.float64)
    alpha[:FADE] = np.arange(1, FADE + 1) / FADE
    alpha[-FADE:] = alpha[:FADE][::-1]
    yield "genuine", pcm, None, None
    for role in ROLE_ORDER[1:]:
        if role == "gain":
            proposal = .65 * source
        elif role == "filter":
            proposal = .75 * source + .25 * pcm[lo - 1:hi - 1]
        else:
            proposal = dub(donor_audio(folder, rows[role]), pcm, lo, hi)
        wave = pcm.copy()
        wave[lo:hi] = np.clip(np.rint((1 - alpha) * source + alpha * proposal), -32768, 32767).astype("<i2")
        require(np.array_equal(wave[:lo], pcm[:lo]) and np.array_equal(wave[hi:], pcm[hi:])
                and np.count_nonzero(wave[lo:hi] != pcm[lo:hi]) > HOP, "The sample intervention must match its half-open extent")
        return_truth = [start, end] if role == "different_words" else None
        yield role, wave, return_truth, [lo, hi]


def best_range(values, active):
    """Compare a measured core against both separately measured flanks."""
    best = None
    compared = 0
    for width in (12, 20, 28):
        for start in range(8, len(values) - width - 7, 2):
            end = start + width
            bounds = ((start - 8, start), (start, end), (end, end + 8))
            used = [np.flatnonzero(active[lo:hi]) + lo for lo, hi in bounds]
            if len(used[0]) < 2 or len(used[2]) < 2 or len(used[1]) < max(6, (2 * width + 4) // 5):
                continue
            before, during, after = (float(np.median(values[indices])) for indices in used)
            strength = min(during - before, during - after)
            compared += 1
            if best is None or strength > best["score"]:
                best = {"score": strength, "extent_ticks": [start, end], "context_ticks": [int(indices[len(indices)//2])
                         for indices in used], "medians": [before, during, after]}
    return {"compared": compared, "best": best}


def annotate(trial, grid, table, origin, pcm, commitments):
    if (best := trial["best"]) is not None:
        best["support"] = []
        for role, tick in zip(("before", "during", "after"), best["context_ticks"], strict=True):
            start, end = tick * HOP, (tick + 1) * HOP
            index = grid.frames[tick]
            frame = commitments.get(index)
            best["support"].append({"role": role, "tick": tick, "frame": frame, "frame_time_base": table.time_base,
                                    "audio_samples": [start, end], "audio_native_s": [str(origin + Fraction(tick, FPS)),
                                     str(origin + Fraction(tick + 1, FPS))], "pcm_sha256": prior.sha(pcm[start:end])})
    return trial


def run(folder, split):
    data, learned = frozen(folder), read(folder / "fit.json")
    require(learned["freeze_sha256"] == sha(folder / "freeze.json"), "The fit is not from the frozen panel")
    if split == "held":
        seal = read(folder / "calibration.json")
        require(seal["fit_sha256"] == sha(folder / "fit.json") and seal["journal_sha256"] == sha(folder / "calibration.jsonl"),
                "Held scoring requires sealed fit and separate calibration actors")
    values = {key: np.asarray(learned[key], dtype=np.float64) for key in ("x_mean", "x_std", "y_mean", "y_std", "weights")}
    yunet = load_yunet(prior.common.FACE_FILE)
    with (folder / f"{split}.jsonl").open("x") as out:
        for group, rows in groups(data, split):
            grid, table, origin, pcm, _, faces, patches, committed = prior.inputs(folder, rows["target"], yunet)
            v, available = visual(patches, grid.ticks)
            predicted = ((v - values["x_mean"]) / values["x_std"]) @ values["weights"]
            _, original_rms = acoustic(pcm, grid.ticks)
            scored = []
            for role, wave, truth, altered in cases(folder, rows, pcm, grid.ticks):
                a, rms = acoustic(wave, grid.ticks)
                standard = (a - values["y_mean"]) / values["y_std"]
                support = (rms >= READOUT_RMS) & available
                current = incumbent.current(faces, patches, wave.astype(np.float32) / 32768.0, grid)
                arms = {"audio_only": best_range(np.mean(standard**2, axis=1), support),
                        "paired": best_range(np.mean((standard - predicted)**2, axis=1), support)}
                for trial in arms.values():
                    annotate(trial, grid, table, origin, wave, committed)
                error_db = None
                if altered:
                    lo, hi = (index // HOP for index in altered)
                    good = original_rms[lo:hi] >= FIT_RMS
                    if good.any():
                        ratio = (rms[lo:hi][good] + 1e-9) / original_rms[lo:hi][good]
                        error_db = float(np.median(np.abs(20 * np.log10(ratio))))
                scored.append({"role": role, "truth_ticks": truth, "changed_samples": altered,
                               "pcm_sha256": prior.sha(wave), "active_edit_median_rms_error_db": error_db,
                               "current": current, "arms": arms})
            out.write(json.dumps({"group": group, "origin_s": str(origin), "native_pictures": len(committed),
                                  "source_sha256": rows["target"]["sha256"], "cases": scored}, sort_keys=True, allow_nan=False) + "\n")
            out.flush()
            print(split, group, grid.ticks, len(patches), flush=True)


def journal(folder, data, split):
    rows = list(map(json.loads, (folder / f"{split}.jsonl").read_text().splitlines()))
    require([r["group"] for r in rows] == [group for group, _ in groups(data, split)]
            and all(tuple(c["role"] for c in row["cases"]) == ROLE_ORDER for row in rows), "Incomplete matched comparison")
    return rows


def calibrate(folder):
    data = frozen(folder)
    rows = journal(folder, data, "calibration")
    thresholds = {}
    for name in ARMS[1:]:
        controls = [best["score"] for row in rows for case in row["cases"] if case["truth_ticks"] is None
                    if (best := case["arms"][name]["best"]) is not None]
        require(len(controls) >= 8, "Too few independently measured controls to freeze a point")
        thresholds[name] = max(.025, max(controls) + .025)
    write(folder / "calibration.json", {"fit_sha256": sha(folder / "fit.json"), "journal_sha256": sha(folder / "calibration.jsonl"),
          "point_rule": "Larger of .025 and the maximum measured ordinary calibration score plus .025", "points": thresholds})


def decision(case, name, point):
    if name == "current":
        measured = bool(case["current"]["measured_windows"])
        spans = [row["extent_ticks"] for row in case["current"]["ranges"]]
    else:
        trial = case["arms"][name]
        best = trial["best"]
        measured = bool(trial["compared"])
        spans = [best["extent_ticks"]] if best is not None and best["score"] > point else []
    if not measured:
        return "unassessed"
    if not spans:
        return "measured_no_range"
    truth = case["truth_ticks"]
    if truth is None:
        return "ordinary_range"
    return "correct_range" if any(prior.overlap(span, truth) >= Fraction(1, 2) for span in spans) else "wrong_range"


def readout(folder):
    data = frozen(folder)
    rows = journal(folder, data, "held")
    points = read(folder / "calibration.json")["points"]
    totals = {name: Counter() for name in ARMS}
    by_condition = {role: {name: Counter() for name in ARMS} for role in ROLE_ORDER}
    errors = {role: [] for role in ROLE_ORDER}
    for row in rows:
        for case in row["cases"]:
            role = case["role"]
            for name in ARMS:
                outcome = decision(case, name, points.get(name))
                totals[name][outcome] += 1
                by_condition[role][name][outcome] += 1
            if (level := case["active_edit_median_rms_error_db"]) is not None:
                errors[role].append(level)
    paired = totals["paired"]
    clears = (paired["correct_range"] >= max(6, 3 + max(totals[k]["correct_range"] for k in ARMS[:2]))
              and paired["ordinary_range"] <= totals["current"]["ordinary_range"] + 2)
    result = {"schema": "chipmunk.video-mouth-content-transfer-result/1", "freeze_sha256": sha(folder / "freeze.json"),
              "held_journal_sha256": sha(folder / "held.jsonl"), "calibration_sha256": sha(folder / "calibration.json"),
              "held_groups": len(rows), "fit_active_ticks": read(folder / "fit.json")["active_ticks"], "thresholds": points,
              "native_pictures_committed": sum(row["native_pictures"] for row in rows), "totals": totals,
              "by_condition": by_condition, "median_active_edit_rms_error_db": {role: float(np.median(items))
                for role, items in errors.items() if items}, "pilot_advances": clears}
    write(folder / "aggregate.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "calibration", "calibrate", "held", "readout"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    if args.stage in ("calibration", "held"):
        run(args.folder, args.stage)
    else:
        {"freeze": freeze, "fit": fit, "calibrate": calibrate, "readout": readout}[args.stage](args.folder)


if __name__ == "__main__":
    main()
