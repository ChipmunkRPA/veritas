# SPDX-License-Identifier: Apache-2.0
"""Test a learned frame-pair alignment score on source-separated short slips."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit

from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import lipsync, xcorr
from app.detectors.video.decode import FrameTable
from scripts import video_lipsync_local_delay_transfer as previous
from scripts import video_lipsync_short_bracket as baseline
from scripts import video_reference_h264_grid_reserve as grid
from scripts.video_common_sources_v89 import read, require, sha, write

HOP = lipsync.HOP
LAGS = np.arange(-8, 9)
FIT_LAGS = (-8, -5, -3, 3, 5, 8)
WIDTHS = (12, 16, 20, 24)
FLANK = 10
FADE = 160
MIN_IOU = Fraction(2, 5)
CASES = ("recorded", "local_gain", "local_lowpass", "local_noise", "constant_early", "constant_late",
         "short_early", "short_late", "long_early", "long_late")
METHOD = {
    "fit": "L2 logistic pair interaction on synchronized and within-utterance shifted active source ticks",
    "spectral_edges_hz": np.geomspace(140, 6200, 11).tolist(),
    "fit_lags_ticks": list(FIT_LAGS), "lags_ticks": LAGS.tolist(), "search_widths_ticks": list(WIDTHS),
    "flank_ticks": FLANK, "fade_samples": FADE, "shift_tolerance_ticks": 2, "minimum_iou": str(MIN_IOU),
    "point": "Max of 0.025 and the largest measured ordinary calibration response plus 0.025",
    "gate": "At least four more correctly localized signed shifts than current across three held components; preserve all current correct shifts; at most one added control flag; at least 75% held positive and ordinary coverage.",
}


def source_plan(plan):
    completed = read(plan.parent / "result.json")
    require(completed["private_plan_sha256"] == sha(plan), "Completed source-plan bytes differ")
    rows = read(plan)["sources"]
    pilot = {row["component_index"] for row in rows if row["role"] == "pilot"}
    reserve = {row["component_index"] for row in rows if row["role"] == "reserve"}
    require(len(rows) == 32 and len(pilot) == len(reserve) == 8 and not pilot & reserve,
            "Exact source-separated GRID pilot and reserve required")
    rank = sorted(pilot, key=lambda group: hashlib.sha256(f"grid-lip-pair/1/{group}".encode()).digest())
    splits = {**dict.fromkeys(rank[:6], "fit"), **dict.fromkeys(rank[6:], "point"), **dict.fromkeys(reserve, "held")}
    selected = []
    for row in rows:
        entry, group = row["input"], row["component_index"]
        require(row["plan_status"] == "ready" and entry["source_license"] == "CC-BY-4.0"
                and entry["audio_origin_seconds"] == entry["video_origin_seconds"] == "0", "Unqualified source")
        selected.append({"ordinal": row["ordinal"], "group": group, "split": splits[group], "source": entry["source_sha256"],
                         "pcm": entry["pcm_file_sha256"]})
    require(Counter(row["split"] for row in selected) == {"fit": 12, "point": 4, "held": 16}, "Unexpected roster sizes")
    return rows, selected


def bindings(plan):
    return {"source_plan": sha(plan), "picture_reader": sha(Path(grid.__file__)), **previous.bindings()}


def frozen(folder, plan, *, create=False):
    source, roster = source_plan(plan)
    expected = {"schema": "chipmunk.grid-lipsync-local-contrast/1", "implementation": sha(Path(__file__)),
                "bindings": bindings(plan), "method": METHOD, "conditions": list(CASES), "roster": roster}
    if create:
        folder.mkdir(parents=True, mode=0o700)
        write(folder / "freeze.json", expected)
    else:
        require(read(folder / "freeze.json") == expected, "Comparison freeze differs")
    return list(zip(source, roster, strict=True))


def native_pcm(row):
    entry = row["input"]
    path = Path(entry["pcm_file"])
    require(sha(Path(entry["source_file"])) == entry["source_sha256"] and sha(path) == entry["pcm_file_sha256"],
            "Qualified source or decoded PCM differs")
    signal = np.load(path, allow_pickle=False)
    require(signal.shape == (47648,) and np.isfinite(signal).all()
            and entry["whole_pcm_source_ranges"] == [{"start_sample": 0, "end_sample": len(signal),
                "source_start_pts": 0, "source_end_pts": len(signal), "time_base": "1/16000"}], "PCM clock differs")
    return np.rint(signal * 32768).astype("<i2")


def observe(row, detector):
    entry, pcm = row["input"], native_pcm(row)
    table = FrameTable.from_json(entry["native_video_table"])
    grid.expected_clock(table)
    ticks = lipsync.tick_grid(table.times(), 3.0, len(pcm), audio_start_s=0.)
    require(ticks.frames == tuple(range(74)), "A timing tick is not bound to an exact native picture")
    faces, patches, evidence = {}, {}, {}
    for index, rgb in enumerate(grid.pictures(row)[:ticks.ticks]):
        face = detect_faces(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        if len(face) != 1:
            continue
        face = face[0]
        record = {"frame_index": index, "pts": table.pts[index], "landmarks": face.landmarks,
                  "bbox": dict(zip(("x", "y", "w", "h"), face.box(), strict=True))}
        x, y, w, h = lipsync.mouth_box(record)
        x0, y0 = int(np.clip(np.floor(x), 0, table.width-1)), int(np.clip(np.floor(y), 0, table.height-1))
        x1, y1 = int(np.clip(np.ceil(x+w), x0+1, table.width)), int(np.clip(np.ceil(y+h), y0+1, table.height))
        patches[index], faces[index] = lipsync.mouth_patch(rgb, record), record
        evidence[index] = {"frame": index, "pts": table.pts[index], "tb": table.time_base,
                           "picture_sha256": previous.sha(rgb), "mouth_patch_sha256": previous.sha(patches[index]),
                           "xyxy": [x0, y0, x1, y1]}
    return pcm, ticks, faces, patches, evidence


def context(values):
    feature = np.full((len(values), 3 * values.shape[1]), np.nan)
    feature[1:-1] = np.column_stack((values[1:-1], values[1:-1] - values[:-2], values[2:] - values[1:-1]))
    return feature


def visual_features(patches, ticks):
    features = np.full((ticks, 12), np.nan)
    for i, patch in patches.items():
        cut = patch[4:20, 4:28]
        scaled = np.clip((cut - np.median(patch)) / max(12, np.std(patch)), -3, 3)
        cep = cv2.dct(cv2.resize(scaled, (8, 8), interpolation=cv2.INTER_AREA))[:3, :4].ravel()
        features[i] = np.r_[xcorr.aperture(patch), cep[1:]]
    return context(features)


def acoustic_features(pcm):
    audio = pcm.astype(np.float64) / 32768
    speech = lipsync.vad(audio).speech
    windows = audio[:len(speech)*HOP].reshape(-1, HOP)
    power = np.abs(np.fft.rfft(windows * np.hanning(HOP), axis=1)) ** 2
    freq = np.fft.rfftfreq(HOP, 1 / lipsync.SR)
    edges = METHOD["spectral_edges_hz"]
    bands = np.column_stack([np.log10(np.maximum(power[:, (freq >= lo) & (freq < hi)].sum(axis=1), 1e-8))
                             for lo, hi in zip(edges, edges[1:], strict=False)])
    return context(bands - bands.mean(axis=1, keepdims=True)), speech


def shifted(pcm, delay):
    return previous.shift(pcm, delay*HOP)


def blend(pcm, replacement, start, stop):
    lo, hi = start*HOP, stop*HOP
    alpha = np.ones(hi - lo)
    alpha[:FADE] = np.arange(1, FADE + 1) / FADE
    alpha[-FADE:] = alpha[:FADE][::-1]
    out = pcm.copy()
    out[lo:hi] = np.clip(np.rint(pcm[lo:hi]*(1-alpha) + replacement[lo:hi]*alpha), -32768, 32767).astype("<i2")
    require(np.array_equal(out[:lo], pcm[:lo]) and np.array_equal(out[hi:], pcm[hi:])
            and np.count_nonzero(out[lo:hi] != pcm[lo:hi]) >= HOP, "Timed input does not have bounded measured changes")
    return out


def variants(pcm, ordinal):
    start = 26 + ordinal % 5
    end = start + 16
    noise = np.random.default_rng(36120 + ordinal).normal(0, 1, len(pcm))
    rms = np.sqrt(np.mean(pcm[start*HOP:end*HOP].astype(np.float64)**2))
    low = sosfiltfilt(butter(4, 1900, fs=lipsync.SR, output="sos"), pcm)
    yield "recorded", pcm, None, None
    yield "local_gain", blend(pcm, .68*pcm, start, end), None, None
    yield "local_lowpass", blend(pcm, low, start, end), None, None
    yield "local_noise", blend(pcm, pcm + .12*rms*noise, start, end), None, None
    yield "constant_early", shifted(pcm, -5), None, None
    yield "constant_late", shifted(pcm, 5), None, None
    for tag, interval, size in (("short", (start, end), 5), ("long", (start-3, end+3), 7)):
        for direction, d in (("early", -size), ("late", size)):
            yield tag + "_" + direction, blend(pcm, shifted(pcm, d), *interval), list(interval), d


def samples(visual, acoustic, speech):
    visual_ok = np.isfinite(visual).all(axis=1)
    audio_ok = speech & np.isfinite(acoustic).all(axis=1)
    for lag in (0, *FIT_LAGS):
        video_tick = np.arange(max(0, -lag), min(len(visual), len(acoustic) - lag))
        audio_tick = video_tick + lag
        valid = visual_ok[video_tick] & audio_ok[video_tick] & audio_ok[audio_tick]
        yield visual[video_tick[valid]], acoustic[audio_tick[valid]], lag == 0


def standardize(values, offset, scale):
    return np.clip((values - offset) / scale, -4, 4)


def fit(folder, pairs, detector):
    pieces = []
    for row, rec in pairs:
        if rec["split"] != "fit":
            continue
        pcm, ticks, _, patches, _ = observe(row, detector)
        pieces.extend(samples(visual_features(patches, ticks.ticks), *acoustic_features(pcm)))
    video = np.concatenate([v for v, _, _ in pieces])
    audio = np.concatenate([a for _, a, _ in pieces])
    labels = np.concatenate([np.full(len(v), y, dtype=np.float64) for v, _, y in pieces])
    positives = int(labels.sum())
    require(positives >= 180 and len(labels)-positives >= 600, "Insufficient independent active fit ticks")
    vmean, amean = video.mean(axis=0), audio.mean(axis=0)
    vstd, astd = np.maximum(video.std(axis=0), .03), np.maximum(audio.std(axis=0), .03)
    v, a = standardize(video, vmean, vstd), standardize(audio, amean, astd)
    x = (v[:, :, None] * a[:, None, :]).reshape(len(v), -1)
    weights = np.where(labels, .5 / positives, .5 / (len(labels) - positives))

    def objective(theta):
        logits = x @ theta
        penalty = .08
        return (np.sum(weights*(np.logaddexp(0, logits) - labels*logits)) + .5*penalty*(theta @ theta),
                x.T @ (weights*(expit(logits)-labels)) + penalty*theta)

    fit_result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 150})
    require(fit_result.success and np.isfinite(fit_result.x).all(), "Pair alignment fit did not converge")
    write(folder / "head.json", {"freeze_sha256": sha(folder / "freeze.json"), "aligned_pairs": positives,
          "shifted_pairs": len(labels) - positives, "vmean": vmean.tolist(), "vstd": vstd.tolist(),
          "amean": amean.tolist(), "astd": astd.tolist(), "matrix": fit_result.x.reshape(v.shape[1], a.shape[1]).tolist()})


def candidate(video, audio, speech, head):
    v = standardize(video, head["vmean"], head["vstd"])
    a = standardize(audio, head["amean"], head["astd"])
    logits = np.full((len(v), len(LAGS)), np.nan)
    active = np.zeros_like(logits, dtype=bool)
    transformed = v @ np.asarray(head["matrix"])
    for i, lag in enumerate(LAGS):
        times = np.arange(max(0, -lag), min(len(v), len(a) - lag))
        aligned = times + lag
        logits[times, i] = expit((transformed[times] * a[aligned]).sum(axis=1))
        active[times, i] = speech[aligned]

    def profile(start, stop):
        block = logits[start:stop]
        anchors = np.isfinite(block).all(axis=1)
        if anchors.sum() < (stop-start) * .80:
            return np.full(len(LAGS), np.nan)
        spoken = active[start:stop][anchors].mean(axis=0) >= .35
        return np.where(spoken, block[anchors].mean(axis=0), np.nan)

    best = None
    measured = 0
    for width in WIDTHS:
        for start in range(FLANK + 8, len(v) - width - FLANK - 7, 2):
            end = start + width
            roles = [[start-FLANK, start], [start, end], [end, end+FLANK]]
            before, middle, after = (profile(*role) for role in roles)
            shared = np.isfinite(before) & np.isfinite(middle) & np.isfinite(after)
            if shared.sum() < 8:
                continue
            measured += 1
            reference = int(np.argmax(np.where(shared, np.minimum(before, after), -np.inf)))
            allowed = shared & (np.abs(LAGS - LAGS[reference]) >= 3)
            for target in np.flatnonzero(allowed):
                score = min(middle[target] - middle[reference], 2*(before[reference] - before[target]),
                            2*(after[reference] - after[target]))
                if score > 0 and (best is None or score > best["score"]):
                    best = {"score": float(score), "extent_ticks": [start, end], "roles": roles,
                            "flank_lag_ticks": int(LAGS[reference]), "changed_lag_ticks": int(LAGS[target]),
                            "relative_lag_ticks": int(LAGS[target] - LAGS[reference])}
    return {"measured": measured, "best": best}


def exhibit(best, evidence, pcm):
    if best is None:
        return None
    output = {}
    for role, (lo, hi) in zip(("before", "during", "after"), best["roles"], strict=True):
        supported = [evidence[i] for i in range(lo, hi) if i in evidence]
        require(supported, "Measured role has no native mouth image")
        windows = {}
        for label, key in (("reference", "flank_lag_ticks"), ("alternate", "changed_lag_ticks")):
            lag = best[key]
            start, end = (lo + lag)*HOP, (hi + lag)*HOP
            require(0 <= start < end <= len(pcm), "Evidence audio slice left the native clock")
            windows[label] = {"samples": [start, end], "sha256": previous.sha(pcm[start:end])}
        output[role] = {"picture": supported[len(supported)//2], "pcm": windows}
    return output


def score(folder, pairs, detector, split):
    head = read(folder / "head.json")
    require(head["freeze_sha256"] == sha(folder / "freeze.json"), "Pair head differs from freeze")
    if split == "held":
        point = read(folder / "point.json")
        require(point["head_sha256"] == sha(folder / "head.json") and point["journal_sha256"] == sha(folder / "point.jsonl"),
                "Pre-held comparison point differs")
    start = time.monotonic()
    with (folder / f"{split}.jsonl").open("x") as stream:
        for row, rec in pairs:
            if rec["split"] != split:
                continue
            native, ticks, faces, patches, evidence = observe(row, detector)
            visual = visual_features(patches, ticks.ticks)
            cases = []
            for name, pcm, interval, delay in variants(native, rec["ordinal"]):
                trial = candidate(visual, *acoustic_features(pcm), head)
                incumbent = baseline.current(faces, patches, pcm.astype(np.float32) / 32768, ticks)
                cases.append({"case": name, "known_extent_ticks": interval, "known_delay_ticks": delay,
                              "pcm_sha256": previous.sha(pcm), "current": incumbent, "candidate": trial,
                              "evidence": exhibit(trial["best"], evidence, pcm)})
            stream.write(json.dumps({**rec, "mouth_pictures": len(evidence), "cases": cases}, allow_nan=False, sort_keys=True) + "\n")
            stream.flush()
            print(split, rec["ordinal"], len(evidence), "native mouth pictures", flush=True)
    if split == "held":
        write(folder / "runtime.json", {"elapsed_s": time.monotonic() - start, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "point_sha256": sha(folder / "point.json"), "held_sha256": sha(folder / "held.jsonl")})


def rows(folder, stage):
    return list(map(json.loads, (folder / f"{stage}.jsonl").read_text().splitlines()))


def set_point(folder):
    point = rows(folder, "point")
    require(len(point) == 4, "All distinct point recordings required")
    controls = [case["candidate"] for row in point for case in row["cases"] if case["known_extent_ticks"] is None]
    require(len(controls) == 24 and sum(bool(case["measured"]) for case in controls) >= 18, "Insufficient ordinary point coverage")
    value = max((case["best"]["score"] for case in controls if case["best"]), default=0)
    write(folder / "point.json", {"value": value + .025, "ordinary": len(controls),
          "measured": sum(bool(case["measured"]) for case in controls),
          "journal_sha256": sha(folder / "point.jsonl"), "head_sha256": sha(folder / "head.json")})


def qualifies(location, interval, delay):
    lag = location["relative_lag_ticks"]
    return previous.overlap(location["extent_ticks"], interval) >= MIN_IOU and lag is not None and abs(lag - delay) <= 2


def readout(folder):
    held, point, runtime = rows(folder, "held"), read(folder / "point.json"), read(folder / "runtime.json")
    require(len(held) == 16 and runtime["held_sha256"] == sha(folder / "held.jsonl")
            and runtime["point_sha256"] == sha(folder / "point.json"), "Complete sealed held journal required")
    totals, strata, groups, preserved = {}, {}, set(), True
    for method in ("current", "candidate"):
        strata[method] = {name: Counter() for name in CASES}
        for row in held:
            require([case["case"] for case in row["cases"]] == list(CASES), "Held case order differs")
            for case in row["cases"]:
                result, truth, delay = case[method], case["known_extent_ticks"], case["known_delay_ticks"]
                if method == "candidate":
                    measured, best = bool(result["measured"]), result["best"]
                    flags = [best] if best is not None and best["score"] > point["value"] else []
                else:
                    measured, flags = bool(result["measured_windows"]), result["ranges"]
                correct = bool(truth and any(qualifies(flag, truth, delay) for flag in flags))
                if method == "candidate" and truth:
                    if correct:
                        groups.add(row["group"])
                    preserved &= correct or not any(qualifies(flag, truth, delay) for flag in case["current"]["ranges"])
                strata[method][case["case"]].update({"views": 1, "measured": measured, "flagged": bool(flags),
                    "correct": correct, "misplaced_or_wrong_lag": bool(truth and flags and not correct)})
        totals[method] = {label: dict(sum((strata[method][name] for name in names), Counter()))
                          for label, names in (("ordinary", CASES[:6]), ("edited", CASES[6:]))}
    new, current = totals["candidate"], totals["current"]
    gate = {"adds_correct": new["edited"].get("correct", 0) >= current["edited"].get("correct", 0) + 4,
            "held_source_groups": len(groups) >= 3, "preserves_current": preserved,
            "ordinary_flags": new["ordinary"].get("flagged", 0) <= current["ordinary"].get("flagged", 0) + 1,
            "coverage": all(r.get("measured", 0) >= .75*r["views"] for r in new.values())}
    write(folder / "aggregate.json", {"advance": all(gate.values()), "criteria": gate, "qualified_groups": len(groups),
          "by_condition": strata, "counts": totals, "runtime": runtime, "point": point, "frozen_method": METHOD})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "point", "calibrate", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--plan", type=Path, default=grid.PLAN)
    args = parser.parse_args()
    pairs = frozen(args.folder, args.plan, create=args.phase == "freeze")
    if args.phase == "freeze":
        return
    if args.phase in {"calibrate", "report"}:
        {"calibrate": set_point, "report": readout}[args.phase](args.folder)
        return
    cv2.setNumThreads(1)
    detector = load_yunet(previous.common.FACE_FILE)
    if args.phase == "fit":
        fit(args.folder, pairs, detector)
    else:
        score(args.folder, pairs, detector, args.phase)


if __name__ == "__main__":
    main()
