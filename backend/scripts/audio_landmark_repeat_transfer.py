"""Compare within-recording spectral landmarks with existing source-bound repeat checks."""
from __future__ import annotations

import argparse
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal.windows import hann

from app.detectors.audio.decode import decode_audio, wav_bytes
from app.evidence import audio_exact_repetition, audio_similarity, audio_waveform_similarity
from scripts.audio_mono_dereverb_transfer import read, sha, write
from scripts.audio_phase_corroboration import parent_receipts, sample_bytes

HOP = 80
FFT = 512
RATE = 16_000
LENGTH = 3 * RATE
MIN_RUN = 2400
METHOD = {
    "spectrum": "512-sample Hann; 80-sample hop; salient local peaks between 187.5 Hz and 4.5 kHz.",
    "hash": "Two quantized peak frequencies plus the elapsed frame count, with a one-frame tolerance.",
    "verify": "Vote across distinct anchor times, with anchors in two bands and across at least three 50 ms cells.",
    "rank": "Distinct anchor count plus frequency-band and temporal coverage bonuses.",
    "selection": "Cutoff above the second highest development source-level false maximum across deliveries.",
    "materialization": "500 ms or 650 ms foreground copy with 20 ms tapers; clean and -5 dB room mixtures; PCM and 32 kb/s Opus.",
    "baseline": "Exact and two-band ranges, with waveform fallback when both are empty.",
    "truth": "Both spans overlap at least 150 ms of the two copy interiors, with lag error at most 10 ms.",
    "gate": "At least 8 additional useful locations across four held source groups, <=2 extra held false views "
            "from <=1 group, and no flags on native room controls.",
}


def estimate(data):
    pcm = np.frombuffer(data, "<i2").astype(float)
    frames = np.lib.stride_tricks.sliding_window_view(pcm, FFT)[::HOP]
    spec = np.abs(np.fft.rfft(frames * hann(FFT, sym=False)))[:, 6:145]
    db = 20 * np.log10(np.maximum(1, spec))
    local_max = (db == maximum_filter(db, size=(5, 5), mode="nearest")) & (db >= np.max(db, axis=1)[:, None] - 21)
    local_max &= db > 50
    coordinates = [tuple(map(int, pair)) for pair in np.argwhere(local_max)]
    landmarks = defaultdict(list)
    for i, (start, f1) in enumerate(coordinates):
        eligible = []
        for j in range(i+1, len(coordinates)):
            end, f2 = coordinates[j]
            delta = end - start
            if delta > 24:
                break
            if delta >= 5 and abs(f2-f1) >= 4:
                eligible.append((float(db[end, f2]), delta, f2))
        for _, delta, f2 in sorted(eligible, reverse=True)[:5]:
            landmarks[f1//2, f2//2, delta].append((start, f1))
    votes = defaultdict(set)
    for (f1, f2, dt), entries in landmarks.items():
        if len(entries) > 12:
            continue
        for delta in (dt - 1, dt, dt + 1):
            for a, band in entries:
                for b, _ in landmarks.get((f1, f2, delta), ()):
                    if b - a >= 38:
                        votes[b-a].add((a, band))
    suggestions = []
    for lag, direct in votes.items():
        if len(direct) < 3:
            continue
        support = set.union(direct, votes.get(lag-1, set()), votes.get(lag+1, set()))
        anchors = sorted({t for t, _ in support})
        partitions = np.split(anchors, np.flatnonzero(np.diff(anchors) > 18) + 1)
        for part in partitions:
            if len(part) < 4 or part[-1] - part[0] < 22:
                continue
            band_count = len({f//24 for t, f in support if part[0] <= t <= part[-1]})
            time_cells = len({int(t)//10 for t in part})
            if band_count < 2 or time_cells < 3:
                continue
            start = int(part[0]) * HOP
            stop = min(int(part[-1]) * HOP + FFT + 12*HOP, start + lag*HOP, LENGTH-lag*HOP)
            if stop - start >= MIN_RUN:
                suggestions.append((len(part) + 2*band_count + time_cells, start, stop, lag*HOP))
    return max(suggestions, default=None)


def build_copy(source, duration):
    src = np.frombuffer(source, "<i2").astype(float)
    starts = np.arange(1600, 9601, 1600)
    anchor = max(map(int, starts), key=lambda i: np.linalg.norm(src[i+320:i+duration-320]))
    target = 28_800
    taper = .5-.5*np.cos(np.linspace(0, np.pi, 320))
    blend = np.r_[taper, np.ones(duration-640), taper[::-1]]
    altered = src.copy()
    altered[target:target+duration] = src[anchor:anchor+duration]*blend + src[target:target+duration]*(1-blend)
    return src, altered, [anchor+320, anchor+duration-320, target+320, target+duration-320]


def mix(genuine, edited, room):
    if room is None:
        return genuine, edited
    room = room * (np.linalg.norm(genuine) / np.linalg.norm(room)) * 10**.25
    scale = min(1., 30_000 / max(np.max(np.abs(genuine)+np.abs(room)), np.max(np.abs(edited)+np.abs(room))))
    return (genuine + room) * scale, (edited + room) * scale


def deliver(folder, data, codec):
    if codec == "pcm":
        return data
    path, source = folder / "current.opus", folder / "current.wav"
    source.write_bytes(wav_bytes(np.frombuffer(data, "<i2").astype(np.float32)/32768))
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
                    "-ar", "48000", "-c:a", "libopus", "-b:a", "32k", str(path)], check=True, timeout=30)
    decoded = decode_audio(path, source_sha256=sha(path), max_seconds=3)
    if len(decoded.pcm) != LENGTH:
        raise ValueError("Decoded delivery changed the exact sample count")
    return np.rint(decoded.pcm * 32768).astype("<i2").tobytes()


def prepare(folder, parent):
    rows, _, proof = parent_receipts(parent)
    recipes = {}
    for role in ("archived", "transfer"):
        candidates = [r for r in rows if r["role"] == role and r["placement"] == "native"]
        scenes = {}
        for mode in ("sequential", "simultaneous"):
            by_room = {}
            for row in candidates:
                if row["mode"] == mode:
                    by_room.setdefault(row["room"], row["id"])
            selected_room = sorted(by_room)[int(mode == "simultaneous")]
            scenes[mode] = by_room[selected_room]
        recipes[role] = {"sources": [r["id"] for r in rows if r["role"] == role and r["placement"] == "clean"],
                         "scenes": scenes, "native": [r["id"] for r in candidates]}
    groups = [{rows[i]["group"] for i in plan["sources"]} for plan in recipes.values()]
    if tuple(map(len, groups)) != (17, 16) or groups[0] & groups[1]:
        raise ValueError("Expected separated original foreground groups")
    folder.mkdir(parents=True)
    write(folder / "freeze.json", {"schema": "chipmunk.audio-landmark-repeat/1", "method": METHOD,
          "code_sha256": sha(Path(__file__)), "parent": proof, "recipes": recipes})


def run(folder, parent, role):
    rows, _, proof = parent_receipts(parent)
    lock = read(folder / "freeze.json")
    if lock["parent"] != proof or lock["code_sha256"] != sha(Path(__file__)):
        raise ValueError("Candidate code or sources differ from the fixed plan")
    plan, cases = lock["recipes"][role], []
    if role == "transfer":
        points = read(folder / "points.json")
        if points["freeze_sha256"] != sha(folder / "freeze.json") or points["development_sha256"] != sha(folder / "archived.json"):
            raise ValueError("Transfer points lack exact development binding")

    def measure(data, *, source, scenario, codec, truth):
        nominee = estimate(data)
        exact = audio_exact_repetition.find_repeated_spans(data)
        two_band = audio_similarity.find_similar_spans(data)
        base = (*exact.spans, *two_band.spans) or audio_waveform_similarity.find_similar_spans(data).spans
        cases.append({"source": source, "scenario": scenario, "codec": codec, "truth": truth,
                      "sha256": sha(data), "candidate": nominee,
                      "baseline": [[*span.first, *span.second] for span in base]})

    for ordinal, source in enumerate(plan["sources"]):
        genuine, edited, truth = build_copy(sample_bytes(parent, rows[source]), 8000 if ordinal % 2 else 10400)
        for name, i in {"clean": None, **plan["scenes"]}.items():
            room = None if i is None else np.frombuffer(sample_bytes(parent, rows[i]), "<i2").astype(float)
            a, b = mix(genuine, edited, room)
            for codec in ("pcm", "opus"):
                measure(deliver(folder, np.rint(a).astype("<i2").tobytes(), codec), source=source,
                        scenario=name, codec=codec, truth=None)
                measure(deliver(folder, np.rint(b).astype("<i2").tobytes(), codec), source=source,
                        scenario=name, codec=codec, truth=truth)
    for source in plan["native"]:
        for codec in ("pcm", "opus"):
            measure(deliver(folder, sample_bytes(parent, rows[source]), codec), source=source,
                    scenario="native", codec=codec, truth=None)
    write(folder / f"{role}.json", {"freeze_sha256": sha(folder / "freeze.json"), "cases": cases})
    if role == "archived":
        group_max = defaultdict(int)
        for case in cases:
            if case["truth"] is None and case["candidate"] and case["scenario"] != "native":
                group_max[case["source"]] = max(group_max[case["source"]], case["candidate"][0])
        high = sorted([*group_max.values(), 0, 0], reverse=True)
        write(folder / "points.json", {"freeze_sha256": sha(folder / "freeze.json"),
              "development_sha256": sha(folder / "archived.json"), "minimum": high[1] + 1})
    for name in ("current.opus", "current.wav"):
        (folder / name).unlink(missing_ok=True)
    print(role, len(cases), flush=True)


def relevant(spans, truth):
    a, b, c, d = truth
    return any(min(y, b)-max(x, a) >= MIN_RUN and min(v, d)-max(u, c) >= MIN_RUN
               and abs((u-x)-(c-a)) <= 2*HOP for x, y, u, v in spans)


def summarize(folder):
    lock = read(folder / "freeze.json")
    point = read(folder / "points.json")
    fit, held = (read(folder / f"{r}.json") for r in ("archived", "transfer"))
    freeze_sha = sha(folder / "freeze.json")
    if any(r["freeze_sha256"] != freeze_sha for r in (point, fit, held)):
        raise ValueError("Journal and point do not refer to this comparison")
    groups, false_groups, cells = set(), set(), defaultdict(Counter)
    for case in held["cases"]:
        key = f"{case['scenario']}/{case['codec']}"
        truth, base, candidate = (case[k] for k in ("truth", "baseline", "candidate"))
        spans = []
        if candidate and candidate[0] >= point["minimum"]:
            _, a, b, lag = candidate
            spans.append((a, b, a+lag, b+lag))
        cell = cells[key]
        if truth:
            base_hit, hit = relevant(base, truth), relevant(spans, truth)
            added = hit and not base_hit
            cell.update(edits=1, baseline_localized=base_hit, candidate_localized=hit, additional=added)
            if added:
                groups.add(case["source"])
        else:
            extra = bool(spans and not base)
            cell.update(controls=1, baseline_flagged=bool(base), candidate_flagged=bool(spans), extra_false=extra)
            if extra and case["scenario"] != "native":
                false_groups.add(case["source"])
    adds = sum(c["additional"] for c in cells.values())
    false = sum(c["extra_false"] for c in cells.values())
    native = sum(c["candidate_flagged"] for k, c in cells.items() if k.startswith("native/"))
    result = {"schema": lock["schema"], "method": METHOD, "freeze_sha256": freeze_sha,
              "fit_sha256": sha(folder / "archived.json"), "point_sha256": sha(folder / "points.json"),
              "held_sha256": sha(folder / "transfer.json"), "views": [len(fit["cases"]), len(held["cases"])],
              "minimum": point["minimum"], "cells": cells, "extra_correct": adds, "extra_false": false,
              "added_source_groups": len(groups), "false_source_groups": len(false_groups),
              "native_flags": native, "advance": adds >= 8 and len(groups) >= 4 and false <= 2
              and len(false_groups) <= 1 and native == 0}
    write(folder / "aggregate.json", result)
    print({k: result[k] for k in ("views", "extra_correct", "extra_false", "advance")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "archived", "transfer", "summarize"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.folder, args.parent)
    elif args.stage == "summarize":
        summarize(args.folder)
    else:
        run(args.folder, args.parent, args.stage)
