"""Replay source, filter, admission and decision arithmetic without application imports."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from scipy.fft import irfft, rfft

SAMPLES, WIDTH, STEP = 48_000, 512, 128
PAD = WIDTH // 2
WINDOW = .5 - .5 * np.cos(2 * np.pi * np.arange(WIDTH) / WIDTH)
CLOCK = np.arange(SAMPLES // STEP + 1) * STEP
CENTER = np.arange(WIDTH)
DELAYS = 6 + np.arange(10)


def sha(value):
    if isinstance(value, Path):
        value = value.read_bytes()
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decoded(raw):
    require(len(raw) == SAMPLES * 2, "A mono waveform does not cover its exact clock")
    return np.frombuffer(raw, "<i2").astype(np.float64)


def analyze(pcm):
    padded = np.pad(pcm, PAD)
    return rfft(padded[CLOCK[:, None] + CENTER] * WINDOW).T / WINDOW.sum()


def predictors(spec):
    clocks = np.arange(DELAYS[-1], len(CLOCK))
    return spec[:, clocks], spec[:, clocks[:, None] - DELAYS]


def predict(spec):
    y, delayed = predictors(spec)
    adjoint = delayed.conj().swapaxes(1, 2)
    residual = y
    for _ in range(3):
        padded = np.pad(abs(residual)**2, [(0, 0), (4, 4)], mode="edge")
        runs = np.concatenate((np.zeros((len(spec), 1)), np.cumsum(padded, axis=1)), axis=1)
        variances = (runs[:, 9:] - runs[:, :-9]) / 9
        variance = np.maximum(variances, np.maximum(.01 * np.mean(abs(spec)**2, axis=1)[:, None], 1e-10))
        weighted = adjoint / variance[:, None]
        gram = weighted @ delayed
        ridge = np.real(np.trace(gram, axis1=1, axis2=2)) * .0001 + 1e-10
        coefficients = np.linalg.solve(gram + ridge[:, None, None] * np.eye(10),
                                       weighted @ y[:, :, None])
        residual = y - (delayed @ coefficients)[..., 0]
    coefficients[:5] = 0
    coefficients[209:] = 0
    return coefficients


def transform(spec, coefficients):
    target, delayed = predictors(spec)
    filtered = spec.copy()
    filtered[:, DELAYS[-1]:] = target - .8 * (delayed @ coefficients)[..., 0]
    frames = irfft(filtered.T) * (WINDOW * WINDOW.sum())
    raw = np.zeros(SAMPLES + 2 * PAD)
    weights = np.zeros_like(raw)
    for begin, frame in zip(CLOCK, frames, strict=True):
        raw[begin:begin + WIDTH] += frame
        weights[begin:begin + WIDTH] += WINDOW**2
    return raw[PAD:-PAD] / weights[PAD:-PAD]


def decision(raw, calibration):
    if raw is None:
        return "unassessed", None
    require(type(raw) is float and math.isfinite(raw) and 0 <= raw <= 1, "Invalid raw response")
    clipped = min(1 - 1e-6, max(1e-6, raw))
    logit = math.log(clipped / (1 - clipped))
    parameters = calibration["parameters"]
    p = 1 / (1 + math.exp(-(parameters["slope"] * logit + parameters["intercept"])))
    labels = []
    for label, score in (("0", p), ("1", 1 - p)):
        guard = calibration["guards"][label]
        if (1 + sum(limit is None or limit >= score for limit in guard)) / (1 + len(guard)) > calibration["alpha"]:
            labels.append(label)
    return ("ai" if labels == ["1"] and p >= .8 else
            "original" if labels == ["0"] and p <= .2 + 1e-12 else "inconclusive"), logit


def verify_native_intervals(harmonic_run, annotations, lock, rows):
    old_roster = harmonic_run / "roster.json"
    old_freeze = harmonic_run / "freeze.json"
    require(sha(annotations) == lock["annotations_sha256"] and
            sha(old_freeze) == lock["earlier_room_cut_freeze_sha256"] and
            sha(old_roster) == read(old_freeze)["roster_sha256"], "Prior room offsets or annotations differ")
    prior = read(old_roster)
    checked = 0
    with ZipFile(annotations) as archive:
        for row in rows:
            if row["role"] != "transfer" or row["treatment"] != "native":
                continue
            room, record = row["room"], row["source"]
            start = record["source_start_sample"]
            require(all(abs(start - r["source_start_sample"]) >= SAMPLES for r in prior
                        if r.get("room") == room and "source_start_sample" in r), "A transfer cut overlaps an earlier cut")
            occupancy = np.zeros((4, 300), dtype=bool)
            first = start // 160
            for i, speaker in enumerate("ABCD"):
                for word in ElementTree.fromstring(archive.read(f"words/{room}.{speaker}.words.xml")):
                    a, b = word.get("starttime"), word.get("endtime")
                    if word.tag != "w" or word.get("punc") or a is None or b is None:
                        continue
                    lo, hi = round(100 * float(a)) - first, round(100 * float(b)) - first
                    if max(lo, 0) < min(hi, 300):
                        occupancy[i, max(lo, 0):min(hi, 300)] = True
            overlaps = occupancy.sum(axis=0) >= 2
            edges = np.flatnonzero(np.r_[False, overlaps] != np.r_[overlaps, False]).tolist()
            intervals = [[left * 160, right * 160] for left, right in zip(edges[::2], edges[1::2], strict=True)]
            require(start % 160 == 0 and record["source_end_sample"] == start + SAMPLES and
                    record["time_base"] == "1/16000" and intervals == record["annotated_overlap"] and
                    occupancy.any(axis=0).sum() >= 245 and
                    (overlaps.sum() <= 10 if row["mode"] == "sequential" else overlaps.sum() >= 120),
                    "A native case does not match its annotation-derived interference condition")
            checked += 1
    return checked


def verify_sources(folder, archived, ami, transfer_manifest, lock, rows):
    require(all(sha(archived / name) == lock["archive"][key] for name, key in (
        ("aggregate.json", "aggregate_sha256"), ("freeze.json", "freeze_sha256"), ("roster.json", "roster_sha256"),
        ("gate.json", "gate_sha256"), ("scores.json", "scores_sha256"),
    )), "Prior panel artifacts differ")
    prior_roster = read(archived / "roster.json")
    prior_scores = read(archived / "scores.json")["rows"]
    source, group_rows, backgrounds = {}, {}, {}
    for ordinal, row in enumerate(rows):
        blob = (folder / "pcm" / f"{ordinal:03}.pcm").read_bytes()
        require(row["id"] == ordinal and sha(blob) == row["sha256"], "Roster and exact PCM differ")
        source[ordinal] = decoded(blob)
        if row["role"] == "retrospective":
            require(row["sha256"] == prior_roster[ordinal]["pcm_sha256"] and row["baseline"] == prior_scores[ordinal].get("mono")
                    and blob == (archived / "pcm" / f"{ordinal:03}.pcm").read_bytes(), "Archived waveform or response differs")
        elif row["treatment"] == "native":
            path = ami / f"{row['room']}.Array1-01.wav"
            require(sha(path) == lock["rooms"][row["room"]]["sha256"], "Native microphone source differs")
            with wave.open(str(path), "rb") as stream:
                require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) == (1, 2, 16000),
                        "Unexpected native sample clock")
                stream.setpos(row["source"]["source_start_sample"])
                require(stream.readframes(SAMPLES) == blob, "Native extraction was not sample-identical")
            backgrounds[ordinal] = source[ordinal]
        else:
            group_rows.setdefault(row["group"], []).append(row)
    require(sha(transfer_manifest) == lock["transfer_source_manifest_sha256"], "Transfer foreground manifest differs")
    candidates = defaultdict(list)
    for row in read(transfer_manifest)["foregrounds"]:
        if row["status"] == "qualified":
            candidates[row["group_id"]].append(row)
    require(len(group_rows) == 16 and len(backgrounds) == 4, "Unexpected transfer source counts")
    for group, cohort in group_rows.items():
        clean, *variants = cohort
        require(len(variants) == 4 and clean["treatment"] == "clean" and
                {row["native_id"] for row in variants} == backgrounds.keys(), "A source lost a matched counterpart")
        match = False
        for seed in candidates[group]:
            wav = Path(seed["file"])
            require(sha(wav) == seed["wav_sha256"], "A qualified foreground changed")
            with wave.open(str(wav), "rb") as stream:
                voice = decoded(stream.readframes(SAMPLES))
            noise = [backgrounds[row["native_id"]] for row in variants]
            scaled = [raw * (np.linalg.norm(voice) / np.linalg.norm(raw)) * 10**.25 for raw in noise]
            gain = min(1., 31_129 / max(np.max(abs(voice) + abs(raw)) for raw in scaled))
            clean_pcm = np.rint(voice * gain)
            if not np.array_equal(clean_pcm, source[clean["id"]]):
                continue
            match = True
            for row, raw in zip(variants, scaled, strict=True):
                residual = np.rint(raw * gain)
                require(row["clean_id"] == clean["id"] and np.array_equal(clean_pcm + residual, source[row["id"]])
                        and abs(20 * math.log10(np.linalg.norm(clean_pcm) / np.linalg.norm(residual)) - row["snr_db"]) < 1e-10,
                        "A matched transfer mixture does not reconstruct")
            break
        require(match, "A transfer clean input does not match any qualified source")
    return source


def audit(folder, archived, harmonic_run, ami, annotations, transfer_manifest, calibration_path):
    lock, rows, scores, aggregate = (read(folder / f"{name}.json") for name in ("freeze", "roster", "scores", "aggregate"))
    require(len(rows) == len(scores["rows"]) == 185 and sha(folder / "roster.json") == lock["roster_sha256"]
            and sha(folder / "freeze.json") == scores["freeze_sha256"] == aggregate["freeze_sha256"]
            and sha(folder / "scores.json") == aggregate["scores_sha256"] and sha(calibration_path) == lock["calibration_sha256"],
            "Evaluation artifacts changed")
    calibration = read(calibration_path)
    source = verify_sources(folder, archived, ami, transfer_manifest, lock, rows)
    native_intervals = verify_native_intervals(harmonic_run, annotations, lock, rows)
    cells, effects, snr, voices = defaultdict(Counter), defaultdict(Counter), defaultdict(list), defaultdict(list)
    groups, rooms, families = defaultdict(set), defaultdict(set), defaultdict(set)
    calls = Counter()
    for row, observed in zip(rows, scores["rows"], strict=True):
        i = row["id"]
        require((observed["id"], observed["source_sha256"]) == (i, row["sha256"]), "Evidence source binding differs")
        a, b = observed["incumbent"], observed["candidate"]
        direct, log_odds = decision(a["raw"], calibration)
        side, side_odds = decision(b["raw"], calibration)
        if direct == "original" and (a["probe_status"] == "unavailable" or a["stability_log_odds_change"] > math.log(2)):
            direct = "inconclusive"
        require(direct == a["decision"] and side == b["decision"], "The recorded decisions do not reproduce")
        if row["role"] == "transfer":
            calls.update(incumbent_activity=94, classifier=int(a["raw"] is not None) + int(a["probe_status"] == "computed"))
        spec = analyze(source[i])
        coef = predict(spec)
        conditioned = transform(spec, coef)
        if "output_sha256" in observed:
            calls.update(derived_activity=94, classifier=int(b["raw"] is not None), exact_conditioned_waveforms=1)
            require(np.max(conditioned) < 32768 and np.min(conditioned) > -32769
                    and sha(np.rint(conditioned).astype("<i2").tobytes()) == observed["output_sha256"],
                    "Independent filter did not reproduce the exact classifier input")
        elif direct in {"ai", "inconclusive"}:
            require(observed["candidate"]["probe_status"] == "clipped" and
                    (np.max(np.rint(conditioned)) > 32767 or np.min(np.rint(conditioned)) < -32768), "Missing conditioned input")
        label = "genuine" if row["family"] == "genuine" else "synthetic"
        role, room, treatment = row["role"], row["room"], row["treatment"]
        key = "/".join((role, room or "-", treatment, row["mode"] or "-", label))
        withheld = direct == "ai" and side in {"original", "inconclusive"} and log_odds - side_odds > math.log(2)
        recovered = direct == "inconclusive" and side == "ai"
        cells[key].update(planned=1, suppressed=int(withheld), rescued=int(recovered),
                          **{f"incumbent_{direct}": 1, f"conditioned_{side}": 1})
        for name, event in (("suppression", withheld), ("rescue", recovered)):
            if event:
                effects[f"{role}/{name}"][f"{treatment}_{label}"] += 1
                if treatment == "mixed":
                    groups[role, name, label].add(row["group"])
                    rooms[role, name, label].add(room)
                    families[role, name, label].add(row["family"])
        if treatment == "mixed":
            clean = source[row["clean_id"]]
            noise = source[i] - clean
            a_out, b_out = transform(analyze(clean), coef), transform(analyze(noise), coef)
            level = 20 * math.log10(np.linalg.norm(a_out) / np.linalg.norm(clean))
            gain = level - 20 * math.log10(np.linalg.norm(b_out) / np.linalg.norm(noise))
            acoustic = observed["acoustic"]
            require(abs(level - acoustic["voice_level_change_db"]) < 1e-9 and abs(gain - acoustic["snr_gain_db"]) < 1e-9
                    and np.max(abs(a_out + b_out - conditioned)) < 1e-7, "Acoustic component replay differs")
            voices[key].append(level)
            snr[key].append(gain)
            calls["acoustic_replays"] += 1
    require(cells == aggregate["cells"] and calls["classifier"] == scores["classifier_forwards"]
            and calls["incumbent_activity"] + calls["derived_activity"] == scores["speech_activity_forwards"],
            "Case or inference totals differ")
    for key, values in (*snr.items(), *voices.items()):
        label = "median_snr_gain_db" if values is snr.get(key) else "median_voice_level_change_db"
        require(abs(median(values) - aggregate[label][key]) < 1e-9, "Aggregate acoustic median differs")
    coverage = all(cell["planned"] - cell["incumbent_unassessed"] >= 6
                   for key, cell in cells.items() if key.startswith("transfer/") and "/mixed/" in key and key.endswith("/genuine"))
    for role in ("retrospective", "transfer"):
        for action, target in (("suppression", "genuine"), ("rescue", "synthetic")):
            observed = effects[f"{role}/{action}"]
            require(observed == aggregate["effects"].get(f"{role}/{action}", {}), "Paired transition totals differ")
            if role == "transfer":
                counts = groups[role, action, target]
                good = (coverage and len(rooms[role, action, target]) == 2 and len(counts) >= 3
                        and not any(value for key, value in observed.items() if not key.startswith("mixed_")))
                if action == "suppression":
                    good &= observed["mixed_synthetic"] <= 2
                else:
                    good &= observed["mixed_genuine"] <= 1 and len(families[role, action, target]) >= 2
                require(good == aggregate["advance"][action], "Independent gate differs")
    require(coverage == aggregate["transfer_genuine_coverage_met"] and
            {"/".join(key): len(value) for key, value in groups.items()} == aggregate["affected_groups"],
            "Aggregate coverage or group count differs")
    return {"verified_views": len(rows), "verified_transfer_mixtures": 64, "verified_native_intervals": native_intervals, **calls,
            "condition_cells": len(cells), "policy_advances": aggregate["advance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("folder", "archived", "harmonic_run", "ami", "annotations", "transfer_manifest", "calibration"):
        parser.add_argument(argument, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder, args.archived, args.harmonic_run, args.ami, args.annotations, args.transfer_manifest, args.calibration), sort_keys=True))


if __name__ == "__main__":
    main()
