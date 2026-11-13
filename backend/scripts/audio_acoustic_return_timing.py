"""Measure acoustic return at candidate short speech-edit intervals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median, quantiles

from scripts import audio_source_isolation_diagnostic as prior
from scripts.audio_trimmed_mel_normalization import PARENT_SHA, parent_rows, read_parent

WIDTH, STARTS = prior.GRIDS["dense"]
FLANK, GAP = 5120, 1280
SCHEMA = "chipmunk.audio-acoustic-return-timing/1"
METHOD = {
    "sources": "Repeat the original 40 development and 52 held mixtures from the isolation diagnostic. Only submitted PCM and previously committed original dense voice scores enter features. This is a new analysis on previously exposed audio.",
    "spectral_profile": "512-sample Hann frames advance by 128 samples. Sum power into 24 logarithmic frequency bands from 250 to 6400 Hz; normalize each active frame to unit band mass, take componentwise segment medians and renormalize. Use the middle 840 ms of the one-second candidate and two 320 ms flanks separated from each candidate edge by 80 ms.",
    "contrasts": "Compute base-2 Jensen-Shannon divergence among each center and flank profile, and between the two flanks. Compare the weaker center-to-flank divergence with the same value minus between-flank divergence. A positive return suggests one transient spectral change, not AI origin.",
    "timing_coverage": "Use only the existing admitted one-second positions with both flanks and supported nonzero frames. Missing flanks do not imply an unchanged recording.",
    "combination": "Standardize dense voice and acoustic return scores with per-feature median and IQR from ordinary development windows. Use the voice score plus half the nonnegative acoustic score capped at four IQR units. Keep the voice term when acoustic timing is unassessed. Retain the same one-second extent.",
    "points": "Fix each new operating point at the maximum ordinary development score for its own readout. Preserve archived dense and original two-second points. Report fixed and retrospective matched false-group counts. Count useful at 50 percent coverage and 0.30 IoU.",
    "advancement": "At least two new combined locations across two generator families, keep the existing useful location, at most one new ordinary group and no isolated-room flag. No serving eligibility from this exposed cohort.",
}
ARMS = ("weaker_edge", "acoustic_return", "combined")


def profiles(pcm):
    import numpy as np

    signal = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.
    frames = np.lib.stride_tricks.sliding_window_view(signal, 512)[::128]
    centers = 256 + np.arange(len(frames)) * 128
    power = np.abs(np.fft.rfft(frames * np.hanning(512))) ** 2
    bins = np.fft.rfftfreq(512, 1 / prior.RATE)
    edges = np.geomspace(250, 6400, 25)
    # Some low log bands are narrower than one DFT bin.
    bands = np.stack([power[:, (bins >= a) & (bins < b)].sum(axis=1)
                      for a, b in zip(edges[:-1], edges[1:], strict=True)], axis=1)
    mass = bands.sum(axis=1)
    return centers, bands / np.maximum(mass[:, None], 1e-20), mass > 1e-12


def readings(pcm, admitted):
    import numpy as np

    centers, values, active = profiles(pcm)

    def profile(start, end):
        subset = values[(centers >= start) & (centers < end) & active]
        if len(subset) < 16:
            return None
        middle = np.median(subset, axis=0)
        return middle / middle.sum() if middle.sum() > 0 else None

    def divergence(first, second):
        center = (first + second) * .5
        return float(sum(np.sum(part * np.log2(np.maximum(part, 1e-20) / np.maximum(center, 1e-20)))
                         for part in (first, second)) * .5)

    result = []
    for start, speech in zip(STARTS, admitted, strict=True):
        stop = start + WIDTH
        if speech is None or start < FLANK + GAP or stop + GAP + FLANK > prior.SAMPLES:
            result.append(None)
            continue
        left = profile(start - GAP - FLANK, start - GAP)
        middle = profile(start + GAP, stop - GAP)
        right = profile(stop + GAP, stop + GAP + FLANK)
        if left is None or middle is None or right is None:
            result.append(None)
            continue
        result.append([divergence(left, middle), divergence(middle, right), divergence(left, right)])
    return result


def dense(row):
    return [window[0]["score"] for window in row["measurements"]["mixture"]["dense"]]


def scales(rows, evidence):
    controls = [row for row in rows if row["truth"] is None]
    voice = [value for row in controls for value in dense(row) if value is not None]
    acoustics = [min(values[:2]) - values[2] for row in controls for values in evidence[row["id"]] if values is not None]

    def scale(values):
        low, _, high = quantiles(values, n=4, method="inclusive")
        prior.check(high - low > 1e-8, "Control scores lack a usable spread")
        return [median(values), high - low]

    return {"voice": scale(voice), "timing": scale(acoustics)}


def arms(rows, evidence, factors):
    voice_center, voice_scale = factors["voice"]
    timing_center, timing_scale = factors["timing"]
    by_arm = {name: [] for name in ARMS}
    for row in rows:
        native = row["measurements"]["mixture"]["dense"]
        derived = {name: [] for name in ARMS}
        for (item,), timing in zip(native, evidence[row["id"]], strict=True):
            model = item["score"]
            edge = min(timing[:2]) if timing is not None else None
            contrast = edge - timing[2] if edge is not None else None
            bonus = 0 if contrast is None else max(0., min(4., (contrast - timing_center) / timing_scale))
            combined = None if model is None else (model - voice_center) / voice_scale + .5 * bonus
            for name, value in zip(ARMS, (edge, contrast, combined), strict=True):
                derived[name].append([{**item, "score": value}])
        for name, windows in derived.items():
            by_arm[name].append({**row, "measurements": {"mixture": {"dense": windows}}})
    return by_arm


def fit(rows, evidence):
    factors = scales(rows, evidence)
    thresholds = {}
    for name, bank in arms(rows, evidence, factors).items():
        ordinary = [prior.best(row, "mixture", "dense") for row in bank if row["truth"] is None]
        thresholds[name] = max(value[0] for value in ordinary if value is not None)
    return {"factors": factors, "thresholds": thresholds}


def evaluate(parent, calibration, observations):
    held = parent["assessment"]
    routes = {name: (rows, "dense", calibration["thresholds"][name])
              for name, rows in arms(held, observations, calibration["factors"]).items()}
    routes["dense"] = (held, "dense", parent["points"]["mixture/dense"]["point"])
    routes["stock"] = (held, "stock", parent["freeze"]["prior_point"])
    results, equal = {}, {}
    for name, (rows, grid, point) in routes.items():
        results[name] = prior.result(rows, "mixture", grid, point)
        groups = {}
        for row in rows:
            score = prior.best(row, "mixture", grid)
            if row["truth"] is None and score is not None:
                group = row["group"]
                groups[group] = max(score[0], groups.get(group, -math.inf))
        equal[name] = [prior.result(rows, "mixture", grid, cut)[0] for cut in sorted(groups.values(), reverse=True)[:4]]
    current, proposed = results["stock"], results["combined"]
    gained, lost = proposed[1] - current[1], current[1] - proposed[1]
    new_groups = proposed[2] - current[2]
    families = {row["family"] for row in held if row["id"] in gained}
    gate = {"gained": len(gained), "lost": len(lost), "gain_families": len(families), "new_false_groups": len(new_groups)}
    gate["follow_up"] = len(gained) >= 2 and len(families) >= 2 and not lost and len(new_groups) <= 1 and not proposed[0].get("isolated_room_flags", 0)
    return {"fixed": {name: value[0] for name, value in results.items()}, "equal_false_groups": equal, "gate": gate}


def measure(roster, rows):
    expected = {row["id"]: row for row in rows}
    observed = {}
    for source in roster:
        ident = prior.digest(source["pcm_sha256"].encode())
        row = expected.get(ident)
        if row is not None:
            raw = next(prior.source_data(prior.DATA / source["bank"], source))
            observed[ident] = readings(raw, dense(row))
    prior.check(expected.keys() == observed.keys(), "Missing original mixture")
    return observed


def verify(result, parent):
    freeze = result["freeze"]
    prior.check(freeze["schema"] == SCHEMA and freeze["method"] == METHOD and freeze["parent_sha256"] == PARENT_SHA,
                "Frozen comparison differs")
    for phase in ("development", "assessment"):
        expected = parent_rows(parent, phase)
        observed = result[phase]
        prior.check(expected.keys() == observed.keys(), "Incomplete receipt")
        for ident, values in observed.items():
            for start, original, item in zip(STARTS, dense(expected[ident]), values, strict=True):
                if item is not None:
                    prior.check(original is not None and FLANK + GAP <= start <= prior.SAMPLES - WIDTH - FLANK - GAP and
                                len(item) == 3 and all(type(v) in (int, float) and 0 <= v <= 1 for v in item),
                                "Invalid source clock or distance")
    fitted = fit(parent["development"], result["development"])
    summary = evaluate(parent, fitted, result["assessment"])
    prior.check(fitted == result["calibration"] and summary == result["summary"], "Outcome arithmetic differs")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "assessment", "review"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    parent = read_parent()
    if args.phase == "review":
        print(json.dumps(verify(prior.read(args.run), parent), sort_keys=True))
        return
    roster, inputs = prior.sources()
    lock = {"schema": SCHEMA, "method": METHOD, "parent_sha256": PARENT_SHA, "source_inputs": inputs,
            "runner_sha256": prior.path_sha(Path(__file__))}
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700, exist_ok=True)
        prior.write(args.run / "freeze.json", lock)
        return
    prior.check(prior.read(args.run / "freeze.json") == lock, "Frozen sources or method changed")
    point_path = args.run / "selection.json"
    if args.phase == "development":
        evidence = measure(roster, parent["development"])
        calibration = fit(parent["development"], evidence)
        prior.write(point_path, {"evidence": evidence, "calibration": calibration})
        return
    selection = prior.read(point_path)
    prior.check(fit(parent["development"], selection["evidence"]) == selection["calibration"], "Development point changed")
    held = measure(roster, parent["assessment"])
    summary = evaluate(parent, selection["calibration"], held)
    receipt = {"freeze": lock, "development": selection["evidence"], "assessment": held,
               "calibration": selection["calibration"], "summary": summary}
    verify(receipt, parent)
    prior.write(args.run / "public.json", receipt)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
