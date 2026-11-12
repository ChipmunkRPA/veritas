"""Test pitch-aligned excitation measurements on exact short speech edits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import butter, sosfiltfilt

from scripts import audio_partial_residual_transfer as previous
from scripts.audio_partial_residual_room_replay import sources as meeting_sources

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PARENT = BASE / "audio-partial-residual-2026-11-12"
MEETING = BASE / "audio-overlap-window-transfer-2026-11-12-final"
RATE, CELL = 16_000, 3_200
FRAME, HOP, FFT = 1024, 320, 2048
ORDER = 12
LAGS = np.arange(40, 223)
WINDOW = np.hanning(FRAME)
HARMONICS = np.arange(1, 8)
BASELINE, CANDIDATE, ROOM_BASELINE = "incumbent_frames", "periodic_excitation", "incumbent_two_second"
SCHEMA = "chipmunk.audio-periodic-excitation-transfer/1"
METHOD = {
    "source": "Reuse exact prior four-second matched splices with real competing speech and separate no-refit six-second AMI mixtures. Underlying clips had prior project exposure; each original source role remains unchanged.",
    "frame": "Unpadded 64 ms frame at 20 ms hops on a zero-phase fourth-order Butterworth 80–4000 Hz band-pass. Remove the frame mean. At >= -50 dBFS use the best local normalized autocorrelation peak among 40–222 sample lags, with a .035 log2(lag/40) octave penalty. Require unpenalized correlation >= .55.",
    "excitation": "Apply 12th-order autocorrelation linear prediction fitted on the Hann-windowed frame after .95 pre-emphasis and .001 relative diagonal loading. Apply the fitted polynomial to the unwindowed pre-emphasis frame. Discard first 12 samples for residual features. Measure residual cycle correlation at selected lag, log crest factor and phase curvature at harmonics 2–6 using neighboring complex coefficients from a Hann-windowed 8192-point FFT and linear interpolation of complex bins.",
    "cell": "Eight features per 200 ms cell: median speech cycle correlation, residual cycle correlation, phase curvature concentration, and log crest; median absolute adjacent log2 period change, phase curvature change, residual periodicity change; and supported-frame fraction. Require >=3 admitted frames and >=2 adjacent admitted pairs. Append the difference from median supported flank cells at offsets ±2 and ±3 (zero difference when no flank qualifies).",
    "fit": "Use only matched splice triples from eight original fitting hosts in three conditions. Fit a balanced L2 ridge with target equal to exact AI overlap fraction, standardized with .03 scale floor. Fitting mass is split equally across positive-overlap and ordinary exposure. Use 16 features and penalty .10, with an unpenalized intercept.",
    "point": "Use strict excess of maximum admitted genuine track response in the original independent confidence_cal role. A track response is the highest mean over three successive 200 ms cells, with at least two qualified cells. Report no measurement separately.",
    "location": "Use the original half-open 0.6 s sample-clock span; useful requires IoU >= .30 and edit recall >= .50. Evaluate strongest-range and matched-control rankings before applying points. Sweep equal held genuine source-group false counts 0, 1 and 2 as retrospective diagnostics.",
    "room_baseline": "Retain fixed six-second 0.6 s incumbent frame flags from the earlier room replay. For score-ranked comparisons, use the earlier measured strongest admitted 2 s window at 1 or 3 seconds and its existing point. These baselines have different temporal widths; report them separately.",
    "advance": "On strong noise need at least two net new useful locations across two generator families, at most one additional genuine false source group, no new isolated-background flag and no reduction in dry useful locations. Also need a useful held AMI noisy location and no extra AMI noisy genuine false flags versus prior 0.6 s incumbent. Advance to new-source research only.",
    "limits": "A periodic component is not a speaker identifier. Overlapping voices, music and preprocessing may dominate the same measurements. No new weights, authenticity confidence or app decision.",
}
read, write, sha = previous.read, previous.write, previous.sha


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ncc(frames):
    size = frames.shape[-1]
    spectrum = np.fft.rfft(frames, FFT)
    auto = np.fft.irfft(abs(spectrum)**2, FFT)[..., :LAGS[-1]+2]
    powers = np.cumsum(np.concatenate((np.zeros((*frames.shape[:-1], 1)), frames**2), axis=-1), axis=-1)
    lags = np.arange(auto.shape[-1])
    denominator = np.sqrt(np.maximum(powers[..., size-lags] * (powers[..., -1:] - powers[..., lags]), 1e-24))
    return np.clip(auto / denominator, -1, 1)


def phase_curvature(residual, lag):
    spectrum = np.fft.rfft(residual * np.hanning(len(residual)), 8192)
    location = 8192 * HARMONICS / lag
    bins = location.astype(int)
    fraction = location - bins
    values = spectrum[bins] * (1-fraction) + spectrum[bins+1] * fraction
    amp = abs(values)
    unit = values / np.maximum(amp, 1e-12)
    closure = unit[:-2] * unit[2:] * np.conj(unit[1:-1])**2
    weight = np.minimum(np.minimum(amp[:-2], amp[1:-1]), amp[2:])
    total = float(weight.sum())
    if total <= 1e-10:
        return np.zeros_like(closure), 0.
    return closure * (weight / total), float(abs(closure @ weight) / total)


def frame_values(samples):
    band = sosfiltfilt(butter(4, [80, 4000], btype="bandpass", fs=RATE, output="sos"), samples)
    frames = np.lib.stride_tricks.sliding_window_view(band, FRAME)[::HOP]
    frames = frames - frames.mean(axis=-1, keepdims=True)
    correlations = ncc(frames)
    peak = (correlations[:, LAGS] > correlations[:, LAGS-1]) & (correlations[:, LAGS] >= correlations[:, LAGS+1])
    preferred = np.where(peak, correlations[:, LAGS] - .035 * np.log2(LAGS / LAGS[0]), -np.inf)
    lags = LAGS[preferred.argmax(axis=1)]
    period = correlations[np.arange(len(frames)), lags]
    usable = peak.any(axis=1) & (period >= .55) & (np.sqrt(np.mean(frames**2, axis=1)) >= 10 ** (-50 / 20))
    values = np.zeros((len(frames), 4))
    closure = np.zeros((len(frames), 5), dtype=np.complex128)
    for i in np.flatnonzero(usable):
        frame = frames[i]
        emphasized = np.r_[frame[0], frame[1:] - .95 * frame[:-1]]
        windowed = emphasized * WINDOW
        autocorrelation = np.array([windowed[:FRAME-k] @ windowed[k:] for k in range(ORDER+1)])
        autocorrelation[0] *= 1.001
        coefficients = solve_toeplitz(autocorrelation[:ORDER], -autocorrelation[1:])
        residual = np.convolve(emphasized, np.r_[1., coefficients], "full")[ORDER:FRAME]
        crest = math.log(max(1., np.max(abs(residual)) / max(1e-10, np.sqrt(np.mean(residual**2)))))
        closure[i], concentration = phase_curvature(residual, lags[i])
        values[i] = period[i], ncc(residual)[lags[i]], concentration, crest
    return lags, usable, values, closure


def features(raw):
    samples = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    require(len(samples) in {4*RATE, 6*RATE}, "Expected exact 4 or 6 second PCM")
    lags, usable, values, closure = frame_values(samples)
    adjacent = usable[1:] & usable[:-1]
    drift = abs(np.log2(lags[1:] / lags[:-1]))
    coupling = closure[1:] * closure[:-1].conj()
    phase_change = 1. - coupling.real.sum(axis=1) / np.maximum(abs(coupling).sum(axis=1), 1e-12)
    residual_change = abs(np.diff(values[:, 1]))
    cells = (np.arange(len(usable)) * HOP + FRAME//2) // CELL
    result = np.full((len(samples)//CELL, 8), np.nan)
    for cell in range(len(result)):
        inside = cells == cell
        pairs = inside[1:] & inside[:-1] & adjacent
        if (inside & usable).sum() >= 3 and pairs.sum() >= 2:
            result[cell] = np.r_[np.median(values[inside & usable], axis=0), np.median(drift[pairs]),
                                np.median(phase_change[pairs]), np.median(residual_change[pairs]), usable[inside].mean()]
    supported = np.isfinite(result).all(axis=1)
    delta = np.zeros_like(result)
    for cell in np.flatnonzero(supported):
        flanks = [pos for pos in (cell-3, cell-2, cell+2, cell+3) if 0 <= pos < len(result) and supported[pos]]
        if flanks:
            delta[cell] = result[cell] - np.median(result[flanks], axis=0)
    return np.concatenate((result, delta), axis=-1)


def bind(parent, meeting):
    manifest = read(parent / "inputs.json")
    verification = read(parent / "inputs-verified.json")
    result = read(parent / "result.json")
    require(verification["inputs_sha256"] == sha(parent / "inputs.json") and verification["rows"] == 316
            and manifest["freeze_sha256"] == sha(parent / "inputs-freeze.json")
            and result["decisions_sha256"] == sha(parent / "decisions.json")
            and result["fit_sha256"] == sha(parent / "fit.json"), "Prior source or baseline commitments differ")
    rows = previous.ordered_rows(manifest)
    require(len(rows) == len({r["pcm_sha256"] for r in rows}) == 316, "Four-second PCM sources must remain unique")
    ancestry = {role: {r[key] for r in rows if r["role"] == role for key in ("host_group", "donor_group", "background_group") if r[key]}
                for role in previous.ROLES}
    require(sum(map(len, ancestry.values())) == len(set().union(*ancestry.values()))
            and all(r["role"] == "confirmation" or r["family"] != "expressive" for r in rows), "Source roles or expressive holdout overlap")
    room_inputs = meeting_sources(meeting)
    room_read = read(parent / "meeting-replay.json")
    room_freeze = read(parent / "meeting-replay-freeze.json")
    require(room_read["freeze_sha256"] == sha(parent / "meeting-replay-freeze.json")
            and room_freeze["parent_roster_sha256"] == sha(meeting / "roster.json")
            and len(room_inputs) == len(room_read["decisions"]) == 48, "Prior AMI source binding changed")
    for (row, raw), prior in zip(room_inputs, room_freeze["inputs"], strict=True):
        require([row["id"], digest(raw)] == prior[:2], "AMI source hashes differ from incumbent inputs")
    names = ("inputs.json", "inputs-freeze.json", "inputs-verified.json", "fit.json", "decisions.json", "result.json",
             "meeting-replay.json", "meeting-replay-freeze.json")
    method = {"schema": SCHEMA, "method": METHOD, "script_sha256": sha(Path(__file__)),
              "parent": {n: sha(parent / n) for n in names},
              "meeting": {n: sha(meeting / n) for n in ("freeze.json", "roster.json", "selection.json", "confirmation.json", "points.json")},
              "runtime": {n: importlib.metadata.version(n) for n in ("numpy", "scipy")},
              "four_second_roles": dict(Counter(r["role"] for r in rows)),
              "four_second_ancestry_groups": {role: len(groups) for role, groups in ancestry.items()},
              "meeting_views": len(room_inputs), "new_neural_forwards": 0}
    return method, rows, room_inputs


def fit_ridge(rows, matrices):
    retained = [(r, f) for r, f in zip(rows, matrices, strict=True) if r["arm"] in (*previous.HUMAN, previous.SYNTHETIC)]
    features_fit = np.concatenate([f for _, f in retained])
    labels = np.concatenate([previous.overlap_weights(r["truth"]) for r, _ in retained])
    use = np.isfinite(features_fit).all(axis=1)
    data, y = features_fit[use], labels[use]
    require(0 < y.sum() < len(y), "Not enough covered synthetic or genuine cells to fit")
    center, scale = data.mean(axis=0), np.maximum(data.std(axis=0), .03)
    standard = np.column_stack(((data-center)/scale, np.ones(len(data))))
    mass = y/(2*y.sum()) + (1-y)/(2*(1-y).sum())
    penalty = np.diag([.10]*data.shape[1] + [0.])
    gram = standard.T @ (mass[:, None] * standard) + penalty
    target = standard.T @ (mass*y)
    beta = np.linalg.solve(gram, target)
    require(np.max(abs(gram @ beta - target)) <= 1e-10, "Ridge fit did not reach its normal equations")
    return {"center": center.tolist(), "scale": scale.tolist(), "weights": beta[:-1].tolist(), "bias": float(beta[-1]),
            "fit_views": len(retained), "covered_fit_cells": int(use.sum()), "positive_overlap_fit_cells": int((y > 0).sum())}


def measure(raw, head, truth):
    matrix = features(raw)
    scores = (matrix - head["center"]) / head["scale"] @ head["weights"] + head["bias"]
    windows = np.lib.stride_tricks.sliding_window_view(scores, 3)
    counts = np.isfinite(windows).sum(axis=1)
    values = np.where(counts >= 2, np.nansum(windows, axis=1) / np.maximum(counts, 1), -np.inf)
    support = np.flatnonzero(np.isfinite(scores))
    if not np.isfinite(values).any():
        return {"score": None, "range": None, "located": False, "covered_cells": int(support.size), "range_support": []}
    index = int(values.argmax())
    start, end = index*CELL, (index+3)*CELL
    exact = support[(support >= index) & (support < index+3)]
    return {"score": float(values[index]), "range": [start, end], "located": previous.useful_range(start, end, truth),
            "covered_cells": int(support.size), "range_support": [[int(c*CELL), int((c+1)*CELL)] for c in exact]}


def train(run, parent, rows):
    fit_rows = [r for r in rows if r["role"] == "fusion_fit"]
    matrices = [features(previous.pcm_bytes(parent, r)) for r in fit_rows]
    head = fit_ridge(fit_rows, matrices)
    selection = [r for r in rows if r["role"] == "confidence_cal" and not r["label"]]
    results = [{"id": r["id"], "pcm_sha256": r["pcm_sha256"], **measure(previous.pcm_bytes(parent, r), head, None)} for r in selection]
    usable = [r for r in results if r["score"] is not None]
    require(len(usable) >= len(selection)//2, "Genuine operating-point coverage is too low")
    worst = max(usable, key=lambda r: r["score"])
    write(run / "fit.json", {"freeze_sha256": sha(run / "freeze.json"), "head": head, "point": worst["score"],
                            "point_witness": worst["id"], "selection": results})
    print("Frozen excitation head and independent genuine operating point", flush=True)


def decision(value, point):
    return {**value, "flag": value["score"] is not None and value["score"] > point}


def observations(parent, meeting, rows, rooms, seal):
    earlier = {r["id"]: r for r in read(parent / "decisions.json")}
    output = []
    for row in rows:
        if row["role"] not in {"selection", "confirmation"}:
            continue
        prior = earlier[row["id"]]
        require(prior["truth"] == row["truth"] and prior["pair"] == row["pair_id"], "Incumbent decision binds to another edit")
        old = prior["checks"][BASELINE]
        score = measure(previous.pcm_bytes(parent, row), seal["head"], row["truth"])
        output.append({"panel": "matched", "role": row["role"], "id": row["id"], "condition": row["condition"],
                       "family": row["family"], "arm": row["arm"], "truth": row["truth"],
                       "group": row["pair_id"] if row["arm"] != "background_only" else row["background_group"],
                       "pcm_sha256": row["pcm_sha256"], "checks": {BASELINE: {**old, "located": old["best_range_useful"]},
                                                                  CANDIDATE: decision(score, seal["point"])}})
    replay = {r["id"]: r for r in read(parent / "meeting-replay.json")["decisions"]}
    short = {r["id"]: r for role in ("selection", "confirmation") for r in read(meeting / f"{role}.json")["rows"]}
    point = read(meeting / "points.json")["points"]["short_odd"]["point"]
    for row, raw in rooms:
        old, separate = replay[row["id"]], short[row["id"]]
        require(old["truth"] == row["truth"] and separate["input_sha256"] == row["source_pcm_sha256"], "AMI baseline input differs")
        incumbent = old["checks"][BASELINE]
        value, bounds = separate["features"]["short_odd"]
        score = measure(raw, seal["head"], row["truth"])
        output.append({"panel": "meeting", "role": row["panel"], "id": row["id"], "condition": row["condition"],
                       "family": row["family"], "arm": row["kind"], "truth": row["truth"], "group": row["host_group"],
                       "pcm_sha256": row["source_pcm_sha256"], "checks": {
                           BASELINE: {**incumbent, "located": incumbent["location"]},
                           ROOM_BASELINE: decision({"score": value, "range": bounds,
                                                   "located": previous.useful_range(*bounds, row["truth"])}, point),
                           CANDIDATE: decision(score, seal["point"])}})
    return output


def counts(rows, arm, budget=None):
    def score(row):
        return row["checks"][arm].get("score")

    if budget is not None:
        maxima = {}
        for r in rows:
            value = score(r)
            if r["truth"] is None and value is not None:
                maxima[r["group"]] = max(value, maxima.get(r["group"], -math.inf))
        observed = sorted(maxima.values(), reverse=True)
        if len(observed) <= budget:
            return None
        cut = observed[budget]
        flagged = [r for r in rows if score(r) is not None and score(r) > cut]
    else:
        flagged = [r for r in rows if r["checks"][arm]["flag"]]
    false = [r for r in flagged if r["truth"] is None]
    return {"synthetic": sum(r["truth"] is not None for r in rows), "genuine": sum(r["truth"] is None for r in rows),
            "true_flags": sum(r["truth"] is not None for r in flagged), "genuine_false_flags": len(false),
            "genuine_false_groups": len({r["group"] for r in false}),
            "useful": sum(r["checks"][arm]["located"] for r in flagged),
            "threshold_free_useful": sum(r["checks"][arm]["located"] for r in rows),
            "no_measurement": sum(r["checks"][arm].get("range") is None for r in rows),
            "background_only_flags": sum(r["arm"] == "background_only" for r in false)}


def pair_ranking(rows, arm, control_arms):
    outcomes = Counter(comparable=0, above_all_controls=0)
    groups = {}
    for row in rows:
        group = groups.setdefault(row["group"], {})
        group[row["arm"]] = row
    for group in groups.values():
        positive = next((r for r in group.values() if r["truth"] is not None), None)
        if positive is None:
            continue
        require(set(control_arms) <= group.keys(), "Known edit lacks a matching genuine control")
        score = positive["checks"][arm].get("score")
        controls = [group[key]["checks"][arm].get("score") for key in control_arms]
        if score is not None and all(value is not None for value in controls):
            outcomes.update(comparable=1, above_all_controls=int(score > max(controls)))
    return dict(outcomes)


def summarize(rows):
    cells = {}
    roles = sorted({(r["panel"], r["role"], r["condition"]) for r in rows})
    for panel, role, condition in roles:
        subset = [r for r in rows if (r["panel"], r["role"], r["condition"]) == (panel, role, condition)]
        cell = cells["/".join((panel, role, condition))] = {arm: counts(subset, arm) for arm in subset[0]["checks"]}
        if role == "confirmation":
            comparator = BASELINE if panel == "matched" else ROOM_BASELINE
            cell["equal_false_groups"] = {str(n): {arm: counts(subset, arm, n) for arm in (comparator, CANDIDATE)} for n in (0, 1, 2)}
            controls = previous.HUMAN if panel == "matched" else ("ordinary",)
            cell["matched_ranking"] = {arm: pair_ranking(subset, arm, controls) for arm in (comparator, CANDIDATE)}
    held_noisy = cells["matched/confirmation/speech_minus5db"]
    held_dry = cells["matched/confirmation/dry"]
    held_room = cells["meeting/confirmation/room"]
    added_families = {r["family"] for r in rows if r["panel"] == "matched" and r["role"] == "confirmation" and r["condition"] == "speech_minus5db"
                      and r["checks"][CANDIDATE]["flag"] and r["checks"][CANDIDATE]["located"]
                      and not (r["checks"][BASELINE]["flag"] and r["checks"][BASELINE]["located"])}
    gate = {"noisy_useful_net": held_noisy[CANDIDATE]["useful"] - held_noisy[BASELINE]["useful"],
            "noisy_added_families": sorted(added_families),
            "noisy_extra_false_groups": held_noisy[CANDIDATE]["genuine_false_groups"] - held_noisy[BASELINE]["genuine_false_groups"],
            "new_background_false": held_noisy[CANDIDATE]["background_only_flags"],
            "dry_useful_net": held_dry[CANDIDATE]["useful"] - held_dry[BASELINE]["useful"],
            "ami_noisy_useful": held_room[CANDIDATE]["useful"],
            "ami_noisy_extra_false": held_room[CANDIDATE]["genuine_false_flags"] - held_room[BASELINE]["genuine_false_flags"]}
    gate["qualifies"] = (gate["noisy_useful_net"] >= 2 and len(added_families) >= 2 and gate["noisy_extra_false_groups"] <= 1
                         and gate["new_background_false"] == 0 and gate["dry_useful_net"] >= 0
                         and gate["ami_noisy_useful"] >= 1 and gate["ami_noisy_extra_false"] <= 0)
    return {"cells": cells, "gate": gate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "report", "audit"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--meeting", type=Path, default=MEETING)
    args = parser.parse_args()
    run = args.run
    lock, rows, rooms = bind(args.parent, args.meeting)
    started = time.monotonic()
    if args.phase == "freeze":
        run.mkdir(mode=0o700)
        for row in rows:
            previous.pcm_bytes(args.parent, row)
        write(run / "freeze.json", lock)
        print("Frozen 316 exact matched views and 48 exact AMI room views", flush=True)
    else:
        require(lock == read(run / "freeze.json"), "Frozen method or sources differ")
        if args.phase == "fit":
            train(run, args.parent, rows)
        else:
            seal = read(run / "fit.json")
            require(seal["freeze_sha256"] == sha(run / "freeze.json"), "Fit differs from the frozen method")
            results = observations(args.parent, args.meeting, rows, rooms, seal)
            summary = summarize(results)
            if args.phase == "report":
                write(run / "observations.json", results)
                write(run / "result.json", {**lock, **summary, "commitments": {name: sha(run / f"{name}.json") for name in
                          ("freeze", "fit", "observations")}, "observation_views": len(results), "seconds": round(time.monotonic()-started, 3)})
                print(summary["gate"], flush=True)
            else:
                result = read(run / "result.json")
                require(all(sha(run / f"{name}.json") == value for name, value in result["commitments"].items())
                        and read(run / "observations.json") == results and all(result[k] == value for k, value in summary.items()),
                        "PCM features, ranges or report counts differ from exact replay")
                print(f"Verified {len(results)} exact source-bound observation rows and {len(summary['cells'])} outcome cells.")


if __name__ == "__main__":
    main()
