# SPDX-License-Identifier: Apache-2.0
"""Compare local spectrotemporal modulation and cepstral cues on held speech edits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn

from scripts import audio_cepstral_partial_transfer as cepstral
from scripts import audio_periodic_excitation_transfer as prior

SCHEMA = "chipmunk.audio-modulation-partial-transfer/1"
REPLICATION = prior.BASE / "audio-remote-neighbor-replication-2026-11-12"
SPECTRAL = ((0, 1), (1, 4), (4, 9), (9, 16), (16, 32))
TEMPORAL = ((1, 3), (3, 6), (6, 10), (10, 18))
METHOD = {
    "frame_hop_fft": [cepstral.FRAME, cepstral.HOP, cepstral.FFT],
    "cell_samples": prior.CELL,
    "mel_bands": 32,
    "temporal_dct_bins": TEMPORAL,
    "spectral_dct_bins": SPECTRAL,
    "local_range_samples": 3 * prior.CELL,
    "views": ["modulation", "with_cepstral"],
    "point": "Strict maximum on the independent genuine point controls.",
    "advance": "At least two net useful locations and two added families in 12 held room edits, with at most one additional ordinary group and no background-only flags.",
}


def modulation_map(log_energy):
    values = log_energy - log_energy.mean(axis=-2, keepdims=True)
    energy = dctn(values, axes=(-2, -1), norm="ortho") ** 2
    return np.stack([energy[:, a:b, c:d].sum(axis=(1, 2))
                     for a, b in TEMPORAL for c, d in SPECTRAL], axis=1)


def features(raw, mel):
    wave = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    cells = len(wave) // prior.CELL
    blocks = wave[:cells * prior.CELL].reshape(cells, prior.CELL)
    frames = np.lib.stride_tricks.sliding_window_view(blocks, cepstral.FRAME, axis=1)[:, ::cepstral.HOP]
    spectrum = np.fft.rfft((frames - frames.mean(axis=-1, keepdims=True)) * np.hanning(cepstral.FRAME), n=cepstral.FFT)
    energies = modulation_map(np.log(np.maximum(np.abs(spectrum) ** 2 @ mel, 1e-10)))
    values = np.log1p(energies)
    normalized = np.log((energies + .001) / (energies.sum(axis=1, keepdims=True) + .001 * energies.shape[1]))
    neighbors = np.vstack((normalized[0], normalized, normalized[-1]))
    contrast = np.abs(normalized - (neighbors[:-2] + neighbors[2:]) / 2)
    local = np.concatenate((values, normalized, contrast), axis=1)
    local[np.mean(blocks**2, axis=1) < 1e-5] = np.nan
    return {"modulation": local, "with_cepstral": np.concatenate((local, cepstral.features(raw, "mel")), axis=1)}


def baseline(row, truth, point):
    value, bounds = row["responses"]["current"]
    located = bounds is not None and prior.previous.useful_range(*bounds, truth)
    return prior.decision({"score": value, "range": bounds, "located": located}, point)


def summarize(rows):
    result = {}
    for condition in ("matched_dry", "matched_loud_speech", "recorded_room_speech"):
        cases = [row for row in rows if row["condition"] == condition]
        arms = cases[0]["checks"]
        result[condition] = {arm: prior.counts(cases, arm) for arm in arms}
        result[condition]["equal_false_groups"] = {
            str(n): {arm: prior.counts(cases, arm, n) for arm in arms} for n in (0, 1, 2)
        }
    cases = [row for row in rows if row["condition"] == "recorded_room_speech"]
    current = result["recorded_room_speech"]["incumbent"]
    gates = {}
    for arm in METHOD["views"]:
        outcome = result["recorded_room_speech"][arm]
        added = {row["family"] for row in cases if row["checks"][arm]["flag"] and row["checks"][arm]["located"]
                 and not (row["checks"]["incumbent"]["flag"] and row["checks"]["incumbent"]["located"])}
        gates[arm] = {"added_families": sorted(added), "qualifies": outcome["useful"] >= current["useful"] + 2 and len(added) >= 2
                      and outcome["genuine_false_groups"] <= current["genuine_false_groups"] + 1
                      and outcome["background_only_flags"] == 0}
    return {"conditions": result, "gates": gates}


def run(parent, meeting, replication, folder, review):
    # A stationary spectral envelope has no nonzero temporal modulation.
    stationary = np.broadcast_to(np.arange(32, dtype=float), (1, 18, 32))
    assert np.max(np.abs(modulation_map(stationary))) < 1e-20
    varying = stationary + np.cos(np.pi * (np.arange(18) + .5) * 4 / 18)[None, :, None]
    spectrum = modulation_map(varying)[0]
    assert spectrum.argmax() == len(SPECTRAL) and spectrum.max() > 100

    ancestry, parent_rows, _ = prior.bind(parent, meeting)
    old_freeze = prior.read(replication / "freeze.json")
    old_summary = prior.read(replication / "result.json")
    names = ("freeze", "roster", "current", "neighbor")
    commitments = {name: prior.sha(replication / f"{name}.json") for name in names}
    prior.require(commitments == old_summary["commitments"] and commitments["roster"] == old_freeze["roster_sha256"],
                  "Replicated panel or current-model inputs changed")
    old = prior.read(replication / "decisions.json")
    roster = prior.read(replication / "roster.json")
    current = prior.read(replication / "current.json")["records"]
    prior.require(len(roster) == len(old) == len(current) == 52 and [row["id"] for row in old] == [row["id"] for row in roster],
                  "Unexpected room panel roster")
    sealed = {"schema": SCHEMA, "method": json.loads(json.dumps(METHOD)), "script": prior.sha(Path(__file__)),
              "cepstral_script": prior.sha(Path(cepstral.__file__)), "prior": ancestry,
              "replication": commitments | {"decisions": prior.sha(replication / "decisions.json")}}
    if review:
        prior.require(prior.read(folder / "freeze.json") == sealed, "Source freeze differs")
    else:
        folder.mkdir(mode=0o700)
        prior.write(folder / "freeze.json", sealed)

    mel = cepstral.filterbank("mel")
    fit = [row for row in parent_rows if row["role"] == "fusion_fit"]
    selection = [row for row in parent_rows if row["role"] == "confidence_cal" and not row["label"]]
    fit_views = [features(prior.previous.pcm_bytes(parent, row), mel) for row in fit]
    point_views = [features(prior.previous.pcm_bytes(parent, row), mel) for row in selection]
    heads = {name: prior.fit_ridge(fit, [view[name] for view in fit_views]) for name in METHOD["views"]}
    selection_scores = {name: [cepstral.score(view[name], head, None)["score"] for view in point_views] for name, head in heads.items()}
    points = {name: max(score for score in values if score is not None) for name, values in selection_scores.items()}
    operating = {"heads": heads, "selection_scores": selection_scores, "points": points}
    if review:
        prior.require(prior.read(folder / "points.json") == operating, "Independent points differ")
    else:
        prior.write(folder / "points.json", operating)

    def measure(raw, row, condition, incumbent, group):
        views = features(raw, mel)
        truth = row["truth"]
        checks = {name: prior.decision(cepstral.score(views[name], head, truth), points[name]) for name, head in heads.items()}
        return {"condition": condition, "group": group, "family": row["family"], "arm": row["arm"], "truth": truth,
                "pcm_sha256": prior.digest(raw), "checks": {"incumbent": incumbent, **checks}}

    earlier = {row["id"]: row for row in prior.read(parent / "decisions.json")}
    matched_point = prior.read(parent / "fit.json")["points"][prior.BASELINE]["point"]
    room_point = old_freeze["points"]["current"]
    cases = []
    for row in parent_rows:
        if row["role"] != "confirmation" or row["condition"] not in {"dry", "speech_minus5db"}:
            continue
        reference = earlier[row["id"]]
        prior.require(reference["truth"] == row["truth"] and reference["pair"] == row["pair_id"], "Incumbent labels differ")
        incumbent = reference["checks"][prior.BASELINE]
        condition = "matched_dry" if row["condition"] == "dry" else "matched_loud_speech"
        group = row["background_group"] if row["arm"] == "background_only" else row["pair_id"]
        cases.append(measure(prior.previous.pcm_bytes(parent, row), row, condition,
                             {**incumbent, "located": incumbent["best_range_useful"]}, group))
    for row, outcome in zip(roster, old, strict=True):
        raw = (replication / "pcm" / (row["id"] + ".pcm")).read_bytes()
        prior.require(len(raw) == 192000 and prior.digest(raw) == row["pcm_sha256"] == outcome["pcm_sha256"]
                      and row["truth"] == outcome["truth"] and row["group"] == outcome["group"], "Replicated input differs")
        cases.append(measure(raw, row, "recorded_room_speech", baseline(outcome, row["truth"], room_point), row["group"]))
    receipt = {"schema": SCHEMA, "freeze_sha256": prior.sha(folder / "freeze.json"), "points": points,
               "incumbent_points": {"matched_dry": matched_point, "matched_loud_speech": matched_point,
                                    "recorded_room_speech": room_point},
               "selection_scores": selection_scores, "cases": cases, "result": summarize(cases)}
    if review:
        prior.require(prior.read(folder / "receipt.json") == receipt, "Native measurements differ")
    else:
        prior.write(folder / "receipt.json", receipt)
    print(json.dumps(receipt["result"], sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "review"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--parent", type=Path, default=prior.PARENT)
    parser.add_argument("--meeting", type=Path, default=prior.MEETING)
    parser.add_argument("--replication", type=Path, default=REPLICATION)
    args = parser.parse_args()
    run(args.parent, args.meeting, args.replication, args.folder, args.mode == "review")
