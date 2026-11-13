# SPDX-License-Identifier: Apache-2.0
"""Test complementary local cepstral measurements on short voice edits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.fft import dct

from scripts import audio_periodic_excitation_transfer as prior

SCHEMA = "chipmunk.audio-cepstral-partial-transfer/1"
RATE, CELL = prior.RATE, prior.CELL
FRAME, HOP, FFT = 400, 160, 512
ARM = prior.CANDIDATE
METHOD = {
    "frames": [FRAME, HOP, FFT],
    "bands": 32,
    "coefficients": [1, 12],
    "cell_samples": CELL,
    "views": ["linear", "mel", "combined"],
    "pool": "Per-cell mean and dispersion of cepstral coefficients, first differences and residual from adjacent cells.",
    "point": "Independent genuine maximum; same sample intervals and advancement rule as the excitation transfer.",
    "scope": "Use existing matched source splits and original mixture only. No encoder or detector weights.",
}


def filterbank(kind):
    low, high = 60, 7800
    if kind == "mel":
        spacing = np.linspace(np.log1p(low / 700), np.log1p(high / 700), METHOD["bands"] + 2)
        centers = 700 * np.expm1(spacing)
    else:
        centers = np.linspace(low, high, METHOD["bands"] + 2)
    f = np.fft.rfftfreq(FFT, 1 / RATE)[:, None]
    a, b, c = centers[:-2], centers[1:-1], centers[2:]
    filters = np.maximum(0, np.minimum((f - a) / (b - a), (c - f) / (c - b)))
    return filters / filters.sum(axis=0)


def features(raw, kind):
    wave = np.frombuffer(raw, "<i2").astype(np.float64) / 32768
    cells = len(wave) // CELL
    blocks = wave[: cells * CELL].reshape(cells, CELL)
    frames = np.lib.stride_tricks.sliding_window_view(blocks, FRAME, axis=1)[:, ::HOP, :]
    power = (
        np.abs(np.fft.rfft((frames - frames.mean(axis=-1, keepdims=True)) * np.hanning(FRAME), n=FFT)) ** 2
    )
    cepstrum = dct(np.log(np.maximum(power @ filterbank(kind), 1e-10)), axis=-1, norm="ortho")[..., 1:13]
    mean = cepstrum.mean(axis=1)
    spread = cepstrum.std(axis=1)
    delta = np.abs(np.diff(cepstrum, axis=1)).mean(axis=1)
    neighbors = np.vstack([mean[0], mean, mean[-1]])
    contrast = np.abs(mean - (neighbors[:-2] + neighbors[2:]) / 2)
    result = np.concatenate((mean, spread, delta, contrast), axis=1)
    audible = np.mean(blocks**2, axis=-1) >= 1e-5
    result[~audible] = np.nan
    return result


def feature_views(raw):
    linear, mel = (features(raw, kind) for kind in ("linear", "mel"))
    return {"linear": linear, "mel": mel, "combined": np.concatenate((linear, mel), axis=1)}


def score(matrix, head, truth):
    values = (matrix - head["center"]) / head["scale"] @ head["weights"] + head["bias"]
    windows = np.lib.stride_tricks.sliding_window_view(values, 3)
    counts = np.isfinite(windows).sum(axis=1)
    eligible = np.where(counts >= 2, np.nansum(windows, axis=1) / np.maximum(counts, 1), -np.inf)
    if not np.isfinite(eligible).any():
        return {"score": None, "range": None, "located": False}
    idx = int(eligible.argmax())
    a, b = idx * CELL, (idx + 3) * CELL
    return {
        "score": float(eligible[idx]),
        "range": [a, b],
        "located": prior.previous.useful_range(a, b, truth),
    }


def build(parent, meeting, folder, phase):
    lock, rows, rooms = prior.bind(parent, meeting)
    seal = {"schema": SCHEMA, "runner": prior.sha(Path(__file__)), "method": METHOD, "parent": lock}
    matched = [(row, prior.previous.pcm_bytes(parent, row)) for row in rows]
    if phase == "run":
        folder.mkdir(mode=0o700)
        prior.write(folder / "freeze.json", seal)
    else:
        prior.require(seal == prior.read(folder / "freeze.json"), "Frozen procedure changed")
    fit = [(r, feature_views(raw)) for r, raw in matched if r["role"] == "fusion_fit"]
    select = [
        (r, feature_views(raw)) for r, raw in matched if r["role"] == "confidence_cal" and not r["label"]
    ]
    heads = {
        name: prior.fit_ridge([r for r, _ in fit], [views[name] for _, views in fit])
        for name in METHOD["views"]
    }
    selection = {
        name: [score(views[name], head, None)["score"] for _, views in select] for name, head in heads.items()
    }
    points = {name: max(value for value in values if value is not None) for name, values in selection.items()}
    operating = {"heads": heads, "points": points, "selection": selection}
    if phase == "run":
        prior.write(folder / "point.json", operating)
    else:
        prior.require(prior.read(folder / "point.json") == operating, "Head or independent point differs")
    earlier = {row["id"]: row for row in prior.read(parent / "decisions.json")}
    room_old = {row["id"]: row for row in prior.read(parent / "meeting-replay.json")["decisions"]}
    room_short = {
        r["id"]: r
        for role in ("selection", "confirmation")
        for r in prior.read(meeting / f"{role}.json")["rows"]
    }
    short_point = prior.read(meeting / "points.json")["points"]["short_odd"]["point"]
    common, outcomes = [], {name: [] for name in heads}
    for row, raw in matched + rooms:
        room = "panel" in row
        if not room and row["role"] not in {"selection", "confirmation"}:
            continue
        truth = row["truth"]
        if room:
            incumbent = room_old[row["id"]]["checks"][prior.BASELINE]
            value, bounds = room_short[row["id"]]["features"]["short_odd"]
            checks = {
                prior.BASELINE: {**incumbent, "located": incumbent["location"]},
                prior.ROOM_BASELINE: prior.decision(
                    {"score": value, "range": bounds, "located": prior.previous.useful_range(*bounds, truth)},
                    short_point,
                ),
            }
        else:
            prior_row = earlier[row["id"]]
            prior.require(
                prior_row["truth"] == truth and prior_row["pair"] == row["pair_id"], "Prior decision differs"
            )
            incumbent = prior_row["checks"][prior.BASELINE]
            checks = {prior.BASELINE: {**incumbent, "located": incumbent["best_range_useful"]}}
        group = (
            row["host_group"]
            if room
            else (row["background_group"] if row["arm"] == "background_only" else row["pair_id"])
        )
        base = {
            "panel": "meeting" if room else "matched",
            "role": row["panel"] if room else row["role"],
            "id": row["id"],
            "condition": row["condition"],
            "family": row["family"],
            "arm": row["kind"] if room else row["arm"],
            "truth": truth,
            "group": group,
            "pcm_sha256": prior.digest(raw),
            "checks": checks,
        }
        common.append(base)
        views = feature_views(raw)
        for name, head in heads.items():
            outcomes[name].append(prior.decision(score(views[name], head, truth), points[name]))
    summaries = summarize(common, outcomes)
    result = {
        "schema": SCHEMA,
        "freeze_sha256": prior.sha(folder / "freeze.json"),
        "method": METHOD,
        "points": points,
        "selection": selection,
        "baseline": common,
        "outcomes": outcomes,
        "summary": summaries,
    }
    if phase == "run":
        prior.write(folder / "receipt.json", result)
    else:
        prior.require(prior.read(folder / "receipt.json") == result, "Native input or output differs")
    print(summaries, flush=True)


def summarize(common, outcomes):
    return {
        name: prior.summarize(
            [
                row | {"checks": row["checks"] | {ARM: check}}
                for row, check in zip(common, readings, strict=True)
            ]
        )
        for name, readings in outcomes.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "review"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--parent", type=Path, default=prior.PARENT)
    parser.add_argument("--meeting", type=Path, default=prior.MEETING)
    args = parser.parse_args()
    build(args.parent, args.meeting, args.folder, args.phase)
