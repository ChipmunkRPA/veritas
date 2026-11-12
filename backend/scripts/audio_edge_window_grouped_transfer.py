"""Freeze grouped ordinary points, then measure Full and Combined on a disjoint noisy panel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts import audio_edge_window_grouped_sources as sources
from scripts import audio_edge_window_transfer as probe
from scripts import review_audio_edge_window_transfer as prior

BASE, ROOT = probe.BASE, probe.ROOT
read, write, sha, digest, require = probe.read, probe.write, probe.sha, probe.digest, probe.require
RUN = BASE / "audio-edge-window-grouped-2026-11-12"
SCHEMA = "chipmunk.audio-edge-window-grouped/1"
ARMS = ("full", "combined")
PRIMARY, SENSITIVITY = 2, 1
BUDGETS = (PRIMARY, SENSITIVITY)
METHOD = {
    "calibration": "For each grid take the maximum admitted score within each of the 16 groups of 40 sealed ordinary controls. "
                   "Strictly exceed the third-highest group score for the primary two-group budget. Freeze a one-group sensitivity point using rank two.",
    "held": "Use 12 fresh synthetic inserts, matched genuine same-reader and other-reader splices and untouched controls, and two isolated room controls. "
            "Keep two separate rooms with strong annotated competing speech at -5 or -8 dB and four edge edits.",
    "localization": "Best speech-admitted interval by raw logit, tie to earliest start. A useful range has IoU >=0.30 and edit coverage >=0.50.",
    "comparison": "Report flags and misses, wrong locations, genuine false files and groups, room-control flags and both overlap rules. "
                  "Thresholds use ordinary calibration only. A direction or customer probability is outside this experiment.",
}


def group_fit(rows):
    fit = {}
    for arm in ARMS:
        grid = probe.GRIDS[arm]
        by_group = {}
        for row in rows:
            require(row["truth"] is None, "An edited file entered ordinary-only calibration")
            score, _ = prior.strongest(row, grid)
            key = row["group"]
            by_group[key] = max(score, by_group.get(key, -math.inf))
        ordered = sorted(by_group.values(), reverse=True)
        require(len(ordered) == 16 and all(math.isfinite(point) for point in ordered[:PRIMARY + 1]),
                "Insufficient calibration group support")
        fit[arm] = {str(budget): {"point": ordered[budget], "calibration_false_groups": sum(v > ordered[budget] for v in ordered)}
                    for budget in BUDGETS}
    return fit


def freeze(held):
    reference = read(prior.RECEIPT)
    prior.validate(reference)
    require(sum(1 for _ in sources.cases(held)) == 50, "Unexpected held denominator")
    return {
        "schema": SCHEMA,
        "method": METHOD,
        "held_sources": sha(held / "sources.json"),
        "ordinary_receipt": sha(prior.RECEIPT),
        "ordinary_rows": reference["commitments"]["selection_public_rows"],
        "primary_budget": PRIMARY,
        "sensitivity_budget": SENSITIVITY,
        "grids": {arm: [list(w) for w in probe.GRIDS[arm]] for arm in ARMS},
        "assets": probe.native.transfer.OLD.voice.ASSETS,
        "implementation": {name: sha(ROOT / name) for name in (
            "backend/scripts/audio_edge_window_grouped_sources.py",
            "backend/scripts/audio_edge_window_grouped_transfer.py",
            "backend/scripts/audio_edge_window_transfer.py",
            "backend/scripts/audio_edge_window_sources.py",
            "backend/scripts/run_audio_specialists_v15.py",
            "backend/app/detectors/audio/upgraded.py",
            "backend/app/detectors/audio/content.py",
            "backend/app/detectors/audio/calibration.py",
        )},
    }, reference["selection"]


def score(run, held, weights):
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    lock, ordinary = freeze(held)
    points = read(run / "points.json")
    lock_sha = sha(run / "freeze.json")
    require(read(run / "freeze.json") == lock and points == {"freeze": lock_sha, "operating_points": group_fit(ordinary)},
            "Frozen study or ordinary-only grouped points changed")
    probe.native.transfer.OLD.spectra.incumbent.network_free()
    require(all(sha(weights / name) == pin for name, pin in lock["assets"].items()), "Pinned speech or voice assets changed")
    model = load_voice({"weights": str(weights), "assets": lock["assets"]})
    activity = SpeechActivity(weights / "silero_vad.jit")
    held_scores = probe.score_cases(((row, raw, {}) for row, raw in sources.cases(held)), model, activity, "grouped holdout")
    write(run / "held.json", {"freeze": lock_sha, "points": sha(run / "points.json"), "records": held_scores})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("sources", "freeze", "score"))
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--held", type=Path, default=sources.HELD)
    parser.add_argument("--weights", type=Path, default=probe.native.WEIGHTS)
    args = parser.parse_args()
    if args.phase == "sources":
        sources.prepare(args.held)
    elif args.phase == "freeze":
        lock, ordinary = freeze(args.held)
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        points = group_fit(ordinary)
        write(args.run / "points.json", {"freeze": sha(args.run / "freeze.json"), "operating_points": points})
        print(f"Froze both budgets on 16 ordinary source groups before held scoring: {sha(args.run / 'points.json')}", flush=True)
        print(points, flush=True)
    else:
        score(args.run, args.held, args.weights)


if __name__ == "__main__":
    main()
