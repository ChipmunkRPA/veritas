# SPDX-License-Identifier: Apache-2.0
"""Fit direct speech-interval rankings on exact same-clock comparisons."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio import interval_features as intervals
from scripts import audio_focused_context_transfer as files
from scripts import audio_long_context_fit as parent
from scripts import audio_long_context_review as reference

SCHEMA = "chipmunk.audio-interval-ranking/1"
RIDGE = .08
PARTS = {"pooled": slice(0, 192), "contrast": slice(192, 384), "combined": slice(None)}
DESIGN = {
    "scope": "One 16 kHz six-second input with 250–1000 ms windows every 50 ms and the same speech-activity rule.",
    "features": "Take mean, spread and frame-to-frame movement in the three established projected speech layers. "
                "Fit these interval measures, their difference from equal-width flanking intervals, and both together.",
    "ranking": "Rank the best-overlap candidate at each duration above exactly matched ordinary, resampled and background-only "
               "windows, and disjoint same-file intervals. The repeat fit also uses the highest-scoring wrong windows in development only.",
    "calibration": "Independent genuine readers and quiet/noisy deliveries set per-duration scales and both whole-file operating points.",
    "evaluation": "Keep the existing app's ranges. Compare standalone candidates and adding a candidate when there is no existing range.",
}


def grouped(rows):
    batches = defaultdict(dict)
    for row in rows:
        if row["role"] == "fit":
            batches[row["source"], row["condition"]][row["kind"]] = row
    return batches


def true_and_wrong(truth):
    a, b = truth
    ranges = intervals.WINDOWS
    intersections = np.maximum(0, np.minimum(ranges[:, 1], b) - np.maximum(ranges[:, 0], a))
    union = ranges[:, 1] - ranges[:, 0] + b - a - intersections
    matches = (2 * intersections >= b-a) & (10 * intersections >= 3 * union)
    best = []
    for width in range(len(intervals.WIDTHS)):
        options = np.flatnonzero(matches & np.equal(intervals.GROUPS, width))
        if options.size:
            best.append(options[np.argmax(intersections[options] / union[options])])
    if not best:
        raise ValueError("No valid proposal can cover the labeled fitting interval")
    return np.asarray(best), np.flatnonzero(intersections == 0)


def fitting_rows(batches, load, preliminary=None):
    draws = defaultdict(list)
    random = np.random.default_rng(611419)
    for index, cases in enumerate(batches.values(), 1):
        grid = {kind: intervals.features(load(row, "short")) for kind, row in cases.items()}
        for kind, row in cases.items():
            if not row["truth"]:
                continue
            family = kind.split("/", 1)[1]
            target, outside = true_and_wrong(row["truth"]["short"])
            edit = grid[kind]
            positive = edit[target]
            ordinary = ["ordinary", f"resampled/{family}"] + (["background"] if "background" in grid else [])
            decoys = random.choice(outside, size=len(target), replace=False)
            base = [positive - grid[control][target] for control in ordinary]
            base.append(positive - edit[decoys])
            for arm in intervals.ARMS:
                evidence = base.copy()
                if preliminary:
                    wrong = outside[np.argmax(intervals.ranks(edit[outside], preliminary[arm]))]
                    evidence.append(positive - edit[wrong])
                    for control in ordinary:
                        counterexample = grid[control]
                        strongest_wrong = counterexample[np.argmax(intervals.ranks(counterexample, preliminary[arm]))]
                        evidence.append(positive - strongest_wrong)
                selected = np.concatenate(evidence)[:, PARTS[arm]]
                draws[arm].append((selected, np.full(len(selected), 1.0 / len(selected))))
        if index % 32 == 0 or index == len(batches):
            print(f"Prepared {index}/{len(batches)} prompt conditions for the {'repeat' if preliminary else 'first'} fit", flush=True)
    return draws


def fit_heads(draws):
    answer = {}
    for arm, samples in draws.items():
        values = np.concatenate([part for part, _ in samples]).astype(np.float64)
        weights = np.concatenate([weight for _, weight in samples])
        weights /= weights.sum()
        scale = np.maximum(.04, np.sqrt(weights @ (values**2)))
        scaled = values / scale
        mean = weights @ scaled
        covariance = (scaled.T * weights) @ scaled + RIDGE * np.eye(scaled.shape[1])
        coefficient = np.zeros(384, dtype=np.float32)
        coefficient[PARTS[arm]] = np.linalg.solve(covariance, mean) / scale
        answer[arm] = {"weights": coefficient.tolist(), "rank_pairs": len(values)}
    return answer


def fit(bank, output):
    samples, manifest = parent.feature_manifest(bank / "inputs")
    load = parent.feature_reader(bank / "features", manifest)
    batches = grouped(samples["rows"])
    initial = fit_heads(fitting_rows(batches, load))
    learned = fit_heads(fitting_rows(batches, load, initial))
    controls = []
    for row in samples["rows"]:
        if row["role"] != "calibration" or row["truth"]:
            continue
        item = next(item for item in row["inputs"] if item["context"] == "short")
        grid = intervals.features(load(row, "short"))
        spans = files.read(bank / "features" / f"{item['sha256']}.json")
        controls.append({key: row[key] for key in ("reader", "condition", "source", "kind", "room")} |
                        {"pcm_sha256": item["sha256"], "readings": {arm: intervals.readings(grid, head, spans)
                                                                  for arm, head in learned.items()}})
    points = {arm: parent.calibrate([row | {"readings": row["readings"][arm]} for row in controls]) for arm in learned}
    roots = {str(row["reader"]) for row in samples["sources"]}
    locked = {"schema": SCHEMA, "design": DESIGN, "source_sha256": manifest["source_sha256"], "ridge": RIDGE,
              "source_readers": sorted(roots), "fit_sources": sorted({row["source"] for row in samples["rows"] if row["role"] == "fit"}),
              "held_observations_at_fit": 0, "code_sha256": parent.file_sha(Path(__file__)),
              "features_sha256": parent.file_sha(Path(intervals.__file__)), "encoder": manifest["encoder"],
              "speech_sha256": manifest["speech_sha256"], "heads": learned, "calibration": controls, "points": points}
    files.save(output, locked)
    print(json.dumps({"fitting_prompts": len(locked["fit_sources"]), "calibration_deliveries": len(controls),
                      "pairs": {key: head["rank_pairs"] for key, head in learned.items()}}, sort_keys=True), flush=True)


def readout(bank, prior, new):
    source = files.read(bank / "files.json")
    fields = ("id", "condition", "kind", "family", "group", "samples", "pcm_sha256", "scope", "truth", "room")
    for index, row in enumerate(prior["rows"], 1):
        digest = row["pcm_sha256"]
        folder = bank / digest
        file, speech = folder / "short.npy", folder / "speech.json"
        record = source[digest]
        if parent.file_sha(file) != record["features"]["short"] or parent.file_sha(speech) != record["speech_sha256"]:
            raise ValueError("Held interval features or speech activity differ from locked input")
        if [record["views"]["short"][0], sum(record["views"]["short"])] != row["scope"]:
            raise ValueError("The prior interval locations use a different source clock")
        if str(row["group"]) in new["source_readers"]:
            raise ValueError("A held speaker appears in fit or calibration")
        grid = intervals.features(np.load(file, allow_pickle=False))
        spans = files.read(speech)
        baseline, assessed = reference.predictions(row, prior["calibration"]["points"], "wider")
        yield {key: row[key] for key in fields} | {"current": [list(span) for span in baseline["current_plus_six"]],
                                                 "current_assessed": assessed["current_plus_six"],
                                                 "readings": {arm: intervals.readings(grid, head, spans) for arm, head in new["heads"].items()},
                                                 "feature_sha256": record["features"]["short"]}
        if index % 60 == 0 or index == len(prior["rows"]):
            print(f"Compared directly ranked intervals on {index}/{len(prior['rows'])} deliveries", flush=True)


def compare(bank, lock, output):
    from scripts.audio_interval_ranking_review import summarize, verify

    fitted = files.read(lock)
    if (fitted["schema"] != SCHEMA or fitted["features_sha256"] != parent.file_sha(Path(intervals.__file__))
            or fitted["code_sha256"] != parent.file_sha(Path(__file__))):
        raise ValueError("The comparison requires the same pre-fitted ranking code")
    history = bank / "prospective-result.json.gz"
    prior = files.read(history)
    if (prior["source_plan"] != files.read(bank / "prospective/freeze.json")
            or prior["fit"]["source_sha256"] != fitted["source_sha256"]):
        raise ValueError("Prior outcome data use a different source plan")
    expected = files.read(bank / "prospective-features/freeze.json")["source_sha256"]
    if expected != parent.file_sha(bank / "prospective/freeze.json"):
        raise ValueError("The exact feature cache differs from the held plan")
    rows = list(readout(bank / "prospective-features", prior, fitted))
    result = {"schema": SCHEMA, "design": DESIGN, "fit": fitted, "rows": rows,
              "fit_sha256": parent.file_sha(lock), "prior_sha256": parent.file_sha(history),
              "summary": summarize(rows, fitted["points"])}
    verify(result)
    files.save(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode, fields in (("fit", ("bank", "output")), ("compare", ("bank", "lock", "output"))):
        command = commands.add_parser(mode)
        for key in fields:
            command.add_argument(key, type=Path)
    args = vars(parser.parse_args())
    action = args.pop("mode")
    (fit if action == "fit" else compare)(**args)
