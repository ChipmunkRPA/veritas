# SPDX-License-Identifier: MIT
"""Calibrate and compare ways to combine the app's two face readings."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

from scripts.video_face_temporal_fit import POINTS, overview, quantile, read, save, sha

METHODS = ("rescaled_maximum", "same_frame_maximum", "same_frame_agreement", "persistent_agreement")
DESIGN = {
    "normalization": "Pool both deliveries of each development original. Scale each detector's clip medians by the "
                     "median and 90th-minus-10th percentile of the ordinary responses, with a .02 minimum scale.",
    "reference": "Freeze the current paired whole-face decision and both native responses per sampled frame.",
    "rescaled_maximum": "Normalize the two clip medians and use their maximum. This isolates recalibration.",
    "same_frame_maximum": "Take the larger normalized response on each jointly measured face, then take the median.",
    "same_frame_agreement": "Average the two normalized responses on the same face, then take the median.",
    "persistent_agreement": "Use .75 times that median plus .25 times the lower quartile of the same-face readings.",
    "coverage": "At least four native responses per check, or four paired native responses for same-frame comparisons.",
    "points": "Calibrate each method using leave-one-source-group-out ordinary scores and group maxima over both "
              "deliveries. Use frozen 90th and 80th percentiles of these maxima. No codec label is an input.",
}


def records(folder, journal):
    manifest = read(folder)
    header, *rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert header["manifest_sha256"] == sha(folder)
    targets = {row["file"]: row for row in manifest["rows"] if row["role"] == header["split"] and "file" in row}
    assert set(targets) == {row["file"] for row in rows} and len(targets) == len(rows)
    data = []
    for row in rows:
        target = targets[row["file"]]
        assert target["sha256"] == row["sha256"]
        item = {key: target[key] for key in ("file", "group", "family", "delivery", "sha256")} | {"reference": row["review"]}
        if row["review"]:
            assert len(row["review"]["samples"]) == 6
            for sample in item["reference"]["samples"]:
                sample.pop("box", None)
        data.append(item)
    return header, data


def paired_readings(row):
    proof = row["reference"]
    if proof is None:
        return []
    return [[float.fromhex(value) if value is not None else None for value in sample["responses_hex"]]
            for sample in proof["samples"]]


def _normalizers(genuine):
    columns = [[], []]
    for row in genuine:
        for index, arm in enumerate(zip(*paired_readings(row), strict=True)):
            values = [value for value in arm if value is not None]
            if len(values) >= 4:
                columns[index].append(median(values))
    if any(len(col) < 6 for col in columns):
        return None
    return [[median(col), max(.02, quantile(col, .90) - quantile(col, .10))] for col in columns]


def calculate(row, terms):
    result = dict.fromkeys(METHODS)
    if terms is None:
        return result
    readings = [[(value - median_value) / scale if value is not None else None
                 for value, (median_value, scale) in zip(sample, terms, strict=True)] for sample in paired_readings(row)]
    if not readings:
        return result
    arms = [[value for value in arm if value is not None] for arm in zip(*readings, strict=True)]
    if all(len(arm) >= 4 for arm in arms):
        result["rescaled_maximum"] = max(median(arm) for arm in arms)
    both = [(a, b) for a, b in readings if a is not None and b is not None]
    if len(both) >= 4:
        average = [(a + b) / 2 for a, b in both]
        result["same_frame_maximum"] = median(max(a, b) for a, b in both)
        result["same_frame_agreement"] = median(average)
        result["persistent_agreement"] = .75 * median(average) + .25 * quantile(average, .25)
    return result


def calibrate(dev):
    originals = defaultdict(list)
    for row in dev:
        if row["family"] == "ordinary":
            originals[row["group"]].append(row)
    maximum = defaultdict(list)
    for group, current in originals.items():
        terms = _normalizers([row for name, delivery in originals.items() if name != group for row in delivery])
        left_out = [calculate(row, terms) for row in current]
        for method in METHODS:
            measured = [score[method] for score in left_out if score[method] is not None]
            if measured:
                maximum[method].append(max(measured))
    terms = _normalizers([row for values in originals.values() for row in values])
    return {"normalizers": terms, "groups": len(originals), "ordinary_group_maxima": dict(maximum),
            "points": {method: {key: quantile(values, percentile) for key, percentile in POINTS.items()}
                       for method, values in maximum.items()}}


def paired_flags(held):
    from app.evidence.whole_face_binding import REVIEW_POINT, paired_score

    flags = {}
    for row in held:
        if proof := row["reference"]:
            score = paired_score(*zip(*paired_readings(row), strict=True))
            assert score is not None and score.hex() == proof["score_hex"] and (score > REVIEW_POINT) == proof["flagged"]
            flags[row["file"]] = proof["flagged"]
        else:
            flags[row["file"]] = None
    return flags


def evaluate(held, fit):
    scores = {}
    for row in held:
        values = calculate(row, fit["normalizers"])
        scores[row["file"]] = {method: {"response": signal, "flags": {point: signal > cut for point, cut in
                               fit["points"].get(method, {}).items()} if signal is not None else {}}
                               for method, signal in values.items()}
    return scores, overview(held, scores, paired_flags(held), dict.fromkeys(METHODS))


def lock(manifest, dev, held_manifest, output):
    header, development = records(manifest, dev)
    fit = calibrate(development)
    assert header["split"] == "development" and set(fit["points"]) == set(METHODS)
    save(output, {"design": DESIGN, "code_sha256": sha(Path(__file__)), "dev_inputs_sha256": sha(dev),
                  "dev_manifest_sha256": sha(manifest), "held_manifest_sha256": sha(held_manifest),
                  "development": development, "dev_models": header, "calibration": fit})
    print(f"Locked {len(METHODS)} combinations before the new held model readings.")


def report(held_manifest, held, frozen, receipt):
    header, test = records(held_manifest, held)
    fixed = read(frozen)
    assert fixed["design"] == DESIGN and fixed["code_sha256"] == sha(Path(__file__))
    assert fixed["held_manifest_sha256"] == sha(held_manifest) and header["split"] == "held"
    assert fixed["dev_models"]["weights"] == header["weights"]
    assert not ({row["group"] for row in test} & {row["group"] for row in fixed["development"]})
    scores, summary = evaluate(test, fixed["calibration"])
    save(receipt, fixed | {"held": test, "held_models": header, "scores": scores, "summary": summary})
    print(json.dumps(summary, sort_keys=True, indent=2))


def verify(path):
    data = read(path)
    assert calibrate(data["development"]) == data["calibration"]
    scores, summary = evaluate(data["held"], data["calibration"])
    assert scores == data["scores"] and summary == data["summary"]
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for phase, paths in (
        ("freeze", ("manifest", "development", "held_manifest", "output")),
        ("report", ("held_manifest", "held", "frozen", "receipt")),
        ("verify", ("receipt",)),
    ):
        action = commands.add_parser(phase)
        for path in paths:
            action.add_argument(path, type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        lock(args.manifest, args.development, args.held_manifest, args.output)
    elif args.action == "report":
        report(args.held_manifest, args.held, args.frozen, args.receipt)
    else:
        verify(args.receipt)
