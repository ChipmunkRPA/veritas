# SPDX-License-Identifier: Apache-2.0
"""Test short waveform-residual review on two retained competing-speech panels."""
from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
from collections import defaultdict
from pathlib import Path

from review_audio_short_codec_transfer import GRID, dump, held_metrics, load, require, rows_digest

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
CODEC = PANELS / "audio-short-codec-transfer-2026-11-12.json"
CONFIRMATION = PANELS / "audio-short-mp3-confirmation-2026-11-12.json"
SCHEMA = "chipmunk.audio-mp3-residual-transfer/1"
WINDOWS = GRID[2:]
RATE, SAMPLES = 16000, 96000
LAGS = (1, 2, 4, 8, 16, 32)
QUANTILES = (.25, .50, .75)
CASE_FIELDS = ("id", "group", "condition", "family", "truth")
COMPARISON = {"min_added_useful_total": 2, "min_added_useful_per_panel": 1, "max_added_ordinary_groups_total": 2}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def public_rows():
    codec, confirmation = load(CODEC), load(CONFIRMATION)
    return {"calibration": codec["calibration"], "transfer": codec["held"], "confirmation": confirmation["held"]}


def source_records(panel, run, public):
    rows = [row for row in load(run / "inputs.json")
            if panel == "confirmation" or row["role"] == ("held" if panel == "transfer" else panel)]
    require(len(rows) == len(public), "The retained files do not cover the earlier public panel")
    for row, label in zip(rows, public, strict=True):
        require(all(row[field] == label[field] for field in CASE_FIELDS), "A retained case label differs")
        delivery = row["delivery"] if panel == "confirmation" else row["deliveries"]["mp3"]
        path = run / delivery.get("file", delivery.get("path"))
        require(path.parent == run and sha(path) == delivery["bytes"]["sha256"]
                and delivery["clock"] == {"start_sample": 0, "end_sample": SAMPLES, "source_pts": 1105},
                "A carrier path, byte commitment, or sample clock differs")
        yield label, path, delivery["pcm"]["sha256"]


def pcm16(path, expected):
    import numpy as np

    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file",
                          "-i", f"file:{path.resolve()}", "-map", "0:a:0", "-f", "s16le", "-c:a", "pcm_s16le", "pipe:1"],
                         check=True, capture_output=True, timeout=8).stdout
    require(len(pcm) == 2 * SAMPLES and hashlib.sha256(pcm).hexdigest() == expected,
            "The MP3 decoder did not reproduce the committed samples")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768


def feature(samples):
    import numpy as np

    smooth = np.convolve(np.pad(samples, (2, 2), mode="edge"), [1/16, 4/16, 6/16, 4/16, 1/16], "valid")
    residual = samples - smooth
    original_frames = np.lib.stride_tricks.sliding_window_view(samples, 320)[::160]
    residual_frames = np.lib.stride_tricks.sliding_window_view(residual, 320)[::160]
    epsilon = 1e-12
    source_power = np.mean(original_frames ** 2, axis=1)
    residual_power = np.mean(residual_frames ** 2, axis=1)
    values = [10 * np.log10((residual_power + epsilon) / (source_power + epsilon))]
    for lag in LAGS:
        left, right = residual_frames[:, :-lag], residual_frames[:, lag:]
        numerator = np.sum(left * right, axis=1)
        denominator = np.sqrt(np.sum(left ** 2, axis=1) * np.sum(right ** 2, axis=1))
        values.append(numerator / np.maximum(denominator, epsilon))
    return np.quantile(values, QUANTILES, axis=1).reshape(-1)


def selected_features(panel, run, public):
    for label, path, digest in source_records(panel, run, public):
        samples = pcm16(path, digest)
        admission = label["scores"]["mp3"][2:]
        yield label, [feature(samples[start:end]) if allowed is not None else None
                      for (start, end), allowed in zip(WINDOWS, admission, strict=True)]


def split(rows):
    groups = sorted({row["group"] for row in rows},
                    key=lambda name: hashlib.sha256(f"mp3-residual-fit:{name}".encode()).digest())
    require(len(groups) == 16 and all(row["truth"] is None for row in rows), "Ordinary calibration groups changed")
    return groups[:8], groups[8:]


def fit(records, groups):
    import numpy as np

    samples = [np.asarray(records[group]) for group in groups]
    require(all(len(views) for views in samples), "A fitted group has no admitted source views")
    mean = np.mean([np.mean(views, axis=0) for views in samples], axis=0)
    cov = np.mean([(views - mean).T @ (views - mean) / len(views) for views in samples], axis=0)
    diag = np.maximum(np.diag(cov), 1e-9)
    inverse = np.linalg.inv(.50 * cov + .50 * np.diag(diag))
    return {"mean": mean.tolist(), "inverse": inverse.tolist()}


def scores(views, model):
    import numpy as np

    inverse = np.asarray(model["inverse"])
    output = []
    for view in views:
        if view is None:
            output.append(None)
            continue
        delta = view - model["mean"]
        output.append(float(delta @ inverse @ delta))
    return output


def point(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].extend(value for value in row["residual"] if value is not None)
    require(len(groups) == 8 and all(groups.values()), "A point group has no admitted source views")
    maximum = sorted((max(views) for views in groups.values()), reverse=True)
    require(maximum[0] > maximum[1], "The ordinary operating point is tied")
    return maximum[1]


def score_row(label, values):
    return {field: label[field] for field in CASE_FIELDS} | {"residual": values}


def freeze(calibration, confirmation, output):
    output.mkdir(mode=0o700)
    public = public_rows()["calibration"]
    fit_groups, point_groups = split(public)
    records, fitting = list(selected_features("calibration", calibration, public)), defaultdict(list)
    for label, views in records:
        if label["group"] in fit_groups:
            fitting[label["group"]].extend(view for view in views if view is not None)
    model = fit(fitting, fit_groups)
    selection = [score_row(label, scores(views, model)) for label, views in records if label["group"] in point_groups]
    lock = {"schema": SCHEMA, "runner_sha256": sha(Path(__file__)), "codec_receipt_sha256": sha(CODEC),
            "confirmation_receipt_sha256": sha(CONFIRMATION), "fit_groups": fit_groups, "point_groups": point_groups,
            "carrier_inputs_sha256": {"transfer": sha(calibration / "inputs.json"),
                                     "confirmation": sha(confirmation / "inputs.json")},
            "fit_sha256": rows_digest(model), "point_rows_sha256": rows_digest(selection), "cutoff": point(selection),
            "comparison": COMPARISON}
    dump(output / "point.json", {"lock": lock, "model": model, "selection": selection})
    print("Wrote the eight-group fitted reference and separate eight-group operating point.")


def compare(rows, public, cutoff, baseline_cutoff):
    require(len(rows) == len(public), "The evaluation file count differs")
    reformatted = []
    for row, label in zip(rows, public, strict=True):
        require(all(row[field] == label[field] for field in CASE_FIELDS), "An evaluation label differs")
        candidate = row["residual"]
        require(len(candidate) == len(WINDOWS) and all(
            (value is None and prior is None) or
            (prior is not None and type(value) in {float, int} and math.isfinite(value) and value >= 0)
            for value, prior in zip(candidate, label["scores"]["mp3"][2:], strict=True)), "Invalid window admission or response")
        reformatted.append(row | {"scores": {"mp3": [None, None, *candidate]}})
    primary = held_metrics(public, "mp3", "combined", baseline_cutoff)
    residual = held_metrics(reformatted, "mp3", "combined", cutoff)
    return {"primary": primary, "residual": residual,
            "added_useful_groups": sorted(set(residual["useful_groups"]) - set(primary["useful_groups"])),
            "added_ordinary_groups": sorted(set(residual["false_groups"]) - set(primary["false_groups"]))}


def summarize(receipt):
    lock = receipt["lock"]
    public = public_rows()
    fit_groups, point_groups = split(public["calibration"])
    require(lock["schema"] == SCHEMA and lock["runner_sha256"] == sha(Path(__file__))
            and lock["codec_receipt_sha256"] == sha(CODEC) and lock["confirmation_receipt_sha256"] == sha(CONFIRMATION)
            and lock["fit_groups"] == fit_groups and lock["point_groups"] == point_groups
            and lock["point_rows_sha256"] == rows_digest(receipt["selection"]) and lock["cutoff"] == point(receipt["selection"])
            and receipt["held_sha256"] == rows_digest(receipt["held"]) and lock["comparison"] == COMPARISON,
            "The predeclared sources or point changed")
    baseline_cutoff = load(CONFIRMATION)["cutoffs"]["combined"]
    calibration = [row for row in public["calibration"] if row["group"] in point_groups]
    compare(receipt["selection"], calibration, lock["cutoff"], baseline_cutoff)
    results = {panel: compare(receipt["held"][panel], public[panel], lock["cutoff"], baseline_cutoff)
               for panel in ("transfer", "confirmation")}
    useful = [len(outcome["added_useful_groups"]) for outcome in results.values()]
    false = sum(len(outcome["added_ordinary_groups"]) for outcome in results.values())
    return {"panels": results, "total_new_useful": sum(useful), "total_new_ordinary": false,
            "screen_passed": (sum(useful) >= COMPARISON["min_added_useful_total"]
                              and min(useful) >= COMPARISON["min_added_useful_per_panel"]
                              and false <= COMPARISON["max_added_ordinary_groups_total"])}


def measure(codec, confirmation, output, target):
    frozen = load(output / "point.json")
    lock, model = frozen["lock"], frozen["model"]
    require(lock["runner_sha256"] == sha(Path(__file__)) and lock["fit_sha256"] == rows_digest(model)
            and lock["carrier_inputs_sha256"] == {"transfer": sha(codec / "inputs.json"),
                                                  "confirmation": sha(confirmation / "inputs.json")},
            "The study or fit changed after freezing")
    public = public_rows()
    held = {panel: [score_row(label, scores(views, model)) for label, views in selected_features(panel, directory, public[panel])]
            for panel, directory in (("transfer", codec), ("confirmation", confirmation))}
    receipt = {"lock": lock, "selection": frozen["selection"], "held": held, "held_sha256": rows_digest(held)}
    receipt["summary"] = summarize(receipt)
    dump(target, receipt)
    return receipt["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("target", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    if args.phase != "review" and (args.codec is None or args.confirmation is None):
        parser.error("Native calibration and measurement require --codec and --confirmation")
    if args.phase == "measure" and args.run is None:
        parser.error("Measurement requires --run with its frozen operating point")
    if args.phase == "freeze":
        freeze(args.codec, args.confirmation, args.target)
    elif args.phase == "measure":
        print(measure(args.codec, args.confirmation, args.run, args.target))
    else:
        receipt = load(args.target)
        summary = summarize(receipt)
        require(summary == receipt["summary"], "Published outcome differs from the recomputed counts")
        print(summary)


if __name__ == "__main__":
    main()
