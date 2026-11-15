# SPDX-License-Identifier: MIT
"""Compare dense phone evidence with the sparse lip-timing check on fixed native clocks."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from app.detectors.audio.phone_events import (
    CENTER,
    MODEL_REVISION,
    STRIDE,
    WEIGHT_PIN,
    collapse,
    phone_posteriors,
)
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_viseme_transfer as prior

SCHEMA = "chipmunk.video-lip-dense-posteriors/1"
CARRIER_SCHEMA = "chipmunk.video-lip-dense-carrier/1"
CARRIER_SOURCE = prior.prior.PANEL / "video-lip-viseme-carrier-2026-11-15.json.gz"
UNIT = 1_000_000
WINDOW = (1, 2, 3, 2, 1)
RIDGE = .08
MIN_SPEECH = .10
APERTURE_ARMS = ("aperture", "normalized_aperture")
MOTION_ARMS = ("aperture_and_motion", "normalized_aperture_and_motion")
ARMS = APERTURE_ARMS + MOTION_ARMS
METHOD = {
    "sound": "Retain the sum of probabilities for each of four visible speech classes plus all other phone "
             "probability on the verified 20 ms clock. Average five consecutive hops with triangular weights.",
    "fit": "Fit a ridge prediction of normalized mouth aperture using the current and adjacent 60 ms "
           "acoustic context, on the first-cohort development originals only. Give each recording equal weight.",
    "comparison": "Compare the predicted and observed aperture at common measured samples for all candidate "
                  "lags. The motion arm also compares centered changes over 80 ms. Use the same best constant "
                  "lag versus a short signed change and two supporting flanks as the existing study.",
    "units": "Keep the initial squared-aperture readings as a control. Normalize both error terms by the "
             "sparse comparison's squared aperture scale before applying its ambiguity margin.",
    "point": "Set each operating point from maximum scores per speaker in the second development cohort, "
             "including ordinary filtering, gain, and constant-delay controls. Evaluate both held cohorts "
             "without changing those points, and retain the sparse-event comparator.",
}
MODEL = {"revision": MODEL_REVISION, "weights_sha256": WEIGHT_PIN}


def reader(checkpoint):
    posterior, vocabulary, device = phone_posteriors(checkpoint)
    groups = [[vocabulary[name] for name in names] for names in timing.PHONES.values()]
    included = {index for ids in groups for index in ids}
    excluded = {vocabulary[token] for token in ("<pad>", "<s>", "</s>", "<unk>", "|") if token in vocabulary}
    other = sorted(set(vocabulary.values()) - included - excluded)
    labels = {vocabulary[name]: label for label, names in timing.PHONES.items() for name in names}

    def measure(pcm, *, include_events=False):
        probabilities = posterior(pcm)
        masses = [probabilities[:, ids].sum(1).tolist() for ids in (*groups, other)]
        series = [[round(UNIT * value) for value in row] for row in zip(*masses, strict=True)]
        return (series, collapse(probabilities, labels)) if include_events else series

    return measure, device


def capture(args):
    rows = prior.timings()["rows"]
    measure, device = reader(args.checkpoint)
    with ZipFile(args.sources / "lombardgrid_audio.zip") as archive, args.output.open("x", buffering=1) as out:
        for i, row in enumerate(rows, 1):
            spec = row["members"]["audio"]
            member = archive.read(spec["name"])
            prior.phone.require(prior.phone.digest(member) == spec["sha256"], "Publisher waveform changed")
            cases = {}
            for role, pcm in prior.parent.treatments(prior.phone.pcm(member), row["edit"]).items():
                if prior.is_fit(row) and role != "original":
                    continue
                digest = prior.phone.digest(pcm.tobytes())
                prior.phone.require(digest == row["cases"][role]["pcm_sha256"], "Staged waveform changed")
                cases[role] = {"pcm_sha256": digest, "groups": measure(pcm)}
            out.write(json.dumps({"source": row["source"], "device": device, "cases": cases}, sort_keys=True) + "\n")
            if i % 8 == 0:
                print(f"Measured {i}/{len(rows)} native speech recordings.", flush=True)


def acoustic(groups):
    nearby = [None] * len(groups)
    for i in range(2, len(groups) - 2):
        interval = groups[i - 2:i + 3]
        values = [sum(weight * row[j] for weight, row in zip(WINDOW, interval, strict=True)) for j in range(5)]
        mass = sum(values)
        if mass:
            nearby[i] = [v / mass for v in values] + [mass / (UNIT * sum(WINDOW))]
    return nearby


def features(nearby, index):
    start, stop = index - 3, index + 3
    if start < 2 or stop >= len(nearby) - 2:
        return None
    values = [nearby[i] for i in (start, index, stop)]
    if any(point is None for point in values) or values[1][-1] < MIN_SPEECH:
        return None
    return [1.] + [value for point in values for value in point]


def fit(rows, captured):
    import numpy as np

    pieces = []
    for row in rows:
        if not prior.is_fit(row) or (pictures := timing.observations(row)) is None:
            continue
        groups = captured[row["source"]]["cases"]["original"]["groups"]
        nearby = acoustic(groups)
        samples = []
        for i in range(len(groups)):
            picture = timing.nearest(pictures, CENTER + STRIDE * i)
            reading = features(nearby, i)
            if picture is not None and reading is not None:
                samples.append((reading, picture[0]))
        if not samples:
            continue
        x = np.asarray([pair[0] for pair in samples], dtype=np.float64)
        y = np.asarray([pair[1] for pair in samples], dtype=np.float64)
        pieces.append((x.T @ x / len(samples), x.T @ y / len(samples), len(samples), row["speaker"]))
    if len(pieces) < 12 or len({x[3] for x in pieces}) < 4:
        raise ValueError("Too few sources with overlapping speech and visible mouth observations")
    xx = sum(x[0] for x in pieces) / len(pieces)
    xy = sum(x[1] for x in pieces) / len(pieces)
    regularizer = np.eye(len(xy)) * RIDGE
    regularizer[0, 0] = 0.
    weights = np.linalg.solve(xx + regularizer, xy)
    return {"weights": [round(float(v), 9) for v in weights], "examples": sum(x[2] for x in pieces), "clips": len(pieces)}


def predict(groups, fitted):
    nearby = acoustic(groups)
    predictions = []
    for i in range(len(groups)):
        row = features(nearby, i)
        estimate = None if row is None else min(1.5, max(-.5, math.fsum(a * b for a, b in zip(row, fitted, strict=True))))
        predictions.append(None if estimate is None else round(UNIT * estimate))
    return predictions


def match_scores(row, predictions):
    points = timing.observations(row)
    if points is None:
        return {arm: [] for arm in ARMS}
    output = {arm: [] for arm in ARMS}
    # At 20 ms posterior cadence, use every other hop. For velocity compare
    # the adjacent two hops on both sides, i.e. an 80 ms interval.
    for i in range(2, len(predictions) - 2, 2):
        if predictions[i] is None:
            continue
        sample = CENTER + i * STRIDE
        values = [timing.nearest(points, sample - lag) for lag in timing.LAGS]
        if any(value is None for value in values):
            continue
        aperture = predictions[i] / UNIT
        static = [min(1.5, abs(level - aperture)) ** 2 for level, _ in values]
        frames = [index for _, index in values]
        for arm, scale in zip(APERTURE_ARMS, (1., timing.SCALE ** 2), strict=True):
            output[arm].append((sample, "aperture", [round(-timing.SCORE_UNITS * x / scale) for x in static], frames))
        left, right = predictions[i - 2], predictions[i + 2]
        if left is None or right is None:
            continue
        derivative = (right - left) / UNIT
        low = [timing.nearest(points, sample - 2 * STRIDE - lag) for lag in timing.LAGS]
        high = [timing.nearest(points, sample + 2 * STRIDE - lag) for lag in timing.LAGS]
        if any(a is None or b is None or a[1] >= b[1] for a, b in zip(low, high, strict=True)):
            continue
        motion = [min(1.5, abs(b[0] - a[0] - derivative)) ** 2 for a, b in zip(low, high, strict=True)]
        for arm, scale in zip(MOTION_ARMS, (1., timing.SCALE ** 2), strict=True):
            output[arm].append((sample, "motion",
                [round(-timing.SCORE_UNITS * (x + y) / scale) for x, y in zip(static, motion, strict=True)], frames))
    return output


def case_scores(row, values, expected):
    prediction = values["prediction"]
    samples = row["samples"]
    sample_count = (samples - 2 * CENTER) // STRIDE + 1
    if (values["pcm_sha256"] != expected or len(prediction) != sample_count
            or any(v is not None and (type(v) is not int or not -UNIT // 2 <= v <= UNIT * 3 // 2) for v in prediction)):
        raise ValueError("Soundtrack identity, cadence or aperture range differs")
    return {arm: timing.hypotheses(inputs, samples) for arm, inputs in match_scores(row, prediction).items()}


def evaluate(rows, dense):
    sparse = prior.load(prior.prior.PANEL / "video-lip-viseme-transfer-2026-11-15.json.gz")
    by_source = {row["source"]: row for row in sparse["acoustic"]["rows"]}
    refs = sparse["result"]["references"]
    sparse_point = sparse["result"]["points"]["all_visible"]["tight"]
    normal = {arm: defaultdict(float) for arm in ARMS}
    held = []
    for row in rows:
        if prior.is_fit(row):
            continue
        block = dense[row["source"]]
        if set(block) != set(row["cases"]):
            raise ValueError("Missing soundtrack delivery")
        for role, values in block.items():
            scores = case_scores(row, values, row["cases"][role]["pcm_sha256"])
            shift = prior.parent.EDITS.get(role)
            kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
            if prior.is_calibration(row):
                if not shift:
                    for arm, score in scores.items():
                        key = row["speaker"]
                        normal[arm][key] = max(normal[arm][key], score.get("score", 0.))
                continue
            existing = timing.hypotheses(timing.compatible(row, by_source[row["source"]]["cases"][role]["phones"], refs),
                                         row["samples"])
            held.append((row, role, kind, shift, {"sparse": existing} | scores))
    points = {}
    for arm, groups in normal.items():
        maxima = sorted(groups.values(), reverse=True)
        if len(maxima) != 4:
            raise ValueError("Calibration must retain four independent speaker maxima")
        points[arm] = {"tight": maxima[0], "wider": maxima[1]}
    points["sparse"] = {"tight": sparse_point, "wider": sparse["result"]["points"]["all_visible"]["wider"]}
    return {"points": points, **summarize(held, points)}


def summarize(held, points):
    counts, evidence = defaultdict(Counter), {}
    for row, role, kind, shift, readings in held:
        for level in ("tight", "wider"):
            flags = {name: score.get("score", 0.) > points[name][level] for name, score in readings.items()}
            correct = {name: bool(shift and flags[name] and prior.parent.useful(
                score["span"], score["change"], row["edit"], shift)) for name, score in readings.items()}
            for name, score in readings.items():
                counts[f"{name}/{level}/{kind}"].update(
                    files=1, assessed=score["status"] not in {"insufficient_events", "ambiguous_change", "incomplete_context"},
                    flagged=flags[name], useful=correct[name], misplaced=bool(shift and flags[name] and not correct[name]),
                    union_useful=correct[name] or correct["sparse"], union_flagged=flags[name] or flags["sparse"],
                    eligible_events=score["eligible_events"])
                if shift and correct[name] and level == "tight":
                    key = f"{name}/{row['source']}/{role}"
                    evidence[key] = timing.picture_evidence(row["frames"], score, row["samples"])
    return {"outcomes": dict(counts), "evidence": evidence}


def seal(path, receipt):
    with path.open("xb") as output:
        output.write(gzip.compress((json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode(), mtime=0))


def report(args):
    base = prior.timings()
    captured = {}
    for entry in map(json.loads, args.raw.read_text().splitlines()):
        if entry["source"] in captured:
            raise ValueError("Repeated source recording")
        captured[entry["source"]] = entry
    if set(captured) != {r["source"] for r in base["rows"]}:
        raise ValueError("The source recordings differ")
    fitted = fit(base["rows"], captured)
    dense = {}
    for row in base["rows"]:
        if prior.is_fit(row):
            continue
        cases = captured[row["source"]]["cases"]
        dense[row["source"]] = {role: {"pcm_sha256": case["pcm_sha256"],
                                       "prediction": predict(case["groups"], fitted["weights"])}
                                for role, case in cases.items()}
    result = evaluate(base["rows"], dense)
    receipt = {"schema": SCHEMA, "method": METHOD, "source_sha256": prior.phone.sha(prior.prior.SOURCE), "model": MODEL,
               "device": sorted({item["device"] for item in captured.values()}), "fit": fitted, "predictions": dense,
               "result": result}
    seal(args.output, receipt)
    print(json.dumps(result["outcomes"], sort_keys=True, indent=2))


def review(path):
    receipt = prior.load(path)
    rows = prior.timings()["rows"]
    if (receipt["schema"] != SCHEMA or receipt["method"] != METHOD or receipt["model"] != MODEL
            or receipt["source_sha256"] != prior.phone.sha(prior.prior.SOURCE)
            or set(receipt["predictions"]) != {row["source"] for row in rows if not prior.is_fit(row)}
            or receipt["result"] != evaluate(rows, receipt["predictions"])):
        raise ValueError("Dense posterior evaluation differs from its replay")
    return receipt


def carrier_results(rows, dense, points, references):
    held = defaultdict(list)
    if set(dense) != {row["source"] for row in rows}:
        raise ValueError("The held carrier source roster changed")
    for row in rows:
        values = dense[row["source"]]
        if set(values) != set(prior.CODECS):
            raise ValueError("Both PCM and AAC predictions are required")
        for codec, deliveries in row["cases"].items():
            if set(values[codec]) != set(deliveries):
                raise ValueError("A held carrier delivery is missing")
            for role, case in deliveries.items():
                scores = case_scores(row, values[codec][role], case["sha256"])
                sparse = timing.hypotheses(timing.compatible(row, case["phones"], references), row["samples"])
                shift = prior.parent.EDITS.get(role)
                kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
                held[codec].append((row, role, kind, shift, {"sparse": sparse} | scores))
    return {codec: summarize(cases, points) for codec, cases in held.items()}


def transfer(args):
    primary = review(args.primary)
    source = prior.load(CARRIER_SOURCE)
    measure, device = reader(args.checkpoint)
    rows = source["rows"]
    dense = {}
    with (ZipFile(args.sources / "lombardgrid_audio.zip") as archive,
          TemporaryDirectory(prefix="chipmunk-dense-lip-") as scratch):
        for i, row in enumerate(rows, 1):
            spec = row["members"]["audio"]
            member = archive.read(spec["name"])
            prior.phone.require(prior.phone.digest(member) == spec["sha256"], "Publisher waveform changed")
            values = dense[row["source"]] = {codec: {} for codec in prior.CODECS}
            for role, wave in prior.parent.treatments(prior.phone.pcm(member), row["edit"]).items():
                submitted = prior.prior.aac_clocked(wave, Path(scratch) / "carrier.m4a")
                for codec, pcm in zip(prior.CODECS, (wave, submitted), strict=True):
                    digest = prior.phone.digest(pcm.tobytes())
                    prior.phone.require(digest == row["cases"][codec][role]["sha256"], "Carrier waveform changed")
                    values[codec][role] = {"pcm_sha256": digest,
                                           "prediction": predict(measure(pcm), primary["fit"]["weights"])}
            print(f"Reconstructed and measured PCM and AAC for {i}/{len(rows)} held recordings.", flush=True)
    result = carrier_results(rows, dense, primary["result"]["points"], source["references"])
    seal(args.output, {"schema": CARRIER_SCHEMA, "primary_sha256": prior.phone.sha(args.primary),
                      "source_sha256": prior.phone.sha(CARRIER_SOURCE), "model": MODEL,
                      "device": device, "predictions": dense, "result": result})
    print(json.dumps({codec: scores["outcomes"] for codec, scores in result.items()}, sort_keys=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    native = phases.add_parser("capture")
    native.add_argument("output", type=Path)
    native.add_argument("--checkpoint", type=Path, required=True)
    native.add_argument("--sources", type=Path, required=True)
    replay = phases.add_parser("report")
    replay.add_argument("raw", type=Path)
    replay.add_argument("output", type=Path)
    phases.add_parser("review").add_argument("receipt", type=Path)
    carrier = phases.add_parser("transfer")
    carrier.add_argument("primary", type=Path)
    carrier.add_argument("output", type=Path)
    carrier.add_argument("--sources", type=Path, required=True)
    carrier.add_argument("--checkpoint", type=Path, required=True)
    carrier_review = phases.add_parser("review-transfer")
    carrier_review.add_argument("primary", type=Path)
    carrier_review.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "capture":
        capture(args)
    elif args.phase == "report":
        report(args)
    elif args.phase == "review":
        print(json.dumps(review(args.receipt)["result"]["outcomes"], sort_keys=True, indent=2))
    elif args.phase == "transfer":
        transfer(args)
    else:
        primary = review(args.primary)
        source = prior.load(CARRIER_SOURCE)
        receipt = prior.load(args.receipt)
        if (receipt["schema"] != CARRIER_SCHEMA or receipt["primary_sha256"] != prior.phone.sha(args.primary)
                or receipt["source_sha256"] != prior.phone.sha(CARRIER_SOURCE) or receipt["model"] != MODEL
                or receipt["result"] != carrier_results(source["rows"], receipt["predictions"],
                                                       primary["result"]["points"], source["references"])):
            raise ValueError("Held carrier evaluation differs from its replay")
        print(json.dumps({codec: entries["outcomes"] for codec, entries in receipt["result"].items()}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
