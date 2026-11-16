# SPDX-License-Identifier: MIT
"""Compare pooled and measured-quantization face calibration on fresh paired sources."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import median

from scripts.video_face_pair_calibration import _normalizers, calculate, paired_flags, records
from scripts.video_face_temporal_fit import overview, quantile, read, save, sha

POINTS = {"tighter": .90, "wider": .80}
MODES = ("pooled", "quality_cutoff", "quality_scale_and_cutoff")
REGIMES = ("direct", "tandem")
DESIGN = {
    "input": "Use the median initial luma QP of inter pictures with exact verified PTS. Unseen QP outside calibration support abstains.",
    "base": "Mean of standardized detector responses on the same face, then clip median with at least four paired samples.",
    "anchors": "The ordinary direct and tandem deliveries define each endpoint. Their observed median QP anchors continuous interpolation.",
    "comparisons": "Refit pooled normalization on all eligible originals. Compare it with a QP-conditioned cutoff alone, "
                   "and with QP-conditioned arm centers, scales and cutoff. Use the same held decoded crops.",
    "points": "Hold out each entire source group in the ordinary development panel. Use 90th and 80th percentile "
              "operating points per delivery, or pooled per-delivery for the shared arm.",
    "input_blindness": "Held scores use measured QP alone to choose or interpolate the conditional parameters. "
                       "The known encoder setting is used for reporting only.",
}


def bind(manifest, features, journals):
    quality = read(features)
    assert quality["manifest_sha256"] == sha(manifest)
    measured = {row["file"]: row for row in quality["rows"]}
    assert len(measured) == len(quality["rows"])
    used = set()
    headers, result = [], []
    for journal in journals:
        header, rows = records(manifest, journal)
        headers.append(header)
        for row in rows:
            name = row["file"]
            source = measured[name]
            assert name not in used and source["sha256"] == row["sha256"]
            used.add(name)
            quantization = source["quantization"]
            if quantization is not None:
                frames = {pts: (index, kind, qp) for index, (pts, kind, qp) in enumerate(quantization["frames"])}
                assert len(frames) == len(quantization["frames"]) == 100
                values = [qp for _, kind, qp in frames.values() if kind in ("P", "B")]
                assert median(values) == quantization["inter_picture_median"]
                if proof := row["reference"]:
                    assert proof["time_base"] == source["time_base"]
                    assert all(frames[sample["pts"]][0] == sample["frame_index"] for sample in proof["samples"])
            result.append(row | {"qp": quantization["inter_picture_median"] if quantization is not None else None})
    assert used == set(measured)
    return headers, result


def measure_terms(ordinary):
    grouped = {name: [row for row in ordinary if row["delivery"] == name] for name in REGIMES}
    low, high = (median(row["qp"] for row in grouped[name]) for name in REGIMES)
    assert low < high
    qp = [row["qp"] for row in ordinary]
    return {
        "pooled": _normalizers(ordinary),
        "by_delivery": {name: _normalizers(grouped[name]) for name in REGIMES},
        "anchors": [low, high],
        "observed": [min(qp), max(qp)],
    }


def blend(a, b, position):
    return a + position * (b - a)


def position(qp, terms):
    if qp is None or not terms["observed"][0] <= qp <= terms["observed"][1]:
        return None
    low, high = terms["anchors"]
    return min(1., max(0., (qp - low) / (high - low)))


def normalize(row, terms, mode):
    if mode != "quality_scale_and_cutoff":
        return terms["pooled"]
    weight = position(row["qp"], terms)
    if weight is None:
        return None
    arms = [terms["by_delivery"][key] for key in REGIMES]
    return [[blend(a, b, weight) for a, b in zip(left, right, strict=True)]
            for left, right in zip(*arms, strict=True)]


def response(row, terms, mode):
    if mode != "pooled" and position(row["qp"], terms) is None:
        return None
    return calculate(row, normalize(row, terms, mode))["same_frame_agreement"]


def calibrate(rows):
    original = [row for row in rows if row["family"] == "ordinary" and row["qp"] is not None and row["reference"]]
    source_groups = {row["group"] for row in original}
    ordinary = defaultdict(list)
    for name in sorted(source_groups):
        fold = [row for row in original if row["group"] != name]
        terms = measure_terms(fold)
        for row in original:
            if row["group"] != name:
                continue
            for mode in MODES:
                result = response(row, terms, mode)
                if result is not None:
                    ordinary[mode, "pool" if mode == "pooled" else row["delivery"]].append(result)
    cutoffs = {mode: {group: {point: quantile(ordinary[mode, group], percentile) for point, percentile in POINTS.items()}
                      for group in (("pool",) if mode == "pooled" else REGIMES)} for mode in MODES}
    return {"terms": measure_terms(original), "source_groups": len(source_groups), "ordinary_deliveries": len(original),
            "leave_out_scores": {"/".join(key): values for key, values in ordinary.items()}, "cutoffs": cutoffs}


def evaluate(rows, fit):
    terms = fit["terms"]
    played = {}
    for row in rows:
        queries = {}
        for mode in MODES:
            value = response(row, terms, mode)
            if value is None:
                queries[mode] = {"response": None, "flags": {}}
                continue
            operating = fit["cutoffs"][mode]
            cuts = operating["pool"] if mode == "pooled" else {
                point: blend(operating["direct"][point], operating["tandem"][point], position(row["qp"], terms))
                for point in POINTS
            }
            queries[mode] = {"response": value, "cutoffs": cuts, "flags": {p: value > cut for p, cut in cuts.items()}}
        played[row["file"]] = queries
    return played, overview(rows, played, paired_flags(rows), dict.fromkeys(MODES))


def compare_weights(headers):
    assert headers and all(value["weights"] == headers[0]["weights"] for value in headers)
    assert all(value["detector"] == headers[0]["detector"] for value in headers)


def freeze(args):
    first, first_rows = bind(args.first_manifest, args.first_quality, (args.first_dev, args.first_held))
    second, second_rows = bind(args.second_manifest, args.second_quality, (args.second_held,))
    training = first_rows + second_rows
    calibration = calibrate(training)
    compare_weights(first + second)
    assert not ({row["group"] for row in first_rows} & {row["group"] for row in second_rows})
    fresh = read(args.held_manifest)
    assert not ({row["group"] for row in training} & {row["group"] for row in fresh["rows"]})
    pins = {"fit": sha(Path(__file__)), "trace": sha(Path(__file__).parents[1] / "app/evidence/h264_quantization.py"),
            "reference": sha(Path(__file__).with_name("video_face_temporal_reference.py"))}
    save(args.output, {"design": DESIGN, "code_sha256": pins,
                       "held_manifest_sha256": sha(args.held_manifest), "training": training,
                       "training_headers": first + second, "calibration": calibration})
    print("Frozen", calibration["source_groups"], "source groups,", calibration["ordinary_deliveries"], "ordinary deliveries.")
    print("QP anchors", calibration["terms"]["anchors"], "observed", calibration["terms"]["observed"])


def report(args):
    data = read(args.protocol)
    assert data["design"] == DESIGN and data["held_manifest_sha256"] == sha(args.held_manifest)
    assert data["code_sha256"]["fit"] == sha(Path(__file__))
    assert data["code_sha256"]["trace"] == sha(Path(__file__).parents[1] / "app/evidence/h264_quantization.py")
    assert data["code_sha256"]["reference"] == sha(Path(__file__).with_name("video_face_temporal_reference.py"))
    header, test = bind(args.held_manifest, args.held_quality, (args.held_journal,))
    compare_weights(data["training_headers"] + header)
    assert header[0]["split"] == "held"
    assert not ({row["group"] for row in test} & {row["group"] for row in data["training"]})
    scores, summary = evaluate(test, data["calibration"])
    save(args.output, data | {"held": test, "held_header": header[0], "scores": scores, "summary": summary})
    print_summary(summary)


def print_summary(summary):
    for cell, result in summary.items():
        print(cell, result["total"], "incumbent", result["reference"])
        for mode in MODES:
            print(" ", mode, result[mode])


def verify(args):
    data = read(args.receipt)
    fit = calibrate(data["training"])
    assert fit == data["calibration"]
    scores, overview_ = evaluate(data["held"], fit)
    assert scores == data["scores"] and overview_ == data["summary"]
    print_summary(overview_)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, action, paths in (
        ("freeze", freeze, ("first_manifest", "first_quality", "first_dev", "first_held",
                            "second_manifest", "second_quality", "second_held", "held_manifest", "output")),
        ("report", report, ("held_manifest", "held_quality", "held_journal", "protocol", "output")),
        ("verify", verify, ("receipt",)),
    ):
        command = commands.add_parser(name)
        for path in paths:
            command.add_argument(path, type=Path)
        command.set_defaults(handler=action)
    args = parser.parse_args()
    args.handler(args)
