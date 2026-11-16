# SPDX-License-Identifier: MIT
"""Calibrate temporal face checks by source group and replay their decisions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

VIEWS = ("fixed", "flow", "both")
FAMILIES = ("inswapper", "uniface")
POINTS = {"tighter": .90, "wider": .80}
MIN_WINDOWS = 2
RULE = {
    "scope": "Whole-clip face-swap screening on fixed 640x360 H.264 deliveries; one face track.",
    "fits": "The two views separately and their concatenation. Fit both generators together, then each alone.",
    "ordinary_scale": "Median and 90th-minus-10th percentile on ordinary development deliveries with a .02 floor.",
    "learning": "Match each fake to its ordinary source at the same delivery quality. Fit pairwise logistic differences "
                "with equal group weight and .25 L2 regularization, 160 steps at .10.",
    "points": "Leave each development group entirely out, then take the .90 and .80 quantiles of ordinary group maxima.",
    "coverage": "At least two fully measured windows; incomplete checks are reported separately.",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def save(path, value):
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    with path.open("xb") as stream:
        stream.write(gzip.compress(encoded, mtime=0) if path.suffix == ".gz" else encoded + b"\n")


def quantile(values, probability):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * probability) - 1)]


def vector(row, view):
    windows = row.get("windows", ())
    if len(windows) < MIN_WINDOWS:
        return None
    arms = ("fixed", "flow") if view == "both" else (view,)
    return tuple(median(values) for arm in arms
                 for values in zip(*(entry["measurements"][arm] for entry in windows), strict=True))


def standardize(values, center, scale):
    return tuple(max(-4., min(4., (value - a) / b)) for value, a, b in zip(values, center, scale, strict=True))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def fit(rows, view, train, exclude=None):
    known = defaultdict(dict)
    for row in rows:
        if row["group"] != exclude and "file" in row and (values := vector(row, view)) is not None:
            known[row["group"]][row["delivery"], row["family"]] = values
    genuine = [x for group in known.values() for (_, arm), x in group.items() if arm == "ordinary"]
    if len(genuine) < 6:
        return None
    ordinary_columns = list(zip(*genuine, strict=True))
    center = [median(values) for values in ordinary_columns]
    scale = [max(.02, quantile(values, .9) - quantile(values, .1)) for values in ordinary_columns]
    groups = []
    for group in known.values():
        contrasts = []
        for (delivery, family), raw in group.items():
            ordinary = group.get((delivery, "ordinary"))
            if family in train and ordinary is not None:
                positive = standardize(raw, center, scale)
                negative = standardize(ordinary, center, scale)
                contrasts.append(tuple(a - b for a, b in zip(positive, negative, strict=True)))
        if contrasts:
            groups.append(contrasts)
    if len(groups) < 6:
        return None
    weights = [0.] * len(center)
    for _ in range(160):
        gradient = [.25 * coef for coef in weights]
        for group in groups:
            for pair in group:
                factor = 1. / (len(group) * len(groups) * (1. + math.exp(dot(weights, pair))))
                for i, value in enumerate(pair):
                    gradient[i] -= factor * value
        weights = [old - .10 * update for old, update in zip(weights, gradient, strict=True)]
    return {"center": center, "scale": scale, "weights": weights, "fit_groups": len(groups)}


def score(model, values):
    if model is None or values is None:
        return None
    return dot(model["weights"], standardize(values, model["center"], model["scale"]))


def calibration(rows):
    groups = sorted({row["group"] for row in rows})
    profiles = {}
    for view in VIEWS:
        for training in (FAMILIES, *((family,) for family in FAMILIES)):
            name = view + "/" + "+".join(training)
            ordinary = []
            for group in groups:
                model = fit(rows, view, training, exclude=group)
                values = [value for row in rows if row["group"] == group and row["family"] == "ordinary"
                          if (value := score(model, vector(row, view))) is not None]
                if values:
                    ordinary.append(max(values))
            model = fit(rows, view, training)
            profiles[name] = {"model": model, "ordinary_group_maxima": ordinary,
                              "points": {key: quantile(ordinary, fraction) for key, fraction in POINTS.items()}
                              if model and len(ordinary) >= 8 else {}}
    return profiles


def candidates(held, profiles):
    choices = {}
    for row in held:
        if "file" not in row:
            continue
        path = row["file"]
        for name, configuration in profiles.items():
            view = name.split("/")[0]
            value = score(configuration["model"], vector(row, view))
            choices.setdefault(path, {})[name] = {
                "response": value,
                "flags": {point: value > cut for point, cut in configuration["points"].items()} if value is not None else {},
            }
    return choices


def baseline(rows, reference):
    from app.evidence.whole_face_binding import REVIEW_POINT, paired_score

    delivered = {row["file"]: row for row in rows if "file" in row}
    output = {}
    for item in reference:
        bound = delivered[item["file"]]
        assert item["sha256"] == bound["sha256"] and item["file"] not in output
        proof = item["review"]
        assert (item["status"] == "ok") == bool(proof)
        if proof:
            readings = [[float.fromhex(value) if value is not None else None for value in row["responses_hex"]]
                        for row in proof["samples"]]
            value = paired_score(*zip(*readings, strict=True))
            assert value is not None and value.hex() == proof["score_hex"]
            assert (value > REVIEW_POINT) == proof["flagged"]
            output[item["file"]] = proof["flagged"]
        else:
            output[item["file"]] = None
    assert set(output) == set(delivered)
    return output


def overview(rows, scores, reference, profiles):
    per = defaultdict(list)
    for row in rows:
        if "file" in row:
            per[row["delivery"], row["family"]].append(row)
    output = {}
    for (delivery, family), subset in sorted(per.items()):
        ref = [reference[row["file"]] for row in subset]
        details = {"total": len(subset), "reference": {"flags": sum(item is True for item in ref),
                    "unassessed": sum(item is None for item in ref)}}
        for name in profiles:
            results = [scores[row["file"]][name] for row in subset]
            points = {}
            for point in POINTS:
                selected = [case["flags"].get(point) for case in results]
                points[point] = {"flags": sum(item is True for item in selected),
                                 "unassessed": sum(item is None for item in selected),
                                 "added": sum(new is True and current is not True
                                              for new, current in zip(selected, ref, strict=True)),
                                 "reference_only": sum(new is not True and current is True
                                                       for new, current in zip(selected, ref, strict=True)),
                                 "union": sum(new is True or current is True for new, current in zip(selected, ref, strict=True))}
            details[name] = points
        output[f"{delivery}/{family}"] = details
    return output


def freeze(dev_path, destination):
    dev = read(dev_path)
    assert dev["split"] == "development"
    paths = ("video_face_temporal_panel", "video_face_temporal_delivery", "video_face_temporal_readout",
             "video_face_temporal_reference", "video_face_temporal_fit")
    methods = {name: sha(Path(__file__).with_name(name + ".py")) for name in paths}
    profiles = calibration(dev["rows"])
    save(destination, {"rule": RULE, "code_sha256": methods, "development_sha256": sha(dev_path),
                       "delivery_manifest_sha256": dev["manifest_sha256"], "profiles": profiles})
    print("Locked", len(profiles), "profiles before reading held temporal outputs.")


def summarize(dev_path, held_path, frozen_path, ref_path, output):
    dev, held, lock = (read(path) for path in (dev_path, held_path, frozen_path))
    assert lock["development_sha256"] == sha(dev_path) and held["split"] == "held"
    assert all(sha(Path(__file__).with_name(name + ".py")) == checksum for name, checksum in lock["code_sha256"].items())
    assert dev["manifest_sha256"] == held["manifest_sha256"] == lock["delivery_manifest_sha256"]
    assert not ({row["group"] for row in dev["rows"]} & {row["group"] for row in held["rows"]})
    reference_header, *reference = [json.loads(line) for line in ref_path.read_text().splitlines()]
    assert reference_header["manifest_sha256"] == held["manifest_sha256"] and reference_header["split"] == "held"
    profiles = lock["profiles"]
    assert calibration(dev["rows"]) == profiles
    reference_flags = baseline(held["rows"], reference)
    scores = candidates(held["rows"], profiles)
    summary = overview(held["rows"], scores, reference_flags, profiles)
    save(output, {"protocol": lock, "reference_header": reference_header, "reference": reference,
                  "development": dev, "held": held, "candidates": scores, "summary": summary})
    print_summary(summary)


def print_summary(summary):
    for cell, row in summary.items():
        print(cell, row["total"], "incumbent", row["reference"])
        for name in sorted(key for key in row if "/" in key):
            print(" ", name, row[name])


def verify(receipt):
    data = read(receipt)
    dev, held = data["development"]["rows"], data["held"]["rows"]
    profiles = calibration(dev)
    assert profiles == data["protocol"]["profiles"]
    scores = candidates(held, profiles)
    assert scores == data["candidates"]
    summary = overview(held, scores, baseline(held, data["reference"]), profiles)
    assert summary == data["summary"]
    print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="action", required=True)
    first = phases.add_parser("freeze")
    first.add_argument("development", type=Path)
    first.add_argument("destination", type=Path)
    second = phases.add_parser("summarize")
    for key in ("development", "held", "frozen", "reference", "output"):
        second.add_argument(key, type=Path)
    third = phases.add_parser("verify")
    third.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.development, args.destination)
    elif args.action == "summarize":
        summarize(args.development, args.held, args.frozen, args.reference, args.output)
    else:
        verify(args.receipt)
