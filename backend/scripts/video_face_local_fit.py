"""Fit and compare localized face changes with source-group-separated calibration."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from scripts import video_common_sources_v89 as io

SCHEMA = "chipmunk.video-face-local-comparison/1"
METHODS = {"same_appearance": (0, 1), "relative_detail": (2, 3), "combined": (0, 1, 2, 3)}
POINTS = (0, 2)


def overlap(a, b):
    if a is None or b is None:
        return 0.
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))


def measured(journal):
    with journal.open() as stream:
        method = json.loads(next(stream))
        rows = [json.loads(line) for line in stream]
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("An input was measured more than once")
    return method, rows


def verify_manifest(header, rows, manifest):
    data = io.read(manifest)
    selected = {(row["file"], row["input_sha256"]) for row in rows}
    if io.sha(manifest) != header["manifest_sha256"] or selected != {(row["file"], row["input_sha256"]) for row in data["rows"]}:
        raise ValueError("Some scheduled videos were not measured against the exact manifest")


def training(rows):
    values = []
    for row in rows:
        candidates = row["proposals"]
        positives = [candidate for candidate in candidates if overlap(candidate[:2], row["truth"]) >= .50]
        negatives = [candidate for candidate in candidates if overlap(candidate[:2], row["truth"]) < .15]
        group = row["group"]
        for found in sorted(positives, key=lambda candidate: (-overlap(candidate[:2], row["truth"]), candidate[0]))[:2]:
            values.append((group, 1, found[2:]))
        if negatives:
            hard = {tuple(max(negatives, key=lambda candidate: candidate[column])) for column in range(2, 6)}
            for found in sorted(hard):
                values.append((group, -1, found[2:]))
    return values


def fit_heads(rows):
    import numpy as np

    choices = training(rows)
    labels = np.asarray([label for _, label, _ in choices])
    counts = Counter((group, label) for group, label, _ in choices)
    membership = Counter(label for _, label in counts)
    if set(membership) != {1, -1} or min(membership.values()) < 3:
        raise ValueError("Too few source groups have usable correct and incorrect intervals")
    balanced = np.asarray([len(choices) / (2 * membership[label] * counts[group, label]) for group, label, _ in choices])
    values = np.asarray([value for _, _, value in choices])
    heads = {}
    for name, columns in METHODS.items():
        x = values[:, columns]
        mean = np.average(x, weights=balanced, axis=0)
        scale = np.maximum(.005, np.sqrt(np.average((x - mean) ** 2, weights=balanced, axis=0)))
        x = (x - mean) / scale
        weights = np.linalg.solve((x.T * balanced) @ x + np.eye(len(columns)) * 12., x.T @ (balanced * labels))
        heads[name] = {"columns": columns, "center": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist()}
    return heads, {"intervals": len(choices), "source_groups": dict(membership)}


def score(values, head):
    selected = (values[column] for column in head["columns"])
    return math.fsum((x - center) * weight / scale for x, center, scale, weight in zip(
        selected, head["center"], head["scale"], head["weights"], strict=True))


def reading(row, head):
    candidates = ((score(values, head), a, b) for a, b, *values in row["proposals"])
    result = max(candidates, key=lambda item: (item[0], item[2] - item[1], -item[1]), default=None)
    if result is None:
        return None
    response, start, end = result
    before = start - 2
    during = start + 2 * ((end - start) // 4)
    pins = {item["index"]: item for item in row["anchors"]}
    return {"range": [start, end], "score": response,
            "anchors": {label: pins[i] for label, i in zip(("before", "during", "after"), (before, during, end), strict=True)},
            "ticks": [row["first_tick"] + i * row["step_tick"] for i in (start, end)], "time_base": row["time_base"]}


def readout(rows, heads):
    records = []
    for row in rows:
        record = {key: row[key] for key in ("file", "group", "family", "delivery", "source_sha256", "input_sha256",
                                           "truth", "edit_frames", "existing", "existing_assessed")}
        record["candidate"] = {name: reading(row, head) for name, head in heads.items()}
        records.append(record)
    return records


def selected(row, arm=None, point=None):
    previous = row["existing"]
    candidate = row["candidate"].get(arm)
    return previous or (candidate["range"] if candidate is not None and candidate["score"] > point else None)


def counts(rows, arm=None, point=None):
    result = Counter(planned=len(rows))
    for row in rows:
        previous = row["existing"]
        candidate = row["candidate"].get(arm)
        final = selected(row, arm, point)
        decided = row["existing_assessed"] or candidate is not None
        if row["truth"]:
            result["edits"] += 1
            result["edit_unassessed"] += not decided
            result["useful"] += overlap(final, row["truth"]) >= .50
            result["new_useful"] += overlap(final, row["truth"]) >= .50 and overlap(previous, row["truth"]) < .50
            result["exact"] += final == row["truth"]
            result["within_two_frames"] += final is not None and max(abs(x - y) for x, y in zip(final, row["truth"], strict=True)) <= 2
            result["misplaced"] += final is not None and overlap(final, row["truth"]) < .50
        else:
            result["controls"] += 1
            result["control_unassessed"] += not decided
            result["ordinary_flags"] += final is not None
            result["added_ordinary_flags"] += final is not None and previous is None
    return dict(result)


def calibration(rows):
    choices = {}
    for arm in METHODS:
        controls = sorted((row["candidate"][arm]["score"] for row in rows
                           if not row["truth"] and not row["existing"] and row["candidate"][arm]), reverse=True)
        if len(controls) <= max(POINTS):
            raise ValueError("Too few ordinary files have eligible calibration intervals")
        for budget in POINTS:
            point = controls[budget]
            choices[f"{arm}/{budget}"] = {"arm": arm, "point": point, "additional_control_budget": budget,
                                           "counts": counts(rows, arm, point)}
    return choices


def strongest(choices):
    def rank(result):
        c = result["counts"]
        return (c["new_useful"] - c["added_ordinary_flags"], c["new_useful"], -c["added_ordinary_flags"],
                c["within_two_frames"], -c["misplaced"], result["arm"] == "combined")

    return max(choices, key=lambda name: rank(choices[name]))


def require_disjoint(*banks):
    seen = set()
    for rows in banks:
        groups = {row["group"] for row in rows}
        if groups & seen or len({row["file"] for row in rows}) != len(rows):
            raise ValueError("Source groups or repeated measurements cross the evaluation roles")
        seen.update(groups)


def freeze(fit, fit_manifest, controls, control_manifest, output):
    train_method, training_rows = measured(fit)
    selection_method, selection_rows = measured(controls)
    verify_manifest(train_method, training_rows, fit_manifest)
    verify_manifest(selection_method, selection_rows, control_manifest)
    require_disjoint(training_rows, selection_rows)
    method = {key: train_method[key] for key in ("implementation_sha256", "readout_sha256", "frequency_sha256", "weights")}
    if any(selection_method[key] != value for key, value in method.items()):
        raise ValueError("Fit and calibration used different measurements")
    heads, sample_counts = fit_heads(training_rows)
    rows = readout(selection_rows, heads)
    choices = calibration(rows)
    chosen = strongest(choices)
    io.write(output, {"schema": SCHEMA, "method": method, "fit_code_sha256": io.sha(Path(__file__)),
                      "fit": io.sha(fit), "fit_groups": sorted({row["group"] for row in training_rows}),
                      "source_training_counts": sample_counts, "selection": io.sha(controls), "heads": heads,
                      "choices": choices, "selected": chosen, "selection_rows": rows})
    print(json.dumps({"selected": chosen, "current": counts(rows), "choices": choices}, indent=2))


def metrics(rows, lock):
    result = {}
    subsets = {"all": rows}
    for field in ("family", "delivery"):
        for value in sorted({row[field] for row in rows}):
            subsets[f"{field}/{value}"] = [row for row in rows if row[field] == value]
    for key, subset in subsets.items():
        result[key] = {"existing": counts(subset)} | {
            name: counts(subset, option["arm"], option["point"]) for name, option in lock["choices"].items()
        }
    return result


def assess(lock_path, journal, manifest, output):
    lock = io.read(lock_path)
    header, confirmation = measured(journal)
    verify_manifest(header, confirmation, manifest)
    if io.sha(Path(__file__)) != lock["fit_code_sha256"] or any(header[key] != value for key, value in lock["method"].items()):
        raise ValueError("Confirmation implementation differs from the freeze")
    fit = [{"group": group, "file": group} for group in lock["fit_groups"]]
    require_disjoint(fit, lock["selection_rows"], confirmation)
    rows = readout(confirmation, lock["heads"])
    results = metrics(rows, lock)
    report = {"lock": lock, "lock_sha256": io.sha(lock_path), "confirmation": header, "rows": rows, "metrics": results}
    with output.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as payload:
        payload.write(json.dumps(report, allow_nan=False, sort_keys=True).encode())
    views = {key: {name: vals[name] for name in ("existing", lock["selected"])} for key, vals in results.items()}
    print(json.dumps(views, indent=2))


def verify(report):
    with gzip.open(report, "rt") as stream:
        value = json.load(stream)
    lock = value["lock"]
    if lock["schema"] != SCHEMA or calibration(lock["selection_rows"]) != lock["choices"] or strongest(lock["choices"]) != lock["selected"]:
        raise ValueError("Stored operating points do not reproduce the calibration")
    if metrics(value["rows"], lock) != value["metrics"]:
        raise ValueError("Stored held counts do not reproduce the labeled input results")
    print(json.dumps(value["metrics"]["all"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, fields in (("freeze", ("fit", "fit_manifest", "calibration", "calibration_manifest", "output")),
                           ("assess", ("lock", "journal", "manifest", "output")), ("verify", ("receipt",))):
        sub = commands.add_parser(action)
        for field in fields:
            sub.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.fit, args.fit_manifest, args.calibration, args.calibration_manifest, args.output)
    elif args.action == "assess":
        assess(args.lock, args.journal, args.manifest, args.output)
    else:
        verify(args.receipt)
