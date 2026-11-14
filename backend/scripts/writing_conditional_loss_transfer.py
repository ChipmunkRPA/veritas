"""Replay conditional writing-pattern loss against the frozen raw ratio."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from scripts import writing_missed_passage_transfer as causal

SOURCE = causal.pool.PANEL_DIR / "writing-missed-passage-transfer-2026-11-14.json"
SOURCE_SHA256 = "63c78faf85837a28e2e8f2dbb7df4837c8ef5e10c3fc86363ed57494f6c3aae5"
SCHEMA = "chipmunk.writing-conditional-loss-transfer/1"
ARMS = ("raw_ratio", "conditional_loss", "relative_conditional_loss", "absolute_and_relative_loss")
BUDGETS = (20, 40)
METHOD = {
    "fit": "Fit observed loss against cross-entropy and log token count on the original fit sentences only. "
           "Use a ridge of 0.3 on the two slopes and use expected minus observed loss as the conditional response.",
    "arms": "Compare the raw ratio, the conditional response, its contrast to the stronger of both adjacent sentences, "
            "and the sum of the absolute response and this contrast.",
    "points": "Use two budgets among ordinary calibration works with no primary flag. Set the point by the "
              "rank of each such work's highest response. Keep all prior primary findings.",
    "scope": "The source observations were measured earlier. The existing single-upload writing-pattern ratio "
             "is a proxy for serving. Compare changes in both useful locations and ordinary review volume.",
}


def features(item):
    return None if item is None else (float.fromhex(item["obs_hex"]), float.fromhex(item["cross_hex"]), math.log(item["tokens"]))


def fit(rows):
    fit_rows = [values for row in rows if row["role"] == "fit" and row["truth"] is None
                for item in row["ratio"] if (values := features(item)) is not None]
    if len(fit_rows) < 20:
        raise ValueError("Insufficient genuine fit measurements")
    means = [math.fsum(values) / len(fit_rows) for values in zip(*fit_rows, strict=True)]
    d = [[v - avg for v, avg in zip(row, means, strict=True)] for row in fit_rows]
    observed, cross, length = zip(*d, strict=True)
    a = math.fsum(x*x for x in cross) + .3
    b = math.fsum(x*y for x, y in zip(cross, length, strict=True))
    d = math.fsum(y*y for y in length) + .3
    r = math.fsum(y*x for y, x in zip(observed, cross, strict=True))
    s = math.fsum(y*x for y, x in zip(observed, length, strict=True))
    det = a*d - b*b
    entropy_coef, length_coef = (d*r - b*s) / det, (a*s - b*r) / det
    return [means[0] - entropy_coef * means[1] - length_coef * means[2], entropy_coef, length_coef]


def scores(row, coefficients):
    intercept, beta_entropy, beta_length = coefficients
    raw, conditional = [], []
    for record in row["ratio"]:
        item = features(record)
        raw.append(-item[0] / item[1] if item else None)
        conditional.append(intercept + beta_entropy * item[1] + beta_length * item[2] - item[0] if item else None)
    relative = [None] * len(conditional)
    joint = relative.copy()
    for index, (left, center, right) in enumerate(zip(conditional, conditional[1:], conditional[2:]), 1):
        if left is not None and center is not None and right is not None:
            relative[index] = center - max(left, right)
            joint[index] = center + relative[index]
    return dict(zip(ARMS, (raw, conditional, relative, joint), strict=True))


def report():
    if causal.pool.file_hash(SOURCE) != SOURCE_SHA256:
        raise ValueError("Frozen loss measurements differ")
    document = causal.read(SOURCE)
    # Keep the source experiment's registry commitment. Later unrelated model
    # registrations do not change these already frozen loss observations.
    if document["definition"] != causal.manifest() | {"model_manifest_sha256": document["definition"]["model_manifest_sha256"]}:
        raise ValueError("Frozen method or source roster changed")
    rows = list(causal.combine(document["readings"]))
    coef = fit(rows)
    control_scores = [scores(row, coef) for row in rows if row["role"] == "calibration"
                      and row["truth"] is None and not causal.pool.baseline(row)]
    if len(control_scores) < 10:
        raise ValueError("Insufficient ordinary calibration inputs")
    thresholds, outcomes = {}, {}
    for budget in BUDGETS:
        allowed = len(control_scores) * budget // 100
        points = {arm: sorted((max((value for value in readings[arm] if value is not None), default=-math.inf)
                               for readings in control_scores), reverse=True)[allowed] for arm in ARMS}
        if not all(map(math.isfinite, points.values())):
            raise ValueError("A calibration arm lacked support")
        thresholds[str(budget)] = points
        for role in ("held", "transfer"):
            counts = {name: Counter() for name in ("primary", "published_union", *ARMS)}
            for row in rows:
                if row["role"] != role:
                    continue
                base = causal.pool.baseline(row)
                assessments = {"primary": base, "published_union": causal.measured_spans(row)}
                for arm, values in scores(row, coef).items():
                    assessments[arm] = base + [span for span, reading in zip(row["spans"], values, strict=True)
                                                if reading is not None and reading > points[arm]]
                for name, spans in assessments.items():
                    outcome = counts[name]
                    if row["truth"] is None:
                        outcome.update(ordinary=1, ordinary_flagged=bool(spans))
                    else:
                        hit = any(causal.pool.useful(row, span) for span in spans)
                        outcome.update(edited=1, useful=hit, misplaced=bool(spans) and not hit)
            outcomes[f"{budget}/{role}"] = {name: dict(result) for name, result in counts.items()}
    return {"schema": SCHEMA, "method": METHOD, "source_sha256": SOURCE_SHA256, "fit_coefficients": coef,
            "ordinary_calibration_works": len(control_scores), "points": thresholds, "results": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    output = report()
    if args.action == "run":
        causal.pool.write(args.receipt, output)
    elif json.loads(args.receipt.read_text()) != output:
        raise ValueError("The study no longer reproduces")
    print(json.dumps(output["results"], indent=2))


if __name__ == "__main__":
    main()
