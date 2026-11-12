# SPDX-License-Identifier: Apache-2.0
"""Evaluate inferred local writing ranks on exact saved overlapping model inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logit

from app.detectors.text.extract import from_string
from app.detectors.text.segment import segment
from scripts import writing_checkpoint_agreement as prior
from scripts.writing_agreement_document_format_transfer import RECEIPT, sha, source_inputs

SCHEMA = "chipmunk.writing-context-mixture-transfer/1"
RIDGE, SMOOTH = .10, .25
ARMS = ("primary", "secondary", "agreement", "inverted_primary", "inverted_secondary", "inverted_agreement", "inverted_peak")
BUDGETS = (0, 1, 2, 4, 6)
METHOD = {
    "role": "Existing grammar-seams panel sets genuine operating points; group-disjoint option-order panel confirms. All sources and responses have prior project exposure. No new model inference.",
    "matrix": "Rows are observed scored contexts, columns are exact extracted sentences. Uniform 1/k weight on the 2 or 3 sentences read by each window. Verify every existing context hash and focus mapping.",
    "solve": "Map each checkpoint's raw response to a clipped logit (1e-6 to 1-1e-6). Use least squares against context responses, .10 ridge toward the excerpt mean logit and .25 squared first differences between adjacent sentences. Do not tune penalties.",
    "arms": "Raw first and second focus scores, minimum of both; corresponding inferred scores; and inferred minimum minus its excerpt median. No output is a calibrated probability.",
    "location": "Retain the single highest focus for each arm, ties within 1e-10 choose the first. Check 0.30 non-whitespace IoU for both the focus sentence and its existing exact full model context; all arms use the same geometry.",
    "points": "Strictly exceed the selection panel's ranked genuine group maxima at budgets 0, 1, 2, 4 and 6; evaluate those points unchanged. Separately compare rankings at identical held false-group counts using held maxima. Count actual false groups after ties.",
    "paired": "An edit wins its source comparison only if its maximum exceeds both its original and matched human splice. Report publisher and generator diversity for useful held changes.",
    "limits": "Model logits need not be linear averages. Ridge estimates are diagnostic ranks with mixed support and cannot be presented as model measurements of single sentences.",
}


def bind(panels):
    rows, responses, parent_marks = source_inputs(panels)
    groups = {panel: {row["group"] for row in rows if row["panel"] == panel} for panel in prior.PANEL_NAMES}
    prior.require(not set.intersection(*groups.values()), "The score-setting and comparison source groups overlap")
    lock = {"schema": SCHEMA, "method": METHOD, "runner_sha256": sha(Path(__file__)), "parent_receipt_sha256": sha(RECEIPT),
            "selection": prior.PANEL_NAMES[0], "confirmation": prior.PANEL_NAMES[1],
            "rows": len(rows), "group_counts": {panel: len(values) for panel, values in groups.items()},
            "panel_rows": dict(Counter(row["panel"] for row in rows)), "new_model_forwards": 0}
    return lock, rows, responses, parent_marks


def incidence(row):
    sentences = [(s.start, s.end) for s in segment(from_string(row["text"]))]
    lookup = {pair: index for index, pair in enumerate(sentences)}
    windows = row["windows"]
    prior.require(len(lookup) == len(sentences) == len(windows), "Context incidence has different source geometry")
    matrix = np.zeros((len(windows), len(sentences)))
    for i, window in enumerate(windows):
        spans = list(map(tuple, window["spans"]))
        prior.require(len(spans) in (2, 3) and len(set(spans)) == len(spans) and sentences[i] in spans
                      and prior.text_sha(" ".join(row["text"][a:b] for a, b in spans)) == window["sha256"],
                      "Original focus or context input no longer matches the source")
        matrix[i, [lookup[span] for span in spans]] = 1/len(spans)
    return sentences, matrix


def reconstruct(matrix, scores, *, audit):
    identity = np.eye(matrix.shape[1])
    differences = np.diff(identity, axis=0)
    average = np.broadcast_to(scores.mean(axis=0), scores.shape)
    if audit:
        stacked = np.vstack((matrix, np.sqrt(RIDGE)*identity, np.sqrt(SMOOTH)*differences))
        target = np.vstack((scores, np.sqrt(RIDGE)*average, np.zeros((len(differences), scores.shape[1]))))
        return np.linalg.lstsq(stacked, target, rcond=None)[0]
    system = matrix.T @ matrix + RIDGE*identity + SMOOTH*(differences.T @ differences)
    right = matrix.T @ scores + RIDGE*average
    local = np.linalg.solve(system, right)
    prior.require(np.max(abs(system @ local - right)) < 1e-10, "Inversion did not solve the frozen objective")
    return local


def measure(row, responses, *, audit):
    sentences, matrix = incidence(row)
    score = logit(np.clip([(w["current"], responses[w["sha256"]]["score"]) for w in row["windows"]], 1e-6, 1-1e-6))
    local = reconstruct(matrix, score, audit=audit)
    agreement = local.min(axis=1)
    values = (*score.T, score.min(axis=1), *local.T, agreement, agreement-np.median(agreement))
    observed = {}
    for arm, points in zip(ARMS, values, strict=True):
        best = int(np.flatnonzero(points >= points.max()-1e-10)[0])
        focus, context = sentences[best], row["windows"][best]["spans"]
        human = row["kind"] in prior.HUMAN
        observed[arm] = {"score": round(float(points[best]), 10), "index": best, "focus": list(focus), "context": context,
                         "useful_focus": not human and prior.useful(row["text"], row["truth"], [focus]),
                         "useful_context": not human and prior.useful(row["text"], row["truth"], context)}
    return {**{key: row[key] for key in ("panel", "publisher", "group", "kind", "text_sha256")}, "arms": observed}


def point(rows, arm, budget):
    maximum = {}
    for row in rows:
        if row["kind"] in prior.HUMAN:
            key, score = row["group"], row["arms"][arm]["score"]
            maximum[key] = max(maximum.get(key, -np.inf), score)
    return sorted(maximum.values(), reverse=True)[budget]


def report_point(rows, arm, cut):
    hits = [row for row in rows if row["arms"][arm]["score"] > cut]
    false = [r for r in hits if r["kind"] in prior.HUMAN]
    useful = [r for r in hits if r["arms"][arm]["useful_focus"]]
    return {"genuine_false_groups": len({r["group"] for r in false}), "genuine_false_files": len(false),
            "focus_useful": len(useful), "context_useful": sum(row["arms"][arm]["useful_context"] for row in hits),
            "useful_publishers": sorted({r["publisher"] for r in useful}), "useful_generators": sorted({r["kind"] for r in useful})}


def rankings(rows, arm):
    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], {})[row["kind"]] = row["arms"][arm]["score"]
    totals = Counter()
    for group in by_group.values():
        prior.require(group.keys() >= prior.HUMAN, "Matched genuine controls absent")
        cut = max(group[role] for role in prior.HUMAN)
        for role, score in group.items():
            if role not in prior.HUMAN:
                totals.update(compared=1, above_both=int(score > cut))
    return dict(totals)


def summarize(results, parent_marks):
    selection, held = ([r for r in results if r["panel"] == panel] for panel in prior.PANEL_NAMES)
    prior_held = [r for r in parent_marks if r["panel"] == prior.PANEL_NAMES[1]]
    parent = {arm: {"context_useful": sum(r["useful"][arm] for r in prior_held),
                    "genuine_false_groups": len({r["group"] for r in prior_held if r["kind"] in prior.HUMAN and r["marks"][arm]})}
              for arm in ("current", "agreement")}
    return {"held_panel": {"edits": sum(r["kind"] not in prior.HUMAN for r in held),
                           "genuine_files": sum(r["kind"] in prior.HUMAN for r in held),
                           "genuine_groups": len({r["group"] for r in held if r["kind"] in prior.HUMAN})},
            "prior_multi_window_held": parent,
            "unthresholded": {arm: {"focus_useful": sum(r["arms"][arm]["useful_focus"] for r in held),
                                    "context_useful": sum(r["arms"][arm]["useful_context"] for r in held),
                                    "paired": rankings(held, arm)} for arm in ARMS},
            "held_frozen_points": {str(n): {arm: report_point(held, arm, point(selection, arm, n))
                                           for arm in ARMS} for n in BUDGETS},
            "held_equal_false_groups": {str(n): {arm: report_point(held, arm, point(held, arm, n))
                                                for arm in ARMS} for n in BUDGETS}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "run", "audit"))
    cli.add_argument("output", type=Path)
    cli.add_argument("panels", type=Path)
    args = cli.parse_args()
    lock, rows, scores, marks = bind(args.panels)
    freeze = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        prior.save(freeze, lock)
        return
    prior.require(lock == prior.load(freeze), "Writing mixture protocol changed")
    data = [measure(row, scores, audit=args.phase == "audit") for row in rows]
    summary = summarize(data, marks)
    ledger, result = (args.output / name for name in ("observations.json", "result.json"))
    if args.phase == "run":
        prior.save(ledger, data)
        prior.save(result, {**lock, **summary, "freeze_sha256": sha(freeze), "observations_sha256": sha(ledger)})
        print(summary["held_panel"])
    else:
        expected = prior.load(result)
        prior.require(prior.load(ledger) == data and expected["observations_sha256"] == sha(ledger)
                      and all(expected[key] == value for key, value in summary.items()), "Augmented least-squares replay differs")
        print(f"Augmented least squares reproduced {len(data)} exact source rankings and all false-group comparisons.")


if __name__ == "__main__":
    main()
