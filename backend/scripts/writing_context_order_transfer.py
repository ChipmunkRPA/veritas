# SPDX-License-Identifier: MIT
"""Measure whether a single writing signal survives changing only nearby context order."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from scripts import writing_joint_tail_transfer as prior

pool = prior.pool
SCHEMA = "chipmunk.writing-context-order-transfer/1"
PARENT = pool.PANEL_DIR / "writing-joint-tail-matched-2026-11-14.json"
TRANSFER = pool.PANEL_DIR / "writing-joint-tail-transfer-2026-11-14.json"
VIEWS = ("adjacent", "reversed", "isolated")
CANDIDATES = ("adjacent", "order_agreement", "order_and_isolation", "order_and_separation")
SHARES = {"tight": .10, "moderate": .25, "broad": .50}
DESIGN = {
    "readings": "Use the same center sentence and same Apache-2.0 checkpoint in three contexts: native neighbor order, "
                "the two neighbors swapped, and the center alone. Reduce the head to exactly the center tokens in each view. "
                "Counterfactual inputs are used only in the experiment. They are never submitted as source prose.",
    "reference": "Per genre and view, normalize with the empirical distribution of ordinary work maxima for each of "
                 "three source-word length ranges (up to 18, 19–30, 31+). Require at least 12 distinct ordinary works per bin. "
                 "The reference percentile is the number strictly lower divided by n+1.",
    "rank": "Compare the native percentile, the smaller of native and reversed percentiles, and the smallest of all three. "
            "The fourth rule subtracts the strongest other center's order "
            "agreement from the proposed center. All policies require a unique strongest center.",
    "points": "Calibrate the maximum eligible result per file on a second disjoint set of ordinary works. Use strict "
              "exceedance at 10%, 25%, and 50% empirical control targets. No setting is a public confidence threshold.",
    "held": "Replay the two source-disjoint news/abstract rewrite cohorts and compare against the published classifier/ratio "
            "union proxy and raw same-center agreement. Count retained, lost and added exact useful locations and ordinary flags. "
            "All panels had prior project exposure. No training or held tuning is performed.",
}


def committed():
    return {"matched": pool.file_hash(PARENT), "transfer": pool.file_hash(TRANSFER)}


def indexed(split):
    filename = TRANSFER if split == "transfer" else PARENT
    role = "held" if split == "transfer" else split
    return [row for row in prior.sources.read(filename)["rows"] if row["role"] == role]


def inputs(row):
    sentences = [row["text"][a:b] for a, b in row["spans"]]
    for index in prior.HELD_CENTERS:
        left, focus, right = sentences[index-1:index+2]
        contexts = ((left, focus, right), (right, focus, left), (focus,))
        for context in contexts:
            start = len(context[0]) + 1 if len(context) == 3 else 0
            yield " ".join(context), [(start, start + len(focus))]


def compare_source(row, known):
    keys = ("sha256", "group", "domain", "position", "spans", "truth", "generator", "role")
    if any(row[key] != known[key] for key in keys):
        raise ValueError("The new score does not belong to the frozen source")


def roster_rows(path, split):
    role = "held" if split == "transfer" else split
    for row, source in zip((r for r in prior.sources.read(path)["rows"] if r["role"] == role), indexed(split), strict=True):
        compare_source(row, source)
        if pool.sha(row["text"]) != row["sha256"]:
            raise ValueError("Native input text changed")
        yield row


def readout_inputs(row, samples):
    text, spans = row["text"], row["spans"]
    data = []
    for i, center in enumerate(prior.HELD_CENTERS):
        a, b = spans[center]
        offset = i * len(VIEWS)
        data.append({"index": center, "words": len(text[a:b].split()),
                     "sha256": [pool.sha(item[0]) for item in samples[offset:offset + len(VIEWS)]]})
    return data


def measure(args):
    import torch

    from app.detectors.text.classifier import FourClassTextClassifier

    for name, digest in pool.BASELINE_ASSETS.items():
        if pool.file_hash(pool.SNAPSHOT / name) != digest:
            raise ValueError("Checkpoint bytes differ from the pinned version")
    rows = list(roster_rows(args.roster, args.phase))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_num_threads(2)
    classifier = FourClassTextClassifier.from_snapshot(pool.SNAPSHOT, device)
    records = []
    for index, row in enumerate(rows, 1):
        samples = list(inputs(row))
        scores = pool.score_batch(samples, classifier)
        readings = readout_inputs(row, samples)
        for i, reading in enumerate(readings):
            offset = i * len(VIEWS)
            reading["measured"] = [r[1][0][0] if r[1][0] is not None else None for r in scores[offset:offset + len(VIEWS)]]
        records.append({"source": row["sha256"], "centers": readings})
        if index % 16 == 0:
            print(args.phase, index, len(rows), flush=True)
    pool.write(args.path, {"schema": SCHEMA, "split": args.phase, "device": device, "parents": committed(), "records": records})


def bound(receipt, split):
    if receipt["schema"] != SCHEMA or receipt["split"] != split or receipt["parents"] != committed():
        raise ValueError("A measured panel differs from the frozen study")
    for row, scores in zip(indexed(split), receipt["records"], strict=True):
        centers = scores["centers"]
        if (row["sha256"] != scores["source"] or tuple(c["index"] for c in centers) != prior.HELD_CENTERS
                or any(type(c["words"]) is not int or c["words"] <= 0 or len(c["measured"]) != len(VIEWS)
                       or len(c["sha256"]) != len(VIEWS) or not all(prior.SHA256.fullmatch(h) for h in c["sha256"])
                       or any(v is not None and not math.isfinite(v) for v in c["measured"]) for c in centers)):
            raise ValueError("Score or location does not match the source")
        yield row, centers


def audit_inputs(receipt, native):
    split = receipt["split"]
    files = views = 0
    for (old, centers), new in zip(bound(receipt, split), roster_rows(native, split), strict=True):
        generated = readout_inputs(new, list(inputs(new)))
        if old["sha256"] != new["sha256"] or generated != [{k: v for k, v in c.items() if k != "measured"} for c in centers]:
            raise ValueError("Recreated context hashes or lengths differ from measurement")
        files += 1
        views += sum(len(row["sha256"]) for row in generated)
    return {"split": split, "files": files, "contexts": views}


def bucket(words):
    return sum(words > edge for edge in (18, 30))


def fit(rows):
    bins = defaultdict(list)
    for source, centers in rows:
        if source["truth"] is not None:
            raise ValueError("Reference distribution requires ordinary files")
        best = {}
        for center in centers:
            group = bucket(center["words"])
            for i, value in enumerate(center["measured"]):
                if value is not None:
                    best[group, i] = max(value, best.get((group, i), -math.inf))
        for (group, index), value in best.items():
            bins[f"{source['domain']}/{group}/{VIEWS[index]}"].append(value)
    return {key: sorted(values) for key, values in bins.items()}


def ranks(source, centers, reference):
    readings = {}
    for center in centers:
        prefix = f"{source['domain']}/{bucket(center['words'])}"
        percentiles = []
        for view, score in zip(VIEWS, center["measured"], strict=True):
            controls = reference.get(f"{prefix}/{view}", [])
            if score is None or len(controls) < 12:
                break
            percentiles.append(bisect.bisect_left(controls, score) / (len(controls) + 1))
        if len(percentiles) == len(VIEWS):
            first, swapped, isolated = percentiles
            shared = min(first, swapped)
            readings[center["index"]] = {"adjacent": first, "order_agreement": shared,
                                        "order_and_isolation": min(shared, isolated)}
    for index, measures in readings.items():
        others = [m["order_agreement"] for j, m in readings.items() if j != index]
        if others:
            measures["order_and_separation"] = measures["order_agreement"] - max(others)
    return readings


def strongest(readings, candidate):
    ranked = sorted(((scores[candidate], index) for index, scores in readings.items() if candidate in scores), reverse=True)
    if not ranked:
        return None, None
    score, center = ranked[0]
    return score, center if len(ranked) == 1 or ranked[1][0] < score else None


def calibrate(fit_receipt, point_receipt):
    reference = fit(bound(fit_receipt, "fit"))
    points = defaultdict(list)
    for row, centers in bound(point_receipt, "point"):
        readouts = ranks(row, centers, reference)
        for candidate in CANDIDATES:
            value, _ = strongest(readouts, candidate)
            points[f"{row['domain']}/{candidate}"].append(value if value is not None else -2.)
    cuts = {key: {name: sorted(scores, reverse=True)[int(share * len(scores))] for name, share in SHARES.items()}
            for key, scores in points.items()}
    return {"reference": reference, "points": cuts}


def selections(row, centers, calibrated):
    yield from islice(prior.policies(row, {}), 2)
    readings = ranks(row, centers, calibrated["reference"])
    for candidate in CANDIDATES:
        score, index = strongest(readings, candidate)
        for point in SHARES:
            show = index is not None and score > calibrated["points"][f"{row['domain']}/{candidate}"][point]
            yield f"{candidate}/{point}", {index: row["spans"][index]} if show else {}, score is not None


def evaluate(receipt, split, calibrated):
    totals = defaultdict(Counter)
    for row, centers in bound(receipt, split):
        choices = list(selections(row, centers, calibrated))
        truth, spaces = row["truth"], set(row["spaces"])
        previous = choices[0][1]
        baseline = bool(truth and prior.covers(previous.values(), truth, spaces))
        for mode, spans, assessed in choices:
            for genre in (row["domain"], "all"):
                stats = totals[f"{genre}/{mode}/{'rewrite' if truth else 'ordinary'}"]
                stats.update(files=1, assessed=assessed, flags=bool(spans))
                if truth:
                    useful = prior.covers(spans.values(), truth, spaces)
                    stats.update(useful=useful, preserved=useful and baseline, added=useful and not baseline,
                                 lost=baseline and not useful, misplaced=bool(spans) and not useful)
                else:
                    stats.update(new_ordinary_flags=bool(spans) and not previous,
                                 removed_ordinary_flags=bool(previous) and not spans)
    return {key: dict(value) for key, value in sorted(totals.items())}


def report(args):
    fit_receipt, point_receipt = (prior.sources.read(args.path / f"{name}.json") for name in ("fit", "point"))
    expected = calibrate(fit_receipt, point_receipt)
    if args.phase == "freeze":
        pool.write(args.path / "points.json", {"schema": SCHEMA, "design": DESIGN, "parents": committed(), "calibration": expected})
    else:
        frozen = prior.sources.read(args.path / "points.json")
        if frozen != {"schema": SCHEMA, "design": DESIGN, "parents": committed(), "calibration": expected}:
            raise ValueError("The pre-held points do not replay")
        study = {}
        for split in ("held", "transfer"):
            packet = prior.sources.read(args.path / f"{split}.json")
            study[split] = evaluate(packet, split, expected)
        print(json.dumps(study, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "point", "held", "transfer", "freeze", "review", "audit-inputs"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--roster", type=Path)
    args = parser.parse_args()
    if args.phase in ("freeze", "review"):
        report(args)
    else:
        if args.roster is None:
            parser.error("This command requires the native roster")
        if args.phase == "audit-inputs":
            print(json.dumps(audit_inputs(prior.sources.read(args.path), args.roster), sort_keys=True))
        else:
            measure(args)
