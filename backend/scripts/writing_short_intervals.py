"""Fit short writing ranges and test them on independent report versions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.bench.independent_locations import _matches
from app.evidence import document_passage_intervals as current
from scripts import writing_multi_passage_panel as panel
from scripts import writing_profile_calibration as profiles
from scripts import writing_short_selection as short

SCHEMA = "chipmunk.writing-short-intervals/1"
ROOT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
DEVELOPMENT = (
    (ROOT / "writing-report-confirmation-2026-11-16.json.gz", ("works",)),
    (ROOT / "writing-profile-calibration-2026-11-16.json.gz", ("source", "works")),
)
METHODS = ("context", "narrow", "overlap")
POINTS = {"strict": 0.0, "wider": .06}
DESIGN = {
    "proposals": "One to three complete sentences, evaluated at their actual character positions.",
    "views": "Compare a fixed center-context score with a fitted narrow response and a fit that also measures "
             "how context responses change as surrounding windows include more of the proposed text.",
    "fit": "Match edits to source-aligned ranges when unchanged contexts identify the location. Compare similar "
           "ordinary and wrong-site ranges of the same width, with weaker weight on extreme wrong-site readings.",
    "points": "Fix the strict and wider operating points on per-source maxima of untouched reports across two layouts.",
    "held": "Retain the current results and count added matches with 30% non-whitespace IoU and 50% target recall. "
            "Count new ordinary file alerts separately from additional highlights on already flagged files.",
}


def digest(path):
    return panel.digest(path.read_bytes())


def positions(sentences, first, stop):
    return [sentences[first][2], sentences[stop - 1][3]]


def iou(left, right):
    common = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    return common / (max(left[1], right[1]) - min(left[0], right[0]))


def eligible(left, right):
    return iou(left, right) >= .30 and 2 * (min(left[1], right[1]) - max(left[0], right[0])) >= right[1] - right[0]


def feature_rows(variant):
    geometry, readings = profiles.recorded(variant)
    if readings is None:
        return []
    return [(first, stop, positions(variant["sentences"], first, stop), baseline, extra)
            for first, stop, baseline, extra in short.features(geometry, readings)]


def one_view(row, method):
    return row[3] + row[4] if method == "overlap" else row[3]


def matching_site(original, altered, first, stop):
    """Register a candidate to its ordinary counterpart using unchanged context hashes."""
    prior = original["inputs_sha256"]
    current_sha = altered["inputs_sha256"]
    count = Counter(prior)
    unique = {value: index for index, value in enumerate(prior) if count[value] == 1}
    before = next((index for index in range(first - 2, -1, -1) if current_sha[index] in unique), None)
    after = next((index for index in range(stop + 1, len(current_sha)) if current_sha[index] in unique), None)
    if before is None or after is None:
        return None
    left = first + unique[current_sha[before]] - before
    right = stop + unique[current_sha[after]] - after
    return (left, right) if 0 <= left < right <= len(prior) else None


def collect_pairs(works):
    pairs = defaultdict(list)
    for work in works:
        originals = {v["layout"]: (v, feature_rows(v)) for v in work["variants"] if v["version"] == "ordinary"}
        for variant in work["variants"]:
            known = [item["compact"] for item in variant["truth"]]
            if not known:
                continue
            seen = feature_rows(variant)
            original, ordinary = originals[variant["layout"]]
            ordinary_by_span = {(row[0], row[1]): row for row in ordinary}
            away = [row for row in seen if all(iou(row[2], box) == 0 for box in known)]
            for target in known:
                good = [row for row in seen if eligible(row[2], target)]
                if not good:
                    continue
                best = max(good, key=lambda row: (iou(row[2], target), -row[0]))
                width = best[1] - best[0]
                group = pairs[work["group_sha256"]]
                if aligned := ordinary_by_span.get(matching_site(original, variant, best[0], best[1])):
                    group.append((best, aligned, 3.))
                for candidates in (ordinary, away):
                    same_width = [row for row in candidates if row[1] - row[0] == width]
                    if not same_width:
                        continue
                    same_size = sorted(same_width, key=lambda row: abs(best[3][6] - row[3][6]))[:6]
                    for negative in same_size[:4]:
                        group.append((best, negative, 1.))
                    hardest = max(same_size, key=lambda row: row[3][3])
                    group.append((best, hardest, .5))
    return {key: rows for key, rows in pairs.items() if rows}


def train(works):
    import numpy as np

    pairs = collect_pairs(works)
    if not pairs:
        raise ValueError("No earlier labeled rewrite fits the short interval geometry")
    totals = {group: math.fsum(row[2] for row in rows) for group, rows in pairs.items()}
    weights = np.array([importance / (len(pairs) * totals[group])
                        for group, rows in pairs.items() for _, _, importance in rows])
    heads = {
        "context": {"center": [0.] * 8, "scale": [1.] * 8, "coefficients": [0., 0., 0., 1., 0., .5, 0., 0.]},
    }
    for method in ("narrow", "overlap"):
        positive, negative = (np.array([one_view(pair[side], method) for rows in pairs.values() for pair in rows])
                              for side in (0, 1))
        midpoint = weights @ ((positive + negative) / 2)
        spread = np.maximum(np.sqrt(weights @ ((positive - midpoint)**2 + (negative - midpoint)**2) / 2), .001)
        delta = (positive - negative) / spread
        covariance = (delta.T * weights) @ delta + .4 * np.eye(midpoint.size)
        coefficients = np.linalg.solve(covariance, weights @ delta)
        heads[method] = {"center": midpoint.tolist(), "scale": spread.tolist(), "coefficients": coefficients.tolist()}
    return heads, {"paired_groups": len(pairs), "comparisons": int(len(weights))}


def candidates(works, heads):
    outcomes = []
    for work in works:
        cases = []
        for variant in work["variants"]:
            geometry, readings = profiles.recorded(variant)
            ranges, _ = current._reviewed_choices(geometry, readings, document_baseline=True, report_review=True)
            measurements = {key: short.candidates(geometry, readings, head, use_overlap=key == "overlap")
                            for key, head in heads.items()}
            # Store one response row per sentence interval. All held cutoffs derive from earlier sources.
            compact = variant["sentences"]
            cases.append({"layout": variant["layout"], "version": variant["version"],
                          "generator": variant.get("generator", work.get("generator_backend")),
                          "truth": [item["compact"] for item in variant["truth"]],
                          "current": [positions(compact, a, b) for a, b, _, _ in ranges],
                          "geometry": compact, "measurements": measurements, "assessed": readings is not None})
        outcomes.append({"document": work["document"], "source": work["group_sha256"], "genre": work["genre"], "cases": cases})
    return outcomes


def threshold(rows):
    values = {}
    for method in METHODS:
        maxima = []
        for work in rows:
            scores = [score for case in work["cases"] if case["version"] == "ordinary"
                      for _, _, score in case["measurements"][method]]
            if scores:
                maxima.append(max(scores))
        if not maxima:
            raise ValueError("No complete original reports to calibrate short ranges")
        ordered = sorted(maxima, reverse=True)
        for point, budget in POINTS.items():
            values[f"{method}/{point}"] = ordered[math.floor(budget * len(maxima))]
    return values


def tally(rows, method, cutoff):
    report = defaultdict(Counter)
    for work in rows:
        for case in work["cases"]:
            current_ranges = case["current"]
            proposals = case["measurements"][method]
            truth = case["truth"]
            target = report[f"{work['genre']}/{case['layout']}/{case['generator'] if truth else 'ordinary'}"]
            selected = short.select(proposals, cutoff)
            added = [positions(case["geometry"], first, stop) for first, stop, _ in selected]
            choices = list({tuple(item) for item in current_ranges + added})
            useful = _matches(truth, choices, .30, .50)
            before = _matches(truth, current_ranges, .30, .50)
            contributed = _matches(truth, added, .30, .50)
            target.update(files=1, assessed=case["assessed"], truth=len(truth), current_alerts=bool(current_ranges),
                          added_alerts=bool(added) and not current_ranges, candidate_alerts=bool(added),
                          candidate_located=contributed, current_located=before, combined_located=useful,
                          added_unmatched=len(added) - contributed,
                          true_rank1=_matches(truth, [positions(case["geometry"], x[0], x[1])
                                                      for x in sorted(proposals, key=lambda item: -item[2])[:1]], .30, .50))
    return {key: dict(values) for key, values in sorted(report.items())}


def comparison(rows, points):
    return {tag: tally(rows, tag.split("/")[0], value) for tag, value in points.items()}


def choose(report):
    def quality(tag):
        published = [row for key, row in report[tag].items() if "/published/" in key]
        gain = sum(row["combined_located"] - row["current_located"] for row in published)
        flags = sum(row["candidate_alerts"] for row in published if not row["truth"])
        wrong = sum(row["added_unmatched"] for row in published if row["truth"])
        return (-gain, flags, wrong, tag)
    return min(report, key=quality)


def train_sources():
    works, inputs = {}, {}
    for path, keys in DEVELOPMENT:
        body = panel.read(path)
        for key in keys:
            body = body[key]
        for item in body:
            works.setdefault(item["group_sha256"], item)
        inputs[path.name] = digest(path)
    return list(works.values()), inputs


def fit(validation, output):
    source = panel.read(validation)
    works, fits = train_sources()
    train_groups = sorted(work["group_sha256"] for work in works)
    if any(row["group_sha256"] in train_groups or row["split"] != "validation" for row in source["works"]):
        raise ValueError("Only separate calibration reports can set short-range points")
    heads, pair_counts = train(works)
    measured = candidates(source["works"], heads)
    points = threshold(measured)
    summary = comparison(measured, points)
    chosen = choose(summary)
    panel.save(output, {"schema": SCHEMA, "design": DESIGN, "implementation_sha256": digest(Path(short.__file__)),
                        "trained": pair_counts, "heads": heads, "points": points, "chosen": chosen,
                        "fitted_groups": train_groups, "earlier_receipts": fits, "calibration_sha256": digest(validation),
                        "calibration": source["works"], "calibration_summary": summary})
    print(json.dumps({"frozen_check": chosen, "calibration_groups": len(measured), **pair_counts}, sort_keys=True), flush=True)


def summarize(lock, held):
    if lock["schema"] != SCHEMA or lock["design"] != DESIGN:
        raise ValueError("Unknown short passage fit")
    fit_groups = set(lock["fitted_groups"])
    calibration_groups = {row["group_sha256"] for row in lock["calibration"]}
    held_groups = {row["group_sha256"] for row in held}
    if (fit_groups & calibration_groups or held_groups & (fit_groups | calibration_groups)
            or len(held_groups) != len(held) or any(row["split"] != "test" for row in held)):
        raise ValueError("A short passage source crosses fixed study roles")
    calibrated = candidates(lock["calibration"], lock["heads"])
    points = threshold(calibrated)
    result = comparison(calibrated, points)
    if (points, result, choose(result)) != (lock["points"], lock["calibration_summary"], lock["chosen"]):
        raise ValueError("Short passage calibration differs from the frozen choice")
    rows = candidates(held, lock["heads"])
    return comparison(rows, points)


def assess(freeze_file, input_file, output):
    lock, source = panel.read(freeze_file), panel.read(input_file)
    if lock["implementation_sha256"] != digest(Path(short.__file__)):
        raise ValueError("The held inputs require the frozen selection method")
    summary = summarize(lock, source["works"])
    meta = {key: value for key, value in source.items() if key not in {"works", "model_assets"}}
    meta["model_asset_digests"] = [{"file": name, "sha256": sha} for name, sha in sorted(source["model_assets"].items())]
    panel.save(output, {"freeze": lock, "held_input_sha256": digest(input_file), "held": source["works"],
                        "held_metadata": meta, "summary": summary})
    print(json.dumps({"held_groups": len(source["works"]), "choice": lock["chosen"]}, sort_keys=True), flush=True)


def verify(receipt):
    if summarize(receipt["freeze"], receipt["held"]) != receipt["summary"]:
        raise ValueError("Saved short passage counts differ from the native interval replay")
    return {"held_groups": len(receipt["held"]), "choice": receipt["freeze"]["chosen"],
            "summaries": sum(map(len, receipt["summary"].values()))}


def verify_native(receipt, plan):
    """Read back held sentence positions and context digests from the exact source text."""
    if panel.digest(json.dumps(plan, sort_keys=True).encode()) != receipt["held_metadata"]["input_plan_sha256"]:
        raise ValueError("The source plan differs from the one used for held inference")
    checks = Counter()
    for reference, measured in zip(plan["works"], receipt["held"], strict=True):
        if reference["group_sha256"] != measured["group_sha256"]:
            raise ValueError("The held source group differs")
        for original, observed in zip(reference["variants"], measured["variants"], strict=True):
            if any(value != observed[key] for key, value in original.items() if key != "text"):
                raise ValueError("Recorded sentence or label geometry differs from the test plan")
            raw = original["text"]
            if panel.digest(raw.encode()) != observed["native_text_sha256"]:
                raise ValueError("The extracted source text has changed")
            prefix = panel._prefix(raw)
            for a, b, compact_a, compact_b, _, _ in observed["sentences"]:
                if not 0 <= a < b <= len(raw) or (prefix[a], prefix[b]) != (compact_a, compact_b):
                    raise ValueError("The source sentence positions do not match")
            _, context = panel._inputs(original)
            if [panel.digest(text.encode()) for text in context] != observed["inputs_sha256"]:
                raise ValueError("The overlapping context inputs differ from the source")
            checks.update(files=1, windows=len(context), labeled_spans=len(observed["truth"]))
    return dict(checks)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    training, testing, review = (commands.add_parser(name) for name in ("fit", "assess", "verify"))
    for parser, names in ((training, ("validation", "output")), (testing, ("freeze", "held", "output")),
                          (review, ("result",))):
        for name in names:
            parser.add_argument(name, type=Path)
    review.add_argument("--plan", type=Path)
    args = cli.parse_args()
    if args.stage == "fit":
        fit(args.validation, args.output)
    elif args.stage == "assess":
        assess(args.freeze, args.held, args.output)
    else:
        result = panel.read(args.result)
        checks = verify(result)
        if args.plan:
            checks["native"] = verify_native(result, panel.read(args.plan))
        print(json.dumps(checks, sort_keys=True))
