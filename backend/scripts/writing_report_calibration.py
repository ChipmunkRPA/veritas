"""Replay a bounded second look for highlighted economic-report passages."""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import document_passage_intervals as serving
from app.evidence import report_passage_calibration as report
from scripts import writing_multi_passage_panel as panel
from scripts import writing_profile_calibration as earlier
from scripts.writing_multi_passage_selection import _match

SCHEMA = "chipmunk.writing-report-confirmation/1"
PRIMARY = "confirmed"
METHODS = ("current", PRIMARY, "report_only")
DESIGN = {
    "profile": "Use the earlier combined shape-and-response profile for the economic genre only.",
    "points": "Use the tighter previously fitted report point, without fitting on the new deliveries.",
    "primary": "Keep the old highlight when an overlapping report range independently meets its fitted point. "
               "If the profile is uncertain, keep all original nominations.",
    "comparison": "Also evaluate selecting only the report ranges, which can introduce new nominations.",
    "truth": "Evaluate overlap in non-whitespace codepoints, with 30% IoU and half of the target characters.",
}
SOURCE = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/writing-profile-calibration-2026-11-16.json.gz"


def asset(receipt):
    fit = receipt["freeze"]["fit"]
    return {
        "profile": fit["features"]["combined"],
        "scales": fit["scales"][report.REPORT],
        "point": fit["points"][report.REPORT]["tighter"],
    }


def location(rows, first, stop, basis):
    return {"compact": [rows[first][2], rows[stop-1][3]], "native": [rows[first][0], rows[stop-1][1]], "basis": basis}


def evaluate(works, calibration):
    measured = []
    for work in works:
        cases = []
        for variant in work["variants"]:
            geometry, readings = earlier.recorded(variant)
            original = serving._choices(geometry, readings, document_baseline=True)
            pooled = {"isolated": serving.ISOLATED, "context": serving.CONTEXT}
            adjusted = report.review(geometry, readings, pooled, calibration)
            confirmed = report.confirm(original, adjusted)
            replacement = original if adjusted is None else [(a, b, value, "report") for a, b, value in adjusted]
            scopes = {key: [location(variant["sentences"], a, b, basis) for a, b, _, basis in choices]
                      for key, choices in zip(METHODS, (original, confirmed, replacement), strict=True)}
            cases.append({
                "version": variant["version"],
                "layout": variant["layout"],
                "generator": variant.get("generator", work.get("generator_backend")),
                "ranges": scopes,
                "truth": [item["compact"] for item in variant["truth"]],
                "routed": adjusted is not None,
                "assessed": readings is not None,
            })
        measured.append({"document": work["document"], "source": work["group_sha256"], "genre": work["genre"], "cases": cases})
    return measured


def summarize(measured):
    totals = defaultdict(Counter)
    for work in measured:
        for case in work["cases"]:
            targets = case["truth"]
            matches = {name: _match(ranges, targets) for name, ranges in case["ranges"].items()}
            old = {target for _, target in matches["current"]}
            family = case["generator"] if targets else "ordinary"
            for name, ranges in case["ranges"].items():
                new = {target for _, target in matches[name]}
                tally = totals[f"{work['genre']}/{case['layout']}/{family}/{name}"]
                tally.update(files=1, assessed=case["assessed"], routed=case["routed"], flagged=bool(ranges),
                             targets=len(targets), located=len(new), lost=len(old - new), gained=len(new - old),
                             unmatched_highlights=len(ranges) - len(new))
    return {key: dict(value) for key, value in sorted(totals.items())}


def digest(path):
    return panel.digest(path.read_bytes())


def freeze(output):
    prior = panel.read(SOURCE)
    fit = asset(prior)
    prior_readings = prior["source"]["works"]
    retained = evaluate(prior_readings, fit)
    groups = [row["group_sha256"] for row in prior_readings] + prior["freeze"]["fit_groups"] + prior["freeze"]["selection_groups"]
    lock = {"schema": SCHEMA, "design": DESIGN, "primary": PRIMARY, "asset": fit, "code": digest(Path(report.__file__)),
            "evaluated_prior_sha256": digest(SOURCE), "excluded_sources": sorted(set(groups)), "retained": summarize(retained)}
    panel.save(output, lock)
    print(json.dumps({"frozen": PRIMARY, "earlier_sources": len(lock["excluded_sources"])}, sort_keys=True))


def assess(freeze_file, inputs, output):
    lock = panel.read(freeze_file)
    if lock["schema"] != SCHEMA or lock["code"] != digest(Path(report.__file__)) or lock["design"] != DESIGN:
        raise ValueError("The report check differs from the version fixed before testing")
    sources = [panel.read(path) for path in inputs]
    works = [work for source in sources for work in source["works"]]
    groups = {work["group_sha256"] for work in works}
    if len(works) != len(groups) or groups.intersection(lock["excluded_sources"]):
        raise ValueError("A test group was already used for this calibration")
    outcomes = evaluate(works, lock["asset"])
    metadata = []
    for source in sources:
        info = {key: value for key, value in source.items() if key not in {"works", "model_assets"}}
        info["model_asset_digests"] = [{"file": name, "sha256": sha}
                                       for name, sha in sorted(source["model_assets"].items())]
        metadata.append(info)
    receipt = {"freeze": lock, "input_sha256": [digest(path) for path in inputs],
               "input_metadata": metadata, "works": works, "outcomes": outcomes, "summary": summarize(outcomes)}
    panel.save(output, receipt)
    print(json.dumps(verify(receipt), sort_keys=True), flush=True)


def verify(receipt):
    from app.bench.independent_locations import _matches

    lock = receipt["freeze"]
    if (lock["schema"], lock["primary"], lock["design"]) != (SCHEMA, PRIMARY, DESIGN):
        raise ValueError("Unknown report review comparison")
    outcomes = evaluate(receipt["works"], lock["asset"])
    if outcomes != receipt["outcomes"] or summarize(outcomes) != receipt["summary"]:
        raise ValueError("Saved passage positions or counts differ from the report review")
    groups = {work["source"] for work in outcomes}
    if len(groups) != len(outcomes) or groups.intersection(lock["excluded_sources"]):
        raise ValueError("Source groups cross evaluation roles")
    independent = Counter()
    for work in outcomes:
        for case in work["cases"]:
            family = case["generator"] if case["truth"] else "ordinary"
            for method, found in case["ranges"].items():
                key = f"{work['genre']}/{case['layout']}/{family}/{method}"
                independent[key] += _matches(case["truth"], [item["compact"] for item in found], .30, .50)
    if {key: metrics["located"] for key, metrics in receipt["summary"].items()} != dict(independent):
        raise ValueError("Independent interval assignment disagrees with the benchmark")
    return {"works": len(groups), "files": sum(len(work["cases"]) for work in outcomes),
            "genres": dict(Counter(work["genre"] for work in outcomes)), "recounted_strata": len(independent)}


def verify_source_plans(receipt, plans):
    measured = {work["document"]: work for work in receipt["works"]}
    outcomes = {work["document"]: work["cases"] for work in receipt["outcomes"]}
    counts = Counter()
    for plan, meta in zip(plans, receipt["input_metadata"], strict=True):
        if panel.digest(json.dumps(plan, sort_keys=True).encode()) != meta["input_plan_sha256"]:
            raise ValueError("Original text plan differs from the recorded inputs")
        for work in plan["works"]:
            for variant, reading, outcome in zip(work["variants"], measured[work["document"]]["variants"],
                                                 outcomes[work["document"]], strict=True):
                text = variant["text"]
                if panel.digest(text.encode()) != reading["native_text_sha256"]:
                    raise ValueError("Original native text differs from the scored file")
                _, context = panel._inputs(variant)
                if [panel.digest(value.encode()) for value in context] != reading["inputs_sha256"]:
                    raise ValueError("Context text differs from the measured inputs")
                characters = [index for index, character in enumerate(text) if not character.isspace()]
                for options in outcome["ranges"].values():
                    for selected in options:
                        a, b = selected["native"]
                        compact = [bisect_left(characters, value) for value in (a, b)]
                        if not 0 <= a < b <= len(text) or compact != selected["compact"]:
                            raise ValueError("A selected range differs from the actual text positions")
                        counts["ranges"] += 1
                counts.update(files=1, context_windows=len(context))
    return dict(counts)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="stage", required=True)
    lock = commands.add_parser("freeze")
    lock.add_argument("output", type=Path)
    test = commands.add_parser("assess")
    test.add_argument("freeze", type=Path)
    test.add_argument("output", type=Path)
    test.add_argument("readings", nargs="+", type=Path)
    review = commands.add_parser("verify")
    review.add_argument("receipt", type=Path)
    review.add_argument("--plans", nargs="+", type=Path)
    args = cli.parse_args()
    if args.stage == "freeze":
        freeze(args.output)
    elif args.stage == "assess":
        assess(args.freeze, args.readings, args.output)
    else:
        result = panel.read(args.receipt)
        verified = verify(result)
        if args.plans:
            verified["native"] = verify_source_plans(result, [panel.read(path) for path in args.plans])
        print(json.dumps(verified, sort_keys=True))
