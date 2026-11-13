"""Recalculate exact packet positions and frozen control-based writing decisions."""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import writing_revision_trajectory_transfer as source
from scripts.writing_revision_worker_priority import GROUPS_PER_BATCH
from scripts.writing_revision_worker_transfer import CHECKPOINT, CHECKS, SCHEMA

METHODS = ("primary_absolute", "primary_relative", "relative_agreement")
PUBLIC_SCHEMA = "chipmunk.writing-revision-worker-public/1"
POINT_SCHEMA = "chipmunk.writing-revision-worker-points/1"
MAX_PACKETS = 6
HASH = re.compile(r"[0-9a-f]{64}")
# This digest binds the first frozen point file, written before the receipt
# reader gained its fit-only projection. Current arithmetic must still match.
FIRST_POINT_READER = "ffa5a91c4edca32862a9d38153f4761aee386bbdda48b4fb96ed547bfad01fe3"


def number(value):
    score = float.fromhex(value)
    source.require(math.isfinite(score) and score.hex() == value, "Invalid recorded scalar")
    return score


def candidates(row):
    primary = [number(value) for _, value in row["raw"]]
    log_odds = [source.odds(value) for value in primary]
    center = median(log_odds)
    ratios = [number(value["raw"]) for value in row["ratio"] if value is not None and not value["neighbor_expansion"]]
    ratio_center = median(ratios) if ratios else None
    classifier_scale = max(0.25, median(abs(value - center) for value in log_odds))
    ratio_scale = max(0.01, median(abs(value - ratio_center) for value in ratios)) if ratios else None
    absolute, relative, agreement = [], [], []
    for index, (p, odds) in enumerate(zip(primary, log_odds, strict=True)):
        # Both primary arms show the full measured input. The agreement arm
        # shows the paired ratio's exact sentence reduction, with the primary
        # context used only to corroborate where to look.
        a, b = max(0, index - 1), min(index + 2, len(primary))
        absolute.append((p, a, b))
        relative.append((odds - center, a, b))
        paired = row["ratio"][index]
        if paired is not None and not paired["neighbor_expansion"]:
            weaker = min((odds - center) / classifier_scale, (ratio_center - number(paired["raw"])) / ratio_scale)
            agreement.append((weaker, index, index + 1))
    return dict(zip(METHODS, (absolute, relative, agreement), strict=True))


def fit_cutoffs(rows):
    source.require(len(rows) == 16, "Sixteen genuine source groups required for point selection")
    maxima = {method: [] for method in METHODS}
    for row in rows:
        for method, choices in candidates(row).items():
            source.require(bool(choices), "Genuine source had no comparable location")
            maxima[method].append(max(score for score, _, _ in choices))
    return {method: sorted(maxima[method])[-2].hex() for method in METHODS}


def points(path):
    ledger = source.read(path)
    source.require(ledger["phase"] == "fit" and not ledger["limited"], "Complete unaltered fit worker run required")
    return {"schema": POINT_SCHEMA, "fit_worker_sha256": source.digest(path), "source_roster_sha256": ledger["source_roster_sha256"],
            "reader_sha256": source.digest(Path(__file__)), "policy": "Strictly above each arm's second-highest genuine document maximum",
            "decision": ("Recommend a separate source-linked inspection cue only if it finds a strict useful range missed by the worker "
                         "on at least three distinct reports across both generator families and flags at most two held originals. "
                         "Retain ordinary and missed ranges; zero confidence contribution."),
            "cutoffs": fit_cutoffs(ledger["rows"])}


def matches_points(saved, path):
    current = points(path)
    return (saved.get("reader_sha256") in {current["reader_sha256"], FIRST_POINT_READER}
            and saved == {**current, "reader_sha256": saved["reader_sha256"]})


def sentence_indices(row, start, end):
    ranges = row["ranges"]
    starts, ends = {value["start"]: i for i, value in enumerate(ranges)}, {value["end"]: i + 1 for i, value in enumerate(ranges)}
    source.require(start in starts and end in ends and starts[start] < ends[end], "Public span is not on exact sentence boundaries")
    return starts[start], ends[end]


def metrics(row, plates):
    ranges = row["ranges"]
    hits = []
    all_indices = set()
    for spans in plates:
        own = {i for a, b in spans for i in range(a, b)}
        source.require(all(0 <= i < len(ranges) for i in own), "Selected range outside exact native positions")
        overlap = sum(ranges[i]["ai_characters"] for i in own)
        union = row["truth_characters"] + sum(ranges[i]["characters"] for i in own) - overlap
        hits.append(overlap > 0 and 2 * overlap >= row["truth_characters"] and 10 * overlap >= 3 * union)
        all_indices.update(own)
    found = [sum(ranges[i]["gold_overlaps"][j] for i in all_indices) * 2 >= size for j, size in enumerate(row["gold_characters"])]
    return {"file_highlighted": bool(plates), "strict_location": any(hits), "has_half_covered_span": any(found),
            "half_covered_spans": sum(found), "ai_chars_highlighted": sum(ranges[i]["ai_characters"] for i in all_indices),
            "ordinary_chars_highlighted": sum(ranges[i]["characters"] - ranges[i]["ai_characters"] for i in all_indices)}


def intervals(row, cutoffs):
    actual = [[sentence_indices(row, *span) for span in packet["spans"]] for packet in row["packets"]]
    historical = [[(a, b)] for i, (_, p) in enumerate(row["prior_raw"])
                  if p >= .5 for a, b in [(max(0, i - 1), min(i + 2, len(row["ranges"])))]][:MAX_PACKETS]
    primary = [[(a, b)] for i, (_, p) in enumerate(row["raw"])
               if number(p) >= .5 for a, b in [(max(0, i - 1), min(i + 2, len(row["ranges"])))]][:MAX_PACKETS]
    present_with_primary = [spans for packet, spans in zip(row["packets"], actual, strict=True) if packet["primary_support"]]
    exact_agreement = [spans for packet, spans in zip(row["packets"], actual, strict=True)
                       if packet["primary_support"] and packet["basis"] == "ratio_reduction"]
    result = {"published_full": actual, "published_primary_supported": present_with_primary,
              "published_exact_agreement": exact_agreement, "prior_primary_proxy": historical, "worker_primary_proxy": primary}
    for method, items in candidates(row).items():
        threshold = number(cutoffs[method])
        eligible = sorted((item for item in items if item[0] > threshold), key=lambda item: (-item[0], item[1]))
        result[method] = [[(a, b)] for _, a, b in eligible[:MAX_PACKETS]]
    return result


def summarize(rows, cutoffs):
    strata = defaultdict(list)
    controls = {row["group"]: row for row in rows if row["role"] == "held" and row["family"] == "original"}
    aliases = []
    gains = {method: [] for method in METHODS}
    losses = {method: [] for method in METHODS}
    for row in rows:
        if row["role"] != "held":
            continue
        by_method = {method: metrics(row, plates) for method, plates in intervals(row, cutoffs).items()}
        if row["family"] == "original":
            names = ["held_original"]
        elif row["text_sha256"] == controls[row["group"]]["text_sha256"]:
            names = ["identical_operation"]
            original = controls[row["group"]]
            source.require(row["document_decision"] == original["document_decision"] and row["packets"] == original["packets"],
                           "Byte-identical native deliveries produced different public ranges or decisions")
            aliases.append(row["record_id"])
        else:
            names = ["text_changed", row["family"], row["family"] + "/" + row["version"]]
            actual = by_method["published_full"]["strict_location"]
            for method in METHODS:
                candidate = by_method[method]["strict_location"]
                if candidate != actual:
                    target = gains if candidate else losses
                    target[method].append({key: row[key] for key in ("group", "record_id", "family", "version")})
        for name in names:
            strata[name].append((row, by_method))
    summary = {}
    for name, cases in sorted(strata.items()):
        summary[name] = {"files": len(cases), "works": len({row["group"] for row, _ in cases}),
                         "document_decisions": dict(Counter(row["document_decision"] for row, _ in cases)),
                         "methods": {method: {key: sum(result[method][key] for _, result in cases) for key in values}
                                     for method, values in cases[0][1].items()}}
    decisions = {}
    for method in METHODS:
        distinct = len({row["group"] for row in gains[method]})
        families = sorted({row["family"] for row in gains[method]})
        false_flags = summary["held_original"]["methods"][method]["file_highlighted"]
        decisions[method] = {"new_strict": gains[method], "new_strict_groups": distinct, "new_strict_families": families,
                             "misses_worker_strict": losses[method], "flagged_ordinary": false_flags,
                             "passes": distinct >= 3 and len(families) == 2 and false_flags <= 2}
    return {"summary": summary, "identical_operation_ids": sorted(aliases), "decision": decisions,
            "detector_statuses": {check: dict(Counter(row["runs"][check]["status"] for row in rows)) for check in CHECKS},
            "published_packets": sum(len(row["packets"]) for row in rows if row["role"] == "held"),
            "published_packets_without_primary_support": sum(not packet["primary_support"] for row in rows if row["role"] == "held"
                                                             for packet in row["packets"])}


def expected_batches(prior):
    groups = {}
    for row in prior["rows"]:
        if row["role"] == "held":
            groups.setdefault(row["group"], []).append(row["record_id"])
    ordered = sorted(groups, key=lambda group: source.sha("writing-worker-held-priority/1/" + group))
    return [set(record for group in ordered[i:i + GROUPS_PER_BATCH] for record in groups[group])
            for i in range(0, len(ordered), GROUPS_PER_BATCH)]


def validate_rows(rows, prior, completed_batches, *, fit_only=False):
    expected = {row["record_id"]: row for row in prior["rows"]}
    batches = expected_batches(prior)
    source.require((completed_batches == 0 and fit_only) or 1 <= completed_batches <= len(batches), "Invalid completed batch count")
    expected_ids = {row["record_id"] for row in prior["rows"] if row["role"] == "fit"}.union(*batches[:completed_batches])
    source.require(len(rows) == len(expected_ids) and {row["record_id"] for row in rows} == expected_ids, "Incomplete worker ledger")
    preserved = ("role", "group", "family", "version", "text_sha256", "ranges", "gold_characters", "truth_characters")
    for row in rows:
        old = expected[row["record_id"]]
        source.require(all(row[key] == old[key] for key in preserved) and row["prior_raw"] == old["primary"], "Prior source geometry changed")
        source.require(set(row["runs"]) == set(CHECKS) and row["runs"][CHECKPOINT]["status"] == "ok"
                       and len(row["raw"]) == len(row["ranges"]) == len(row["ratio"])
                       and [h for h, _ in row["raw"]] == [h for h, _ in old["primary"]], "Primary coverage or input identity changed")
        source.require(all(0 <= number(value) <= 1 for _, value in row["raw"]), "Invalid primary response")
        for span, pair in zip(row["ranges"], row["ratio"], strict=True):
            if pair is not None:
                sentence = [span["start"], span["end"]]
                source.require(HASH.fullmatch(pair["input_sha256"]) and len(pair["spans"]) == 1 and number(pair["raw"]) >= 0
                               and (pair["spans"] == [sentence]) is (not pair["neighbor_expansion"]), "Invalid exact paired scope")
                for extent in pair["spans"]:
                    sentence_indices(row, *extent)
        source.require(len(row["packets"]) <= MAX_PACKETS, "Worker exceeded bounded evidence scope")


def attach_sources(rows, prior):
    old = {row["record_id"]: row for row in prior["rows"]}
    bound = []
    for result in rows:
        prior_row = old[result["record_id"]]
        fields = {key: prior_row[key] for key in ("role", "group", "family", "version", "text_sha256", "ranges", "gold_characters", "truth_characters")}
        bound.append({**fields, **result, "prior_raw": prior_row["primary"]})
    return bound


def fit_summary(rows):
    return {"ordinary_files": len(rows), "ordinary_files_highlighted": sum(bool(row["packets"]) for row in rows),
            "published_packets": sum(len(row["packets"]) for row in rows),
            "packets_without_primary_support": sum(not p["primary_support"] for row in rows for p in row["packets"]),
            "ordinary_chars_highlighted": sum(metrics(row, [[sentence_indices(row, *span) for span in packet["spans"]]
                for packet in row["packets"]])["ordinary_chars_highlighted"] for row in rows),
            "document_decisions": dict(Counter(row["document_decision"] for row in rows)),
            "detector_statuses": {check: dict(Counter(row["runs"][check]["status"] for row in rows)) for check in CHECKS}}


def report_proof(proof, rows):
    [row] = [row for row in rows if row["text_sha256"] == proof["source_sha256"]]
    source.require(proof["packets"] == len(row["packets"]) and proof["pdf_source_spans"] == len({tuple(span)
        for packet in row["packets"] for span in packet["spans"]}) and proof["pdf_ranges_match_api_and_json"] is True
        and proof["internal_model_names_hidden"] is True, "PDF receipt does not bind the genuine case")


def publish_fit(prior_path, fit_path, point_path, report_path, output):
    prior, fit, fitted, report = (source.read(path) for path in (prior_path, fit_path, point_path, report_path))
    source.require(fit["schema"] == SCHEMA and fit["source_roster_sha256"] == prior["freeze"]["roster_sha256"]
                   and matches_points(fitted, fit_path), "Genuine worker fit differs")
    rows = attach_sources(fit["rows"], prior)
    validate_rows(rows, prior, 0, fit_only=True)
    report_proof(report, rows)
    receipt = {"schema": PUBLIC_SCHEMA, "role": "fit_only", "prior_sha256": source.digest(prior_path),
               "source_roster_sha256": fit["source_roster_sha256"], "worker_sha256": {"fit": source.digest(fit_path)},
               "runner_sha256": fit["runner_sha256"], "fitted": fitted, "device": fit["device"],
               "runtime_s": fit["worker_duration_s"], "payment": {"amount_cents": 999, "currency": "usd",
               "unpaid_jobs": 0, "simulated_paid_jobs": len(rows)}, "pdf_sample": report, "rows": rows, "results": fit_summary(rows)}
    source.seal(output, receipt)
    return receipt["results"]


def publish(prior_path, fit_path, held_paths, point_path, batch_path, output):
    prior, fit, fitted, batch_plan = (source.read(path) for path in (prior_path, fit_path, point_path, batch_path))
    source.require([set(items) for items in batch_plan["batches"]] == expected_batches(prior), "Invalid source-group batch plan")
    held = [source.read(path) for path in held_paths]
    source.require(fit["schema"] == SCHEMA and fit["phase"] == "fit" and not fit["limited"] and matches_points(fitted, fit_path),
                   "Invalid full fit output")
    for index, batch in enumerate(held):
        source.require(batch["schema"] == SCHEMA and batch["phase"] == "held" and batch["batch"] == index
                       and not batch["limited"] and fit["device"] == batch["device"]
                       and fit["source_roster_sha256"] == batch["source_roster_sha256"] == prior["freeze"]["roster_sha256"]
                       and batch["fit_worker_sha256"] == source.digest(fit_path)
                       and batch["points_sha256"] == source.digest(point_path)
                       and batch["batches_sha256"] == source.digest(batch_path)
                       and [row["record_id"] for row in batch["rows"]] == batch_plan["batches"][index], "Unbound held worker output")
    rows = attach_sources([*fit["rows"], *(row for batch in held for row in batch["rows"])], prior)
    validate_rows(rows, prior, len(held))
    receipt = {"schema": PUBLIC_SCHEMA, "prior_sha256": source.digest(prior_path), "source_roster_sha256": fit["source_roster_sha256"],
               "worker_sha256": {"fit": source.digest(fit_path), "held": [source.digest(path) for path in held_paths]},
               "runner_sha256": {"fit": fit["runner_sha256"], "held": [batch["runner_sha256"] for batch in held]},
               "fitted": fitted, "device": fit["device"], "planned_batches": len(batch_plan["batches"]),
               "completed_batches": len(held), "batch_plan_sha256": source.digest(batch_path),
               "runtime_s": {"fit": fit["worker_duration_s"], "held": [batch["worker_duration_s"] for batch in held]},
               "payment": {"amount_cents": 999, "currency": "usd", "unpaid_jobs": 0, "simulated_paid_jobs": len(rows)},
               "rows": rows, "results": summarize(rows, fitted["cutoffs"])}
    source.seal(output, receipt)
    return receipt["results"]


def review(path, prior_path):
    data, prior = source.read(path), source.read(prior_path)
    source.require(data["schema"] == PUBLIC_SCHEMA and source.digest(prior_path) == data["prior_sha256"]
                   and data["source_roster_sha256"] == prior["freeze"]["roster_sha256"], "Changed source receipt")
    rows = data["rows"]
    if data.get("role") == "fit_only":
        validate_rows(rows, prior, 0, fit_only=True)
        source.require(fit_cutoffs(rows) == data["fitted"]["cutoffs"], "Genuine control points differ")
        report_proof(data["pdf_sample"], rows)
        result = fit_summary(rows)
        source.require(result == data["results"], "Published fit arithmetic differs")
        return result
    source.require(data["planned_batches"] == len(expected_batches(prior))
                   and data["completed_batches"] == len(data["worker_sha256"]["held"]), "Invalid batch coverage")
    validate_rows(rows, prior, data["completed_batches"])
    cutoffs = fit_cutoffs([row for row in rows if row["role"] == "fit"])
    source.require(cutoffs == data["fitted"]["cutoffs"] and data["fitted"]["fit_worker_sha256"] == data["worker_sha256"]["fit"],
                   "Published control point does not replay")
    result = summarize(rows, cutoffs)
    source.require(result == data["results"], "Published packet arithmetic differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("worker", type=Path)
    fit.add_argument("output", type=Path)
    public = sub.add_parser("publish")
    for name in ("prior", "fit", "points", "batches", "output"):
        public.add_argument(name, type=Path)
    public.add_argument("held", type=Path, nargs="+")
    calibration = sub.add_parser("publish-fit")
    for name in ("prior", "fit", "points", "report", "output"):
        calibration.add_argument(name, type=Path)
    check = sub.add_parser("review")
    check.add_argument("path", type=Path)
    check.add_argument("prior", type=Path)
    args = parser.parse_args()
    if args.phase == "fit":
        source.seal(args.output, points(args.worker))
    elif args.phase == "publish":
        print(source.serialize(publish(args.prior, args.fit, args.held, args.points, args.batches, args.output)), end="")
    elif args.phase == "publish-fit":
        print(source.serialize(publish_fit(args.prior, args.fit, args.points, args.report, args.output)), end="")
    else:
        print(source.serialize(review(args.path, args.prior)), end="")


if __name__ == "__main__":
    main()
