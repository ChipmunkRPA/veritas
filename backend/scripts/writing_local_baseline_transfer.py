# SPDX-License-Identifier: MIT
"""Compare pooled writing scores, document baselines and matched flanks."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from app.core.canonical import FLOAT_DP
from app.evidence import document_passage_intervals as serving
from app.evidence.passage_interval_selection import (
    add_disjoint_ranges,
    center_on_document,
    combine_responses,
    flank_adjusted_candidates,
    interval_candidates,
    lower_quarter,
    normalize,
    select_intervals,
)
from scripts import writing_multi_passage_panel as source
from scripts import writing_multi_passage_selection as previous

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = ROOT / "docs/evaluation-panels/writing-multi-passage-transfer-2026-11-16.json.gz"
SCHEMA = "chipmunk.writing-local-baseline-transfer/1"
ARMS = ("pooled", "document_baseline", "flanks", "both")
VIEWS = ("isolated", "context")
PROFILES = {"tighter": .90, "wider": .75}
PUBLISHER_REVISION = "8b31cdf2971ec4e959776e16744d0e5b3159ced8"
FRESH_PINS = {
    "overnight_batch_03_mixed": "dd6360409655745e72d692017ff727a4adc05011c3467ddc0d2b75f99b37a55b",
    "gpt_general_mixed": "f981f0f45dea86f820475f5c14b3a61f1fd4b9e1126bf4da150d6f34f7e720c1",
}
FRESH_SOURCES = {"overnight_batch_03_mixed": "gutenberg_selected", "gpt_general_mixed": "beigebook"}
FRESH_COUNT = 20
METHOD = {
    "inputs": "Use the same isolated and three-sentence classifier inputs and shared word-count scales.",
    "document_baseline": "Estimate a shift from the lower quarter of responses in each complete document view relative to "
                         "the median of ordinary calibration document lower quarters. Bound the shift to one normalized unit, "
                         "subtract it and repeat the same combined-response interval selection.",
    "flanks": "Give an interval extra support when its mean fitted response exceeds the medians in both surrounding "
              "five-sentence excerpts. Require two complete flank sentences on each side, cap the contrast at two "
              "normalized units, and add one third of that contrast scaled by the square root of interval length. "
              "Keep the ordinary score at document edges.",
    "operating_points": "Choose each method's wider and tighter operating points independently from the same "
                        "75th and 90th percentiles of ordinary document maxima over both layouts.",
    "addition": "Keep all pooled wider choices. Add highest scoring combined local choices at their tighter point, "
                "with at least one unselected sentence between the new range and all retained ranges. Preserve the six-range cap.",
    "measurement": "Same exact non-whitespace locations for all arms. Count one-to-one edit matches (recall ≥ 50%, "
                   "intersection over union ≥ 0.30), genuine files flagged, missed edits and unmatched proposals.",
}


def _views(variant):
    scales = {"isolated": serving.ISOLATED, "context": serving.CONTEXT}
    return {view: [normalize(item[1], *scales[view][item[0]]) if item is not None else None
                   for item in previous._inputs(variant, view)] for view in VIEWS}


def fit(calibration):
    ordinary = [variant for work in calibration["works"] if work["split"] == "validation" for variant in work["variants"]]
    views = [_views(variant) for variant in ordinary]
    anchors = {view: median(lower_quarter(row[view]) for row in views) for view in VIEWS}
    peaks = defaultdict(list)
    for work in calibration["works"]:
        if work["split"] == "validation":
            choices = [proposals(variant, anchors) for variant in work["variants"]]
            for arm in ARMS:
                peaks[arm].append(max(score for plan in choices for _, _, score in plan[arm]))
    points = {arm: {label: previous._quantile(scores, fraction) for label, fraction in PROFILES.items()}
              for arm, scores in peaks.items()}
    assert abs(points["pooled"]["wider"] - serving.POINT) < 1e-8
    return {"anchors": anchors, "points": points, "calibration_documents": len(calibration["works"]) - sum(
        work["split"] == "test" for work in calibration["works"])}


def proposals(variant, anchors):
    views = _views(variant)
    centered = {view: center_on_document(readings, anchors[view]) for view, readings in views.items()}
    values = {key: combine_responses(row["isolated"], row["context"]) for key, row in (("normal", views), ("centered", centered))}
    original = interval_candidates(values["normal"])
    adjusted = interval_candidates(values["centered"])
    return {"pooled": original, "document_baseline": adjusted,
            "flanks": flank_adjusted_candidates(values["normal"]),
            "both": flank_adjusted_candidates(values["centered"])}


def _findings(ordinary, altered, regular, rewritten):
    targets = [item["compact"] for item in altered["truth"]]
    return {
        "ordinary": regular, "mixed": rewritten, "targets": targets,
        "hits": [j for _, j in previous._match(rewritten, targets)],
        "ordinary_at_edit": [j for j, row in enumerate(ordinary["matched_targets"])
                             if any(previous._overlap(item["compact"], row) for item in regular)],
        "complete": all(item is not None for variant in (ordinary, altered) for view in VIEWS for item in variant[view]),
    }


def _addition(variant, planned, points):
    count = len(variant["sentences"])
    baseline = select_intervals(planned["pooled"], points["pooled"]["wider"], count)
    local = select_intervals(planned["both"], points["both"]["tighter"], count)
    options = add_disjoint_ranges(baseline, local)
    selections = []
    for first, stop, score, basis in options:
        start, end = variant["sentences"][first], variant["sentences"][stop - 1]
        selections.append({"sentences": [first, stop], "native": [start[0], end[1]], "compact": [start[2], end[3]],
                           "response": round(score, 9), "basis": basis})
    return selections


def decisions(held, parameters):
    output = {}
    for work in held:
        candidates = defaultdict(dict)
        for variant in work["variants"]:
            candidates[variant["layout"]][variant["version"]] = variant, proposals(variant, parameters["anchors"])
        work_decisions = {}
        for arm in ARMS:
            work_decisions[arm] = {}
            for label, point in parameters["points"][arm].items():
                work_decisions[arm][label] = {}
                for layout, pair in candidates.items():
                    ordinary, regular = pair["ordinary"]
                    altered, mixed = pair["mixed"]
                    ordinary_hits = previous._choice(arm, ordinary, regular[arm], point)
                    proposed = previous._choice(arm, altered, mixed[arm], point)
                    work_decisions[arm][label][layout] = _findings(ordinary, altered, ordinary_hits, proposed)
        work_decisions["addition"] = {"wider": {}}
        for layout, pair in candidates.items():
            ordinary, regular = pair["ordinary"]
            altered, mixed = pair["mixed"]
            work_decisions["addition"]["wider"][layout] = _findings(
                ordinary, altered, _addition(ordinary, regular, parameters["points"]), _addition(altered, mixed, parameters["points"]))
        output[work["document"]] = work_decisions
    return output


def evaluate(calibration, held):
    selection = fit(calibration)
    ready = decisions(held, selection)
    totals = {arm: {point: previous._summarize(held, ready, arm, point) for point in PROFILES} for arm in ARMS}
    totals["addition"] = {"wider": previous._summarize(held, ready, "addition", "wider")}
    return {"schema": SCHEMA, "method": METHOD, "selection": selection, "decisions": ready, "summary": totals}


def verify_serving(packet):
    anchors = packet["selection"]["anchors"]
    assert (anchors["isolated"], anchors["context"]) == (serving.DOCUMENT_BASELINE_ISOLATED, serving.DOCUMENT_BASELINE_CONTEXT)
    assert packet["selection"]["points"]["both"]["tighter"] == serving.DOCUMENT_BASELINE_POINT
    for work in packet["held"]:
        for variant in work["variants"]:
            geometry = [[row[0], row[1], row[-1]] for row in variant["sentences"]]
            readings = {view: [round(value, FLOAT_DP) for value in variant[view]] for view in VIEWS}
            label = "ordinary" if variant["version"] == "ordinary" else "mixed"
            for arm, with_baseline in (("pooled", False), ("addition", True)):
                expected = packet["decisions"][work["document"]][arm]["wider"][variant["layout"]][label]
                reported = [([geometry[first][0], geometry[stop - 1][1]], basis) for first, stop, _, basis in
                            serving._choices(geometry, readings, document_baseline=with_baseline)]
                assert reported == [(row["native"], row.get("basis", "pooled")) for row in expected], (
                    work["document"], variant["layout"], label, arm)


def verify_comparison(calibration, packet):
    expected = evaluate(calibration, packet["held"])
    fitted = expected.pop("selection")
    recorded = packet["selection"]
    assert expected == {key: packet[key] for key in expected}
    assert fitted.keys() == recorded.keys() and fitted["calibration_documents"] == recorded["calibration_documents"]
    assert fitted["points"].keys() == recorded["points"].keys()
    pairs = [(fitted["anchors"], recorded["anchors"])]
    pairs.extend((fitted["points"][arm], recorded["points"][arm]) for arm in ARMS)
    # Python releases accumulate floating-point sums differently. Source geometry and tallies still match exactly.
    for recalculated, saved in pairs:
        assert recalculated.keys() == saved.keys()
        assert all(math.isclose(value, saved[key], rel_tol=0, abs_tol=1e-12) for key, value in recalculated.items())


def _neutral(raw):
    return "".join(" " if char.isspace() else char for char in raw)


def variants(row):
    output = []
    for layout, fix in (("published", str), ("neutral", _neutral)):
        article, original = fix(row["text"]), fix(row["source_text"])
        ai = [span for span in row["spans"] if span["label"] == "ai"]
        words = source._prefix(original)
        target = [[s["start"], s["end"]] for s in ai]
        original_ranges = [[words[span["source_start"]], words[span["source_end"]]] for span in ai]
        clean = source.native(original, ())
        mixed = source.native(article, target)
        if clean is None or mixed is None:
            return None
        output += [clean | {"layout": layout, "version": "ordinary", "matched_targets": original_ranges},
                   mixed | {"layout": layout, "version": "mixed"}]
    return output


def prepare(publisher, output, calibration):
    from collections import Counter

    import pyarrow.parquet as pq

    seen_groups = {work["group_sha256"] for work in calibration["works"]}
    works = []
    for subset, pin in FRESH_PINS.items():
        expected_source = FRESH_SOURCES[subset]
        file = publisher / f"{subset}.parquet"
        assert source.digest(file.read_bytes()) == pin
        rows = pq.read_table(file, filters=[("source_dataset", "==", expected_source)]).to_pylist()
        assert all(row["split"] == "train" for row in rows)
        chosen, counts, by_writer = [], Counter(), Counter()
        for row in sorted(rows, key=lambda item: source.digest(f"{SCHEMA}/{item['id']}".encode())):
            group = source.digest(source._source_id(row).encode())
            writer = row["generator_backend"]
            if group in seen_groups or counts[group] >= 2 or by_writer[writer] >= (10 if expected_source == "beigebook" else 6):
                continue
            spans = row["spans"]
            if not 1 <= sum(item["label"] == "ai" for item in spans) <= 2:
                continue
            source._validate(row)
            if (native := variants(row)) is None:
                continue
            chosen.append({"document": row["id"], "split": "test", "publisher_split": "train", "group_sha256": group,
                           "genre": source.GENRES[expected_source], "source_collection": expected_source,
                           "source_reference": row["source_reference"], "source_author": row["source_author"],
                           "source_title": row["source_title"], "source_license": row["source_license"],
                           "generator_backend": writer, "requested_model": row["requested_model"],
                           "model_identity_status": row["model_identity_status"], "variants": native})
            counts[group] += 1
            by_writer[writer] += 1
            if len(chosen) == FRESH_COUNT:
                break
        assert len(chosen) == FRESH_COUNT and len(counts) >= FRESH_COUNT // 2
        seen_groups.update(counts)
        works.extend(chosen)
        print(f"Froze {len(chosen)} matched originals and rewrites from {len(counts)} groups for {expected_source}.", flush=True)
    assert len({work["document"] for work in works}) == len(works)
    source.save(output, {"schema": SCHEMA, "dataset": source.DATASET, "revision": PUBLISHER_REVISION,
                         "publisher_files_sha256": FRESH_PINS, "method": METHOD, "works": works,
                         "calibration_sha256": source.digest(CALIBRATION.read_bytes())})


def overview(packet):
    for arm, profiles in packet["summary"].items():
        for point, genres in profiles.items():
            for layout in ("published", "neutral"):
                tallies = [row[layout] for row in genres.values()]
                values = {key: sum(row[key] for row in tallies) for key in ("labeled_passages", "located", "both_passages",
                                                                          "ordinary_flags", "unmatched_flags", "unconfounded")}
                print(arm, point, layout, values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    read = commands.add_parser("compare")
    read.add_argument("output", type=Path)
    read.add_argument("--held", type=Path)
    review = commands.add_parser("review")
    review.add_argument("receipt", type=Path)
    review.add_argument("--serving", action="store_true", help="Also check the rounded runtime evidence choices")
    build = commands.add_parser("prepare")
    build.add_argument("publisher", type=Path)
    build.add_argument("output", type=Path)
    args = parser.parse_args()
    calibration = source.read(CALIBRATION)
    if args.action == "prepare":
        prepare(args.publisher, args.output, calibration)
        raise SystemExit(0)
    if args.action == "compare":
        test = source.read(args.held) if args.held else calibration
        held = [work for work in test["works"] if work["split"] == "test"]
        packet = evaluate(calibration, held)
        inputs = {key: test[key] for key in ("dataset", "revision", "input_count", "input_plan_sha256", "publisher_files_sha256", "model_assets")}
        source.save(args.output, packet | {"held": held, "inputs": inputs, "calibration_sha256": source.digest(CALIBRATION.read_bytes())})
    else:
        packet = source.read(args.receipt)
        assert packet["calibration_sha256"] == source.digest(CALIBRATION.read_bytes())
        verify_comparison(calibration, packet)
        if args.serving:
            verify_serving(packet)
    print(json.dumps(packet["selection"], indent=2))
    overview(packet)
