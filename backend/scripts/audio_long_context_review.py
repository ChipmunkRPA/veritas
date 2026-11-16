"""Recount paired-context speech ranges and export a model-independent test panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_focused_context_transfer as event

SCHEMA = "chipmunk.audio-native-context-transfer/1"
CONTEXTS = ("short", "long")
POLICIES = ("wider", "strict")
SCOPE_SAMPLES = 6 * event.RATE
EXPERIMENT = {
    "scope": "Use one center six-second interval in each external input. Both candidates use its exact clock and its speech "
             "activity. The longer candidate should receive up to ten seconds surrounding the same interval. Interpolate "
             "its content and boundary readings onto the shared six-second sample grid before selecting a range.",
    "points": "Repeat the ordinary width and whole-input calibration for the shared search scope. "
              "Both operating points are frozen before scoring the external recordings.",
    "outcomes": "Count only ground-truth events fully inside the shared search scope. Keep the actual serving comparison "
                "with its full-length windows. Report separate, combined and agreeing candidate ranges, ordinary "
                "and isolated-background flags, unassessed cases, and genuine time highlighted.",
}


def common_truth(row):
    a = (row["samples"] - SCOPE_SAMPLES) // 2
    end = a + SCOPE_SAMPLES
    if row["scope"] != [a, end]:
        raise ValueError("Source clock differs from the paired scope")
    return [interval for interval in row["truth"] if a <= interval[0] < interval[1] <= end]


def recalled(readings, point, policy, offset):
    results = []
    for item, scale in zip(readings, point["width_scales"], strict=True):
        if item is None or scale is None:
            continue
        score = (item["score"] - scale[0]) / scale[1]
        if score > point[policy]:
            results.append((score, *(offset + sample for sample in item["range"])))
    if results:
        _, start, end = max(results, key=lambda item: (item[0], -item[1], -item[2]))
        return [(start, end)]
    return []


def predictions(row, points, policy):
    location = row["scope"][0]
    short, long = [recalled(row["readings"][context], points[context], policy, location) for context in CONTEXTS]
    combined = short + [interval for interval in long if not any(event.same_location(interval, old) for old in short)]
    agreed = [interval for interval in long if any(event.same_location(interval, old) for old in short)]
    confirmed = {"six": short, "ten": long, "combined": combined, "agreed": agreed}
    base = [tuple(interval) for interval in row["current"]]
    options = {"current": base} | confirmed | {f"current_plus_{key}": sorted(set(base + ranges))
                                               for key, ranges in confirmed.items()}
    assessed = {"current": row["current_assessed"], "six": any(row["readings"]["short"]), "ten": any(row["readings"]["long"])}
    assessed["combined"] = assessed["six"] or assessed["ten"]
    assessed["agreed"] = assessed["six"] and assessed["ten"]
    assessed.update({f"current_plus_{name}": assessed["current"] or assessed[name] for name in confirmed})
    return options, assessed


def summarize(rows, points):
    tallies, extra_groups = defaultdict(Counter), defaultdict(set)
    for row in rows:
        a, b = row["scope"]
        truth = common_truth(row)
        subsecond = {index for index, (start, stop) in enumerate(truth) if stop - start <= 9600}
        old = event.matches(row["current"], truth)
        for policy in POLICIES:
            options, assessed = predictions(row, points, policy)
            for name, ranges in options.items():
                score = tallies[row["condition"], policy, name]
                score[row["kind"]] += 1
                score[f"{row['kind']}_flagged"] += bool(ranges)
                found = event.matches(ranges, truth)
                score["truth_in_shared_scope"] += len(truth)
                score["all_truth"] += len(row["truth"])
                score["located"] += len(found)
                score["subsecond_truth_in_shared_scope"] += len(subsecond)
                score["subsecond_located"] += len(found & subsecond)
                score["missed"] += len(truth) - len(found)
                score["locatable_additions"] += len(found - old)
                score["old_lost"] += len(old - found)
                if found - old:
                    extra_groups[row["condition"], policy, name].add(row["group"])
                if row["kind"] != "background":
                    visible = [(max(a, start), min(b, stop)) for start, stop in event.merge(ranges) if max(a, start) < min(b, stop)]
                    score["genuine_samples_marked_in_scope"] += sum(
                        stop - start - sum(event.overlap((start, stop), interval) for interval in row["truth"])
                        for start, stop in visible)
                score[f"{row['kind']}_unassessed"] += not assessed[name]
    return {"/".join(key): dict(value, newly_located_source_groups=len(extra_groups[key])) for key, value in sorted(tallies.items())}


def verify_points(calibration):
    def quantile(numbers, percentile):
        return sorted(numbers)[math.ceil(len(numbers) * percentile) - 1]

    for context in CONTEXTS:
        raw = calibration["calibration"][context]
        points = calibration["points"][context]
        sources, rooms = {}, []
        for row in raw:
            parts = [part["score"] if part is not None else None for part in row["readings"]]
            if row["kind"] == "background":
                rooms.append(parts)
            else:
                combined = sources.setdefault(row["source"], [[] for _ in parts])
                for bucket, score in zip(combined, parts, strict=True):
                    if score is not None:
                        bucket.append(score)
        per_width = [[] for _ in points["width_scales"]]
        for widths in sources.values():
            for column, readings in zip(per_width, widths, strict=True):
                if readings:
                    column.append(max(readings))
        scaled = []
        for values in per_width:
            if values:
                center = quantile(values, .50)
                scaled.append([center, round(max(.15, quantile(values, .90) - center), 8)])
            else:
                scaled.append(None)
        if scaled != points["width_scales"]:
            raise ValueError("Width calibration does not reproduce")

        def maximum(values, width_scales):
            return max(((score - scale[0]) / scale[1] for score, scale in zip(values, width_scales, strict=True)
                        if score is not None and scale is not None), default=None)

        sources = [value for widths in sources.values()
                   if (value := maximum([max(scores) if scores else None for scores in widths], scaled)) is not None]
        rooms = [value for scores in rooms if (value := maximum(scores, scaled)) is not None]
        assert (len(sources), len(rooms)) == (points["foreground_prompts"], points["isolated_deliveries"])
        assert points["wider"] == max(max(rooms), quantile(sources, .90))
        assert points["strict"] == max(max(rooms), max(sources))


def verify(document):
    if document["schema"] != SCHEMA or document["experiment"] != EXPERIMENT:
        raise ValueError("Unexpected paired-context receipt")
    verify_points(document["calibration"])
    rows = document["rows"]
    for case, row in zip(document["source_plan"]["cases"], rows, strict=True):
        assert all(row[key] == value for key, value in case.items())
        for readings in row["readings"].values():
            assert all(0 <= item["range"][0] < item["range"][1] <= SCOPE_SAMPLES and math.isfinite(item["score"])
                       for item in readings if item is not None)
    summary = summarize(rows, document["calibration"]["points"])
    if summary != document["summary"]:
        raise ValueError("The reported outcomes do not reproduce")
    return summary


def recount_delta(document):
    """Recount recorded changes and verify their exact source and label references."""
    result, plan = document["delta"], document["source_plan"]
    source = json.dumps(plan, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    cases = {case["id"]: case for case in plan["cases"]}
    if (result["schema"] != SCHEMA or result["scope"] != "incremental changes only"
            or hashlib.sha256(source).hexdigest() != result["input_sha256"]
            or len(cases) != len(plan["cases"]) or len(cases) != result["total_inputs"]):
        raise ValueError("The recorded audio deltas differ from their frozen sources")
    seen, changed_inputs = set(), set()
    stratum, totals = defaultdict(Counter), defaultdict(Counter)
    for case in cases.values():
        tally = stratum[case["condition"]]
        offset = (case["samples"] - SCOPE_SAMPLES) // 2
        tally[case["kind"]] += 1
        tally["labeled_intervals"] += len(common_truth(case | {"scope": [offset, offset + SCOPE_SAMPLES]}))
    for item in result["outcomes"]:
        case = cases[item["case"]]
        key = item["case"], item["policy"], item["candidate"]
        if (key in seen or item["policy"] not in POLICIES or item["candidate"] not in {"six", "ten", "combined", "agreed"}
                or (item["condition"], item["kind"]) != (case["condition"], case["kind"])
                or type(item["new_flag"]) is not bool or type(item["current_assessed"]) is not bool):
            raise ValueError("Invalid recorded audio comparison")
        seen.add(key)
        changed_inputs.add(item["case"])
        offset = (case["samples"] - SCOPE_SAMPLES) // 2
        expected = common_truth(case | {"scope": [offset, offset + SCOPE_SAMPLES]})
        gains, losses = item["new_locations"], item["lost_locations"]
        if any(interval not in expected for interval in gains + losses) or any(interval in losses for interval in gains):
            raise ValueError("A changed audio finding does not refer to a labeled interval in the tested scope")
        tally = totals[item["condition"], item["policy"], item["candidate"]]
        tally["gained"] += len(gains)
        tally["lost"] += len(losses)
        tally["new_flags_" + item["kind"]] += item["new_flag"]
        tally["prior_unassessed"] += not item["current_assessed"]
    if len(changed_inputs) != result["inputs_with_candidate_ranges"]:
        raise ValueError("Candidate trigger counts do not match the recorded changes")
    return {"strata": dict(stratum), "differences": {"/".join(keys): count for keys, count in sorted(totals.items())}}


def export(document, output, policy):
    from app.bench import independent
    from app.bench.independent_locations import TRUTH_SCHEMA, count_events

    summary = verify(document)
    plan = document["source_plan"]
    sources = {item["id"]: item for item in plan["origins"]}
    rooms = {item["room"]: item for item in plan["rooms"]["rooms"]}
    inputs, expected, output_rows = [], [], defaultdict(list)
    for row in document["rows"]:
        truth = common_truth(row)
        if row["kind"] == "edited" and not truth:
            continue  # The exposed comparison covers known edits inside the common time scope.
        original = sources[row["id"].split("/")[0]]
        room = rooms.get(row["room"])
        special = row["kind"] in ("background", "ordinary_processed")
        case = {"id": row["id"], "input_sha256": row["pcm_sha256"]}
        reader = f"reader/{row['group']}"
        inputs.append(case | {"modality": "audio", "split": "confirmation",
                              "source_sha256": room["pcm_sha256"] if row["kind"] == "background" else original["source_sha256"],
                              "condition": row["condition"] + (f"/{row['kind']}" if special else ""),
                              "group": f"ami/{room['room']}" if row["kind"] == "background" else reader,
                              "label": int(bool(truth)), "family": row["family"] or row["kind"],
                              "lineage": [reader, room["pcm_sha256"]] if room else [reader],
                              "license": {"id": "CC BY 4.0 (LlamaPartialSpoof, LibriSpeech, AMI)",
                                          "evaluation_permitted": True, "redistribution_permitted": True},
                              "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                                     "target_units": sum(b-a for a, b in truth)}})
        expected.append(case | {"extent": [row["samples"]], "regions": truth})
        options, assessed = predictions(row, document["calibration"]["points"], policy)
        for method, locations in options.items():
            outcome = ({"status": "ok", "response": int(bool(locations)),
                        "localization": {"status": "ok", "regions": [list(region) for region in locations]}}
                       if assessed[method] else {"status": "not_applicable", "response": None,
                                                 "reason": "No eligible speech window was assessed.",
                                                 "localization": {"status": "not_applicable",
                                                                  "reason": "No eligible speech window was assessed."}})
            output_rows[method].append(case | outcome)
    manifest = {"schema": independent.SCHEMA, "sample_rate_hz": event.RATE, "cases": inputs}
    identity = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    answer = identity | {"schema": TRUTH_SCHEMA, "cases": expected}
    cuts = {row["condition"] for row in document["rows"]}
    output.mkdir(mode=0o700)
    event.save(output / "manifest.json", manifest)
    event.save(output / "truth.json", answer)
    for method, rows in output_rows.items():
        weights = [item["sha256"] for item in document["fit"]["heads"]] if method != "current" else []
        if method == "current" or method.startswith("current_plus_"):
            weights += list(document["baseline_assets"].values())
        response = identity | {"system": method, "implementation_sha256": independent.digest([SCHEMA, EXPERIMENT, method, policy]),
                               "method_kind": "weighted", "score_kind": "raw", "response_units": "A source range was flagged (0 or 1)",
                               "weight_sha256": sorted(set(weights)), "predictions": rows}
        checked = count_events(manifest, answer, response, minimum_iou=.30, minimum_recall=.50)
        for condition in cuts:
            assert sum(row["matched_events"] for row in checked["strata"] if row["label"] == 1 and row["condition"] == condition) == (
                summary[f"{condition}/{policy}/{method}"]["located"])
        event.save(output / f"{method}.json", response)
        event.save(output / f"{method}-events.json", checked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--point", choices=POLICIES, default="wider")
    args = parser.parse_args()
    saved = event.read(args.receipt)
    if args.incremental:
        print(json.dumps(recount_delta(saved), indent=2, sort_keys=True))
    elif args.export:
        export(saved, args.export, args.point)
    else:
        verify(saved)
        print("Reproduced both point calibrations and all matched-scope audio outcomes.")
