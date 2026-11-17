# SPDX-License-Identifier: Apache-2.0
"""Freeze ordinary-control operating points and compare measured speech intervals."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from app.bench.independent_locations import intervals as merge
from scripts import audio_frame_logit_localization as experiment
from scripts.audio_focused_context_transfer import matches, overlap
from scripts.audio_two_second_transfer import locations

FIELDS = ("level", "contrast")
KINDS = {"edited", "ordinary", "ordinary_processed", "background"}
OPERATING_POINTS = {"strict": (0, 0), "wider": (2, 1)}
LIMIT = 5


def journal(plan, raw, incumbent):
    import json

    frozen = experiment.read(plan)
    lines = [json.loads(body) for body in raw.read_text().splitlines()]
    header, *results = lines
    if (header["schema"] != experiment.SCHEMA or header["plan"] != experiment.sha(plan.read_bytes())
            or len(results) != len(frozen["cases"])):
        raise ValueError("Replay is incomplete or does not match the frozen roster")
    serving = {row["id"]: row for row in experiment.read(incumbent)["rows"]}
    rows = []
    for expected, result in zip(frozen["cases"], results, strict=True):
        readings = result.pop("readings")
        error = result.pop("raw_logit_reconstruction_error")
        if result != expected or result["kind"] not in KINDS or not 0 <= error <= 3e-5:
            raise ValueError("A measured recording or reconstructed raw response differs")
        existing = serving[result["id"]]
        if any(existing[key] != result[key] for key in ("pcm_sha256", "samples", "kind", "truth", "room")):
            raise ValueError("The existing result uses different source samples or truth")
        if "current" in existing:
            prior, assessed = existing["current"], existing["current_assessed"]
        else:
            prior = locations(existing)["adjacent_support"]
            assessed = (any(item["direction"] != "unassessed" for item in existing["current_windows"])
                        or any(item["responses"] for item in existing["excerpts"]))
        for row in readings:
            a, b = row["range"]
            if (not 0 <= a < b <= result["samples"]
                    or b - a not in {experiment.STRIDE * size for size in experiment.WIDTHS}
                    or (a - experiment.FIRST_CENTER + experiment.STRIDE // 2) % (experiment.STRIDE * 4)
                    or not (b - a) / 2 <= row["speech_samples"] <= b - a or row["context_support"] < 2):
                raise ValueError("An interval does not follow the frozen sample clock")
        rows.append(result | {"baseline": [list(pair) for pair in prior], "baseline_assessed": assessed,
                              "readings": readings, "reconstruction_error": error})
    return rows, header


def control_points(rows):
    thresholds = {}
    for field in FIELDS:
        ordinary, noise = defaultdict(list), defaultdict(list)
        for row in rows:
            peak = row["peaks"][field]
            if peak is None:
                continue
            if row["kind"] in {"ordinary", "ordinary_processed"} and not row["baseline"]:
                ordinary[row["group"]].append(peak)
            if row["kind"] == "background" and not row["baseline"]:
                noise[row["room"]].append(peak)
        controls = [sorted((max(entries) for entries in source.values()), reverse=True) for source in (ordinary, noise)]
        if any(len(groups) <= max(allowance[i] for allowance in OPERATING_POINTS.values())
               for i, groups in enumerate(controls)):
            raise ValueError("The development controls are not sufficient to calibrate these points")
        thresholds[field] = {name: max(controls[0][genuine], controls[1][background])
                             for name, (genuine, background) in OPERATING_POINTS.items()}
    return thresholds


def selected(readings, field, minimum):
    eligible = [row for row in readings if row[field] is not None and row[field] > minimum]
    accepted = []
    for row in sorted(eligible, key=lambda area: (-area[field], area["range"])):
        a, b = row["range"]
        if any(2 * overlap((a, b), (c, d)) >= min(b - a, d - c) for c, d in accepted):
            continue
        accepted.append((a, b))
        if len(accepted) == LIMIT:
            break
    return [list(area) for area in sorted(accepted)]


def evaluate(rows, points):
    readings = []
    for row in rows:
        predictions = {f"{field}_{label}": selected(row["readings"], field, point)
                       for field in FIELDS for label, point in points[field].items()}
        findings = {label: selected(row["readings"], label, -float("inf")) for label in FIELDS}
        readings.append({key: value for key, value in row.items() if key != "readings"}
                        | {"candidate": predictions, "pre_point": findings,
                           "assessed": {field: row["peaks"][field] is not None for field in FIELDS}})
    return readings


def summarize(rows):
    counts = Counter()
    results = defaultdict(Counter)
    new_groups = defaultdict(set)
    backgrounds = defaultdict(set)
    for row in rows:
        kind = row["kind"]
        counts[kind] += 1
        truth = row["truth"]
        base = row["baseline"]
        baseline_matches = matches(base, truth)
        counts["labeled_intervals"] += len(truth)
        counts["short_intervals"] += sum(b-a < 9600 for a, b in truth)
        counts["existing_locations"] += len(baseline_matches)
        counts["existing_short_locations"] += sum(truth[i][1] - truth[i][0] < 9600 for i in baseline_matches)
        counts["existing_" + kind + "_flags"] += bool(base)
        for arm, areas in row["candidate"].items():
            combined = matches(sorted(set(map(tuple, base + areas))), truth)
            useful = matches(areas, truth)
            result = results[arm]
            result["locations"] += len(combined)
            result["net_new_locations"] += len(combined) - len(baseline_matches)
            result["short_locations"] += sum(truth[i][1] - truth[i][0] < 9600 for i in combined)
            result["candidate_locations"] += len(useful)
            result["misplaced_edited_deliveries"] += bool(truth and areas and not useful)
            if kind == "edited":
                result["candidate_ranges"] += len(areas)
                result["genuine_samples_flagged_on_edited_files"] += sum(
                    end - start - sum(overlap((start, end), span) for span in merge(truth)) for start, end in merge(areas))
            result[f"new_{kind}_flags"] += bool(areas and not base)
            if areas and not base and kind == "background":
                backgrounds[arm].add(row["room"])
            if areas and not base and kind in {"ordinary", "ordinary_processed"}:
                new_groups[arm].add(row["group"])
            if row["family"] and len(combined) > len(baseline_matches):
                result[f"gain_{row['family']}"] += len(combined) - len(baseline_matches)
        for field, nominations in row["pre_point"].items():
            counts[f"{field}_assessed"] += row["assessed"][field]
            counts[f"{field}_top_locations"] += len(matches(nominations, truth))
    return {"source_counts": dict(counts),
            "arms": {arm: dict(result) | {"new_ordinary_groups": len(new_groups[arm]),
                                         "new_background_rooms": len(backgrounds[arm])}
                     for arm, result in results.items()}}


def export_comparison(report, source, output):
    from app.bench import independent
    from app.bench.independent_locations import TRUTH_SCHEMA, count_events

    data, parent = experiment.read(report), experiment.read(source)
    if data["role"] != "transfer" or data.get("input_sha256") != experiment.sha(source.read_bytes()):
        raise ValueError("Expected the original transfer source commitments")
    originals = {row["id"]: row for row in parent["origins"]}
    rooms = {row["room"]: row["pcm_sha256"] for row in parent["rooms"]["rooms"]}
    inputs, truth, predictions = [], [], []
    for row in data["rows"]:
        ident = row["id"]
        input_row = {"id": ident, "input_sha256": row["pcm_sha256"]}
        reader, meeting = "reader/" + row["group"], "ami/" + row["room"]
        isolated = row["kind"] == "background"
        extent = sum(b - a for a, b in merge(row["truth"]))
        source_id = rooms[row["room"]] if isolated else originals[ident.split("/", 1)[0]]["source_sha256"]
        inputs.append(input_row | {"modality": "audio", "split": "confirmation", "source_sha256": source_id,
                                   "group": meeting if isolated else reader, "lineage": [reader, meeting],
                                   "label": int(bool(extent)), "condition": row["condition"] + ("/isolated" if isolated else ""),
                                   "family": row["family"] or row["kind"],
                                   "license": {"id": "CC BY 4.0 (LlamaPartialSpoof, LibriSpeech, AMI)",
                                               "evaluation_permitted": True, "redistribution_permitted": True},
                                   "localization_truth": {"unit": "samples", "source_units": row["samples"], "target_units": extent}})
        truth.append(input_row | {"extent": [row["samples"]], "regions": row["truth"]})
        if row["baseline_assessed"]:
            predictions.append(input_row | {"status": "ok", "response": int(bool(row["baseline"])),
                                            "localization": {"status": "ok", "regions": row["baseline"]}})
        else:
            reason = "The existing check measured no eligible speech window."
            predictions.append(input_row | {"status": "not_applicable", "response": None, "reason": reason,
                                            "localization": {"status": "not_applicable", "reason": reason}})
    manifest = {"schema": independent.SCHEMA, "sample_rate_hz": experiment.RATE, "cases": inputs}
    identity = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    target = identity | {"schema": TRUTH_SCHEMA, "cases": truth}
    implementation = experiment.sha(Path(locations.__code__.co_filename).read_bytes())
    response = identity | {"system": "existing_speech_windows", "implementation_sha256": implementation,
                           "method_kind": "weighted", "score_kind": "raw", "response_units": "A source range was flagged (0 or 1)",
                           "weight_sha256": sorted(data["weights"].values()), "predictions": predictions}
    events = count_events(manifest, target, response, minimum_iou=.30, minimum_recall=.50)
    if sum(row["matched_events"] for row in events["strata"]) != data["summary"]["source_counts"]["existing_locations"]:
        raise ValueError("Independent event recount differs from the recorded comparison")
    output.mkdir(mode=0o700)
    for name, content in (("manifest", manifest), ("truth", target), ("existing", response), ("existing-events", events)):
        experiment.write(output / f"{name}.json", content)


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "replay", "verify", "export"))
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.phase == "export":
        export_comparison(*args.paths)
        return
    if args.phase == "verify":
        report, *parent = args.paths
        data = experiment.read(report)
        if data["summary"] != summarize(data["rows"]):
            raise ValueError("Reported totals do not match recorded source intervals")
        if data["role"] == "development" and data["points"] != control_points(data["rows"]):
            raise ValueError("Recorded operating points differ from the source-group control maxima")
        if parent:
            (development,) = parent
            first = experiment.read(development)
            if (data["role"] != "transfer" or first["role"] != "development"
                    or experiment.sha(development.read_bytes()) != data["development_sha256"]
                    or data["points"] != first["points"] or first["points"] != control_points(first["rows"])
                    or any({row[field] for row in first["rows"]} & {row[field] for row in data["rows"]}
                           for field in ("group", "room", "pcm_sha256"))):
                raise ValueError("Transfer does not match the disjoint, frozen development points")
        print(json.dumps(data["summary"], indent=2))
        return
    plan, raw, incumbent, output, *fix = args.paths
    rows, model = journal(plan, raw, incumbent)
    for row in rows:
        row["peaks"] = {field: max((area[field] for area in row["readings"] if area[field] is not None), default=None)
                        for field in FIELDS}
    receipt = experiment.read(fix[0]) if fix else None
    if args.phase == "fit":
        points = control_points(rows)
    else:
        if (len(fix) != 1 or receipt["weights"] != model["weights"]
                or receipt["role"] != "development" or experiment.read(plan)["role"] != "transfer"):
            raise ValueError("The transfer needs the already fitted development receipt")
        if any({row[field] for row in rows} & {row[field] for row in receipt["rows"]}
               for field in ("group", "room", "pcm_sha256")):
            raise ValueError("The development and transfer source groups, rooms or bytes overlap")
        points = receipt["points"]
    published = evaluate(rows, points)
    frozen = experiment.read(plan)
    record = {"schema": experiment.SCHEMA, "role": frozen["role"], "weights": model["weights"],
              "input_sha256": frozen["input"],
              "model_plan": model["plan"], "baseline_sha256": experiment.sha(incumbent.read_bytes()),
              "development_sha256": experiment.sha(fix[0].read_bytes()) if fix else None,
              "points": points, "rows": published, "summary": summarize(published)}
    experiment.write(output, record)
    print(json.dumps(record["summary"], indent=2))


if __name__ == "__main__":
    main()
