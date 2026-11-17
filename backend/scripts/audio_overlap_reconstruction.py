# SPDX-License-Identifier: Apache-2.0
"""Test whether mixture-derived speech views recover short edits under louder voices."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from scripts import audio_focused_context_transfer as event
from scripts import audio_long_context_review as reference

SCHEMA = "chipmunk.overlapping-speech-reconstruction/1"
METHODS = ("separated", "mixture_phase")
WIDTHS = (4000, 5600, 8000, 9600, 12000, 16000)
POLICIES = ("strict", "wider")
CONDITION = "overlap_-10dB"
BANKS = {"calibration": "audio-six-ten-context-comparison/inputs",
         "held": "audio-six-ten-context-comparison/prospective",
         "matched": "audio-matched-originals/matched-panel"}
DESIGN = {
    "scope": "Use the same center six seconds and native 16 kHz clock as the current focused speech check. "
             "Test voice mixtures with the competing speech 10 dB louder, plus clean ordinary controls.",
    "views": "Use two anonymous decoded estimates and two complementary magnitude masks applied to the original complex spectrum. "
             "Require the two latter waveforms to add back to the input without moving any samples.",
    "score": "Keep the existing speech encoder, activity rule and six-second localization head. At each width use the strongest "
             "window across both views. Fit independent width scales and whole-file points only from the separate calibration files.",
    "fallback": "The existing search has priority. Add the strongest new source interval only when that search has no flag. "
                "Compare both views separately, with the phase-preserving method primary. Speaker identity is outside scope.",
}


def native_row(row, split, *, clean=None):
    scope = [(row["samples"] - reference.SCOPE_SAMPLES) // 2, (row["samples"] + reference.SCOPE_SAMPLES) // 2]
    if row.get("scope", scope) != scope:
        raise ValueError("The previous speech study used a different scope")
    return {key: row[key] for key in ("id", "kind", "condition", "group", "room", "family", "samples", "pcm_sha256", "truth")} | {
        "split": split, "bank": split, "scope": scope, "clean_reference": clean,
        "source": row["id"].rsplit("/", 2)[0],
    }


def prepare(data, output):
    directories = {name: data / folder for name, folder in BANKS.items()}
    paths = {name: directory / "freeze.json" for name, directory in directories.items()}
    ordinary = event.read(paths["calibration"])
    measured = data / "audio-six-ten-context-comparison/prospective-result.json.gz"
    previous = event.read(measured)
    if previous["source_plan"] != event.read(paths["held"]):
        raise ValueError("The serving results do not match the held sources")
    if any(item["context"] == "short" and item["sha256"] != event.digest(
            (data / "audio-six-ten-context-comparison/fit/short.safetensors").read_bytes())
           for item in previous["fit"]["heads"]):
        raise ValueError("The current six-second speech head has changed")
    rows = []
    for row in ordinary["rows"]:
        if row["role"] != "calibration" or row["condition"] != -10 or row["truth"]:
            continue
        item = next(item for item in row["inputs"] if item["context"] == "short")
        rows.append({"split": "calibration", "bank": "calibration", "id": f"{row['source']}/{row['kind']}",
                     "source": row["source"], "group": row["reader"], "kind": row["kind"], "family": None,
                     "room": row["room"], "condition": CONDITION, "samples": item["samples"],
                     "pcm_sha256": item["sha256"], "scope": [0, item["samples"]], "truth": [], "clean_reference": None})
    for row in previous["rows"]:
        if row["condition"] != CONDITION and (row["condition"], row["kind"]) != ("clean", "ordinary"):
            continue
        base, coverage = reference.predictions(row, previous["calibration"]["points"], "wider")
        rows.append(native_row(row, "held") | {"current": [list(span) for span in base["current_plus_six"]],
                                              "current_assessed": coverage["current_plus_six"],
                                              "raw_readings": row["readings"]["short"]})
    matched = event.read(paths["matched"])["cases"]
    by_id = {row["id"]: row for row in matched}
    for row in matched:
        if row["condition"] != CONDITION:
            continue
        clean = by_id.get(row["id"].rsplit("/", 1)[0] + "/clean")
        rows.append(native_row(row, "matched", clean={key: clean[key] for key in ("pcm_sha256", "samples")} if clean else None))
    selected = [row for row in rows if row["split"] == "calibration"]
    held = [row for row in rows if row["split"] != "calibration"]
    for key in ("group", "room", "pcm_sha256"):
        if {row[key] for row in selected} & {row[key] for row in held}:
            raise ValueError(f"Calibration and held recordings share {key}")
    for row in rows:
        row["view_id"] = event.digest(json.dumps([row["pcm_sha256"], row["scope"]], separators=(",", ":")).encode())
    if len({(row["split"], row["id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate prepared trial identifiers")
    result = {"schema": SCHEMA, "banks": BANKS, "design": DESIGN,
              "source_pins": {name: event.digest(file.read_bytes()) for name, file in paths.items()},
              "earlier_comparison_sha256": event.digest(measured.read_bytes()),
              "raw_points": previous["calibration"]["points"]["short"], "rows": rows}
    event.save(output, result)
    print(json.dumps(dict(Counter(row["split"] for row in rows)), sort_keys=True), flush=True)


def predict(row, points, policy):
    raw = reference.recalled(row["raw_readings"], points["raw"], "wider", row["scope"][0])
    current = [tuple(interval) for interval in row.get("current", raw)]
    options = {"raw": raw, "current": current}
    for method in METHODS:
        observed = reference.recalled(row["readings"][method], points[method], policy, row["scope"][0])
        options[method] = observed
        options["plus_" + method] = current or observed
    return options


def assessed(row):
    coverage = {method: any(row["readings"][method]) for method in METHODS}
    coverage["raw"] = any(row["raw_readings"])
    coverage["current"] = row.get("current_assessed", coverage["raw"])
    coverage.update({"plus_" + method: coverage["current"] or coverage[method] for method in METHODS})
    return coverage


def summarize(rows, points):
    outcomes, sources = defaultdict(Counter), defaultdict(set)
    for row in rows:
        truth = reference.common_truth(row)
        for policy in POLICIES:
            options = predict(row, points, policy)
            old = event.matches(options["current"], truth)
            coverage = assessed(row)
            for name, ranges in options.items():
                key = "/".join((row["split"], row["condition"], policy, name))
                tally = outcomes[key]
                hits = event.matches(ranges, truth)
                added = hits - old
                if added:
                    sources[key].add(row["group"])
                tally.update({row["kind"]: 1, row["kind"] + "_flagged": bool(ranges),
                              row["kind"] + "_unassessed": not coverage[name], "events": len(truth), "located": len(hits),
                              "added": len(added), "lost": len(old - hits), "misplaced": row["kind"] == "edited" and bool(ranges) and not hits})
    return {key: dict(counts, added_groups=len(sources[key])) for key, counts in sorted(outcomes.items())}


def reference_diagnostics(rows, points):
    paired = [row for row in rows if "target_si_sdr_db" in row]
    originals = {row["group"]: row for row in paired if row["kind"] == "ordinary"}
    edit_pairs = [row for row in paired if row["kind"] == "edited"]
    if len(originals) != len(edit_pairs) or set(originals) != {row["group"] for row in edit_pairs}:
        raise ValueError("The diagnostic lacks an exact matched ordinary delivery for a reader")
    clean_detected, edit_count, originals_flagged = 0, 0, 0
    for row in paired:
        truth = reference.common_truth(row)
        clean = reference.recalled(row["clean_readings"], points["raw"], "wider", row["scope"][0])
        clean_detected += len(event.matches(clean, truth))
        edit_count += len(truth)
        originals_flagged += row["kind"] == "ordinary" and bool(clean)
    repeated = Counter()
    for changed in edit_pairs:
        original = originals[changed["group"]]
        if (original["scope"], original["room"]) != (changed["scope"], changed["room"]):
            raise ValueError("The paired ordinary and edited clocks or room differ")
        expected = reference.common_truth(changed)
        for policy in POLICIES:
            before = predict(original, points, policy)
            after = predict(changed, points, policy)
            for method in ("raw", *METHODS):
                repeated[f"{policy}/{method}"] += len(event.matches(before[method], expected) & event.matches(after[method], expected))
    proposals = defaultdict(Counter)
    for row in rows:
        if row["split"] != "held" or row["kind"] != "edited":
            continue
        truth = reference.common_truth(row)
        readings = row["readings"] | {"raw": row["raw_readings"]}
        for method, items in readings.items():
            offset = row["scope"][0]
            intervals = [tuple(offset + edge for edge in item["range"]) for item in items if item]
            top = reference.recalled(items, points[method] | {"precutoff": float("-inf")}, "precutoff", offset)
            proposals[method].update(top=len(event.matches(top, truth)), any_width=len(event.matches(intervals, truth)))
    return {"paired_deliveries": len(paired), "clean_reference_events": edit_count, "clean_reference_located": clean_detected,
            "clean_original_flags": originals_flagged, "edit_locations_repeated_on_the_matched_original": dict(repeated),
            "held_proposals_before_the_operating_point": {method: dict(counts) for method, counts in proposals.items()},
            "median_gain_db": {method: round(median(row["target_si_sdr_db"][method] - row["target_si_sdr_db"]["raw"]
                                                    for row in paired), 4) for method in METHODS}}


def verify(result):
    if result["schema"] != SCHEMA or result["design"] != DESIGN:
        raise ValueError("Unexpected two-voice reconstruction study")
    points = result["points"]
    map_to_context = dict(zip(reference.CONTEXTS, METHODS, strict=True))
    normalized = {key: points[arm] for key, arm in map_to_context.items()}
    measured = {key: [row | {"readings": row["readings"][arm]} for row in result["calibration"]]
                for key, arm in map_to_context.items()}
    reference.verify_points({"points": normalized, "calibration": measured})
    roles = defaultdict(set)
    for row in (*result["calibration"], *result["rows"]):
        roles[row["split"]].add(row["group"])
        for readings in (*row["readings"].values(), row["raw_readings"]):
            assert len(readings) == len(WIDTHS)
            for width, rim in zip(WIDTHS, readings, strict=True):
                if rim is not None:
                    a, b = rim["range"]
                    assert 0 <= a < b <= reference.SCOPE_SAMPLES and b - a == width and a % 800 == 0
                    assert math.isfinite(rim["score"])
    assert roles["calibration"].isdisjoint(roles["held"] | roles["matched"])
    summary = summarize(result["rows"], points)
    if summary != result["summary"]:
        raise ValueError("Measured outcome counts failed replay")
    if result["reference_diagnostics"] != reference_diagnostics(result["rows"], points):
        raise ValueError("The clean-reference diagnostics do not reproduce")
    return summary


def export(result, output, policy):
    from app.bench import independent
    from app.bench.independent_locations import TRUTH_SCHEMA, count_events

    summary = verify(result)
    prior = event.read(Path(__file__).resolve().parents[2] / "docs/evaluation-panels/audio-six-ten-context-fresh-complete-2026-11-16.json.gz")
    encoder = result["freeze"]["encoder"]
    if encoder != prior["fit"]["encoder"]:
        raise ValueError("The published serving comparison uses a different speech encoder")
    raw_weights = {encoder["assets_sha256"]["pytorch_model.bin"], result["freeze"]["speech_sha256"]}
    raw_weights.update(item["sha256"] for item in prior["fit"]["heads"] if item["context"] == "short")
    previous_weights = {hash_value for name, hash_value in prior["baseline_assets"].items() if name.endswith((".pt", ".jit"))}
    view_weights = {hash_value for name, hash_value in result["view_assets"]["sha256"].items() if name.endswith(".ckpt")}
    inputs, truth, reports = [], [], defaultdict(list)
    for row in result["rows"]:
        regions = reference.common_truth(row)
        if not regions and row["kind"] == "edited":
            continue
        case = {"id": f"{row['split']}/{row['id']}", "input_sha256": row["pcm_sha256"]}
        group = f"reader/{row['group']}" if row["kind"] != "background" else f"ami/{row['room']}"
        condition = f"{row['split']}/{row['condition']}"
        inputs.append(case | {"modality": "audio", "split": "confirmation", "source_sha256": row["pcm_sha256"],
                              "condition": condition, "group": group, "label": int(bool(regions)),
                              "family": row["family"] or row["kind"], "lineage": [group] + (["ami/" + row["room"]] if row["room"] else []),
                              "license": {"id": "CC BY 4.0 (LlamaPartialSpoof, LibriSpeech, AMI)",
                                          "evaluation_permitted": True, "redistribution_permitted": True},
                              "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                                     "target_units": sum(b - a for a, b in regions)}})
        truth.append(case | {"extent": [row["samples"]], "regions": regions})
        coverage = assessed(row)
        for method, ranges in predict(row, result["points"], policy).items():
            reason = "No eligible speech window was assessed."
            payload = ({"status": "ok", "response": int(bool(ranges)),
                        "localization": {"status": "ok", "regions": [list(span) for span in ranges]}} if coverage[method]
                       else {"status": "not_applicable", "response": None, "reason": reason,
                             "localization": {"status": "not_applicable", "reason": reason}})
            reports[method].append(case | payload)
    manifest = {"schema": independent.SCHEMA, "sample_rate_hz": event.RATE, "cases": inputs}
    signature = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    answer = signature | {"schema": TRUTH_SCHEMA, "cases": truth}
    conditions = {case["condition"] for case in inputs}
    output.mkdir(mode=0o700)
    event.save(output / "manifest.json", manifest)
    event.save(output / "truth.json", answer)
    for method, readings in reports.items():
        weights = raw_weights.copy()
        if method in METHODS or method.startswith("plus_"):
            weights.update(view_weights)
        if method == "current" or method.startswith("plus_"):
            weights.update(previous_weights)
        response = signature | {"system": method, "implementation_sha256": independent.digest([SCHEMA, DESIGN, method, policy]),
                               "method_kind": "weighted", "score_kind": "raw", "response_units": "A source range was flagged (0 or 1)",
                               "weight_sha256": sorted(weights), "predictions": readings}
        checked = count_events(manifest, answer, response, minimum_iou=.30, minimum_recall=.50)
        for condition in conditions:
            assert sum(row["matched_events"] for row in checked["strata"] if row["label"] == 1 and row["condition"] == condition) == (
                summary[f"{condition}/{policy}/{method}"]["located"])
        event.save(output / f"{method}.json", response)
        event.save(output / f"{method}-events.json", checked)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    plan = modes.add_parser("prepare")
    plan.add_argument("data", type=Path)
    plan.add_argument("output", type=Path)
    replay = modes.add_parser("verify")
    replay.add_argument("receipt", type=Path)
    external = modes.add_parser("export")
    external.add_argument("receipt", type=Path)
    external.add_argument("directory", type=Path)
    external.add_argument("--point", choices=POLICIES, default="wider")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.data, args.output)
    elif args.mode == "verify":
        print(json.dumps(verify(event.read(args.receipt)), sort_keys=True))
    else:
        export(event.read(args.receipt), args.directory, args.point)
