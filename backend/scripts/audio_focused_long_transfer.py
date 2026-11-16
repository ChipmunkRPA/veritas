# SPDX-License-Identifier: MIT
"""Test the frozen focused speech detector on longer recordings and competing voices."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import audio_focused_edit_binding as focused
from app.evidence import audio_two_second_binding as two_second
from scripts import audio_two_second_transfer as earlier
from scripts.audio_focused_context_transfer import digest, matches, merge, overlap, read, save

SCHEMA = "chipmunk.focused-long-speech-transfer/1"
METHOD = {
    "schedule": "Measure the frozen 0.5-, 0.75- and 1-second focused detector in each scheduled six-second excerpt. "
                "Use its existing half-speech admission and saved decision point. Hold the current long and two-second rules fixed.",
    "focused": "Add each unique elevated source span to the currently reported ranges.",
    "repeat": "Require another elevated span from a six-second excerpt starting at least one second apart. "
              "The short spans must overlap by half the smaller span or more. Keep both exact measured spans.",
    "outcomes": "Recount one-to-one event matches, ordinary alerts, isolated-background alerts and marked genuine sample counts. "
                "This reuses the completed 240-delivery panel as an exploratory comparison.",
}


def validate_readings(case, rows):
    assert [row["excerpt_start"] for row in rows] == two_second.excerpt_starts(case["samples"])
    for row in rows:
        assert set(row) == {"excerpt_start", "checked_windows", "best"}
        count, best = row["checked_windows"], row["best"]
        assert type(count) is int and 0 <= count <= focused.WINDOWS and (best is None) == (count == 0)
        if best is not None:
            assert set(best) == {"start_sample", "end_sample", "score_units"}
            assert focused.on_grid(best["start_sample"], best["end_sample"]) and type(best["score_units"]) is int


def ranges(rows):
    elevated = []
    for row in rows:
        best = row["best"]
        if best is not None and best["score_units"] > focused.REVIEW_POINT:
            offset = row["excerpt_start"]
            elevated.append((offset, (offset + best["start_sample"], offset + best["end_sample"])))
    corroborated = []
    for offset, span in elevated:
        a, b = span
        if any(abs(other_offset - offset) >= focused.RATE and 2 * overlap(span, other) >= min(b - a, other[1] - other[0])
               for other_offset, other in elevated):
            corroborated.append(span)
    return {"focused": sorted({span for _, span in elevated}), "repeat": sorted(set(corroborated))}


def summarize(parent, measured):
    assert len(parent["rows"]) == len(measured)
    counts = defaultdict(Counter)
    groups = defaultdict(set)
    for case, recorded in zip(parent["rows"], measured, strict=True):
        assert (recorded["id"], recorded["pcm_sha256"]) == (case["id"], case["pcm_sha256"])
        readings = recorded["readings"]
        validate_readings(case, readings)
        extra = ranges(readings)
        truth, kind = case["truth"], case["kind"]
        current = earlier.locations(case)["adjacent_support"]
        before = matches(current, truth)
        for mode, proposed in {"current": current, **{key: sorted(set(current + vals)) for key, vals in extra.items()}}.items():
            key = case["condition"], mode
            found = matches(proposed, truth)
            tally = counts[key]
            tally[kind] += 1
            tally[f"{kind}_flagged"] += bool(proposed)
            if mode != "current":
                tally[f"{kind}_focused_unassessed"] += not any(row["checked_windows"] for row in readings)
            tally["known_edits"] += len(truth)
            tally["located"] += len(found)
            tally["new_matches"] += len(found - before)
            tally["lost_matches"] += len(before - found)
            tally["short_edits_located"] += sum(truth[j][1] - truth[j][0] < int(.6 * focused.RATE) for j in found)
            if found - before:
                groups[key].add(case["group"])
            if kind != "background":
                tally["genuine_samples_marked"] += sum(b - a - sum(overlap((a, b), event) for event in truth) for a, b in merge(proposed))
    return {"/".join(key): dict(tally, groups_with_new_matches=len(groups[key])) for key, tally in sorted(counts.items())}


def score(receipt, media, output):
    import numpy as np
    import torch
    from app.detectors.audio import focused_edit_review
    from app.detectors.audio.short_voice_review import ACTIVITY_MODEL
    from app.detectors.image.adapters.loaders import _entry, _file
    from app.engine.boundary import Deadline
    from scripts.audio_clean_teacher_transfer import runtime

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    parent = read(receipt)
    assert earlier.summarize(parent["freeze"], parent["rows"]) == parent["summary"]
    assert read(media / "freeze.json") == parent["freeze"]
    candidate = _entry(focused_edit_review.MODEL_ID)
    activity_entry = _entry(ACTIVITY_MODEL)
    activity_path = _file(activity_entry, "silero_vad.jit")
    activity_pin, = activity_entry.source.files
    assert digest(activity_path.read_bytes()) == activity_pin.sha256
    inputs = {"schema": SCHEMA, "method": METHOD, "review_point": focused.REVIEW_POINT,
              "baseline_sha256": digest(receipt.read_bytes()), "activity_sha256": activity_pin.sha256,
              "focused_assets": {file.path: file.sha256 for file in candidate.source.files}}
    partial = output.with_name(output.stem + ".partial.json.gz")
    state = read(partial) if partial.is_file() else inputs | {"rows": []}
    assert {key: state[key] for key in inputs} == inputs and len(state["rows"]) <= len(parent["rows"])
    model = focused_edit_review.load_model(focused_edit_review.MODEL_ID, "cpu", "float32")
    activity = runtime(activity_path)
    rows = state["rows"]
    for case, row in zip(parent["rows"][:len(rows)], rows, strict=True):
        assert (row["id"], row["pcm_sha256"]) == (case["id"], case["pcm_sha256"])
        validate_readings(case, row["readings"])
    observed = {row["pcm_sha256"]: row["readings"] for row in rows}
    for index, case in enumerate(parent["rows"][len(rows):], len(rows) + 1):
        pin = case["pcm_sha256"]
        if pin not in observed:
            raw = (media / "pcm" / f"{pin}.pcm").read_bytes()
            assert len(raw) == 2 * case["samples"] and digest(raw) == pin
            pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            readings = []
            for start in two_second.excerpt_starts(len(pcm)):
                excerpt = pcm[start:start + focused.SAMPLES]
                reading = focused_edit_review.scan(excerpt, activity.spans(excerpt), model.obj, Deadline(None))
                readings.append({"excerpt_start": start, **reading})
            validate_readings(case, readings)
            observed[pin] = readings
        rows.append({"id": case["id"], "pcm_sha256": pin, "readings": observed[pin]})
        if index == 1 or index % 4 == 0 or index == len(parent["rows"]):
            scratch = partial.with_name(partial.stem + ".tmp.gz")
            save(scratch, inputs | {"rows": rows})
            scratch.replace(partial)
            print(f"Measured focused speech on {index}/{len(parent['rows'])} deliveries.", flush=True)
    save(output, inputs | {"rows": rows, "summary": summarize(parent, rows)})


def export(parent, result, output):
    from app.bench import independent
    from app.bench.independent_locations import count_events, count_locations

    earlier.export(parent, output)
    manifest, truth, previous = (read(output / name) for name in ("manifest.json", "truth.json", "adjacent_support.json"))
    for arm in ("focused", "repeat"):
        predictions = []
        for case, measured, baseline in zip(parent["rows"], result["rows"], previous["predictions"], strict=True):
            assert case["id"] == measured["id"] == baseline["id"]
            if baseline["status"] != "ok" and not any(row["checked_windows"] for row in measured["readings"]):
                predictions.append(baseline)
                continue
            before = baseline["localization"].get("regions", ())
            extra = ranges(measured["readings"])[arm]
            selected = sorted(set(map(tuple, before)) | set(extra))
            predictions.append({"id": baseline["id"], "input_sha256": baseline["input_sha256"], "status": "ok",
                                "response": int(bool(selected)), "localization": {"status": "ok", "regions": [list(span) for span in selected]}})
        response = previous | {"system": arm, "implementation_sha256": independent.digest([SCHEMA, METHOD, arm]),
                               "weight_sha256": sorted(set(previous["weight_sha256"]) | set(result["focused_assets"].values())
                                                       | {result["activity_sha256"]}), "predictions": predictions}
        events = count_events(manifest, truth, response, minimum_iou=.30, minimum_recall=.50)
        coverage = count_locations(manifest, truth, response)
        for condition in ("clean", "overlap_-5dB", "overlap_-10dB"):
            expected = result["summary"][f"{condition}/{arm}"]
            assert sum(item["matched_events"] for item in events["strata"] if item["condition"] == condition and item["label"] == 1) == expected["located"]
            masks = [row["localization"] for case, row in zip(manifest["cases"], coverage["predictions"], strict=True)
                     if case["condition"] == condition and row["status"] == "ok"]
            assert sum(item["reported_units"] - item["overlap_units"] for item in masks) == expected["genuine_samples_marked"]
            for kind in ("ordinary", "background"):
                assert sum(pred["response"] == 1 for case, pred in zip(parent["rows"], predictions, strict=True)
                           if case["condition"] == condition and case["kind"] == kind) == expected.get(f"{kind}_flagged", 0)
        save(output / f"{arm}.json", response)
        save(output / f"{arm}-events.json", events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, fields in (("score", ("baseline", "media", "output")), ("review", ("baseline", "receipt")),
                         ("export", ("baseline", "receipt", "output"))):
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.action == "score":
        score(args.baseline, args.media, args.output)
    else:
        result = read(args.receipt)
        assert result["schema"] == SCHEMA and result["method"] == METHOD and result["review_point"] == focused.REVIEW_POINT
        assert result["baseline_sha256"] == digest(args.baseline.read_bytes())
        parent = read(args.baseline)
        counts = summarize(parent, result["rows"])
        assert counts == result["summary"]
        if args.action == "export":
            export(parent, result, args.output)
        else:
            print(json.dumps(counts, indent=2))
