# SPDX-License-Identifier: MIT
"""Compare finite input perturbations for narrowing existing speech alerts."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_input_contribution_transfer as sensitivity
from scripts import audio_narrow_flag_transfer as nested
from scripts import audio_shared_mel_context as shared
from scripts import audio_short_edit_detector_training as training

SOURCE = shared.source
SCHEMA = "chipmunk.audio-finite-window-attribution/1"
PRIMARY = shared.PANELS / "audio-short-edit-detector-training-2026-11-14.json"
NEW_SINE = shared.PANELS / "audio-narrow-flag-transfer-2026-11-14.json"
POINTS_PARENT = sensitivity.PUBLISHED_PARENT
PANELS = ("development", "random_placement", "families", "primary", "new_sine")
WIDTH, STEP = 16000, 8000
CONTEXT = 64000
STRIDE, FIRST = 320, 280
UPPER, TAPER, FLANK = 40, 5, 12
MIN_FRAMES = 40
ARMS = ("dense", "both_fills", "same_winner")
METHOD = {
    "geometry": "Keep the existing two-second alert. Examine three contained one-second windows on the half-second grid. "
                "Use the same original four-second context for all three, choosing the context with the larger minimum flank "
                "at the broad alert edges and the earlier context on ties. Use the normal one-second speech admission.",
    "perturbation": "Before inference, taper the upper 40 normalized Mel bands inside one candidate to zero, or to the "
                    "median of the nearest 12 feature positions on each available side. Use a five-frame linear taper. "
                    "Retain all other features and every attention position. Measure the original context logit minus the "
                    "perturbed logit. Keep the lower drop, with a separate agreement route requiring the same unique winner "
                    "for both fills. The density comparator uses the median unperturbed local frame response.",
    "points": "Freeze a strict and wider cut separately for each readout using existing broad flags in the stock development "
              "panel. Use the highest and second-highest genuine source group maxima, respectively; keep a zero floor for "
              "the two finite-drop readouts. When fewer than two groups support a same-winner point, reuse the two-fill "
              "points. Retain the broad window below or on the point, or when no candidate qualifies.",
    "evaluation": "A useful interval covers at least half the true edit and reaches 0.30 temporal IoU. Report correct and "
                  "misplaced refinements, losses, genuine and same-word refinements, and same-clock matches. The methods "
                  "localize existing alerts so they cannot recover missed alerts or add a new file flag.",
}


def input_row(row, *, summary=False, condition=None):
    broad = shared.existing(row | {"existing_summary": summary})
    return {"pcm_sha256": row["pcm_sha256"], "group": row["group"], "kind": row["kind"],
            "condition": condition or "competing_speech", "truth": row["truth"], "broad": list(broad) if broad else None}


def panels():
    results = {panel: [input_row(row, summary=panel == "families") for row in shared.input_rows(panel)] for panel in PANELS[:3]}
    primary = SOURCE.read(PRIMARY)
    noisy = [row for row in primary["sources"]["rows"] if row["role"] == "held" and row["condition"] == "competing_speech"]
    results["primary"] = []
    for row, before in zip(noisy, primary["incumbent"]["results"], strict=True):
        SOURCE.check(row["pcm_sha256"] == before["pcm_sha256"], "Primary decision differs from its source")
        results["primary"].append({key: row[key] for key in ("pcm_sha256", "group", "kind", "condition", "truth")} | {"broad": before["selected"]})
    recent = SOURCE.read(NEW_SINE)
    results["new_sine"] = [input_row(row, condition=row["condition"]) for row in recent["records"]]
    return results


def method(context):
    if context == CONTEXT:
        return METHOD
    if context != 2 * WIDTH:
        raise ValueError("Unsupported context length")
    return METHOD | {"geometry": "Use the exact two-second crop that raised the original alert. Examine three contained "
                                 "one-second windows on the half-second grid with the normal one-second speech admission."}


def bounds(broad, context=CONTEXT):
    start, end = broad
    origin = start if context == 2 * WIDTH else max(
        (offset for offset in SOURCE.CUTS if offset <= start and end <= offset + context),
        key=lambda offset: (min(start - offset, offset + context - end), -offset),
    )
    return origin, list(range(start, end - WIDTH + 1, STEP))


def measure(model, activity, pcm, broad, audit, context=CONTEXT):
    import numpy as np
    import torch

    from app.detectors.audio.content import window_in_speech

    offset, starts = bounds(broad, context)
    signal, features = sensitivity._normalized_input(pcm[offset:offset + context])
    with torch.inference_mode():
        original = sensitivity._frames(model, features)
        baseline = float(original.mean())
        if "full_logit_error" not in audit:
            error = abs(float(model(signal[None])) - baseline)
            SOURCE.check(error <= 0.00002, "Context pooling differs from the existing voice model")
            audit["full_logit_error"] = error
        span = activity.spans(pcm)
        centers = offset + FIRST + np.arange(len(original)) * STRIDE
        source_bands = features.reshape(len(original), 2, 80)
        results = []
        for start in starts:
            selected = np.flatnonzero((centers >= start) & (centers < start + WIDTH))
            if len(selected) < MIN_FRAMES or not window_in_speech(span, start / training.RATE, (start + WIDTH) / training.RATE):
                results.append(None)
                continue
            lo, hi = int(selected[0]), int(selected[-1]) + 1
            held = torch.cat((source_bands[max(0, lo - FLANK):lo], source_bands[hi:hi + FLANK]))
            if len(held) < FLANK:
                results.append(None)
                continue
            original_bands = source_bands[lo:hi, :, UPPER:]
            ramp = torch.arange(1, hi - lo + 1, dtype=features.dtype)
            taper = (torch.minimum(ramp, ramp.flip(0)) / TAPER).clamp(max=1.)[:, None, None]
            median = held[:, :, UPPER:].median(dim=0).values
            replacement_scores = []
            for replacement in (0., median):
                changed = features.clone()
                changed.reshape_as(source_bands)[lo:hi, :, UPPER:] = original_bands * (1 - taper) + replacement * taper
                response = float(sensitivity._frames(model, changed).mean())
                replacement_scores.append(round(baseline - response, 8))
            results.append({"dense": round(float(np.median(original[lo:hi].numpy())), 8),
                            "to_mean": replacement_scores[0], "to_flanks": replacement_scores[1]})
    return {"context_start": offset, "baseline": round(baseline, 8), "windows": results}


def best(row, arm):
    if row["response"] is None:
        return None
    windows = row["response"]["windows"]
    choices = [(index, item) for index, item in enumerate(windows) if item is not None]
    if not choices:
        return None
    if arm == "same_winner":
        winners = []
        for name in ("to_mean", "to_flanks"):
            maximum = max(item[name] for _, item in choices)
            winners.append([index for index, item in choices if item[name] == maximum])
        if winners[0] != winners[1] or len(winners[0]) != 1:
            return None
        choices = [(index, item) for index, item in choices if index == winners[0][0]]
    values = [(item["dense"] if arm == "dense" else min(item["to_mean"], item["to_flanks"]), -index) for index, item in choices]
    score, negative_index = max(values)
    start = row["broad"][0] - negative_index * STEP
    return score, [start, start + WIDTH]


def fit(rows):
    result = {}
    for arm in ARMS:
        groups = {}
        for row in rows:
            if row["truth"] is None and (candidate := best(row, arm)) is not None:
                groups[row["group"]] = max(groups.get(row["group"], -math.inf), candidate[0])
        ordered = sorted(groups.values(), reverse=True)
        floor = 0. if arm != "dense" else -math.inf
        SOURCE.check(len(ordered) >= 2 or arm == "same_winner", "Too few genuine control groups")
        if len(ordered) < 2:
            # With no cross-fill agreement in enough control groups, use the same floor and points
            # as the broader two-fill ranking, rather than infer a tail from one example.
            result[arm] = result["both_fills"] | {"groups": len(ordered), "uses_both_fills_points": True}
        else:
            result[arm] = {"strict": max(floor, ordered[0]), "wider": max(floor, ordered[1]), "groups": len(ordered)}
    return result


def count(rows, arm, point):
    counts = {}
    locations, genuine = [], set()
    control_ranges = defaultdict(list)
    for row in rows:
        condition = row["condition"]
        record = counts.setdefault(condition, Counter())
        broad, truth = row["broad"], row["truth"]
        candidate = best(row, arm)
        selected = candidate[1] if candidate is not None and candidate[0] > point else None
        final = selected or broad
        if truth:
            before = bool(broad and shared.stock.useful(broad, truth))
            after = bool(final and shared.stock.useful(final, truth))
            record.update(edits=1, alerted_edits=bool(broad), useful_before=before, useful_after=after,
                          lost_useful=before and not after, added_useful=after and not before,
                          misplaced_refinements=bool(selected) and not after)
            if after and not before:
                locations.append({key: row[key] for key in ("group", "kind", "condition", "truth", "pcm_sha256")} | {"range": final})
        else:
            category = "isolated" if row["kind"] in shared.interval.BACKGROUNDS else "same_words" if row["kind"] == "resyn" else "ordinary"
            record.update({category: 1, f"{category}_alerts": bool(broad), f"{category}_refined": bool(selected)})
            if final:
                control_ranges[condition, row["group"]].append(final)
            if category == "ordinary" and selected:
                genuine.add((condition, row["group"]))
    for location in locations:
        condition, group = location["condition"], location["group"]
        other = control_ranges[condition, group]
        location["same_clock_control"] = any(shared.stock.useful(interval, location["truth"]) for interval in other)
        counts[condition]["added_useful_without_same_clock_control"] += not location["same_clock_control"]
    return {"by_condition": {k: dict(v) for k, v in counts.items()}, "new_useful": locations,
            "refined_ordinary_groups": {condition: sum(c == condition for c, _ in genuine) for condition in counts}}


def bindings():
    pins = {path.name: SOURCE.path_sha(path) for path in (PRIMARY, NEW_SINE, POINTS_PARENT)}
    SOURCE.check(pins[POINTS_PARENT.name] == sensitivity.PUBLISHED_PARENT_SHA256, "Published comparison differs")
    return SOURCE.read(POINTS_PARENT)["freeze"]["bindings"] | pins


def same_context_check(receipt):
    references = {name: shared.input_rows(name) for name in PANELS[:3]}
    references["new_sine"] = SOURCE.read(NEW_SINE)["records"]
    errors = []
    for name, notes in references.items():
        for row, old in zip(receipt["panels"][name], notes, strict=True):
            if row["response"] is None:
                continue
            scores = old["existing_scores"]
            reported = scores[0] if name == "families" else scores[shared.stock.WINDOWS.index(tuple(row["broad"]))]
            actual = row["response"]["baseline"]
            errors.append(abs(math.tanh(reported / 2) - math.tanh(actual / 2)) / 2)
    SOURCE.check(errors and max(errors) <= .000002, "The same-context forward differs from the archived alert score")
    return {"archived_probabilities_checked": len(errors), "largest_difference": max(errors)}


def review(receipt):
    context = receipt.get("context_samples", CONTEXT)
    SOURCE.check(receipt["schema"] == SCHEMA and receipt["method"] == method(context) and receipt["bindings"] == bindings(), "Frozen protocol differs")
    expected = panels()
    SOURCE.check(set(receipt["panels"]) == set(expected), "A frozen panel is missing")
    for name, rows in receipt["panels"].items():
        for row, original in zip(rows, expected[name], strict=True):
            SOURCE.check({key: row[key] for key in original} == original, "A source or broad result differs")
            response = row["response"]
            if row["broad"] is None:
                SOURCE.check(response is None, "Unexpected refinement without an existing flag")
                continue
            offset, starts = bounds(row["broad"], context)
            SOURCE.check(response["context_start"] == offset and math.isfinite(response["baseline"])
                         and len(response["windows"]) == len(starts), "Refinement clock differs")
            SOURCE.check(all(item is None or (set(item) == {"dense", "to_mean", "to_flanks"}
                         and all(math.isfinite(value) for value in item.values())) for item in response["windows"]), "Invalid response")
    points = fit(receipt["panels"]["development"])
    SOURCE.check(receipt["points"] == points and 0 <= receipt["audit"]["full_logit_error"] <= .00002,
                 "Point or forward check does not reproduce")
    if context == 2 * WIDTH:
        archived = same_context_check(receipt)
        SOURCE.check(archived["archived_probabilities_checked"] == receipt["audit"]["archived_probabilities_checked"]
                     and math.isclose(archived["largest_difference"], receipt["audit"]["largest_difference"], abs_tol=1e-14),
                     "Archived score parity differs from the recorded audit")
    return {panel: {arm: {level: count(rows, arm, points[arm][level]) for level in ("strict", "wider")} for arm in ARMS}
            for panel, rows in receipt["panels"].items()}


def score(args):
    import numpy as np
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    expected = panels()
    args.path.mkdir(mode=0o700)
    rows = {}
    context = args.context_seconds * training.RATE
    freeze = {"schema": SCHEMA, "method": method(context), "bindings": bindings(), "context_samples": context}
    SOURCE.write(args.path / "freeze.json", freeze)
    _, bank, _ = shared.interval.native_rows(args.archive)
    fresh = SOURCE.read(NEW_SINE)
    SOURCE.check(SOURCE.read(args.sine / "freeze.json") == fresh["freeze"], "The later SINE source roster differs")
    for _, pcm in nested.sine.native_rows(args.sine, fresh["freeze"]):
        raw = pcm.tobytes()
        bank[SOURCE.digest(raw)] = raw
    plan, digest = training.load_plan(args.data)
    SOURCE.check(digest == SOURCE.read(PRIMARY)["incumbent"]["source_sha256"], "Primary native source roster differs")
    primary = {row["pcm_sha256"]: row for row in plan["rows"] if row["role"] == "held"}
    network_free()
    torch.set_num_threads(3)
    weights = args.weights
    model = load_voice({"weights": weights, "assets": ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    audit = {}
    for name, roster in expected.items():
        samples = {}
        rows[name] = []
        for row in roster:
            key = row["pcm_sha256"]
            if row["broad"] is not None and key not in samples:
                if name == "primary":
                    pcm = training.pcm(args.data, primary[key])
                else:
                    raw = bank[key]
                    SOURCE.check(SOURCE.digest(raw) == key, "The prepared mixture changed")
                    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.
                samples[key] = measure(model, activity, pcm, row["broad"], audit, context)
                if len(samples) % 16 == 0:
                    print(f"{name}: measured {len(samples)} existing alerts", flush=True)
            rows[name].append(row | {"response": samples[key] if row["broad"] is not None else None})
        SOURCE.write(args.path / f"{name}.json", rows[name])
        print(f"{name}: all {len(roster)} deliveries, {len(samples)} distinct existing alerts", flush=True)
        if name == "development":
            points = fit(rows[name])
            SOURCE.write(args.path / "points.json", points)
    receipt = freeze | {"points": points, "panels": rows, "audit": audit}
    if context == 2 * WIDTH:
        receipt["audit"].update(same_context_check(receipt))
    receipt["results"] = review(receipt)
    SOURCE.write(args.path / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("score", "review"))
    cli.add_argument("path", type=Path)
    cli.add_argument("--archive", type=Path)
    cli.add_argument("--sine", type=Path)
    cli.add_argument("--data", type=Path)
    cli.add_argument("--weights", type=Path, default=SOURCE.ROOT / "models/weights/audio-upgrade")
    cli.add_argument("--context-seconds", type=int, choices=(2, 4), default=4)
    args = cli.parse_args()
    if args.phase == "score" and not all((args.archive, args.sine, args.data)):
        cli.error("Native scoring requires --archive, --sine and --data")
    receipt = score(args) if args.phase == "score" else SOURCE.read(args.path)
    output = review(receipt)
    SOURCE.check(receipt["results"] == output, "Saved outcome differs from the replay")
    print(json.dumps(output, indent=2, sort_keys=True))
