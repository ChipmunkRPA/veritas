# SPDX-License-Identifier: MIT
"""Diagnose short voice localization with the exact isolated source tracks."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_two_voice_components as study

SOURCE, SCORING, RATE = study.SOURCE, study.SCORING, study.RATE
GRID = study.prior.parent.GRID
PARENT = study.PHONEME.with_name("audio-two-voice-components-2026-11-15.json.gz")
PARENT_SHA = "0a42082d8e3e05a7a805caa604a22d8573279f38e8bc7e7665f9f3628d21c5f7"
VIEWS = ("mix", "matched_estimate", "isolated_room", "isolated_dry")
ISOLATED = VIEWS[2:]
STATES = ("untouched", "genuine", "synthetic")
METHOD = {
    "schema": "chipmunk.audio-two-voice-isolation/1",
    "inputs": "Reuse the exact factorial benchmark. Compare the mixture and prior spatial estimates with the exact "
    "room-rendered source alone and its dry source on the same 6 s clock before room convolution. Also replay the "
    "current 2 s app scan on the exact isolated rendered sources without changing its point.",
    "matching": "At each one-second window, match an estimate to a known voice only when its SI-SDR is at least "
    "1 dB closer to that voice than to the other. Choose the best matching estimate with an admitted score, not the "
    "largest classifier response. If none qualify, that voice-window has no estimated reading.",
    "points": "Each one-second view gets its own development genuine maximum. A separate predeclared point uses "
    "the second-highest speaker-pair maximum. Use only the untouched and both-genuinely-edited deliveries. "
    "Select the strongest admitted window with earlier ties and require strict excess.",
    "edit_clock": "In the diagnostic ranking, select the one-second grid position with the best geometric overlap "
    "with that voice's annotated edit, before inspecting scores. At that identical window check whether the "
    "synthetic reading exceeds the genuine-edit reading both with and without the other voice edited.",
}


def parent():
    SOURCE.check(SOURCE.path_sha(PARENT) == PARENT_SHA, "Published parent receipt differs")
    with gzip.open(PARENT, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    SOURCE.check(study.report(payload) == payload["results"], "Parent result disagrees with replay")
    return payload


def references(plan, phase):
    source_arm = {"untouched": ("untouched", "untouched"), "genuine": ("neither", "neither"),
                  "synthetic": ("foreground", "background")}
    for index, scene, condition, _ in study.cases(plan, phase):
        # Each case state uses the same stems across its matched deliveries.
        if condition != "untouched":
            continue
        for side, voice in enumerate(scene["voices"]):
            for state in STATES:
                mix = scene["mixes"][source_arm[state][side]]
                yield index, side, state, voice, mix["stems"][side]


def dry_inputs(phoneme, stock):
    raw = {}
    for cohort in (study.phoneme_clips(phoneme), study.stock_clips(stock)):
        for _, _, _, clips in cohort:
            for clip in clips:
                raw.update((clip["variants"][state], samples) for state, samples in clip["samples"].items())
    return raw


def frozen_plan(root, baseline):
    saved = root / "freeze.json"
    plan = SOURCE.read(saved)
    SOURCE.check(plan == baseline["plan"] and SOURCE.path_sha(saved) == baseline["held"]["plan_sha256"],
                 "Local stem roster does not match the published benchmark")
    return plan


def assemble(plan, record, phase):
    SOURCE.check(record["phase"] == phase and record["method"] == METHOD and record["parent_sha256"] == PARENT_SHA,
                 "Isolation method or parent source differs")
    rows = list(references(plan, phase))
    by_source = {}
    for (index, side, state, voice, stem), row in zip(rows, record["rows"], strict=True):
        SOURCE.check(row["key"] == [index, side, state] and row["stem"] == stem and row["dry"] == voice["variants"][state],
                     "Exact source changed")
        SOURCE.check(set(row["scores"]) == set(ISOLATED) and all(map(study.prior.valid_grid, row["scores"].values())),
                     "Isolated score grid differs")
        by_source[index, side, state] = row["scores"]
    return by_source


def matched_estimate(row, side):
    readings = []
    for position in range(len(GRID)):
        candidates = []
        for index, stream in enumerate(row["streams"]):
            score = stream["scores"][position]
            direct, other = (stream["quality"][position][idx] for idx in (side, 1 - side))
            if score is not None and direct is not None and other is not None and direct - other >= 1:
                candidates.append((direct, -index, score))
        readings.append(max(candidates)[2] if candidates else None)
    return readings


def sample_views(baseline, record, phase):
    plan = baseline["plan"]
    isolated = assemble(plan, record, phase)
    for scene, condition, row in study.bind(plan, baseline[phase], phase):
        index = row["scene"]
        for side, state in enumerate(study.CONDITIONS[condition]):
            yield index, scene, condition, side, state, {"mix": row["mono"], "matched_estimate": matched_estimate(row, side),
                                                       **isolated[index, side, state]}


def calibrate(baseline, development):
    maxima = defaultdict(dict)
    for _, scene, condition, _, _, views in sample_views(baseline, development, "development"):
        if condition not in ("untouched", "neither"):
            continue
        for arm, values in views.items():
            score = max((v for v in values if v is not None), default=None)
            if score is not None:
                group = scene["pair"]
                maxima[arm][group] = max(maxima[arm].get(group, score), score)
    points = {}
    for arm in VIEWS:
        ranked = sorted(maxima[arm].values(), reverse=True)
        SOURCE.check(len(ranked) >= 2, "Too few development pairs with a usable source window")
        points[arm] = {"strict": ranked[0], "one_pair": ranked[1]}
    return points


def measure(args):
    import numpy as np

    baseline = parent()
    plan = frozen_plan(args.inputs, baseline)
    raw = dry_inputs(args.phoneme, args.stock)
    points = calibrate(baseline, SOURCE.read(args.development)) if args.development else None
    SOURCE.check((points is not None) == (args.phase == "held"), "Held inference requires the development readings")
    args.output.mkdir(mode=0o700)
    lock = {"phase": args.phase, "method": METHOD, "parent_sha256": PARENT_SHA, "points": points}
    SOURCE.write(args.output / "freeze.json", lock)
    model, extract, speech = SCORING.load_classifier(args.weights, args.speech)

    def classify(signal):
        pcm = signal.astype(np.float32) / 32768
        return SCORING.window_log_odds(model, extract, pcm, speech.spans(pcm), RATE)

    rows = []
    for index, side, state, voice, stem in references(plan, args.phase):
        key = voice["variants"][state]
        dry = raw[key]
        SOURCE.check(SOURCE.digest(dry.tobytes()) == key, "Dry source bytes differ")
        pre_room = np.rint(study.position(dry, voice["shift"])).astype("<i2")
        rendered = np.rint(study.load_stereo(args.inputs, stem).mean(axis=1)).astype("<i2")
        rows.append({"key": [index, side, state], "stem": stem, "dry": key,
                     "scores": {"isolated_room": classify(rendered), "isolated_dry": classify(pre_room)}})
        if len(rows) % 12 == 0:
            print(f"{args.phase}: measured {len(rows)} source versions", flush=True)
    SOURCE.write(args.output / "scores.json", {**lock, "rows": rows})


def current(args):
    import numpy as np
    import torch

    from app.detectors.audio import short_voice_review
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    plan = frozen_plan(args.inputs, parent())
    network_free()
    model = load_voice({"weights": args.weights, "assets": ASSETS})
    torch.set_num_threads(2)
    speech = SpeechActivity(args.weights / "silero_vad.jit")

    class Deadline:
        def check(self, _stage):
            pass  # Run to completion in the diagnostic CLI.

    rows = []
    for index, side, state, _, stem in references(plan, "held"):
        signal = np.rint(study.load_stereo(args.inputs, stem).mean(axis=1)).astype(np.float32) / 32768
        scan = short_voice_review.scan(signal, speech.spans(signal), model, Deadline(), codec="pcm_s16le")
        selection = scan["selected"]
        window = [selection["start_sample"], selection["end_sample"]] if selection else None
        rows.append({"key": [index, side, state], "stem": stem, "window": window,
                     "speech_counts": scan["speech_counts"], "responses": scan["responses"]})
        if len(rows) % 12 == 0:
            print(f"Current scan: measured {len(rows)} isolated source versions", flush=True)
    SOURCE.write(args.output, {"parent_sha256": PARENT_SHA, "assets": ASSETS, "rows": rows})


def edit_clock(voice, view):
    delay = voice["direct_arrival"] if view == "isolated_dry" else 0
    return [value - delay for value in voice["span"]]


def geometric_window(span):
    lo, hi = span
    overlap = [min(start + RATE, hi) - max(start, lo) for start in GRID]
    best = max(range(len(GRID)), key=overlap.__getitem__)
    SOURCE.check(SCORING._useful([GRID[best], GRID[best] + RATE], span), "The one-second grid cannot reach this edit")
    return best


def fixed_clock(plan, views):
    metrics = defaultdict(Counter)
    for index, scene in enumerate(plan["scenes"]):
        if scene["cohort"] == "development":
            continue
        for side, voice in enumerate(scene["voices"]):
            target = "foreground" if side == 0 else "background"
            competitor = "background" if side == 0 else "foreground"
            for arm in VIEWS:
                window = geometric_window(edit_clock(voice, arm))
                ordinary, solo_edit, other_edit, both = (views[index, side, state][arm][window]
                                                        for state in ("neither", target, competitor, "both"))
                for group in ("all", voice["family"]):
                    row = metrics[scene["cohort"], arm, group]
                    row["sources"] += 1
                    solo_measured = ordinary is not None and solo_edit is not None
                    both_measured = other_edit is not None and both is not None
                    row["comparable_genuine_competitor"] += solo_measured
                    row["comparable_synthetic_competitor"] += both_measured
                    row["higher_genuine_competitor"] += solo_measured and solo_edit > ordinary
                    row["higher_synthetic_competitor"] += both_measured and both > other_edit
                    row["comparable_in_both"] += solo_measured and both_measured
                    row["higher_in_both"] += solo_measured and both_measured and solo_edit > ordinary and both > other_edit
    return {"/".join(key): dict(value) for key, value in sorted(metrics.items())}


def app_views(baseline, record):
    SOURCE.check(record["parent_sha256"] == PARENT_SHA and record["assets"] == baseline["current"]["assets"],
                 "Current scanner did not use the frozen parent weights")
    readings = {}
    for (index, side, state, _, stem), row in zip(references(baseline["plan"], "held"), record["rows"], strict=True):
        SOURCE.check(row["key"] == [index, side, state] and row["stem"] == stem, "Current scan evaluated a different source")
        readings[index, side, state] = study.app_reading(row)
    return readings


def measurements(plan, views, points, current_readings):
    out = defaultdict(Counter)
    for (index, side, state), scan in current_readings.items():
        scene = plan["scenes"][index]
        voice = scene["voices"][side]
        true_change = state == "synthetic"
        condition = ("foreground", "background")[side] if true_change else "neither" if state == "genuine" else state
        readings = [("current_isolated", "serving", scan[0], scan[2])]
        for view, values in views[index, side, condition].items():
            strongest = study.prior.parent.ranked([values])
            for setting, point in points[view].items():
                selected = study.prior.parent.chosen(strongest, point)
                readings.append((view, setting, selected, strongest is not None))
        for arm, setting, interval, assessed in readings:
            counts = out[scene["cohort"], arm, setting, state]
            counts["sources"] += 1
            counts["assessed"] += assessed
            counts["flagged"] += interval is not None
            if true_change:
                precise = SCORING._useful(interval, edit_clock(voice, arm))
                counts["useful"] += precise
                counts["wrong_time"] += bool(interval and not precise)
    return {"/".join(key): dict(value) for key, value in sorted(out.items())}


def app_transition(baseline, exact):
    source = {(row["scene"], row["condition"]): row["window"] for row in baseline["current"]["rows"]}
    counts = defaultdict(Counter)
    for (index, side, state), (localized, _, _) in exact.items():
        if state != "synthetic":
            continue
        scene = baseline["plan"]["scenes"][index]
        truth = scene["voices"][side]["span"]
        before = SCORING._useful(source[index, ("foreground", "background")[side]], truth)
        after = SCORING._useful(localized, truth)
        cohort = counts[scene["cohort"]]
        cohort["retained"] += before and after
        cohort["gained"] += after and not before
        cohort["lost"] += before and not after
        cohort["missed_in_both"] += not before and not after
    return {key: dict(value) for key, value in sorted(counts.items())}


def report(baseline, payload):
    development, held = payload["development"], payload["held"]
    points = calibrate(baseline, development)
    SOURCE.check(development["points"] is None and points == held["points"], "Held operating points changed")
    views = {(index, side, condition): values for index, _, condition, side, _, values in
             sample_views(baseline, held, "held")}
    exact = app_views(baseline, payload["current"])
    return {"fixed_clock": fixed_clock(baseline["plan"], views),
            "localization": measurements(baseline["plan"], views, points, exact),
            "app_transition": {f"{key}/mix_to_exact_room": value for key, value in app_transition(baseline, exact).items()}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="command", required=True)
    run = modes.add_parser("measure")
    run.add_argument("phase", choices=("development", "held"))
    for name in ("inputs", "output", "phoneme", "stock", "weights", "speech"):
        run.add_argument(name, type=Path)
    run.add_argument("--development", type=Path)
    app = modes.add_parser("current")
    for name in ("inputs", "output", "weights"):
        app.add_argument(name, type=Path)
    combine = modes.add_parser("combine")
    for name in ("development", "held", "current", "output"):
        combine.add_argument(name, type=Path)
    modes.add_parser("review").add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.command == "measure":
        measure(args)
    elif args.command == "current":
        current(args)
    elif args.command == "combine":
        payload = {name: SOURCE.read(getattr(args, name)) for name in ("development", "held", "current")}
        payload["results"] = report(parent(), payload)
        SOURCE.write(args.output, payload)
    else:
        open_receipt = gzip.open if args.receipt.suffix == ".gz" else Path.open
        with open_receipt(args.receipt, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        counts = report(parent(), payload)
        SOURCE.check(counts == payload["results"], "The receipt does not match its saved outcomes")
        print(f"Replayed {len(payload['held']['rows'])} held voice versions")
        for category, group in counts.items():
            for scope, values in group.items():
                print(category, scope, values)


if __name__ == "__main__":
    main()
