# SPDX-License-Identifier: MIT
"""Test partial attenuation of each estimated voice on a fixed source clock."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_mixture_phase_isolation as parent
from scripts import review_audio_random_clock_stock as stock

SOURCE = parent.SOURCE
SCORING = parent.scoring
PANELS = ("ami", "icsi")
PARENT = parent.PARENT.with_name("audio-mixture-phase-isolation-2026-11-14.json")
STOCK = parent.PARENT.with_name("audio-random-clock-stock-2026-11-13.json")
METHOD = {
    "schema": "chipmunk.audio-stream-attenuation/1",
    "level": 0.5,
    "minimum_selective_response_log_odds": math.log(2),
    "inputs": "Reuse all 96 frozen six-second mono deliveries at -5 dB and their two mixture-phase estimates. "
    "Subtract half of each estimate in turn. Require both estimates and both partially attenuated readings "
    "at the same one-second source window. Use the existing 50 percent speech requirement and 0.5 s hop.",
    "scores": "Compare the highest response after either attenuation with a selective response that must be "
    "at least log(2) higher when attenuating the other estimate than when attenuating the supported estimate. "
    "Also test a rule that requires support from the submitted mix at the same interval, using the lower score.",
    "points": "For each rule use the second-highest AMI ordinary group maximum, bounded below by the "
    "maximum isolated background score. Use strict excess, choose one range, break ties at the earliest "
    "window, and freeze AMI points before ICSI inference. Compare fallback only where the app has no alert.",
    "source_check": "At the known edit clock, compare the inferred stream against the benchmark foreground "
    "using the parent acoustic diagnostic. Source qualities never enter the score or thresholds.",
}
ARMS = ("partial", "selective", "selective_with_mix")
STOCK_METHOD = {"sources": "All minus-five-dB clips plus the three isolated backgrounds from the fixed random-clock "
                "stock-voice panel. Use the same separator, waveform operations, and AMI cutoffs, without refitting.",
                "baseline": "Replay the live short-review point. Apply the same frozen one-second mixed point. "
                "Count a matched collision on the untouched clip or either genuine splice. The room-only "
                "controls are separate excerpts and count as controls without a matched clock."}


def lock(panel):
    return {"method": METHOD, "geometry": parent.lock(panel), "parent_sha256": SOURCE.path_sha(PARENT)}


def stock_lock():
    return {"method": METHOD, "transfer": STOCK_METHOD, "source_sha256": SOURCE.path_sha(STOCK),
            "development": lock("ami"), "classifier": SCORING._lock("ami"), "separator": parent.SEPARATOR}


def baseline_rows(panel):
    previous = SOURCE.read(PARENT)["panels"][panel]
    roster, _ = SCORING._parent(panel)
    incumbent = {result["pcm_sha256"]: result for result in roster["measurements"]}
    for clip, row in parent.iter_results(previous, panel):
        yield clip, row, incumbent[clip["pcm_sha256"]]


def best(scores, mixture_scores, arm):
    if not scores:
        return None
    chosen = None
    for start, mix, first, second in zip(parent.GRID, mixture_scores, *scores, strict=True):
        if first is None or second is None:
            continue
        high, low = max(first, second), min(first, second)
        if arm != "partial" and high - low < METHOD["minimum_selective_response_log_odds"]:
            continue
        # Attenuating the other estimate preserved the larger response.
        favored_stream = int(first > second) if first != second else None
        if arm == "selective_with_mix":
            if mix is None:
                continue
            high = min(mix, high)
        item = high, -start, favored_stream
        if chosen is None or item > chosen:
            chosen = item
    return chosen


def valid_grid(values):
    return len(values) == len(parent.GRID) and all(
        value is None or type(value) in (float, int) and math.isfinite(value) for value in values)


def joined(panel, record):
    SOURCE.check(record["freeze"] == lock(panel), "Method or original source commitment differs")
    for (clip, parent_row, incumbent), row in zip(baseline_rows(panel), record["rows"], strict=True):
        SOURCE.check(clip["pcm_sha256"] == row["pcm_sha256"], "Original PCM differs")
        hashes, scores = row["variants"], row["scores"]
        expected = 2 if len(parent_row["views"]["mixture_phase"]) == 2 else 0
        SOURCE.check(len(hashes) == len(scores) == expected and all(
            isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
            for value in hashes), "Invalid attenuation commitments")
        SOURCE.check(all(map(valid_grid, scores)), "Invalid response on the frozen grid")
        yield clip, parent_row, incumbent, scores


def calibrate(record):
    groups = defaultdict(dict)
    background = defaultdict(list)
    for clip, parent_row, _, scores in joined("ami", record):
        if clip["truth"] is not None:
            continue
        for arm in ARMS:
            reading = best(scores, parent_row["scores"]["mix"][0], arm)
            if reading is None:
                continue
            value = reading[0]
            if clip["kind"] == "background":
                background[arm].append(value)
            else:
                group = clip["group"]
                groups[arm][group] = max(groups[arm].get(group, -math.inf), value)
    for arm in ARMS:
        SOURCE.check(len(groups[arm]) >= 2 and background[arm], f"Insufficient calibration coverage for {arm}")
    return {arm: max(sorted(groups[arm].values())[-2], max(background[arm])) for arm in ARMS}


def report(rows, points, prior_points):
    rows = list(rows)
    answers = defaultdict(dict)
    for clip, old, baseline, scores in rows:
        key = clip["group"], clip["kind"]
        incumbent = baseline["current"], None, baseline["speech_admitted"]
        answers["incumbent"][key] = incumbent
        for method in ("mix", "mixture_phase"):
            if method not in old["scores"]:
                continue
            response = parent.ranked(old["scores"][method])
            answers[method][key] = parent.chosen(response, prior_points[method]), None, response is not None
        for arm in ARMS:
            response = best(scores, old["scores"]["mix"][0], arm)
            candidate = parent.chosen(response, points[arm])
            qualities = old["qualities"]["mixture_phase"].get("edit", [])
            tracks_edit = (response[2] == int(qualities[1] > qualities[0])
                           if candidate and response[2] is not None and len(qualities) == 2 and None not in qualities
                           and qualities[0] != qualities[1] else None)
            answers[arm][key] = candidate, tracks_edit, response is not None
            answers[f"incumbent_plus_{arm}"][key] = (incumbent if incumbent[0] is not None else
                                                     (candidate, tracks_edit, incumbent[2] or response is not None))

    results = {}
    for method, choices in answers.items():
        counts = Counter()
        ordinary_groups = set()
        for clip, _, _, _ in rows:
            group, kind, truth = clip["group"], clip["kind"], clip["truth"]
            location, tracks_edit, assessed = choices[group, kind]
            role = "edited" if truth else "background" if kind == "background" else "ordinary"
            counts[role] += 1
            counts[f"{role}_assessed"] += assessed
            counts[f"{role}_flags"] += location is not None
            if role == "ordinary" and location:
                ordinary_groups.add(group)
            if truth:
                useful = SCORING._useful(location, truth)
                counts["useful"] += useful
                counts["wrong_edit_window"] += bool(location and not useful)
                matched = clip.get("controls") or [(group, name) for name in (
                    "untouched", kind.replace("edited/", "ordinary/"), "background")]
                controls = (choices[tuple(control)][0] for control in matched)
                counts["useful_without_matched_collision"] += useful and not any(
                    SCORING._useful(control, truth) for control in controls)
                counts["useful_with_known_stream_comparison"] += useful and tracks_edit is not None
                counts["useful_follows_closest_stream"] += useful and tracks_edit is True
        results[method] = dict(counts, ordinary_groups_flagged=len(ordinary_groups))
    return results


def variant_signals(directory, views):
    import numpy as np

    def load(digest):
        raw = (directory / "signals" / f"{digest}.f32").read_bytes()
        SOURCE.check(len(raw) == 4 * parent.SAMPLES and SOURCE.digest(raw) == digest,
                     "Mixture or estimated samples do not match their commitment")
        return np.frombuffer(raw, "<f4")

    if len(views["mixture_phase"]) != 2:
        return []
    mix, first, second = map(load, (*views["mix"], *views["mixture_phase"]))
    return attenuate(mix, first, second)


def attenuate(mix, first, second):
    import numpy as np

    SOURCE.check(np.max(np.abs(first + second - mix)) < 1e-5, "The estimates do not reconstruct the mix")
    return [mix - METHOD["level"] * estimate for estimate in (first, second)]


def measure(args):
    points = calibrate(SOURCE.read(args.development)) if args.development else None
    SOURCE.check((args.panel == "icsi") == (points is not None), "Supply sealed AMI measurements only for ICSI")
    record = {"freeze": lock(args.panel), "points": points, "rows": []}
    args.output.mkdir(mode=0o700)
    SOURCE.write(args.output / "freeze.json", {"freeze": record["freeze"], "points": points})
    model, extract, speech = SCORING.load_classifier(args.weights, args.speech)
    for index, (clip, old, _) in enumerate(baseline_rows(args.panel), 1):
        row = {"pcm_sha256": clip["pcm_sha256"], "variants": [], "scores": []}
        for signal in variant_signals(args.views, old["views"]):
            row["variants"].append(SOURCE.digest(signal.astype("<f4", copy=False).tobytes()))
            row["scores"].append(SCORING.window_log_odds(model, extract, signal, speech.spans(signal), parent.RATE))
        record["rows"].append(row)
        if index % 8 == 0:
            print(f"{args.panel}: scored {index} deliveries", flush=True)
    SOURCE.write(args.output / "scored.json", record)


def stock_rows():
    reference = SOURCE.read(STOCK)
    stock.verify(reference)
    return [row for row in reference["rows"] if row["snr"] in (-5, None)]


def measure_stock(args):
    import numpy as np
    import onnxruntime as ort

    reference = SOURCE.read(STOCK)
    freeze = args.native / "placements-freeze.json"
    SOURCE.check(SOURCE.path_sha(freeze) == reference["native_freeze_sha256"], "Stock native plan differs")
    with np.load(args.native / "placements-native.npz", allow_pickle=False) as bank:
        signals, dry = bank["native"], bank["dry"]
    SOURCE.check(all(array.shape == (83, parent.SAMPLES) and array.dtype == np.dtype("<i2")
                     for array in (signals, dry)), "Stock clock differs")
    pcm, foreground = {}, {}
    existing = {clip["pcm_sha256"] for panel in PANELS for clip, _, _ in baseline_rows(panel)}
    for signal, target, row in zip(signals, dry, reference["rows"], strict=True):
        digest = row["pcm_sha256"]
        SOURCE.check(SOURCE.digest(signal.tobytes()) == digest and digest not in existing, "Stock PCM differs or overlaps")
        pcm[digest] = signal.astype(np.float32) / 32768
        if row["dry_pcm_sha256"] is not None:
            SOURCE.check(SOURCE.digest(target.tobytes()) == row["dry_pcm_sha256"], "Stock target PCM differs")
            foreground[digest] = target.astype(np.float32) / 32768
    SOURCE.check(SOURCE.path_sha(args.separator) == parent.SEPARATOR["sha256"], "Separator digest differs")
    points = calibrate(SOURCE.read(args.development))
    args.output.mkdir(mode=0o700)
    record = {"freeze": stock_lock(), "points": points, "rows": []}
    SOURCE.write(args.output / "freeze.json", {"freeze": record["freeze"], "points": points})
    model, feature, speech = SCORING.load_classifier(args.weights, args.speech)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.separator), sess_options=opts, providers=["CPUExecutionProvider"])

    def infer(signal):
        output, = session.run(["sources"], {"mix": signal})
        SOURCE.check(output.shape == (2, 4 * parent.RATE) and np.isfinite(output).all(), "Invalid estimated signals")
        return output

    def score(signal):
        return SCORING.window_log_odds(model, feature, signal, speech.spans(signal), parent.RATE)

    for i, clip in enumerate(stock_rows(), 1):
        key = clip["pcm_sha256"]
        mix = pcm[key]
        streams, _ = parent.select_streams(mix, infer)
        phase = parent.mixture_phase(mix, streams) if streams else []
        supported = len(phase) == 2 and all(np.linalg.norm(s) >= .08 * np.linalg.norm(mix) for s in phase)
        variants = attenuate(mix, *phase) if supported else []
        quality = []
        if clip["edit_clock"] and supported:
            start, stop = clip["edit_clock"]
            target = foreground[key][start:stop]
            quality = [parent.isolation.acoustic_quality(target, s[start:stop]) for s in phase]
        record["rows"].append({"pcm_sha256": key, "mix": score(mix), "quality": quality,
                               "variants": [SOURCE.digest(s.astype("<f4", copy=False).tobytes()) for s in variants],
                               "scores": [score(s) for s in variants]})
        if i % 5 == 0:
            print(f"Stock transfer: scored {i} deliveries", flush=True)
    SOURCE.write(args.output / "scored.json", record)


def joined_stock(record):
    SOURCE.check(record["freeze"] == stock_lock(), "Stock source commitment differs")
    for clip, row in zip(stock_rows(), record["rows"], strict=True):
        group, kind, truth = clip["group"], clip["kind"], clip["edit_clock"]
        SOURCE.check(row["pcm_sha256"] == clip["pcm_sha256"], "Stock score refers to another clip")
        SOURCE.check(len(row["scores"]) in (0, 2) and len(row["scores"]) == len(row["variants"])
                     and all(stock.prior.sha256(digest) for digest in row["variants"])
                     and len(row["quality"]) == (len(row["scores"]) if truth else 0)
                     and all(value is None or math.isfinite(value) for value in row["quality"]), "Invalid stock witnesses")
        SOURCE.check(all(map(valid_grid, [row["mix"], *row["scores"]])), "Invalid stock response grid")
        window, _, flag = stock.selected(clip, "existing_short")
        controls = [(group, name) for name in stock.ORDINARY]
        yield ({"group": group, "kind": "background" if kind == "isolated" else kind, "truth": truth, "controls": controls},
               {"scores": {"mix": [row["mix"]]}, "qualities": {"mixture_phase": {"edit": row["quality"]}}},
               {"current": list(window) if flag else None, "speech_admitted": window is not None}, row["scores"])


def review(payload):
    records = payload["panels"]
    points = calibrate(records["ami"])
    SOURCE.check(records["ami"]["points"] is None and records["icsi"]["points"] == points,
                 "Held operating points do not match the frozen development data")
    prior_points = SOURCE.read(PARENT)["panels"]["icsi"]["points"]
    results = {panel: report(joined(panel, records[panel]), points, prior_points) for panel in PANELS}
    if "stock" in records:
        SOURCE.check(records["stock"]["points"] == points, "Stock operating points differ from AMI")
        results["stock"] = report(joined_stock(records["stock"]), points, prior_points)
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="phase", required=True)
    score = modes.add_parser("measure")
    score.add_argument("panel", choices=PANELS)
    for name in ("output", "views", "weights", "speech"):
        score.add_argument(name, type=Path)
    score.add_argument("--development", type=Path)
    transfer = modes.add_parser("measure-stock")
    for name in ("output", "native", "weights", "speech", "separator", "development"):
        transfer.add_argument(name, type=Path)
    combine = modes.add_parser("combine")
    for name in (*PANELS, "output"):
        combine.add_argument(name, type=Path)
    combine.add_argument("--stock", type=Path)
    modes.add_parser("review").add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.phase == "measure":
        measure(args)
    elif args.phase == "measure-stock":
        measure_stock(args)
    elif args.phase == "combine":
        payload = {"panels": {panel: SOURCE.read(getattr(args, panel)) for panel in PANELS}}
        if args.stock is not None:
            payload["panels"]["stock"] = SOURCE.read(args.stock)
        payload["results"] = review(payload)
        SOURCE.write(args.output, payload)
    else:
        payload = SOURCE.read(args.receipt)
        result = review(payload)
        SOURCE.check(result == payload["results"], "The saved results do not match the replay")
        print(f"Replayed {sum(len(panel['rows']) for panel in payload['panels'].values())} deliveries")
        for panel, methods in result.items():
            for method, counts in methods.items():
                print(panel, method, counts)


if __name__ == "__main__":
    main()
