# SPDX-License-Identifier: MIT
"""Test a different voice checkpoint under competing speech."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_nested_voice_window_transfer as parent

SOURCE = parent.SOURCE
RATE = parent.RATE
STEP = RATE // 2
WIDTHS = (RATE, 2 * RATE)
GRIDS = {width: tuple(range(0, parent.SAMPLES - width + 1, STEP)) for width in WIDTHS}
ARMS = ("one_second", "both_contexts")
REVISION = "f7050b586236dc910d1157f430def2d0647b02b4"
MODEL = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
ASSETS = {
    "config.json": "bc86d455d78aa7c256844038e342e2beea9399bf0efc48e1748fdd918ddf3934",
    "preprocessor_config.json": "a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335",
    "model.safetensors": "8bff7244579a5c0f22f9994848c7eee2e038a07358bfebfbe75b38bd11d39b90",
}
SPEECH_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
SCHEMA = "chipmunk.audio-second-classifier-overlap/1"
METHOD = {
    "inputs": "Replay the native AMI development and ICSI transfer rosters. ICSI shares the foreground readers, "
              "edited words and generators with AMI, and uses different meeting captures.",
    "scores": "Measure native raw fake-vs-real log-odds on one- and two-second windows with half-second hops. "
              "Use the same speech activity model with at least half speech per window. Compare the one-second "
              "score alone and the smaller of that score and its centered two-second response, clamping the "
              "context at the file edges. Rank by largest value with earliest ties.",
    "calibrate": "For each arm take the larger of the second-highest AMI ordinary foreground group maximum "
                 "and the highest isolated AMI background maximum. Use a strict greater-than comparison and "
                 "freeze both points before ICSI inference. Select at most one range per arm.",
    "evaluation": "Count candidate-only and any-range union with the current two-second check. Count useful "
                  "ranges at 50% edit coverage and at least 0.30 intersection over union. Separately report "
                  "unassessed files, new ordinary flags and new candidate locations that miss the edit.",
}


def _parent(panel):
    path = parent.PANELS / f"audio-nested-voice-window-{panel}-2026-11-14.json"
    return SOURCE.read(path), SOURCE.path_sha(path)


def _lock(panel):
    _, digest = _parent(panel)
    return {"schema": SCHEMA, "panel": panel, "parent_sha256": digest, "model": MODEL,
            "revision": REVISION, "assets": ASSETS, "speech_sha256": SPEECH_SHA256, "method": METHOD}


def prepare(args):
    plan, bank = parent._materialize(args)
    panel = "icsi" if args.icsi else "ami"
    previous, _ = _parent(panel)
    SOURCE.check(plan == previous["freeze"]["plan"], "Prepared PCM roster differs from the published plan")
    args.output.mkdir(mode=0o700)
    for digest, pcm in bank.items():
        (args.output / f"{digest}.pcm").write_bytes(pcm)
    print(f"{panel}: verified {len(plan['clips'])} deliveries, {len(bank)} distinct PCM inputs", flush=True)


def _scores(values):
    one, two = values
    for index, value in enumerate(one):
        if value is None:
            continue
        start = GRIDS[RATE][index]
        yield "one_second", start, value
        context = two[max(0, min(index - 1, len(two) - 1))]
        if context is not None:
            yield "both_contexts", start, min(value, context)


def _best(values):
    best = {}
    for arm, start, score in _scores(values):
        proposal = (score, -start)
        if arm not in best or proposal > best[arm]:
            best[arm] = proposal
    return best


def _joined(record, panel):
    source, _ = _parent(panel)
    SOURCE.check(record["freeze"] == _lock(panel), "Evaluation method or source pin differs")
    for clip, serving, measured in zip(source["freeze"]["plan"]["clips"], source["measurements"],
                                       record["measurements"], strict=True):
        SOURCE.check(clip["pcm_sha256"] == serving["pcm_sha256"] == measured["pcm_sha256"],
                     "Score assigned to the wrong source")
        values = measured["log_odds"]
        SOURCE.check(tuple(map(len, values)) == tuple(len(GRIDS[w]) for w in WIDTHS), "Window grid differs")
        SOURCE.check(all(value is None or isinstance(value, (float, int)) and math.isfinite(value)
                         for grid in values for value in grid), "Invalid response")
        yield clip, serving["current"], values


def calibrate(record):
    ordinary, background = defaultdict(dict), defaultdict(list)
    for clip, _, values in _joined(record, "ami"):
        if clip["truth"] is not None:
            continue
        for arm, (score, _) in _best(values).items():
            if clip["kind"] == "background":
                background[arm].append(score)
            else:
                group = clip["group"]
                ordinary[arm][group] = max(score, ordinary[arm].get(group, -float("inf")))
    return {arm: max(max(background[arm], default=-float("inf")),
                     sorted(ordinary[arm].values(), reverse=True)[1]) for arm in ARMS}


def _useful(region, truth):
    return bool(region and parent.earlier.stock.useful(region, truth))


def _selection(response, point):
    if response is None or response[0] <= point:
        return None
    start = -response[1]
    return [start, start + RATE]


def summarize(record, panel, points):
    metrics = defaultdict(Counter)
    groups = defaultdict(set)
    selections = {}
    recovered = []
    for clip, current, values in _joined(record, panel):
        kind = clip["kind"].split("/", 1)[0]
        best = _best(values)
        for arm in ARMS:
            response = best.get(arm)
            selected = _selection(response, points[arm])
            condition = clip["group"], clip["condition_db"], arm
            selections[condition, clip["kind"]] = selected
            key = clip["condition_db"], kind, arm
            row = metrics[key]
            row["files"] += 1
            row["unassessed"] += response is None
            row["current_flags"] += current is not None
            row["candidate_flags"] += selected is not None
            row["combined_flags"] += current is not None or selected is not None
            if selected is not None and current is None:
                groups[key, "additional_flag_groups"].add(clip["group"])
            truth = clip["truth"]
            if truth is not None:
                prior, candidate = _useful(current, truth), _useful(selected, truth)
                row["current_useful"] += prior
                row["candidate_useful"] += candidate
                row["combined_useful"] += prior or candidate
                row["new_useful_with_no_prior_alert"] += candidate and current is None
                row["wrong_candidate_location"] += selected is not None and not candidate
                target_scores = [score for method, start, score in _scores(values)
                                 if method == arm and _useful([start, start + RATE], truth)]
                row["useful_window_measured"] += bool(target_scores)
                row["useful_window_above_point"] += bool(target_scores and max(target_scores) > points[arm])
                if candidate and not prior:
                    groups[key, "additional_useful_groups"].add(clip["group"])
                    recovered.append((key, condition, clip["kind"], selected))
    for (key, metric), readers in groups.items():
        metrics[key][metric] = len(readers)
    for key, condition, kind, selected in recovered:
        # Diagnostic only. The classifier never sees the paired controls at inference time.
        for comparison in ("untouched", kind.replace("edited/", "ordinary/"), "background"):
            if (condition, comparison) in selections:
                label = comparison.split("/", 1)[0]
                metrics[key][f"recovered_with_{label}_control"] += 1
                metrics[key][f"recovered_at_{label}_control_location"] += selections[condition, comparison] == selected
    return [dict(condition_db=db, kind=kind, arm=arm, **counts)
            for (db, kind, arm), counts in sorted(metrics.items())]


def load_classifier(weights, speech_path):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free

    for name, digest in ASSETS.items():
        SOURCE.check(SOURCE.path_sha(weights / name) == digest, f"Unexpected model file: {name}")
    SOURCE.check(SOURCE.path_sha(speech_path) == SPEECH_SHA256, "Unexpected speech activity weights")
    network_free()
    torch.set_num_threads(2)
    feature = AutoFeatureExtractor.from_pretrained(weights, local_files_only=True)
    model = AutoModelForAudioClassification.from_pretrained(weights, local_files_only=True).eval()
    SOURCE.check(set(model.config.id2label.values()) == {"fake", "real"}, "Unexpected model output labels")
    return model, feature, SpeechActivity(speech_path)


def window_log_odds(model, feature, pcm, spans, width):
    import torch

    from app.detectors.audio.content import window_in_speech

    fake, real = (model.config.label2id[label] for label in ("fake", "real"))
    grid = GRIDS[width]
    good = [(i, pcm[a:a + width]) for i, a in enumerate(grid)
            if window_in_speech(spans, a / RATE, (a + width) / RATE, fraction=.5)]
    scores = [None] * len(grid)
    with torch.inference_mode():
        for begin in range(0, len(good), 4):
            batch = good[begin:begin + 4]
            values = feature([window for _, window in batch], sampling_rate=RATE, return_tensors="pt")
            logits = model(**values).logits.float()
            for (index, _), score in zip(batch, (logits[:, fake] - logits[:, real]).tolist(), strict=True):
                scores[index] = round(score, 7)
    return scores


def measure(args):
    import numpy as np

    points = calibrate(SOURCE.read(args.development)) if args.panel == "icsi" else None
    source, _ = _parent(args.panel)
    clips = source["freeze"]["plan"]["clips"]
    args.output.mkdir(mode=0o700)
    freeze = _lock(args.panel)
    SOURCE.write(args.output / "freeze.json", {"method": freeze, "points": points})
    model, feature, speech = load_classifier(args.weights, args.speech)
    measured, cache = [], {}
    for clip in clips:
        digest = clip["pcm_sha256"]
        if digest not in cache:
            raw = (args.pcm / f"{digest}.pcm").read_bytes()
            SOURCE.check(SOURCE.digest(raw) == digest and len(raw) == 2 * parent.SAMPLES, "Unexpected PCM")
            pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            spans = speech.spans(pcm)
            measured_widths = [window_log_odds(model, feature, pcm, spans, width) for width in WIDTHS]
            cache[digest] = {"pcm_sha256": digest, "log_odds": measured_widths}
        measured.append(cache[digest])
        if len(measured) % 8 == 0:
            print(f"{args.panel}: {len(measured)}/{len(clips)}", flush=True)
    SOURCE.write(args.output / "measured.json", {"freeze": freeze, "points": points,
                                                  "device": "cpu", "measurements": measured})


def replay(ami, icsi):
    points = calibrate(ami)
    SOURCE.check(ami["points"] is None and icsi["points"] == points, "Frozen holdout decision points differ")
    return {name: summarize(record, name, points) for name, record in (("ami", ami), ("icsi", icsi))}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    prep = modes.add_parser("prepare")
    prep.add_argument("output", type=Path)
    for option in ("melo", "aligned", "earlier"):
        prep.add_argument(f"--{option}", type=Path, required=True)
    prep.add_argument("--history", type=Path, default=parent.ROOT / "backend/data/independent-evaluation")
    prep.add_argument("--icsi", type=Path)
    score = modes.add_parser("measure")
    score.add_argument("panel", choices=("ami", "icsi"))
    for field in ("output", "pcm", "weights", "speech"):
        score.add_argument(field, type=Path)
    score.add_argument("--development", type=Path)
    combine = modes.add_parser("combine")
    for field in ("ami", "icsi", "output"):
        combine.add_argument(field, type=Path)
    review = modes.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "measure":
        if (args.panel == "icsi") != (args.development is not None):
            cli.error("Provide the frozen AMI measurements only when measuring ICSI")
        measure(args)
    elif args.mode == "combine":
        ami, icsi = (SOURCE.read(path) for path in (args.ami, args.icsi))
        SOURCE.write(args.output, {"panels": {"ami": ami, "icsi": icsi}, "results": replay(ami, icsi)})
    else:
        record = SOURCE.read(args.receipt)
        result = replay(**record["panels"])
        SOURCE.check(result == record["results"], "Recorded counts do not reproduce")
        for name, panel in result.items():
            for row in panel:
                print(name, row)


if __name__ == "__main__":
    main()
