# SPDX-License-Identifier: MIT
"""Confirm background-supervised short-edit readings on new SINE authorship pairs."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts import audio_short_edit_background_training as training
from scripts import audio_sine_resynthesis_transfer as sine

SOURCE, BASE, ROOM = training.SOURCE, training.BASE, training.room
PANELS = ROOM.PHONEME.parent
SINE = PANELS / "audio-sine-resynthesis-transfer-2026-11-14.json"
NARROW = PANELS / "audio-narrow-flag-transfer-2026-11-14.json"
PREVIEW = PANELS / "audio-sine-preview-2026-11-13.json"
SHARDS = (1, 4, 5, 6)
MIN_READERS, MAX_READERS = 16, 32
SCHEMA = "chipmunk.audio-short-edit-background-sine/1"
METHOD = {
    "selection": "Take the first 192 MiB of shards 1, 4, 5 and 6 at the pinned SINE revision. Exclude the source IDs of all "
                 "earlier SINE preview and named parent study readers. Select one original/marked cut-and-paste pair per "
                 "additional reader, by hash, up to 32. Choose a six-second crop at an eligible 100 ms grid start by hash. "
                 "Keep edits from 0.20 through 1.00 s and 0.25 s of surrounding speech audio on each side.",
    "noise": "Use the same eight documented AMI excerpts from the earlier SINE transfer, not used for fitting any of these "
             "models. Mix each pair with its assigned shared background set 5 dB above the original by RMS. Keep the dry "
             "pair and all eight isolated room negatives. Keep all calibration points frozen from the earlier plan.",
}


def exposure():
    known, previews, narrow = SOURCE.read(SINE), SOURCE.read(PREVIEW), SOURCE.read(NARROW)
    readers = {str(value) for value in previews["source_reader_ids"]}
    for doc in (known, narrow):
        readers.update(row["reader"] for row in doc["freeze"]["pairs"])
        SOURCE.check(not set(SHARDS) & {row["shard"] for row in doc["freeze"]["shards"]}, "A SINE shard had prior project use")
    return readers, known["freeze"]["backgrounds"]


def contract():
    return {"schema": SCHEMA, "method": METHOD, "revision": sine.sine.REVISION, "prefix_bytes": sine.PREFIX,
            "parents": {path.name: SOURCE.path_sha(path) for path in (SINE, NARROW, PREVIEW, training.REFERENCE)}}


def order(tag, *values):
    return SOURCE.digest("/".join((tag, *map(str, values))).encode())


def select_shard(cache, shard, pin, excluded):
    members, labels, reading = sine.shard_members(cache, shard, pin)
    matched = {}
    for key in sorted(members["real"].keys() & members["cut_paste"].keys()):
        reader = key.split("-", 1)[0]
        if reader in excluded:
            continue
        bodies = {"original": members["real"][key], "variant": members["cut_paste"][key]}
        original, edited = (sine.wav(bodies[kind]) for kind in ("original", "variant"))
        clock, label = labels["medium_cut_paste.txt"][f"dev_cut_paste_medium-{key}"]
        SOURCE.check(label == "0", "Publisher segment label differs")
        clock_original, label_original = labels["medium_real.txt"][f"dev_real_medium-{key}"]
        sine.sine.label_clock(clock_original, original[0], "real")
        SOURCE.check(label_original == "1", "Publisher original label differs")
        regions, _ = sine.sine.label_clock(clock, edited[0], "cut_paste")
        a, b, _ = regions[1]
        limit = min(original[0], edited[0]) - 144000
        starts = [tick for tick in range(0, limit + 1, 2400) if tick + 6000 <= a < b <= tick + 138000]
        if not (4800 <= b - a <= 24000 and starts):
            continue
        priority = order("sine-background-training-source/1", key)
        if reader in matched and matched[reader][0] <= priority:
            continue
        start = min(starts, key=lambda tick: order("sine-background-training-clock/1", key, tick))
        row = {"reader": reader, "stem": key, "shard": shard, "native_start": start, "role": "cut_paste",
               "publisher_cut_paste_clock": clock, "truth": [(a-start)*2//3, (b-start)*2//3],
               "files": {name: {"wav_sha256": SOURCE.digest(bodies[name]), "samples": info[0], "pcm_sha256": info[1]}
                         for name, info in zip(("original", "variant"), (original, edited), strict=True)}}
        matched[reader] = (priority, row, bodies)
    return matched, reading


def freeze(args):
    listed = sine.source_listing(args.cache)
    excluded, backgrounds = exposure()
    all_pairs, pins = {}, []
    for shard in SHARDS:
        pairs, pin = select_shard(args.cache, shard, listed[f"data/split-{shard}.tar.gz"], excluded)
        print(f"SINE shard {shard}: {len(pairs)} eligible additional readers", flush=True)
        for reader, offer in pairs.items():
            if reader not in all_pairs or offer[0] < all_pairs[reader][0]:
                all_pairs[reader] = offer
        pins.append(pin)
    args.output.mkdir(mode=0o700)
    (args.output / "sine").mkdir()
    (args.output / "background").mkdir()
    selected = []
    for index, (_, row, bodies) in enumerate(sorted(all_pairs.values(), key=lambda pair: pair[0])[:MAX_READERS]):
        for name, body in bodies.items():
            (args.output / "sine" / f"{index:03}-{name}.wav").write_bytes(body)
        selected.append(row | {"id": index, "room": backgrounds[index % len(backgrounds)]["room"]})
    for row in backgrounds:
        name = row["room"] + ".pcm"
        raw = (args.backgrounds / name).read_bytes()
        SOURCE.check(len(raw) == 2 * BASE.SAMPLES and SOURCE.digest(raw) == row["pcm_sha256"], "Known AMI input changed")
        (args.output / "background" / name).write_bytes(raw)
    plan = {"contract": contract(), "shards": pins, "backgrounds": backgrounds, "pairs": selected}
    check_plan(plan)
    SOURCE.write(args.output / "freeze.json", plan)
    print(f"Sealed {len(selected)} reader-disjoint SINE pairs before running any held detector", flush=True)


def check_plan(plan):
    excluded, backgrounds = exposure()
    SOURCE.check(plan["contract"] == contract() and plan["backgrounds"] == backgrounds
                 and [row["shard"] for row in plan["shards"]] == list(SHARDS), "SINE source plan differs")
    previous = training.reference()["plan"]
    SOURCE.check(not {row["room"] for row in backgrounds} & {group["capture"] for group in previous["groups"]},
                 "Fresh foreground panel reuses a training background room")
    pairs = plan["pairs"]
    SOURCE.check(MIN_READERS <= len(pairs) == len({row["reader"] for row in pairs}) <= MAX_READERS
                 and not excluded.intersection(row["reader"] for row in pairs), "The publisher reader roster overlaps earlier use")
    for index, row in enumerate(pairs):
        a, b = row["truth"]
        original, edited = (row["files"][kind]["samples"] for kind in ("original", "variant"))
        intervals, _ = sine.sine.label_clock(row["publisher_cut_paste_clock"], edited, "cut_paste")
        left, right, _ = intervals[1]
        eligible = [tick for tick in range(0, min(original, edited) - 144000 + 1, 2400)
                    if tick + 6000 <= left < right <= tick + 138000]
        start = min(eligible, key=lambda tick: order("sine-background-training-clock/1", row["stem"], tick), default=None)
        SOURCE.check(row["id"] == index and row["role"] == "cut_paste" and row["shard"] in SHARDS
                     and row["reader"] == row["stem"].split("-", 1)[0]
                     and row["room"] == backgrounds[index % len(backgrounds)]["room"]
                     and start is not None and start == row["native_start"]
                     and (a, b) == ((left-start)*2//3, (right-start)*2//3)
                     and 4000 <= a < b <= 92000 and 3200 <= b-a <= 16000, "The sealed edit clock differs")


def deliveries(inputs, plan):
    for info, samples in sine.native_rows(inputs, plan):
        group, room_label, side, truth = (info[key] for key in ("id", "condition", "side", "truth"))
        isolated = info["role"] == "background_only"
        condition = "dry" if room_label == "clean" else "competing"
        kind = "background" if isolated else "edited/sine" if side == "variant" else "untouched"
        yield {"panel": "sine", "condition": condition, "group": group, "kind": kind,
               "truth": truth, "pcm_sha256": SOURCE.digest(samples.tobytes())}, samples


def prepare(inputs, output):
    plan = SOURCE.read(inputs / "freeze.json")
    check_plan(plan)
    output.mkdir(mode=0o700)
    waves = output / "pcm"
    waves.mkdir()
    cases = []
    for row, samples in deliveries(inputs, plan):
        path = waves / (row["pcm_sha256"] + ".pcm")
        if not path.exists():
            path.write_bytes(samples.tobytes())
        cases.append(row)
    check_cases(plan, cases)
    SOURCE.write(output / "cases.json", cases)


def check_cases(plan, cases):
    expected = [(row["id"], treatment, kind, truth) for row in plan["pairs"] for treatment in ("dry", "competing")
                for kind, truth in (("untouched", None), ("edited/sine", row["truth"]))]
    expected.extend((idx, "competing", "background", None) for idx, _ in enumerate(plan["backgrounds"]))
    keys = [(row["group"], row["condition"], row["kind"], row["truth"]) for row in cases]
    SOURCE.check(keys == expected and len({row["pcm_sha256"] for row in cases}) == len(cases), "Fresh source rows differ")
    main = training.reference()["plan"]
    prior = {value for item in main["groups"] for view in item["observations"].values() for value in view.values()}
    prior.update(row["pcm_sha256"] for row in ROOM.transfer_cases())
    SOURCE.check(not any(row["pcm_sha256"] in prior for row in cases), "Fresh scoring reuses exact previous PCM")


def measure(args):
    known = training.reference()
    prepare(args.inputs, args.output)
    plan = SOURCE.read(args.inputs / "freeze.json")
    cases = SOURCE.read(args.output / "cases.json")
    check_cases(plan, cases)
    trained = SOURCE.read(args.fit / "fit.json")
    training.check_fit(trained, known)
    extract = ROOM.feature_inputs(args.output, ROOM.runtime(args.speech))
    readings = []
    for family in ROOM.FAMILIES:
        initial = training.previous(known, family, "fit")
        trials = [(args.initial, initial), *((args.fit, row) for row in trained["trials"] if row["family"] == family)]
        for directory, trial in trials:
            model, mean, scale = ROOM.load_trial(directory, trial)
            scores = ROOM.score_rows(model, mean, scale, cases, training.ORIGINAL, extract)
            readings.append({"family": family, "arm": trial["arm"], "weights_sha256": trial["weights_sha256"], "rows": scores})
            print(f"SINE transfer measured {len(cases)} files for {family} / {trial['arm']}", flush=True)
    SOURCE.write(args.output / "readings.json", {"schema": SCHEMA, "plan": plan, "cases": cases,
                                                "fit_sha256": ROOM.plan_hash(trained), "readings": readings})


def current(args):
    import torch

    from app.detectors.audio import short_voice_review
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    plan = SOURCE.read(args.inputs / "freeze.json")
    check_plan(plan)
    cases = SOURCE.read(args.data / "cases.json")
    check_cases(plan, cases)
    model = load_voice({"weights": args.weights, "assets": ASSETS})
    activity = ROOM.runtime(args.weights / "silero_vad.jit")
    torch.set_num_threads(2)

    class OfflineDeadline:
        def check(self, _stage):
            pass

    outcomes = []
    for case in cases:
        clip = BASE.pcm(args.data, case)
        raw = short_voice_review.scan(clip, activity.spans(clip), model, OfflineDeadline(), codec="pcm_s16le")
        best = raw["selected"]
        outcomes.append(case | {"window": [best["start_sample"], best["end_sample"]] if best else None,
                                "speech_counts": raw["speech_counts"], "responses": raw["responses"]})
        if len(outcomes) % 12 == 0 or len(outcomes) == len(cases):
            print(f"Current short speech scan measured {len(outcomes)}/{len(cases)} files", flush=True)
    SOURCE.write(args.output, {"plan_sha256": ROOM.plan_hash(plan), "assets": ASSETS, "rows": outcomes})


def report(study, lock, app):
    plan, cases = study["plan"], study["cases"]
    check_plan(plan)
    check_cases(plan, cases)
    SOURCE.check(study["schema"] == SCHEMA and study["fit_sha256"] == ROOM.plan_hash(lock)
                 and app["plan_sha256"] == ROOM.plan_hash(plan), "App or held capture differs from the sealed study")
    known = training.reference()
    training.check_fit(lock, known)
    app_locations, app_coverage = [], []
    grid = [(a * BASE.HOP, (a + length) * BASE.HOP) for length in BASE.LENGTHS
            for a in range(0, BASE.FRAMES - length + 1, BASE.STRIDE)]
    useful = ROOM.two_voice.SCORING._useful
    geometry = {state: {"edited": 0, "admitted_two_second_window_can_match": 0,
                        "short_grid_can_match": 0} for state in ("dry", "competing")}
    for expected, row in zip(cases, app["rows"], strict=True):
        SOURCE.check(all(row[key] == value for key, value in expected.items()), "Current app source differs")
        location, _, assessed = ROOM.two_voice.app_reading(row)
        app_locations.append(location)
        app_coverage.append(True if assessed else None)
        if truth := row["truth"]:
            counts = geometry[row["condition"]]
            counts["edited"] += 1
            counts["short_grid_can_match"] += any(useful(interval, truth) for interval in grid)
            counts["admitted_two_second_window_can_match"] += any(
                useful((entry["start_sample"], entry["end_sample"]), truth) for entry in row["responses"])
    original = [training.previous(known, family, "fit") for family in ROOM.FAMILIES]
    fits = [(family, arm, next(entry for entry in (*original, *lock["trials"])
                               if entry["family"] == family and entry["arm"] == arm))
            for family in ROOM.FAMILIES for arm in (training.ORIGINAL, *training.ARMS)]
    results = {"current": ROOM.summarize(cases, app_coverage, app_locations), "geometry": geometry,
               "ranks": {}, "points": {}, "fallback": {}}
    for (family, arm, fitted), row in zip(fits, study["readings"], strict=True):
        SOURCE.check((row["family"], row["arm"], row["weights_sha256"]) == (family, arm, fitted["weights_sha256"]),
                     "Fresh inference used a different checkpoint")
        label = f"{family}/{arm}"
        raw = row["rows"]
        results["ranks"][label] = training.matched_rank(cases, raw)
        for setting, point in fitted["points"].items():
            selected, results["points"][f"{label}/{setting}"] = training.evaluate(cases, raw, point)
            coverage = [old or read["best"] for old, read in zip(app_coverage, raw, strict=True)]
            fallback = [old if old is not None else new for old, new in zip(app_locations, selected, strict=True)]
            results["fallback"][f"{label}/{setting}"] = ROOM.summarize(cases, coverage, fallback)
    return results


def combine(args):
    fit = SOURCE.read(args.fit)
    held = SOURCE.read(args.held)
    current = SOURCE.read(args.current)
    packet = {"fit": fit, "held": held, "current": current, "result": report(held, fit, current)}
    with args.output.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(packet, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def review(args):
    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        packet = json.load(stream)
    summary = report(packet["held"], packet["fit"], packet["current"])
    SOURCE.check(summary == packet["result"], "Fresh SINE results differ from replay")
    print(f"Replayed all {len(packet['held']['cases'])} source files and {len(summary['points'])} candidate settings")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    for action, fields in (("freeze", ("cache", "backgrounds", "output")), ("measure", ("inputs", "initial", "fit", "speech", "output")),
                           ("current", ("inputs", "data", "weights", "output")), ("combine", ("fit", "held", "current", "output")),
                           ("review", ("receipt",))):
        command = commands.add_parser(action)
        for field in fields:
            command.add_argument(field, type=Path)
    args = cli.parse_args()
    {"freeze": freeze, "measure": measure, "current": current, "combine": combine, "review": review}[args.action](args)


if __name__ == "__main__":
    main()
