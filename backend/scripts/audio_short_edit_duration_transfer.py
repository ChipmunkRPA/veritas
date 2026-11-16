# SPDX-License-Identifier: MIT
"""Compare finer and context-relative timing on frozen short speech-edit models."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import tarfile
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

from scripts import audio_short_edit_background_sine as previous

ROOM, BASE, SOURCE = previous.ROOM, previous.BASE, previous.SOURCE
ARM = previous.training.ORIGINAL
REFERENCE = previous.training.REFERENCE
EARLIER_SINE = previous.PANELS / "audio-short-edit-background-sine-2026-11-15.json.gz"
SCHEMA = "chipmunk.audio-short-edit-duration-transfer/1"
SHARDS = (8, 10, 11, 12)
TREATMENTS = ("dry", "competing")
WIDTHS = (25, 35, *BASE.LENGTHS)
HOP, FRAMES = BASE.HOP, BASE.FRAMES
METHODS = ("fixed", "denser", "width_calibrated", "surrounding_context")
ARCHIVE_ROOT = "chipmunk-short-edit-room.ePcmRA"


def contract():
    return {
        "schema": SCHEMA, "script_sha256": SOURCE.path_sha(Path(__file__)),
        "parents": {p.name: SOURCE.path_sha(p) for p in (REFERENCE, EARLIER_SINE, previous.SINE, previous.NARROW, previous.PREVIEW)},
        "shards": list(SHARDS), "max_fresh_readers": 32, "arm": ARM, "families": list(ROOM.FAMILIES),
        "methods": list(METHODS), "width_frames": list(WIDTHS), "hop_samples": HOP, "dense_step_frames": 5,
        "protocol": ("The same frozen model reads only the submitted waveform. Preserve the earlier fixed scan, then change "
                     "the start spacing alone. Separately compare width-normalized 250–1000 ms maxima and subtract the higher "
                     "mean of two neighboring speech intervals before width normalization. Context uses 50 ms guard gaps and "
                     "25–30 frames on either side. Use 16 matched genuine calibration reader maxima per condition and isolated "
                     "conversation as an additional floor. Freeze scores and thresholds before new held inference."),
    }


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def earlier():
    return previous.training.reference()


def prohibited_readers():
    excluded, rooms = previous.exposure()
    excluded.update(row["reader"] for row in read_gz(EARLIER_SINE)["held"]["plan"]["pairs"])
    return excluded, rooms


def fresh_sources(cache, backgrounds, run):
    excluded, rooms = prohibited_readers()
    if set(SHARDS) & set(previous.SHARDS):
        raise ValueError("The fresh source plan reuses a previous shard")
    listing = previous.sine.source_listing(cache)
    candidates, pins = {}, []
    for shard in SHARDS:
        rows, pin = previous.select_shard(cache, shard, listing[f"data/split-{shard}.tar.gz"], excluded)
        for reader, offer in rows.items():
            if reader not in candidates or offer[0] < candidates[reader][0]:
                candidates[reader] = offer
        pins.append(pin)
        print(f"Selected {len(rows)} additional reader candidates from shard {shard}", flush=True)
    SOURCE.check(len(candidates) >= 16, "Too few fresh declared SINE readers")
    run.mkdir(mode=0o700)
    (run / "sine").mkdir()
    (run / "background").mkdir()
    pairs = []
    for _, row, bodies in sorted(candidates.values(), key=lambda item: item[0])[:32]:
        i = len(pairs)
        pairs.append(row | {"id": i, "room": rooms[i % len(rooms)]["room"]})
        for label, body in bodies.items():
            (run / "sine" / f"{i:03}-{label}.wav").write_bytes(body)
    for row in rooms:
        name = row["room"] + ".pcm"
        raw = (backgrounds / name).read_bytes()
        SOURCE.check(SOURCE.digest(raw) == row["pcm_sha256"] and len(raw) == 2 * BASE.SAMPLES, "AMI controls differ")
        (run / "background" / name).write_bytes(raw)
    return {"pairs": pairs, "backgrounds": rooms, "shards": pins}


def validate_fresh(plan):
    excluded, rooms = prohibited_readers()
    pairs = plan["pairs"]
    readers = {row["reader"] for row in pairs}
    SOURCE.check(16 <= len(pairs) == len(readers) <= 32 and not readers & excluded
                 and plan["backgrounds"] == rooms and [item["shard"] for item in plan["shards"]] == list(SHARDS),
                 "Fresh roster, reader overlap or room commitments differ")
    for i, row in enumerate(pairs):
        start, end = row["truth"]
        total = row["files"]["variant"]["samples"]
        regions, _ = previous.sine.sine.label_clock(row["publisher_cut_paste_clock"], total, "cut_paste")
        a, b, _ = regions[1]
        limit = min(total, row["files"]["original"]["samples"]) - 144000
        clocks = [tick for tick in range(0, limit + 1, 2400) if tick + 6000 <= a < b <= tick + 138000]
        selected = min(clocks, key=lambda tick: previous.order("sine-background-training-clock/1", row["stem"], tick), default=None)
        SOURCE.check(row["id"] == i and row["role"] == "cut_paste" and row["shard"] in SHARDS
                     and row["reader"] == row["stem"].split("-", 1)[0] and row["room"] == rooms[i % len(rooms)]["room"]
                     and selected == row["native_start"] and selected is not None
                     and [start, end] == [(x - selected) * 2 // 3 for x in (a, b)]
                     and 4000 <= start < end <= 92000 and 3200 <= end-start <= 16000, "Fresh edit clock differs")


def row_rosters(parent, fresh):
    by_fold = {}
    archive_cases = list(previous.deliveries(fresh[0], fresh[1]))
    fresh_cases = [row for row, _ in archive_cases]
    for family in ROOM.FAMILIES:
        split = {}
        for role in ("calibration", "held"):
            split[role] = [row for row in ROOM.trial_cases(parent, role, family) if row["condition"] in TREATMENTS]
        by_fold[family] = split | {"fresh": fresh_cases}
    return by_fold, archive_cases


def archived_pcm(path, run, plan, trials, rosters):
    stored = run / "room"
    (stored / "pcm").mkdir(parents=True)
    (stored / "fit").mkdir()
    required = {"sealed/freeze.json": stored / "freeze.json"}
    for trial in trials:
        name = f'{trial["family"]}-{trial["arm"]}.safetensors'
        required["fit/" + name] = stored / "fit" / name
    for split in rosters.values():
        for role in ("calibration", "held"):
            for row in split[role]:
                name = row["pcm_sha256"] + ".pcm"
                required["sealed/pcm/" + name] = stored / "pcm" / name
    if path.is_dir():
        SOURCE.check((fit := path.parent / "fit").is_dir(), "Place the original fit directory alongside the sealed plan")
        for key, target in required.items():
            source = fit / target.name if key.startswith("fit/") else path / key.removeprefix("sealed/")
            shutil.copyfile(source, target)
        required.clear()
    else:
        with tarfile.open(path, "r|") as archive:
            for member in archive:
                key = member.name.removeprefix(ARCHIVE_ROOT + "/")
                if (target := required.pop(key, None)) is not None:
                    SOURCE.check(member.name == ARCHIVE_ROOT + "/" + key and member.isfile() and member.size <= 700000,
                                 "Unexpected model or PCM member")
                    with archive.extractfile(member) as stream:
                        target.write_bytes(stream.read())
                    if not required:
                        break
    SOURCE.check(not required and SOURCE.read(stored / "freeze.json") == plan, "Archived PCM or input plan is incomplete")


def freeze(args):
    parent = earlier()
    plan = parent["plan"]
    ROOM.validate(plan)
    trials = [previous.training.previous(parent, family, "fit") for family in ROOM.FAMILIES]
    fresh_dir = args.run / "fresh"
    args.run.mkdir(mode=0o700)
    fresh = fresh_sources(args.cache, args.background, fresh_dir)
    validate_fresh(fresh)
    rosters, audio = row_rosters(plan, (fresh_dir, fresh))
    archived_pcm(args.archive, args.run, plan, trials, rosters)
    (fresh_dir / "pcm").mkdir()
    for row, samples in audio:
        (fresh_dir / "pcm" / (row["pcm_sha256"] + ".pcm")).write_bytes(samples.tobytes())
    SOURCE.write(args.run / "freeze.json", {"contract": contract(), "fresh": fresh, "rosters": rosters,
                                            "weights": {trial["family"]: trial["weights_sha256"] for trial in trials}})
    print(f"Froze {len(fresh['pairs'])} newly selected reader pairs and both existing model folds", flush=True)


def choices(local, edges, spans):
    import numpy as np

    support = np.zeros(FRAMES, dtype=np.float64)
    ticks = np.arange(FRAMES) * HOP
    for start, end in spans:
        support += np.clip(np.minimum(ticks + HOP, end * BASE.RATE)
                           - np.maximum(ticks, start * BASE.RATE), 0, HOP)
    tolerance = 2e-9 * BASE.RATE
    cumulative = np.r_[0., np.cumsum(local, dtype=np.float64)]
    speech = np.r_[0., np.cumsum(support)]
    maximum = np.lib.stride_tricks.sliding_window_view(np.pad(edges, (4, 5), constant_values=-np.inf), 9).max(axis=1)
    dense, flanked = {}, {}
    for width in WIDTHS:
        starts = np.arange(0, FRAMES - width + 1, 5)
        ends = starts + width
        speech_ok = 2 * (speech[ends] - speech[starts]) >= width * HOP - tolerance
        values = (cumulative[ends] - cumulative[starts]) / width + .075 * (maximum[starts] + maximum[ends])
        dense[width] = pick(values, starts, ends, speech_ok)
        side = min(width, 30)
        left, right = starts - side - 5, ends + side + 5
        bounded = speech_ok & (left >= 0) & (right <= FRAMES)
        left, right = np.maximum(0, left), np.minimum(FRAMES, right)
        ls, rs = np.maximum(0, starts - 5), np.minimum(FRAMES, ends + 5)
        enough = (2 * (speech[ls] - speech[left]) >= side * HOP - tolerance) & (
            2 * (speech[right] - speech[rs]) >= side * HOP - tolerance)
        context = np.maximum(cumulative[ls] - cumulative[left], cumulative[right] - cumulative[rs]) / side
        flanked[width] = pick(values - context, starts, ends, bounded & enough)
    return {"fixed": BASE.best_interval(local, edges, spans),
            "dense": [dense[width] for width in WIDTHS], "context": [flanked[width] for width in WIDTHS]}


def pick(values, starts, ends, allowed):
    import numpy as np

    if not np.any(allowed):
        return None
    index = int(np.argmax(np.where(allowed, values, -np.inf)))
    return {"score": round(float(values[index]), 8), "range": [int(starts[index]) * HOP, int(ends[index]) * HOP]}


def model_readings(model, mean, scale, rows, extract):
    import torch

    cache, readings = {}, []
    with torch.inference_mode():
        for row in rows:
            digest = row["pcm_sha256"]
            if digest not in cache:
                signal, spans = extract(digest, True)
                values = torch.from_numpy(((signal - mean[:, None]) / scale[:, None])[None])
                local, edges = model(values)[0].numpy()
                cache[digest] = choices(local, edges, spans)
            readings.append(cache[digest])
    return readings


def quantile(values, q):
    values = sorted(values)
    return values[math.ceil(len(values) * q) - 1]


def maxima(pool, getter):
    groups = defaultdict(list)
    for case, reading in pool:
        if (result := getter(reading)) is not None:
            groups[case["group"]].append(result["score"])
    return [max(values) for values in groups.values()]


def strongest(items, scales=None):
    rows = []
    for i, item in enumerate(items):
        if item is None or (scales is not None and scales[i] is None):
            continue
        value = item["score"]
        if scales is not None:
            center, spread = scales[i]
            value = round((value - center) / spread, 8)
        rows.append({"score": value, "range": item["range"]})
    return max(rows, key=lambda row: (row["score"], *(-x for x in row["range"])), default=None)


def best(reading, method, scales):
    if method == "fixed":
        return reading["fixed"]
    if method == "denser":
        return strongest(reading["dense"][-len(BASE.LENGTHS):])
    return strongest(reading["dense" if method == "width_calibrated" else "context"], scales)


def fit_points(cases, scores):
    SOURCE.check(len(cases) == len(scores) and all(row["truth"] is None for row in cases), "Held data entered calibration")
    estimates = {}
    for condition in TREATMENTS:
        pairs = [(case, score) for case, score in zip(cases, scores, strict=True) if case["condition"] == condition]
        ordinary = [pair for pair in pairs if pair[0]["kind"] != "background"]
        background = [pair for pair in pairs if pair[0]["kind"] == "background"]
        for method in METHODS:
            scales = None
            if method in METHODS[2:]:
                lane = "dense" if method == "width_calibrated" else "context"
                scales = []
                for i in range(len(WIDTHS)):
                    controls = maxima(ordinary, lambda row, lane=lane, i=i: row[lane][i])
                    if len(controls) < 8:
                        scales.append(None)
                    else:
                        median = quantile(controls, .50)
                        scales.append([median, round(max(.20, quantile(controls, .90) - median), 8)])
            select = partial(best, method=method, scales=scales)
            values = maxima(ordinary, select)
            stems = maxima(background, select)
            SOURCE.check(len(values) >= 8, "Too few assessed ordinary calibration groups")
            floor = max(stems, default=min(values))
            estimates[f"{condition}/{method}"] = {"scales": scales, "strict": max(max(values), floor),
                                                 "wider": max(quantile(values, .90), floor),
                                                 "ordinary_groups": len(values), "isolated_groups": len(stems)}
    return estimates


def capture(args):
    frozen = SOURCE.read(args.run / "freeze.json")
    SOURCE.check(frozen["contract"] == contract(), "Frozen experiment differs")
    parent = earlier()
    activity = ROOM.runtime(args.speech)
    native = ROOM.feature_inputs(args.run / "room", activity)
    fresh = ROOM.feature_inputs(args.run / "fresh", activity)
    calibration_path = args.run / "calibration.json"
    if args.phase == "calibration":
        SOURCE.check(not calibration_path.exists(), "Calibration is already sealed")
    else:
        seal = SOURCE.read(calibration_path)
        SOURCE.check(seal["freeze_sha256"] == SOURCE.path_sha(args.run / "freeze.json"), "Calibration refers to another freeze")
    folders = {}
    for family, roles in frozen["rosters"].items():
        trial = previous.training.previous(parent, family, "fit")
        SOURCE.check(frozen["weights"][family] == trial["weights_sha256"], "Frozen weights differ")
        model, mean, scale = ROOM.load_trial(args.run / "room/fit", trial)
        if args.phase == "calibration":
            readings = model_readings(model, mean, scale, roles["calibration"], native)
            folders[family] = {"readings": readings, "points": fit_points(roles["calibration"], readings)}
        else:
            row = seal["folds"][family]
            SOURCE.check(row["points"] == fit_points(roles["calibration"], row["readings"]), "Operating points differ")
            folders[family] = {role: model_readings(model, mean, scale, roles[role], extractor)
                               for role, extractor in (("held", native), ("fresh", fresh))}
        print(f"{args.phase} readings completed for {family}", flush=True)
    freeze_digest = SOURCE.path_sha(args.run / "freeze.json")
    if args.phase == "calibration":
        SOURCE.write(calibration_path, {"freeze_sha256": freeze_digest, "folds": folders})
    else:
        output = {"freeze": frozen, "calibration": seal, "held": folders}
        report(output)
        args.output.write_bytes(gzip.compress((json.dumps(output, sort_keys=True, allow_nan=False) + "\n").encode(), mtime=0))


def compatible(selection, truth):
    if selection is None:
        return False
    a, b = selection
    c, d = truth
    common = max(0, min(b, d) - max(a, c))
    return 2 * common >= d - c and 10 * common >= 3 * (max(b, d) - min(a, c))


def summarize(cases, readings, point, method, threshold):
    counts = Counter(edited=0, useful=0, useful_clear=0, misplaced=0, unassessed_edits=0,
                     ordinary=0, ordinary_flags=0, unassessed_ordinary=0, isolated=0, isolated_flags=0, unassessed_isolated=0)
    controls = defaultdict(list)
    choices = []
    for case, raw in zip(cases, readings, strict=True):
        item = best(raw, method, point["scales"])
        chosen = item["range"] if item is not None and item["score"] > point[threshold] else None
        choices.append((case, chosen, item is None))
        if case["kind"] != "background" and case["truth"] is None and chosen is not None:
            controls[case["group"]].append(chosen)
    for case, selection, skipped in choices:
        truth = case["truth"]
        if truth is not None:
            useful = compatible(selection, truth)
            counts.update(edited=1, useful=int(useful), misplaced=int(selection is not None and not useful),
                          useful_clear=int(useful and not any(compatible(x, truth) for x in controls[case["group"]])),
                          unassessed_edits=int(skipped))
        elif case["kind"] == "background":
            counts.update(isolated=1, isolated_flags=int(selection is not None), unassessed_isolated=int(skipped))
        else:
            counts.update(ordinary=1, ordinary_flags=int(selection is not None), unassessed_ordinary=int(skipped))
    return dict(counts)


def report(record):
    frozen = record["freeze"]
    SOURCE.check(frozen["contract"] == contract() and record["calibration"]["freeze_sha256"] == ROOM.plan_hash(frozen),
                 "Receipt contract or calibration binding differs")
    validate_fresh(frozen["fresh"])
    primary = earlier()
    SOURCE.check(set(frozen["rosters"]) == set(ROOM.FAMILIES) == set(record["held"]), "Model fold roster differs")
    output = []
    for family, roster in frozen["rosters"].items():
        SOURCE.check(frozen["weights"][family] == previous.training.previous(primary, family, "fit")["weights_sha256"],
                     "Receipt uses a different model")
        for role in ("calibration", "held"):
            expected = [row for row in ROOM.trial_cases(primary["plan"], role, family) if row["condition"] in TREATMENTS]
            SOURCE.check(roster[role] == expected, "Room trial roster differs")
        previous.check_cases(frozen["fresh"], roster["fresh"])
        cal = record["calibration"]["folds"][family]
        SOURCE.check(cal["points"] == fit_points(roster["calibration"], cal["readings"]), "Calibrated cutoffs differ")
        for role, panels in (("held", ("phoneme", "stock")), ("fresh", ("sine",))):
            all_rows = list(zip(roster[role], record["held"][family][role], strict=True))
            for panel in panels:
                for condition in TREATMENTS:
                    pairs = [(case, raw) for case, raw in all_rows if (case["panel"], case["condition"]) == (panel, condition)]
                    rows, scores = zip(*pairs, strict=True)
                    for method in METHODS:
                        point = cal["points"][f"{condition}/{method}"]
                        for threshold in ("strict", "wider"):
                            result = summarize(rows, scores, point, method, threshold)
                            output.append(dict(fold=family, panel=panel, condition=condition, method=method,
                                               point=threshold, **result))
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="phase", required=True)
    lock = commands.add_parser("freeze")
    for option in ("run", "cache", "background", "archive"):
        lock.add_argument(option, type=Path)
    for phase in ("calibration", "held"):
        native = commands.add_parser(phase)
        native.add_argument("run", type=Path)
        native.add_argument("speech", type=Path)
        if phase == "held":
            native.add_argument("output", type=Path)
    review = commands.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.phase == "freeze":
        freeze(args)
    elif args.phase == "review":
        print(json.dumps(report(read_gz(args.receipt)), sort_keys=True, indent=2))
    else:
        capture(args)


if __name__ == "__main__":
    main()
