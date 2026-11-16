# SPDX-License-Identifier: MIT
"""Check whether flagged speech windows survive small changes to the model input."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from app.evidence.audio_short_review_binding import GRID, RATE, REVIEW_POINT
from scripts import audio_short_edit_union_holdout as union

SOURCE, BASE, ROOM = union.SOURCE, union.BASE, union.ROOM
duration = union.duration
PANEL = duration.previous.PANELS / "audio-short-edit-union-holdout-2026-11-15.json.gz"
SCHEMA = "chipmunk.audio-short-edit-input-views/1"
SHARDS = (22, 23, 24, 26, 27, 28)
SHIFT = 1280
SHELVES = {"low": 650, "high": 3200}
ATTENUATION_DB = 4
METHODS = ("current", "band_confirmed", "time_confirmed", "either_confirmed")
PANELS = ("diagnostic", "phoneme", "stock", "fresh")
MIN_READERS, MAX_READERS = 16, 32


def contract(prior):
    return {"schema": SCHEMA, "code_sha256": SOURCE.path_sha(Path(__file__)),
            "prior_sha256": SOURCE.path_sha(PANEL), "room_sha256": SOURCE.path_sha(duration.REFERENCE),
            "shards": list(SHARDS), "current_assets": prior["frozen"]["protocol"]["current_assets"],
            "point": REVIEW_POINT, "shift_samples": SHIFT, "shelf_hz": SHELVES,
            "attenuation_db": ATTENUATION_DB, "max_readers": MAX_READERS,
            "method": ("Keep the existing two-second PCM window and operating point. Re-read an already flagged window "
                       "after two separate mild whole-clip shelves, or at two starts shifted by 80 ms with exact source "
                       "samples and the same speech admission. The output location never moves. Re-use the existing score "
                       "point in every view and count support only above that point. Compare both agreement checks and "
                       "either check as a confirmation rule. No tuning, model training or upload-time background reference.")}


def archives(prior):
    # Retain genuine splices and every labeled generator on the known room panel.
    diagnostic = prior["frozen"]["cases"]
    existing = duration.earlier()["current"]["rows"]
    panels = {"diagnostic": diagnostic}
    for name in ("phoneme", "stock"):
        panels[name] = [
            {key: row[key] for key in ("panel", "group", "speaker", "condition", "kind", "truth", "pcm_sha256")}
            for row in existing if row["panel"] == name and row["kind"] in (
                "untouched", "genuine_splice", "background", *(f"edited/{x}" for x in (*ROOM.FAMILIES, *ROOM.STOCK_FAMILIES)))
        ]
    return panels


def excluded_readers(prior):
    return union.known_readers(union.parent()) | {row["reader"] for row in prior["frozen"]["pairs"]}


def new_inputs(run, cache, background, prior):
    imports = union.SINE.source_listing(cache)
    known = excluded_readers(prior)
    offers, pins = {}, []
    for shard in SHARDS:
        rows, pin = duration.previous.select_shard(cache, shard, imports[f"data/split-{shard}.tar.gz"], known)
        for reader, candidate in rows.items():
            if reader not in offers or candidate[0] < offers[reader][0]:
                offers[reader] = candidate
        pins.append(pin)
        print(f"Shard {shard}: {len(rows)} eligible new readers", flush=True)
    SOURCE.check(len(offers) >= MIN_READERS, "Too few reader-disjoint source pairs")
    _, rooms = duration.previous.exposure()
    for name in ("sine", "background", "pcm"):
        (run / name).mkdir()
    pairs = []
    for _, row, audio in sorted(offers.values(), key=lambda item: item[0])[:MAX_READERS]:
        i = len(pairs)
        paired = row | {"id": i, "room": rooms[i % len(rooms)]["room"]}
        pairs.append(paired)
        for role, raw in audio.items():
            (run / "sine" / f"{i:03}-{role}.wav").write_bytes(raw)
    for room in rooms:
        name = room["room"] + ".pcm"
        raw = (background / name).read_bytes()
        SOURCE.check(SOURCE.digest(raw) == room["pcm_sha256"], "Original room waveform differs")
        (run / "background" / name).write_bytes(raw)
    spine = {"pairs": pairs, "backgrounds": rooms, "shards": pins}
    cases = []
    for case, values in duration.previous.deliveries(run, spine):
        if case["condition"] == "competing":
            (run / "pcm" / (case["pcm_sha256"] + ".pcm")).write_bytes(values.tobytes())
            cases.append(case)
    return spine, cases


def validate(plan, prior):
    SOURCE.check(plan["protocol"] == contract(prior), "Inputs do not match the frozen protocol")
    SOURCE.check({key: plan["cases"][key] for key in PANELS[:-1]} == archives(prior), "Retained rosters differ")
    fresh = plan["fresh_source"]
    _, rooms = duration.previous.exposure()
    pairs = fresh["pairs"]
    readers = {row["reader"] for row in pairs}
    SOURCE.check(fresh["backgrounds"] == rooms and [row["shard"] for row in fresh["shards"]] == list(SHARDS)
                 and MIN_READERS <= len(pairs) == len(readers) <= MAX_READERS and not readers & excluded_readers(prior),
                 "Fresh SINE source roles differ")
    cases = plan["cases"]["fresh"]
    expected = [(row["id"], kind, truth) for row in pairs for kind, truth in
                (("untouched", None), ("edited/sine", row["truth"]))]
    expected.extend((i, "background", None) for i in range(len(rooms)))
    SOURCE.check(expected == [(case["group"], case["kind"], case["truth"]) for case in cases]
                 and len({case["pcm_sha256"] for case in cases}) == len(cases)
                 and all(case["condition"] == "competing" and case["panel"] == "sine" for case in cases),
                 "Fresh case file or truth differs")
    for i, row in enumerate(pairs):
        interval, _ = union.SINE.sine.label_clock(row["publisher_cut_paste_clock"], row["files"]["variant"]["samples"], "cut_paste")
        left, right, _ = interval[1]
        crop_start = row["native_start"]
        last = min(row["files"][key]["samples"] for key in ("variant", "original"))
        SOURCE.check(row["id"] == i and row["room"] == rooms[i % len(rooms)]["room"]
                     and row["reader"] == row["stem"].split("-", 1)[0] and row["shard"] in SHARDS
                     and crop_start % 2400 == 0 and 0 <= crop_start <= last - 144000
                     and crop_start + 6000 <= left < right <= crop_start + 138000
                     and row["truth"] == [(x-crop_start)*2//3 for x in (left, right)], "Fresh native edit clock differs")


def freeze(args):
    prior = duration.read_gz(PANEL)
    args.run.mkdir(mode=0o700)
    SOURCE.write(args.run / "protocol.json", contract(prior))
    fresh, cases = new_inputs(args.run, args.cache, args.background, prior)
    plan = {"protocol": contract(prior), "fresh_source": fresh, "cases": archives(prior) | {"fresh": cases}}
    validate(plan, prior)
    SOURCE.write(args.run / "freeze.json", plan)
    print(f"Sealed {len(fresh['pairs'])} fresh reader pairs before running the transformed checks", flush=True)


def shelves(signal):
    import numpy as np

    padding = SHIFT
    padded = np.pad(signal.astype(np.float64), padding, mode="reflect")
    frequency = np.fft.rfftfreq(len(padded), 1 / RATE)
    spectrum = np.fft.rfft(padded)
    loss = 10**(-ATTENUATION_DB/20) - 1
    for name, cutoff in SHELVES.items():
        ratio = (frequency / cutoff)**4
        taper = (1 / (1 + ratio)) if name == "low" else ratio / (1 + ratio)
        output = np.fft.irfft(spectrum * (1 + loss * taper), n=len(padded))[padding:-padding]
        peak = float(np.max(np.abs(output)))
        if peak > 32767 / 32768:
            output *= (32767 / 32768) / peak
        yield name, output.astype(np.float32)


def adjusted_spans(start):
    # At the file edge the clipped direction coincides with the original window.
    return sorted({max(0, start - SHIFT), min(GRID[-1][0], start + SHIFT)} - {start})


def admitted(spans, start, end):
    return sum(max(0, min(end, round(b * RATE)) - max(start, round(a * RATE))) for a, b in spans) >= RATE


def read_views(pcm, spans, model, base):
    from app.detectors.audio.calibration import logit

    start, end = base
    input_views = {}
    for name, filtered in shelves(pcm):
        input_views[name] = logit(model.predict([filtered[start:end]])[0])
    shifted = []
    for location in adjusted_spans(start):
        end = location + 2 * RATE
        score = logit(model.predict([pcm[location:end]])[0]) if admitted(spans, location, end) else None
        shifted.append({"range": [location, end], "score": score})
    return {"band": input_views, "shifted": shifted}


def parent_readings(prior):
    existing = duration.earlier()["current"]["rows"]
    return {
        "diagnostic": {row["pcm_sha256"]: row["reading"] for row in prior["current"]["rows"]},
        **{panel: {row["pcm_sha256"]: {"range": row["window"], "assessed": bool(row["responses"])}
                   for row in existing if row["panel"] == panel} for panel in ("phoneme", "stock")},
    }


def capture(args):
    from app.detectors.audio import short_voice_review
    from scripts.run_audio_specialists_v15 import load_voice

    class Deadline:
        def check(self, stage):
            pass

    prior = duration.read_gz(PANEL)
    plan = SOURCE.read(args.run / "freeze.json")
    validate(plan, prior)
    model = load_voice({"weights": args.weights, "assets": plan["protocol"]["current_assets"]})
    activity = ROOM.runtime(args.weights / "silero_vad.jit")
    parent_bases = parent_readings(prior)
    results = {}
    for panel in args.panels:
        rows = plan["cases"][panel]
        root = args.run if panel == "fresh" else args.diagnostic if panel == "diagnostic" else args.room
        readings = []
        for case in rows:
            digest = case["pcm_sha256"]
            raw = parent_bases.get(panel, {}).get(digest)
            signal = spans = None
            if raw is None or raw["range"] is not None:
                signal = BASE.pcm(root, case)
                spans = activity.spans(signal)
            if raw is None:
                out = short_voice_review.scan(signal, spans, model, Deadline(), codec="pcm_s16le")
                chosen = out["selected"]
                raw = {"range": [chosen["start_sample"], chosen["end_sample"]] if chosen else None,
                       "assessed": bool(out["responses"])}
            views = read_views(signal, spans, model, raw["range"]) if raw["range"] is not None else None
            readings.append({"pcm_sha256": digest, "current": raw, "views": views})
            if len(readings) % 16 == 0 or len(readings) == len(rows):
                print(f"{panel} measured {len(readings)}/{len(rows)} files", flush=True)
        results[panel] = readings
    SOURCE.write(args.output, {"freeze_sha256": SOURCE.path_sha(args.run / "freeze.json"), "readings": results})


def agreement(row):
    current = row["current"]
    tested = row["views"]
    position = current["range"]
    assessed = current["assessed"]
    SOURCE.check(type(assessed) is bool and (position is None or (assessed and tuple(position) in GRID))
                 and (tested is None) == (position is None), "Input view differs from the original flag")
    if tested is None:
        return dict.fromkeys(METHODS)
    expected = adjusted_spans(position[0])
    SOURCE.check([line["range"] for line in tested["shifted"]] == [[loc, loc + 2 * RATE] for loc in expected]
                 and set(tested["band"]) == set(SHELVES), "Unexpected corroborating geometry")
    values = list(tested["band"].values())
    times = [line["score"] for line in tested["shifted"]]
    SOURCE.check(all(math.isfinite(x) for x in values) and all(x is None or math.isfinite(x) for x in times),
                 "Non-finite corroborating view")
    band = all(x > REVIEW_POINT for x in values)
    time = all(x is not None and x > REVIEW_POINT for x in times)
    return dict(zip(METHODS, (position, position if band else None, position if time else None,
                              position if band or time else None), strict=True))


def summarize(cases, observed):
    states = []
    for case, row in zip(cases, observed, strict=True):
        SOURCE.check(case["pcm_sha256"] == row["pcm_sha256"], "Measurement uses another sealed PCM")
        states.append(agreement(row))
    ordinary_by_group = {}
    for case, flags in zip(cases, states, strict=True):
        if case["truth"] is None and case["kind"] != "background":
            ordinary_by_group.setdefault(case["group"], []).append(flags)
    result = {}
    for method in METHODS:
        counts = {}
        for case, row, flags in zip(cases, observed, states, strict=True):
            truth = case["truth"]
            name = "edited" if truth is not None else "isolated" if case["kind"] == "background" else "genuine"
            selected = flags[method]
            useful = truth is not None and duration.compatible(selected, truth)
            collision = useful and any(duration.compatible(item[method], truth) for item in ordinary_by_group[case["group"]])
            counters = counts.setdefault(name, Counter(deliveries=0, assessed=0, flagged=0, useful=0, useful_clear=0,
                                                        removed_flags=0, lost_useful=0))
            counters.update(deliveries=1, assessed=int(row["current"]["assessed"]), flagged=int(selected is not None),
                            useful=int(useful), useful_clear=int(useful and not collision),
                            removed_flags=int(selected is None and flags["current"] is not None),
                            lost_useful=int(truth is not None and selected is None and duration.compatible(flags["current"], truth)))
        result[method] = counts
    return result


def review(receipt):
    prior = duration.read_gz(PANEL)
    plan = receipt["plan"]
    validate(plan, prior)
    SOURCE.check(receipt["freeze_sha256"] == ROOM.plan_hash(plan) and set(receipt["readings"]) == set(PANELS), "Study receipt differs")
    prior_current = parent_readings(prior)
    for panel in PANELS[:-1]:
        for row in receipt["readings"][panel]:
            SOURCE.check(row["current"] == prior_current[panel][row["pcm_sha256"]], "Prior app result differs")
    return {panel: summarize(plan["cases"][panel], receipt["readings"][panel]) for panel in PANELS}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="phase", required=True)
    source = sub.add_parser("freeze")
    for name in ("run", "cache", "background"):
        source.add_argument(name, type=Path)
    native = sub.add_parser("capture")
    for name in ("run", "weights", "diagnostic", "room", "output"):
        native.add_argument(name, type=Path)
    native.add_argument("panels", nargs="+", choices=PANELS)
    combine = sub.add_parser("combine")
    combine.add_argument("run", type=Path)
    combine.add_argument("output", type=Path)
    combine.add_argument("inputs", type=Path, nargs="+")
    portable = sub.add_parser("review")
    portable.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.phase == "freeze":
        freeze(args)
    elif args.phase == "capture":
        capture(args)
    elif args.phase == "combine":
        plan = SOURCE.read(args.run / "freeze.json")
        digest = SOURCE.path_sha(args.run / "freeze.json")
        readings = {}
        for path in args.inputs:
            part = SOURCE.read(path)
            SOURCE.check(part["freeze_sha256"] == digest and not set(readings) & set(part["readings"]), "Capture split differs")
            readings.update(part["readings"])
        receipt = {"plan": plan, "freeze_sha256": digest, "readings": readings}
        print(json.dumps(review(receipt), sort_keys=True, indent=2))
        args.output.write_bytes(gzip.compress((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode(), mtime=0))
    else:
        print(json.dumps(review(duration.read_gz(args.receipt)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
