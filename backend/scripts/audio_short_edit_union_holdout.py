# SPDX-License-Identifier: MIT
"""Re-test the frozen two-second and shorter-window union on new readers and rooms."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError

from app.evidence.audio_short_review_binding import GRID
from scripts import audio_short_edit_duration_transfer as duration

ROOM, BASE, SOURCE = duration.ROOM, duration.BASE, duration.SOURCE
SINE = duration.previous.sine
PANEL = duration.previous.PANELS / "audio-short-edit-duration-transfer-2026-11-15.json.gz"
PARENT = "63eda8e93883e3f1d6f2c3ee7ccc75bc8c261fb1"
SCHEMA = "chipmunk.audio-short-edit-union-holdout/1"
SHARDS = (7, 14, 15, 16, 18, 19, 20, 21)
FOLD = "MeloTTS"
RANKER = "width_calibrated"
POINT = "competing/width_calibrated"
ROOM_COUNT = 6
MIN_ROOMS = 5
MIN_READERS, MAX_READERS = 20, 32
ROOM_ID = re.compile(r"[EIT]S[0-9]{4}[a-z]")
METHOD = {
    "sources": "Select up to 32 newly declared SINE readers from eight fixed public prefixes, matching the previous intake "
               "geometry. Select six four-person AMI meetings absent from the prior evaluation docs. Their annotated "
               "participants must be disjoint within the held set and from the room model source roles and preceding SINE comparisons. "
               "Take the first six candidates with accessible official PCM and nonoverlapping participants. "
               "Allow five if the annotation-qualified official sources are exhausted. "
               "Record inaccessible publisher ranges and check source availability before inference. "
               "Use 6 s clips and identical room sound within each matched pair at -5 dB foreground-to-room RMS. "
               "Include each isolated room as a genuine control.",
    "comparison": "Keep the existing two-second PCM rule and the exact wider competing-speech width-normalized rule for the "
                  "second fit from the published duration study. The union can report both existing and shorter ranges. "
                  "Also measure the shorter check as a fallback only after a two-second non-flag. Assess useful locations "
                  "and ordinary and isolated flags separately with paired changes and exact sample overlap.",
}


def parent():
    return duration.read_gz(PANEL)


def protocol(old):
    trial = duration.previous.training.previous(duration.earlier(), FOLD, "fit")
    points = old["calibration"]["folds"][FOLD]["points"][POINT]
    return {"schema": SCHEMA, "revision": PARENT, "method": METHOD, "shards": list(SHARDS),
            "code_sha256": SOURCE.path_sha(Path(__file__)), "prior_receipt_sha256": SOURCE.path_sha(PANEL),
            "shorter_fit_sha256": trial["weights_sha256"], "shorter_operating_point": points,
            "current_assets": duration.read_gz(duration.EARLIER_SINE)["current"]["assets"],
            "room_range": [MIN_ROOMS, ROOM_COUNT], "reader_range": [MIN_READERS, MAX_READERS]}


def known_readers(old):
    excluded, _ = duration.prohibited_readers()
    excluded.update(row["reader"] for row in old["freeze"]["fresh"]["pairs"])
    prior_shards = {row["shard"] for row in old["freeze"]["fresh"]["shards"]}
    SOURCE.check(not prior_shards.intersection(SHARDS), "The new source prefixes overlap the previous experiment")
    return excluded


def comparison_rooms():
    original = {name for row in duration.earlier()["plan"]["groups"] for name in ROOM_ID.findall(row["capture"])}
    recent = parent()["freeze"]["fresh"]
    earlier = duration.read_gz(duration.EARLIER_SINE)["held"]["plan"]
    return original | {row["room"] for panel in (recent, earlier) for row in panel["backgrounds"]}


def select_sine(cache, inputs, old):
    excluded = known_readers(old)
    listing = SINE.source_listing(cache)
    (inputs / "sine").mkdir()
    offers = {}
    pins = []
    for shard in SHARDS:
        selected, pin = duration.previous.select_shard(cache, shard, listing[f"data/split-{shard}.tar.gz"], excluded)
        for reader, option in selected.items():
            if reader not in offers or option[0] < offers[reader][0]:
                offers[reader] = option
        pins.append(pin)
        print(f"SINE shard {shard} supplied {len(selected)} eligible readers", flush=True)
    SOURCE.check(len(offers) >= MIN_READERS, "Too few newly declared SINE readers")
    rows = []
    for index, (_, record, bodies) in enumerate(sorted(offers.values(), key=lambda item: item[0])[:MAX_READERS]):
        for name, body in bodies.items():
            (inputs / "sine" / f"{index:03}-{name}.wav").write_bytes(body)
        rows.append(record | {"id": index})
    return rows, pins


def room_targets(annotations):
    from xml.etree import ElementTree
    from zipfile import ZipFile

    from scripts.audio_harmonic_overlap_transfer import room_activity

    ami = SINE.ami
    pin = SOURCE.read(duration.previous.PANELS / "audio-locate-competing-speech-2026-11-13.json")["freeze"]["annotations_sha256"]
    SOURCE.check(SOURCE.path_sha(annotations) == pin, "Unexpected AMI annotation release")
    previous = subprocess.run(["git", "grep", "-h", "-o", "-E", ROOM_ID.pattern, PARENT, "--", "docs"],
                              cwd=SINE.prior.ROOT, check=True, text=True, capture_output=True)
    earlier = set(previous.stdout.splitlines())
    comparison = comparison_rooms()
    with ZipFile(annotations) as archive:
        tree = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        participants = {row.get("observation"): {item.get("global_name") for item in row if item.get("global_name")}
                        for row in tree.iter() if ROOM_ID.fullmatch(row.get("observation") or "")}
        earlier_participants = set().union(*(participants.get(name, set()) for name in comparison))
        available = set(archive.namelist())
        eligible = sorted(participants, key=lambda room: SOURCE.digest(f"short-edit-union-room/1/{room}".encode()))
        targets = []
        for room in eligible:
            group = participants[room]
            if (room in earlier or room in comparison or len(group) != 4 or group & earlier_participants
                    or any(f"words/{room}.{speaker}.words.xml" not in available for speaker in "ABCD")):
                continue
            usage = room_activity(archive, room)
            windows = []
            for offset in range(1800, len(usage) - 600, 300):
                values = usage[offset:offset + 600]
                active, overlap = int((values > 0).sum()), int((values >= 2).sum())
                if active >= 500 and overlap >= 200:
                    windows.append((SOURCE.digest(f"short-edit-union-window/1/{room}/{offset}".encode()), offset, active, overlap))
            if not windows:
                continue
            _, offset, active, overlap = min(windows)
            targets.append({"room": room, "start_sample": 160 * offset, "speech_samples": 160 * active,
                            "overlap_samples": 160 * overlap, "url": ami.capture_url(room),
                            "participant_commitments": sorted(SOURCE.digest(f"AMI-public-person/{name}".encode()) for name in group)})
    SOURCE.check(len(targets) >= ROOM_COUNT, "Insufficient participant-distinct AMI meetings")
    return {"annotations_sha256": pin, "revision": PARENT, "excluded_room_count": len(earlier),
            "participant_exclusion_rooms": sorted(comparison), "targets": targets}


def fetch_room(target, background):
    for attempt in range(2):
        try:
            return SINE.download_background(target, background), "available"
        except HTTPError as error:
            if error.code not in (403, 404, 410, 429, 500, 502, 503, 504):
                raise
            status = str(error.code)
        except (TimeoutError, URLError):
            status = "unavailable"
        if status in ("404", "410") or attempt == 1:
            return None, status


def acquire_rooms(schedule, background):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    rooms, attempts, people = [], [], set()
    batch_size = 5
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        # Fetch in parallel but keep the annotation-only order for admission.
        # Only source availability and participant separation affect selection.
        targets = schedule["targets"]
        for first in range(0, len(targets), batch_size):
            batch = targets[first:first + batch_size]
            acquired = pool.map(partial(fetch_room, background=background), batch)
            for target, (room, status) in zip(batch, acquired, strict=True):
                participants = set(target["participant_commitments"])
                if room is not None:
                    if people & participants:
                        status = "shared_participant"
                    else:
                        rooms.append(room)
                        people.update(participants)
                        status = "acquired"
                attempts.append({"room": target["room"], "status": status})
                print(f"AMI room {target['room']}: {status}", flush=True)
                if len(rooms) == ROOM_COUNT:
                    break
            if len(rooms) == ROOM_COUNT:
                break
    SOURCE.check(MIN_ROOMS <= len(rooms) <= ROOM_COUNT, "Too few planned AMI recordings are accessible from the publisher")
    return rooms, schedule | {"targets": schedule["targets"][:len(attempts)], "attempts": attempts,
                              "offer_count": len(schedule["targets"])}


def native_inputs(args):
    initial = parent()
    protocol_lock = protocol(initial)
    run = args.run
    run.mkdir(mode=0o700)
    SOURCE.write(run / "protocol.json", protocol_lock)
    rows, pins = select_sine(args.cache, run, initial)
    schedule = room_targets(args.annotations)
    SOURCE.write(run / "room-plan.json", schedule)
    background = run / "background"
    background.mkdir()
    rooms, schedule = acquire_rooms(schedule, background)
    for row in rows:
        row["room"] = rooms[row["id"] % len(rooms)]["room"]
    (run / "pcm").mkdir()
    cases = []
    for row, samples in duration.previous.deliveries(run, {"pairs": rows, "backgrounds": rooms}):
        if row["condition"] == "competing":
            (run / "pcm" / (row["pcm_sha256"] + ".pcm")).write_bytes(samples.tobytes())
            cases.append(row)
    frozen = {"protocol": protocol_lock, "shards": pins, "pairs": rows, "rooms": rooms,
              "room_plan": schedule, "cases": cases}
    validate_frozen(frozen)
    SOURCE.write(run / "freeze.json", frozen)
    print(f"Frozen {len(rows)} fresh reader pairs and {len(rooms)} recorded room controls", flush=True)


def validate_frozen(frozen):
    old = parent()
    SOURCE.check(frozen["protocol"] == protocol(old), "Prespecified rules differ")
    excluded = known_readers(old)
    rooms, pairs, cases = (frozen[key] for key in ("rooms", "pairs", "cases"))
    known = {case["pcm_sha256"] for fold in old["freeze"]["rosters"].values()
             for split in fold.values() for case in split}
    sources = {row["reader"] for row in pairs}
    SOURCE.check(MIN_READERS <= len(sources) == len(pairs) <= MAX_READERS and not sources & excluded
                 and MIN_ROOMS <= len({r["room"] for r in rooms}) == len(rooms) <= ROOM_COUNT
                 and len({commitment for r in rooms for commitment in r["participant_commitments"]}) == 4 * len(rooms)
                 and [row["shard"] for row in frozen["shards"]] == list(SHARDS), "Fresh rosters overlap")
    plan = frozen["room_plan"]
    targets = plan["targets"]
    attempts = plan["attempts"]
    admitted = [row for row, trace in zip(targets, attempts, strict=True) if trace["status"] == "acquired"]
    annotation_pin = SOURCE.read(duration.previous.PANELS / "audio-locate-competing-speech-2026-11-13.json")["freeze"]["annotations_sha256"]
    SOURCE.check(plan["revision"] == PARENT and plan["annotations_sha256"] == annotation_pin
                 and plan["participant_exclusion_rooms"] == sorted(comparison_rooms())
                 and [row["room"] for row in targets] == [row["room"] for row in attempts]
                 and all(row["status"] in ("acquired", "shared_participant", "unavailable", "403", "404", "410",
                                           "429", "500", "502", "503", "504") for row in attempts)
                 and (attempts[-1]["status"] == "acquired" if len(rooms) == ROOM_COUNT else len(attempts) == plan["offer_count"])
                 and admitted ==
                 [{key: row[key] for key in target} for row, target in zip(rooms, admitted, strict=True)],
                 "Acquired rooms are outside the annotation plan")
    for i, row in enumerate(pairs):
        raw = row["files"]
        regions, _ = SINE.sine.label_clock(row["publisher_cut_paste_clock"], raw["variant"]["samples"], "cut_paste")
        left, right, _ = regions[1]
        limit = min(raw["original"]["samples"], raw["variant"]["samples"]) - 144000
        starts = [tick for tick in range(0, limit + 1, 2400) if tick + 6000 <= left < right <= tick + 138000]
        selected = min(starts, key=lambda tick: duration.previous.order("sine-background-training-clock/1", row["stem"], tick),
                       default=None)
        SOURCE.check(row["id"] == i and row["shard"] in SHARDS and row["reader"] == row["stem"].split("-", 1)[0]
                     and row["room"] == rooms[i % len(rooms)]["room"]
                     and selected is not None and row["native_start"] == selected
                     and row["truth"] == [(x - selected) * 2 // 3 for x in (left, right)]
                     and 4800 <= right - left <= 24000, "Reader delivery or edit timing differs")
    expected = [(row["id"], label, truth) for row in pairs for label, truth in
                (("untouched", None), ("edited/sine", row["truth"]))]
    expected.extend((i, "background", None) for i in range(len(rooms)))
    SOURCE.check(expected == [(c["group"], c["kind"], c["truth"]) for c in cases]
                 and all(c["condition"] == "competing" and c["panel"] == "sine" for c in cases)
                 and len({c["pcm_sha256"] for c in cases}) == len(cases)
                 and not any(c["pcm_sha256"] in known for c in cases), "Held PCM or delivery roster overlaps the prior study")


def short_fit(run, phase, weights, fit, output):
    frozen = SOURCE.read(run / "freeze.json")
    validate_frozen(frozen)
    activity = ROOM.runtime(weights / "silero_vad.jit")
    rows = frozen["cases"]
    if phase == "current":
        from app.detectors.audio import short_voice_review
        from scripts.run_audio_specialists_v15 import load_voice

        class Deadline:
            def check(self, stage):
                pass

        model = load_voice({"weights": weights, "assets": frozen["protocol"]["current_assets"]})
        readings = []
        for row in rows:
            pcm = BASE.pcm(run, row)
            decision = short_voice_review.scan(pcm, activity.spans(pcm), model, Deadline(), codec="pcm_s16le")
            selected = decision["selected"]
            readings.append({"assessed": bool(decision["responses"]),
                             "range": [selected["start_sample"], selected["end_sample"]] if selected else None})
            if len(readings) % 16 == 0:
                print(f"Current scan measured {len(readings)}/{len(rows)} files", flush=True)
    else:
        trial = duration.previous.training.previous(duration.earlier(), FOLD, "fit")
        model, mean, scale = ROOM.load_trial(fit, trial)
        extract = ROOM.feature_inputs(run, activity)
        choices = duration.model_readings(model, mean, scale, rows, extract)
        point = frozen["protocol"]["shorter_operating_point"]
        readings = [duration.best(row, RANKER, point["scales"]) for row in choices]
    SOURCE.write(output, {"phase": phase, "frozen_sha256": SOURCE.path_sha(run / "freeze.json"),
                          "rows": [{"pcm_sha256": row["pcm_sha256"], "reading": readout}
                                   for row, readout in zip(rows, readings, strict=True)]})


def replay(receipt):
    frozen = receipt["frozen"]
    validate_frozen(frozen)
    cases = frozen["cases"]
    result = {}
    for phase in ("current", "shorter"):
        measured = receipt[phase]
        SOURCE.check(measured["phase"] == phase and measured["frozen_sha256"] == ROOM.plan_hash(frozen)
                     and [row["pcm_sha256"] for row in measured["rows"]] == [row["pcm_sha256"] for row in cases],
                     "Measurements differ from the sealed input")
        result[phase] = [row["reading"] for row in measured["rows"]]
    summary, by_case, comparisons = {}, [], defaultdict(Counter)
    cutoff = frozen["protocol"]["shorter_operating_point"]["wider"]
    for case, incumbent, candidate in zip(cases, result["current"], result["shorter"], strict=True):
        first = incumbent["range"]
        SOURCE.check(type(incumbent["assessed"]) is bool and
                     (first is None or (incumbent["assessed"] and tuple(first) in GRID)), "Current interval is off-grid")
        if candidate is not None:
            a, b = candidate["range"]
            SOURCE.check(0 <= a < b <= BASE.SAMPLES and a % (5 * BASE.HOP) == 0 and
                         (b - a) in {width * BASE.HOP for width in duration.WIDTHS} and math.isfinite(candidate["score"]),
                         "Short interval is off-grid")
        second = candidate["range"] if candidate is not None and candidate["score"] > cutoff else None
        current = [first] if first else []
        shorter = [second] if second else []
        policies = {"current": current, "shorter": shorter, "union": current + shorter, "fallback": current or shorter}
        admitted = {"current": incumbent["assessed"], "shorter": candidate is not None,
                    "union": incumbent["assessed"] or candidate is not None,
                    "fallback": incumbent["assessed"] or candidate is not None}
        by_case.append((case, policies, admitted))
    controls = {case["group"]: selected for case, selected, _ in by_case if case["kind"] == "untouched"}
    for case, views, available in by_case:
        truth = case["truth"]
        label = "edited" if truth is not None else "isolated" if case["kind"] == "background" else "genuine"
        for mode, ranges in views.items():
            counts = summary.setdefault(mode, {}).setdefault(label, Counter(deliveries=0, unassessed=0, flagged=0,
                                                                           useful=0, misplaced=0, useful_clear=0,
                                                                           misplaced_spans=0))
            useful = truth is not None and any(duration.compatible(r, truth) for r in ranges)
            collision = useful and any(duration.compatible(r, truth) for r in controls[case["group"]][mode])
            counts.update(deliveries=1, unassessed=int(not available[mode]), flagged=int(bool(ranges)),
                          useful=int(useful), misplaced=int(truth is not None and bool(ranges) and not useful),
                          misplaced_spans=sum(not duration.compatible(r, truth) for r in ranges) if truth is not None else 0,
                          useful_clear=int(useful and not collision))
            if mode in ("union", "fallback"):
                base = (bool(views["current"]) if truth is None else any(duration.compatible(r, truth) for r in views["current"]))
                tested = bool(ranges) if truth is None else useful
                comparisons[mode].update({label + "_gained": int(tested and not base), label + "_lost": int(base and not tested)})
    return {"readers": len(frozen["pairs"]), "rooms": len(frozen["rooms"]), "summary": summary, "paired_changes": comparisons}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    phase = cli.add_subparsers(dest="phase", required=True)
    native = phase.add_parser("freeze")
    for key in ("run", "cache", "annotations"):
        native.add_argument(key, type=Path)
    for name in ("current", "shorter"):
        cmd = phase.add_parser(name)
        for key in ("run", "weights", "output"):
            cmd.add_argument(key, type=Path)
        if name == "shorter":
            cmd.add_argument("fit", type=Path)
    publish = phase.add_parser("combine")
    for key in ("run", "current", "shorter", "output"):
        publish.add_argument(key, type=Path)
    portable = phase.add_parser("review")
    portable.add_argument("receipt", type=Path)
    args = cli.parse_args()
    if args.phase == "freeze":
        native_inputs(args)
    elif args.phase in ("current", "shorter"):
        short_fit(args.run, args.phase, args.weights, getattr(args, "fit", None), args.output)
    elif args.phase == "combine":
        receipt = {"frozen": SOURCE.read(args.run / "freeze.json"), "current": SOURCE.read(args.current),
                   "shorter": SOURCE.read(args.shorter)}
        print(json.dumps(replay(receipt), sort_keys=True, indent=2))
        args.output.write_bytes(gzip.compress((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode(), mtime=0))
    else:
        print(json.dumps(replay(duration.read_gz(args.receipt)), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
