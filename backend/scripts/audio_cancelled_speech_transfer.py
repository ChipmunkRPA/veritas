# SPDX-License-Identifier: MIT
"""Compare original-channel and difference scores for speech lost in mono."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

RATE = 16000
DELIVERIES = ("isolated", "competing")
VIEWS = ("mono", "left", "difference")
METHODS = (*VIEWS, "difference_gain")
SCORE_GATE = 6400  # 400 ms of detected speech in a measured one-second window.
SCHEMA = "chipmunk.audio-cancelled-speech/1"
EXPECTED_GROUPS = {"phoneme/calibration": 16, "phoneme/held": 24, "stock/held": 8, "sine/held": 18}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def group_sources(rooms, sine_inputs, sine_reads):
    from scripts import audio_short_edit_background_sine as sine
    from scripts import audio_short_edit_room_training as room

    plan = read(rooms / "freeze.json")
    room.validate(plan)
    for row in plan["groups"]:
        if row["role"] == "fit":
            continue
        available = row["observations"]
        roles = ["untouched", "genuine_splice"]
        if row["role"] == "held":
            roles.extend(sorted(kind for kind in row["inputs"] if kind.startswith("edited/")))
        views = {delivery: {kind: (rooms, available[condition][kind]) for kind in roles}
                 for delivery, condition in zip(DELIVERIES, ("room", "competing"), strict=True)}
        yield row["panel"], row["id"], row["role"], row["span"], views, (rooms, available["competing"]["background"])
    held_plan = read(sine_inputs / "freeze.json")
    sine.check_plan(held_plan)
    measured = read(sine_reads / "cases.json")
    sine.check_cases(held_plan, measured)
    index = {(row["group"], row["condition"], row["kind"]): row["pcm_sha256"] for row in measured}
    backgrounds = {row["room"]: index[i, "competing", "background"] for i, row in enumerate(held_plan["backgrounds"])}
    for row in held_plan["pairs"]:
        group = row["id"]
        roles = ("untouched", "edited/sine")
        views = {delivery: {kind: (sine_reads, index[group, condition, kind]) for kind in roles}
                 for delivery, condition in zip(DELIVERIES, ("dry", "competing"), strict=True)}
        yield "sine", group, "held", row["truth"], views, (sine_reads, backgrounds[row["room"]])


def stereo_cases(identity, index, truth, views, background):
    import numpy as np

    from scripts.audio_short_edit_detector_training import pcm

    def load(source):
        root, sha = source
        return pcm(root, {"pcm_sha256": sha})

    start = min(5 * RATE, max(0, (sum(truth) - RATE) // 2))
    excerpt = slice(start, start + RATE)
    window = 1 + int(digest(str(identity).encode())[0], 16) % 2
    # A shared, shifted recording context keeps all conditions matched and
    # supplies a genuine mono foreground even when the opposed speech cancels.
    center = np.roll(load(background), 17003)[:3 * RATE]
    rms = np.sqrt(np.mean(center[window*RATE:(window+1)*RATE] ** 2))
    mid = np.rint(center * min(.010 / max(rms, .002), 3.) * 32768).astype(np.int32)
    for delivery, rows in views.items():
        clips = {kind: load(source)[excerpt] for kind, source in rows.items()}
        reference = clips["untouched"]
        loudness = np.sqrt(np.mean(reference ** 2))
        peak = max(float(np.max(np.abs(clip))) for clip in clips.values())
        gain = min(.080 / max(loudness, .001), .65 / max(peak, .001), 12.)
        for kind, source in rows.items():
            signal = clips[kind]
            side = np.rint(signal * gain * 32768).astype(np.int32)
            # Bundle a shared original context and one opposed window. Both
            # channels preserve the exact unfiltered hidden window before mixing.
            left, right = mid.copy(), mid.copy()
            left[window*RATE:(window+1)*RATE] += side
            right[window*RATE:(window+1)*RATE] -= side
            stereo = np.column_stack((left, right))
            if np.any(np.abs(stereo) > 32767):
                raise ValueError("A planned stereo delivery would clip")
            label = [max(0, truth[0]-start) + window*RATE, min(RATE, truth[1]-start) + window*RATE]
            path = f"{index:03}-{delivery}-{kind.replace('/', '-')}.wav"
            row = {"file": path, "panel": identity[0], "group": index, "role": identity[2], "kind": kind, "delivery": delivery,
                   "window": window, "source_start": start, "truth": label if kind.startswith("edited/") else None,
                   "source_pcm_sha256": source[1], "background_pcm_sha256": background[1],
                   "difference_sha256": digest(side.astype("<i2").tobytes()), "mono_window_sha256":
                   digest(mid[window*RATE:(window+1)*RATE].astype("<i2").tobytes())}
            yield row, stereo.astype("<i2").tobytes()


def freeze(rooms, sine, sine_reads, target):
    import wave

    from scripts.prepare_audio_specialists_v15 import ASSETS

    target.mkdir(mode=0o700)
    cases = []
    group_counts = Counter()
    for index, (panel, group, role, truth, views, background) in enumerate(group_sources(rooms, sine, sine_reads)):
        group_counts[f"{panel}/{role}"] += 1
        for row, stereo in stereo_cases((panel, group, role), index, truth, views, background):
            file = target / row["file"]
            with wave.open(str(file), "wb") as stream:
                stream.setparams((2, 2, RATE, 0, "NONE", ""))
                stream.writeframes(stereo)
            cases.append(row | {"wav_sha256": digest(file.read_bytes())})
    assert group_counts == EXPECTED_GROUPS
    parents = {key: digest(file.read_bytes()) for key, file in (
        ("rooms", rooms / "freeze.json"), ("sine", sine / "freeze.json"), ("sine_pcm", sine_reads / "cases.json"))}
    write(target / "freeze.json", {"schema": SCHEMA, "parents": parents, "assets": ASSETS, "cases": cases,
                                   "code_sha256": digest(Path(__file__).read_bytes())})
    print(f"Sealed {len(cases)} stereo deliveries across {sum(group_counts.values())} source groups", flush=True)


def score(folder, weights, phase, destination, calibration=None):
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.evidence.stereo_binding import destructive_downmix_windows
    from scripts.audio_short_edit_room_training import runtime
    from scripts.run_audio_specialists_v15 import load_voice

    plan_file = folder / "freeze.json"
    plan = read(plan_file)
    assert plan["schema"] == SCHEMA and plan["code_sha256"] == digest(Path(__file__).read_bytes())
    input_sha = digest(plan_file.read_bytes())
    point_lock = None
    if phase == "held":
        assert calibration is not None, "Fit control-only points before any held forwards"
        previous = read(calibration)
        assert previous["phase"] == "calibration" and previous["input_sha256"] == input_sha
        point_lock = digest(calibration.read_bytes())
        cutoffs = points(previous["rows"])
        print(f"Frozen control-only points {digest(json.dumps(cutoffs, sort_keys=True).encode())}", flush=True)
    voice = load_voice({"weights": weights, "assets": plan["assets"]})
    speech = runtime(weights / "silero_vad.jit")
    cache = {}
    outcomes = []
    for row in (case for case in plan["cases"] if case["role"] == phase):
        start, stop = row["window"] * RATE, (row["window"] + 1) * RATE
        source = decode_audio(folder / row["file"], source_sha256=row["wav_sha256"], inspect_stereo=True, retain_stereo=True)
        assert source.stereo_pcm is not None and len(source.pcm) == len(source.stereo_pcm) == 3 * RATE
        measured = source.mapping["stereo_context"]
        entered = destructive_downmix_windows(measured)
        assert entered in ([], [row["window"]])
        mono = source.pcm[start:stop]
        left, right = source.stereo_pcm[start:stop].astype(np.int32).T
        delta = (left - right) // 2
        assert digest(delta.astype("<i2").tobytes()) == row["difference_sha256"]
        assert digest(np.rint(mono * 32768).astype("<i2").tobytes()) == row["mono_window_sha256"]
        channels = (mono, left.astype(np.float32) / 32768., delta.astype(np.float32) / 32768.)
        readings = {}
        for view, values in zip(VIEWS, channels, strict=True):
            sha = digest(values.astype("<f4").tobytes())
            if sha not in cache and (entered or view == "mono"):
                matched = sum(round(b*RATE) - round(a*RATE) for a, b in speech.spans(values))
                response = float(voice.predict([values])[0]) if matched >= SCORE_GATE else None
                cache[sha] = {"sha256": sha, "speech_samples": matched, "score": response}
            readings[view] = cache[sha] if entered or view == "mono" else None
        outcomes.append({**row, "eligible": bool(entered), "stereo": measured["windows"][row["window"]], "readings": readings})
        if len(outcomes) % 20 == 0:
            print(f"{phase}: {len(outcomes)} decoded and scored stereo deliveries", flush=True)
    write(destination, {"schema": SCHEMA, "input_sha256": input_sha, "assets": plan["assets"],
                        "phase": phase, "rows": outcomes, "calibration_sha256": point_lock})


def responses(row):
    scores = {view: entry["score"] if entry else None for view, entry in row["readings"].items()}
    side, mono = scores["difference"], scores["mono"]
    scores["difference_gain"] = side-mono if side is not None and mono is not None else None
    return scores


def points(rows):
    scores = defaultdict(dict)
    for row in rows:
        assert row["role"] == "calibration" and row["truth"] is None
        for view, value in responses(row).items():
            if value is not None:
                key = (row["delivery"], view)
                scores[key][row["group"]] = max(value, scores[key].get(row["group"], -math.inf))
    points = {}
    for (delivery, view), values in sorted(scores.items()):
        ordered = sorted(values.values())
        strict = ordered[-1]
        wider = ordered[math.ceil(len(ordered) * .90)-1]
        if len(ordered) >= 8:
            points.setdefault(delivery, {})[view] = {"strict": strict, "wider": wider, "groups": len(ordered)}
    return points


def calculate(rows, cutoffs):
    summary = defaultdict(Counter)
    measured = {}
    for row in rows:
        values = responses(row)
        identity = row["panel"], row["delivery"], row["group"]
        for view in METHODS:
            point = cutoffs.get(row["delivery"], {}).get(view, {})
            for policy in ("strict", "wider"):
                response = values[view]
                state = (None if response is None or policy not in point else response > point[policy])
                measured[*identity, row["kind"], view, policy] = state
    for row in rows:
        panel, delivery, group, kind = (row[key] for key in ("panel", "delivery", "group", "kind"))
        positive = bool(row["truth"])
        for view in METHODS:
            for policy in ("strict", "wider"):
                key = "/".join((panel, delivery, view, policy))
                counts = summary[key]
                counts["edited" if positive else "genuine"] += 1
                state = measured[panel, delivery, group, kind, view, policy]
                if positive:
                    counts["edited_unassessed"] += state is None
                    counts["edited_flags"] += state is True
                    controls = [measured.get((panel, delivery, group, c, view, policy)) for c in ("untouched", "genuine_splice")]
                    counts["flags_clear_of_matched_genuine"] += state is True and controls[0] is False and not any(c is True for c in controls)
                else:
                    counts["genuine_unassessed"] += state is None
                    counts["genuine_flags"] += state is True
    return dict(sorted(summary.items()))


def review(receipt):
    calibration, held = (receipt[phase] for phase in ("calibration", "held"))
    assert calibration["schema"] == held["schema"] == SCHEMA
    assert calibration["input_sha256"] == held["input_sha256"] and calibration["assets"] == held["assets"]
    groups = defaultdict(set)
    for phase, record in (("calibration", calibration), ("held", held)):
        assert phase == record["phase"]
        assert len({row["file"] for row in record["rows"]}) == len(record["rows"])
        for row in record["rows"]:
            assert row["role"] == phase and row["delivery"] in DELIVERIES and set(row["readings"]) == set(VIEWS)
            groups[f"{row['panel']}/{phase}"].add(row["group"])
            window = row["stereo"]
            start, end = window["start_sample"], window["end_sample"]
            assert start == row["window"] * RATE and end == start + RATE
            correlation = window["correlation"]
            assert row["eligible"] == (window["downmix"]["strong_attenuation"] and correlation is not None and correlation <= -.85)
            if truth := row["truth"]:
                assert start <= truth[0] < truth[1] <= end and row["kind"].startswith("edited/")
            for view, reading in row["readings"].items():
                if reading is None:
                    assert view != "mono" and not row["eligible"]
                    continue
                score = reading["score"]
                assert 0 <= reading["speech_samples"] <= RATE and (score is None) == (reading["speech_samples"] < SCORE_GATE)
                assert score is None or 0 <= score <= 1
    assert {key: len(values) for key, values in groups.items()} == EXPECTED_GROUPS
    cutoffs = points(calibration["rows"])
    assert receipt["points"] == cutoffs
    assert receipt["summary"] == calculate(held["rows"], cutoffs)
    return receipt["summary"]


def combine(calibration, held, output):
    a, b = read(calibration), read(held)
    assert b["calibration_sha256"] == digest(calibration.read_bytes())
    cutoffs = points(a["rows"])
    payload = {"calibration": a, "held": b, "points": cutoffs, "summary": calculate(b["rows"], cutoffs)}
    review(payload)
    with output.open("xb") as stream:
        stream.write(gzip.compress(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    print(json.dumps(payload["summary"], sort_keys=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode, keys in (("freeze", ("rooms", "sine", "sine_reads", "output")), ("score", ("folder", "weights", "output")),
                       ("combine", ("calibration", "held", "output")), ("review", ("receipt",))):
        sub = modes.add_parser(mode)
        for key in keys:
            sub.add_argument(key, type=Path)
        if mode == "score":
            sub.add_argument("--phase", choices=("calibration", "held"), required=True)
            sub.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.rooms, args.sine, args.sine_reads, args.output)
    elif args.mode == "score":
        score(args.folder, args.weights, args.phase, args.output, args.calibration)
    elif args.mode == "combine":
        combine(args.calibration, args.held, args.output)
    else:
        print(json.dumps(review(json.loads(gzip.decompress(args.receipt.read_bytes()))), sort_keys=True, indent=2))
