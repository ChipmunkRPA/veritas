"""Measure paired room and competing-speech effects on voice localization."""

from __future__ import annotations

import argparse
import math
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from scipy.signal import fftconvolve

from scripts import audio_spatial_mask_aachen_transfer as aachen

SCHEMA = "chipmunk.audio-room-stress/1"
ROOMS = tuple(aachen.ROOMS)
WINDOWS = aachen.review.WINDOWS
SAMPLES, RATE = aachen.SAMPLES, aachen.RATE
LEVELS = (5, 0, -5, -10)
STAGES = ("dry", "early_room", "full_room", *[f"snr_{level}" for level in LEVELS])
PUBLIC = aachen.DOCS / "audio-spatial-mask-aachen-transfer-2026-11-13.json"
read, write, require, digest, sha = aachen.read, aachen.write, aachen.require, aachen.hash_file, aachen.sha
METHOD = {
    "sources": "Reuse seven exact Aachen publisher edit clips and measured room/background assignments. Reuse 12 known ordinary clips, reserving two per room for point selection before new inference, and add four previously unselected genuine speakers from the same publisher prefix to point selection, two per room. Reserve all six Aachen background-only clips and eight other genuine voices for assessment. These are exposed Aachen cohorts for controlled diagnostics, not fresh release validation.",
    "ladder": "For each source compare dry, direct arrival and first 8 ms of response, complete six-second room render, and full room plus the same competing speech at +5, 0, −5 and −10 dB. Preserve the original authored label clock. Match dry/early/full foreground RMS and use the same scalar gain at all stages within the group. Set gain for worst-case peak and independently round and exactly add integer stems. Reuse the same genuine controls at each level.",
    "scoring": "Use the two pinned two-second windows, speech admission and commercial voice weights. At noisy levels run native mono and both anonymous prior AuxIVA outputs. Select the strongest admitted response per arm and report its exact window. Clean stages use only mono. A useful location exceeds its point, overlaps at least half of a publisher edit and has interval IoU of at least 0.30. Inconclusive inputs are counted separately.",
    "points": "Use the earlier mono and AuxIVA points as primary comparison. Before evaluation inference, freeze a second AuxIVA point at the strict maximum from the eight ordinary development readers across all four noisy levels. No positives, evaluation controls or background-only controls enter point fitting. Only the development point stage writes this value.",
    "decision": "For follow-up, an additive arm must gain at least two edit groups from two generator families at each of two adjacent levels. Across levels permit at most one added ordinary group and zero added background-only groups. The same source counts once across levels. Paired dry/early/full transition analyses are diagnostic and do not affect the gate.",
}


def rank(value):
    return sha(f"{SCHEMA}:{value}".encode())


def annotation_path():
    return aachen.BASE / "spectra-recovery-2026-11-11/ami/ami_public_manual_1.6.2.zip"


def mean_energy(stereo):
    return float(np.square(stereo.mean(axis=1)).sum())


def peak_quantize(stereo):
    values = np.rint(stereo)
    require(np.max(abs(values)) < 32768, "Rendered input clips")
    return values.astype("<i2")


def read_background(file, support):
    with wave.open(str(file), "rb") as source:
        source.setpos(support["start"])
        raw = source.readframes(SAMPLES)
    require(sha(raw) == support["pcm_sha256"], "Existing background excerpt changed")
    return np.frombuffer(raw, dtype="<i2").astype(np.float64)


def convolve(waveform, impulse):
    return fftconvolve(waveform[:, None], impulse, axes=0)[:SAMPLES]


def ladder(dry, masker, room):
    dry = np.repeat(dry[:, None], 2, axis=1)
    impulse = room["lateral"]
    arrival = min(np.argmax(abs(impulse), axis=0))
    early = impulse[:arrival + 128].copy()
    early[-16:] *= np.linspace(1, 0, 16)[:, None]
    waveform = dry[:, 0]
    foregrounds = {"dry": dry, "early_room": convolve(waveform, early), "full_room": convolve(waveform, impulse)}
    dry_energy = mean_energy(dry)
    for stage in ("early_room", "full_room"):
        foregrounds[stage] *= math.sqrt(dry_energy / mean_energy(foregrounds[stage]))
    bg = convolve(masker, room["balanced"])
    unit = bg * math.sqrt(dry_energy / mean_energy(bg))
    foreground = foregrounds["full_room"]
    maximum = max(*(np.max(abs(x)) for x in foregrounds.values()), np.max(abs(foreground) + math.sqrt(10) * abs(unit)))
    gain = 22500 / maximum
    variants = [(stage, peak_quantize(signal * gain), None) for stage, signal in foregrounds.items()]
    original = variants[-1][1]
    for level in LEVELS:
        added = peak_quantize(unit * 10 ** (-level / 20) * gain)
        mixture = original.astype(np.int32) + added.astype(np.int32)
        measured = 10 * math.log10(mean_energy(original) / mean_energy(added))
        require(abs(measured - level) <= .02 and np.max(abs(mixture)) <= 22501, "Native noise ratio or peak differs")
        variants.append((f"snr_{level}", mixture.astype("<i2"), round(measured, 6)))
    return variants


def sources(args):
    publisher = args.publisher
    old_panel = read(PUBLIC)
    prior_run = args.input / "run"
    old_lock, old_rows = (read(prior_run / name) for name in ("freeze.json", "roster.json"))
    require(digest(prior_run / "freeze.json") == old_panel["freeze_sha256"] and
            digest(prior_run / "roster.json") == old_panel["private_roster_sha256"], "Prior native source commitment changed")
    require(digest(publisher / "LICENSE.txt") == aachen.publisher.PINS["LICENSE.txt"], "Publisher terms changed")
    require(digest(publisher / "locked/roster.json") == old_lock["prior_publisher_roster"], "Former publisher roster changed")
    _, points, used_ids, used_bytes, _ = aachen.prior_sources(args.prior_directional, publisher / "locked/roster.json")
    prefix, rir = (args.input / name for name in ("llama-first-288m.tar", "air_database_release_1_4.zip"))
    require(digest(prefix) == old_lock["prefix_sha256"] and digest(rir) == old_lock["aachen_archive_sha256"],
            "Publisher tar or measured room archive differs")
    cases, _ = aachen.read_publisher(prefix, publisher / "part-a.labels.txt", used_ids, used_bytes)
    by_source = {case["source"]: case for case in cases}
    rooms_meta, rooms, _ = aachen.rir_inputs(rir)
    require(rooms_meta == old_lock["rooms"], "Measured room coefficients differ")
    masks = aachen.ami_inputs(args.ami, annotation_path())
    existing = [row for row in old_rows if row["kind"] != "background_only"]
    used_speakers = {by_source[row["source"]["source"]]["speaker"] for row in existing}
    require(len(used_speakers) == 19 and len(old_rows) == 25, "Prior held source groups differ")
    prior_used = defaultdict(set)
    for row in old_rows:
        prior_used[row["ami"]["meeting"]].add(row["ami"]["start"])
    development = set()
    for room in ROOMS:
        controls = [row["group"] for row in existing if row["room"] == room and row["kind"] == "genuine"]
        development.update(sorted(controls, key=rank)[:2])
    for case in sorted(cases, key=lambda row: rank(row["source"])):
        if case["truth"] or case["speaker"] in used_speakers:
            continue
        slot = len(development) - 4
        require(slot < 4, "The retained bounded source roster changed")
        room = ROOMS[slot % len(ROOMS)]
        group = f"new-control-{slot:02d}"
        meeting = aachen.ROOMS[room][-1]
        _, support = aachen.masker(masks[meeting], prior_used[meeting], meeting, group)
        old_rows.append({"group": group, "room": room, "kind": "genuine", "truth": [], "family": None,
                         "source": aachen.source_receipt(case), "ami": support})
        development.add(group)
        used_speakers.add(case["speaker"])
    require(len(development) == 8 and len(used_speakers) == 23, "Additional point controls differ")

    roster, payloads = [], []
    for row in old_rows:
        support = row["ami"]
        room = rooms[row["room"]]
        masker = read_background(masks[support["meeting"]][0], support)
        if row["kind"] == "background_only":
            background = convolve(masker, room["balanced"])
            audio = peak_quantize(22500 * background / np.max(abs(background)))
            require(sha(audio.tobytes()) == row["stereo_pcm_sha256"], "Prior isolated background differs")
            variants = [("background_only", audio, None)]
        else:
            case = by_source[row["source"]["source"]]
            require(aachen.source_receipt(case) == row["source"] and [list(span) for span in case["truth"]] == row["truth"],
                    "Native dry excerpt or label clock changed")
            dry = aachen.pcm(case)
            variants = ladder(dry, masker, room)
        role = "development" if row["group"] in development else "assessment"
        for stage, stereo, actual in variants:
            roster.append({**{key: row[key] for key in ("group", "room", "kind", "truth", "family")}, "stage": stage,
                           "role": role, "case": len(roster), "actual_snr_db": actual, "sha256": sha(stereo.tobytes())})
            payloads.append(stereo)
    lock = {"schema": SCHEMA, "method": METHOD, "assets": old_lock["assets"], "windows": WINDOWS, "levels_db": LEVELS,
            "runner": digest(Path(__file__)), "ancestor": digest(PUBLIC), "prior_source_freeze": old_panel["freeze_sha256"],
            "prior_source_roster": old_panel["private_roster_sha256"], "prior_points": points,
            "distinct_point_groups": 8, "new_point_groups": 4, "assessment_edit_groups": 7,
            "assessment_ordinary_reader_groups": 8, "assessment_background_groups": 6}
    return lock, roster, payloads


def views(stereo, stage):
    signal = stereo.astype(np.float64)
    outputs = {"mono": [peak_quantize(signal.mean(axis=1))]}
    if stage.startswith("snr_") or stage == "background_only":
        candidates = aachen.auxiva.separate(signal)
        outputs["auxiva"] = [] if candidates is None else [peak_quantize(track) if np.max(abs(np.rint(track))) < 32768 else None
                                                            for track in candidates[0].T]
    return outputs


def measure(args, lock, roster, payloads, role):
    from app.detectors.audio.calibration import logit
    from app.detectors.audio.content import window_in_speech
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    gate = SpeechActivity(args.weights / "silero_vad.jit")
    model = load_voice({"weights": str(args.weights), "assets": lock["assets"]})
    output, cache = [], {}
    started = time.monotonic()
    for row, stereo in zip(roster, payloads, strict=True):
        if row["role"] != role or role == "development" and not row["stage"].startswith("snr_"):
            continue
        results = {}
        for arm, tracks in views(stereo, row["stage"]).items():
            if role == "development" and arm == "mono":
                continue
            arm_reads = []
            for waveform in tracks:
                if waveform is None:
                    arm_reads.append(None)
                    continue
                pcm = waveform.astype(np.float32) / 32768
                speech = gate.spans(pcm)
                scores = []
                for a, b in WINDOWS:
                    score = None
                    if window_in_speech(speech, a / RATE, b / RATE):
                        identity = sha(waveform[a:b].tobytes())
                        if identity not in cache:
                            probability, = model.predict([pcm[a:b]])
                            require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid voice output")
                            cache[identity] = float(logit(probability))
                        score = cache[identity]
                    scores.append(score)
                arm_reads.append(scores)
            results[arm] = arm_reads
        output.append({**{key: value for key, value in row.items() if key != "sha256"}, "responses": results})
        if len(output) % 10 == 0:
            print(role, len(output), "cases", len(cache), "forwards", round(time.monotonic() - started, 2), "seconds", flush=True)
    print(role, len(output), "cases", len(cache), "forwards", round(time.monotonic() - started, 2), "seconds", flush=True)
    return {"rows": output, "forwards": len(cache)}


def chosen(row, arm):
    return max(((score, -index) for track in row["responses"].get(arm, []) if track is not None
                for index, score in enumerate(track) if score is not None), default=None)


def outcomes(rows, arm, point):
    sets = {key: set() for key in ("located", "ordinary", "background", "wrong", "unassessed")}
    ranges = {}
    for row in rows:
        decision = chosen(row, arm)
        group = row["group"]
        if decision is None:
            sets["unassessed"].add(group)
            continue
        score, index = decision
        if score <= point:
            continue
        window = WINDOWS[-index]
        if not row["truth"]:
            sets["ordinary"].add(group)
            if row["kind"] == "background_only":
                sets["background"].add(group)
        elif any(aachen.review.location_matches(interval, window) for interval in row["truth"]):
            sets["located"].add(group)
            ranges[group] = {"family": row["family"], "room": row["room"], "window": list(window), "truth": row["truth"]}
        else:
            sets["wrong"].add(group)
    return sets, ranges


def fit_point(fit):
    maxima = defaultdict(list)
    for row in fit["rows"]:
        require(not row["truth"] and row["kind"] == "genuine" and row["role"] == "development", "Non-development input to point fit")
        reading = chosen(row, "auxiva")
        if reading is not None:
            maxima[row["group"]].append(reading[0])
    require(len(maxima) == 8 and len(fit["rows"]) == 32, "Point needs all eight admitted development groups")
    return max(max(scores) for scores in maxima.values())


def summarize(rows, prior, fitted):
    report, matched = {}, {}
    advances = {key: {} for key in ("auxiva_prior", "auxiva_development")}
    all_ordinary = defaultdict(set)
    for stage in (*STAGES, "background_only"):
        cohort = [row for row in rows if row["stage"] == stage]
        require(cohort and {row["role"] for row in cohort} == {"assessment"}, "Expected complete diagnostic cohort")
        mono, ranges = outcomes(cohort, "mono", prior["mono"])
        record = {"mono": {key: sorted(value) for key, value in mono.items()}, "mono_ranges": ranges}
        matched[stage] = mono["located"]
        if stage.startswith("snr_") or stage == "background_only":
            for name, point in (("auxiva_prior", prior["earlier_either"]), ("auxiva_development", fitted)):
                separated, locations = outcomes(cohort, "auxiva", point)
                new = separated["located"] - mono["located"]
                ordinary = separated["ordinary"] - mono["ordinary"]
                record[name] = {key: sorted(value) for key, value in separated.items()}
                record[name]["additions"] = {group: locations[group] for group in sorted(new)}
                record[name]["added_ordinary"] = sorted(ordinary)
                all_ordinary[name].update(ordinary)
                advances[name][stage] = len(new) >= 2 and len({locations[group]["family"] for group in new}) >= 2
        report[stage] = record

    shifts = {}
    for before, after in zip(STAGES, STAGES[1:]):
        def truth_scores(stage):
            return {row["group"]: max(eligible) for row in rows if row["stage"] == stage and row["truth"]
                    and (eligible := [score for index, score in enumerate(row["responses"]["mono"][0])
                                      if score is not None and any(aachen.review.location_matches(truth, WINDOWS[index])
                                                                   for truth in row["truth"])])}

        a, b = truth_scores(before), truth_scores(after)
        common = a.keys() & b.keys()
        shifts[f"{before}_to_{after}"] = {"paired_truth_window_scores": len(common),
                                           "median_log_odds_change": round(median(b[group] - a[group] for group in common), 6) if common else None,
                                           "lost_locations": sorted(matched[before] - matched[after]),
                                           "gained_locations": sorted(matched[after] - matched[before])}
    decisions = {}
    noisy = tuple(f"snr_{level}" for level in LEVELS)
    for name, passes in advances.items():
        adjacent = any(passes[a] and passes[b] for a, b in zip(noisy, noisy[1:]))
        added_background = report["background_only"][name]["added_ordinary"]
        decisions[name] = {"passes_adjacent_levels": adjacent, "added_genuine_groups": len(all_ordinary[name] - set(added_background)),
                           "added_background_only_groups": len(added_background),
                           "passes": adjacent and len(all_ordinary[name]) <= 1 and not added_background}
    return {"stages": report, "paired_transitions": shifts, "decision": decisions}


def compare(path, value):
    import json

    require(json.dumps(read(path), sort_keys=True) == json.dumps(value, sort_keys=True), "Frozen inputs or executable changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--publisher", type=Path, required=True)
    parser.add_argument("--prior-directional", type=Path, required=True)
    parser.add_argument("--ami", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=aachen.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    lock, roster, payloads = sources(args)
    freeze, inputs = args.run / "freeze.json", args.run / "roster.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(freeze, lock)
        write(inputs, roster)
        print("Frozen", len(roster), "deliveries", Counter((row["role"], row["stage"]) for row in roster),
              "lock", digest(freeze), flush=True)
        return
    compare(freeze, lock)
    compare(inputs, roster)
    point_file, fit_file = args.run / "point.json", args.run / "development.json"
    if args.phase == "fit":
        require(not fit_file.exists() and not point_file.exists(), "Development already recorded")
        fit = measure(args, lock, roster, payloads, "development")
        point = fit_point(fit)
        write(fit_file, fit)
        write(point_file, {"freeze_sha256": digest(freeze), "development_sha256": digest(fit_file), "point": point})
        print("Frozen development point", point, flush=True)
        return
    receipt = args.run / "public.json"
    require(not receipt.exists(), "Assessment already recorded")
    point, fit = read(point_file), read(fit_file)
    require(point == {"freeze_sha256": digest(freeze), "development_sha256": digest(fit_file), "point": fit_point(fit)},
            "Development fit changed")
    scored = measure(args, lock, roster, payloads, "assessment")
    summary = summarize(scored["rows"], lock["prior_points"], point["point"])
    write(receipt, {"schema": SCHEMA, "freeze": lock, "freeze_sha256": digest(freeze), "private_roster_sha256": digest(inputs),
                    "development": fit, "point": point, "assessment": scored, "summary": summary})


if __name__ == "__main__":
    main()
