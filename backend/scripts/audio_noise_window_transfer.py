# SPDX-License-Identifier: Apache-2.0
"""Test one source-clock noise reduction and staggered speech-window comparison."""

from __future__ import annotations

import argparse
import re
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from scripts import audio_partial_publisher_transfer as publisher
from scripts import review_audio_reduced_background_localization as clock
from scripts.audio_reduced_background_localization import SplitArchive, assets_for_scores

common, require = publisher.common, publisher.require
SCHEMA = "chipmunk.audio-noise-window-transfer/1"
RATE, WINDOW = clock.RATE, clock.WINDOW
STEP = RATE * 3 // 4
SCENES = {"selection": ("gain_control", "restaurant", "coastal"), "held": ("gain_control", "vienna", "market")}
PREVIOUS = ("audio-partial-publisher-transfer", "audio-reduced-background-localization", "audio-nemotron-noise-transfer")
MEMBER = re.compile(r"R01TTS\.0\.b/(dev01-([a-z0-9]+)-partial-(cp|oa)_(\d+)_\d+_\d+_\d+)\.wav")
PINS = {
    "part-b-32m.prefix": "f31f2fbd07e71abf59cba961125c17dfcb1eb8f82437aa01d4d62f5eb565cc56",
    "part-b-extension": "3f0a74ed923586fa3cd0f1a68f9c0dae87fa8b593119089b05311a62d1bbdc2c",
    "part-b-slice": "7f2ca6fe76cc4d077ed7098413ea245498e284cc30658bb5ad4e03284164d4b4",
    "part-b.labels.txt": "d0a9b856b3cf7ba09fbbe21507a1224e86aefc85a1893af25736d87454f28e80",
}
BACKGROUNDS = {
    "restaurant": ("restaurant.ogg", "0aeb1fa7a3d9647d3284e97d9880b6375b54db2eda6a517c64e3bb02ac8a51cc", (6, 18, 36, 54)),
    "coastal": ("cori-street.ogg", "50e5eb59b296580340898f16e737f826a9fd7fef6fc45b07bb3e0318a9e58495", (18, 48, 85, 121)),
    "vienna": ("vienna.wav", "a4b1807f396592351d7ffb17873391cec9b9144f9216af3d463b1f2009e90e36", (2, 6, 10, 14)),
    "market": ("market.ogg", "7e677dad7ce4cbebf7b4e9d3555c0af22fe51ceb2514602b5f75a50e6e12ec85", (1, 5, 9, 13)),
}
METHOD = {
    "window_samples": WINDOW, "dense_step": clock.STEP, "staggered_step": STEP,
    "calibration_sha256": clock.METHOD["primary_calibration_sha256"],
    "noise_model": {"fft": 512, "hop": 128, "quantile": .20, "strength": 1.5, "amplitude_floor": .35},
    "separate_view": "Keep original AI responses. Original-leaning windows veto rescue. Require speech in at least half the reduced window.",
    "original_comparator": "The existing sparse rule, fixed 1.5 s grid, and its unchanged 1.5 s overlap rule.",
    "staggered": "Two interleaved grid phases at 0 and .75 s. Intersect three consecutive positive 3 s windows, 1.5 s extent.",
    "decision_point": "Maximum admitted reduced logit on selection genuine and background-only, strict greater.",
    "role_scenes": {role: list(scenes) for role, scenes in SCENES.items()},
    "background_db": -5, "selection_speakers": 12, "held_genuine_speakers": 12,
    "held_partial_speakers": 18, "held_minimum_generators": 6,
    "target": {"minimum_added_partial_groups": 4, "minimum_gained_families": 3,
               "maximum_added_genuine_groups": 2, "maximum_added_genuine_time_fraction": .03,
               "maximum_new_clean_or_isolated_groups": 0},
}
ARMS = ("sparse", "dense", "staggered", "reduced", "combined")


def archive_partials(sources):
    require({name: common.sha(sources / name) for name in PINS} == PINS, "Publisher archive range or labels differ")
    labels = {name: parts for name, *parts in map(str.split, (sources / "part-b.labels.txt").open())}
    require(len(labels) == 64_388, "Unexpected publisher label roster")
    pool, seen, complete = defaultdict(list), set(), Counter()

    def collect(stream, chunk):
        try:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    match = MEMBER.fullmatch(member.name)
                    require(match and 44 < member.size < 2**20 and match[1] in labels, "Unexpected publisher WAV member")
                    with archive.extractfile(member) as file:
                        body = file.read()
                    require(len(body) == member.size, "Incomplete native WAV")
                    complete[chunk] += 1
                    name, family, recipe, group = match.groups()
                    if name in seen:
                        continue
                    seen.add(name)
                    if not 6 <= Decimal(labels[name][0]) <= 14:
                        continue
                    raw = publisher.native(body)
                    samples = len(raw) // 2
                    try:
                        segments = publisher.labels_for(labels[name], samples)
                    except ValueError:
                        complete["invalid_publisher_clocks"] += 1
                        continue
                    edits = clock.intervals_union([[a, b] for a, b, kind in segments if kind == "spoof"])
                    if (6 * RATE <= samples <= 14 * RATE and .10 * samples <= sum(b - a for a, b in edits) <= .60 * samples
                            and any(9600 <= b - a <= 38400 for a, b in edits)):
                        pool[family].append(({"id": name, "group": group, "family": family, "recipe": recipe,
                                              "edits": edits, "source_sha256": common.digest(body),
                                              "source_pcm_sha256": common.digest(raw)}, raw))
        except (tarfile.ReadError, EOFError) as error:
            require(isinstance(error, EOFError) or "end of data" in str(error), "Invalid archive structure")

    with (sources / "part-b-32m.prefix").open("rb") as head, (sources / "part-b-extension").open("rb") as tail:
        collect(SplitArchive(head, tail), "prefix")
    with (sources / "part-b-slice").open("rb") as stream:
        for offset in range(0, 2**21, tarfile.BLOCKSIZE):
            block = stream.read(tarfile.BLOCKSIZE)
            if not block.startswith(b"R01TTS.0.b/"):
                continue
            try:
                member = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
            except tarfile.HeaderError:
                continue
            if member.isfile() and MEMBER.fullmatch(member.name):
                stream.seek(offset)
                break
        else:
            raise ValueError("The archive slice has no aligned WAV member")
        collect(stream, "slice")
    require(set(pool) == set(publisher.GENERATORS), "All six edited-speech families are required")
    return pool, dict(complete)


def prior_groups():
    documents = {name: common.PUBLIC / f"{name}-2026-11-12.json" for name in PREVIOUS}
    a, b, c = (common.read(path) for path in documents.values())
    selection = set()
    held = {row["source"]["group"] for row in b["roster"]
            if row["source"] and row["source"]["label"] == "genuine"}
    seen = ({row["source"]["speaker"] for row in a["roster"] if row["source"]}
            | {row["source"]["group"] for row in b["roster"] if row["source"]}
            | {row["origin"]["group"] for row in c["held"] if row["label"] != "isolated"})
    a_genuine = {row["source"]["speaker"] for row in a["roster"] if row["source"] and row["source"]["kind"] == "genuine"}
    c_genuine = {row["origin"]["group"] for row in c["held"] if row["label"] == "bonafide"}
    for group in sorted((a_genuine | c_genuine) - held, key=lambda group: common.digest(f"{SCHEMA}/{group}".encode())):
        selection.add(group)
        if len(selection) == METHOD["selection_speakers"]:
            break
    require(len(selection) == METHOD["selection_speakers"] and len(held) == METHOD["held_genuine_speakers"],
            "Genuine source roles changed")
    return selection, held, seen, {name: common.sha(path) for name, path in documents.items()}


def select_edited(pool, excluded, seen):
    options = {}
    for family, pairs in pool.items():
        choices = {}
        ranked = sorted(pairs, key=lambda pair: (pair[0]["group"] in seen, common.digest(f"{SCHEMA}/{pair[0]['id']}".encode())))
        for row, pcm in ranked:
            if row["group"] not in excluded:
                choices.setdefault(row["group"], (row, pcm))
        options[family] = choices
    roles = [(family, index) for family in sorted(options, key=lambda family: len(options[family])) for index in range(3)]
    assigned, selected = {}, {}

    def place(role, visited):
        for group, pair in options[role[0]].items():
            if group in visited:
                continue
            visited.add(group)
            if group not in assigned or place(assigned[group], visited):
                assigned[group], selected[role] = role, pair
                return True
        return False

    for role in roles:
        require(place(role, set()), f"Too few distinct eligible speakers for {role[0]}")
    return [selected[role] for role in roles]


def original_pcm(group, candidates):
    ordered = sorted(candidates, key=lambda pair: common.digest(f"{SCHEMA}/{pair[0]['publisher_id']}".encode()))
    row, raw = next(pair for pair in ordered if pair[0]["group"] == group)
    return {"id": row["publisher_id"], "group": group, "family": None, "recipe": None, "edits": [],
            "source_sha256": row["source_wav_sha256"], "source_pcm_sha256": row["source_pcm_sha256"]}, raw


def prepare(run, sources, part_a, labels, sounds):
    from scripts.audio_gtcrn_traffic import float_pcm, pcm
    from scripts.audio_noise_views_transfer import match_views
    from scripts.audio_reduced_background_localization import source_candidates

    selection, controls, seen, history = prior_groups()
    pool, scanned = archive_partials(sources)
    # Allocate genuine controls using publisher source IDs, independent of scores.
    part_a_paths = [part_a / name for name in ("part-a-192m.prefix.tar", "part-a-192m-352m.tarseg",
                                             "slice-2g.tarseg", "slice-6g.tarseg", "slice-10g.tarseg")]
    genuine, scanned_a, _ = source_candidates(*part_a_paths[:2], labels, part_a_paths[2:])
    candidates = genuine["genuine", None]
    available = {row["group"] for row, _ in candidates}
    for groups in (controls, selection):
        require(groups <= available, "Expected Part A genuine speaker unavailable")
    partial = select_edited(pool, selection | controls, seen)
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir(mode=0o700)
    noises = {}
    for scene, (name, pin, _) in BACKGROUNDS.items():
        require(common.sha(sounds / name) == pin, "Background recording differs")
        noises[scene] = float_pcm(common.decode(sounds / name))
    rows = []

    def keep(role, kind, scene, row, raw, offset):
        identifier = len(rows)
        require(6 * RATE <= len(raw) // 2 <= 14 * RATE, "Unexpected waveform size")
        record = {"id": identifier, "role": role, "kind": kind, "scene": scene, "source": row,
                  "samples": len(raw) // 2, "pcm_sha256": common.digest(raw), "offset": offset}
        (run / "pcm" / f"{identifier:03}.pcm").write_bytes(raw)
        rows.append(record)

    genuine_rows = [(role, original_pcm(group, candidates))
                    for role, groups in (("selection", selection), ("held", controls)) for group in sorted(groups)]
    groups = [(role, "genuine", pair) for role, pair in genuine_rows] + [("held", "partial", pair) for pair in partial]
    for index, (role, kind, (row, raw)) in enumerate(groups):
        wave = float_pcm(raw)
        scenes = SCENES[role]
        offsets = {scene: BACKGROUNDS[scene][2][index % 4] * RATE for scene in scenes[1:]}
        excerpts = {scene: noises[scene][offset:offset + len(wave)] for scene, offset in offsets.items()}
        views = match_views(wave, excerpts, METHOD["background_db"])
        for scene in scenes:
            keep(role, kind, scene, row, pcm(views[scene]), offsets.get(scene))
    for role, scenes in SCENES.items():
        for scene in scenes[1:]:
            for second in BACKGROUNDS[scene][2][::2]:
                offset = second * RATE
                keep(role, "isolated", scene, None, pcm(noises[scene][offset:offset + 8 * RATE]), offset)
    require(len({row["pcm_sha256"] for row in rows}) == len(rows), "Repeated delivery bytes")
    common.save(run / "roster.json", rows)
    lock = {"schema": SCHEMA, "method": METHOD, "pins": PINS, "backgrounds": BACKGROUNDS, "previous": history,
            "part_a": {path.name: common.sha(path) for path in part_a_paths},
            "runner_sha256": common.sha(Path(__file__)), "roster_sha256": common.sha(run / "roster.json"),
            "complete_part_a_members": scanned_a, "complete_part_b_members": scanned, "voice_assets": common.assets(),
            "model_responses_at_freeze": 0, "prior_partial_speakers": sum(row["group"] in seen for row, _ in partial)}
    common.save(run / "freeze.json", lock)
    print("Frozen", len(rows), "deliveries", common.sha(run / "freeze.json"), flush=True)


def reduce_noise(pcm):
    import numpy as np
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    config = METHOD["noise_model"]
    size, hop = config["fft"], config["hop"]
    _, _, spec = stft(pcm.astype(np.float64), fs=RATE, window="hann", nperseg=size, noverlap=size - hop)
    power = np.abs(spec) ** 2
    stable = np.quantile(power, config["quantile"], axis=1, keepdims=True)
    keep = np.sqrt(np.maximum(config["amplitude_floor"] ** 2,
                              1 - config["strength"] * stable / np.maximum(power, 1e-12)))
    keep = uniform_filter(keep, size=(3, 5), mode="nearest")
    _, output = istft(spec * keep, fs=RATE, window="hann", nperseg=size, noverlap=size - hop)
    require(len(output) >= len(pcm), "Noise reduction changed the sample clock")
    return output[:len(pcm)]


def geometry(length):
    sparse, dense = clock.schedules(length)
    staggered = [[a, a + WINDOW] for a in range(0, length - WINDOW + 1, STEP)]
    return sparse, dense, staggered


def locked(run):
    lock, rows = (common.read(run / name) for name in ("freeze.json", "roster.json"))
    require(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["runner_sha256"] == common.sha(Path(__file__))
            and lock["roster_sha256"] == common.sha(run / "roster.json"), "Study freeze changed")
    return lock, rows


def score(run, role, assets=None):
    from app.detectors.audio import calibration
    from app.detectors.audio.content import window_in_speech
    from scripts.audio_gtcrn_traffic import float_pcm, pcm
    from scripts.audio_radar_transfer import adapter_result

    lock, rows = locked(run)
    require(METHOD["calibration_sha256"] == calibration.SHA256, "Current calibration changed")
    point_pin = None
    if role == "held":
        point_pin = common.sha(run / "point.json")
        require(common.read(run / "point.json")["selection_sha256"] == common.sha(run / "selection.json"),
                "Reduced-view point lacks its selection commitment")
    voice, vad = assets if assets is not None else assets_for_scores(lock)
    bundle = calibration.load()
    results = []
    for row in rows:
        if row["role"] != role:
            continue
        raw = (run / "pcm" / f"{row['id']:03}.pcm").read_bytes()
        require(common.digest(raw) == row["pcm_sha256"] and len(raw) == 2 * row["samples"], "Frozen waveform changed")
        native = float_pcm(raw)
        reduced_raw = pcm(reduce_noise(native))
        reduced = float_pcm(reduced_raw)
        sparse, _, staggered = geometry(len(native))
        observations = []
        for a, b in sorted(set(map(tuple, sparse + staggered))):
            clip = native[a:b]
            measured = adapter_result(clip, voice, vad)
            direction, original = measured["decision"], measured["raw"]
            require((measured["scope"] and measured["windows"] == [[0, WINDOW]]) or direction == "unassessed",
                    "Original window scope differs")
            processed = None
            prior = "unassessed" if original is None else calibration.initial_decision(calibration.probability(original, bundle), bundle)
            # The unprocessed tendency vetoes a rescue even when the stability
            # probe withholds its original-leaning direction.
            if [a, b] in staggered and direction in clock.RESCUE and prior != "original":
                clip = reduced[a:b]
                if window_in_speech(vad.spans(clip), 0., 3.):
                    processed = calibration.logit(voice.predict([clip])[0])
            observations.append({"span": [a, b], "direction": direction, "original": original.hex() if original is not None else None,
                                 "processed": processed.hex() if processed is not None else None})
        results.append({"id": row["id"], "windows": observations, "reduced_sha256": common.digest(reduced_raw)})
        if len(results) % 8 == 0:
            print(role, len(results), "deliveries", flush=True)
    common.save(run / f"{role}.json", {"freeze_sha256": common.sha(run / "freeze.json"), "point_sha256": point_pin,
                                      "rows": results})
    print(role, "complete", len(results), flush=True)


def fit(run):
    locked(run)
    source = run / "selection.json"
    result = common.read(source)
    require(result["freeze_sha256"] == common.sha(run / "freeze.json") and result["point_sha256"] is None,
            "Selection responses do not match the freeze")
    scores = [float.fromhex(window["processed"]) for row in result["rows"] for window in row["windows"]
              if window["processed"] is not None]
    require(bool(scores), "No admitted selection response")
    common.save(run / "point.json", {"selection_sha256": common.sha(source), "point_hex": max(scores).hex(),
                                     "observations": len(scores)})
    print("Reduced-view point", max(scores), "on", len(scores), "ordinary windows", flush=True)


def ranges(length, checks, point):
    sparse, dense, staggered = geometry(length)
    scheduled = sorted(set(map(tuple, sparse + staggered)))
    require(scheduled == [tuple(check["span"]) for check in checks], "Response windows differ from exact schedule")
    directions = {tuple(check["span"]): check["direction"] for check in checks}
    original = {span for span, direction in directions.items() if direction == "ai"}
    processed = {tuple(check["span"]) for check in checks if check["processed"] is not None
                 and float.fromhex(check["processed"]) > point}
    incumbent = [span for span in sparse if tuple(span) in original]
    if len(incumbent) < 2 or any(directions[tuple(span)] == "original" for span in sparse):
        incumbent = []

    def triplets(positives):
        return [[end[0], start[1]] for start, mid, end in zip(staggered, staggered[1:], staggered[2:])
                if all(tuple(span) in positives for span in (start, mid, end))]

    original_dense = clock.intersections(dense, original)
    rescue = triplets(original | processed)
    return dict(zip(ARMS, (incumbent, original_dense, triplets(original), rescue, original_dense + rescue), strict=True))


def report(run):
    lock, rows = locked(run)
    selected, held, fit_record = (common.read(run / f"{name}.json") for name in ("selection", "held", "point"))
    require(held["freeze_sha256"] == selected["freeze_sha256"] == common.sha(run / "freeze.json")
            and held["point_sha256"] == common.sha(run / "point.json")
            and fit_record["selection_sha256"] == common.sha(run / "selection.json"), "Scoring receipts differ")
    point = float.fromhex(fit_record["point_hex"])
    cells = defaultdict(Counter)
    gains, family_gains, new_genuine, clean_or_isolated = set(), set(), set(), set()
    for row, responses in zip((row for row in rows if row["role"] == "held"), held["rows"], strict=True):
        require(responses["id"] == row["id"], "Response row differs")
        spans = ranges(row["samples"], responses["windows"], point)
        scene, kind, source = row["scene"], row["kind"], row["source"]
        truth = source["edits"] if source else []
        group = source["group"] if source else f"{scene}/{row['offset']}"
        cell = cells[f"{scene}/{kind}"]
        cell["files"] += 1
        cell["assessed_original"] += any(window["direction"] != "unassessed" for window in responses["windows"])
        cell["assessed_reduced"] += any(window["processed"] is not None for window in responses["windows"])
        cell["genuine_samples"] += row["samples"] - sum(b - a for a, b in truth)
        useful = {}
        for arm in ARMS:
            union = clock.intervals_union(spans[arm])
            useful[arm] = clock.useful(spans[arm], truth)
            cell[f"{arm}_flags"] += bool(union)
            cell[f"{arm}_useful"] += useful[arm]
            cell[f"{arm}_genuine_samples_flagged"] += sum(b - a for a, b in union) - sum(
                clock.overlap(span, edit) for span in union for edit in truth)
        if kind == "partial" and scene != "gain_control" and useful["combined"] and not useful["dense"]:
            gains.add(group)
            family_gains.add(source["family"])
        if kind != "partial" and spans["combined"] and not spans["dense"]:
            (clean_or_isolated if scene == "gain_control" or kind == "isolated" else new_genuine).add(group)
    noisy = [row for key, row in cells.items() if not key.startswith("gain_control/")]
    added_time = (sum(row["combined_genuine_samples_flagged"] - row["dense_genuine_samples_flagged"] for row in noisy)
                  / sum(row["genuine_samples"] for row in noisy))
    target = METHOD["target"]
    gates = {"added_noisy_groups": len(gains) >= target["minimum_added_partial_groups"],
             "added_families": len(family_gains) >= target["minimum_gained_families"],
             "new_noisy_genuine_groups": len(new_genuine) <= target["maximum_added_genuine_groups"],
             "new_clean_or_isolated_groups": len(clean_or_isolated) <= target["maximum_new_clean_or_isolated_groups"],
             "genuine_time": added_time <= target["maximum_added_genuine_time_fraction"]}
    summary = {"cells": dict(cells), "new_noisy_partial_groups": len(gains), "new_families": sorted(family_gains),
               "new_noisy_genuine_groups": len(new_genuine), "new_clean_or_isolated_groups": len(clean_or_isolated),
               "added_noisy_genuine_time_fraction": added_time, "fixed_gate": gates, "passes": all(gates.values()),
               "serving": False}
    common.save(run / "receipt.json", {"freeze": lock, "roster": rows, "selection": selected, "held": held,
                                      "point": fit_record, "summary": summary})
    print(common.encoded(summary).decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "selection", "fit", "held", "report", "run"))
    parser.add_argument("run", type=Path)
    for name in ("sources", "part-a", "labels", "sounds"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args.run, args.sources, args.part_a, args.labels, args.sounds)
    elif args.phase in {"selection", "held"}:
        score(args.run, args.phase)
    elif args.phase == "run":
        lock, _ = locked(args.run)
        assets = assets_for_scores(lock)
        score(args.run, "selection", assets)
        fit(args.run)
        score(args.run, "held", assets)
        report(args.run)
    else:
        {"fit": fit, "report": report}[args.phase](args.run)


if __name__ == "__main__":
    main()
