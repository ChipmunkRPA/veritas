# SPDX-License-Identifier: Apache-2.0
"""Test whether overlapping speech decisions locate published partial edits."""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import wave
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from app.evidence import audio_long_review_binding as long_review
from scripts import audio_noise_views_transfer as views
from scripts import audio_radar_transfer as radar

common, require = radar.common, radar.require
RATE = 16_000
WINDOW, STEP = 3 * RATE, 24_000
SCHEMA = "chipmunk.audio-partial-publisher-transfer/1"
PUBLISHER = "https://doi.org/10.5281/zenodo.14214149"
PINS = {
    "part-a-96m.prefix.tar": "4a8a7684db08acc8b41a8e8df3030f9a64281b8c90f285408fc194470b98c18e",
    "part-a.labels.txt": "cbeebd5030f1111f1ec5606bf28a6cbe9600d65f3eea7cd149a7b8dfc422ebda",
    "LICENSE.txt": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
}
MEMBER = re.compile(r"R01TTS\.0\.a/((?:dev-clean|test-clean|dev01-[a-z0-9]+-(?:full|partial-cf))_(\d+)_\d+_\d+_\d+)\.wav")
GENERATORS = ("cosyvoice", "elevenlab", "gptsovits", "ljjets", "xttsv2", "yourtts")
OFFSETS = {"traffic": (1, 4, 7, 10), "cafe": radar.SCENES["cafe"]}
GATE = {"minimum_noisy_added_groups": 4, "minimum_generator_families": 3, "maximum_noisy_added_genuine_groups": 1,
        "maximum_clean_added_genuine_groups": 0, "maximum_isolated_flags": 0, "maximum_lost_useful_dry_or_full": 0}


def native(body):
    with wave.open(io.BytesIO(body), "rb") as stream:
        require((stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype())
                == (1, 2, RATE, "NONE"), "Unexpected publisher WAV format")
        count = stream.getnframes()
        raw = stream.readframes(count)
    require(len(raw) == 2 * count, "Incomplete publisher PCM")
    return raw


def labels_for(items, samples):
    duration, label, *segments = items
    require(abs(round(Decimal(duration) * RATE) - samples) <= RATE // 1000, "Publisher duration differs from the native clock")
    clock = []
    for item in segments:
        a, b, direction = item.split("-")
        clock.append([round(Decimal(a) * RATE), round(Decimal(b) * RATE), direction])
    require(bool(clock) and clock[0][0] == 0 and abs(clock[-1][1] - samples) <= RATE // 1000, "Incomplete label clock")
    clock[-1][1] = samples
    require(all(a < b and direction in {"spoof", "bonafide"} for a, b, direction in clock)
            and all(a[1] == b[0] for a, b in zip(clock, clock[1:]))
            and label == ("spoof" if any(s[2] == "spoof" for s in clock) else "bonafide"), "Invalid publisher segments")
    return clock


def acquire(source):
    require({key: common.sha(source / key) for key in PINS} == PINS, "Publisher bytes differ")
    require((source / "part-a-96m.prefix.tar").stat().st_size == 96 * 1024**2, "Unbounded publisher archive")
    labels = {name: parts for name, *parts in map(str.split, (source / "part-a.labels.txt").read_text().splitlines())}
    require(len(labels) == 76_228, "Publisher label count differs")
    roster, names = defaultdict(list), set()
    try:
        with tarfile.open(source / "part-a-96m.prefix.tar", "r|") as archive:
            for entry in archive:
                if not entry.isfile():
                    continue
                match = MEMBER.fullmatch(entry.name)
                require(match is not None and 44 < entry.size < 2 * 1024**2, "Unexpected publisher member")
                name, speaker = match.groups()
                require(name not in names and name in labels, "Missing or repeated publisher identity")
                with archive.extractfile(entry) as stream:
                    body = stream.read()
                require(len(body) == entry.size, "Incomplete publisher member")
                names.add(name)
                if not 6 <= Decimal(labels[name][0]) <= 14:
                    continue
                raw = native(body)
                samples = len(raw) // 2
                clock = labels_for(labels[name], samples)
                if not 6 * RATE <= samples <= 14 * RATE:
                    continue
                truth = [[start, stop] for start, stop, direction in clock if direction == "spoof"]
                generator = name.split("-")[1] if name.startswith("dev01-") else None
                kind = "genuine" if not truth else "full" if truth == [[0, samples]] else "partial"
                if kind == "partial":
                    edit_length = sum(b - a for a, b in truth)
                    if not (.10 * samples <= edit_length <= .60 * samples and any(9600 <= b - a <= 38400 for a, b in truth)):
                        continue
                row = {"id": name, "speaker": speaker, "kind": kind, "generator": generator, "clock": clock,
                       "truth": truth, "samples": samples, "wav_sha256": common.digest(body), "pcm_sha256": common.digest(raw)}
                roster[kind, generator].append((row, raw))
    except (EOFError, tarfile.ReadError) as error:
        require(isinstance(error, EOFError) or "end of data" in str(error), "Unexpected publisher read failure")
    require(len(names) == 474, "The bounded tar member roster changed")
    return roster


def select(roster):
    selected, speakers = [], set()

    def take(key, count):
        taken = 0
        for row, raw in sorted(roster[key], key=lambda pair: common.digest(f"{SCHEMA}/{pair[0]['id']}".encode())):
            if row["speaker"] not in speakers:
                speakers.add(row["speaker"])
                selected.append((row, raw))
                taken += 1
            if taken == count:
                return
        raise ValueError(f"Too few distinct publisher speakers for {key}")

    for generator in sorted(GENERATORS, key=lambda name: (len({r['speaker'] for r, _ in roster['partial', name]}), name)):
        take(("partial", generator), 2)
    take(("genuine", None), 10)
    for generator in GENERATORS:
        take(("full", generator), 1)
    require(len({r["pcm_sha256"] for r, _ in selected}) == len(selected), "Repeated source PCM")
    return selected


def dense_spans(samples):
    return [[start, start + WINDOW] for start in range(0, samples - WINDOW + 1, STEP)]


def background_paths(background):
    names = tuple(f"{scene}.ogg" for scene in OFFSETS)
    for name in names:
        require(common.sha(background / name) == radar.SOURCE_FILES[name], "Background source changed")
    return {name[:-4]: views.enhancement.float_pcm(common.decode(background / name)) for name in names}


def build(source, background, run):
    selected = select(acquire(source))
    recordings = background_paths(background)
    run.mkdir(mode=0o700)
    (run / "pcm").mkdir(mode=0o700)
    roles, rows = Counter(), []

    def store(values, scene, source_row, offsets):
        raw = views.enhancement.pcm(values)
        n = len(values)
        row = {"key": len(rows), "scene": scene, "source": source_row, "mix_offsets": offsets, "samples": n,
               "pcm_sha256": common.digest(raw), "sparse": [list(x) for x in long_review.intervals(n)], "dense": dense_spans(n)}
        (run / "pcm" / f"{row['key']:03}.pcm").write_bytes(raw)
        rows.append(row)

    for row, raw in selected:
        rank = roles[row["kind"]]
        roles[row["kind"]] += 1
        offsets = {scene: starts[rank % len(starts)] * RATE for scene, starts in OFFSETS.items()}
        clips = {scene: recordings[scene][start:start + row["samples"]] for scene, start in offsets.items()}
        for scene, values in views.match_views(views.enhancement.float_pcm(raw), clips, -5).items():
            store(values, scene, row, offsets)
    for scene, starts in OFFSETS.items():
        for seconds in starts:
            start = seconds * RATE
            store(recordings[scene][start:start + 8 * RATE], scene, None, {scene: start})
    require(len(rows) == 92, "Unexpected prepared panel size")
    views.save(run / "roster.json", rows)
    views.save(run / "freeze.json", {"schema": SCHEMA, "publisher": PUBLISHER, "input_files": PINS,
                                   "backgrounds": {scene: radar.SOURCE_FILES[f'{scene}.ogg'] for scene in OFFSETS},
                                   "code_sha256": common.sha(Path(__file__)), "roster_sha256": common.sha(run / "roster.json"),
                                   "reviewer_sha256": common.sha(Path(__file__).with_name("review_audio_partial_publisher_transfer.py")),
                                   "assets": common.assets(), "calibration_sha256": "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf",
                                   "target_db": -5, "dense_hop_samples": STEP, "rule": GATE, "classifier_forwards": 0})
    print(json.dumps({"freeze_sha256": common.sha(run / 'freeze.json'), "source_groups": dict(roles), "views": len(rows),
                      "unique_window_inputs": sum(len(set(map(tuple, row['sparse'] + row['dense']))) for row in rows)}), flush=True)


def locked(run):
    freeze = common.read(run / "freeze.json")
    require(freeze["code_sha256"] == common.sha(Path(__file__)) and freeze["roster_sha256"] == common.sha(run / "roster.json"),
            "Freeze differs from the runner or the planned roster")
    return freeze, common.read(run / "roster.json")


def score(run):
    from app.core.config import settings
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    freeze, roster = locked(run)
    assets = {name: pin for name, pin in freeze['assets'].items() if name not in common.SOUND_ASSETS}
    require(settings.audio_detector_mode == "candidate" and freeze['calibration_sha256'] == calibration.SHA256,
            "The tested speech decision must use the pinned calibration")
    voice = load_voice({"weights": str(common.WEIGHTS), "assets": assets})
    vad = SpeechActivity(common.WEIGHTS / "silero_vad.jit")
    decisions = []
    for row in roster:
        raw = (run / "pcm" / f"{row['key']:03}.pcm").read_bytes()
        require(common.digest(raw) == row["pcm_sha256"] and len(raw) == row["samples"] * 2, "Prepared PCM changed")
        pcm = views.enhancement.float_pcm(raw)
        result = []
        for start, stop in sorted(set(map(tuple, row["sparse"] + row["dense"]))):
            chunk = pcm[start:stop]
            response = radar.adapter_result(chunk, voice, vad)
            require((response["scope"] and response["windows"] == [[0, WINDOW]]) or response["decision"] == "unassessed",
                    "Window did not follow the current short-speech assessment")
            result.append({"span": [start, stop], "direction": response["decision"], "pcm_sha256": common.digest(raw[2 * start:2 * stop])})
        decisions.append({"key": row["key"], "windows": result})
        if len(decisions) % 12 == 0 or len(decisions) == len(roster):
            print(f"{len(decisions)}/{len(roster)} clips scored", flush=True)
    views.save(run / "decisions.json", {"freeze_sha256": common.sha(run / "freeze.json"), "rows": decisions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--background", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        build(args.source, args.background, args.run)
    else:
        score(args.run)


if __name__ == "__main__":
    main()
