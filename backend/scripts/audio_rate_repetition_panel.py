# SPDX-License-Identifier: Apache-2.0
"""Compare native stereo playback-rate matches on source-disjoint local panels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from scripts.audio_aac_repetition_panel import HELD_VOICES
from scripts.audio_native_alignment_panel import (
    DURATION,
    GAINS,
    SHIFTS,
    digest,
    iou,
    read,
    save,
    save_wav,
    segment,
)

SCHEMA = "chipmunk.audio-rate-repetition-panel/1"
SCENES = {"development": ("DLIVING", "OOFFICE"), "confirmation": ("DKITCHEN", "NFIELD", "STRAFFIC", "TMETRO")}
MEETINGS = {"development": "ES2004a", "confirmation": "TS3004a"}
SCENE_STARTS = (18, 48, 78, 102, 132, 166, 201, 275)
VOICE_STARTS = (33, 48, 97, 132, 156, 190, 247, 303)
# These values were selected for the recipe independently of the search implementation.
DELIVERY_RATIOS = ((4, 5), (9, 10), (7, 8), (19, 20), (10, 11), (5, 6), (9, 10), (19, 20))
CONTROLS = ("recorded", "ordinary_mix", "new_room_excerpt", "moving_voice", "retimed_voice", "same_speed", "two_tones")


def _deliveries(room, different, voice, index):
    import numpy as np
    from scipy.signal import resample_poly

    from scripts.audio_channel_alignment_panel import delayed
    from scripts.audio_channel_difference_panel import rms

    starts = (16_802 + index * 629, 100_163 + index * 551)
    count = (12_480, 19_200, 24_960, 32_640)[(index + index // 4) % 4]
    ratio = DELIVERY_RATIOS[index]
    faster = index % 2
    longer_start, shorter_start = starts[1 - faster], starts[faster]

    def changed(signal, rate):
        result = signal.copy()
        copied = signal[longer_start:longer_start + count]
        converted = np.rint(resample_poly(copied.astype(float), *rate, axis=0))
        result[shorter_start:shorter_start + len(converted)] = converted
        lengths = (len(converted), count) if faster == 0 else (count, len(converted))
        return result, tuple((start, start + length) for start, length in zip(starts, lengths, strict=True))

    edited, truth = changed(room, ratio)
    same_speed, _ = changed(room, (1, 1))
    retimed_voice, _ = changed(voice, ratio)
    new_room = room.copy()
    new_room[shorter_start:shorter_start + count] = different[:count]
    # Use a constant headroom for all paired deliveries. Quantize only the final mix.
    relative = rms(room) / max(1., rms(voice))
    peak_voice = max(np.abs(voice).max(), np.abs(retimed_voice).max())
    peak_room = max(np.abs(sample).max() for sample in (room.astype(float), edited.astype(float), new_room.astype(float)))
    headroom = min(1., 29_000 / (peak_room + peak_voice * relative * 10 ** .9 * 1.26))

    def mixed(background, foreground, db, *, moving=False):
        offset = SHIFTS[index]
        left = foreground * (headroom * relative * 10 ** (db / 20))
        right = GAINS[index] * delayed(left, offset)
        if moving:
            right[len(right) // 2:] = delayed(left, -offset)[len(right) // 2:] / GAINS[index]
        values = np.rint(headroom * background + np.column_stack((left, right)))
        if np.abs(values).max() > 32767:
            raise ValueError("The staged stereo delivery clipped")
        return values.astype("<i2")

    time = np.arange(len(room)) / 48_000
    tone = rms(room) * (np.sin(2 * np.pi * 260 * time) + np.sin(2 * np.pi * 650 * time))
    yield "recorded", room, None
    yield "ordinary_mix", mixed(room, voice, 10), None
    yield "new_room_excerpt", mixed(new_room, voice, 10), None
    yield "moving_voice", mixed(room, voice, 10, moving=True), None
    yield "retimed_voice", mixed(room, retimed_voice, 10), None
    yield "same_speed", mixed(same_speed, voice, 10), None
    yield "two_tones", mixed(np.column_stack((tone, np.roll(tone, 144))), voice, 10), None
    for db in (10, 18):
        yield f"repeat_{db}db", mixed(edited, voice, db), {
            "ranges": truth, "faster": ("first", "second")[faster], "ratio": ratio,
        }


def _check(delivery, output):
    import numpy as np

    from app.detectors.audio.decode import decode_audio, decode_native_stereo
    from app.evidence import audio_native_alignment, audio_rate_repetition

    row, stereo = delivery
    file = output / row["delivery"]
    save_wav(file, stereo)
    sha = digest(file)
    decoded = decode_audio(file, source_sha256=sha)
    original = decode_native_stereo(file, decoded.mapping)
    if original != stereo.tobytes() or decoded.mapping["timeline"][0]["source_pts"] != 0:
        raise ValueError("The decoded stereo samples differ from the staged source")
    values = np.frombuffer(original, dtype="<i2").reshape(-1, 2)
    fitted, baseline_reason, baseline = audio_native_alignment.search(values)
    _, reason, matches = audio_rate_repetition.search(values, fitted)
    return row | {"input_sha256": sha, "baseline_reason": baseline_reason, "baseline": baseline,
                  "reason": reason, "matches": [asdict(match) for match in matches]}


def _located(pair, ranges):
    return all(iou(a, b) >= .5 for a, b in zip(pair, ranges, strict=True))


def counts(rows):
    groups = defaultdict(Counter)
    for row in rows:
        role, truth = row["role"], row["truth"]
        count = groups[role]
        count.update(files=1, assessed=row["reason"] is None, flagged=bool(row["matches"]),
                     baseline_assessed=row["baseline_reason"] is None, baseline_flagged=bool(row["baseline"]))
        if truth is None:
            continue

        baseline_hit = any(_located(pair, truth["ranges"]) for pair in row["baseline"])
        strict = [match for match in row["matches"] if _located((match["first"], match["second"]), truth["ranges"])
                  and match["faster"] == truth["faster"] and tuple(match["ratio"]) == tuple(truth["ratio"])]
        count.update(baseline_correct=baseline_hit, correct=bool(strict), combined_correct=baseline_hit or bool(strict),
                     additional=bool(strict) and not baseline_hit, incorrect_pairs=len(row["matches"]) - len(strict))
    return groups


def measure(room_root, voice_root, split, output):
    import numpy as np
    from scipy.signal import resample_poly

    from app.evidence import audio_rate_repetition

    output.mkdir(mode=0o700)
    scenes = [row for row in read(room_root / "recorded-sources.json") if row["scene"] in SCENES[split]]
    if {row["scene"] for row in scenes} != set(SCENES[split]):
        raise ValueError("Missing frozen stereo scene")
    meeting = MEETINGS[split]
    voice_source = voice_root / f"{meeting}.Array1-05.wav"
    if digest(voice_source) != HELD_VOICES[meeting]:
        raise ValueError("The foreground recording changed")
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for position, scene in enumerate(scenes):
            name = scene["scene"]
            files = [room_root / name / Path(item["member"]).name for item in scene["members"]]
            if any(digest(file) != record["sha256"] for file, record in zip(files, scene["members"], strict=True)):
                raise ValueError("A publisher background channel changed")
            jobs = []
            for index, start in enumerate(SCENE_STARTS):
                room = np.column_stack([segment(file, start, DURATION, 48_000) for file in files])
                different = np.column_stack([segment(file, start + 9, 32_640, 48_000) for file in files])
                speech_start = VOICE_STARTS[index] + 7 * position
                voice = resample_poly(segment(voice_source, speech_start, DURATION // 3, 16_000).astype(float), 3, 1)
                for role, stereo, truth in _deliveries(room, different, voice, index):
                    jobs.append(pool.submit(_check, ({"scene": name, "start_seconds": start, "meeting": meeting,
                                                     "speech_start_seconds": speech_start, "role": role, "truth": truth,
                                                     "delivery": f"{name}-{index}-{role}.wav"}, stereo), output))
            rows.extend(job.result() for job in jobs)
            print(name, len(rows), flush=True)
    result = counts(rows)
    save(output / "result.json.gz", {"schema": SCHEMA, "split": split, "rows": rows, "counts": result, "scenes": scenes,
                                    "foreground_sha256": {meeting: HELD_VOICES[meeting]}, "recipe_sha256": digest(Path(__file__)),
                                    "implementation_sha256": digest(Path(audio_rate_repetition.__file__))})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    trial = actions.add_parser("measure")
    trial.add_argument("room_root", type=Path)
    trial.add_argument("voice_root", type=Path)
    trial.add_argument("split", choices=SCENES)
    trial.add_argument("output", type=Path)
    actions.add_parser("review").add_argument("receipt", type=Path)
    args = vars(parser.parse_args())
    action = args.pop("action")
    if action == "measure":
        measure(**args)
    else:
        receipt = read(args["receipt"])
        result = counts(receipt["rows"])
        if receipt["schema"] != SCHEMA or result != receipt["counts"]:
            raise ValueError("The measured results do not recount")
        print(json.dumps(result, sort_keys=True))
