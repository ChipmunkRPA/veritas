"""Compare soundtrack speech activity and periodicity on separate SINE readers."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from app.evidence import mouth_hold_sound as binding
from scripts import audio_sine_resynthesis_transfer as sine
from scripts import audio_vocal_event_mixtures as io
from scripts.video_mouth_hold_activity_transfer import MODEL_SHA256

PARENT = io.PUBLIC / "audio-narrow-flag-transfer-2026-11-14.json"
PRIOR = io.PUBLIC / "video-mouth-hold-noise-transfer-2026-11-14.json"
PRIOR_SOURCES = io.PUBLIC / "audio-sine-resynthesis-transfer-2026-11-14.json"
SCHEMA = "chipmunk.mouth-hold-speech-context-transfer/1"
SCENES = ("rain", "train", "competing_speech")
WINDOW = 6400
SHA = re.compile("[0-9a-f]{64}")


def interval(pair):
    center = sum(pair["truth"]) // 2
    start = min(max(0, center - WINDOW // 2), 6 * binding.RATE - WINDOW)
    return [start, start + WINDOW]


def summarize(rows):
    cells = {}
    for row in rows:
        key = row["role"] + "/" + row["scene"]
        counter = cells.setdefault(key, Counter())
        measurement = row["sound"]
        counter.update(windows=1, periodic=measurement["sustained_periodicity"],
                       speech=measurement["speech_activity"]["samples"] >= binding.MIN_ACTIVITY_SAMPLES)
    return dict(sorted((key, dict(value)) for key, value in cells.items()))


def review(doc):
    parent = io.read(PARENT)
    old = io.read(PRIOR)["plan"]["groups"]
    old_readers = {item["id"]: item["reader"] for item in io.read(PRIOR_SOURCES)["freeze"]["pairs"]}
    pairs = parent["freeze"]["pairs"]
    if (doc["schema"] != SCHEMA or doc["sources"] != [io.sha(path) for path in (PARENT, PRIOR, PRIOR_SOURCES)]
            or doc["activity_sha256"] != MODEL_SHA256
            or {pair["reader"] for pair in pairs} & {old_readers[row["source_id"]] for row in old}):
        raise ValueError("Scope differs from the frozen, reader-disjoint transfer")
    roster = {(pair["id"], role, scene): interval(pair)
              for pair in pairs for role in ("original", "variant", "background")
              for scene in (("clean", *SCENES) if role != "background" else SCENES)}
    for row in doc["rows"]:
        key = tuple(row[field] for field in ("id", "role", "scene"))
        sound = row["sound"]
        start, end = sound["samples"]
        possible = len(binding.block_starts(start, end))
        if (roster.pop(key, None) != [start, end] or sound["schema"] != binding.ACTIVITY_SCHEMA
                or not SHA.fullmatch(row["pcm_sha256"]) or sound["source_pts"] != [start, end]
                or sound["speech_activity"]["context"] != [0, 6 * binding.RATE]
                or type(sound["speech_activity"]["samples"]) is not int
                or not 0 <= sound["speech_activity"]["samples"] <= WINDOW or sound["blocks"] != possible
                or not 0 <= sound["periodic_blocks"] <= possible
                or sound["sustained_periodicity"] != (3 * sound["periodic_blocks"] >= 2 * possible)):
            raise ValueError("Source clocks, speech measurements or periodic block counts differ")
    totals = summarize(doc["rows"])
    if roster or totals != doc["results"]:
        raise ValueError("Missing comparison rows or changed counts")
    return totals


def sources(bank, sounds):
    import numpy as np

    from scripts.audio_reference_variable_window import read_noise

    parent = io.read(PARENT)
    frozen = parent["freeze"]
    if io.read(bank / "freeze.json") != frozen:
        raise ValueError("The retained SINE source roster differs")
    pins = {(r["group"], r["condition"], r["side"]): r["pcm_sha256"] for r in parent["records"]}
    views = {}
    for row, pcm in sine.native_rows(bank, frozen):
        key = (row["id"], row["condition"], row["side"])
        if io.digest(pcm.tobytes()) != pins[key]:
            raise ValueError("The original or decoded comparison input differs from the verified study")
        views[key] = pcm
    environments = read_noise(sounds)
    backgrounds = {row["room"]: views[index, "competing_speech", row["room"]]
                   for index, row in enumerate(frozen["backgrounds"])}
    for pair in frozen["pairs"]:
        i = pair["id"]
        first, second = (views[i, "clean", s] for s in ("original", "variant"))
        yield pair, "original", "clean", first
        yield pair, "variant", "clean", second
        for scene, chunks in environments.items():
            background = chunks[i % len(chunks)].astype(np.float64)
            added = background * np.linalg.norm(first) / np.linalg.norm(background) * 10 ** .25
            scale = min(1., 32000 / max(np.max(np.abs(x)) for x in (first + added, second + added, added)))
            for role, values in (("original", first + added), ("variant", second + added), ("background", added)):
                yield pair, role, scene, np.rint(values * scale).astype("<i2")
        yield pair, "background", "competing_speech", backgrounds[pair["room"]]
        for role in ("original", "variant"):
            yield pair, role, "competing_speech", views[i, "competing_speech", role]


def run(bank, sounds, checkpoint):
    import torch

    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.mouth_hold_sound import measure

    if io.sha(checkpoint) != MODEL_SHA256:
        raise ValueError("The local speech-activity weights differ")
    torch.set_num_threads(1)
    activity = SpeechActivity(checkpoint)
    rows = []
    for pair, role, scene, pcm in sources(bank, sounds):
        mapping = {"samples": len(pcm), "timeline_time_base": "1/16000",
                   "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}]}
        measured = measure(pcm.astype("float32") / 32768, mapping,
                           {"schema": "chipmunk.video-mouth-local-hold/1", "status": "matched",
                            "match": {"frames": [0, 1]}, "pts": interval(pair)},
                           "1/16000", schema=binding.ACTIVITY_SCHEMA, activity=activity)
        rows.append({"id": pair["id"], "role": role, "scene": scene, "pcm_sha256": io.digest(pcm.tobytes()), "sound": measured})
    return {"schema": SCHEMA, "sources": [io.sha(path) for path in (PARENT, PRIOR, PRIOR_SOURCES)], "activity_sha256": MODEL_SHA256,
            "rows": rows, "results": summarize(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--sounds", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.bank:
        if not (args.sounds and args.checkpoint) or args.receipt.exists():
            parser.error("Native replay requires both audio inputs, checkpoint and a new receipt path")
        doc = run(args.bank, args.sounds, args.checkpoint)
        review(doc)
        io.save(args.receipt, doc)
    else:
        doc = io.read(args.receipt)
    print(json.dumps(review(doc), indent=2))


if __name__ == "__main__":
    main()
