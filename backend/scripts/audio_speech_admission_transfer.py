"""Measure voice-window admission on fresh, annotated meeting excerpts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import audio_locate_competing_speech as source
from scripts import audio_paired_loud_background as io

SCHEMA = "chipmunk.audio-speech-admission-transfer/1"
BASE = "e3d0d2b"
CHECKPOINT = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
METHOD = {
    "sampling": "Four participant-disjoint AMI meeting cohorts unseen in the repository at freeze. Six excerpts per room and stratum at least eight seconds apart, selected by fixed hashes from mixer-independent manual word timings.",
    "strata": "High has at least 75% annotated speech, low has at most 25%. Two boundary strata bracket 50% with 40–49% and 50–59%.",
    "comparison": "Run the existing three-second speech-activity gate at original volume and on a clipped-sample-free +6 dB copy. The union admits if either does. The intersection requires both. Keep the established half-window rule in both views.",
    "interpretation": "Manual words are only a proxy for audible speech. This test measures window admission, not AI classification. Gain copies do not enter analysis.",
}
STRATA = {"high": (225, 300), "low": (0, 75), "near_below": (120, 149), "near_above": (150, 179)}
ARMS = ("original", "gain", "union", "intersection")


def freeze(folder, annotations):
    import subprocess
    from xml.etree import ElementTree
    from zipfile import ZipFile

    from scripts.audio_harmonic_overlap_transfer import room_activity

    checksum = io.read(source.PANEL)["freeze"]["annotations_sha256"]
    io.check(io.digest(annotations) == checksum, "Manual timing archive differs")
    grep = subprocess.run(["git", "grep", "-h", "-o", "-E", "(ES|IS|TS)[0-9]{4}[a-d]", BASE, "--", "docs"],
                          check=True, cwd=io.ROOT, capture_output=True, text=True)
    earlier = set(grep.stdout.split())
    rows, accepted = [], []
    with ZipFile(annotations) as archive:
        tree = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {item.get("observation"): {node.get("global_name") for node in item if node.get("global_name")}
                  for item in tree.iter() if item.get("observation")}
        occupied = set().union(*(people[room] for room in earlier))
        series = {room[:-1] for room in earlier}
        for room in sorted(people, key=lambda name: source.sha(f"{SCHEMA}/room/{name}".encode())):
            members = people[room]
            if room[:-1] in series or len(members) != 4 or members & occupied:
                continue
            try:
                activity = room_activity(archive, room)
            except KeyError:
                continue
            starts, selected = [], []
            for label, (low, high) in STRATA.items():
                choices = []
                for start in range(400, len(activity) - 300, 400):
                    words = int((activity[start:start + 300] > 0).sum())
                    if low <= words <= high:
                        choices.append((source.sha(f"{SCHEMA}/{room}/{label}/{start}".encode()), start, words))
                count = 0
                for _, start, words in sorted(choices):
                    if any(abs(start - prior) < 800 for prior in starts):
                        continue
                    starts.append(start)
                    selected.append({"room": room, "start_sample": start * 160, "stratum": label, "word_cells": words})
                    count += 1
                    if count == 6:
                        break
                if count != 6:
                    break
            if len(selected) != 24:
                continue
            rows.extend(selected)
            accepted.append(room)
            occupied.update(members)
            series.add(room[:-1])
            if len(accepted) == 4:
                break
    io.check(len(accepted) == 4, "Four fresh rooms are required")
    folder.mkdir(mode=0o700)
    io.write(folder / "freeze.json", {"schema": SCHEMA, "method": METHOD, "base": BASE, "annotations_sha256": checksum,
        "study_sha256": io.digest(Path(__file__)), "runner_sha256": io.digest(Path(source.__file__)),
        "rooms": accepted, "cases": [row | {"id": index} for index, row in enumerate(rows)]})
    print("Frozen rooms:", ", ".join(accepted), flush=True)


def measure(folder, checkpoint):
    import numpy as np
    import torch

    from app.detectors.audio.upgraded import SpeechActivity

    lock, inputs = (io.read(folder / name) for name in ("freeze.json", "input.json"))
    io.check(lock["study_sha256"] == io.digest(Path(__file__)) and io.digest(checkpoint) == CHECKPOINT
             and inputs["freeze_sha256"] == io.digest(folder / "freeze.json"), "The freeze or weights differ")
    torch.set_num_threads(2)
    model = SpeechActivity(checkpoint)
    rows = []
    for case, item in zip(lock["cases"], inputs["rows"], strict=True):
        io.check(case["id"] == item["id"], "Source sample order differs")
        values = None
        if item["pcm_sha256"] is not None:
            raw = (folder / "pcm" / f"{case['id']:03}.pcm").read_bytes()
            io.check(len(raw) == 96_000 and hashlib.sha256(raw).hexdigest() == item["pcm_sha256"], "Original samples differ")
            pcm = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            multiplier = min(10**(6/20), (32767 / 32768) / max(float(np.max(np.abs(pcm))), 1e-9))
            values = [sum(round(b * source.RATE) - round(a * source.RATE) for a, b in model.spans(view))
                      for view in (pcm, pcm * multiplier)]
        rows.append(case | {"pcm_sha256": item["pcm_sha256"], "admitted_speech_samples": values})
    receipt = {"freeze": lock, "captures": inputs["captures"], "checkpoint_sha256": CHECKPOINT, "rows": rows}
    io.write(folder / "receipt.json", receipt | {"summary": summarize(receipt)})


def summarize(receipt):
    lock = receipt["freeze"]
    io.check(lock["method"] == METHOD and lock["schema"] == SCHEMA and lock["base"] == BASE
             and receipt["checkpoint_sha256"] == CHECKPOINT, "Speech admission method differs")
    out = {label: Counter() for label in STRATA}
    rooms = {room: {label: Counter() for label in STRATA} for room in lock["rooms"]}
    io.check(len(rooms) == 4 and len(lock["cases"]) == 96 == len(receipt["rows"]), "Missing fresh rooms or windows")
    for candidate, row in zip(lock["cases"], receipt["rows"], strict=True):
        io.check(all(row[key] == value for key, value in candidate.items()), "Source geometry differs from the freeze")
        low, high = STRATA[row["stratum"]]
        io.check(low <= row["word_cells"] <= high and row["start_sample"] % 64000 == 0, "Wrong annotation stratum or clock")
        for count in (out[row["stratum"]], rooms[row["room"]][row["stratum"]]):
            count["planned"] += 1
            values = row["admitted_speech_samples"]
            io.check((values is None) == (row["pcm_sha256"] is None), "Unbound coverage response")
            if values is None:
                count["unavailable"] += 1
                continue
            io.check(len(values) == 2 and all(type(v) is int and 0 <= v <= 48000 for v in values), "Invalid speech counts")
            original, gain = (samples >= 24000 for samples in values)
            count.update(dict(zip(ARMS, (original, gain, original or gain, original and gain), strict=True)))
    io.check(all(count["planned"] == 6 for room in rooms.values() for count in room.values()), "Incomplete room strata")
    return {"total": out, "rooms": rooms}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "acquire", "measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--annotations", default=source.ANNOTATIONS, type=Path)
    parser.add_argument("--checkpoint", default=io.ROOT / "models/weights/audio-upgrade/silero_vad.jit", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.path, args.annotations)
    elif args.phase == "acquire":
        source.acquire(args.path)
    elif args.phase == "measure":
        measure(args.path, args.checkpoint)
    else:
        receipt = io.read(args.path)
        summary = summarize(receipt)
        io.check(summary == receipt["summary"], "Public speech admission results do not replay")
        print(json.dumps(summary["total"], indent=2))


if __name__ == "__main__":
    main()
