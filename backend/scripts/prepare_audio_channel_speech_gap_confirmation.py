"""Freeze a participant-disjoint AMI panel for unchanged channel speech coverage."""

from __future__ import annotations

import argparse
import shutil
import wave
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts import audio_channel_speech_gap_transfer as study
from scripts.audio_ami_native_stereo_v291 import MIRROR
from scripts.audio_gtcrn_traffic import read, sha, write_new
from scripts.prepare_natural_overlap_transfer import ANNOTATION_SHA256, EARLIER_ROOMS, track_words

PRIOR_FREEZE = "bd84160f5483e7782f00b8f8975ca94d8d66d3f9deda6fa65b7d4fb20648ba9c"
EXPOSED = EARLIER_ROOMS | {"ES2006d", "ES2009d", "ES2015d"}
PREFIX_SAMPLES = 16_800_000
MICROPHONES = ("Array1-01", "Array1-05")


def annotation_candidates(tracks):
    prefix = min(max(b for track in tracks for _, b in track), PREFIX_SAMPLES // study.STEP)
    if prefix < 30_000:
        return None
    activity = np.zeros((4, prefix), dtype=bool)
    for index, track in enumerate(tracks):
        for a, b in track:
            activity[index, a:b] = True
    choices = defaultdict(list)
    for start in range(300, prefix - 299, 300):
        active = activity[:, start:start + 300].sum(axis=0)
        quiet, overlap, speaking = int((active == 0).sum()), int((active >= 2).sum()), int((active > 0).sum())
        if quiet >= 296:
            choices["quiet"].append(start)
        elif overlap <= 10 and speaking >= 90:
            choices["little_overlap"].append(start)
        elif overlap >= 75 and speaking >= 100:
            choices["overlap"].append(start)
    if any(len(choices[k]) < n for k, n in study.SELECT.items()):
        return None
    return activity.any(axis=0), choices


def choose(annotations):
    if sha(annotations) != ANNOTATION_SHA256:
        raise ValueError("AMI manual annotations changed")
    with ZipFile(annotations) as corpus:
        document = ElementTree.fromstring(corpus.read("corpusResources/meetings.xml"))
        rooms = {node.get("observation"): {child.get("global_name") for child in node if child.get("global_name")}
                 for node in document.iter() if node.get("observation")}
        previously_seen = set().union(*(rooms[name] for name in EXPOSED))
        selected = {}
        for room in sorted(rooms, key=lambda item: study.salt(("fresh-room", item))):
            people = rooms[room]
            if room[:2] not in {"ES", "IS"} or len(people) != 4 or people & previously_seen:
                continue
            try:
                tracks = [track_words(corpus.read(f"words/{room}.{person}.words.xml")) for person in "ABCD"]
            except KeyError:
                continue
            candidate = annotation_candidates(tracks)
            if candidate is None:
                continue
            active, choices = candidate
            selected[room] = [(condition, start, [[a * study.STEP, b * study.STEP] for a, b in
                                                   study.intervals(active[start:start + 300])])
                              for condition, count in study.SELECT.items()
                              for start in sorted(choices[condition], key=lambda value: study.salt((room, value)))[:count]]
            previously_seen.update(people)
            if len(selected) == 4:
                return selected
    raise ValueError("Insufficient participant-disjoint source groups")


def freeze(folder, annotations, previous):
    if sha(previous / "freeze.json") != PRIOR_FREEZE:
        raise ValueError("Earlier frozen input changed")
    original_lock, original_rows = study.frozen(previous)
    source_plan = read(folder / "source-plan.json")
    acquisition = read(folder / "acquisition.json")["files"]
    selected = choose(annotations)
    if (source_plan["rooms"], source_plan["prefix_cap_samples"], source_plan["candidates_per_room"],
            tuple(source_plan["physical_channels"]), source_plan["prior_freeze_sha256"]) != (
            list(selected), PREFIX_SAMPLES, study.SELECT, MICROPHONES, PRIOR_FREEZE):
        raise ValueError("Preacquisition plan differs from reproducible annotations")
    expected = {f"{room}.{mic}.wav" for room in selected for mic in MICROPHONES}
    if set(acquisition) != expected:
        raise ValueError("Incomplete physical microphone acquisition")
    rows, commitments = [], {"annotations": ANNOTATION_SHA256}
    (folder / "stereo").mkdir()
    (folder / "mono").mkdir()
    for index, (room, cases) in enumerate(selected.items(), 1):
        sources = []
        for mic in MICROPHONES:
            name = f"{room}.{mic}.wav"
            path = folder / "native" / name
            bound = acquisition[name]
            if (bound["url"], bound["bytes"], bound["sha256"]) != (
                    f"{MIRROR}/{room}/audio/{name}", path.stat().st_size, sha(path)):
                raise ValueError("Physical microphone acquisition differs")
            commitments[name] = bound["sha256"]
            sources.append(path)
        with wave.open(str(sources[0]), "rb") as left, wave.open(str(sources[1]), "rb") as right:
            if any((item.getnchannels(), item.getsampwidth(), item.getframerate()) != (1, 2, study.RATE) for item in (left, right)):
                raise ValueError("Physical microphone sample clock differs")
            for condition, start, labels in cases:
                source_start = start * study.STEP
                channels = []
                for item in (left, right):
                    item.setpos(source_start)
                    channel = np.frombuffer(item.readframes(study.SAMPLES), dtype="<i2")
                    if len(channel) != study.SAMPLES:
                        raise ValueError("Truncated physical microphone")
                    channels.append(channel)
                stereo = np.column_stack(channels)
                wav = folder / "stereo" / f"{len(rows):03}.wav"
                study.write_wave(wav, stereo)
                wav_hash = sha(wav)
                decoded = decode_audio(wav, source_sha256=wav_hash, max_seconds=3.)
                mono = np.rint(decoded.pcm * 32768).astype("<i2").tobytes()
                if (len(mono) != 2 * study.SAMPLES or decoded.mapping["source_channels"] != 2 or decoded.mapping["timeline"] !=
                        [{"start_sample": 0, "end_sample": study.SAMPLES, "source_pts": 0}]):
                    raise ValueError("Canonical downmix changed the input clock")
                (folder / "mono" / f"{len(rows):03}.pcm").write_bytes(mono)
                rows.append({"id": len(rows), "panel": "native", "family": "genuine", "group": f"meeting:{index}",
                             "condition": condition, "source_start_sample": source_start, "known_word_samples": labels,
                             "wav_sha256": wav_hash, "stereo_sha256": sha(stereo.tobytes()), "mono_sha256": sha(mono)})
    for row in original_rows:
        if row["panel"] != "controlled":
            continue
        if row["id"] != len(rows):
            raise ValueError("Controlled view position changed")
        study.pcm(previous, row)
        for directory, extension in (("stereo", "wav"), ("mono", "pcm")):
            rel = Path(directory) / f"{len(rows):03}.{extension}"
            shutil.copyfile(previous / rel, folder / rel)
        rows.append(row)
    write_new(folder / "roster.json", rows)
    write_new(folder / "freeze.json", {**study.binding(), "schema": "chipmunk.audio-channel-speech-gap-freeze/1",
                                       "views": len(rows), "sources": commitments, "roster_sha256": sha(folder / "roster.json"),
                                       "controlled_journal_sha256": original_lock["controlled_journal_sha256"],
                                       "confirmation": {"previous_freeze_sha256": PRIOR_FREEZE, "source_plan_sha256": sha(folder / "source-plan.json"),
                                                        "acquisition_sha256": sha(folder / "acquisition.json"),
                                                        "preparation_code_sha256": sha(Path(__file__)), "replayed_controls": 200}})
    print(f"Frozen {len(selected)} participant-disjoint meetings and {len(rows)} total views", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    freeze(args.folder, args.annotation, args.previous)
