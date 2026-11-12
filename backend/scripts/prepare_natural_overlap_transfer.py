"""Reproduce the annotation-only AMI source selection for the stream-presence test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np

ANNOTATION_SHA256 = "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d"
EARLIER_ROOMS = frozenset({"ES2002a", "IS1003a", "ES2005a", "IS1006a", "ES2004a", "IS1009a"})
TICKS = 300


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def salt(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def track_words(xml: bytes) -> list[tuple[int, int]]:
    track = []
    for word in ElementTree.fromstring(xml):
        start, end = word.get("starttime"), word.get("endtime")
        if word.tag == "w" and not word.get("punc") and start is not None and end is not None:
            left, right = round(float(start) * 100), round(float(end) * 100)
            if 0 <= left < right:
                track.append((left, right))
    return track


def candidate_windows(tracks: list[list[tuple[int, int]]]) -> dict[int, list[int]]:
    if len(tracks) != 4:
        raise ValueError("The room reference needs four annotated participant tracks")
    last = max((end for track in tracks for _, end in track), default=0)
    active = np.zeros((4, last), dtype=bool)
    for index, track in enumerate(tracks):
        for start, end in track:
            active[index, start:end] = True
    candidates = {2: [], 3: []}
    for start in range(TICKS, min(last - TICKS, 150000), TICKS):
        clip = active[:, start:start + TICKS]
        duration = clip.sum(axis=1)
        overlap = clip.sum(axis=0)
        for expected, starts in candidates.items():
            if sum(duration >= 35) == expected and sum(duration > 8) == expected and sum(overlap == expected) >= 45:
                starts.append(start)
    return candidates


def make_plan(annotation: Path) -> dict:
    annotation_sha = digest(annotation)
    if annotation_sha != ANNOTATION_SHA256:
        raise ValueError("Exact AMI manual annotation release required")
    with ZipFile(annotation) as archive:
        meetings = ElementTree.fromstring(archive.read("corpusResources/meetings.xml"))
        people = {node.get("observation"): {child.get("global_name") for child in node if child.get("global_name")}
                  for node in meetings.iter() if node.get("observation")}
        previous = set().union(*(people[name] for name in EARLIER_ROOMS))
        rooms = []
        for name, participants in people.items():
            if name[:2] not in {"ES", "IS"} or len(participants) != 4 or participants & previous:
                continue
            try:
                tracks = [track_words(archive.read(f"words/{name}.{speaker}.words.xml")) for speaker in "ABCD"]
            except KeyError:
                continue
            options = candidate_windows(tracks)
            if min(map(len, options.values())) >= 4:
                rooms.append((salt(f"hard-room-natural/1:{name}"), name, participants, options))
    selected = []
    for _, name, participants, options in sorted(rooms):
        if participants & previous:
            continue
        previous.update(participants)
        cases = []
        for expected, times in options.items():
            starts = sorted(times, key=lambda time: salt(f"hard-room-window/1:{name}:{time}"))[:6]
            cases.extend({"expected_voices": expected, "begin_centiseconds": start} for start in starts)
        source = f"https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{name}/audio/{name}.Array1-01.wav"
        selected.append({"meeting": name, "annotation_sha256": annotation_sha, "url": source, "cases": cases,
                         "eligible": {str(expected): len(times) for expected, times in options.items()}})
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise ValueError("The declared source roster is incomplete")
    return {"schema": "chipmunk.natural-overlap-inputs/1", "source": "AMI CC BY 4.0", "rooms": selected}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("annotation", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    plan = make_plan(args.annotation)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
