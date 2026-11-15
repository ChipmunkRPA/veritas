# SPDX-License-Identifier: MIT
"""Prepare small, participant-separated original ICSI meeting test excerpts."""

from __future__ import annotations

import argparse
import itertools
import re
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import accumulate
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

from scripts.audio_source_isolation_diagnostic import check, digest, path_sha, read, write

SCHEMA = "chipmunk.audio-icsi-overlap-intake/1"
ANNOTATIONS = "https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_core_NXT.zip"
ANNOTATION_SHA256 = "cf4860245b9ca9c9ed11a66e5a74cfea2a99a686a52f7580aa0a6bc2225895a9"
REPOSITORY = "aaronyee/icsi-meeting-corpus"
REVISION = "ea358f7db63cd68c59b22d7990918f326480215f"
RATE, TICK, WINDOW = 16000, 160, 600
SAMPLES = WINDOW * TICK
METHOD = {
    "room_groups": "Choose a deterministic three-meeting combination with four or more declared participants in each "
                   "and no declared participant shared among the three. Use published words and dialogue-act participant IDs only.",
    "geometry": "Use nonoverlapping 6 s windows on a 10 ms source grid, starting at 12 s and ending within the hour. "
                "Require 4.5 s with at least one speaker. Select four windows with at least 1.2 s of overlapping words, "
                "plus four with at most 0.1 s, per meeting. Select using fixed SHA-256 order before audio access.",
    "native": "Use exact original 16 kHz mono 16-bit PCM from the pinned interaction WAV. Retrieve only the header "
              "and selected sample ranges. Keep known widths and clocks without resampling. No source inference.",
}


def _rank(value):
    return digest(f"{SCHEMA}/{value}".encode())


def _words(root):
    for word in root:
        start, end = word.get("starttime"), word.get("endtime")
        if word.tag != "w" or not start or not end:
            continue
        a, b = round(float(start) * 100), round(float(end) * 100)
        if 0 <= a < b:
            yield a, b


def _meeting_tracks(annotation):
    from app.bench.independent_locations import intervals

    meetings = defaultdict(dict)
    with ZipFile(annotation) as archive:
        names = set(archive.namelist())
        for name in sorted(names):
            prefix = re.fullmatch(r"ICSI/Words/(B[a-z]+\d+)\.([A-Z])\.words\.xml", name)
            if not prefix:
                continue
            meeting, agent = prefix.groups()
            text = ElementTree.fromstring(archive.read(name))
            spans = list(_words(text))
            if not spans:
                continue
            label = f"ICSI/DialogueActs/{meeting}.{agent}.dialogue-acts.xml"
            acts = ElementTree.fromstring(archive.read(label)) if label in names else ()
            ids = {p.get("participant") for p in acts if p.get("participant")}
            if len(ids) == 1:
                participant, = ids
                meetings[meeting].setdefault(participant, []).extend(spans)
            else:
                # A room with unbound timed words cannot establish participant separation.
                meetings[meeting][None] = []
    return {room: {participant: intervals(spans) for participant, spans in tracks.items()}
            for room, tracks in meetings.items() if None not in tracks and len(tracks) >= 4}


def _positions(tracks):
    count = min(360_000, max(end for spans in tracks.values() for _, end in spans))
    change = [0] * (count + 1)
    for spans in tracks.values():
        for a, b in spans:
            if a < count:
                change[a] += 1
                change[min(b, count)] -= 1
    concurrent = list(accumulate(change[:-1]))
    result = defaultdict(list)
    for tick in range(2 * WINDOW, count - WINDOW + 1, WINDOW):
        counts = concurrent[tick:tick + WINDOW]
        active = sum(bool(value) for value in counts)
        overlap = sum(value > 1 for value in counts)
        if active >= 450 and (overlap >= 120 or overlap <= 10):
            edges = [i for i, (a, b) in enumerate(itertools.pairwise([False, *(value > 1 for value in counts), False])) if a != b]
            result["strong" if overlap >= 120 else "minimal"].append({"start_sample": tick * TICK,
                "active_samples": active * TICK, "overlap_samples": overlap * TICK,
                "overlap_ranges": [[a*TICK, b*TICK] for a, b in zip(edges[::2], edges[1::2], strict=True)]})
    return result


def plan(annotation):
    check(path_sha(annotation) == ANNOTATION_SHA256, "ICSI core annotations differ")
    rooms = _meeting_tracks(annotation)
    positions = {room: _positions(tracks) for room, tracks in rooms.items()}
    eligible = sorted(room for room, windows in positions.items() if min(len(windows["strong"]), len(windows["minimal"])) >= 4)
    triples = (r for r in itertools.combinations(eligible, 3) if all(not rooms[a].keys() & rooms[b].keys()
                                                                  for a, b in itertools.combinations(r, 2)))
    selected = min(triples, key=_rank)
    cases = []
    for room in selected:
        for condition in ("strong", "minimal"):
            windows = sorted(positions[room][condition], key=lambda row: _rank((room, condition, row["start_sample"])))[:4]
            cases.extend(row | {"room": room, "condition": condition} for row in windows)
    return {"schema": SCHEMA, "method": METHOD, "annotations_url": ANNOTATIONS, "annotations_sha256": ANNOTATION_SHA256,
            "repository": REPOSITORY, "revision": REVISION, "sample_count": SAMPLES, "sample_rate": RATE,
            "known_speaker_counts": {room: len(rooms[room]) for room in selected}, "cases": cases}


def _range(room, start, count, total=None):
    check(re.fullmatch(r"B[a-z]+\d+", room) is not None, "Unexpected meeting ID")
    stop = start + count - 1
    url = (f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/Signals/{room}/{room}.interaction.wav"
           f"?download=true&chipmunk-range={start}")
    headers = {"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity", "User-Agent": "Chipmunk-Verify-evaluation/1"}
    with urlopen(Request(url, headers=headers), timeout=28) as reply:
        found = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", reply.headers.get("Content-Range", ""))
        check(found is not None and reply.status == 206 and (int(found[1]), int(found[2])) == (start, stop)
              and total in (None, int(found[3])) and int(found[3]) > stop
              and reply.headers.get("Content-Encoding", "identity") == "identity", "Mirror returned a different byte range")
        body = reply.read(count + 1)
    check(len(body) == count, "Downloaded range is incomplete")
    return body, int(found[3])


def acquire(root, annotation):
    frozen = read(root / "plan.json")
    check(frozen == plan(annotation), "Annotation plan changed before acquisition")
    captures = {}
    for room in frozen["known_speaker_counts"]:
        raw, total = _range(room, 0, 44)
        check(struct.unpack("<4sI4s4sIHHIIHH4sI", raw) ==
              (b"RIFF", total-8, b"WAVE", b"fmt ", 16, 1, 1, RATE, 2*RATE, 2, 16, b"data", total-44),
              "Expected native mono 16-bit PCM and a continuous clock")
        captures[room] = {"total_bytes": total, "header_sha256": digest(raw), "samples": (total-44)//2}

    def capture(case):
        room, start = case["room"], case["start_sample"]
        body, _ = _range(room, 44 + 2*start, 2*SAMPLES, captures[room]["total_bytes"])
        pcm = root / f"{room}-{start}.pcm"
        try:
            with pcm.open("xb") as output:
                output.write(body)
        except FileExistsError:
            check(pcm.read_bytes() == body, "An existing excerpt differs from the pinned source")
        return case | {"pcm_sha256": digest(body)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        cases = list(pool.map(capture, frozen["cases"]))
    return {"plan": frozen, "captures": captures, "cases": cases, "detection_attempts": 0}


def replay(root, annotation, receipt):
    frozen = plan(annotation)
    check(receipt["plan"] == frozen, "ICSI annotation plan differs")
    check(receipt["captures"].keys() == frozen["known_speaker_counts"].keys()
          and all(item["total_bytes"] == 44 + 2 * item["samples"] for item in receipt["captures"].values()),
          "Unexpected ICSI native capture extent")
    for expected, checked in zip(frozen["cases"], receipt["cases"], strict=True):
        check(expected == {k: v for k, v in checked.items() if k != "pcm_sha256"}, "Unexpected sample labels")
        room, start = expected["room"], expected["start_sample"]
        raw = (root / f"{room}-{start}.pcm").read_bytes()
        check(len(raw) == 2*SAMPLES and digest(raw) == checked["pcm_sha256"]
              and start + SAMPLES <= receipt["captures"][room]["samples"], "Original PCM or sample extent differs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "acquire", "replay"))
    parser.add_argument("root", type=Path)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()
    if args.phase == "plan":
        prepared = plan(args.annotations)
        write(args.root / "plan.json", prepared)
    elif args.phase == "acquire":
        receipt = acquire(args.root, args.annotations)
        replay(args.root, args.annotations, receipt)
        write(args.root / "receipt.json", receipt)
        prepared = receipt["plan"]
    else:
        receipt = read(args.root / "receipt.json")
        replay(args.root, args.annotations, receipt)
        prepared = receipt["plan"]
    print(f"{args.phase}: {len(prepared['cases'])} native-clock excerpts across {len(prepared['known_speaker_counts'])} meetings")
