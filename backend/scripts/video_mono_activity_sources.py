# SPDX-License-Identifier: Apache-2.0
"""Reserve fresh publisher inputs for the mono speech-activity comparison."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from scripts import video_covariance_sources as sources
from scripts import video_speech_dropout_sources as previous
from scripts.video_silent_mouth_clock_panel import CREMA_REV, LICENSE_SHA, SOURCE_INDEX_SHA, sha

SCHEMA = "chipmunk.video-mono-activity-sources/1"
PREFIX = "video-speech-activity-gap-"
FILES = tuple(f"{PREFIX}{role}-2026-11-17.json.gz" for role in ("plan", "sources", "inputs", "development"))


def order(value):
    return sha(f"{SCHEMA}\0{value}".encode())


def reserve(index, license_file, archive, panels, output):
    if sources.digest(index) != SOURCE_INDEX_SHA or sources.digest(license_file) != LICENSE_SHA:
        raise ValueError("CREMA-D source metadata or reuse terms differ from the pinned revision")
    sources.check_archive(archive)
    plan, inventory, inputs, calibration = [sources.read(panels / name) for name in FILES]
    if (inventory["plan_sha256"] != sources.digest(panels / FILES[0])
            or inputs["source_sha256"] != sources.digest(panels / FILES[1])):
        raise ValueError("The previous speech-activity source chain differs")
    actors = set(plan["prior_speakers_excluded"]) | {row["speaker"] for row in inventory["entries"]}
    used_files = set(plan["prior_source_hashes_excluded"]) | {row["sha256"] for row in inventory["entries"]}
    earlier = [row for name in (*sources.PRIOR[:2], previous.PREVIOUS_INPUTS, FILES[2])
               for row in sources.read(panels / name)["sources"]]
    room_names = {row["room"] for row in earlier}
    room_digests = {row["room_sha256"] for row in earlier}
    timing = sources.read(panels / sources.PRIOR[2])
    dropouts = sources.read(panels / sources.PRIOR[3])
    older_timing = {row["speaker"] for row in timing["rows"]} | {
        row["speaker"] for row in dropouts["clocks"] if "speaker" in row
    }
    options = defaultdict(list)
    with index.open(newline="") as stream:
        for row in csv.DictReader(stream):
            file = row["Filename"]
            if file.endswith("_NEU_XX"):
                speaker = sha(f"CREMA-D:{file.split('_')[0]}".encode())
                if speaker not in actors:
                    options[speaker].append(file + ".flv")
    selected = sorted(options, key=order)[:sources.HELD_SPEAKERS]
    categories = sorted({row["room"][:-6] for row in earlier}, key=order)
    named_rooms = [name + "06.wav" for name in categories] + [name + "02.wav" for name in categories[:2]]
    with ZipFile(archive) as bundle:
        rooms = [{"name": name, "sha256": sha(bundle.read(name))} for name in named_rooms]
    if (len(selected) != sources.HELD_SPEAKERS or len(categories) != 10
            or room_names.intersection(named_rooms) or room_digests.intersection(room["sha256"] for room in rooms)):
        raise ValueError("Fresh speakers and room recordings must be disjoint from the preceding activity studies")
    held = [{"speaker": speaker, "options": sorted(options[speaker], key=order), "room": room,
             "earlier_project_speaker": speaker in older_timing} for speaker, room in zip(selected, rooms, strict=True)]
    sources.save(output, {"schema": sources.SCHEMA, "evaluation": SCHEMA, "publisher_revision": CREMA_REV,
                          "source_index_sha256": SOURCE_INDEX_SHA, "license_sha256": LICENSE_SHA,
                          "archive_sha1": sources.ARCHIVE_SHA1, "held": held, "prior_source_hashes_excluded": sorted(used_files),
                          "prior_speakers_excluded": sorted(actors),
                          "comparison_file_sha256": {name: sources.digest(panels / name) for name in FILES},
                          "operating_points": {key: arm["selected"] for key, arm in calibration["decisions"].items()}})
    print(f"Reserved {len(held)} distinct speakers and rooms.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "license_file", "archive", "panels", "output"):
        cli.add_argument(name, type=Path)
    reserve(**vars(cli.parse_args()))
