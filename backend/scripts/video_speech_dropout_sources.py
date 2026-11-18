# SPDX-License-Identifier: Apache-2.0
"""Reserve new speaker and room recordings for the speech-activity dropout check."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from scripts import video_covariance_sources as stereo
from scripts.video_silent_mouth_clock_panel import CREMA_REV, LICENSE_SHA, SOURCE_INDEX_SHA, sha

PANEL = "speech-dropout-activity/1"
PREVIOUS_PLAN = "video-stereo-covariance-plan-2026-11-17.json.gz"
PREVIOUS_SOURCES = "video-stereo-covariance-sources-2026-11-17.json.gz"
PREVIOUS_INPUTS = "video-stereo-covariance-inputs-2026-11-17.json.gz"


def order(value):
    return sha(f"{PANEL}\0{value}".encode())


def reserve(index, license_file, archive, panels, output):
    if stereo.digest(index) != SOURCE_INDEX_SHA or stereo.digest(license_file) != LICENSE_SHA:
        raise ValueError("CREMA-D index or license differs from its pinned revision")
    stereo.check_archive(archive)
    previous = stereo.read(panels / PREVIOUS_PLAN)
    inventory = stereo.read(panels / PREVIOUS_SOURCES)
    inputs = stereo.read(panels / PREVIOUS_INPUTS)
    if (inventory["plan_sha256"] != stereo.digest(panels / PREVIOUS_PLAN)
            or inputs["source_sha256"] != stereo.digest(panels / PREVIOUS_SOURCES)):
        raise ValueError("The previous held input inventory does not match its recorded reservation")

    previous_stereo = [item for name in stereo.PRIOR[:2] for item in stereo.read(panels / name)["sources"]]
    speakers = set(previous["prior_speakers_excluded"]) | {item["speaker"] for item in inventory["entries"]}
    hashes = set(previous["prior_source_hashes_excluded"]) | {item["sha256"] for item in inventory["entries"]}
    room_names = {item["room"] for item in previous_stereo + inputs["sources"]}
    room_hashes = {item["room_sha256"] for item in previous_stereo + inputs["sources"]}
    known_speakers = {item["speaker"] for item in stereo.read(panels / stereo.PRIOR[2])["rows"]}
    known_speakers.update(item["speaker"] for item in stereo.read(panels / stereo.PRIOR[3])["clocks"] if "speaker" in item)
    options = defaultdict(list)
    with index.open(newline="") as stream:
        for row in csv.DictReader(stream):
            filename = row["Filename"]
            if filename.endswith("_NEU_XX"):
                speaker = sha(f"CREMA-D:{filename.split('_')[0]}".encode())
                if speaker not in speakers:
                    options[speaker].append(filename + ".flv")

    development = previous["development"]
    categories = [item["room"][:-6] for item in development]
    selected = sorted(options, key=order)[:stereo.HELD_SPEAKERS]
    room_files = [name + "10.wav" for name in categories]
    room_files += [name + "07.wav" for name in sorted(categories, key=order)[:2]]
    with ZipFile(archive) as bundle:
        rooms = [{"name": name, "sha256": sha(bundle.read(name))} for name in room_files]
    if (len(selected) != stereo.HELD_SPEAKERS or len(set(categories)) != 10 or room_names.intersection(room_files)
            or room_hashes.intersection(item["sha256"] for item in rooms)
            or len({item["sha256"] for item in rooms}) != stereo.HELD_SPEAKERS):
        raise ValueError("A new held speaker or room recording overlaps the previous stereo comparisons")

    held = [{"speaker": speaker, "options": sorted(options[speaker], key=order), "room": room,
             "earlier_project_speaker": speaker in known_speakers}
            for speaker, room in zip(selected, rooms, strict=True)]
    result = {
        "schema": stereo.SCHEMA, "evaluation": PANEL, "publisher_revision": CREMA_REV,
        "source_index_sha256": SOURCE_INDEX_SHA, "license_sha256": LICENSE_SHA, "archive_sha1": stereo.ARCHIVE_SHA1,
        "compare": {name: stereo.digest(panels / name)
                    for name in (*stereo.PRIOR, PREVIOUS_PLAN, PREVIOUS_SOURCES, PREVIOUS_INPUTS)},
        "development": development, "prior_speakers_excluded": sorted(speakers),
        "prior_source_hashes_excluded": sorted(hashes), "held": held,
        "selection": "Order neutral clips and speakers by the new panel hash. Use twelve eligible speakers and "
                     "room take 10 in all ten categories plus take 07 in the first two hashed categories. "
                     "Apply the earlier seven controlled treatments at both levels and in both encodings.",
    }
    stereo.save(output, result)
    print(f"Reserved {len(held)} speakers and rooms. "
          f"{sum(item['earlier_project_speaker'] for item in held)} speakers appeared in earlier timing studies.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("index", "license_file", "archive", "panels", "output"):
        parser.add_argument(field, type=Path)
    reserve(**vars(parser.parse_args()))
