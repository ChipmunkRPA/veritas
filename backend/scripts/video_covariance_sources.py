# SPDX-License-Identifier: Apache-2.0
"""Reserve and rebuild source-disjoint stereo speech-dropout comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from zipfile import ZipFile

from scripts.acquire_video_common_sources_v91 import lfs_pointer
from scripts.video_lip_multishot_panel import RATE, digest, ffmpeg, read, save
from scripts.video_shared_sound_panel import ARCHIVE_SHA1, STREAM_HASH, acoustic_sources, observe
from scripts.video_silent_mouth_clock_panel import (
    CREMA,
    CREMA_REV,
    FRAME_PERIOD,
    GEOMETRY_SHA,
    LICENSE_SHA,
    SOURCE_INDEX_SHA,
    fetch,
    fit_sound,
    sha,
    source_sound,
)

SCHEMA = "chipmunk.video-stereo-covariance-sources/1"
HELD_SPEAKERS = 12
MAX_MEDIA_BYTES = 2 * 1024**2
PRIOR = (
    "video-shared-sound-development-2026-11-17.json.gz",
    "video-shared-sound-held-2026-11-17.json.gz",
    "video-lip-delivery-native-2026-11-16.json.gz",
    "video-sound-dropouts-30fps-2026-11-17.json.gz",
)


def order(value):
    return sha(f"{SCHEMA}\0{value}".encode())


def check_archive(archive):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha1").hexdigest() != ARCHIVE_SHA1:
            raise ValueError("DCASE archive differs from the publisher checksum")


def plan(index, license_file, archive, panels, output):
    if digest(index) != SOURCE_INDEX_SHA or digest(license_file) != LICENSE_SHA:
        raise ValueError("CREMA-D metadata differs from the pinned revision")
    old_development, old_held, old_timing, old_dropouts = (read(panels / item) for item in PRIOR)
    consumed = [*old_development["sources"], *old_held["sources"]]
    excluded_speakers = {item["speaker"] for item in consumed}
    excluded_files = {item["sha256"] for item in consumed}
    excluded_files.update(item["source"] for item in old_timing["rows"])
    excluded_files.update(item["source_sha256"] for item in old_dropouts["clocks"] if "speaker" in item)
    older_speakers = {item["speaker"] for item in old_timing["rows"]}
    older_speakers.update(item["speaker"] for item in old_dropouts["clocks"] if "speaker" in item)
    pool = defaultdict(list)
    with index.open(newline="") as stream:
        for row in csv.DictReader(stream):
            filename = row["Filename"]
            if filename.endswith("_NEU_XX"):
                actor = filename.split("_")[0]
                speaker = sha(f"CREMA-D:{actor}".encode())
                if speaker not in excluded_speakers:
                    pool[speaker].append(filename + ".flv")
    selected = sorted(pool, key=order)[:HELD_SPEAKERS]
    if len(selected) != HELD_SPEAKERS or len(excluded_speakers) != len(consumed) or len(old_development["sources"]) != 10:
        raise ValueError("The comparison needs the known development recordings and twelve different held speakers")

    # No change of clip, room or panning based on model scores. Use another
    # take from each acoustic category, then two additional independent takes.
    development_rooms = dict(acoustic_sources(archive))["development"][:10]
    if development_rooms != [item["room"] for item in old_development["sources"]]:
        raise ValueError("The development room ordering differs from its published measurements")
    hold_rooms = [name.replace("01.wav", "08.wav") for name in development_rooms]
    hold_rooms += [name.replace("01.wav", "09.wav") for name in sorted(development_rooms, key=order)[:2]]
    prior_rooms = {item["room"] for item in consumed}
    prior_room_hashes = {item["room_sha256"] for item in consumed}
    with ZipFile(archive) as bundle:
        rooms = [{"name": name, "sha256": sha(bundle.read(name))} for name in hold_rooms]
    if prior_rooms & set(hold_rooms) or prior_room_hashes.intersection(room["sha256"] for room in rooms) or len(
        {room["sha256"] for room in rooms}
    ) != HELD_SPEAKERS:
        raise ValueError("The held room recordings must be independent of the published stereo panel")
    planned_rows = [
        {"speaker": speaker, "options": sorted(pool[speaker], key=order), "room": room,
         "earlier_project_speaker": speaker in older_speakers}
        for speaker, room in zip(selected, rooms, strict=True)
    ]
    plan_doc = {
        "schema": SCHEMA,
        "publisher_revision": CREMA_REV,
        "source_index_sha256": SOURCE_INDEX_SHA,
        "license_sha256": LICENSE_SHA,
        "archive_sha1": ARCHIVE_SHA1,
        "compare": {name: digest(panels / name) for name in PRIOR},
        "development": old_development["sources"],
        "prior_speakers_excluded": sorted(excluded_speakers),
        "prior_source_hashes_excluded": sorted(excluded_files),
        "held": planned_rows,
        "selection": "Order neutral recordings by the fixed schema hash. Take the first qualifying video for each held "
                     "speaker. Reject previously uploaded exact sources and incompatible picture or sound clocks. Use "
                     "the original seven-arm panning, gain and speech-mute comparisons at both levels and encodes.",
    }
    save(output, plan_doc)
    print(f"Reserved {len(selected)} held speakers and distinct room takes. "
          f"{sum(item['earlier_project_speaker'] for item in planned_rows)} speakers appeared in earlier timing work.", flush=True)


def qualify(clip, checksum):
    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import probe_frames

    table = probe_frames(clip)
    if (not 40 <= len(table) <= 120 or table.rotation or table.truncated or table.interpolated
            or any(abs((pts - table.pts[0]) * table.tb - i * FRAME_PERIOD) > Fraction(3, 2000)
                   for i, pts in enumerate(table.pts))):
        return "picture_scope"
    try:
        pcm, _ = source_sound(decode_audio(clip, source_sha256=checksum), table)
        fit_sound(pcm, round(len(table) * RATE / 30))
    except ValueError:
        return "sound_clock"
    return None


def download(frozen, output):
    document = read(frozen)
    if document["schema"] != SCHEMA or document["publisher_revision"] != CREMA_REV:
        raise ValueError("Frozen reservation differs from the expected source")
    exclusion = set(document["prior_source_hashes_excluded"])
    output.mkdir(mode=0o700)

    def materialize(item):
        failures = []
        for name in item["options"]:
            pin, size = lfs_pointer(fetch(f"{CREMA}/VideoFlash/{name}", 1024))
            if pin in exclusion:
                failures.append({"name": name, "sha256": pin, "reason": "exact_source_reused"})
                continue
            if not 0 < size <= MAX_MEDIA_BYTES:
                raise ValueError(f"Publisher video is outside the expected size for {name}")
            base = f"https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/{CREMA_REV}/VideoFlash"
            body = fetch(f"{base}/{name}", size)
            if len(body) != size or sha(body) != pin:
                raise ValueError(f"Media differs from its pinned Git LFS pointer for {name}")
            clip = output / name
            clip.write_bytes(body)
            if reason := qualify(clip, pin):
                failures.append({"name": name, "sha256": pin, "reason": reason})
                clip.unlink()
                continue
            print(f"Verified {name} ({size} bytes)", flush=True)
            return {key: value for key, value in item.items() if key != "options"} | {
                "name": name, "sha256": pin, "bytes": size, "scope_exclusions": failures, "split": "held",
            }
        raise ValueError(f"Speaker {item['speaker']} has no qualifying fresh recording")

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = list(workers.map(materialize, document["held"]))
    if len({item["sha256"] for item in rows}) != HELD_SPEAKERS:
        raise ValueError("Held entries must be distinct publisher files")
    save(output / "sources.json.gz", {"schema": SCHEMA, "plan_sha256": digest(frozen), "entries": rows})


def mouth_frames(picture, clock, assets):
    from app.detectors.video.decode import probe_frames
    from app.detectors.video.viseme_worker import capture_mouth_frames

    table = probe_frames(picture)
    ticks = clock["ticks"]
    if (table.tb != Fraction(1, 30_000) or list(table.pts) != ticks
            or ffmpeg("-i", str(picture), *STREAM_HASH).decode().strip() != clock["picture_stream"]):
        raise ValueError("Picture stream or presentation times differ from the verified source clock")
    frames, _, _ = capture_mouth_frames(picture, table, assets / "face_landmarker.task", motion=False, shots=[[0, len(table)]])
    return [{**frame, "sample": round(tick * RATE / 30_000)} for frame, tick in zip(frames, ticks, strict=True)]


def load_assets(assets):
    if digest(assets / "face_landmarker.task") != GEOMETRY_SHA:
        raise ValueError("Unexpected mouth landmark asset")
    sys.path.insert(0, str(assets / "runtime"))


def replay_mouth(receipt, videos, assets, output):
    load_assets(assets)
    source = read(receipt)["sources"]
    measure = {}
    for item in source:
        group = item["group"]
        measure[group] = mouth_frames(videos / f"{group}-pictures.mp4", item, assets)
    save(output, measure)
    print(f"Replayed mouth coordinates on {len(measure)} source clocks", flush=True)


def prepare(frozen, sources, archive, assets, output):
    check_archive(archive)
    load_assets(assets)
    inventory = sources / "sources.json.gz"
    reservation = read(frozen)
    source = read(inventory)
    if (source["schema"] != SCHEMA or source["plan_sha256"] != digest(frozen)
            or len(source["entries"]) != HELD_SPEAKERS):
        raise ValueError("Held preparation needs the twelve frozen source recordings")
    excluded = set(reservation["prior_source_hashes_excluded"])
    for item, planned in zip(source["entries"], reservation["held"], strict=True):
        if (item["speaker"] != planned["speaker"] or item["room"] != planned["room"]
                or item["name"] not in planned["options"] or item["sha256"] in excluded):
            raise ValueError("The held inventory differs from the frozen source reservation")
    output.mkdir(mode=0o700)
    rows, clocks, mouth = [], [], {}
    with ZipFile(archive) as bundle:
        for number, item in enumerate(source["entries"], 1):
            clip = sources / item["name"]
            room = item["room"]
            raw = bundle.read(room["name"])
            if digest(clip) != item["sha256"] or sha(raw) != room["sha256"]:
                raise ValueError("A source differs from the frozen publisher commitments")
            observations, clock = observe(clip, raw, number, assets, output)
            group = clock["group"]
            mouth[group] = mouth_frames(output / f"{group}-pictures.mp4", clock, assets)
            clocks.append(item | clock | {"room": room["name"], "room_sha256": room["sha256"]})
            rows.extend(observations)
            print(f"Prepared {number}/{len(source['entries'])} speakers, {len(rows)} PCM/AAC deliveries", flush=True)
    mouth_path = output / "mouth.json.gz"
    save(mouth_path, mouth)
    save(output / "held-inputs.json.gz", {"schema": SCHEMA, "split": "held", "source_sha256": digest(inventory),
                                         "mouth_sha256": digest(mouth_path), "sources": clocks, "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, fields in {
        "plan": ("index", "license_file", "archive", "panels", "output"),
        "download": ("frozen", "output"),
        "prepare": ("frozen", "sources", "archive", "assets", "output"),
        "mouth": ("receipt", "videos", "assets", "output"),
    }.items():
        sub = commands.add_parser(name)
        for field in fields:
            sub.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    {"plan": plan, "download": download, "prepare": prepare, "mouth": replay_mouth}[args.pop("command")](**args)
