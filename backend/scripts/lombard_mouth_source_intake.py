# SPDX-License-Identifier: Apache-2.0
"""Select and range-read a bounded new frontal Lombard speech source cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from binascii import crc32
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from scripts.grid_clip_plan_v136 import DirectoryReader
from scripts.video_common_sources_v89 import read, require, sha, write

RECORD = 3736465
SALT = "chipmunk-native-lombard-release/1"
METADATA = {"publisher-json.zip": "lombardgrid_json.zip", "alignment.zip": "lombardgrid_alignment.zip"}
MEDIA = {"front": "lombardgrid_front.zip", "audio": "lombardgrid_audio.zip"}
PROTOCOL = {
    "speaker_roles": ["development", "development", *(["held"] * 6)],
    "source_order": "SHA-256 order of published speaker ids, then known recording ids per condition. No observed scores enter.",
    "recordings_per_condition": 2,
    "stop_prior": "Earliest internal lexical /b/ or /p/, after 0.52 and before 1.80 seconds, followed by a nonsilence phone. The label is only a coarse neighborhood for independent signal and picture observers.",
    "evaluation_unit": "Publisher speaker code. No known overlap with the original GRID release; cross-corpus identities were not inferred.",
}


class MemberReader(DirectoryReader):
    def seekable(self):
        return True


def md5(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()


def key(value):
    return hashlib.sha256(f"{SALT}/{value}".encode()).hexdigest()


def publisher(folder):
    record = read(folder / "record.json")
    require(
        record["id"] == RECORD and record["metadata"]["license"]["id"] == "cc-by-4.0",
        "Expected official CC BY 4.0 record",
    )
    by_name = {row["key"]: row for row in record["files"]}
    for name, remote in METADATA.items():
        file = folder / name
        require(
            file.stat().st_size == by_name[remote]["size"]
            and by_name[remote]["checksum"] == "md5:" + md5(file),
            "Publisher metadata archive differs",
        )
    return {
        role: {
            "url": f"https://zenodo.org/records/{RECORD}/files/{name}",
            "bytes": by_name[name]["size"],
            "published_checksum": by_name[name]["checksum"],
        }
        for role, name in MEDIA.items()
    }


def target(phones):
    for a, b in zip(phones, phones[1:]):
        end = Fraction(str(a["offset"])) + Fraction(str(a["duration"]))
        if (
            a["phone"] in {"b_B", "p_B"}
            and Fraction(13, 25) <= end <= Fraction(9, 5)
            and not b["phone"].startswith("SIL")
        ):
            return {
                "phoneme": a["phone"].split("_")[0],
                "aligned_start_s": str(Fraction(str(a["offset"]))),
                "aligned_following_phone_s": str(end),
            }
    return None


def index_zip(archive, extension):
    entries = {}
    for row in archive.infolist():
        p = PurePosixPath(row.filename)
        require(
            not p.is_absolute()
            and ".." not in p.parts
            and "\\" not in row.filename
            and "\x00" not in row.filename,
            "Unsafe archive member",
        )
        if p.name.startswith("._") or "__MACOSX" in p.parts or p.suffix.lower() != extension:
            continue
        require(
            p.stem not in entries
            and row.compress_type in {ZIP_STORED, ZIP_DEFLATED}
            and not row.flag_bits & 1
            and 0 < row.file_size < 2 * 1024**2,
            "Invalid duplicate or oversized member",
        )
        entries[p.stem] = row
    return entries


def source_plan(folder):
    archives = publisher(folder)
    roster = defaultdict(dict)
    with ZipFile(folder / "publisher-json.zip") as metadata, ZipFile(folder / "alignment.zip") as alignments:
        aligned = index_zip(alignments, ".json")
        for file in index_zip(metadata, ".json").values():
            for row in json.loads(metadata.read(file)):
                speaker, condition, utterance = (row[k] for k in ("SPKR", "COND", "UTTERANCE"))
                uid = f"{speaker}_{condition}_{utterance}"
                if row["STATUS"] != "CORRECT" or condition not in {"p", "l"} or uid not in aligned:
                    continue
                label = alignments.read(aligned[uid])
                if (event := target(json.loads(label)[uid])) is not None:
                    roster[speaker][uid] = {
                        "condition": condition,
                        "alignment_sha256": hashlib.sha256(label).hexdigest(),
                        **event,
                    }
    readers = {role: DirectoryReader(archive) for role, archive in archives.items()}
    with ZipFile(readers["front"]) as front, ZipFile(readers["audio"]) as audio:
        by_role = {"front": index_zip(front, ".mov"), "audio": index_zip(audio, ".wav")}
    sources = []
    speakers = []
    expected_roles = PROTOCOL["speaker_roles"]
    for speaker in sorted(roster, key=key):
        eligible = [
            uid for uid in sorted(roster[speaker], key=key) if all(uid in rows for rows in by_role.values())
        ]
        arms = [
            [uid for uid in eligible if roster[speaker][uid]["condition"] == condition][:2]
            for condition in ("p", "l")
        ]
        if min(map(len, arms)) < 2:
            continue
        stage = expected_roles[len(speakers)]
        speakers.append(speaker)
        for uid in [*arms[0], *arms[1]]:
            ordinal = len(sources) + 1
            sources.append(
                {
                    "ordinal": ordinal,
                    "group": len(speakers),
                    "speaker": speaker,
                    "split": stage,
                    "source": uid,
                    "members": {
                        role: {
                            "name": items[uid].filename,
                            "bytes": items[uid].file_size,
                            "crc32": items[uid].CRC,
                        }
                        for role, items in by_role.items()
                    },
                    **roster[speaker][uid],
                }
            )
        if len(speakers) == len(expected_roles):
            break
    require(len(speakers) == len(expected_roles), "Incomplete disjoint matched speaker roster")
    return {
        "schema": "chipmunk.lombard-mouth-sources/1",
        "spec": PROTOCOL,
        "archives": archives,
        "rows": sources,
        "directory_reads": {role: reader.network_bytes for role, reader in readers.items()},
        "metadata_sha256": {name: sha(folder / name) for name in (*METADATA, "record.json")},
        "worker_sha256": sha(Path(__file__)),
    }


def acquire(folder):
    lock = read(folder / "freeze.json")
    require(
        lock["archives"] == publisher(folder) and lock["worker_sha256"] == sha(Path(__file__)),
        "Frozen intake changed",
    )
    deliveries = folder / "sources"
    deliveries.mkdir(mode=0o700, exist_ok=True)
    rows = [dict(row, inputs={}) for row in lock["rows"]]
    cache = {}

    def member(plan):
        role, archive, row = plan
        pinned = row["members"][role]
        suffix = "mov" if role == "front" else "wav"
        name = f"{row['ordinal']:02}-{role}.{suffix}"
        path = deliveries / name
        network_bytes = 0
        if path.exists():
            data = path.read_bytes()
        else:
            reader = MemberReader(archive)
            reader.cache = list(cache[role])
            with ZipFile(reader) as contents:
                item = contents.getinfo(pinned["name"])
                require(
                    item.file_size == pinned["bytes"] and pinned["crc32"] == item.CRC,
                    "Member commitment changed",
                )
                data = contents.read(item)
            network_bytes = reader.network_bytes
            with path.open("xb") as stream:
                stream.write(data)
        require(
            len(data) == pinned["bytes"] and crc32(data) == pinned["crc32"],
            "Local member fails frozen archive CRC",
        )
        row["inputs"][role] = {
            "file": name,
            "sha256": hashlib.sha256(data).hexdigest(),
            "zip_range_bytes": network_bytes,
        }
        print(role, row["ordinal"], flush=True)

    for role, archive in lock["archives"].items():
        suffix = "mov" if role == "front" else "wav"
        if any(not (deliveries / f"{row['ordinal']:02}-{role}.{suffix}").exists() for row in rows):
            reader = DirectoryReader(archive)
            with ZipFile(reader):
                cache[role] = reader.cache
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                member, [(role, archive, row) for role, archive in lock["archives"].items() for row in rows]
            )
        )
    write(folder / "source-receipt.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "acquire"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        write(args.folder / "freeze.json", source_plan(args.folder))
    else:
        acquire(args.folder)


if __name__ == "__main__":
    main()
