# SPDX-License-Identifier: Apache-2.0
"""Verify the bounded public archive index and timing labels for a video source panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
import zipfile
import zlib
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.video_gros_range_intake import SuffixReader

RECORD = 20548572
NAME = "temporally-localized-video-manipulation-detection-dataset.zip"
SIZE = 14390339732
ARCHIVE_MD5 = "2971f564807dd0ba9ed406dcac84ceea"
ROOT = "zenodo_dataset_package"
INDEX_PARTS = {
    "tail": (14390208660, 14390339731, "7992bdd664efb1f9ee031f9896850d002438f17ae3b6433cd14f53ab198487b5"),
    "metadata": (14390263862, 14390283564, "c5eaada109cf3f77f5bad8422fb1b04973802fdcba4e6b4c6759ad5be1d8726f"),
    "readme": (14390283565, 14390284879, "e3487870fb3c74d7b510527cbc8e9a17eb8bd6366f11aaff9e675e8062cc4105"),
}
MEDIA = {
    "authentic": ("", ".mp4"),
    "fake_segments": ("_fake", ".mp4"),
    "merged": ("_merged", ".mp4"),
    "images": ("", ".png"),
}
SCHEMA = "chipmunk.video-temporal-insert-source-intake/1"


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def archive_part(folder, label):
    start, stop, expected = INDEX_PARTS[label]
    content = (folder / f"{label}.bin").read_bytes()
    headers = (folder / f"{label}.headers").read_text().casefold().splitlines()
    require(len(content) == stop - start + 1 and sha(content) == expected
            and f"content-range: bytes {start}-{stop}/{SIZE}" in headers, "Bounded publisher range differs")
    return content


def stored_member(info, data, base_offset):
    offset = info.header_offset - base_offset
    require(0 <= offset <= len(data) - 30 and info.compress_type == zipfile.ZIP_STORED, "Unsupported ZIP member")
    signature, _, flags, method, _, _, _, _, _, name_size, extra_size = struct.unpack_from("<4s5H3L2H", data, offset)
    name_start = offset + 30
    body_start = name_start + name_size + extra_size
    body = data[body_start:body_start + info.file_size]
    require(signature == b"PK\x03\x04" and not flags & 1 and method == zipfile.ZIP_STORED
            and data[name_start:name_start + name_size].decode("ascii") == info.filename
            and len(body) == info.file_size and info.file_size == info.compress_size and zlib.crc32(body) == info.CRC,
            "Published ZIP member does not match its local header and central index")
    return body


def csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


def measure(folder):
    record = json.loads((folder / "record.json").read_bytes())
    [package] = record["files"]
    require(record["id"] == RECORD and record["metadata"]["version"] == "v1.0"
            and record["metadata"]["license"]["id"] == "cc-by-4.0"
            and (package["key"], package["size"], package["checksum"]) == (NAME, SIZE, f"md5:{ARCHIVE_MD5}"),
            "Publisher record, version, archive or license differs")
    tail, metadata, readme = (archive_part(folder, label) for label in ("tail", "metadata", "readme"))
    with zipfile.ZipFile(SuffixReader(tail, SIZE)) as archive:
        members = {row.filename: row for row in archive.infolist()}
        require(len(members) == len(archive.infolist()) == 409, "ZIP index is incomplete or repeated")
    payloads = {
        name: stored_member(members[f"{ROOT}/metadata/{name}"], metadata, INDEX_PARTS["metadata"][0])
        for name in ("dataset_manifest.csv", "frame_insertion_metadata.csv")
    }
    payloads["README.txt"] = stored_member(members[f"{ROOT}/README.txt"], readme, INDEX_PARTS["readme"][0])
    files = csv_rows(payloads["dataset_manifest.csv"])
    labels = csv_rows(payloads["frame_insertion_metadata.csv"])
    cases = {int(row["video_name"].removeprefix("Ornek_")): row for row in labels}
    require(len(cases) == len(labels) == 100 and set(cases) == set(range(1, 101)), "Incorrect frame-label roster")
    keyed = {(int(row["example_id"]), row["subset"]): row for row in files}
    require(len(keyed) == len(files) == 400 and set(keyed) == {(case, subset) for case in cases for subset in MEDIA},
            "File manifest has an incomplete or repeated example")
    for (case, subset), row in keyed.items():
        suffix, ext = MEDIA[subset]
        expected = f"Ornek_{case}{suffix}{ext}"
        member = members[f"{ROOT}/{subset}/{expected}"]
        require(row["file_name"] == expected and int(row["size_bytes"]) == member.file_size,
                "Publisher manifest does not match the central directory")
    rate_pairs = Counter()
    lengths = []
    equal_cadence = []
    for case, row in sorted(cases.items()):
        originals, generated, start, end, merged = (int(row[key]) for key in
            ("original_frames", "fake_frames_added", "start_frame", "end_frame", "total_frames_after_merge"))
        require(100 == start < end <= merged and originals >= start - 1 and generated == end - start and merged == originals + generated,
                "Publisher insertion rows are arithmetically inconsistent")
        rates = tuple(Fraction(row[key]) for key in ("fps_original", "fps_fake"))
        require(rates[0] > 0 and rates[1] == 24, "Unexpected publisher cadence")
        rate_pairs[tuple(row[key] for key in ("fps_original", "fps_fake"))] += 1
        lengths.append(generated)
        if rates[0] != rates[1]:
            continue
        triplet = {}
        for subset in ("authentic", "fake_segments", "merged"):
            key = keyed[case, subset]["file_name"]
            info = members[f"{ROOT}/{subset}/{key}"]
            require(info.compress_type == zipfile.ZIP_STORED and info.file_size == info.compress_size, "Unexpected ZIP storage")
            triplet[subset] = {"name": key, "header_offset": info.header_offset, "bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
        equal_cadence.append({"example": case, "publisher_frames": [start, end], "original_frames": originals, "merged_frames": merged,
                              "bytes": sum(row["bytes"] for row in triplet.values()), "members": triplet})
    smallest = sorted(equal_cadence, key=lambda row: (row["bytes"], row["example"]))[:4]
    return {
        "schema": SCHEMA, "publisher_record": RECORD, "archive_bytes": SIZE, "archive_md5": ARCHIVE_MD5, "dataset_license": "CC-BY-4.0",
        "ranges": {name: {"first": first, "last": last, "sha256": digest} for name, (first, last, digest) in INDEX_PARTS.items()},
        "annotation_member_sha256": {name: sha(content) for name, content in payloads.items()},
        "zip_entries": len(members), "verified_manifest_rows": len(files), "verified_timing_rows": len(labels),
        "publisher_insert_frames_min_max": [min(lengths), max(lengths)], "same_cadence_24fps": len(equal_cadence),
        "mixed_publisher_cadence": len(labels) - len(equal_cadence),
        "cadence_pairs": [{"original": a, "generated": b, "examples": count} for (a, b), count in sorted(rate_pairs.items())],
        "smallest_equal_cadence_triplets": smallest, "selected_media_bytes": sum(row["bytes"] for row in smallest),
        "native_media_files_read": 0, "model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    value = measure(args.inputs)
    if args.compare:
        require(value == json.loads(args.compare.read_bytes()), "Public source receipt differs")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
