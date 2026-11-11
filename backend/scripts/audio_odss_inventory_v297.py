"""Verify the ODSS release inventory without opening any recordings."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import PurePosixPath

from scripts.grid_clip_plan_v136 import DirectoryReader, require

URL = "https://zenodo.org/records/8370669/files/odss.zip?download=1"
ARCHIVE_BYTES = 2_403_767_464
ENTRIES = 30_463
DIRECTORY_OFFSET, DIRECTORY_BYTES = 2_399_666_751, 4_100_691
DIRECTORY_SHA256 = "497157e4b1dc4324cc2dcce0565905dc76e654e318fb8f84d5c6a5420bf1faaa"
METADATA = {
    "odss/LICENSE": "891aa3b7a433bed4facbea43d073418ebcb1c34f6c1bfb58245201a404d82db1",
    "odss/speaker_info.csv": "78663c6747afeb43e408845761e1b9bf3310ca5180f0875442c88dd3c32a0dea",
}
FOOTER = struct.Struct("<4s4H2LH")
HEADER = struct.Struct("<4s5H3I2H")
ORIGINS = ("natural", "vits", "fastpitch-hifigan")


def member(reader, info):
    require(
        info.filename in METADATA and info.file_size <= 4096, "Only the pinned license and CSV are readable"
    )
    reader.seek(info.header_offset)
    signature, _, flags, method, *_, name_size, extra_size = HEADER.unpack(reader.read(HEADER.size))
    name = info.filename.encode("ascii")
    require(
        signature == b"PK\x03\x04"
        and flags == info.flag_bits
        and not flags & 1
        and method == info.compress_type
        and method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        and name_size == len(name)
        and extra_size <= 1024
        and 0 < info.compress_size <= 4096
        and reader.tell() + name_size + extra_size + info.compress_size <= DIRECTORY_OFFSET,
        "Expected bounded local ZIP metadata header",
    )
    encoded = reader.read(name_size + extra_size + info.compress_size)
    require(encoded[:name_size] == name, "Local and central member names differ")
    body = encoded[name_size + extra_size :]
    if method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        body = decoder.decompress(body, info.file_size + 1)
        require(decoder.eof and not decoder.unused_data, "Complete single metadata deflate stream required")
    require(
        len(body) == info.file_size
        and zlib.crc32(body) == info.CRC
        and hashlib.sha256(body).hexdigest() == METADATA[info.filename],
        "Pinned metadata bytes differ",
    )
    return body.decode("utf-8-sig")


def match_inventory(entries, speakers):
    paths = [PurePosixPath(entry.filename) for entry in entries if not entry.is_dir()]
    require(len(paths) == len(set(paths)) == 30_028, "Expected unique release member paths")
    grouped = {origin: set() for origin in ORIGINS}
    for path in paths:
        if path.suffix != ".wav" or path.parts[2:3] != ("hifi-tts",):
            continue
        require(
            len(path.parts) == 5 and path.parts[0] == "odss" and path.parts[1] in ORIGINS,
            "Expected the pinned Hi-Fi TTS member layout",
        )
        grouped[path.parts[1]].add(path.parts[3:])
    natural = grouped["natural"]
    counts = Counter(speaker for speaker, _ in natural)
    require(
        all(items == natural for items in grouped.values())
        and len(natural) == 3778
        and len(counts) == len(speakers) == 8
        and counts.keys() == speakers,
        "Expected exact relative paths and publisher speaker groups for every origin",
    )
    return len(natural), len(counts), [min(counts.values()), max(counts.values())]


def audit(reader):
    reader.seek(ARCHIVE_BYTES - FOOTER.size)
    footer = reader.read(FOOTER.size)
    require(
        FOOTER.unpack(footer)
        == (b"PK\x05\x06", 0, 0, ENTRIES, ENTRIES, DIRECTORY_BYTES, DIRECTORY_OFFSET, 0),
        "Pinned single-volume ZIP footer required",
    )
    reader.seek(DIRECTORY_OFFSET)
    split = 2 * 1024**2
    catalog = reader.read(split) + reader.read(DIRECTORY_BYTES - split)
    require(hashlib.sha256(catalog).hexdigest() == DIRECTORY_SHA256, "Pinned ZIP directory differs")
    # The in-memory ZIP exposes the catalog only. Metadata bodies use exact ranges above.
    local_footer = FOOTER.pack(b"PK\x05\x06", 0, 0, ENTRIES, ENTRIES, DIRECTORY_BYTES, 0, 0)
    with zipfile.ZipFile(io.BytesIO(catalog + local_footer)) as archive:
        entries = archive.infolist()
        require(len(entries) == ENTRIES, "Exact catalog entry count required")
        license_notice, speaker_table = (member(reader, archive.getinfo(name)) for name in METADATA)
    require(
        "creativecommons.org/licenses/by-sa/4.0/" in license_notice, "Published CC BY-SA 4.0 notice required"
    )
    rows = list(csv.DictReader(io.StringIO(speaker_table)))
    speakers = {row["speaker"] for row in rows if row["dataset"] == "hifi-tts" and row["language"] == "en"}
    triplets, groups, sizes = match_inventory(entries, speakers)
    return {
        "schema": "chipmunk.audio-odss-inventory/297",
        "status": "metadata_verified",
        "release_doi": "10.5281/zenodo.8370669",
        "license": "CC-BY-SA-4.0",
        "directory_sha256": DIRECTORY_SHA256,
        "metadata_sha256": METADATA,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_entries": ENTRIES,
        "origins": ORIGINS,
        "exact_relative_path_triplets": triplets,
        "publisher_speaker_groups": groups,
        "per_group_triplet_range": sizes,
        "range_requests": reader.requests,
        "metadata_bytes": reader.network_bytes,
        "recordings_opened": 0,
        "model_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(audit(DirectoryReader({"url": URL, "bytes": ARCHIVE_BYTES})), indent=2))
