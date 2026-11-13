"""Verify a bounded native Half-Truth test sample and its publisher time labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
import wave
import zlib
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.audio-had-native-intake/1"
ARCHIVE_SIZE = 8_073_665_280
ARCHIVE_MD5 = "4daef62a7cf20c71b052635c968ece1c"
RANGES = {
    "tail": (8_073_599_723, 8_073_665_279, "1b64c86810f356e324c25e6ce621378c16c15097e27b543a61bdc4b9be244f5b"),
    "testlabels": (1_719_034_985, 1_719_115_621, "01a5141ef889f0bd50c24058ae3a2d776c7d9d45679d55d468d7046ee0a83a35"),
    "testprefix": (1_719_115_622, 1_723_309_925, "ef7cb9e326a3eb4a8a2dba6853c07c63fdd8b8b7925fa18c38fb1b474b80cd5f"),
}
LOCAL = struct.Struct("<4s5H3L2H")
RATE = 16_000
TICK = RATE // 100
ITEM = re.compile(r"HAD_test_\d{8}")
TIME = re.compile(r"(\d{1,2})\.(\d{2})")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_range(folder, name):
    first, last, digest = RANGES[name]
    path = folder / f"{name}.bin"
    require(path.stat().st_size == last - first + 1, f"Wrong byte count for {name}")
    data = path.read_bytes()
    headers = (folder / f"{name}.headers").read_text().casefold().splitlines()
    require("http/1.1 206 partial_content" in headers or any(re.match(r"http/\S+ 206\b", line) for line in headers),
            f"Publisher did not return a partial response for {name}")
    require(f"content-range: bytes {first}-{last}/{ARCHIVE_SIZE}" in headers and sha(data) == digest,
            f"Publisher bytes differ for {name}")
    return data


def local_member(data, start):
    magic, _, flags, method, _, _, crc, packed, unpacked, name_size, extra_size = LOCAL.unpack_from(data, start)
    require(magic == b"PK\x03\x04" and flags == 0 and method in {0, 8}, "Unsupported ZIP local record")
    name_start = start + LOCAL.size
    body_start = name_start + name_size + extra_size
    require(0 < name_size <= 120 and body_start <= len(data), "Incomplete ZIP local header")
    name = data[name_start:name_start + name_size].decode("ascii")
    return name, method, body_start, body_start + packed, unpacked, crc


def inflate(data, limit, *, complete):
    decoder = zlib.decompressobj(-15)
    result = decoder.decompress(data, limit + 1)
    require(len(result) <= limit and not decoder.unconsumed_tail, "Inflation exceeds the expected extent")
    if complete:
        require(decoder.eof and not decoder.unused_data, "Incomplete or overlong ZIP stream")
    return result


def full_member(data, member, *, limit):
    name, method, start, end, unpacked, crc = member
    require(end <= len(data) and unpacked <= limit, f"Incomplete or oversized ZIP member {name}")
    body = data[start:end]
    decoded = inflate(body, unpacked, complete=True) if method == 8 else body
    require(len(decoded) == unpacked and zlib.crc32(decoded) == crc, f"ZIP length or CRC differs for {name}")
    return decoded


def label_clock(text):
    match = TIME.fullmatch(text)
    require(match is not None, "Unexpected publisher time unit")
    return 100 * int(match[1]) + int(match[2])


def labels(data):
    values = {}
    for line in data.decode("ascii").splitlines():
        name, segments, status = line.split()
        require(ITEM.fullmatch(name) and name not in values and status in {"0", "1"}, "Unexpected test label row")
        previous = 0
        result = []
        for item in segments.split("/"):
            start, end, kind = item.split("-")
            a, b = label_clock(start), label_clock(end)
            require(previous == a < b and kind in {"T", "F"}, "Publisher label clock is discontinuous")
            result.append((a, b, kind))
            previous = b
        require((status == "0") == any(kind == "F" for _, _, kind in result), "Overall and local labels disagree")
        values[name] = result
    require(len(values) == 9072, "Publisher test label count differs")
    return values


def valid_geometry(parts):
    total = parts[-1][1]
    return (len(parts) == 3 and [kind for _, _, kind in parts] == ["T", "F", "T"]
            and 400 <= total <= 900 and 50 <= parts[1][1] - parts[1][0] <= 120
            and parts[0][1] >= 40 and total - parts[2][0] >= 40)


def wav_row(name, data, parts):
    with wave.open(io.BytesIO(data)) as sample:
        require((sample.getframerate(), sample.getnchannels(), sample.getsampwidth(), sample.getcomptype()) ==
                (RATE, 1, 2, "NONE"), f"Unexpected WAV format for {name}")
        samples = sample.getnframes()
        pcm = sample.readframes(samples)
    labels_in_samples = [[a * TICK, b * TICK, kind] for a, b, kind in parts]
    remainder = samples - labels_in_samples[-1][1]
    require(len(pcm) == 2 * samples and 0 <= remainder < TICK, f"Native sample clock differs for {name}")
    return {"id": name, "wav_sha256": sha(data), "pcm_sha256": sha(pcm), "samples": samples,
            "labels": labels_in_samples, "unlabeled_tail_samples": remainder, "eligible": valid_geometry(parts)}


def review(folder, *, media_dir=None):
    tail, annotated, prefix = (read_range(folder, name) for name in RANGES)
    end_offset = tail.rindex(b"PK\x06\x06")
    signature, size, _, _, disk, central_disk, count, total, length, start = struct.unpack_from(
        "<4sQ2H2L4Q", tail, end_offset
    )
    require((signature, size, disk, central_disk, count, total, length, start) ==
            (b"PK\x06\x06", 44, 0, 0, 13, 13, 1351, 8_073_663_831), "ZIP64 directory differs")
    label = local_member(annotated, 0)
    readme = local_member(annotated, label[3])
    require(label[0] == "HAD/HAD_test/HAD_test_label.txt" and readme[0] == "HAD/HAD_test/README.md"
            and readme[3] == len(annotated), "Unexpected publisher test metadata members")
    text = full_member(annotated, label, limit=500_000)
    instructions = full_member(annotated, readme, limit=1_000)
    timestamps = labels(text)
    wrapper = local_member(prefix, 0)
    require(wrapper[0] == "HAD/HAD_test/test.zip" and wrapper[1] == 8 and wrapper[3] > len(prefix),
            "Unexpected outer audio member")
    nested = inflate(prefix[wrapper[2]:], 8 * 1024**2, complete=False)
    start = 0
    rows = []
    selected = {}
    partial_member = None
    while start + LOCAL.size <= len(nested):
        member = local_member(nested, start)
        member_name = member[0]
        start = member[3]
        if start > len(nested):
            partial_member = member_name
            break
        if member_name == "test/":
            continue
        require(member_name == f"test/HAD_test_{len(rows) + 1:08d}.wav", "Unexpected nested member order")
        name = member_name.removeprefix("test/").removesuffix(".wav")
        data = full_member(nested, member, limit=1_000_000)
        row = wav_row(name, data, timestamps[name])
        rows.append(row)
        if row["eligible"]:
            selected[name] = data
    require(rows and partial_member and selected, "Native prefix did not yield a qualified intake")
    if media_dir is not None:
        media_dir.mkdir()
        for name, data in selected.items():
            (media_dir / f"{name}.wav").write_bytes(data)
    census = Counter("genuine" if all(kind == "T" for _, _, kind in row)
                     else "fully_synthetic" if all(kind == "F" for _, _, kind in row)
                     else "partial" for row in timestamps.values())
    return {"schema": SCHEMA, "record": 10377492, "license": "CC BY 4.0", "archive_bytes": ARCHIVE_SIZE,
            "publisher_archive_md5": ARCHIVE_MD5, "bytes_acquired": sum(b - a + 1 for a, b, _ in RANGES.values()),
            "source_ranges": {name: {"first_byte": a, "last_byte": b, "sha256": digest}
                              for name, (a, b, digest) in RANGES.items()},
            "labels_sha256": sha(text), "readme_sha256": sha(instructions), "publisher_test_counts": census,
            "generator": "GST-Tacotron / LPCNet", "language": "Mandarin", "declared_speaker_ids": False,
            "declared_parent_utterance_ids": False, "label_resolution_samples": TICK,
            "method": {"samples": [4 * RATE, 9 * RATE], "one_splice_samples": [RATE // 2, 6 * RATE // 5],
                       "minimum_genuine_flank_samples": 2 * RATE // 5, "native_audio_format": "16000 Hz mono PCM16"},
            "native_complete": len(rows), "incomplete_next_member": partial_member, "eligible_positives": len(selected),
            "detector_forwards": 0, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--media-dir", type=Path, help="New local directory for eligible publisher WAVs")
    args = parser.parse_args()
    result = review(args.folder, media_dir=args.media_dir)
    if args.compare:
        require(result == json.loads(args.compare.read_bytes()), "Native HAD receipt does not replay")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
