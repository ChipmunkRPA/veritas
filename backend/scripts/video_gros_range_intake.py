# SPDX-License-Identifier: Apache-2.0
"""Fetch four fixed CC BY motion-control clips using only member ranges."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
import subprocess
import zipfile
import zlib
from pathlib import Path

PARTS = {1: "xgg8xcscr5", 2: "f47pm7rwt3"}
PREFIX = b"chipmunk-gros-held-head-motion-v1\0"
MAX_MEMBER = 230 * 1024**2


class SuffixReader(io.RawIOBase):
    def __init__(self, suffix, size):
        self.suffix, self.size, self.pos = suffix, size, 0

    def seek(self, offset, whence=0):
        self.pos = (0, self.pos, self.size)[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        start = self.pos - (self.size - len(self.suffix))
        if start < 0:
            raise ValueError("Central directory exceeds the bounded suffix")
        data = self.suffix[start:] if n < 0 else self.suffix[start:start+n]
        self.pos += len(data)
        return data


def _require(condition, detail):
    if not condition:
        raise ValueError(detail)


def plan(folder):
    rows = []
    for part, dataset in PARTS.items():
        receipt = json.loads((folder / f"gros-part{part}.json").read_text())[0]
        suffix = (folder / f"part{part}-tail.bin").read_bytes()
        metadata = receipt["content_details"]
        with zipfile.ZipFile(SuffixReader(suffix, receipt["size"])) as archive:
            files = {info.filename: info for info in archive.infolist()
                     if re.fullmatch(r"\d+/(?:med|warm|cold|low|high)\.MP4", info.filename)}
        subjects = sorted({name.split("/")[0] for name in files},
                          key=lambda value: hashlib.sha256(PREFIX + value.encode()).digest())[:2]
        for lighting, subject in zip(("med", "warm") if part == 1 else ("cold", "low"), subjects, strict=True):
            info = files[f"{subject}/{lighting}.MP4"]
            _require(0 < info.compress_size <= MAX_MEMBER and 0 < info.file_size <= MAX_MEMBER
                     and info.compress_type in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED), "Unsupported source member")
            rows.append({"part": part, "dataset_version": 1, "dataset": dataset, "publisher_file_id": receipt["id"],
                         "publisher_url": metadata["download_url"], "archive_size": receipt["size"],
                         "publisher_archive_sha256": metadata["sha256_hash"], "zip_tail_sha256": hashlib.sha256(suffix).hexdigest(),
                         "source_member": info.filename, "lighting": lighting, "header_offset": info.header_offset,
                         "compressed": info.compress_size, "size": info.file_size, "method": info.compress_type, "crc32": info.CRC})
    _require(len({row["source_member"].split("/")[0] for row in rows}) == 4 and sum(row["size"] for row in rows) < 900 * 1024**2,
             "Need four distinct and bounded sources")
    with (folder / "gros-roster.json").open("x") as stream:
        json.dump({"license": "CC-BY-4.0", "selection": "two subjects per part in fixed hash order, with predefined light settings",
                   "rows": rows}, stream, sort_keys=True, indent=2)
        stream.write("\n")


def fetch(folder, destination):
    destination.mkdir(mode=0o700, exist_ok=False)
    roster = json.loads((folder / "gros-roster.json").read_text())
    results = []
    for slot, row in enumerate(roster["rows"], 1):
        start = row["header_offset"]
        offset = start + 30 + 65535 + 65535  # Maximum filename and extra-field bytes.
        stop = min(row["archive_size"] - 1, offset + row["compressed"])
        header = folder / f"gros-member-{slot}-response.headers"
        filename = f"gros-{slot:02d}.mp4"
        final, partial = destination / filename, destination / f"{filename}.part"
        proc = subprocess.Popen(["curl", "-fsSL", "--max-time", "150", "--max-filesize", str(stop-start+1),
                                 "--range", f"{start}-{stop}", "-D", str(header), row["publisher_url"]],
                                stdout=subprocess.PIPE)
        try:
            with proc.stdout as stream, partial.open("xb") as out:
                fields = struct.unpack("<4s5H3L2H", stream.read(30))
                magic, method, name_size, extra_size = fields[0], fields[3], fields[-2], fields[-1]
                _require(magic == b"PK\x03\x04" and method == row["method"], "Invalid member header")
                _require(stream.read(name_size).decode("ascii") == row["source_member"], "Member name differs")
                _require(len(stream.read(extra_size)) == extra_size, "Incomplete member header")
                decoder = zlib.decompressobj(-15) if method == zipfile.ZIP_DEFLATED else None
                digest, crc, emitted, pending = hashlib.sha256(), 0, 0, row["compressed"]
                while pending:
                    data = stream.read(min(1024**2, pending))
                    _require(data, "Incomplete ranged member")
                    pending -= len(data)
                    pixels = data if decoder is None else decoder.decompress(data)
                    emitted += len(pixels)
                    _require(emitted <= row["size"], "ZIP member exceeds publisher index")
                    out.write(pixels)
                    digest.update(pixels)
                    crc = zlib.crc32(pixels, crc)
                _require(decoder is None or decoder.eof, "Incomplete deflate stream")
                while stream.read(1024**2):
                    pass
                _require(proc.wait(timeout=10) == 0 and emitted == row["size"] and crc == row["crc32"],
                         "Publisher member length or CRC differs")
            headers = header.read_text(errors="replace").lower()
            _require(f"content-range: bytes {start}-{stop}/{row['archive_size']}" in headers,
                     "Publisher response did not serve the exact range")
            partial.rename(final)
            results.append({"slot": slot, "file": filename, "sha256": digest.hexdigest(), "bytes": emitted, "crc32": crc})
            print("Acquired licensed motion control", slot, emitted, flush=True)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            partial.unlink(missing_ok=True)
    with (folder / "gros-files.json").open("x") as stream:
        json.dump(results, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "fetch"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        plan(args.folder)
    else:
        if args.destination is None:
            parser.error("fetch requires --destination")
        fetch(args.folder, args.destination)


if __name__ == "__main__":
    main()
