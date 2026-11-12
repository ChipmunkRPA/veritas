# SPDX-License-Identifier: Apache-2.0
"""Retrieve only the selected public GRID WAVs with bounded ZIP byte ranges."""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from scripts.grid_reserve_word_clocks import plan, publisher_record
from scripts.video_common_sources_v89 import require, write

MIB = 1024**2


class RangeReader:
    def __init__(self, asset):
        self.asset = asset
        self.position = 0
        self.ranges = []
        self.cache = []
        self.transferred = 0

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.asset["size"]}
        position = base[whence] + offset
        if not 0 <= position <= self.asset["size"]:
            raise OSError("Seek outside official archive size")
        self.position = position
        return position

    def read(self, amount=-1):
        remaining = self.asset["size"] - self.position
        size = remaining if amount == -1 else min(remaining, amount)
        if not 0 <= size <= 6*MIB:
            raise OSError("Publisher range exceeds individual limit")
        if size == 0:
            return b""
        start, end = self.position, self.position+size
        for left, right, data in self.cache:
            if left <= start and end <= right:
                self.position = end
                return data[start-left:end-left]
        stop = min(self.asset["size"], max(end, start+256*1024))
        if len(self.ranges) >= 40 or self.transferred + stop-start > 10*MIB:
            raise OSError("Publisher range exceeds aggregate limit")
        request = Request(self.asset["links"]["self"], headers={"Range": f"bytes={start}-{stop-1}", "Accept-Encoding": "identity"})
        with urlopen(request, timeout=12) as reply:
            if (reply.status != 206 or reply.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{self.asset['size']}"
                    or reply.headers.get("Content-Encoding", "identity") != "identity"):
                raise OSError("Expected the exact bounded publisher byte range")
            data = reply.read(stop-start+1)
        if len(data) != stop-start:
            raise OSError("Truncated publisher byte range")
        self.cache.append((start, stop, data))
        self.transferred += len(data)
        self.ranges.append({"start": start, "end_exclusive": stop, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.position = end
        return data[:size]


def acquire(output):
    asset = publisher_record()["audio_25k.zip"]
    reader = RangeReader(asset)
    sources, _ = plan()
    output.mkdir(mode=0o700, parents=True)
    extracted = []
    with ZipFile(reader) as archive:
        unique = defaultdict(list)
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.suffix.lower() == ".wav" and not path.name.startswith("._") and "__MACOSX" not in path.parts:
                unique[path.stem].append(info)
        require(len(unique) == 34000, "Expected the exact publisher WAV inventory")
        for ordinal, source in sources.items():
            stem = Path(source["input"]["planned"]["member"]["name"]).stem
            [info] = unique[stem]
            require(0 < info.file_size < 256*1024 and info.compress_type in {ZIP_STORED, ZIP_DEFLATED} and not info.flag_bits & 1,
                    "Expected a bounded publisher WAV ZIP entry")
            payload = archive.read(info)  # ZipFile checks the stored entry CRC.
            filename = f"{ordinal:02}.wav"
            (output / filename).write_bytes(payload)
            extracted.append({"ordinal": ordinal, "file": filename, "member": info.filename, "crc32": info.CRC,
                              "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    write(output / "receipt.json", {"publisher_asset": asset, "official_unique_stems": len(unique),
                                   "http_ranges": reader.ranges, "wavs": extracted})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    acquire(parser.parse_args().output)


if __name__ == "__main__":
    main()
