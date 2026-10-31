"""JPEG structure read from the bitstream (I-1, IF §4.1): quantisation tables, component sampling and frame type.

The original bytes are walked marker by marker up to the first start-of-scan, without decoding, so the tables are the
file's own (luma and chroma, 8- or 16-bit). Tables are returned in natural (row-major) order, as Pillow's `quantization`.

- `quality_luma`, `quality_chroma`: the v0.3 estimate per table (mean ratio to the IJG standard table, then IJG scaling).
  `quality_luma` equals `forensics.jpeg_info`, which fills the manifest's `jpeg_q_est`, so the fusion context is unchanged.
- `ijg_quality`: the libjpeg quality whose scaled standard luma and chroma tables equal the file's, else None.
- `fingerprint`: sha256 over the component tables and sampling factors, an in-house key for the I-4 table database. It is
  not ExifTool's JPEGDigest.
- Not read: tables redefined after the first scan (possible between progressive scans) and anything after the frame
  header of lossless or hierarchical files.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.provenance.metadata_rules import STD_CHROMA, STD_LUMA, standard_quality

ZIGZAG_TO_NATURAL = (0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
                     7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
                     39, 46, 53, 60, 61, 54, 47, 55, 62, 63)
FRAMES = {0xC0: "baseline", 0xC1: "extended sequential", 0xC2: "progressive", 0xC3: "lossless",
          0xC5: "differential sequential", 0xC6: "differential progressive", 0xC7: "differential lossless",
          0xC9: "extended sequential, arithmetic", 0xCA: "progressive, arithmetic", 0xCB: "lossless, arithmetic",
          0xCD: "differential sequential, arithmetic", 0xCE: "differential progressive, arithmetic",
          0xCF: "differential lossless, arithmetic"}
STANDALONE = frozenset({0x01, *range(0xD0, 0xD9)})  # TEM, RST0–RST7, SOI: no length field
SUBSAMPLING = {(1, 1): "4:4:4", (2, 1): "4:2:2", (2, 2): "4:2:0", (4, 1): "4:1:1", (1, 2): "4:4:0", (4, 2): "4:1:0"}
DQT, SOS, EOI, DRI = 0xDB, 0xDA, 0xD9, 0xDD


class JpegParseError(ValueError):
    """A JPEG whose marker structure is truncated or invalid before the first scan."""


@dataclass(frozen=True)
class Component:
    id: int
    h: int
    v: int
    table_id: int


@dataclass(frozen=True)
class JpegStructure:
    frame: str
    precision: int
    width: int
    height: int
    components: tuple[Component, ...]
    tables: dict[int, tuple[int, ...]]  # table id → 64 values in natural order
    table_bits: dict[int, int]  # table id → 8 or 16
    restart_interval: int = 0

    @property
    def sampling_factors(self) -> str:
        return ",".join(f"{c.h}x{c.v}" for c in self.components)

    @property
    def subsampling(self) -> str:
        """4:4:4, 4:2:2, 4:2:0 … for YCbCr-like files, `grayscale` for one component, else the raw sampling factors."""
        if len(self.components) == 1:
            return "grayscale"
        if len(self.components) == 3:
            y, cb, cr = self.components
            if (cb.h, cb.v) == (cr.h, cr.v) and cb.h and cb.v and y.h % cb.h == 0 and y.v % cb.v == 0:
                label = SUBSAMPLING.get((y.h // cb.h, y.v // cb.v))
                if label is not None:
                    return label
        return self.sampling_factors

    def table(self, index: int) -> tuple[int, ...] | None:
        """The table of the index-th component (0: luma, 1: chroma); None when absent or never defined."""
        return self.tables.get(self.components[index].table_id) if index < len(self.components) else None

    @property
    def quality_luma(self) -> int | None:
        return estimate_quality(self.table(0), STD_LUMA)

    @property
    def quality_chroma(self) -> int | None:
        return estimate_quality(self.table(1), STD_CHROMA)

    @property
    def ijg_quality(self) -> int | None:
        luma, chroma = self.table(0), self.table(1)
        return None if luma is None or chroma is None else standard_quality({0: luma, 1: chroma})

    @property
    def fingerprint(self) -> str:
        payload = {"sampling": self.sampling_factors, "tables": [list(self.tables.get(c.table_id, ())) for c in self.components]}
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {"frame": self.frame, "precision": self.precision, "width": self.width, "height": self.height,
                "components": [{"id": c.id, "h": c.h, "v": c.v, "table_id": c.table_id} for c in self.components],
                "sampling_factors": self.sampling_factors, "subsampling": self.subsampling,
                "tables": {str(k): list(v) for k, v in sorted(self.tables.items())},
                "table_bits": {str(k): v for k, v in sorted(self.table_bits.items())}, "restart_interval": self.restart_interval,
                "quality_luma": self.quality_luma, "quality_chroma": self.quality_chroma, "ijg_quality": self.ijg_quality,
                "fingerprint": self.fingerprint}


def estimate_quality(table: tuple[int, ...] | None, standard: np.ndarray) -> int | None:
    """v0.3 estimate (`forensics.jpeg_info`): mean ratio to the standard table × 100, inverted through IJG scaling."""
    if table is None:
        return None
    scale = float(np.mean(np.asarray(table, dtype=np.float64) / np.asarray(standard, dtype=np.float64))) * 100
    quality = 50 * 100 / scale if scale >= 100 else 100 - scale / 2
    return int(round(max(1, min(100, quality))))


def _u16(buf: memoryview, at: int) -> int:
    return int.from_bytes(buf[at:at + 2], "big")


def _dqt(segment: memoryview, tables: dict[int, tuple[int, ...]], bits: dict[int, int]) -> None:
    i = 0
    while i < len(segment):
        precision, table_id = segment[i] >> 4, segment[i] & 0x0F
        i += 1
        if precision not in (0, 1) or table_id > 3:
            raise JpegParseError(f"invalid quantisation table header (precision {precision}, id {table_id})")
        size = 64 * (precision + 1)
        if i + size > len(segment):
            raise JpegParseError(f"quantisation table {table_id} is truncated")
        zigzag = list(segment[i:i + 64]) if precision == 0 else [_u16(segment, i + 2 * k) for k in range(64)]
        natural = [0] * 64
        for k, index in enumerate(ZIGZAG_TO_NATURAL):
            natural[index] = zigzag[k]
        tables[table_id], bits[table_id] = tuple(natural), 8 * (precision + 1)
        i += size


def parse(data: bytes | memoryview) -> JpegStructure | None:
    """The structure before the first scan; None when the data does not start with SOI. Raises `JpegParseError`."""
    buf = memoryview(data)
    if len(buf) < 2 or buf[0] != 0xFF or buf[1] != 0xD8:
        return None
    tables: dict[int, tuple[int, ...]] = {}
    bits: dict[int, int] = {}
    frame: tuple[str, int, int, int, tuple[Component, ...]] | None = None
    restart = 0
    pos, n = 2, len(buf)
    while True:
        if pos >= n:
            raise JpegParseError("the file ends before the first scan")
        if buf[pos] != 0xFF:
            raise JpegParseError(f"expected a marker at byte {pos}")
        while pos < n and buf[pos] == 0xFF:  # fill bytes
            pos += 1
        if pos >= n:
            raise JpegParseError("the file ends before the first scan")
        marker, pos = buf[pos], pos + 1
        if marker in STANDALONE:
            continue
        if marker == EOI:
            raise JpegParseError("end of image before the first scan")
        if pos + 2 > n or (length := _u16(buf, pos)) < 2 or pos + length > n:
            raise JpegParseError(f"segment 0x{marker:02X} at byte {pos - 2} is truncated")
        segment, pos = buf[pos + 2:pos + length], pos + length
        if marker == DQT:
            _dqt(segment, tables, bits)
        elif marker in FRAMES and frame is None:
            if len(segment) < 6 or segment[5] == 0 or len(segment) < 6 + 3 * segment[5]:
                raise JpegParseError("the frame header is truncated")
            components = tuple(Component(segment[6 + 3 * i], segment[7 + 3 * i] >> 4, segment[7 + 3 * i] & 0x0F, segment[8 + 3 * i])
                               for i in range(segment[5]))
            frame = (FRAMES[marker], segment[0], _u16(segment, 3), _u16(segment, 1), components)
        elif marker == DRI and len(segment) >= 2:
            restart = _u16(segment, 0)
        elif marker == SOS:
            if frame is None:
                raise JpegParseError("no frame header before the first scan")
            name, precision, width, height, components = frame
            return JpegStructure(frame=name, precision=precision, width=width, height=height, components=components,
                                 tables=tables, table_bits=bits, restart_interval=restart)


def read(data: bytes | memoryview) -> JpegStructure | None:
    """`parse`, with None for a malformed JPEG as well as for other formats."""
    try:
        return parse(data)
    except JpegParseError:
        return None
