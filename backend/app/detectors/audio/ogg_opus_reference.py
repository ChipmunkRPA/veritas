"""Verify the exact presentation clock of a bounded single-stream Ogg Opus file."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

PRESENTED = 288_000
PACKET = 960
PACKET_BYTES = 160  # 20 ms at 64 kb/s.
PRE_SKIP = 312
MAX_BYTES = 128 * 1024


def crc_table():
    table = []
    for index in range(256):
        work = index << 24
        for _ in range(8):
            work = ((work << 1) ^ (0x04C11DB7 if work & 0x80000000 else 0)) & 0xFFFFFFFF
        table.append(work)
    return table


def page_counts(data, *, channels=1, presented=PRESENTED):
    """Check complete native pages of the fixed 20 ms trial profiles."""
    table = crc_table()
    serial = None
    offset = 0
    preskip = None
    sizes = []
    sequence = 0
    while offset < len(data):
        header = data[offset:offset + 27]
        if len(header) != 27 or header[:5] != b"OggS\x00":
            raise ValueError("Incomplete Ogg page")
        flags = header[5]
        granule = int.from_bytes(header[6:14], "little")
        page_serial = header[14:18]
        page_sequence = int.from_bytes(header[18:22], "little")
        if sequence == 0:
            serial = page_serial
        if serial != page_serial or page_sequence != sequence:
            raise ValueError("Ogg stream has a missing or unrelated page")
        lace_end = offset + 27 + header[26]
        lace = data[offset + 27:lace_end]
        end = lace_end + sum(lace)
        if not lace or lace[-1] == 255 or end > len(data):
            raise ValueError("Missing Ogg lacing bytes or an unfinished packet")
        expected_flag = 2 if sequence == 0 else 4 if end == len(data) else 0
        if flags != expected_flag:
            raise ValueError("Ogg page flags differ from a single complete stream")
        page = data[offset:offset + 22] + bytes(4) + data[offset + 26:end]
        crc = 0
        for byte in page:
            crc = ((crc << 8) & 0xFFFFFFFF) ^ table[(crc >> 24) ^ byte]
        if crc != int.from_bytes(header[22:26], "little"):
            raise ValueError("Ogg page checksum mismatch")

        if sequence < 2:
            if granule != 0 or sum(value < 255 for value in lace) != 1:
                raise ValueError("Ogg headers require a zero granule and one contained packet")
            payload = data[lace_end:end]
            if sequence == 0:
                if (len(payload) != 19 or payload[:10] != b"OpusHead\x01" + bytes([channels])
                        or payload[12:] != b"\x80\xbb\x00\x00\x00\x00\x00"
                        or len(lace) != 1):
                    raise ValueError("OpusHead differs from the expected 48 kHz channel profile")
                preskip = int.from_bytes(payload[10:12], "little")
                if not 0 < preskip < PACKET:
                    raise ValueError("Pre-skip must be within the first 20 ms packet")
            elif not payload.startswith(b"OpusTags"):
                raise ValueError("Invalid OpusTags page")
        else:
            size = 0
            for value in lace:
                size += value
                if value < 255:
                    if not size:
                        raise ValueError("Empty Opus audio packet")
                    sizes.append(size)
                    size = 0
            expected = presented + preskip if flags == 4 else len(sizes) * PACKET
            if granule != expected:
                raise ValueError("Opus page granules disagree with its presentation clock")
        offset = end
        sequence += 1
    if preskip is None or len(sizes) != (presented + preskip + PACKET - 1) // PACKET or not 0 < len(sizes) * PACKET - granule < PACKET:
        raise ValueError("Expected the full bounded Ogg presentation and verified end trim")
    return preskip, granule, sizes


def native_clock(path: Path, *, packet_bytes: int | None = None, channels: int = 1, presented: int = PRESENTED) -> dict:
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Ogg Opus exceeds the bounded verification limit")
    preskip, granule, sizes = page_counts(data, channels=channels, presented=presented)
    if packet_bytes is not None and any(size != packet_bytes for size in sizes):
        raise ValueError("Opus packets do not match the qualified constant bitrate")
    command = ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries",
               "packet=pts,dts,duration,size:packet_side_data=side_data_type,skip_samples,discard_padding:"
               "stream=codec_name,codec_type,sample_rate,channels,time_base,start_pts,duration_ts,extradata_size",
               "-of", "json", str(path)]
    result = json.loads(subprocess.run(command, check=True, capture_output=True, timeout=15).stdout)
    stream = {"codec_name": "opus", "codec_type": "audio", "sample_rate": "48000", "channels": channels,
              "time_base": "1/48000", "start_pts": 0, "duration_ts": granule, "extradata_size": 19}
    if result["streams"] != [stream] or len(result["packets"]) != len(sizes):
        raise ValueError("Opus probe does not match the native page inventory")
    trim = len(sizes) * PACKET - granule
    for index, (packet, size) in enumerate(zip(result["packets"], sizes, strict=True)):
        first, last = index == 0, index == len(sizes) - 1
        pts = index * PACKET - preskip
        expected = (pts, pts, PACKET - trim if last else PACKET, size)
        actual = packet["pts"], packet["dts"], packet["duration"], int(packet["size"])
        expected_side = ([{"side_data_type": "Skip Samples", "skip_samples": preskip if first else 0,
                           "discard_padding": trim if last else 0}] if first or last else [])
        if actual != expected or packet.get("side_data_list", []) != expected_side:
            raise ValueError("Opus packet timing or padding disagrees with the native pages")
    return {"sample_rate": 48_000, "channels": channels, "presentation_start_pts": 0,
            "presentation_end_exclusive_pts": presented, "end_granule": granule, "pre_skip": preskip,
            "end_trim": trim, "packets": len(sizes), "packet_payload_bit_rate": round(sum(sizes) * 8 * 48000 / presented)}


def read(path: Path) -> dict | None:
    """Return proof only for the locally qualified 64 kb/s profile."""
    try:
        clock = native_clock(path, packet_bytes=PACKET_BYTES)
        return clock if clock["pre_skip"] == PRE_SKIP else None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None
