"""Read initial luma quantization on a verified one-slice H.264 picture clock."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from statistics import median

SCHEMA = "h264-initial-slice-qp/1"
PREFIX = re.compile(r"^\[trace_headers @ 0x[0-9a-fA-F]+\] (.*)$")
FIELD = re.compile(r"^\s*\d+\s+(\w+)\s+[01]+\s+=\s+(-?\d+)\s*$")
PTS = re.compile(r"\bpts (-?\d+),")
SECTIONS = {"Sequence Parameter Set": "sps", "Picture Parameter Set": "pps", "Slice Header": "slice"}
TYPES = ("P", "B", "I")


class QuantizationUnavailable(ValueError):
    """The encoding trace has no complete, supported quantization reading."""


def parse_trace(lines, expected_pts):
    """Join each slice with its parameter set and the verified presentation tick."""
    active = None
    section = {}
    parameter_sets = {}
    readings = {}
    packet_pts = None
    packet_count = 0
    for line in lines:
        match = PREFIX.fullmatch(line)
        if not match:
            continue
        value = match[1]
        if value.startswith("Packet: "):
            packet = PTS.search(value)
            packet_pts = int(packet[1]) if packet else None
            packet_count += 1
            active = None
            continue
        if value in SECTIONS:
            active = SECTIONS[value]
            section = {}
            continue
        if active is None or not (item := FIELD.fullmatch(value)):
            continue
        name, raw = item.groups()
        field = int(raw)
        section[name] = field
        if active == "sps" and name == "bit_depth_luma_minus8" and field != 0:
            raise QuantizationUnavailable("Only eight-bit H.264 quantization is covered")
        if active == "pps" and name == "pic_init_qp_minus26":
            identifier = section.get("pic_parameter_set_id")
            if identifier is None or not 0 <= identifier <= 255 or not -26 <= field <= 25:
                raise QuantizationUnavailable("Incomplete H.264 picture parameter set")
            parameter_sets[identifier] = field
        if active == "slice" and name == "slice_qp_delta":
            kind = section.get("slice_type", -1)
            identifier = section.get("pic_parameter_set_id")
            if (packet_pts is None or packet_pts in readings or section.get("first_mb_in_slice") != 0
                    or not 0 <= kind <= 9 or kind % 5 >= len(TYPES) or identifier not in parameter_sets):
                raise QuantizationUnavailable("A picture lacks a supported unique H.264 slice")
            # ITU-T H.264 (08/2024), section 7.4.3, equation 7-32.
            qp = 26 + parameter_sets[identifier] + field
            if not 0 <= qp <= 51:
                raise QuantizationUnavailable("Initial slice quantization is outside the eight-bit range")
            readings[packet_pts] = [packet_pts, TYPES[kind % 5], qp]
    ticks = tuple(expected_pts)
    if (not ticks or packet_count != len(ticks) or len(readings) != len(ticks)
            or tuple(sorted(readings)) != ticks):
        raise QuantizationUnavailable("Coding headers differ from the observed picture clock")
    frames = [readings[tick] for tick in ticks]
    inter = [qp for _, kind, qp in frames if kind in ("P", "B")]
    if len(inter) < len(frames) / 2:
        raise QuantizationUnavailable("Too few inter pictures for quantization review")
    return {"schema": SCHEMA, "inter_picture_median": median(inter), "frames": frames}


def measure(path: Path, expected_pts, *, timeout=20):
    """Use FFmpeg's H.264 tracing bitstream filter with a bounded local input."""
    exe = shutil.which("ffmpeg")
    ticks = tuple(expected_pts)
    if exe is None or not ticks:
        raise QuantizationUnavailable("H.264 tracing or the picture clock is unavailable")
    args = [exe, "-hide_banner", "-nostdin", "-v", "info", "-xerror", "-protocol_whitelist", "file",
            "-i", "file:" + str(path.resolve()), "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
            "-bsf:v", "trace_headers", "-frames:v", str(len(ticks) + 1), "-f", "null", "-"]
    try:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                check=True, timeout=timeout, text=True, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError) as error:
        raise QuantizationUnavailable("H.264 slice trace was unavailable") from error
    return parse_trace(result.stderr.splitlines(), ticks)
