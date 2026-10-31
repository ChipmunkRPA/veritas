"""Codec lane, model-free (V-6; VF §7, §9.6).

Every reading here is explanatory: it describes how the file was encoded and muxed, which splicing, trimming, re-muxing
or dubbing can change, and which ordinary editing and re-sharing change as well. Nothing here is decisive.

- **Streams:** ffprobe stream and format facts (codecs, nominal and average frame rate, start times, durations, tags).
- **GOP** from the frame table's key frames: GOP lengths between consecutive key frames. The last GOP can be cut short by
  the end of the file, so the dominant length is the most common complete GOP, established from at least 2 complete GOPs.
  A key frame after a GOP of another length is *irregular*; one within 1 frame of a detected shot cut is a scene-change
  key frame and is not counted. A first key frame after frame 0 means decoding starts mid-GOP.
- **Timestamps:** frame steps in pts ticks. Duplicates (step 0) and non-monotonic steps (< 0) are counted. The stream is
  variable frame rate when more than 5 % of the steps differ from the median step by more than 10 %. A gap is a step
  above 1.5 × the median, and it counts as an irregularity only in a constant frame-rate stream.
- **Packets:** first video stream packets: discarded packets (flag D, the frames an edit list hides) and non-monotonic dts.
- **Edit lists:** the `elst` boxes of an ISO BMFF (MP4 / MOV) file per track, with times in seconds (an empty edit delays
  the track). Other containers have none.
- **Encoder SEI:** the x264 and x265 identification strings in the first 256 MB of the file. Each separately encoded part
  carries one; more than one distinct configuration in a file is counted as an irregularity.
- **A/V:** audio minus video start time and duration; a difference above 0.5 s is an irregularity (VF §7).
- **Probability:** none without an irregularity. Otherwise 0.55 + 0.05 per further irregularity kind, capped at 0.70, with
  outcome weight 0.3 (VF §7 house rule: explanatory signals stay within [0.40, 0.70] and weight ≤ 0.3). Hand-set,
  uncalibrated.
"""
from __future__ import annotations

import json
import re
import struct
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from app.detectors.video.decode import DecodeError, FrameTable, _input, _tool

P_FIRST, P_STEP, P_CAP, OUTCOME_WEIGHT = 0.55, 0.05, 0.70, 0.3
AV_MISMATCH_S = 0.5
GAP_RATIO = 1.5
VFR_TOLERANCE, VFR_FRACTION = 0.10, 0.05
CUT_TOLERANCE_FRAMES = 1
MIN_GOPS = 2
MAX_PACKETS = 108_000
SEI_SCAN_BYTES = 256 * 1024 * 1024
SEI_CHUNK = 16 * 1024 * 1024
SEI_OVERLAP = 4096  # longer than any accepted identification string
MOOV_MAX_BYTES = 64 * 1024 * 1024
PROBE_TIMEOUT_S = 300.0
_X264 = re.compile(rb"x264 - core (\d+)[ -~]{0,200}? - options: ([ -~]{1,2000})")
_X265 = re.compile(rb"x265 \(build (\d+)\)[ -~]{0,2000}")
_BMFF_TOP = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"uuid", b"pdin", b"moof", b"mfra", b"sidx", b"styp", b"meta"}


def _float(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _run(argv: list[str]) -> str:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise DecodeError("Video could not be decoded: the codec probe timed out") from e
    if r.returncode != 0:
        raise DecodeError("Video could not be decoded: the codec probe failed")
    return r.stdout


# --- ffprobe facts ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Streams:
    video: dict[str, Any]
    audio: dict[str, Any] | None
    format: dict[str, Any]

    def tags(self) -> dict[str, str]:
        return {**{k.lower(): str(v)[:200] for k, v in (self.format.get("tags") or {}).items()},
                **{k.lower(): str(v)[:200] for k, v in (self.video.get("tags") or {}).items()}}


def stream_facts(path: Path) -> Streams:
    out = _run([_tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file", "-show_entries",
                "stream=index,codec_type,codec_name,profile,level,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,"
                "start_time,duration,nb_frames,has_b_frames,sample_rate,channels:stream_tags:format=format_name,start_time,"
                "duration,bit_rate:format_tags", "-of", "json", _input(path)])
    doc = json.loads(out or "{}")
    streams = doc.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise DecodeError("Video could not be decoded: no video stream")
    return Streams(video, next((s for s in streams if s.get("codec_type") == "audio"), None), doc.get("format") or {})


@dataclass(frozen=True)
class Packets:
    count: int
    discarded: int
    non_monotonic_dts: int
    capped: bool


def packet_facts(path: Path, cap: int = MAX_PACKETS) -> Packets:
    out = _run([_tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0", "-show_entries",
                "packet=dts,flags", "-of", "csv=p=0", _input(path)])
    count = discarded = non_monotonic = 0
    previous: int | None = None
    for line in out.splitlines():
        if not line.strip():
            continue
        if count >= cap:
            return Packets(count, discarded, non_monotonic, True)
        dts, _, flags = line.partition(",")
        count += 1
        discarded += "D" in flags
        if dts not in ("", "N/A"):
            value = int(dts)
            non_monotonic += previous is not None and value <= previous
            previous = value
    return Packets(count, discarded, non_monotonic, False)


# --- encoder SEI and edit lists ------------------------------------------------------------------------------------------

def encoder_sei(path: Path, limit: int = SEI_SCAN_BYTES) -> tuple[tuple[str, ...], int, bool]:
    """(distinct x264 / x265 identification strings, bytes scanned, scan capped)."""
    found: list[str] = []
    scanned = 0
    tail = b""
    with open(path, "rb") as f:
        while scanned < limit:
            chunk = f.read(min(SEI_CHUNK, limit - scanned))
            if not chunk:
                break
            scanned += len(chunk)
            data = tail + chunk
            complete = len(data) if len(chunk) < SEI_CHUNK else len(data) - 1  # a match touching the end may be truncated
            for m in _X264.finditer(data):
                text = f"x264 core {m.group(1).decode()}: {m.group(2).decode('ascii', 'replace').strip()}"
                if m.end() < complete and text not in found:
                    found.append(text)
            for m in _X265.finditer(data):
                text = m.group(0).decode("ascii", "replace").strip()
                if m.end() < complete and text not in found:
                    found.append(text)
            tail = data[-SEI_OVERLAP:]
        capped = bool(f.read(1))
    return tuple(found), scanned, capped


@dataclass(frozen=True)
class EditList:
    handler: str  # vide | soun | …
    entries: tuple[tuple[float, float | None], ...]  # (segment duration s, media time s or None for an empty edit)


def _boxes(data: bytes, start: int, end: int):
    while start + 8 <= end:
        size, kind = struct.unpack(">I4s", data[start:start + 8])
        header = 8
        if size == 1:
            if start + 16 > end:
                return
            size, header = struct.unpack(">Q", data[start + 8:start + 16])[0], 16
        elif size == 0:
            size = end - start
        if size < header or start + size > end:
            return
        yield kind, start + header, start + size
        start += size


def _timescale(data: bytes, a: int) -> int:
    return struct.unpack(">I", data[a + 12:a + 16] if data[a] == 0 else data[a + 20:a + 24])[0]


def _read_moov(path: Path) -> bytes | None:
    size = path.stat().st_size
    with open(path, "rb") as f:
        position, first = 0, True
        while position + 8 <= size:
            f.seek(position)
            header = f.read(16)
            box, kind = struct.unpack(">I4s", header[:8])
            if first and kind not in _BMFF_TOP:
                return None
            first = False
            length = 8
            if box == 1:
                box, length = struct.unpack(">Q", header[8:16])[0], 16
            elif box == 0:
                box = size - position
            if box < length:
                return None
            if kind == b"moov":
                f.seek(position + length)
                return f.read(min(box - length, MOOV_MAX_BYTES))
            position += box
    return None


def _track(moov: bytes, start: int, end: int, track: dict[str, Any]) -> dict[str, Any]:
    """Collect a trak box's edit entries (in movie and media ticks), handler type and media timescale."""
    for sub, x, y in _boxes(moov, start, end):
        if sub in (b"edts", b"mdia"):
            _track(moov, x, y, track)
        elif sub == b"elst" and y - x >= 8:
            version, n = moov[x], struct.unpack(">I", moov[x + 4:x + 8])[0]
            width = 20 if version == 1 else 12
            for k in range(min(n, (y - x - 8) // width)):
                p = x + 8 + k * width
                track["raw"].append(struct.unpack(">Qq", moov[p:p + 16]) if version == 1 else struct.unpack(">Ii", moov[p:p + 8]))
        elif sub == b"hdlr" and y - x >= 12:
            track["handler"] = moov[x + 8:x + 12].decode("ascii", "replace")
        elif sub == b"mdhd" and y - x >= 24:
            track["media_scale"] = _timescale(moov, x)
    return track


def edit_lists(path: Path) -> tuple[EditList, ...]:
    moov = _read_moov(path)
    if moov is None:
        return ()
    movie_scale = next((_timescale(moov, a) for kind, a, _ in _boxes(moov, 0, len(moov)) if kind == b"mvhd"), 0)
    out = []
    for kind, a, b in _boxes(moov, 0, len(moov)):
        if kind != b"trak":
            continue
        track = _track(moov, a, b, {"raw": [], "handler": "?", "media_scale": 0})
        if track["raw"] and movie_scale and track["media_scale"]:
            out.append(EditList(track["handler"], tuple((round(d / movie_scale, 6), None if t < 0 else round(t / track["media_scale"], 6))
                                                         for d, t in track["raw"])))
    return tuple(out)


# --- GOP and timestamps ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Gop:
    key_frames: tuple[int, ...]
    lengths: tuple[int, ...]  # frames from each key frame to the next (the last to the end of the table)
    dominant: int | None  # None: fewer than 2 complete GOPs
    irregular: tuple[int, ...]  # key frames after a complete GOP of another length, away from shot cuts
    at_cuts: tuple[int, ...]  # irregular key frames within 1 frame of a detected shot cut
    b_frames: int

    @property
    def starts_mid_gop(self) -> bool:
        return bool(self.key_frames) and self.key_frames[0] != 0

    @property
    def regular(self) -> bool:
        return self.dominant is not None and not self.irregular and not self.at_cuts


def gop_analysis(table: FrameTable, cuts: Sequence[int] = ()) -> Gop:
    keys = list(table.key_frames)
    lengths = [b - a for a, b in pairwise([*keys, len(table)])] if keys else []
    complete = lengths[:-1]
    dominant = Counter(complete).most_common(1)[0][0] if len(complete) >= MIN_GOPS else None
    off = [keys[i + 1] for i, length in enumerate(complete) if dominant is not None and length != dominant]
    near_cut = [k for k in off if any(abs(k - c) <= CUT_TOLERANCE_FRAMES for c in cuts)]
    return Gop(tuple(keys), tuple(lengths), dominant, tuple(k for k in off if k not in near_cut), tuple(near_cut),
               table.pict_types.count("B"))


@dataclass(frozen=True)
class Timing:
    median_step_s: float | None
    gaps: tuple[tuple[int, float], ...]  # (frame index after the gap, step in seconds)
    duplicates: int
    non_monotonic: int
    vfr: bool


def timing(table: FrameTable) -> Timing:
    steps = np.diff(np.asarray(table.pts, np.int64))
    positive = steps[steps > 0]
    if len(positive) == 0:
        return Timing(None, (), int((steps == 0).sum()), int((steps < 0).sum()), False)
    median = float(np.median(positive))
    vfr = len(positive) >= 10 and float(np.mean(np.abs(positive - median) > VFR_TOLERANCE * median)) > VFR_FRACTION
    tb = float(table.tb)
    gaps = tuple((i + 1, round(float(steps[i]) * tb, 6)) for i in np.flatnonzero(steps > GAP_RATIO * median))
    return Timing(round(median * tb, 6), gaps, int((steps == 0).sum()), int((steps < 0).sum()), vfr)


# --- the report --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Irregularity:
    kind: str  # irregular_key_frame | encoder_configurations | stream_copy_trim | av_start_mismatch | av_duration_mismatch | pts_gaps
    observed: str
    reasoning: str
    time_s: float | None = None


@dataclass(frozen=True)
class CodecReport:
    streams: Streams
    packets: Packets
    sei: tuple[str, ...]
    sei_scanned_bytes: int
    sei_capped: bool
    edits: tuple[EditList, ...]
    gop: Gop
    timing: Timing
    av_start_delta_s: float | None
    av_duration_delta_s: float | None
    irregularities: tuple[Irregularity, ...]

    @property
    def probability(self) -> float | None:
        k = len(self.irregularities)
        return None if k == 0 else round(min(P_CAP, P_FIRST + P_STEP * (k - 1)), 4)


def _delta(audio: dict[str, Any] | None, video: dict[str, Any], key: str) -> float | None:
    if audio is None:
        return None
    a, v = _float(audio.get(key)), _float(video.get(key))
    return None if a is None or v is None else round(a - v, 6)


def analyse(path: Path, table: FrameTable, cuts: Sequence[int] = (), shots_detected: bool = True) -> CodecReport:
    streams = stream_facts(path)
    packets = packet_facts(path)
    sei, scanned, capped = encoder_sei(path)
    edits = edit_lists(path)
    gop = gop_analysis(table, cuts)
    stamps = timing(table)
    start_delta, duration_delta = _delta(streams.audio, streams.video, "start_time"), _delta(streams.audio, streams.video, "duration")
    found: list[Irregularity] = []
    if gop.irregular:
        first = gop.irregular[0]
        note = "" if shots_detected else " Shot cuts were not detected for this run, so scene-change key frames cannot be told apart."
        found.append(Irregularity(
            "irregular_key_frame",
            f"{len(gop.irregular)} key frame(s) break the {gop.dominant}-frame GOP away from a detected shot cut (first at frame {first}, "
            f"{table.seconds(first):.2f} s)",
            "An extra key frame inside an otherwise fixed GOP is consistent with separately encoded parts joined at that point, and "
            "equally with an encoder that starts a GOP at a scene change the shot detector did not see (gradual transitions, "
            "flashes)." + note, table.seconds(first)))
    if len(sei) > 1:
        found.append(Irregularity(
            "encoder_configurations", f"{len(sei)} different encoder configurations in the stream",
            "Several encoder identification strings in one file are consistent with parts encoded separately and joined without "
            "re-encoding, and equally with a platform that stitches an intro or outro onto a clip."))
    if packets.discarded:
        media = next((e.entries[0][1] for e in edits if e.handler == "vide" and e.entries and e.entries[0][1] is not None), None)
        found.append(Irregularity(
            "stream_copy_trim",
            f"{packets.discarded} video packet(s) are discarded before the first shown frame"
            + (f" (the edit list starts {media:.2f} s into the stored stream)" if media is not None else ""),
            "Discarded leading packets are consistent with a trim by stream copy, which ordinary editing tools and platforms also "
            "produce."))
    if start_delta is not None and abs(start_delta) > AV_MISMATCH_S:
        found.append(Irregularity(
            "av_start_mismatch", f"the audio starts {start_delta:+.2f} s relative to the video",
            "A start difference above 0.5 s is consistent with a remux or a replaced sound track, and equally with muxers that "
            "delay one stream."))
    if duration_delta is not None and abs(duration_delta) > AV_MISMATCH_S:
        found.append(Irregularity(
            "av_duration_mismatch", f"the audio is {duration_delta:+.2f} s longer than the video",
            "A duration difference above 0.5 s is consistent with a replaced or re-cut sound track, and equally with recordings "
            "whose streams were stopped at different times."))
    if stamps.gaps and not stamps.vfr:
        largest = max(stamps.gaps, key=lambda g: g[1])
        found.append(Irregularity(
            "pts_gaps",
            f"{len(stamps.gaps)} timestamp gap(s) in a constant frame-rate stream (largest {largest[1] * 1000:.0f} ms before "
            f"frame {largest[0]}, {table.seconds(largest[0]):.2f} s)",
            "A gap in the frame timestamps is consistent with removed frames, and equally with frames dropped during capture or "
            "a screen recording.", table.seconds(largest[0])))
    return CodecReport(streams, packets, sei, scanned, capped, edits, gop, stamps, start_delta, duration_delta, tuple(found))
