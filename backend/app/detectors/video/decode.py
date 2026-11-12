"""pts-accurate video decode (ARCHITECTURE §3.1 `probe` and `decode`; VF §8.1).

- **Frame table:** `ffprobe -show_entries frame=pts,best_effort_timestamp,pict_type,key_frame` on the first video stream,
  capped at 108k frames. Table order is the decoder's output order, so frame index `n` is the n-th decoded frame. A frame
  without `pts` takes `best_effort_timestamp`; a frame with neither is interpolated and listed in `interpolated`.
- **Decode:** one ffmpeg child process per pass, fixed argv, no shell, `file` protocol only. `select` keeps the requested
  frame indices (`n`), `showinfo` reports each kept frame's pts, and rgb24 frames are read from a rawvideo pipe. ffmpeg
  applies the display matrix, so frames are in display orientation. `-copyts` keeps stream timestamps, and every decoded
  pts is checked against the table, so a frame's time is exactly `pts × time_base` (never `index / fps`).
- **Low-resolution stream:** `max_side=640` scales frames so the long side is at most 640 px (never upscaled).
- **Process isolation:** the decoder runs in its own process. A crash or hang on a malformed file ends that process, not
  the worker: a watchdog kills it after the timeout, `check` (the analysis deadline) runs between frames, and the child is
  killed on every exit path, including a consumer that stops reading early.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from app.core.logging import get_logger

log = get_logger(__name__)

MAX_TABLE_FRAMES = 108_000  # ARCHITECTURE §3.1: one hour at 30 fps
LOW_RES_PX = 640
PROBE_TIMEOUT_S = 900.0
DECODE_TIMEOUT_S = 1800.0
INFO_TIMEOUT_S = 60.0  # the showinfo line of a frame arrives before its pixels; this only guards a broken pipe
_SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+|NOPTS)")
_TIME_BASE = re.compile(r"config in time_base:\s*(\d+)/(\d+)")
_EOF = object()


class DecodeError(ValueError):
    """User-facing decode failure: messages contain "could not be decoded" (`core.errors.FRIENDLY`) and no paths."""


@dataclass(frozen=True)
class DecodedFrame:
    index: int  # frame-table index
    pts: int
    rgb: np.ndarray  # uint8 (h, w, 3), display orientation
    pts_verified: bool = True  # decoder supplied a timestamp and it was not inferred in the table


@dataclass(frozen=True)
class FrameTable:
    time_base: str  # "1/30000"
    pts: tuple[int, ...]
    key_frames: tuple[int, ...]  # table indices of key frames
    pict_types: str  # one character per frame: I, P, B, S or ?
    width: int  # display orientation (after the display matrix)
    height: int
    coded_width: int
    coded_height: int
    rotation: int = 0  # display-matrix rotation in degrees, as ffprobe reports it
    truncated: bool = False  # the stream has more frames than the cap
    interpolated: tuple[int, ...] = ()  # frames with neither pts nor best_effort_timestamp

    def __len__(self) -> int:
        return len(self.pts)

    @property
    def tb(self) -> Fraction:
        return parse_time_base(self.time_base)

    def seconds(self, index: int) -> float:
        """Exactly `pts × time_base`, in seconds."""
        return float(self.pts[index] * self.tb)

    def times(self) -> np.ndarray:
        return np.asarray([self.seconds(i) for i in range(len(self.pts))], dtype=np.float64)

    def to_json(self) -> dict[str, Any]:
        return {"time_base": self.time_base, "pts": list(self.pts), "key_frames": list(self.key_frames),
                "pict_types": self.pict_types, "width": self.width, "height": self.height, "coded_width": self.coded_width,
                "coded_height": self.coded_height, "rotation": self.rotation, "truncated": self.truncated,
                "interpolated": list(self.interpolated)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FrameTable:
        return cls(time_base=str(data["time_base"]), pts=tuple(int(p) for p in data["pts"]),
                   key_frames=tuple(int(k) for k in data["key_frames"]), pict_types=str(data["pict_types"]),
                   width=int(data["width"]), height=int(data["height"]), coded_width=int(data["coded_width"]),
                   coded_height=int(data["coded_height"]), rotation=int(data.get("rotation") or 0),
                   truncated=bool(data.get("truncated")), interpolated=tuple(int(i) for i in data.get("interpolated") or ()))


def parse_time_base(value: str) -> Fraction:
    num, _, den = str(value).partition("/")
    tb = Fraction(int(num), int(den or 1))
    if tb <= 0:
        raise ValueError(f"invalid time base {value!r}")
    return tb


def scaled_size(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    if not max_side or max(width, height) <= max_side:
        return width, height
    s = max_side / max(width, height)
    return max(1, round(width * s)), max(1, round(height * s))


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise DecodeError(f"Video could not be decoded: {name} is not installed")
    return exe


def _input(path: Path) -> str:
    return "file:" + str(Path(path).resolve())  # an explicit protocol, so a file name is never read as a URL or option


def _stream_info(path: Path) -> dict[str, Any]:
    argv = [_tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,time_base:stream_side_data=rotation", "-of", "json", _input(path)]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        raise DecodeError("Video could not be decoded: the stream probe failed") from e
    if not streams or not streams[0].get("width") or not streams[0].get("time_base"):
        log.warning("video.decode.no_stream", stderr=(r.stderr or "")[-300:])
        raise DecodeError("Video could not be decoded: no video stream")
    return streams[0]


def _rotation(stream: dict[str, Any]) -> int:
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            return int(round(float(side["rotation"])))
    return 0


def _fields(line: str) -> dict[str, str]:
    return dict(part.split("=", 1) for part in line.rstrip("\n").split("|")[1:] if "=" in part)


def probe_frames(path: Path, *, cap: int = MAX_TABLE_FRAMES, check: Callable[[], None] | None = None,
                 timeout_s: float = PROBE_TIMEOUT_S, max_pixels: int | None = None) -> FrameTable:
    """The per-frame pts table of the first video stream (decodes every frame once, in a child process)."""
    stream = _stream_info(path)
    rotation = _rotation(stream)
    coded_w, coded_h = int(stream["width"]), int(stream["height"])
    width, height = (coded_h, coded_w) if abs(rotation) % 180 == 90 else (coded_w, coded_h)
    if max_pixels is not None and width * height > max_pixels:
        raise DecodeError(f"Video could not be decoded for comparison above {max_pixels:,} pixels per picture.")
    decoder_limit = ["-max_pixels", str(max_pixels)] if max_pixels is not None else []
    argv = [_tool("ffprobe"), "-v", "error", "-protocol_whitelist", "file", *decoder_limit, "-select_streams", "v:0",
            "-show_entries", "frame=pts,best_effort_timestamp,pict_type,key_frame", "-of", "compact", _input(path)]
    pts: list[int | None] = []
    keys: list[int] = []
    types: list[str] = []
    truncated = False
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    timed_out = threading.Event()

    def watchdog() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, watchdog)
    timer.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if not line.startswith("frame|"):
                continue
            if len(pts) >= cap:
                truncated = True
                break
            f = _fields(line)
            value = f.get("pts") if f.get("pts") not in (None, "N/A") else f.get("best_effort_timestamp")
            pts.append(int(value) if value not in (None, "N/A") else None)
            if f.get("key_frame") == "1":
                keys.append(len(pts) - 1)
            types.append((f.get("pict_type") or "?")[:1])
            if check is not None and len(pts) % 1000 == 0:
                check()
    finally:
        if proc.poll() is None:
            proc.kill()
        stderr = proc.communicate()[1] if proc.stderr else ""
        timer.cancel()
    if timed_out.is_set():
        raise DecodeError("Video could not be decoded: the frame probe timed out")
    if not pts:
        log.warning("video.decode.no_frames", returncode=proc.returncode, stderr=(stderr or "")[-300:])
        raise DecodeError("Video could not be decoded: no decodable video frames")
    filled, interpolated = _fill_missing(pts)
    return FrameTable(time_base=str(stream["time_base"]), pts=tuple(filled), key_frames=tuple(keys), pict_types="".join(types),
                      width=width, height=height, coded_width=coded_w, coded_height=coded_h, rotation=rotation,
                      truncated=truncated, interpolated=tuple(interpolated))


def _fill_missing(pts: list[int | None]) -> tuple[list[int], list[int]]:
    """Frames with no timestamp at all take the previous pts plus the median step (1 when unknown)."""
    known = [p for p in pts if p is not None]
    steps = np.diff(np.asarray(known, dtype=np.int64)) if len(known) > 1 else np.asarray([], dtype=np.int64)
    positive = steps[steps > 0]
    step = int(np.median(positive)) if len(positive) else 1
    out: list[int] = []
    interpolated: list[int] = []
    for i, p in enumerate(pts):
        if p is None:
            p = (out[-1] + step) if out else (known[0] - step * (pts.index(known[0]) - i) if known else i * step)
            interpolated.append(i)
        out.append(int(p))
    return out, interpolated


FLAT_SELECT_TERMS = 64


def _balanced_sum(terms: Sequence[str]) -> str:
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return f"({_balanced_sum(terms[:middle])})+({_balanced_sum(terms[middle:])})"


def select_expression(indices: Sequence[int]) -> str:
    """`select` expression over frame indices, with runs as `between` terms (commas escaped for the filter graph).

    Up to `FLAT_SELECT_TERMS` terms are a flat sum. Beyond that the sum is a balanced tree: ffmpeg 8.1 refuses a flat sum
    of 150 or more terms ("Cannot allocate memory" when the filter graph is built; 100 pass), while balanced sums of
    1,600 terms select exactly the requested frames (measured 2026-09-13, `data/src/v3_select/`)."""
    terms: list[str] = []
    start = prev = None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            terms.append(f"eq(n\\,{start})" if start == prev else f"between(n\\,{start}\\,{prev})")
            start = prev = i
    if start is not None:
        terms.append(f"eq(n\\,{start})" if start == prev else f"between(n\\,{start}\\,{prev})")
    return "+".join(terms) if len(terms) <= FLAT_SELECT_TERMS else _balanced_sum(terms)


def _read_stderr(stream: Any, infos: queue.Queue, tail: deque, time_bases: list[Fraction]) -> None:
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", "replace")
        if "showinfo" in line:
            m = _TIME_BASE.search(line)
            if m:
                time_bases.append(Fraction(int(m[1]), int(m[2])))
                continue
            m = _SHOWINFO.search(line)
            if m:
                infos.put(None if m[2] == "NOPTS" else int(m[2]))
                continue
            continue
        tail.append(line.strip())
    infos.put(_EOF)


def _checked_pts(table: FrameTable, index: int, raw: int | None, time_bases: list[Fraction]) -> int:
    expected = table.pts[index]
    if raw is None or index in table.interpolated:
        return expected
    tb = time_bases[-1] if time_bases else table.tb
    decoded = raw if tb == table.tb else round(Fraction(raw) * tb / table.tb)
    if decoded != expected:
        raise DecodeError(f"Video could not be decoded frame-accurately: frame {index} decoded with pts {decoded}, "
                          f"the frame table has {expected}")
    return expected


def decode_frames(path: Path, table: FrameTable, indices: Sequence[int] | None, *, max_side: int | None = None,
                  check: Callable[[], None] | None = None, timeout_s: float = DECODE_TIMEOUT_S,
                  max_pixels: int | None = None) -> Iterator[DecodedFrame]:
    """Decode the frames at `indices` (sorted, unique; None = every frame) in one pass, in table order.

    Frames the decoder does not deliver (a truncated tail) are simply not yielded; the caller compares.
    """
    wanted = None if indices is None else sorted({int(i) for i in indices})
    if wanted is not None:
        if not wanted:
            return
        if wanted[0] < 0 or wanted[-1] >= len(table):
            raise ValueError(f"frame index out of range for a table of {len(table)} frames")
    w, h = scaled_size(table.width, table.height, max_side)
    filters = ([f"select='{select_expression(wanted)}'"] if wanted is not None else []) + ["showinfo"]
    if (w, h) != (table.width, table.height):
        filters.append(f"scale={w}:{h}:flags=area")
    filters.append("format=rgb24")
    decoder_limit = ["-max_pixels", str(max_pixels)] if max_pixels is not None else []
    argv = [_tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "info", "-protocol_whitelist", "file", "-copyts",
            *decoder_limit, "-i", _input(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough", "-vf", ",".join(filters),
            "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    infos: queue.Queue = queue.Queue()
    tail: deque = deque(maxlen=20)
    time_bases: list[Fraction] = []
    reader = threading.Thread(target=_read_stderr, args=(proc.stderr, infos, tail, time_bases), daemon=True)
    reader.start()
    timed_out = threading.Event()

    def watchdog() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, watchdog)
    timer.start()
    frame_bytes = w * h * 3
    count = 0
    try:
        assert proc.stdout is not None
        while wanted is None or count < len(wanted):
            buf = bytearray(frame_bytes)
            view, got = memoryview(buf), 0
            while got < frame_bytes:
                n = proc.stdout.readinto(view[got:])
                if not n:
                    break
                got += n
            if got == 0:
                break
            if got != frame_bytes:
                raise DecodeError("Video could not be decoded: the decoder returned a truncated frame")
            try:
                raw = infos.get(timeout=INFO_TIMEOUT_S)
            except queue.Empty as e:
                raise DecodeError("Video could not be decoded: no frame information from the decoder") from e
            if raw is _EOF:
                raise DecodeError("Video could not be decoded: no frame information from the decoder")
            index = wanted[count] if wanted is not None else count
            if index >= len(table):
                break  # frames beyond a truncated table
            pts = _checked_pts(table, index, raw, time_bases)
            yield DecodedFrame(index, pts, np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3),
                               pts_verified=raw is not None and index not in table.interpolated)
            count += 1
            if check is not None:
                check()
        if wanted is not None and count == len(wanted):
            return  # every requested frame is out: stop the decoder instead of decoding the tail
        proc.wait(timeout=30)
        if timed_out.is_set():
            raise DecodeError("Video could not be decoded: the decoder timed out")
        if count == 0 and proc.returncode != 0:
            log.warning("video.decode.failed", returncode=proc.returncode, stderr=" | ".join(tail)[-300:])
            raise DecodeError(f"Video could not be decoded: the decoder exited with status {proc.returncode}")
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()
        reader.join(timeout=5)


def decode_stream(path: Path, table: FrameTable, *, max_side: int | None = LOW_RES_PX,
                  check: Callable[[], None] | None = None, timeout_s: float = DECODE_TIMEOUT_S) -> Iterator[DecodedFrame]:
    """Every frame once, in table order, at low resolution (the stream `shots` and `faces` read)."""
    return decode_frames(path, table, None, max_side=max_side, check=check, timeout_s=timeout_s)
