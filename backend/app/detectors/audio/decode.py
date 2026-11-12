"""A reproducible audio input and its source timeline, independent of browser decoding.

The first audio stream is decoded once into the exact mono s16 / 16 kHz input
used by voice checks. Each emitted frame records its timestamp before the WAV
timeline is reset. Gaps therefore remain visible in the mapping instead of
silently implying that concatenated PCM has a continuous source timeline.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.detectors.video.decode import _input
from app.evidence.stereo_binding import MAX_SECONDS as STEREO_MAX_SECONDS

SAMPLE_RATE = 16_000
VERSION = "audio-input/1"
_FRAME = re.compile(r"\bn:(\d+)\s+pts:(-?\d+).*?\bnb_samples:(\d+)")


class AudioStreamUnavailable(ValueError):
    """The container has no audio stream to decode."""


@dataclass(frozen=True)
class DecodedAudio:
    pcm: np.ndarray
    wav: bytes
    mapping: dict[str, Any]
    stereo_pcm: np.ndarray | None = None


def read_wav(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, SAMPLE_RATE, "NONE"):
            raise ValueError("The analysed audio does not have its recorded PCM format")
        if not 0 < wav.getnframes() <= SAMPLE_RATE * 600:
            raise ValueError("The analysed audio exceeds its bounded sample count")
        frames = wav.readframes(wav.getnframes())
        if len(frames) != 2 * wav.getnframes():
            raise ValueError("The analysed audio is truncated")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def wav_bytes(pcm: np.ndarray) -> bytes:
    values = np.asarray(pcm, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Invalid mono PCM")
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(np.clip(np.rint(values * 32768.0), -32768, 32767).astype("<i2").tobytes())
    return out.getvalue()


def _number(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _mapping(log: str, samples: int) -> list[dict[str, int]]:
    blocks, offset = [], 0
    for n, pts, count in _FRAME.findall(log):
        count = min(int(count), samples - offset)
        if count <= 0:
            break
        if int(n) != len(blocks):
            raise ValueError("The decoded audio timeline is incomplete")
        blocks.append({"start_sample": offset, "end_sample": offset + count, "source_pts": int(pts)})
        offset += count
    if offset != samples:
        raise ValueError("The decoded audio timeline does not cover the analysed samples")
    # Coalesce only truly adjacent timestamps, preserving every discontinuity.
    merged: list[dict[str, int]] = []
    for block in blocks:
        if merged and merged[-1]["source_pts"] + merged[-1]["end_sample"] - merged[-1]["start_sample"] == block["source_pts"]:
            merged[-1]["end_sample"] = block["end_sample"]
        else:
            merged.append(dict(block))
    return merged


def _named_timeline(log: str, name: str) -> str:
    return "\n".join(line for line in log.splitlines() if line.startswith(f"[ashowinfo@{name} "))


def _read_stereo(path: Path, limit: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        count = wav.getnframes()
        if ((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
                != (2, 2, SAMPLE_RATE, "NONE") or not 0 < count <= limit):
            raise ValueError("The stereo evidence PCM format or duration is invalid")
        frames = wav.readframes(count)
        if len(frames) != 4 * count:
            raise ValueError("The stereo evidence PCM is truncated")
    return np.frombuffer(frames, dtype="<i2").reshape(-1, 2)


def decode_audio(path: Path, *, source_sha256: str, max_seconds: float = 600.0,
                 inspect_stereo: bool = False) -> DecodedAudio:
    if not (0 < max_seconds <= 600):
        raise ValueError("Audio decoding is bounded to 600 seconds")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required for audio evidence")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source_sha256:
            raise ValueError("The audio decoder input differs from the recorded original")
    try:
        entries = ("stream=index,channels,channel_layout,sample_rate,codec_name,bit_rate,bits_per_raw_sample,time_base,start_pts,duration"
                   ":format=format_name,start_time,duration")
        probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a",
                                "-show_entries", entries, "-of", "json", _input(path)],
                               capture_output=True, text=True, timeout=60, check=True, stdin=subprocess.DEVNULL)
        info = json.loads(probe.stdout)
        streams = info.get("streams") or []
        if not streams:
            raise AudioStreamUnavailable("The file has no audio stream")
        stream = streams[0]
        maximum = math.floor(max_seconds * SAMPLE_RATE)
        filters = f"aformat=sample_fmts=s16:channel_layouts=mono,aresample={SAMPLE_RATE},atrim=end_sample={maximum},ashowinfo,asetpts=N/SR/TB"
        layout = stream.get("channel_layout")
        stereo_pcm, stereo_clock = None, None
        capture_stereo = False
        if inspect_stereo:
            from app.detectors.audio import stereo
            capture_stereo = maximum >= SAMPLE_RATE and stereo.supports_stereo(int(stream["channels"]), layout)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "decoded.wav"
            command = ["ffmpeg", "-nostdin", "-v", "info", "-y", "-protocol_whitelist", "file", "-copyts", "-i", _input(path)]
            mono_output = ["-vn", "-t", str(max_seconds), "-c:a", "pcm_s16le", "-f", "wav", str(output)]
            if capture_stereo:
                limit = min(maximum, STEREO_MAX_SECONDS * SAMPLE_RATE)
                stereo_path = Path(temp) / "stereo.wav"
                mono_filters = filters.replace("ashowinfo", "ashowinfo@mono")
                graph = (f"[0:a:0]asplit=2[mix][channels];[mix]{mono_filters}[mono];"
                         f"[channels]aresample={SAMPLE_RATE},atrim=end_sample={limit},"
                         "ashowinfo@stereo,asetpts=N/SR/TB[stereo]")
                run = subprocess.run(
                    command + ["-filter_complex", graph, "-map", "[mono]"] + mono_output
                    + ["-map", "[stereo]", "-vn", "-c:a", "pcm_s16le", "-f", "wav", str(stereo_path)],
                    capture_output=True, text=True, timeout=180, stdin=subprocess.DEVNULL,
                )
                if run.returncode == 0:
                    try:
                        stereo_pcm = _read_stereo(stereo_path, limit)
                        stereo_clock = {"samples": len(stereo_pcm), "timeline_time_base": f"1/{SAMPLE_RATE}",
                                        "timeline": _mapping(_named_timeline(run.stderr, "stereo"), len(stereo_pcm))}
                    except (OSError, ValueError, wave.Error, EOFError):
                        stereo_pcm = None
                else:
                    capture_stereo = False
            if capture_stereo:
                try:
                    pcm = read_wav(output.read_bytes())
                    timeline = _mapping(_named_timeline(run.stderr, "mono"), len(pcm))
                except (OSError, ValueError, wave.Error, EOFError):
                    capture_stereo = False
                    stereo_pcm, stereo_clock = None, None
            if not capture_stereo:
                run = subprocess.run(command + ["-map", "0:a:0", "-af", filters] + mono_output,
                                     capture_output=True, text=True, timeout=180, check=True, stdin=subprocess.DEVNULL)
                pcm = read_wav(output.read_bytes())
                timeline = _mapping(run.stderr, len(pcm))
        fmt = info.get("format") or {}
        mapping = {
            "version": VERSION, "source_sha256": source_sha256,
            "sample_rate": SAMPLE_RATE, "channels": 1, "sample_format": "s16le",
            "samples": len(pcm), "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
            "source_stream_index": int(stream["index"]), "source_sample_rate": int(stream["sample_rate"]),
            "source_channels": int(stream["channels"]), "source_codec": stream.get("codec_name"),
            "source_time_base": stream.get("time_base"), "source_start_pts": stream.get("start_pts"),
            "container_start_s": _number(fmt.get("start_time")),
            "source_duration_s": _number(stream.get("duration")) or _number(fmt.get("duration")),
            "timeline_time_base": f"1/{SAMPLE_RATE}", "timeline": timeline,
            "decoder": next((line for line in run.stderr.splitlines() if line.startswith("ffmpeg version ")), "ffmpeg"),
            "transform": "First audio stream; decoder trims codec priming; mono mix; 16000 Hz resampling; signed 16-bit quantisation. WAV starts at analysed sample zero.",
            "source_timing_precision_s": 1 / SAMPLE_RATE,
            "limit_samples": maximum,
        }
        if mapping["source_codec"] == "flac":
            mapping["source_container"] = fmt.get("format_name")
            bits = _number(stream.get("bits_per_raw_sample"))
            mapping["source_bit_depth"] = int(bits) if bits is not None else None
        if mapping["source_codec"] == "pcm_s16le" and fmt.get("format_name") == "wav":
            mapping.update(source_container="wav", source_bit_depth=16)
        if mapping["source_codec"] == "mp3":
            mapping["source_container"] = fmt.get("format_name")
            bit_rate = _number(stream.get("bit_rate"))
            mapping["source_bit_rate"] = int(bit_rate) if bit_rate is not None else None
        if len(streams) > 1:
            mapping["audio_stream_indices"] = [int(track["index"]) for track in streams]
        verified_stereo = None
        if inspect_stereo:
            mapping["stereo_context"] = stereo.build(mapping, pcm, layout, stereo_pcm, stereo_clock)
            if stereo_pcm is not None and mapping["stereo_context"]["stereo_pcm_sha256"] is not None:
                verified_stereo = stereo_pcm[:len(mapping["stereo_context"]["windows"]) * SAMPLE_RATE]
        return DecodedAudio(pcm, wav_bytes(pcm), mapping, verified_stereo)
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, IndexError, OSError) as exc:
        raise ValueError("The audio stream could not be decoded with a verified sample timeline") from exc


def source_ranges(mapping: dict[str, Any], start: int, end: int) -> list[dict[str, Any]]:
    """Map a half-open analysed interval to rational source timestamps, retaining gaps."""
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= mapping["samples"]):
        raise ValueError("Audio evidence interval lies outside the analysed signal")
    if mapping.get("timeline_time_base") != f"1/{SAMPLE_RATE}":
        raise ValueError("Audio source timeline has an unexpected time base")
    previous = 0
    for block in mapping["timeline"]:
        values = (block.get("start_sample"), block.get("end_sample"), block.get("source_pts"))
        if (any(not isinstance(v, int) or isinstance(v, bool) for v in values)
                or values[0] != previous or not values[0] < values[1] <= mapping["samples"]):
            raise ValueError("Audio source timeline does not partition the analysed PCM")
        previous = values[1]
    if previous != mapping["samples"]:
        raise ValueError("Audio source timeline does not cover the analysed PCM")
    result = []
    for block in mapping["timeline"]:
        left, right = max(start, block["start_sample"]), min(end, block["end_sample"])
        if left >= right:
            continue
        source_start = block["source_pts"] + left - block["start_sample"]
        result.append({"start_sample": left, "end_sample": right, "source_start_pts": source_start,
                       "source_end_pts": source_start + right - left, "time_base": mapping["timeline_time_base"]})
    if sum(r["end_sample"] - r["start_sample"] for r in result) != end - start:
        raise ValueError("Audio evidence interval has an incomplete source mapping")
    return result
