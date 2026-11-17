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
from app.evidence.audio_short_review_binding import (
    AAC_CONTAINER,
    AAC_FRAME,
    LOSSLESS_FORMATS,
    LOSSLESS_NATIVE_CLOCK,
    LOSSLESS_RATE,
)
from app.evidence.audio_two_second_binding import MAXIMUM as LONG_WINDOW_MAXIMUM
from app.evidence.audio_two_second_binding import MINIMUM as LONG_WINDOW_MINIMUM
from app.evidence.stereo_binding import MAX_SECONDS as STEREO_MAX_SECONDS

SAMPLE_RATE = 16_000
VIDEO_AUDIO_MAX_SECONDS = 120
VERSION = "audio-input/1"
_FRAME = re.compile(r"\bn:(\d+)\s+pts:(-?\d+).*?\bnb_samples:(\d+)")


class AudioStreamUnavailable(ValueError):
    """The container has no audio stream to decode."""


def _probe(path: Path, *, exact_duration: bool = False) -> dict:
    fields = "index,channels,channel_layout,sample_rate,codec_name,bit_rate,bits_per_raw_sample,time_base,start_pts,duration"
    if exact_duration:
        fields += ",duration_ts,profile"
    entries = f"stream={fields}:format=format_name,start_time,duration"
    probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a",
                            "-show_entries", entries, "-of", "json", _input(path)],
                           capture_output=True, text=True, timeout=60, check=True, stdin=subprocess.DEVNULL)
    return json.loads(probe.stdout)


def _aac_presentation_stream(path: Path) -> dict | None:
    try:
        info = _probe(path, exact_duration=True)
        if info.get("format", {}).get("format_name") != "mov,mp4,m4a,3gp,3g2,mj2":
            return None
        stream, = info["streams"]
        duration = stream["duration_ts"]
        return stream if stream.get("codec_name") == "aac" and type(duration) is int and duration > 0 else None
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError):
        return None


def aac_native_extent(path: Path) -> int | None:
    """Read the exact presentation extent of a 16 kHz mono M4A."""
    stream = _aac_presentation_stream(path)
    if stream and (stream.get("sample_rate"), stream.get("channels"), stream.get("time_base"),
                   stream.get("start_pts")) == ("16000", 1, "1/16000", 0):
        return stream["duration_ts"]
    return None


def aac48_presentation(path: Path) -> dict | None:
    """Read the native clock and channel count of a supported 48 kHz AAC-LC source."""
    stream = _aac_presentation_stream(path)
    if (stream and (stream.get("sample_rate"), stream.get("time_base"), stream.get("start_pts"), stream.get("profile"))
            == ("48000", "1/48000", 0, "LC") and (stream.get("channels"), stream.get("channel_layout"))
            in {(1, "mono"), (2, "stereo")} and stream["duration_ts"] % 3 == 0):
        return {"sample_rate": 48_000, "channels": stream["channels"], "time_base": "1/48000",
                "presentation_end_exclusive_pts": stream["duration_ts"]}
    return None


def audio_stream_indices(path: Path) -> tuple[int, ...]:
    """Return the ordered audio stream indices from the source container."""
    try:
        return tuple(int(stream["index"]) for stream in _probe(path).get("streams", ()))
    except (subprocess.SubprocessError, TypeError, ValueError, KeyError, OSError) as error:
        raise ValueError("The audio stream inventory could not be read") from error


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


def _exact_zero_based_clock(log: str, samples: int) -> bool:
    if sum(int(count) for _, _, count in _FRAME.findall(log)) != samples:
        return False
    try:
        return _mapping(log, samples) == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}]
    except ValueError:
        return False


def _native_presentation_clock(log: str, extent: int, padding: int) -> bool:
    total = sum(int(count) for _, _, count in _FRAME.findall(log))
    return extent <= total <= extent + padding and _exact_zero_based_clock(log, total)


def _read_stereo(path: Path, limit: int, *, rate: int = SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        count = wav.getnframes()
        if ((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype())
                != (2, 2, rate, "NONE") or not 0 < count <= limit):
            raise ValueError("The stereo evidence PCM format or duration is invalid")
        frames = wav.readframes(count)
        if len(frames) != 4 * count:
            raise ValueError("The stereo evidence PCM is truncated")
    return np.frombuffer(frames, dtype="<i2").reshape(-1, 2)


def decode_native_stereo(path: Path, mapping: dict[str, Any], *, aac: bool = False) -> bytes | None:
    """Keep short 48 kHz stereo samples only when both decoded clocks agree."""
    from app.evidence.audio_channel_alignment_binding import (
        NATIVE_FACTOR,
        NATIVE_RATE,
        aac_presented_samples,
        native_scope,
    )

    if aac:
        count = aac_presented_samples(mapping)
    else:
        count = mapping["samples"] * NATIVE_FACTOR if native_scope(mapping) else None
    if count is None:
        return None
    expected_pts = mapping["timeline"][0]["source_pts"] * NATIVE_FACTOR
    # AAC can decode beyond the presented final sample. Lossless inputs must
    # end on the same sample in both decodes without trimming any extra samples.
    limit = count if aac else count + 1
    filters = (f"aformat=sample_fmts=s16,asettb=1/{NATIVE_RATE},atrim=end_sample={limit},"
               "ashowinfo,asetpts=N/SR/TB")
    try:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "native-stereo.wav"
            run = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "info", "-y", "-protocol_whitelist", "file", "-copyts", "-i", _input(path),
                 "-map", f"0:{mapping['source_stream_index']}", "-af", filters, "-vn", "-c:a", "pcm_s16le", "-f", "wav", str(output)],
                capture_output=True, text=True, timeout=20, check=True, stdin=subprocess.DEVNULL,
            )
            pcm = _read_stereo(output, limit, rate=NATIVE_RATE)
        if (len(pcm) != count or _mapping(run.stderr, count) != [
                {"start_sample": 0, "end_sample": count, "source_pts": expected_pts}]):
            return None
    except (subprocess.SubprocessError, OSError, ValueError, wave.Error, EOFError):
        return None
    return pcm.tobytes()


def decode_audio(path: Path, *, source_sha256: str, max_seconds: float = 600.0,
                 inspect_stereo: bool = False, stream_position: int = 0,
                 retain_stereo: bool = False) -> DecodedAudio:
    if not (0 < max_seconds <= 600):
        raise ValueError("Audio decoding is bounded to 600 seconds")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg and ffprobe are required for audio evidence")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source_sha256:
            raise ValueError("The audio decoder input differs from the recorded original")
    if type(stream_position) is not int or stream_position < 0:
        raise ValueError("Invalid audio stream position")
    try:
        info = _probe(path, exact_duration=True)
        streams = info.get("streams") or []
        if not streams:
            raise AudioStreamUnavailable("The file has no audio stream")
        if stream_position >= len(streams):
            raise AudioStreamUnavailable("The selected audio stream is not present")
        stream = streams[stream_position]
        fmt = info.get("format") or {}
        input_stream = f"0:a:{stream_position}"
        maximum = math.floor(max_seconds * SAMPLE_RATE)
        filters = f"aformat=sample_fmts=s16:channel_layouts=mono,aresample={SAMPLE_RATE},atrim=end_sample={maximum},ashowinfo,asetpts=N/SR/TB"
        layout = stream.get("channel_layout")
        source_format = (fmt.get("format_name"), stream.get("codec_name"))
        native_extent = stream.get("duration_ts")
        lossless_clock = source_format in LOSSLESS_FORMATS
        aac_clock = (source_format == (AAC_CONTAINER, "aac") and stream.get("profile") == "LC"
                     and stream.get("start_pts") == 0)
        longer_extent = (type(native_extent) is int and native_extent % 3 == 0
                         and LONG_WINDOW_MINIMUM <= native_extent // 3 <= LONG_WINDOW_MAXIMUM)
        six_second_lossless = lossless_clock and native_extent == 6 * LOSSLESS_RATE and max_seconds >= 7
        inspect_native_clock = (
            len(streams) == 1 and (stream.get("channels"), stream.get("sample_rate"), stream.get("time_base"))
            == (1, str(LOSSLESS_RATE), f"1/{LOSSLESS_RATE}")
            and (six_second_lossless or (longer_extent and max_seconds >= 15 and (lossless_clock or aac_clock)))
        )
        native_clock_verified = False
        stereo_pcm, stereo_clock = None, None
        capture_stereo = False
        if inspect_stereo or retain_stereo:
            from app.detectors.audio import stereo
            capture_stereo = (retain_stereo or maximum >= SAMPLE_RATE) and stereo.supports_stereo(int(stream["channels"]), layout)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "decoded.wav"
            command = ["ffmpeg", "-nostdin", "-v", "info", "-y", "-protocol_whitelist", "file", "-copyts", "-i", _input(path)]
            mono_output = ["-vn", "-t", str(max_seconds), "-c:a", "pcm_s16le", "-f", "wav", str(output)]
            if capture_stereo:
                limit = min(maximum, STEREO_MAX_SECONDS * SAMPLE_RATE)
                stereo_path = Path(temp) / "stereo.wav"
                mono_filters = filters.replace("ashowinfo", "ashowinfo@mono")
                graph = (f"[{input_stream}]asplit=2[mix][channels];[mix]{mono_filters}[mono];"
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
                input_filters = ("ashowinfo@original," + filters.replace("ashowinfo", "ashowinfo@mono")
                                 if inspect_native_clock else filters)
                run = subprocess.run(command + ["-map", input_stream, "-af", input_filters] + mono_output,
                                     capture_output=True, text=True, timeout=180, check=True, stdin=subprocess.DEVNULL)
                pcm = read_wav(output.read_bytes())
                timeline = _mapping(_named_timeline(run.stderr, "mono") if inspect_native_clock else run.stderr, len(pcm))
                if inspect_native_clock:
                    native_clock_verified = _native_presentation_clock(
                        _named_timeline(run.stderr, "original"), native_extent, AAC_FRAME - 1 if aac_clock else 0,
                    )
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
        if (fmt.get("format_name") == "mov,mp4,m4a,3gp,3g2,mj2"
                and mapping["source_codec"] == "aac" and mapping["source_sample_rate"] == 48_000
                and type(stream.get("duration_ts")) is int):
            mapping["source_duration_pts"] = stream["duration_ts"]
            mapping["source_container"] = fmt["format_name"]
            mapping["source_profile"] = stream.get("profile")
            bit_rate = _number(stream.get("bit_rate"))
            mapping["source_bit_rate"] = int(bit_rate) if bit_rate is not None else None
        if mapping["source_codec"] == "flac":
            mapping["source_container"] = fmt.get("format_name")
            bits = _number(stream.get("bits_per_raw_sample"))
            mapping["source_bit_depth"] = int(bits) if bits is not None else None
        if mapping["source_codec"] == "pcm_s16le" and fmt.get("format_name") == "wav":
            mapping.update(source_container="wav", source_bit_depth=16)
        if ((mapping.get("source_container"), mapping["source_codec"]) in LOSSLESS_FORMATS
                and mapping["source_sample_rate"] == LOSSLESS_RATE and type(stream.get("duration_ts")) is int):
            mapping["source_duration_pts"] = stream["duration_ts"]
        if native_clock_verified:
            mapping[LOSSLESS_NATIVE_CLOCK] = True
        if mapping["source_codec"] == "mp3":
            mapping["source_container"] = fmt.get("format_name")
            bit_rate = _number(stream.get("bit_rate"))
            mapping["source_bit_rate"] = int(bit_rate) if bit_rate is not None else None
        if len(streams) > 1:
            mapping["audio_stream_indices"] = [int(track["index"]) for track in streams]
        if stream_position:
            mapping["transform"] = mapping["transform"].replace("First audio stream", "Selected audio stream", 1)
        verified_stereo = None
        if inspect_stereo:
            mapping["stereo_context"] = stereo.build(mapping, pcm, layout, stereo_pcm, stereo_clock)
            if stereo_pcm is not None and mapping["stereo_context"]["stereo_pcm_sha256"] is not None:
                verified_stereo = stereo_pcm[:len(mapping["stereo_context"]["windows"]) * SAMPLE_RATE]
        if (retain_stereo and stereo_clock is not None and stereo_pcm is not None and len(stereo_pcm) == len(pcm)
                and stereo_clock["timeline"] == timeline):
            verified_stereo = stereo_pcm
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
