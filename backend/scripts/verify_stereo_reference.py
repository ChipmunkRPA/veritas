"""Independently verify the stereo section of an earlier-audio JSON export."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import re
import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path
from struct import iter_unpack

RATE = 16000
BIN = 320
MAX_SAMPLES = RATE * 30
MAX_BYTES = 48 * 1024 * 1024
SCHEMA = "chipmunk.audio-reference-comparison/5"
SIDES = ("earlier", "submitted")
FRAME = re.compile(rb"\bn:(\d+)\s+pts:(-?\d+).*?\bnb_samples:(\d+)")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def command(*args):
    return subprocess.run(args, capture_output=True, check=True, timeout=60, stdin=subprocess.DEVNULL)


def decode(path, identity, source):
    size = path.stat().st_size
    with path.open("rb") as stream:
        require(0 < size <= MAX_BYTES and identity == {"bytes": size, "sha256": hashlib.file_digest(stream, "sha256").hexdigest()},
                "Original bytes differ from the export")
    report = command("ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0", "-show_entries",
                     "stream=index,channels,channel_layout,sample_rate", "-of", "json", str(path.resolve()))
    streams = json.loads(report.stdout)["streams"]
    require(len(streams) == 1 and streams[0]["channels"] == 2 and streams[0].get("channel_layout") in (None, "unknown", "stereo"),
            "The first audio stream needs recorded stereo channels")
    stream = streams[0]
    result = command("ffmpeg", "-v", "info", "-nostdin", "-protocol_whitelist", "file", "-copyts", "-i", str(path.resolve()),
                     "-map", "0:a:0", "-af", f"aresample={RATE},atrim=end_sample={MAX_SAMPLES + 1},ashowinfo",
                     "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1")
    raw = result.stdout
    require(len(raw) % 4 == 0 and 0 < len(raw) // 4 <= MAX_SAMPLES, "Decoded stereo duration exceeds the comparison scope")
    expected = {"channels": 2, "sample_rate": int(stream["sample_rate"]), "stream_index": stream["index"],
                "stereo_pcm_sha256": digest(raw)}
    require(source == expected, "Decoded stereo samples or stream identity differ")
    timing = [(int(i), int(pts), int(samples)) for i, pts, samples in FRAME.findall(result.stderr)]
    require(timing, "FFmpeg did not report a sample clock")
    origin = timing[0][1]
    count = 0
    for index, (frame, pts, samples) in enumerate(timing):
        require(frame == index and pts == origin + count and samples > 0, "Decoded stereo clock is discontinuous")
        count += samples
    require(count == len(raw) // 4, "Sample clock does not cover the decoded stereo")
    return tuple(zip(*iter_unpack("<hh", raw), strict=True)), origin


def waveform(samples: Sequence[int], *, residual=False):
    values = [sample / 32768 for sample in samples]
    digits = 8 if residual else 4
    result = {"bin_s": .01, "t": [], "low": [], "high": [], "rms": []}
    for start in range(0, len(values), 160):
        block = values[start:start + 160]
        result["t"].append(round((start + len(block) / 2) / RATE, 4))
        result["low"].append(round(min(block), digits))
        result["high"].append(round(max(block), digits))
        result["rms"].append(round(math.sqrt(math.fsum(sample ** 2 for sample in block) / len(block)), digits))
    return result


def verify_clip(encoded, reference, samples, name):
    raw = base64.b64decode(encoded[name], validate=True)
    require(reference == {"file": name, "media_type": "audio/wav", "sha256": digest(raw)}, "WAV asset commitment differs")
    with wave.open(io.BytesIO(raw), "rb") as decoded:
        require((decoded.getnchannels(), decoded.getsampwidth(), decoded.getframerate(), decoded.getnframes()) == (1, 2, RATE, len(samples)),
                "Preview has a different sample format or extent")
        actual = tuple(value for (value,) in iter_unpack("<h", decoded.readframes(len(samples))))
    require(actual == samples, "Preview differs from the corresponding source channel")


def close(actual, expected):
    return math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12)


def verify_channel(label, record, assets, first, second, origins):
    delta = [b - a for a, b in zip(first, second, strict=True)]
    counts = [sum(value != 0 for value in delta[a:a + BIN]) for a in range(0, len(delta), BIN)]
    ranges = []
    start = None
    for bin_index, changed in enumerate([*counts, 0]):
        if changed and start is None:
            start = bin_index
        elif not changed and start is not None:
            ranges.append((start, bin_index))
            start = None
    require((record["sample_rate"], record["bin_samples"], record["changed_samples"], record["bin_counts_sha256"])
            == (RATE, BIN, sum(counts), digest(json.dumps(counts, separators=(",", ":")).encode())), "Channel sample count differs")
    selected = set(sorted(ranges, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))[:3])
    clips = 0
    for number, (span, (low, high)) in enumerate(zip(record["ranges"], ranges, strict=True), 1):
        a, b = low * BIN, min(high * BIN, len(delta))
        offsets = [index for index, value in enumerate(delta[a:b], a) if value]
        expect = {"id": f"{label[0].upper()}{number:02d}", "start_sample": a, "end_sample": b,
                  "start_s": a / RATE, "end_s": b / RATE, "first_differing_sample": offsets[0],
                  "last_differing_sample": offsets[-1], "changed_samples": len(offsets),
                  "source_ranges": {side: {"start_pts": origin + a, "end_exclusive_pts": origin + b}
                                    for side, origin in zip(SIDES, origins, strict=True)}}
        require({key: span.get(key) for key in expect} == expect, "Measured source range differs")
        peak = max(abs(value) for value in delta[a:b]) / 32768
        rms = math.sqrt(math.fsum(value ** 2 for value in delta[a:b]) / (b - a)) / 32768
        require(close(span["difference"]["peak"], peak) and close(span["difference"]["rms"], rms), "Difference statistics differ")
        require(("preview" in span) == ((low, high) in selected), "Preview selection differs")
        if "preview" not in span:
            continue
        preview = span["preview"]
        end = min(b, a + 2 * RATE)
        require((preview["start_sample"], preview["end_sample"], preview["end_s"]) == (a, end, end / RATE), "Preview clock differs")
        require(preview["difference_waveform"] == waveform(delta[a:end], residual=True), "Difference waveform differs")
        for side, source in zip(SIDES, (first, second), strict=True):
            require(preview["waveforms"][side] == waveform(source[a:end]), "Source waveform differs")
            verify_clip(assets, preview["assets"][side], source[a:end], f"{label}-{side}-audio-{number:02d}.wav")
            clips += 1
    return {"differing_samples": sum(counts), "ranges": len(ranges), "previews_verified": clips}


def verify(earlier, submitted, export):
    package = json.loads(export.read_bytes())
    record, assets = package["comparison"], package["assets"]
    stereo = record["stereo_comparison"]
    require(record["schema"] == SCHEMA and stereo["status"] == "measured"
            and stereo["source_time_base"] == f"1/{RATE}" and stereo["affects_overall_confidence"] is False,
            "The export needs a measured version 5 stereo comparison")
    require(set(stereo["channels"]) == {"left", "right"}, "Stereo comparison must include both channels")
    first, first_pts = decode(earlier, record["earlier"], stereo["sources"]["earlier"])
    second, second_pts = decode(submitted, record["submitted"], stereo["sources"]["submitted"])
    require(first_pts == second_pts and len(first[0]) == len(second[0]), "The two source clocks differ")
    return {label: verify_channel(label, stereo["channels"][label], assets, before, after, (first_pts, second_pts))
            for label, before, after in zip(("left", "right"), first, second, strict=True)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("submitted", type=Path)
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.earlier, args.submitted, args.export), sort_keys=True))
    except (KeyError, TypeError, ValueError, OSError, wave.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f"Stereo evidence verification failed. {error}\n")
