"""Independently verify a matched excerpt in an earlier-audio JSON export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import wave
from fractions import Fraction
from pathlib import Path
from struct import iter_unpack

import numpy as np

from scripts.review_video_reference_soundtrack_panel import localize
from scripts.verify_stereo_reference import (
    FRAME,
    MAX_BYTES,
    RATE,
    SIDES,
    command,
    digest,
    require,
    verify_clip,
    waveform,
)

SCHEMA = "chipmunk.audio-reference-comparison/6"
WINDOW = 2560
MIN_SAMPLES = 24000
MAX_SAMPLES = 192000


def decode(path, original):
    path = str(path.resolve())
    with open(path, "rb") as source:
        size = Path(path).stat().st_size
        require(0 < size <= MAX_BYTES, "Recording size exceeds the comparison scope")
        require(original == {"bytes": size, "sha256": hashlib.file_digest(source, "sha256").hexdigest()},
                "Original file differs from the export")
    probe = command(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,codec_name,bits_per_raw_sample,time_base:format=format_name", "-of", "json", path,
    )
    info = json.loads(probe.stdout)
    stream, = info["streams"]
    codec, container = stream["codec_name"], info["format"]["format_name"]
    eligible = (codec, container) == ("pcm_s16le", "wav") or (
        (codec, container) == ("flac", "flac") and stream.get("bits_per_raw_sample") == "16"
    )
    require(eligible and int(stream["sample_rate"]) == RATE and stream["channels"] == 1
            and Fraction(stream["time_base"]) == Fraction(1, RATE),
            "Timing evidence requires native 16 kHz mono PCM16 WAV or FLAC")
    audio = command(
        "ffmpeg", "-v", "info", "-nostdin", "-protocol_whitelist", "file", "-copyts", "-i", path,
        "-map", "0:a:0", "-af", f"atrim=end_sample={MAX_SAMPLES + 1},ashowinfo", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
    )
    frames = [(int(i), int(tick), int(length)) for i, tick, length in FRAME.findall(audio.stderr)]
    require(frames and len(audio.stdout) % 2 == 0, "Missing decoded sample clock")
    origin, count = frames[0][1], 0
    for i, (index, tick, length) in enumerate(frames):
        require(index == i and tick == origin + count and length > 0, "Decoded clock is discontinuous")
        count += length
    require(count * 2 == len(audio.stdout) and MIN_SAMPLES <= count <= MAX_SAMPLES, "Recording extent differs")
    return audio.stdout, origin


def verify_export(earlier, submitted, export):
    package = json.loads(export.read_bytes())
    record, assets = package["comparison"], package["assets"]
    timing = record["local_timing"]
    require(record["schema"] == SCHEMA and timing["status"] == "local_shift" and timing["sample_rate"] == RATE
            and timing["affects_overall_confidence"] is False, "The export needs a measured local audio shift")
    span = timing["range"]
    source = [decode(path, record[side]) for path, side in zip((earlier, submitted), SIDES, strict=True)]
    samples = [np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768 for raw, _ in source]
    require(source[0][1] == source[1][1] and len(samples[0]) == len(samples[1]), "The source clocks differ")
    found, match_count = localize(*samples)
    require(found is not None and match_count == timing["matched_windows"], "Independent timing measurement differs")
    bounds, offset = found
    prior = [sample - offset for sample in bounds]
    require(bounds == span["submitted_samples"] and prior == span["earlier_samples"] and offset == span["shift_samples"]
            and timing["offset_ms"] == offset * 1000 / RATE, "Independent signed offset or range differs")
    fields = {"submitted_samples", "earlier_samples", "shift_samples", "supporting_windows", "minimum_correlation",
              "window_start_samples", "same_time_support"}
    measured = {key: span[key] for key in fields}
    commitment = digest(json.dumps(measured, sort_keys=True, separators=(",", ":")).encode())
    require(commitment == span["support_sha256"], "Supporting window commitment differs")
    starts = span["window_start_samples"]
    require(len(starts) == span["supporting_windows"] and len(starts) >= 4
            and starts == sorted(set(starts)) and all(0 <= x <= len(samples[1]) - WINDOW and x % 640 == 0 for x in starts)
            and .86 <= span["minimum_correlation"] <= 1, "Invalid supporting windows")
    # Close lags can differ by a few samples. Verify each listed window has
    # corresponding audio near the recorded lag and the flanks match at zero.
    def similarity(first, second):
        first = first.astype(np.float64) - first.mean(dtype=np.float64)
        second = second.astype(np.float64) - second.mean(dtype=np.float64)
        return float(first @ second / max(math.sqrt((first @ first) * (second @ second)), 1e-12))

    def supports(start, lag, tolerance, minimum):
        query = samples[1][start:start + WINDOW]
        if np.std(query) < .003:
            return False
        base = start - lag
        return any(
            similarity(samples[0][i:i + WINDOW], query) >= minimum
            for i in range(max(0, base - tolerance), min(len(samples[0]) - WINDOW, base + tolerance) + 1)
        )

    require(all(supports(start, offset, 32, span["minimum_correlation"] - 1e-6) for start in starts),
            "Listed excerpts do not support the signed offset")
    flanks = span["same_time_support"]
    for key in ("before", "after"):
        ticks = flanks[key]
        require(len(ticks) == 2 and ticks == sorted(set(ticks))
                and all(0 <= x <= len(samples[1]) - WINDOW and x % 640 == 0 and supports(x, 0, 16, .86 - 1e-6) for x in ticks),
                "Same-time support does not match the original recordings")
    require(flanks["before"][-1] + WINDOW <= bounds[0] and flanks["after"][0] >= bounds[1],
            "Same-time windows do not bracket the excerpt")
    for side, (raw, origin), (start, end) in zip(SIDES, source, (prior, bounds), strict=True):
        row = span[side]
        require(0 <= start < end <= len(raw) // 2 and row["samples"] == [start, end]
                and row["source_pts"] == [origin + start, origin + end]
                and row["start_s"] == start / RATE and row["end_s"] == end / RATE, "Exported source clock differs")
        excerpt = raw[2 * start:2 * end]
        values = tuple(x for (x,) in iter_unpack("<h", excerpt))
        require(row["pcm_sha256"] == digest(excerpt) and row["waveform"] == waveform(values),
                "Source PCM or waveform differs")
        verify_clip(assets, row["audio"], values, f"local-timing-{side}.wav")
    return {"offset_ms": timing["offset_ms"], "earlier_samples": prior, "submitted_samples": bounds, "excerpts_verified": 2}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("submitted", type=Path)
    parser.add_argument("export", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_export(args.earlier, args.submitted, args.export), sort_keys=True))
    except (KeyError, TypeError, ValueError, OSError, wave.Error, subprocess.SubprocessError) as error:
        parser.exit(1, f"Local timing evidence verification failed. {error}\n")


if __name__ == "__main__":
    main()
