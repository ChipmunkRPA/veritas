"""Freeze and replay native two-channel fixed-gain repetition transfer."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import subprocess
import tempfile
import time
import wave
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from app.evidence import audio_exact_repetition as exact
from app.evidence import audio_gain_repetition as gain
from app.evidence import audio_gain_repetition_binding as binding
from app.evidence.stereo_repetition import native_pcm

ROOT = Path(__file__).resolve().parents[2]
DEV_ROSTER = ROOT / "backend/data/independent-evaluation/audio-measured-spatial-transfer-final-2026-11-12"
AMI = ROOT / "backend/data/independent-evaluation/audio-ami-native-stereo-v291/acquired.json"
SESSIONS = ("ES2004a", "IS1009a")
MICROPHONES = ("Array1-01", "Array1-05")
RATE = 16000
DURATION = 3 * RATE
GRID = tuple(75 + 30 * index for index in range(12))
BOUNDARY_TOLERANCE = 8
DECODE_POSITIONS = {"development": {0, 15}, "confirmation": {0, 11}}
CORE = ((5401, 8450), (28617, 31666))
RATIOS = {
    "attenuated": (3, 4, "second"),
    "inverted": (-1, 1, "second"),
    "amplified": (3, 2, "second"),
    "earlier_attenuated": (1, 2, "first"),
}
CONTROLS = ("unchanged", "global_gain", "unsupported_gain", "competing_microphone", "clipped_repeat", "short_repeat")
PINNED = (Path(__file__), Path(gain.__file__), Path(binding.__file__), Path(exact.__file__), AMI, DEV_ROSTER / "roster.json")
MONO_FILTER = f"aresample=osr={RATE}:ochl=mono:osf=s16"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def plan(dev, ffmpeg):
    receipt = read(dev / "receipt.json")
    expected = {f"{index:02d}-unmodified.wav" for index in range(16)}
    if len(receipt["cases"]) != 48 or not all((dev / name).is_file() for name in expected):
        raise ValueError("Expected all 16 retained development WAVs and their receipt")
    return {
        "schema": "chipmunk.audio.gain-repetition-transfer-freeze/1",
        "code_and_intake": {str(path.resolve().relative_to(ROOT)): digest(path) for path in PINNED},
        "numpy": np.__version__,
        "ffmpeg_sha256": digest(Path(ffmpeg)),
        "mono_filter": MONO_FILTER,
        "production_decode_checks": {role: sorted(indices) for role, indices in DECODE_POSITIONS.items()},
        "development_source_files": {name: digest(dev / name) for name in sorted(expected | {"receipt.json"})},
        "confirmation_sources": read(AMI)["files"],
        "confirmation_source_role": "source-time-separated from 60+30i intake, same two known AMI sessions",
        "confirmation_time_starts_s": list(GRID),
        "window_samples": DURATION,
        "source_pair_core_samples": list(map(list, CORE)),
        "source_pair_geometry": "add index times 13 to first start and 7 to second start, both ends shift with start",
        "gain_variants": {key: list(value) for key, value in RATIOS.items()},
        "other_cases": ["exact_copy", *CONTROLS],
        "authored_stress": ["silence", "periodic_tone", "ordinary_scaled_alert"],
        "boundary_tolerance_samples": BOUNDARY_TOLERANCE,
        "reciprocal_orientation": "accept an exact reciprocal in the opposite temporal direction",
        "gate": {"gain_variant_recall_minimum": 0.80, "no_relation_control_extra_flags_maximum": 0,
                 "wrong_channel_or_gain_or_ranges_maximum": 0, "correct_in_both_sessions": True},
    }


def decode_wav(reader, start, count):
    reader.setpos(start)
    body = reader.readframes(count)
    if len(body) != 2 * count * reader.getnchannels():
        raise ValueError("Incomplete source samples")
    return np.frombuffer(body, dtype="<i2").reshape(count, reader.getnchannels())


def check_profile(reader, channels):
    if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) != (channels, 2, RATE, "NONE"):
        raise ValueError("Expected native 16 kHz signed-16-bit PCM")


def source_groups(dev, ami, freeze):
    dev_rows = [row for row in read(dev / "receipt.json")["cases"] if row["variant"] == "unmodified"]
    old_sources = [row for row in read(DEV_ROSTER / "roster.json") if row["arm"] in {"untouched", "background_only"}]
    if len(dev_rows) != 16 or len(old_sources) != 16:
        raise ValueError("Development roster changed")
    for index, (row, previous) in enumerate(zip(dev_rows, old_sources, strict=True)):
        if row["case"] != index:
            raise ValueError("Development roster changed")
        with wave.open(str(dev / f"{index:02d}-unmodified.wav"), "rb") as reader:
            check_profile(reader, 2)
            original = decode_wav(reader, 0, reader.getnframes())
        raw = (DEV_ROSTER / "pcm" / f"{previous['id']}-stereo.pcm").read_bytes()
        if sha(raw) != row["original_pcm_sha256"] or original.tobytes() != raw[4 * 1301:4 * 50802]:
            raise ValueError("Development PCM differs from retained receipt")
        yield "development", row["role"], index, 0, original[:DURATION]

    for session in SESSIONS:
        with ExitStack() as stack:
            readers = []
            for microphone in MICROPHONES:
                name = f"{session}.{microphone}.wav"
                source = ami / "raw" / name
                proof = freeze["confirmation_sources"][name]
                if source.stat().st_size != proof["bytes"] or digest(source) != proof["sha256"]:
                    raise ValueError("AMI source does not match its first-party receipt")
                reader = stack.enter_context(wave.open(str(source), "rb"))
                check_profile(reader, 1)
                readers.append(reader)
            if readers[0].getnframes() != readers[1].getnframes():
                raise ValueError("AMI microphone timeline differs")
            for index, seconds in enumerate(GRID):
                start = seconds * RATE
                yield "confirmation", session, index, start, np.column_stack(
                    [decode_wav(reader, start, DURATION)[:, 0] for reader in readers]
                )


def sample_ranges(index):
    return tuple((start + index * step, end + index * step) for (start, end), step in zip(CORE, (13, 7), strict=True))


def scaled(values, numerator, denominator):
    return np.rint(values.astype(np.int32) * numerator / denominator).astype(np.int32)


def into_i16(values):
    return np.clip(values, -32768, 32767).astype("<i2")


def deliveries(original, index):
    ranges = sample_ranges(index)
    for ordinal, variant in enumerate((*RATIOS, "exact_copy", *CONTROLS)):
        channel = (index + ordinal) % 2
        copy = original.copy()
        first, second = (slice(*bounds) for bounds in ranges)
        truth = None
        if variant in RATIOS:
            numerator, denominator, occurrence = RATIOS[variant]
            source, target = (first, second) if occurrence == "second" else (second, first)
            values = scaled(copy[source, channel], numerator, denominator)
            if np.any((values < -32768) | (values > 32767)):
                # Keep clipping as a separate control. Condition only the donor here.
                copy[source, channel] = into_i16(scaled(copy[source, channel], 1, 2))
                values = scaled(copy[source, channel], numerator, denominator)
            copy[target, channel] = into_i16(values)
            truth = (channel, ranges, numerator, denominator, occurrence)
        elif variant == "exact_copy":
            copy[second, channel] = copy[first, channel]
            truth = (channel, ranges, 1, 1, "second")
        elif variant == "global_gain":
            copy = into_i16(scaled(copy, 3, 4))
        elif variant == "unsupported_gain":
            copy[second, channel] = into_i16(scaled(copy[first, channel], 7, 8))
        elif variant == "competing_microphone":
            copy[second, channel] = into_i16(scaled(copy[first, channel], 3, 4) + scaled(copy[second, 1 - channel], 1, 3))
        elif variant == "clipped_repeat":
            source = copy[first, channel]
            source[::193] = np.where(np.arange(len(source[::193])) % 2, -25000, 25000)
            copy[second, channel] = into_i16(scaled(source, 3, 2))
        elif variant == "short_repeat":
            source, target = (slice(start, start + 1199) for start, _ in ranges)
            copy[target, channel] = into_i16(scaled(copy[source, channel], 3, 4))
        yield variant, copy, truth


def authored_stress():
    silent = np.zeros((DURATION, 2), dtype="<i2")
    yield "silence", silent, None
    tone = np.tile(into_i16(1400 * np.sin(np.arange(64) * (2 * np.pi / 64))), DURATION // 64)
    yield "periodic_tone", np.column_stack((tone, tone)), None
    first, second = (slice(*bounds) for bounds in CORE)
    count = CORE[0][1] - CORE[0][0]
    phase = np.arange(count) / RATE
    alert = into_i16(1600 * np.sin(2 * np.pi * (480 * phase + 300 * phase**2)) * np.sin(np.pi * np.arange(count) / count))
    ordinary = silent.copy()
    ordinary[first, 0] = alert
    ordinary[second, 0] = into_i16(scaled(alert, 3, 4))
    yield "ordinary_scaled_alert", ordinary, (0, CORE, 3, 4, "second")


def pcm_wav(stereo):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams((2, 2, RATE, len(stereo), "NONE", "not compressed"))
        writer.writeframes(stereo.tobytes())
    return buffer.getvalue()


def pcm_mono(stereo, ffmpeg):
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
         "-i", "pipe:0", "-af", MONO_FILTER, "-f", "s16le", "pipe:1"],
        input=pcm_wav(stereo), capture_output=True, check=True, timeout=10,
    ).stdout
    if len(result) != 2 * len(stereo):
        raise ValueError("Production FFmpeg mono sample clock changed")
    return result


def error(span, ranges):
    return max(abs(a - b) for measured, intended in zip((span.first, span.second), ranges, strict=True)
               for a, b in zip(measured, intended, strict=True))


def correct_gain(span, numerator, denominator, occurrence):
    if span.scaled_occurrence == occurrence:
        return span.numerator * denominator == numerator * span.denominator
    return span.numerator * numerator == denominator * span.denominator


def gain_readout(samples, spans):
    for span in spans:
        if not binding.matches_clips(
            {"sample_count": span.sample_count, "scaled_occurrence": span.scaled_occurrence,
             "gain_numerator": span.numerator, "gain_denominator": span.denominator},
            *(samples[start:end].tobytes() for start, end in (span.first, span.second)),
        ):
            raise ValueError("Returned relationship does not independently verify on source samples")


def score(stereo, truth, ffmpeg, *, production_check=False):
    mono = pcm_mono(stereo, ffmpeg)
    if production_check:
        wav = pcm_wav(stereo)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "comparison.wav"
            path.write_bytes(wav)
            decoded = decode_audio(path, source_sha256=sha(wav), inspect_stereo=True, retain_stereo=True)
        if native_pcm(decoded) != stereo.tobytes() or (decoded.pcm * 32768).astype("<i2").tobytes() != mono:
            raise ValueError("Native samples or mono differ from production decode")
    samples = [stereo[:, channel].copy() for channel in range(2)]
    channels = [channel.tobytes() for channel in samples]
    start = time.perf_counter()
    gained = [gain.find_gain_repeats(channel) for channel in channels]
    gain_time = time.perf_counter() - start
    start = time.perf_counter()
    exact_channels = [exact.find_repeated_spans(channel) for channel in channels]
    exact_mono = exact.find_repeated_spans(mono)
    exact_time = time.perf_counter() - start
    for source, result in zip(samples, gained, strict=True):
        gain_readout(source, result.spans)
    row = {
        "native_pcm_sha256": sha(stereo.tobytes()),
        "mono_pcm_sha256": sha(mono),
        "production_decode_checked": production_check,
        "gain": [[asdict(item) for item in result.spans] for result in gained],
        "gain_status": [[result.status, result.reason] for result in gained],
        "exact_channel": [[asdict(item) for item in result.spans] for result in exact_channels],
        "exact_channel_status": [[result.status, result.reason] for result in exact_channels],
        "exact_mono": [asdict(item) for item in exact_mono.spans],
        "exact_mono_status": [exact_mono.status, exact_mono.reason],
    }
    if truth is None:
        row.update(gain_useful=False, exact_channel_useful=False, exact_mono_useful=False)
    else:
        channel, ranges, numerator, denominator, occurrence = truth
        errors = [error(span, ranges) for span in gained[channel].spans
                  if correct_gain(span, numerator, denominator, occurrence)]
        best = min(errors, default=None)
        row.update(
            truth=truth, gain_bound_error=best, gain_useful=best is not None and best <= BOUNDARY_TOLERANCE,
            exact_channel_useful=any(error(span, ranges) <= BOUNDARY_TOLERANCE for span in exact_channels[channel].spans),
            exact_mono_useful=any(error(span, ranges) <= BOUNDARY_TOLERANCE for span in exact_mono.spans),
        )
    row["other_gain_spans"] = sum(len(result.spans) for result in gained) - int(row["gain_useful"])
    return row, (gain_time, exact_time)


def totals(rows):
    collected = Counter()
    errors = []
    for row in rows:
        collected["deliveries"] += 1
        collected["production_decode_checks"] += int(row["production_decode_checked"])
        for metric in ("gain_useful", "exact_channel_useful", "exact_mono_useful", "other_gain_spans"):
            collected[metric] += int(row[metric])
        for metric in ("gain", "exact_channel", "exact_mono"):
            collected[f"{metric}_flagged_files"] += int(any(row[metric]))
        for status, reason in row["gain_status"]:
            collected[f"gain_channel_{status}"] += 1
            if reason:
                collected[f"gain_reason_{reason}"] += 1
        if row["gain_useful"]:
            errors.append(row["gain_bound_error"])
    collected["maximum_useful_boundary_error_samples"] = max(errors, default=0)
    return dict(collected)


def summary(rows):
    report = {}
    for role in ("development", "confirmation", "stress"):
        group = [row for row in rows if row["role"] == role]
        report[role] = {
            "by_variant": {variant: totals(row for row in group if row["variant"] == variant)
                           for variant in sorted({row["variant"] for row in group})},
            "by_source_group": {session: totals(row for row in group if row["group"] == session)
                                for session in sorted({row["group"] for row in group})},
        }
    confirmation = [row for row in rows if row["role"] == "confirmation"]
    positives = [row for row in confirmation if row["variant"] in RATIOS]
    controls = [row for row in confirmation if row["variant"] in CONTROLS]
    passing = (
        sum(row["gain_useful"] for row in positives) >= 0.8 * len(positives)
        and all(not row["other_gain_spans"] for row in confirmation)
        and all(any(row["gain_useful"] and row["group"] == session for row in positives) for session in SESSIONS)
    )
    return {"counts": report, "production_decode_checks": sum(row["production_decode_checked"] for row in rows),
            "confirmation_gain_positives": totals(positives),
            "confirmation_no_relation_controls": totals(controls), "passed": passing}


def run(dev, ami, ffmpeg, freeze, *, phase):
    rows, gain_times, exact_times = [], [], []
    for role, group, index, start, original in source_groups(dev, ami, freeze):
        for variant, stereo, truth in deliveries(original, index):
            row, (gain_seconds, exact_seconds) = score(stereo, truth, ffmpeg, production_check=index in DECODE_POSITIONS[role])
            rows.append({"role": role, "group": group, "window": index, "source_start_sample": start,
                         "variant": variant, "source_pcm_sha256": sha(original.tobytes()), **row})
            gain_times.append(gain_seconds)
            exact_times.append(exact_seconds)
        if (index + 1) % 6 == 0:
            print(phase, role, group, index + 1, flush=True)
    for variant, stereo, truth in authored_stress():
        row, (gain_seconds, exact_seconds) = score(stereo, truth, ffmpeg)
        rows.append({"role": "stress", "group": "authored_nonvoice", "variant": variant, **row})
        gain_times.append(gain_seconds)
        exact_times.append(exact_seconds)
    timings = {name: {"median_ms": round(1000 * statistics.median(values), 3), "max_ms": round(1000 * max(values), 3)}
               for name, values in (("gain_two_channels", gain_times), ("exact_two_channels_and_mono", exact_times))}
    return json.loads(json.dumps(rows)), timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "review"))
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--ami", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg")
    args = parser.parse_args()
    frozen = plan(args.development, args.ffmpeg)
    freeze_path = args.output / "freeze.json"
    results_path = args.output / "readout.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze_path, frozen)
        print("Frozen before candidate evaluation:", digest(freeze_path))
        return
    if read(freeze_path) != frozen:
        raise ValueError("Prospective source or code freeze changed")
    if args.phase == "run" and results_path.exists():
        raise ValueError("Preserve the original readout")
    start = time.perf_counter()
    rows, timings = run(args.development, args.ami, args.ffmpeg, frozen, phase=args.phase)
    report = summary(rows)
    elapsed = round(time.perf_counter() - start, 3)
    if args.phase == "run":
        write(results_path, {"rows": rows, "summary": report, "timings": timings, "elapsed_seconds": elapsed})
    else:
        earlier = read(results_path)
        if earlier["rows"] != rows or earlier["summary"] != report:
            raise ValueError("Independent native interval replay changed the readout")
        write(args.receipt, {
            "schema": "chipmunk.audio.gain-repetition-transfer/1", "freeze": frozen,
            "freeze_sha256": digest(freeze_path), "private_readout_sha256": digest(results_path),
            "exact_native_replays": len(rows), "first_run_seconds": earlier["elapsed_seconds"],
            "replay_seconds": elapsed, "first_run_timing": earlier["timings"], "replay_timing": timings, **report,
        })
    print(json.dumps({"elapsed_seconds": elapsed, "timings": timings,
                      "positives": report["confirmation_gain_positives"],
                      "controls": report["confirmation_no_relation_controls"], "passed": report["passed"]}), flush=True)


if __name__ == "__main__":
    main()
