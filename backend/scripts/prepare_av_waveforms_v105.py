"""Native audio/sample qualification and source-timed 16-kHz conditioning caches.

No model, generator or app pipeline is executed. Raw voices remain private.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import sys
import wave
from fractions import Fraction
from pathlib import Path

from scripts.review_video_native_inserts_v96 import Budget

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PARENT = BASE / "av-sentence-support-v104"
NATIVE = BASE / "video-native-sources-v92"
ACQUIRED = BASE / "video-common-acquisition-v91"
TRIAL = BASE / "av-waveforms-v105"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
CACHE = ARCHIVE / "av-waveforms-v105"
PUBLIC = ROOT / "docs/evaluation-panels/av-waveform-freeze-v105.json"
CODE = (
    "backend/scripts/prepare_av_waveforms_v105.py",
    "backend/tests/test_av_waveforms_v105.py",
    "docs/av-waveforms-v105-method.md",
    "backend/scripts/review_video_native_inserts_v96.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sample_count(data, channels):
    require(
        type(channels) is int and channels in (1, 2) and len(data) % (2 * channels) == 0,
        "Complete one/two-channel signed-16 PCM frames required",
    )
    require(bool(data), "Decoded PCM cannot be empty")
    return len(data) // (2 * channels)


def values(data):
    require(len(data) % 2 == 0, "Complete signed-16 samples required")
    result = array.array("h")
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def pcm(values_):
    data = array.array("h", values_)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def audio_command(path, *, rate=None, channels=None, hash_output=False):
    command = [
        shutil.which("ffmpeg"),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-protocol_whitelist",
        "file",
        "-i",
        "file:" + str(path.resolve()),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-c:a",
        "pcm_s16le",
    ]
    if channels is not None:
        command += ["-ac", str(channels)]
    if rate is not None:
        command += ["-ar", str(rate)]
    command += ["-f", "hash", "-hash", "sha256", "pipe:1"] if hash_output else ["-f", "s16le", "pipe:1"]
    return command


def parse_hash(data):
    match = re.fullmatch(rb"SHA256=([0-9a-f]{64})\s*", data)
    require(match is not None, "One complete SHA256 audio hash required")
    return match[1].decode()


def save_wav(path, data, rate, channels):
    count = sample_count(data, channels)
    require(type(rate) is int and rate > 0 and not path.exists(), "New native-rate WAV required")
    with path.open("xb") as stream:
        with wave.open(stream, "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(2)
            target.setframerate(rate)
            target.writeframes(data)
    with wave.open(str(path), "rb") as check:
        require(
            (check.getnchannels(), check.getsampwidth(), check.getframerate(), check.getnframes())
            == (channels, 2, rate, count)
            and check.readframes(count) == data,
            "WAV header or PCM payload changed",
        )
    return {
        "file": path.name,
        "sha256": sha(path),
        "pcm_sha256": digest(data),
        "sample_rate": rate,
        "channels": channels,
        "sample_frames": count,
    }


def coordinates(clock, samples):
    require(type(samples) is int and samples > 0, "Observed conditioning sample count required")
    origin = Fraction(clock["source_audio_origin_seconds"])
    rows = []
    for frame in clock["audio_frame_mappings"]:
        source = Fraction(frame["source_time_seconds"])
        point = (source - origin) * 16000
        lo, hi = math.floor(point), math.ceil(point)
        rows.append(
            {
                "video_frame_index": frame["video_frame_index"],
                "source_time_seconds": str(source),
                "sample_position_fraction": str(point),
                "bracketing_samples": [lo, hi],
                "status": "supported" if 0 <= lo <= hi < samples else "unavailable",
                "native_clock_allowance_seconds": clock["native_audio_clock_quantization_allowance_seconds"],
            }
        )
    return rows


def phase_fixture(channels):
    require(channels in (1, 2), "Mono/stereo authored phase fixture required")
    count = 88200
    impulses = [0, 13230, 34287, 66150, count - 1]
    output = [0] * (count * channels)
    for i, position in enumerate(impulses):
        output[position * channels + (i % channels)] = 12000
    return pcm(output), impulses


def check_phase(data, impulses):
    output = values(data)
    require(len(output) == 32000, "Complete two-second authored resampling required")
    rows = []
    for point in impulses:
        expected = Fraction(point * 16000, 44100)
        candidates = range(max(0, math.floor(expected) - 4), min(len(output), math.ceil(expected) + 5))
        peak = max(candidates, key=lambda i: abs(output[i]))
        require(
            abs(output[peak]) > 0 and abs(Fraction(peak) - expected) <= 1,
            "Authored impulse phase differs by more than one output sample",
        )
        rows.append(
            {
                "source_sample": point,
                "expected_output_sample_fraction": str(expected),
                "observed_peak_sample": peak,
            }
        )
    return rows


def identity():
    return {
        "schema": "chipmunk.av-waveform-freeze/105",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                PARENT / "preparation/plan.json",
                PARENT / "preparation/result.json",
                NATIVE / "qualified-plan.json",
                ACQUIRED / "asset-plan.json",
            )
        },
        "ffmpeg_sha256": sha(Path(shutil.which("ffmpeg"))),
        "psutil_version": importlib.metadata.version("psutil"),
        "phase_tolerance_output_samples": 1,
        "sources": 48,
        "new_neural_calls": 0,
        "scope": "Native PCM sample counts, exact repeated source hash, unchanged mono16k content binding, authored resampling phase and source-time coordinates; not achieved phonetic synchronization, generator qualification or detector accuracy.",
    }


def freeze():
    TRIAL.mkdir(exist_ok=False)
    body = identity()
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def prepare():
    lock = read(TRIAL / "freeze.json")
    require(lock == identity() == read(PUBLIC), "Published unchanged waveform preparation required")
    output = TRIAL / "preparation"
    output.mkdir(exist_ok=False)
    CACHE.mkdir(exist_ok=False)
    budget = Budget(output)
    status, error_type = "failed", None
    try:
        budget.observe(initial=True)
        phase = []
        for channels in (1, 2):
            source, impulses = phase_fixture(channels)
            path = output / f"authored-phase-{channels}ch.wav"
            save_wav(path, source, 44100, channels)
            resampled = budget.command(audio_command(path, rate=16000, channels=1))
            phase.append({"channels": channels, "checks": check_phase(resampled, impulses)})
        plan = read(PARENT / "preparation/plan.json")
        require(
            sha(PARENT / "preparation/plan.json")
            == read(PARENT / "preparation/result.json")["private_plan_sha256"],
            "Locked sentence/timing plan required",
        )
        native = {r["index"]: r for r in read(NATIVE / "qualified-plan.json")["assets"]}
        acquired = read(ACQUIRED / "asset-plan.json")
        assets = {r["index"]: r for r in acquired["assets"]}
        sources = Path(acquired["media_root"])
        require(
            sources.parent == ARCHIVE and len(plan["records"]) == 48,
            "Approved complete source inventory required",
        )
        records = []
        with (output / "waveforms.jsonl").open("x") as journal:
            for row in plan["records"]:
                item = native[row["asset_index"]]
                asset = assets[row["asset_index"]]
                path = sources / asset["file"]
                require(
                    path.parent == sources and sha(path) == asset["sha256"] == row["input_sha256"],
                    "Exact acquired source bytes required",
                )
                audio = item["audio_clocks"]
                data = budget.command(audio_command(path))
                count = sample_count(data, audio["channels"])
                require(
                    count
                    == sum(f["samples"] for f in audio["frames"])
                    == row["clock_support"]["native_audio_samples_declared"],
                    "Native decoded samples differ from recorded frame counts",
                )
                repeated = parse_hash(budget.command(audio_command(path, hash_output=True)))
                require(repeated == digest(data), "Second source-decode PCM digest differs")
                native_wav = save_wav(
                    CACHE / f"asset-{row['asset_index']:03d}-native.wav",
                    data,
                    audio["sample_rate"],
                    audio["channels"],
                )
                conditioned = budget.command(audio_command(path, rate=16000, channels=1))
                n = sample_count(conditioned, 1)
                require(
                    n == item["canonical_audio"]["samples"]
                    and digest(conditioned) == item["canonical_audio"]["pcm_sha256"],
                    "Conditioning input differs from earlier exact mono16k source content",
                )
                wav = save_wav(CACHE / f"asset-{row['asset_index']:03d}-mono16k.wav", conditioned, 16000, 1)
                frame_coordinates = coordinates(row["clock_support"], n)
                nonzero = any(values(conditioned))
                record = {
                    "asset_index": row["asset_index"],
                    "source_group": row["id"],
                    "study_role": row["study_role"],
                    "source_sha256": row["input_sha256"],
                    "native_wav": native_wav,
                    "mono16k_wav": wav,
                    "audio_origin_seconds": row["clock_support"]["source_audio_origin_seconds"],
                    "video_origin_seconds": row["clock_support"]["source_video_origin_seconds"],
                    "audio_minus_video_origin_seconds": row["clock_support"][
                        "audio_minus_video_origin_seconds"
                    ],
                    "frame_sample_coordinates": frame_coordinates,
                    "native20_edit_eligible": row["clock_support"]["native20_edit_eligible"],
                    "conditioning_waveform_ready": nonzero
                    and all(p["status"] == "supported" for p in frame_coordinates),
                    "generator_conditioning_qualified": False,
                    "nonzero_mono_signal": nonzero,
                }
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
                records.append(record)
                budget.observe()
                print(
                    json.dumps(
                        {"completed_sources": len(records), "planned_sources": 48, "new_neural_calls": 0}
                    ),
                    flush=True,
                )
        require(
            len(records) == 48 and not {"torch", "numpy", "transformers", "scipy"}.intersection(sys.modules),
            "Complete non-neural waveform preparation required",
        )
        result = {
            "schema": "chipmunk.av-waveform-result/105",
            "status": "complete",
            "sources": len(records),
            "native_sample_frames": sum(r["native_wav"]["sample_frames"] for r in records),
            "mono16k_samples": sum(r["mono16k_wav"]["sample_frames"] for r in records),
            "stereo_sources_preserved": sum(r["native_wav"]["channels"] == 2 for r in records),
            "conditioning_waveforms_ready": sum(r["conditioning_waveform_ready"] for r in records),
            "supported_picture_sample_coordinates": sum(
                p["status"] == "supported" for r in records for p in r["frame_sample_coordinates"]
            ),
            "unavailable_picture_sample_coordinates": sum(
                p["status"] != "supported" for r in records for p in r["frame_sample_coordinates"]
            ),
            "native_source_repeat_hashes_matched": len(records),
            "earlier_mono16k_hashes_matched": len(records),
            "authored_phase_checks": sum(len(p["checks"]) for p in phase),
            "maximum_phase_error_output_samples": float(
                max(
                    abs(Fraction(c["observed_peak_sample"]) - Fraction(c["expected_output_sample_fraction"]))
                    for p in phase
                    for c in p["checks"]
                )
            ),
            "freeze_sha256": sha(TRIAL / "freeze.json"),
            "private_journal_sha256": sha(output / "waveforms.jsonl"),
            "private_cache_files": len(records) * 2,
            "new_neural_calls": 0,
            "defaults_changed": False,
            "scope": lock["scope"],
        }
        write(output / "phase.json", phase)
        write(output / "result.json", result)
        print(json.dumps(result), flush=True)
        status = "complete"
    except BaseException as error:
        error_type = type(error).__name__
        raise
    finally:
        import time

        budget.stream.close()
        write(
            output / "profile.json",
            {
                "status": status,
                "error_type": error_type,
                "duration_s": time.monotonic() - budget.started,
                "maximum_observed_combined_rss_bytes": max(
                    (r["combined_rss_bytes"] for r in budget.rows), default=0
                ),
                "minimum_available_ram_bytes": min(
                    (r["available_ram_bytes"] for r in budget.rows), default=None
                ),
                "sample_errors": len(budget.errors),
                "terminal_observations": budget.exits,
                "resource_journal_sha256": sha(output / "resources.jsonl"),
                "new_neural_calls": 0,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare"))
    freeze() if parser.parse_args().command == "freeze" else prepare()
