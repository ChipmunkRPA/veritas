"""Bind existing stereo backgrounds and qualify native 16 kHz PCM handling."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np

from app.detectors.audio.decode import SAMPLE_RATE, _mapping, decode_audio, source_ranges
from app.detectors.video.decode import _input
from scripts import audio_demand_controls_v204 as development
from scripts import audio_fresh_shared_noise_v217 as holdout

ROOT, PANEL = holdout.ROOT, holdout.PANEL
require, sha = holdout.require, holdout.sha
RATE = SAMPLE_RATE
SAMPLES = 3 * RATE
TAPE_SAMPLES = 4_800_064
CHANNELS = ("ch02.wav", "ch16.wav")
WINDOWS = tuple((200 + 6 * index) * RATE for index in range(16))
RECEIPTS = {
    development.PUBLIC: "200b773bb3aaff134465a989da93c495fd35a398bdfaf7d1afa2069072f63801",
    holdout.ASSET_RECORD: holdout.ASSET_RECORD_SHA,
    PANEL / "audio-short-edit-materialization-complete-v169c.json":
        "681389a3f959833de637a4e5f778e193b2e4bcecb62a437f3eede38c75896a1e",
    PANEL / "audio-short-edit-cases-result-v172.json":
        "9df9facc4dd63586ec8c24bafdce048b96ec2df731a397cf689d3692b6c25a61",
}
CODE = (
    "backend/app/detectors/audio/decode.py",
    "backend/app/detectors/video/decode.py",
    "backend/scripts/audio_demand_controls_v204.py",
    "backend/scripts/audio_fresh_shared_noise_v217.py",
    "backend/scripts/audio_native_stereo_preflight_v274.py",
    "backend/tests/test_audio_native_stereo_preflight_v274.py",
    "docs/audio-native-stereo-preflight-v274-method.md",
)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def window_assignment(family_ordinal):
    """Cycle the same fixed windows within each role and origin family."""
    require(type(family_ordinal) is int and family_ordinal >= 0, "A nonnegative family ordinal is required")
    return WINDOWS[family_ordinal % len(WINDOWS)], family_ordinal % 2


def plan():
    require(
        WINDOWS[0] >= 0 and WINDOWS[-1] + SAMPLES <= TAPE_SAMPLES
        and all(left + SAMPLES <= right for left, right in zip(WINDOWS, WINDOWS[1:])),
        "Every frozen window must be bounded and nonoverlapping within its tape",
    )
    return {
        "roles": {"development": dict(development.TAPES), "holdout": dict(holdout.TAPE_FILES)},
        "members": list(CHANNELS),
        "window_starts_samples": list(WINDOWS),
        "window_samples": SAMPLES,
        "sample_rate_hz": RATE,
        "assignment": "Zero-based v169c manifest order within each origin family and role; ordinal modulo 16 selects the window and ordinal modulo 2 selects the quieter channel for the future forced-imbalance condition.",
        "foreground_slots_for_future_composition": {"development": 37, "holdout": 40},
    }


def tape_member(archive, channel):
    matches = [item for item in archive.infolist()
               if not item.is_dir() and PurePosixPath(item.filename).name == channel]
    require(len(matches) == 1 and matches[0].file_size <= 12_000_000, "One bounded publisher channel member required")
    with archive.open(matches[0]) as stream, wave.open(stream, "rb") as wav:
        require(
            (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype())
            == (1, 2, RATE, TAPE_SAMPLES, "NONE"),
            "Exact native 16 kHz publisher channel header required",
        )
        pcm = wav.readframes(TAPE_SAMPLES)
        require(len(pcm) == 2 * TAPE_SAMPLES and not wav.readframes(1), "Incomplete publisher channel PCM")
        stream.read()  # Finish the bounded ZIP member so CRC verification includes any trailing chunks.
    return pcm


def tape_readout(path):
    with zipfile.ZipFile(path) as archive:
        channels = {channel: tape_member(archive, channel) for channel in CHANNELS}
    require(len({digest(pcm) for pcm in channels.values()}) == len(CHANNELS), "Distinct recorded channels required")
    return {
        "samples_per_channel": TAPE_SAMPLES,
        "members": {channel: {"pcm_sha256": digest(pcm), "windows_sha256": digest(b"".join(
            pcm[2 * start:2 * (start + SAMPLES)] for start in WINDOWS
        ))} for channel, pcm in channels.items()},
    }


def archive_checks():
    require(all(sha(path) == expected for path, expected in RECEIPTS.items()), "Exact prior source and archive receipts required")
    dev = holdout.read(development.PUBLIC)
    require(
        dev["environment_license"] == "CC-BY-SA-3.0" and dev["publisher_record"] == development.RECORD
        and development.asset_identity() == dev["publisher_assets"],
        "Development DEMAND assets differ from their pinned publisher receipt",
    )
    held = holdout.publisher_assets()
    require(held["publisher_record"] == development.RECORD, "Exact DEMAND publisher record required")
    return {
        role: {environment: {"archive": filename, **tape_readout(folder / filename)}
               for environment, filename in files.items()}
        for role, folder, files in (
            ("development", development.SOURCES, development.TAPES),
            ("holdout", holdout.TAPE_ROOT, holdout.TAPE_FILES),
        )
    }


def fixtures():
    left = np.zeros(SAMPLES, dtype="<i2")
    right = np.zeros(SAMPLES, dtype="<i2")
    left[[0, 511, 24001, SAMPLES - 2]] = [800, -2300, 4400, 1700]
    right[[1, 513, 24001, SAMPLES - 1]] = [-900, 1800, -1300, -4200]
    return {
        "signed_impulses": (left, right),
        "swapped_channels": (right, left),
        "silent_right": (left, np.zeros_like(left)),
        "silent_left": (np.zeros_like(left), left),
        "opposite_phase": (left, -left),
    }


def ffmpeg_pcm(path, transform):
    filters = f"{transform},atrim=end_sample={SAMPLES},ashowinfo,asetpts=N/SR/TB"
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-protocol_whitelist", "file", "-copyts",
         "-i", _input(path), "-map", "0:a:0", "-vn", "-af", filters, "-t", "3",
         "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"],
        stdin=subprocess.DEVNULL, capture_output=True, timeout=15, check=True,
    )
    require(len(result.stdout) == 2 * SAMPLES, "Exact bounded FFmpeg PCM length required")
    return result.stdout, _mapping(result.stderr.decode(), SAMPLES)


def fixture_readout(directory, name, channels):
    source = directory / f"{name}.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((2, 2, RATE, SAMPLES, "NONE", "not compressed"))
        wav.writeframes(np.column_stack(channels).astype("<i2").tobytes())

    production = decode_audio(source, source_sha256=sha(source), max_seconds=SAMPLES / RATE)
    mixed, timeline = ffmpeg_pcm(source, f"aresample=osr={RATE}:ochl=mono:osf=s16")
    mapping = production.mapping
    require(
        development.read_wave(production.wav) == mixed
        and mapping["source_channels"] == 2 and mapping["source_sample_rate"] == RATE
        and mapping["samples"] == SAMPLES and mapping["timeline"] == timeline,
        "Production downmix or sample clock differs from the explicit FFmpeg resampler",
    )
    require(source_ranges(mapping, 0, SAMPLES) == [{"start_sample": 0, "end_sample": SAMPLES,
             "source_start_pts": 0, "source_end_pts": SAMPLES, "time_base": f"1/{RATE}"}],
            "Authored source clock must be continuous and start at zero")

    demuxed = []
    for index, expected in enumerate(channels):
        transform = f"pan=mono|c0=c{index},aformat=sample_fmts=s16:channel_layouts=mono,aresample={RATE}"
        pcm, channel_timeline = ffmpeg_pcm(source, transform)
        require(pcm == expected.tobytes() and channel_timeline == timeline,
                "Explicit channel extraction must preserve exact signed samples and source clock")
        demuxed.append(digest(pcm))
    return mixed, {"mixed_pcm_sha256": digest(mixed), "channel_pcm_sha256": demuxed, "source_timeline": timeline}


def authored_checks(directory):
    mixed, rows = {}, {}
    inputs = fixtures()
    for name, channels in inputs.items():
        mixed[name], rows[name] = fixture_readout(directory, name, channels)
    support = np.flatnonzero(np.frombuffer(mixed["signed_impulses"], dtype="<i2"))
    expected_support = np.flatnonzero(np.any(np.column_stack(inputs["signed_impulses"]), axis=1))
    require(
        mixed["signed_impulses"] == mixed["swapped_channels"]
        and np.array_equal(support, expected_support)
        and mixed["silent_left"] == mixed["silent_right"]
        and mixed["silent_left"] != b"\0" * (2 * SAMPLES)
        and mixed["opposite_phase"] == b"\0" * (2 * SAMPLES),
        "Authored downmix failed impulse support, swap, silent-side, or cancellation parity",
    )
    return rows


def tool_hash(name):
    executable = shutil.which(name)
    require(executable is not None, f"{name} is required for exact stereo PCM verification")
    return sha(Path(executable).resolve(strict=True))


def main():
    holdout.model_free()
    frozen_plan = plan()
    codecs = {name: tool_hash(name) for name in ("ffmpeg", "ffprobe")}
    tapes = archive_checks()
    with tempfile.TemporaryDirectory(prefix="chipmunk-stereo-v274-authored-") as temporary:
        authored = authored_checks(Path(temporary))
    holdout.model_free()
    print(json.dumps({
        "schema": "chipmunk.audio-native-stereo-preflight/274",
        "status": "complete",
        "archive_license": "CC-BY-SA-3.0",
        "publisher_record": development.RECORD,
        "plan": frozen_plan,
        "prior_receipts_sha256": {path.name: expected for path, expected in RECEIPTS.items()},
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "codec_binaries_sha256": codecs,
        "tapes": tapes,
        "authored_native_16khz": authored,
        "foreground_files_opened": 0,
        "model_calls": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
