"""Test two repetition cues through fixed lossy codecs on consumed GRID PCM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import types
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import grid_audio_splice_probe_v167 as prior
from scripts import video_appearance_runtime_v108 as runtime
from scripts.audio_exact_repetition import MIN_REPEAT_SAMPLES, find_repeated_spans
from scripts.audio_repetition_similarity import find_similar_spans
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
TRIAL = BASE / "grid-audio-repetition-codecs-v170"
PUBLIC = PANEL / "grid-audio-repetition-codecs-freeze-v170.json"
RESULT = PANEL / "grid-audio-splice-probe-complete-v167.json"
SCRIPT = "scripts.grid_audio_repetition_codecs_v170"
SR = 16000
CHANNELS = {
    "pcm": (".wav", []),
    "aac_64k": (".m4a", ["-c:a", "aac", "-b:a", "64k"]),
    "mp3_64k": (".mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    "opus_48k": (".opus", ["-c:a", "libopus", "-b:a", "48k"]),
}
CUES = {"exact": find_repeated_spans, "similarity": find_similar_spans}
MAX_APPROXIMATE_LAG_ERROR = 32
CODE = (
    "backend/scripts/grid_audio_repetition_codecs_v170.py",
    "backend/scripts/audio_repetition_similarity.py",
    "backend/tests/test_audio_repetition_similarity.py",
    "backend/scripts/audio_exact_repetition.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/app/detectors/audio/decode.py",
    "docs/grid-audio-repetition-codecs-v170-method.md",
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def decoder_tools():
    tools = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        require(path is not None, "Both local codec tools are required")
        tools[name] = sha(Path(path).resolve())
    return tools


def identity():
    prior.verify()
    require(prior.readout() == read(RESULT), "Complete verified prior source study required")
    return {
        "schema": "chipmunk.grid-audio-repetition-codecs-freeze/170",
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "prior_source_freeze_sha256": sha(prior.PUBLIC),
        "prior_source_result_sha256": sha(RESULT),
        "codec_tool_sha256": decoder_tools(),
        "fixed_channels": {key: [suffix, args] for key, (suffix, args) in CHANNELS.items()},
        "minimum_copy_witness_samples": MIN_REPEAT_SAMPLES,
        "maximum_similarity_copy_lag_error_samples": MAX_APPROXIMATE_LAG_ERROR,
        "planned_cases": 16 * len(prior.ARMS) * len(CHANNELS),
        "scope": "Consumed GRID waveform similarity engineering. No fitting, AI directions or customer change.",
    }


def verify():
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == identity(), "Exact frozen codec study required")


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve earlier attempts")
    lock = identity()
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"status": "frozen", "planned_cases": lock["planned_cases"], "new_neural_calls": 0}))


def run_command(argv, *, data=None):
    return subprocess.run(argv, input=data, capture_output=True, check=True, timeout=30).stdout


def compressed(pcm, channel, folder):
    from app.detectors.audio.decode import decode_audio, source_ranges

    suffix, options = CHANNELS[channel]
    with tempfile.TemporaryDirectory(prefix="channel-", dir=folder) as temp:
        file = Path(temp) / f"sample{suffix}"
        run_command(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-f",
                "s16le",
                "-ar",
                str(SR),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-map_metadata",
                "-1",
                *options,
                str(file),
            ],
            data=pcm,
        )
        digest = sha(file)
        decoded = decode_audio(file, source_sha256=digest)
        info = json.loads(
            run_command(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=time_base,start_pts,duration_ts",
                    "-of",
                    "json",
                    str(file),
                ]
            )
        )["streams"]
        require(len(info) == 1, "Exactly one encoded audio stream required")
        stream = info[0]
        ranges = source_ranges(decoded.mapping, 0, len(decoded.pcm))
        expected_origin = Fraction(stream["time_base"]) * int(stream["start_pts"]) * SR
        require(
            len(ranges) == 1 and ranges[0]["source_start_pts"] == expected_origin,
            "The decoded origin differs from the codec stream clock",
        )
        expected = len(pcm) // 2
        tail = len(decoded.pcm) - expected
        if channel == "aac_64k":
            declared = Fraction(stream["time_base"]) * int(stream["duration_ts"]) * SR
            require(declared == expected and 0 <= tail < 1024, "Unverified AAC padding extent")
        else:
            require(tail == 0, "Gapless output length differs")
        view = prior.speech.pcm_lattice(decoded.pcm[:expected])
    return view, {
        "encoded_sha256": digest,
        "origin_16k_samples": int(expected_origin),
        "decoded_tail_samples_excluded": tail,
        "analysis_samples": expected,
    }


def outcome(pcm, support, fade, name):
    result = CUES[name](pcm)
    observed = None
    if fade is not None:
        lag = support["shift_samples"]
        start, end = support["output_samples_half_open"]
        left, right = start + fade, min(start + lag, end - fade)
        tolerance = MAX_APPROXIMATE_LAG_ERROR if name == "similarity" else 0
        observed = any(
            abs(span.lag_samples - lag) <= tolerance
            and min(right, span.second[1]) - max(left, span.second[0]) >= MIN_REPEAT_SAMPLES
            and min(right - lag, span.first[1]) - max(left - lag, span.first[0]) >= MIN_REPEAT_SAMPLES
            for span in result.spans
        )
    return {
        "status": result.status,
        "reason": result.reason,
        "spans": [asdict(s) for s in result.spans],
        "witnessed_copy": observed,
    }


def summary(rows):
    require(
        [(r["ordinal"], r["arm"], r["channel"]) for r in rows]
        == [(i, arm, channel) for i in range(16) for arm in prior.ARMS for channel in CHANNELS],
        "Exact complete input order required",
    )
    counts = {}
    for channel in CHANNELS:
        counts[channel] = {}
        for arm in prior.ARMS:
            subset = [row for row in rows if row["channel"] == channel and row["arm"] == arm]
            valid = [row for row in subset if row["status"] == "decoded"]
            counts[channel][arm] = {
                "planned": len(subset),
                "decode_status": dict(Counter(row["status"] for row in subset)),
                "decoded_origins_16k_samples": sorted({row["origin_16k_samples"] for row in valid}),
                "maximum_decoded_tail_samples_excluded": max(
                    (row["decoded_tail_samples_excluded"] for row in valid), default=None
                ),
                **{
                    name: {
                        "statuses": dict(Counter(row[name]["status"] for row in valid)),
                        "with_reported_spans": sum(bool(row[name]["spans"]) for row in valid),
                        "matching_copy_support": sum(row[name]["witnessed_copy"] is True for row in valid),
                    }
                    for name in CUES
                },
            }
    return counts


def worker(output):
    stage = TRIAL / "monitor"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline data-only guard required",
    )
    verify()
    output.mkdir(mode=0o700, exist_ok=False)
    journal = output / "private-observations.jsonl"
    rows = []
    with journal.open("x") as stream:
        for source in read(prior.TRIAL / "private-inputs.json"):
            path = Path(source["pcm_file"])
            require(sha(path) == source["pcm_file_sha256"], "Bound source waveform changed")
            original = np.load(path, allow_pickle=False)
            for arm, fade in prior.ARMS.items():
                pcm = prior.speech.pcm_lattice(prior.render(original, source["copy"], fade))
                require(
                    hashlib.sha256(pcm).hexdigest() == source["pcm_sha256_by_arm"][arm], "PCM recipe differs"
                )
                for channel in CHANNELS:
                    row = {"ordinal": source["ordinal"], "arm": arm, "channel": channel}
                    try:
                        view, evidence = (
                            compressed(pcm, channel, output)
                            if channel != "pcm"
                            else (
                                pcm,
                                {
                                    "origin_16k_samples": 0,
                                    "decoded_tail_samples_excluded": 0,
                                    "analysis_samples": len(pcm) // 2,
                                },
                            )
                        )
                    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as error:
                        row.update(status="unavailable", error_type=type(error).__name__, error=str(error))
                    else:
                        row.update(status="decoded", **evidence)
                        row.update({name: outcome(view, source["copy"], fade, name) for name in CUES})
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    rows.append(row)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.grid-audio-repetition-codecs-result/170",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "planned_cases": len(rows),
            "new_neural_calls": 0,
            "channels": summary(rows),
        },
    )


def routed_popen(argv, **kwargs):
    output = TRIAL / "monitor/output"
    require(
        argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "monitor", str(output)],
        "Exact data-only monitor command required",
    )
    return subprocess.Popen([*argv[:3], SCRIPT, "worker", str(output)], **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda _command, _batch: True,
            "__file__": __file__,
        },
    )("monitor")


def readout():
    verify()
    stage = TRIAL / "monitor"
    journal = stage / "output/private-observations.jsonl"
    result = prior.speech.qualification_profile(stage, 0, {}, Path(__file__).name, command="monitor")
    require(
        result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["channels"] == summary(prior.speech.planning.reviewed.lines(journal)),
        "Exact completed codec receipt required",
    )
    profile = read(stage / "profile.json")
    return {
        **result,
        "private_resource_profile_sha256": sha(stage / "profile.json"),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.command == "worker") == (args.output is not None), "Only the worker takes an output path")
    if args.command == "worker":
        worker(args.output)
    elif args.command == "readout":
        print(json.dumps(readout(), indent=2, allow_nan=False))
    else:
        (freeze if args.command == "freeze" else run)()
