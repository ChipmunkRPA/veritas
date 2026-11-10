"""Lossless native-picture/mono-PCM assembly qualification; no neural inference."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import types
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import native_mouth_inputs_v116 as inputs
from scripts import native_mouth_runtime_v118 as renderer
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = inputs.ROOT, inputs.BASE, inputs.PANEL
TRIAL = BASE / "native-av-mux-v119"
CACHE = inputs.CACHE.parent / "native-av-mux-v119"
PUBLIC = PANEL / "native-av-mux-freeze-v119.json"
AV_RUNTIME = BASE / "native-av-runtime-v119"
AV_SITE = AV_RUNTIME / "site"
sys.path.insert(0, str(AV_SITE))
CODE = (
    "backend/scripts/native_av_mux_v119.py",
    "backend/tests/test_native_av_mux_v119.py",
    "docs/native-av-mux-v119-method.md",
    "backend/app/detectors/audio/decode.py",
    "backend/app/detectors/video/decode.py",
)
read, write, sha, require = inputs.read, inputs.write, inputs.sha, inputs.require


def runtime_identity():
    plan = read(PANEL / "native-av-runtime-v119.json")
    require(
        plan == read(AV_RUNTIME / "plan.json")
        and plan["name"] == "av"
        and plan["version"] == "19.0.2"
        and plan["license"] == "BSD-3-Clause"
        and sha(AV_RUNTIME / plan["filename"]) == plan["sha256"],
        "Pinned isolated publisher wheel required",
    )
    dist = importlib.metadata.distribution("av")
    require(dist.version == plan["version"], "Exact isolated PyAV version required")
    files = []
    for file in dist.files:
        if file.hash is None:
            continue
        path = Path(dist.locate_file(file)).resolve()
        require(
            path.is_relative_to(AV_SITE.resolve()) and file.hash.mode == "sha256",
            "Isolated runtime source/binary required",
        )
        digest = hashlib.sha256(path.read_bytes()).digest()
        require(
            base64.urlsafe_b64encode(digest).decode().rstrip("=") == file.hash.value,
            "Installed PyAV RECORD differs",
        )
        files.append((str(file), digest.hex()))
    require(files, "Complete installed runtime records required")
    import av

    return {
        "package": plan,
        "installed_files": len(files),
        "installed_manifest_sha256": hashlib.sha256(
            json.dumps(sorted(files), separators=(",", ":")).encode()
        ).hexdigest(),
        "codec_libraries": {k: list(v) for k, v in av.library_versions.items()},
        "preserved_prefreeze_failure_sha256": sha(AV_RUNTIME / "preparation-failure.json"),
    }


def validate_clock(pts, audio_start):
    require(
        len(pts) >= 2
        and len(pts) <= 200
        and all(type(t) is int and t >= 0 for t in pts)
        and all(a < b for a, b in zip(pts, pts[1:], strict=False)),
        "Two to 200 increasing nonnegative native millisecond picture timestamps required",
    )
    require(
        isinstance(audio_start, Fraction) and audio_start >= 0 and (audio_start * 1000).denominator == 1,
        "Exact nonnegative millisecond audio origin required",
    )
    return [b - a for a, b in zip(pts, pts[1:], strict=False)]


def encode(path, frames, pts, shape, pcm, audio_start):
    import av

    steps = validate_clock(pts, audio_start)
    require(
        isinstance(pcm, np.ndarray)
        and pcm.ndim == 1
        and pcm.dtype == np.dtype("<i2")
        and 640 <= len(pcm) <= 480000,
        "Complete bounded mono16k signed-16 PCM required",
    )
    require(
        len(shape) == 3 and shape[2] == 3 and 0 < shape[0] <= 1080 and 0 < shape[1] <= 1920,
        "Native RGB dimensions required",
    )
    last_step = sorted(steps)[len(steps) // 2]
    durations = {t: d for t, d in zip(pts, [*steps, last_step], strict=True)}
    hashes = []
    with path.open("xb") as file:
        with av.open(file, mode="w", format="matroska", options={"avoid_negative_ts": "disabled"}) as output:
            video = output.add_stream("ffv1", rate=25)
            video.width, video.height = shape[1], shape[0]
            video.pix_fmt = "bgr0"
            video.time_base = video.codec_context.time_base = Fraction(1, 1000)
            video.codec_context.thread_count = 1
            audio = output.add_stream("pcm_s16le", rate=16000)
            audio.layout = "mono"
            audio.time_base = Fraction(1, 16000)
            audio.codec_context.time_base = Fraction(1, 16000)
            for rgb, t in zip(frames, pts, strict=True):
                require(
                    rgb.dtype == np.uint8 and rgb.shape == shape, "Complete exact native RGB input required"
                )
                hashes.append(inputs.speech.speech.fingerprint(rgb))
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
                frame.pts = t
                frame.time_base = Fraction(1, 1000)
                for packet in video.encode(frame):
                    require(
                        packet.time_base == Fraction(1, 1000) and packet.pts in durations,
                        "Native encoder clock changed",
                    )
                    packet.duration = durations[packet.pts]
                    output.mux(packet)
            for packet in video.encode():
                require(
                    packet.time_base == Fraction(1, 1000) and packet.pts in durations,
                    "Native flush clock changed",
                )
                packet.duration = durations[packet.pts]
                output.mux(packet)
            for start in range(0, len(pcm), 1024):
                frame = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(pcm[start : start + 1024].reshape(1, -1)),
                    format="s16",
                    layout="mono",
                )
                frame.sample_rate = 16000
                frame.time_base = Fraction(1, 16000)
                frame.pts = int(audio_start * 16000) + start
                for packet in audio.encode(frame):
                    output.mux(packet)
            for packet in audio.encode():
                output.mux(packet)
    return {
        "rgb_sha256": hashes,
        "picture_pts_ms": list(pts),
        "constructed_last_picture_duration_ms": last_step,
        "audio_start_seconds": str(audio_start),
        "audio_samples": len(pcm),
        "pcm_s16_sha256": __import__("hashlib").sha256(pcm.tobytes()).hexdigest(),
    }


def verify_media(path, expected, shape):
    from app.detectors.audio.decode import decode_audio, source_ranges
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(path)
    require(
        table.width == shape[1]
        and table.height == shape[0]
        and list(table.pts) == expected["picture_pts_ms"]
        and table.time_base == "1/1000"
        and not table.interpolated
        and not table.truncated,
        "Complete exact output picture clock required",
    )
    frames = [
        inputs.speech.speech.fingerprint(f.rgb)
        for f in decode_frames(path, table, range(len(table)), timeout_s=120)
    ]
    require(frames == expected["rgb_sha256"], "Decoded RGB differs from native inputs")
    sound = decode_audio(path, source_sha256=sha(path), max_seconds=30)
    pcm = np.rint(sound.pcm * 32768).astype("<i2")
    require(
        len(pcm) == expected["audio_samples"]
        and __import__("hashlib").sha256(pcm.tobytes()).hexdigest() == expected["pcm_s16_sha256"],
        "Decoded soundtrack sample bytes differ",
    )
    ranges = source_ranges(sound.mapping, 0, len(sound.pcm))
    require(
        len(ranges) == 1
        and Fraction(ranges[0]["source_start_pts"], 16000) == Fraction(expected["audio_start_seconds"]),
        "Output audio origin or continuity changed",
    )
    return {
        "pictures": len(table),
        "audio_samples": len(pcm),
        "picture_clock_exact": True,
        "rgb_exact": True,
        "pcm_exact": True,
        "audio_origin_exact": True,
    }


def identity():
    renderer.context()["verify"]()
    stage = renderer.TRIAL / "authored"
    result, profile = read(stage / "output/result.json"), read(stage / "profile.json")
    require(
        result["status"] == "complete"
        and result["new_neural_calls"] == 17
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(stage / "output/result.json"),
        "Successful native renderer contract required",
    )
    return {
        "schema": "chipmunk.native-av-mux-freeze/119",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "renderer_freeze_sha256": sha(renderer.PUBLIC),
        "renderer_result_sha256": sha(stage / "output/result.json"),
        "native_input_freeze_sha256": sha(inputs.PUBLIC),
        "private_native_plan_sha256": sha(inputs.TRIAL / "plan.json"),
        "waveform_journal_sha256": sha(inputs.speech.WAVES),
        "packages": {p: importlib.metadata.version(p) for p in ("av", "numpy", "psutil")},
        "isolated_av_runtime": runtime_identity(),
        "authored_cases": 2,
        "fitting_source_rewrites": 1,
        "new_neural_calls": 0,
        "scope": "Full decoded native picture clocks/RGB and mono16k sample/origin preservation only; final display duration is constructed from the upper median picture step. No new generation, detector score or achieved synchronization.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged assembly contract required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed data qualification can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.native_av_mux_v119"
    return subprocess.Popen(routed, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda command, batch: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


def qualify(output):
    from app.detectors.video.decode import FrameTable, decode_frames

    CACHE.mkdir(exist_ok=False)
    cases = []
    for index, (pts, origin, samples) in enumerate(
        (
            ([27, 60, 94, 127, 160, 260, 294, 327], Fraction(0), 6400),
            ([67 + 40 * i for i in range(12)], Fraction(21, 500), 7001),
        )
    ):
        frames = []
        yy, xx = np.indices((64, 96))
        for i in range(len(pts)):
            frames.append(
                np.stack(
                    ((xx + i * 7) % 256, (yy * 3 + i * 11) % 256, (xx + yy + i * 13) % 256), axis=2
                ).astype(np.uint8)
            )
        pcm = ((np.arange(samples, dtype=np.int64) * 29) % 65536 - 32768).astype("<i2")
        file = CACHE / f"authored-{index}.mkv"
        expected = encode(file, frames, pts, (64, 96, 3), pcm, origin)
        cases.append(
            {
                "case": "authored",
                "index": index,
                "file_sha256": sha(file),
                "expected": expected,
                "review": verify_media(file, expected, (64, 96, 3)),
            }
        )
    source = read(inputs.TRIAL / "plan.json")[0]
    require(source["study_role"] == "fusion_fit", "Fixed first fitting source required")
    table = FrameTable.from_json(source["table"])
    index = source["asset_index"]
    parent = next(
        r for r in read(inputs.native.TRIAL / "qualified-plan.json")["assets"] if r["index"] == index
    )
    clock, wav = inputs.speech.source_records()[0]
    path = Path(source["source_path"])
    require(sha(path) == source["input_sha256"], "Exact fitting source required")
    wavepath = inputs.speech.clock.CACHE / wav["file"]
    require(sha(wavepath) == wav["sha256"], "Exact qualified waveform required")
    with wave.open(str(wavepath), "rb") as stream:
        require(
            (stream.getnchannels(), stream.getframerate(), stream.getsampwidth(), stream.getnframes())
            == (1, 16000, 2, wav["sample_frames"]),
            "Qualified waveform header required",
        )
        pcm = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()

    def frames():
        for frame in decode_frames(path, table, range(len(table)), timeout_s=120):
            require(
                inputs.speech.speech.fingerprint(frame.rgb) == parent["rgb_sha256"][frame.index],
                "Original source RGB changed",
            )
            yield frame.rgb

    origin = Fraction(clock["frames"][0]["source_audio_origin_seconds"])
    file = CACHE / "fitting-source-rewrite.mkv"
    expected = encode(file, frames(), table.pts, (table.height, table.width, 3), pcm, origin)
    cases.append(
        {
            "case": "fitting_source",
            "file_sha256": sha(file),
            "expected": expected,
            "review": verify_media(file, expected, (table.height, table.width, 3)),
        }
    )
    write(output / "private-cases.json", cases)
    result = {
        "schema": "chipmunk.native-av-mux-result/119",
        "status": "complete",
        "cases": 3,
        "authored_cases": 2,
        "fitting_source_rewrites": 1,
        "pictures": sum(r["review"]["pictures"] for r in cases),
        "audio_samples": sum(r["review"]["audio_samples"] for r in cases),
        "all_picture_clocks_rgb_pcm_audio_origins_exact": True,
        "private_case_results_sha256": sha(output / "private-cases.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Observed picture centers/RGB and mono16k PCM/origin only; constructed terminal display duration, not source exposure/duration proof. Same codec library through PyAV encoding and command decoding, not an independent implementation or detector accuracy.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze has no arguments")
        freeze()
    elif args.command == "run":
        require(not args.arguments, "Run has no arguments")
        run()
    else:
        verify()
        require(len(args.arguments) == 2 and args.arguments[1] == "0", "Exact guarded fixed output required")
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive private output required")
        output.mkdir(exist_ok=False)
        qualify(output)
