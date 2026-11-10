"""Qualify actual source speech states and native-clock routing; no detector fit."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
import wave
from collections import Counter
from pathlib import Path

from scripts import av_conditioning_clock_v109 as clock
from scripts import av_speech_native_v110 as speech
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = speech.ROOT, speech.BASE, speech.PANEL
TRIAL = BASE / "av-source-encoder-v114"
CACHE = clock.CACHE.parent / "av-source-encoder-v114"
PUBLIC = PANEL / "av-source-encoder-freeze-v114.json"
PLAN = clock.TRIAL / "preparation/clock-plan.jsonl"
WAVES = clock.WAVEFORMS / "waveforms.jsonl"
CODE = (
    "backend/scripts/av_source_encoder_v114.py",
    "backend/tests/test_av_source_encoder_v114.py",
    "docs/av-source-encoder-v114-method.md",
    "backend/scripts/video_appearance_corrected_v112.py",
)
read, write, sha, require = speech.read, speech.write, speech.sha, speech.require


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def source_records():
    plans, waves = lines(PLAN), lines(WAVES)
    require(len(plans) == len(waves) == 48, "Complete original source panel required")
    require(len({r["asset_index"] for r in plans}) == 48, "Unique original source records required")
    by_id = {r["asset_index"]: r for r in waves}
    output = []
    for plan in plans:
        wav = by_id[plan["asset_index"]]
        require(
            plan["source_group"] == wav["source_group"]
            and plan["study_role"] == wav["study_role"]
            and plan["mono_wav_sha256"] == wav["mono16k_wav"]["sha256"]
            and not plan["generator_conditioning_qualified"],
            "Exact source/role/waveform binding required",
        )
        output.append((plan, wav["mono16k_wav"]))
    return output


def read_pcm(record):
    import numpy as np

    name = record["file"]
    require(Path(name).name == name, "Safe cached waveform basename required")
    path = clock.CACHE / name
    require(sha(path) == record["sha256"], "Exact bound waveform bytes required")
    with wave.open(str(path), "rb") as stream:
        require(
            (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getnframes())
            == (16000, 1, 2, record["sample_frames"]),
            "Exact mono16k signed-16 header required",
        )
        raw = stream.readframes(stream.getnframes())
    import hashlib

    require(
        len(raw) == record["sample_frames"] * 2 and hashlib.sha256(raw).hexdigest() == record["pcm_sha256"],
        "Exact complete PCM bytes required",
    )
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / np.float32(32768)
    speech.validate_pcm(pcm)
    return pcm


def audit_chunks(hidden, frames):
    """Compare array selector with separate explicit tick/layer copying."""
    import numpy as np

    zeros = 0
    for frame in frames:
        require(frame["source_sample_status"] == "supported", "Keep unavailable sources explicit")
        expected = np.zeros((50, 384), np.float32)
        for slot, tick in enumerate(frame["tick_indices"]):
            if 0 <= tick < frame["retained_tick_prefix"]:
                for layer in range(5):
                    expected[slot * 5 + layer] = hidden[tick, layer]
            else:
                zeros += 1
        actual = clock.select_chunk(hidden, frame)
        require(actual.tobytes() == expected.tobytes(), "Native source tick/layer/zero routing differs")
    return {"routed_frames": len(frames), "explicit_zero_tick_slots": zeros}


def qualification():
    speech.verify()
    for name in ("monitor", "authored"):
        stage = speech.TRIAL / name
        result, profile = read(stage / "output/result.json"), read(stage / "profile.json")
        require(
            result["status"] == "complete"
            and profile["child_exit_code"] == 0
            and profile["guard"] is None
            and profile["sample_errors"] == 0
            and profile["worker_receipt_sha256"] == sha(stage / "output/result.json"),
            "Successful speech authored/owned-descendant qualification required",
        )
        if name == "monitor":
            require(
                result["owned_descendant_sampled"]
                and result["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
                "Exact owned descendant sampler required",
            )


def identity():
    qualification()
    records = source_records()
    require(
        read(clock.TRIAL / "preparation/result.json")["private_plan_sha256"] == sha(PLAN),
        "Unchanged qualified source-clock plan required",
    )
    return {
        "schema": "chipmunk.av-source-encoder-freeze/114",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "speech_contract_freeze_sha256": sha(speech.PUBLIC),
        "qualified_authored_result_sha256": sha(speech.TRIAL / "authored/output/result.json"),
        "qualified_monitor_result_sha256": sha(speech.TRIAL / "monitor/output/result.json"),
        "source_plan_sha256": sha(PLAN),
        "waveform_journal_sha256": sha(WAVES),
        "sources": 48,
        "frame_coordinates": sum(len(p["frames"]) for p, _ in records),
        "role_counts": dict(Counter(p["study_role"] for p, _ in records)),
        "batches": 8,
        "sources_per_batch": 6,
        "planned_encoder_forwards": 56,
        "scope": "All source states and clock routing only; no authenticity prediction, fitting, generated mouth alignment, finite evidence locality or serving change.",
    }


def freeze():
    body = identity()
    require(body["frame_coordinates"] == 3642, "Complete unchanged coordinate panel required")
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "sources": 48, "new_neural_calls": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged source contract required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "extract"],
        "Only a fixed extraction invocation can be routed",
    )
    routed = list(argv)
    routed[3] = "scripts.av_source_encoder_v114"
    return subprocess.Popen(routed, **kwargs)


def run(batch):
    require(type(batch) is int and 0 <= batch < 8, "One of eight explicit fixed batches required")
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    runner = scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda command, batch: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )
    runner("extract", batch)


def extract(batch, output):
    import numpy as np

    require(type(batch) is int and 0 <= batch < 8, "Fixed source batch required")
    cache = CACHE / f"batch-{batch:02d}"
    cache.mkdir(parents=True, exist_ok=False)
    encoder = speech.NativeEncoder()
    records = source_records()[batch * 6 : (batch + 1) * 6]
    frames, zero_slots = 0, 0
    with (output / "source-states.jsonl").open("x") as journal:
        for offset, (plan, wav) in enumerate(records):
            pcm = read_pcm(wav)
            hidden, proof = encoder.encode(pcm)
            if offset == 0:
                repeat, other = encoder.encode(pcm.copy())
                require(
                    hidden.tobytes() == repeat.tobytes() and proof == other,
                    "Actual source native repeat differs",
                )
                del repeat
            audit = audit_chunks(hidden, plan["frames"])
            file = cache / f"source-{offset:02d}.npy"
            with file.open("xb") as stream:
                np.save(stream, hidden, allow_pickle=False)
            restored = np.load(file, allow_pickle=False)
            require(restored.tobytes() == hidden.tobytes(), "Saved source states changed")
            record = {
                "asset_index": plan["asset_index"],
                "source_group": plan["source_group"],
                "study_role": plan["study_role"],
                "mono_wav_sha256": plan["mono_wav_sha256"],
                "hidden_file": str(file),
                "hidden_file_sha256": sha(file),
                "hidden_shape": list(hidden.shape),
                "proof": proof,
                **audit,
                "encoder_routing_qualified": True,
                "generator_conditioning_qualified": False,
            }
            journal.write(json.dumps(record, allow_nan=False) + "\n")
            journal.flush()
            frames += audit["routed_frames"]
            zero_slots += audit["explicit_zero_tick_slots"]
            del hidden, restored
    require(encoder.forward_count == 7 and len(records) == 6, "Exactly six sources plus one repeat required")
    result = {
        "schema": "chipmunk.av-source-encoder-result/114",
        "status": "complete",
        "batch": batch,
        "sources": 6,
        "routed_frames": frames,
        "explicit_zero_tick_slots": zero_slots,
        "source_journal_sha256": sha(output / "source-states.jsonl"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": encoder.forward_count,
        "defaults_changed": False,
        "scope": "Actual source CPU encoder/routing/save qualification only. No detector accuracy, achieved mouth synchronization or finite temporal evidence locality.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "extract"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze has no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One batch required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(len(args.arguments) == 2, "Exact guarded output/batch required")
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Private exclusive stage required")
        output.mkdir(exist_ok=False)
        extract(int(args.arguments[1]), output)
