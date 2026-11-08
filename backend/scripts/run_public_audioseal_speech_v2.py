"""Separate frozen real-speech watermark/noise study; no AI-authorship inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import acquire_public_audioseal_v2 as assets
from scripts import prepare_public_audioseal_speech_v2 as preparation
from scripts import public_audio_watermark_v1 as method
from scripts.run_public_audio_watermark_v1 import codec

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["clean", "white_0db", "noise_0db", "music_0db", "speech_0db", "mp3_64k"]
STATES = ["unmarked", "marked", "partial"]
CODE = [
    "backend/scripts/acquire_public_audioseal_v2.py",
    "backend/scripts/public_audioseal_assets_v2.json",
    "backend/scripts/prepare_public_audioseal_speech_v2.py",
    "backend/scripts/run_public_audioseal_speech_v2.py",
    "backend/scripts/review_public_audioseal_speech_v2.py",
    "backend/scripts/public_speech_watermark_runtime_v2.py",
    "backend/scripts/public_audio_watermark_v1.py",
    "backend/scripts/run_public_audio_watermark_v1.py",
    "backend/scripts/review_public_audio_watermark_v1.py",
    "backend/scripts/specialist_runtime_v4.py",
    "backend/app/detectors/audio/decode.py",
    "backend/tests/test_public_audioseal_speech_v2.py",
    "docs/public-audioseal-speech-v2-protocol.md",
]
POLICY = {
    "schema": "chipmunk.public-speech-watermark-policy/2",
    "speakers": 32,
    "cases": 576,
    "source_selection": "Known-unused project speaker identities from fixed public train360 first8 shards; source plan frozen before neural inference. External pretraining overlap unknown. Backgrounds are project-consumed, licensed and canonical-speaker disjoint.",
    "source_extent": "First128000 mono16kHz s16 source samples (0–8s), duration eligibility only; no response-based source choice.",
    "normalization": "Unmarked foreground float32 at RMS0.1; add native full-clip public mark, fixed alternating16bit message. Partial only copies marked source samples32000:96000; rest bit-identical.",
    "conditions": CONDITIONS,
    "states": STATES,
    "background": "Four matched0dB backgrounds per speaker: deterministic white noise and exact licensed environmental/music/other-speech excerpts. Gain uses the unmarked foreground RMS, identical for every mark state.",
    "gain": "One float32 peak-safe gain shared across every mark state and condition for a speaker; min(1,0.8/max absolute pre-codec sample). No hidden clipping.",
    "codec": "64kbps seekable MP3 of each clean gained state, decoded mono16kHz float32 with exact sample-count requirement; no alignment trimming or padding.",
    "readout": "Frozen v1 native strict sample>0.5; quarter-second bin selected only when more than half its samples exceed0.5. Every native sample and bin retained.",
    "planned_generator_forwards": 32,
    "planned_detector_forwards": 581,
    "helper_parity_cases": 4,
    "repeat_cases": 1,
    "reporting": "Per-condition unmarked false responses, full/partial misses, selected interval IoU/coverage/outside time; same32 speakers repeated, not576 independent sources. Fixed Wilson95 intervals for file rates.",
    "serving_admission": False,
    "confidence_admission": False,
    "provider_authentication": False,
    "early_stop_eligible": False,
    "scope": "Public-family localization and robustness only. Deliberately marked real speech is not AI-generated speech.",
}


def rms(values):
    return math.sqrt(float(np.mean(values.astype(np.float64) ** 2)))


def normalize(values, target):
    value = rms(values)
    method.require(value > 0 and np.isfinite(value), "Non-silent finite source required")
    return (values.astype(np.float64) * (target / value)).astype(np.float32)


def mixtures(unmarked, marked, backgrounds):
    partial = unmarked.copy()
    partial[32000:96000] = marked[32000:96000]
    states = {"unmarked": unmarked, "marked": marked, "partial": partial}
    pre = {
        (state, condition): (audio if condition == "clean" else audio + backgrounds[condition])
        for state, audio in states.items()
        for condition in CONDITIONS
        if condition != "mp3_64k"
    }
    maximum = max(float(np.max(np.abs(v))) for v in pre.values())
    gain = np.float32(min(1.0, 0.8 / maximum))
    output = {key: (audio * gain).astype(np.float32) for key, audio in pre.items()}
    for condition in CONDITIONS:
        if condition == "mp3_64k":
            continue
        u, p = output[("unmarked", condition)], output[("partial", condition)]
        method.require(
            np.array_equal(u[:32000], p[:32000]) and np.array_equal(u[96000:], p[96000:]),
            "Outside partial interval changed",
        )
    return output, float(gain)


def truth(state):
    return [] if state == "unmarked" else [[0, 128000]] if state == "marked" else [[32000, 96000]]


def metrics(intervals, state):
    selected = np.zeros(128000, bool)
    actual = np.zeros(128000, bool)
    for a, b in truth(state):
        actual[a:b] = True
    for span in intervals:
        selected[span["start_sample"] : span["end_sample"]] = True
    intersection = int(np.count_nonzero(selected & actual))
    union = int(np.count_nonzero(selected | actual))
    return {
        "any_selected": bool(selected.any()),
        "iou": intersection / union if actual.any() else None,
        "mark_coverage": intersection / int(actual.sum()) if actual.any() else None,
        "outside_mark_s": int(np.count_nonzero(selected & ~actual)) / 16000,
    }


def freeze(source, inputs, trial):
    assets.verify(source, json.loads(assets.LOCK.read_text()))
    plan = method.read(inputs / "source-plan.json")
    method.require(
        len(plan["trials"]) == 32 and plan["foreground_project_overlap"] == 0,
        "Unused32-speaker plan required",
    )
    for row in plan["trials"]:
        for item in [row["foreground"], *row["backgrounds"].values()]:
            method.require(
                method.digest(inputs / item["input_path"]) == item["input_sha256"], "Prepared input changed"
            )
    trial.mkdir(exist_ok=False)
    lock = {
        "policy": POLICY,
        "source": str(source),
        "inputs": str(inputs),
        "input_plan_sha256": method.digest(inputs / "source-plan.json"),
        "code_sha256": {p: method.digest(ROOT / p) for p in CODE},
        "asset_acquisition_sha256": method.digest(source / "acquisition.json"),
        "packages": {
            p: importlib.metadata.version(p) for p in ["torch", "numpy", "scipy", "PyYAML", "einops"]
        },
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0],
        "neural_forwards_before_freeze": 0,
        "customer_inputs": 0,
        "serving_admitted": False,
    }
    method.write(trial / "freeze.json", lock)
    return {
        "frozen": True,
        "cases": 576,
        "planned_native_forwards": 613,
        "source_plan_sha256": lock["input_plan_sha256"],
    }


def check(trial):
    lock = method.read(trial / "freeze.json")
    method.require(
        lock["policy"] == POLICY and lock["code_sha256"] == {p: method.digest(ROOT / p) for p in CODE},
        "Frozen study code/policy changed",
    )
    method.require(
        lock["packages"] == {p: importlib.metadata.version(p) for p in lock["packages"]},
        "Native environment changed",
    )
    source, inputs = Path(lock["source"]), Path(lock["inputs"])
    method.require(
        method.digest(inputs / "source-plan.json") == lock["input_plan_sha256"], "Source plan changed"
    )
    method.require(
        method.digest(source / "acquisition.json") == lock["asset_acquisition_sha256"],
        "Acquisition receipt changed",
    )
    assets.verify(source, json.loads(assets.LOCK.read_text()))
    return lock, source, inputs


def run(trial):
    lock, source, inputs = check(trial)
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261108)
    models, loaded = method.load_models(source)
    output = trial / "private"
    output.mkdir(exist_ok=False)
    plan = method.read(inputs / "source-plan.json")
    rows = []
    gforwards = dforwards = 0
    message = torch.tensor([[i % 2 for i in range(16)]], dtype=torch.int64)
    for ordinal, case in enumerate(plan["trials"]):
        foreground = case["foreground"]
        name = preparation.stable(foreground["id"])
        s16 = np.load(inputs / foreground["input_path"], allow_pickle=False)
        method.require(
            method.digest(inputs / foreground["input_path"]) == foreground["input_sha256"]
            and s16.dtype == np.int16
            and s16.shape == (128000,),
            "Frozen source PCM required",
        )
        unmarked = normalize(s16.astype(np.float32) / 32768.0, 0.1)
        with torch.inference_mode():
            watermark = (
                models["generator"]
                .get_watermark(torch.from_numpy(unmarked)[None, None], message=message)
                .numpy()
                .reshape(-1)
            )
        gforwards += 1
        method.require(
            watermark.shape == (128000,) and watermark.dtype == np.float32 and np.isfinite(watermark).all(),
            "Native mark geometry invalid",
        )
        backgrounds = {}
        white = np.random.default_rng(int(name[:16], 16)).normal(size=128000).astype(np.float32)
        backgrounds["white_0db"] = normalize(white, rms(unmarked))
        for kind, ref in case["backgrounds"].items():
            path = inputs / ref["input_path"]
            method.require(method.digest(path) == ref["input_sha256"], "Background changed")
            signal = np.load(path, allow_pickle=False)
            method.require(
                signal.dtype == np.int16 and signal.shape == (128000,), "Exact background PCM required"
            )
            backgrounds[kind + "_0db"] = normalize(signal.astype(np.float32) / 32768.0, rms(unmarked))
        mixtures_pcm, gain = mixtures(unmarked, unmarked + watermark, backgrounds)
        base = output / (name + "-base.npz")
        np.savez_compressed(base, unmarked=unmarked, watermark=watermark, **backgrounds)
        for state in STATES:
            path = output / (name + "-" + state + ".mp3")
            mixtures_pcm[(state, "mp3_64k")] = codec(mixtures_pcm[(state, "clean")], path)
        for condition in CONDITIONS:
            for state in STATES:
                audio = np.ascontiguousarray(mixtures_pcm[(state, condition)], dtype=np.float32)
                method.require(
                    audio.shape == (128000,) and np.isfinite(audio).all() and np.max(np.abs(audio)) <= 1,
                    "Finite unclipped exact8s detector input required",
                )
                tensor = torch.from_numpy(audio)[None, None]
                with torch.inference_mode():
                    scores, bits = models["detector"](tensor)
                dforwards += 1
                method.require(
                    tuple(scores.shape) == (1, 2, 128000)
                    and torch.isfinite(scores).all()
                    and torch.allclose(scores.sum(dim=1), torch.ones((1, 128000)), atol=1e-6, rtol=0),
                    "Native geometry or normalization changed",
                )
                values = scores[0, 1].numpy().copy()
                evidence = method.summarize(values)
                if ordinal == 0 and (state, condition) in [
                    ("unmarked", "clean"),
                    ("marked", "clean"),
                    ("partial", "clean"),
                    ("partial", "music_0db"),
                ]:
                    with torch.inference_mode():
                        fraction, decoded = models["detector"].detect_watermark(tensor)
                    dforwards += 1
                    method.require(
                        abs(float(fraction[0]) - evidence["positive_sample_fraction"]) <= 1e-7
                        and torch.equal(decoded, (bits > 0.5).int()),
                        "Native helper parity failed",
                    )
                if ordinal == 0 and (state, condition) == ("marked", "clean"):
                    with torch.inference_mode():
                        repeat, _ = models["detector"](tensor)
                    dforwards += 1
                    method.require(torch.equal(scores, repeat), "Native repeat differs")
                path = output / (name + "-" + state + "-" + condition + ".npz")
                np.savez_compressed(path, pcm=audio, watermark_response=values)
                rows.append(
                    {
                        "speaker": foreground["speaker"],
                        "foreground_id": foreground["id"],
                        "state": state,
                        "condition": condition,
                        "expected_marked_spans": truth(state),
                        "base_path": str(base),
                        "base_sha256": method.digest(base),
                        "gain": gain,
                        "private_arrays_path": str(path),
                        "private_arrays_sha256": method.digest(path),
                        "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
                        "evidence": evidence,
                        "metrics": metrics(evidence["intervals"], state),
                        "decoded_bits_correct_of16": int(((bits > 0.5).int() == message).sum()),
                    }
                )
        method.write(output / (name + "-cases.json"), {"cases": rows[-18:]})
        print(
            json.dumps(
                {
                    "speakers_completed": ordinal + 1,
                    "cases_completed": len(rows),
                    "generator_forwards": gforwards,
                    "detector_forwards": dforwards,
                }
            ),
            flush=True,
        )
    method.require(
        len(rows) == 576 and gforwards == 32 and dforwards == 581, "Complete declared native plan required"
    )
    method.write(trial / "cases-private.json", {"cases": rows})
    result = {
        "schema": "chipmunk.public-speech-watermark-result/2",
        "policy": POLICY,
        "generator_forwards": gforwards,
        "detector_forwards": dforwards,
        "load_receipts": loaded,
        "case_count": len(rows),
        "source_plan_sha256": lock["input_plan_sha256"],
        "outcomes": aggregate(rows),
        "serving_admitted": False,
        "provider_authenticated": False,
        "used_in_confidence": False,
        "early_stop_eligible": False,
    }
    method.write(trial / "result.json", result)
    print(json.dumps({"completed": len(rows), "outcomes": result["outcomes"]}), flush=True)


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["state"])].append(row["metrics"])
    return [
        {
            "condition": condition,
            "state": state,
            "n": len(items),
            "selected_cases": sum(r["any_selected"] for r in items),
            "selected_rate_wilson95": wilson(sum(r["any_selected"] for r in items), len(items)),
            "misses": None if state == "unmarked" else sum(not r["any_selected"] for r in items),
            "mean_interval_iou": None if state == "unmarked" else float(np.mean([r["iou"] for r in items])),
            "mean_mark_coverage": None
            if state == "unmarked"
            else float(np.mean([r["mark_coverage"] for r in items])),
            "mean_outside_mark_s": float(np.mean([r["outside_mark_s"] for r in items])),
        }
        for (condition, state), items in sorted(groups.items())
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("source", type=Path)
    p.add_argument("inputs", type=Path)
    p.add_argument("trial", type=Path)
    p = sub.add_parser("run")
    p.add_argument("trial", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        print(json.dumps(freeze(args.source, args.inputs, args.trial), indent=2))
    else:
        run(args.trial)
