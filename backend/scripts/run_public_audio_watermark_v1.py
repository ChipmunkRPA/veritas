"""Freeze/run public AudioSeal on authored marked and unmarked signal controls."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts import public_audio_watermark_v1 as method

ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "backend/scripts/public_audio_watermark_v1.py",
    "backend/scripts/run_public_audio_watermark_v1.py",
    "backend/scripts/public_watermark_runtime_v1.py",
    "backend/scripts/specialist_runtime_v4.py",
    "backend/tests/test_public_audio_watermark_v1.py",
    "docs/public-audio-watermark-v1-protocol.md",
)
POLICY = {
    "schema": "chipmunk.public-audio-watermark-authored-policy/1",
    "scope": "Authored signal implementation test, not real-speech or deployed-provider accuracy validation.",
    "device": "CPUfloat32, frozen generator/detector, two Torch threads, no serving imports.",
    "sources": "Four8s16kHz mono carriers: harmonic pulses, speech-shaped noise pulses, chirp mixture, silence. Seed20261108.",
    "variants": [
        "unmarked",
        "marked",
        "partial",
        "unmarked_noise",
        "marked_noise",
        "partial_noise",
        "unmarked_mp3",
        "partial_mp3",
    ],
    "mark": "Fixed alternating16bit message; full or source samples32000:96000 only. Unmarked samples outside partial span must be bit-identical.",
    "noise": "Same white-noise realization at10dB relative to carrier RMS, floor.001; no watermark added to negative controls.",
    "mp3": "64kbps seekable MP3, decoded16kHzfloat32; exact sample-count preservation required, no hidden alignment trimming/padding.",
    "readout": "Native per-sample watermark response, strict>0.5; quarter-second bins selected only when more than half their samples exceed0.5. Preserve sample coordinates.",
    "parity": "Four high-level helper fractions/bits match direct native outputs (fraction tolerance1e-7); first marked native response repeats bit-exactly.",
    "identity": "No provider/creator authentication, AI confidence contribution or detection early stop from this public-family response.",
    "planned_cases": 32,
    "planned_generator_forwards": 4,
    "planned_detector_forwards": 37,
    "serving_admitted": False,
}


def freeze(source, trial):
    method.verify_source(source)
    trial.mkdir(parents=True, exist_ok=False)
    lock = {
        "policy": POLICY,
        "source_root": str(source),
        "code_sha256": {p: method.digest(ROOT / p) for p in CODE},
        "asset_receipts_sha256": {
            p: method.digest(source / p)
            for p in (
                "source-review.json",
                "downloads.json",
                "publisher-model-tree.json",
                "isolated-python-files.json",
                "restricted-weight-inspection.json",
            )
        },
        "packages": {
            p: importlib.metadata.version(p) for p in ("torch", "numpy", "scipy", "PyYAML", "einops")
        },
        "weights": method.WEIGHTS,
        "native_forwards_before_freeze": 0,
        "source_code_revision": method.CODE_REVISION,
        "source_model_revision": method.MODEL_REVISION,
        "customer_inputs": 0,
        "provider_calls": 0,
        "serving_admitted": False,
    }
    method.write(trial / "freeze.json", lock)
    print(
        json.dumps({"frozen": True, "cases": 32, "neural_forwards": 41, "serving_admitted": False}),
        flush=True,
    )


def check(trial):
    lock = method.read(trial / "freeze.json")
    method.require(
        lock["policy"] == POLICY
        and lock["code_sha256"] == {p: method.digest(ROOT / p) for p in CODE}
        and lock["packages"] == {p: importlib.metadata.version(p) for p in lock["packages"]},
        "Frozen code/runtime changed",
    )
    root = Path(lock["source_root"])
    method.require(
        all(method.digest(root / p) == sha for p, sha in lock["asset_receipts_sha256"].items()),
        "Public asset receipts changed",
    )
    method.verify_source(root)
    return root


def carriers():
    from scipy.signal import butter, sosfiltfilt

    rate = method.RATE
    t = np.arange(8 * rate, dtype=np.float64) / rate
    envelope = np.maximum(0, np.sin(2 * np.pi * 2 * t)) ** 2
    rng = np.random.default_rng(20261108)
    harmonic = (
        sum(np.sin(2 * np.pi * freq * t) / i for i, freq in enumerate((140, 280, 420, 560, 700), 1))
        * 0.07
        * envelope
    )
    noise = sosfiltfilt(
        butter(3, [150, 3800], btype="bandpass", fs=rate, output="sos"), rng.normal(size=len(t))
    )
    noise = noise / np.sqrt(np.mean(noise**2)) * 0.07 * envelope
    chirp = (
        (np.sin(2 * np.pi * (180 * t + 80 * t * t)) + 0.4 * np.sin(2 * np.pi * 1100 * t))
        * 0.08
        * (0.4 + 0.6 * envelope)
    )
    return {
        "harmonic": harmonic.astype(np.float32),
        "pulsed_noise": noise.astype(np.float32),
        "chirp_mix": chirp.astype(np.float32),
        "silence": np.zeros(len(t), np.float32),
    }


def codec(pcm, path):
    method.require(np.abs(pcm).max() <= 1.0, "Do not silently clip codec input")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            "-threads",
            "1",
            str(path),
        ],
        input=pcm.astype("<f4").tobytes(),
        check=True,
        timeout=30,
    )
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    method.require(
        len(decoded.stdout) == len(pcm) * 4,
        "Gapless decoded sample count differs; do not hide timing uncertainty",
    )
    return np.frombuffer(decoded.stdout, dtype="<f4").copy()


def run(trial):
    source = check(trial)
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261108)
    models, load_receipts = method.load_models(source)
    folder = trial / "authored-private"
    folder.mkdir(exist_ok=False)
    message = torch.tensor([[i % 2 for i in range(16)]], dtype=torch.int64)
    rows = []
    gforwards = dforwards = 0
    repeat_difference = None
    for number, (name, pcm) in enumerate(carriers().items()):
        with torch.inference_mode():
            watermark = (
                models["generator"]
                .get_watermark(torch.from_numpy(pcm)[None, None], message=message)
                .numpy()
                .reshape(-1)
            )
        gforwards += 1
        method.require(
            watermark.shape == pcm.shape and watermark.dtype == np.float32 and np.isfinite(watermark).all(),
            "Finite full-length watermark required",
        )
        marked = pcm + watermark
        partial = pcm.copy()
        partial[32000:96000] += watermark[32000:96000]
        method.require(
            np.array_equal(partial[:32000], pcm[:32000]) and np.array_equal(partial[96000:], pcm[96000:]),
            "Partial watermark changed unmarked source samples",
        )
        noise = np.random.default_rng(100 + number).normal(size=len(pcm))
        noise = (
            noise
            / np.sqrt(np.mean(noise**2))
            * max(float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))), 0.001)
            / np.sqrt(10)
        ).astype(np.float32)
        variants = {
            "unmarked": pcm,
            "marked": marked,
            "partial": partial,
            "unmarked_noise": pcm + noise,
            "marked_noise": marked + noise,
            "partial_noise": partial + noise,
            "unmarked_mp3": codec(pcm, folder / (name + "-unmarked.mp3")),
            "partial_mp3": codec(partial, folder / (name + "-partial.mp3")),
        }
        for variant in POLICY["variants"]:
            audio = np.ascontiguousarray(variants[variant], dtype=np.float32)
            tensor = torch.from_numpy(audio)[None, None]
            with torch.inference_mode():
                scores, bits = models["detector"](tensor)
            dforwards += 1
            method.require(
                tuple(scores.shape) == (1, 2, len(audio))
                and torch.isfinite(scores).all()
                and torch.allclose(scores.sum(dim=1), torch.ones((1, len(audio))), atol=1e-6, rtol=0),
                "Native response/time geometry or normalization differs",
            )
            values = scores[0, 1].numpy().copy()
            evidence = method.summarize(values)
            if number == 0 and variant in ("unmarked", "marked", "partial", "partial_noise"):
                with torch.inference_mode():
                    fraction, decoded = models["detector"].detect_watermark(tensor)
                dforwards += 1
                method.require(
                    abs(float(fraction[0]) - evidence["positive_sample_fraction"]) <= 1e-7
                    and torch.equal(decoded, (bits > 0.5).int()),
                    "High-level native helper differs",
                )
            if number == 0 and variant == "marked":
                with torch.inference_mode():
                    repeated, _ = models["detector"](tensor)
                dforwards += 1
                repeat_difference = float(torch.max(torch.abs(scores - repeated)))
                method.require(repeat_difference == 0, "Authored native repeat differs")
            archive = folder / (name + "-" + variant + ".npz")
            np.savez_compressed(archive, pcm=audio, watermark_response=values)
            expected = (
                [] if variant.startswith("unmarked") else [{"start_sample": 32000, "end_sample": 96000}]
            )
            if variant.startswith("marked"):
                expected = [{"start_sample": 0, "end_sample": len(audio)}]
            rows.append(
                {
                    "carrier": name,
                    "variant": variant,
                    "expected_marked_spans": expected,
                    "evidence": evidence,
                    "decoded_bits_correct_of16": int(((bits > 0.5).int() == message).sum()),
                    "private_arrays_sha256": method.digest(archive),
                    "private_arrays_path": str(archive),
                    "source_pcm_sha256": __import__("hashlib").sha256(audio.tobytes()).hexdigest(),
                }
            )
            print(
                json.dumps(
                    {
                        "completed": len(rows),
                        "carrier": name,
                        "variant": variant,
                        "positive_sample_fraction": evidence["positive_sample_fraction"],
                        "intervals": evidence["intervals"],
                    }
                ),
                flush=True,
            )
    method.require(
        len(rows) == 32 and gforwards == 4 and dforwards == 37,
        "All declared cases and native forwards required",
    )
    method.write(trial / "cases-private.json", {"cases": rows, "loader": load_receipts})
    groups = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    aggregate = {
        variant: {
            "cases": len(cases),
            "cases_with_selected_interval": sum(bool(r["evidence"]["intervals"]) for r in cases),
            "mean_positive_sample_fraction": float(
                np.mean([r["evidence"]["positive_sample_fraction"] for r in cases])
            ),
        }
        for variant, cases in groups.items()
    }
    check(trial)
    method.write(
        trial / "result.json",
        {
            "schema": "chipmunk.public-audio-watermark-authored-result/1",
            "policy": POLICY,
            "freeze_sha256": method.digest(trial / "freeze.json"),
            "cases_private_sha256": method.digest(trial / "cases-private.json"),
            "cases": 32,
            "generator_forwards": gforwards,
            "detector_forwards": dforwards,
            "repeat_maximum_difference": repeat_difference,
            "loader": load_receipts,
            "aggregate": aggregate,
            "provider_calls": 0,
            "customer_inputs": 0,
            "serving_admitted": False,
            "deployment_compatibility_verified": False,
            "outside_human_reviews": 0,
        },
    )
    print(
        json.dumps(
            {
                "completed": 32,
                "generator_forwards": gforwards,
                "detector_forwards": dforwards,
                "aggregate": aggregate,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("source", type=Path)
    parser.add_argument("trial", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.source.resolve(), args.trial.resolve())
    else:
        run(args.trial.resolve())
