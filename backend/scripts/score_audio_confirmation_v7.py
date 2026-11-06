"""Freeze and score the complete fresh panel against the actual guarded service."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration as serving
from app.detectors.audio.content import window_in_speech
from app.engine.boundary import Deadline
from app.engine.detector import Profile
from app.engine.profiles import SAMPLING
from scripts import audio_guard_v7 as method
from scripts import audio_residual_head_v6 as learner
from scripts.audio_calibration_v3_methods import band_view
from scripts.audio_confirmation_v7 import HEAD_SHA256, PREPARED_SHA256
from scripts.prepare_audio_confirmation_v7 import digest
from scripts.prepare_audio_guard_v7 import implementation as original_implementation
from scripts.prepare_audio_guard_v7 import write
from scripts.run_audio_guard_v7 import capture, load_rows


def implementation():
    root = Path(__file__).resolve().parents[2]
    names = set(original_implementation()) | {
        "backend/scripts/audio_confirmation_v7.py",
        "backend/scripts/prepare_audio_confirmation_variants_v7.py",
        "backend/scripts/score_audio_confirmation_v7.py",
        "backend/scripts/review_audio_confirmation_v7.py",
        "backend/scripts/review_audio_guard_v7.py",
        "backend/scripts/prepare_audio_confirmation_v7.py",
        "backend/scripts/prepare_audio_confirmation_inputs_v7.py",
        "backend/app/detectors/audio/content.py",
        "backend/app/detectors/video/audio.py",
        "backend/app/detectors/video/adapters/audio.py",
        "backend/app/engine/profiles.py",
        "backend/app/engine/boundary.py",
        "backend/app/detectors/audio/calibrations/local-speech-v3.json",
        "docs/audio-guard-v7-confirmation.md",
    }
    return {p: digest(root / p) for p in sorted(names)}


def runtime():
    import scipy
    import sklearn
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "dtype": "float32",
        "threads": 4,
        "packages": {
            name: importlib.metadata.version(name) for name in ("transformers", "pyloudnorm", "soundfile")
        },
        "platform": platform.system(),
        "machine": platform.machine(),
    }


def weights():
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    result = {}
    for identifier in ("pella.voice_v2", "silero.speech_activity"):
        spec = get_registry().manifest.model(identifier)
        if spec.licence.class_ != "commercial":
            raise ValueError("Verified commercial-permission checkpoint required")
        result[identifier] = [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in ensure_verified(spec, force=True)
        ]
    return result


def freeze(root, head_path, output):
    variants_path = root / "variants.json"
    variants = json.loads(variants_path.read_text())
    if digest(root / "prepared-sources.json") != PREPARED_SHA256 or digest(head_path) != HEAD_SHA256:
        raise ValueError("Original prepared sources or fixed v7 head changed")
    if variants["prepared_sources_sha256"] != PREPARED_SHA256 or len(variants["variants"]) != 1080:
        raise ValueError("All1080 exact variants required before freezing")
    project = Path(__file__).resolve().parents[2]
    if variants["preparation_code_sha256"] != digest(
        project / "backend/scripts/prepare_audio_confirmation_variants_v7.py"
    ):
        raise ValueError("Variant preparation code changed")
    head = json.loads(head_path.read_text())
    receipt = head_path.parent.parent / "representations.jsonl"
    if digest(receipt) != head["receipt_sha256"]:
        raise ValueError("Candidate fitting representation receipt changed")
    representation, _ = load_rows(receipt)
    serving.load()
    parity_path = root / "consumed-parity.json"
    parity = json.loads(parity_path.read_text())
    if (
        not parity["passed"]
        or len(parity["cases"]) != 13
        or parity["implementation_sha256"] != implementation()
    ):
        raise ValueError("Exact current runner must pass consumed-source parity before fresh scoring")
    for row in variants["variants"]:
        if digest(root / row["path"]) != row["sha256"]:
            raise ValueError("Frozen variant bytes changed before inference")
    body = {
        "schema": "chipmunk.audio-fresh-confirmation-lock/7",
        "variants_path": str(variants_path.resolve()),
        "variants_sha256": digest(variants_path),
        "prepared_sources_path": str((root / "prepared-sources.json").resolve()),
        "prepared_sources_sha256": PREPARED_SHA256,
        "head_path": str(head_path.resolve()),
        "head_sha256": HEAD_SHA256,
        "serving_calibration_sha256": serving.SHA256,
        "serving_bundle": serving.load(),
        "representation_header": representation,
        "implementation_sha256": implementation(),
        "runtime": runtime(),
        "weights": weights(),
        "consumed_parity_path": str(parity_path.resolve()),
        "consumed_parity_sha256": digest(parity_path),
        "planned": 1080,
        "deployment_eligible": False,
        "scope": "One prospectively frozen fresh120-source comparison with nine conditions and connected groups; no fitting, point selection or outside validation.",
    }
    write(output, body)
    print(
        json.dumps(
            {
                "lock_sha256": digest(output),
                "planned": 1080,
                "head_sha256": HEAD_SHA256,
                "detector_inference_run": False,
            }
        ),
        flush=True,
    )


def check(lock_path):
    lock = json.loads(lock_path.read_text())
    if lock["implementation_sha256"] != implementation() or lock["runtime"] != runtime():
        raise ValueError("Frozen scoring/review method or numerical runtime changed")
    for prefix in ("variants", "prepared_sources", "head", "consumed_parity"):
        if digest(Path(lock[prefix + "_path"])) != lock[prefix + "_sha256"]:
            raise ValueError("Frozen source/candidate artifact changed")
    if (
        lock["head_sha256"] != HEAD_SHA256
        or lock["serving_calibration_sha256"] != serving.SHA256
        or lock["serving_bundle"] != serving.load()
    ):
        raise ValueError("Fixed candidate or actual-service calibration changed")
    return lock


def speech_windows(pcm, activity):
    from app.detectors.video.audio import MAX_VOICE_WINDOWS, VOICE_WINDOW_S, voice_windows

    spans = activity.spans(pcm)
    params = SAMPLING[Profile("full")]
    windows = voice_windows(
        pcm,
        params.voice_hop_s or VOICE_WINDOW_S,
        max_windows=params.audio_windows or MAX_VOICE_WINDOWS,
        keep=functools.partial(window_in_speech, spans),
    )
    return spans, windows


def score_one(pcm, model, activity, head, captured):
    spans, windows = speech_windows(pcm, activity)
    if windows and (len(windows) != 1 or windows[0][0] != 0 or len(windows[0][1]) != len(pcm)):
        raise ValueError("Unexpected source window; no silently changed calibration scope")
    raw, embedding = capture(model, windows[0][1], captured) if windows else (None, None)
    if raw is not None and (not serving.valid(raw) or not np.isfinite(embedding).all()):
        raise ValueError("Nonfinite original representation/response")
    record = {"raw": raw, "embedding": embedding.tolist() if embedding is not None else None}
    candidate_p = learner.predict(record, head["fitted"])
    probe = {"status": "not_required", "raw": None, "embedding": None, "p": None}

    class Recorder:
        def predict(self, chunks):
            if len(chunks) != 1:
                raise ValueError("One exact complete stability window required")
            probe["status"] = "failed"
            q, feature = capture(model, chunks[0], captured)
            if not serving.valid(q) or not np.isfinite(feature).all():
                raise ValueError("Nonfinite stability representation")
            probe.update(
                status="computed",
                raw=q,
                embedding=feature.tolist(),
                embedding_sha256=learner.digest_embedding(feature),
                samples=len(chunks[0]),
                filtered_pcm_sha256=hashlib.sha256(np.asarray(chunks[0], "<f4").tobytes()).hexdigest(),
            )
            return [q]

    recorder = Recorder()
    baseline = serving.evaluate(raw, pcm, windows, recorder, Deadline(300))
    candidate_proposed = method.initial(candidate_p, head["guards"])
    if candidate_proposed == "original" and probe["status"] == "not_required":
        try:
            recorder.predict([band_view(pcm)])
        except Exception as error:
            probe.update(status="failed", error_type=type(error).__name__)
    if probe["status"] == "computed":
        probe["p"] = learner.predict({"raw": probe["raw"], "embedding": probe["embedding"]}, head["fitted"])
    candidate = method.final(candidate_p, probe["p"], head["guards"])
    return {
        **record,
        "embedding_sha256": learner.digest_embedding(embedding) if embedding is not None else None,
        "status": "ok" if windows else "not_applicable",
        "speech_spans_s": [list(s) for s in spans],
        "windows": [
            {"start_sample": round(start * 16000), "end_sample": round(start * 16000) + len(chunk)}
            for start, chunk in windows
        ],
        "original_p": baseline["probability_ai"],
        "original_decision": baseline["decision"],
        "baseline_detail": baseline,
        "candidate_p": candidate_p,
        "candidate_decision": candidate["decision"],
        "candidate_detail": candidate,
        "probe": probe,
        "probe_needed_by_service": baseline["probe_status"] != "not_required_for_decision",
        "probe_needed_by_candidate": candidate_proposed == "original",
    }


def models():
    import torch

    from app.detectors.audio.upgraded import load_speech_activity, load_voice

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    return load_voice("pella.voice_v2", "cpu", "float32").obj, load_speech_activity(
        "silero.speech_activity", "cpu", "float32"
    ).obj


def score(lock_path, output):
    from app.detectors.audio.decode import decode_audio

    lock = check(lock_path)
    if weights() != lock["weights"]:
        raise ValueError("Commercial checkpoint bytes changed")
    head = json.loads(Path(lock["head_path"]).read_text())
    variants = json.loads(Path(lock["variants_path"]).read_text())
    root = Path(variants["source_root"])
    header = {
        "schema": "chipmunk.audio-fresh-confirmation-responses/7",
        "lock_sha256": digest(lock_path),
        "weights": lock["weights"],
        "runtime": lock["runtime"],
        "planned": 1080,
    }
    completed = []
    if output.exists():
        previous, completed = load_rows(output)
        if previous != header or [r["id"] for r in completed] != [
            r["id"] for r in variants["variants"][: len(completed)]
        ]:
            raise ValueError("Cannot resume changed inputs or non-prefix outcomes")
    else:
        write_header = json.dumps(header, allow_nan=False) + "\n"
        with output.open("x") as stream:
            stream.write(write_header)
    model, activity = models()
    reference = lock["representation_header"]
    direction = model.head.weight.detach().cpu().numpy()[0]
    bias = float(model.head.bias.detach().cpu()[0])
    if (
        not np.array_equal(direction, np.asarray(reference["head_weight"], np.float32))
        or bias != reference["head_bias"]
    ):
        raise ValueError("Frozen representation head differs")
    captured = []
    hook = model.head.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().cpu().numpy().copy())
    )
    try:
        with output.open("a") as stream:
            for index, row in enumerate(variants["variants"][len(completed) :], len(completed) + 1):
                started = time.monotonic()
                result = {k: v for k, v in row.items() if k not in ("path", "production_input")}
                try:
                    source = root / row["path"]
                    if digest(source) != row["sha256"]:
                        raise ValueError("Source waveform changed")
                    decoded = decode_audio(source, source_sha256=row["sha256"])
                    if (
                        len(decoded.pcm) != row["samples"]
                        or decoded.mapping["pcm_sha256"] != row["pcm_sha256"]
                    ):
                        raise ValueError("Production PCM differs from frozen evidence")
                    result.update(score_one(decoded.pcm, model, activity, head, captured))
                except Exception as error:
                    result.update(
                        status="failed",
                        error_type=type(error).__name__,
                        raw=None,
                        embedding=None,
                        original_p=None,
                        original_decision="inconclusive",
                        candidate_p=None,
                        candidate_decision="inconclusive",
                        probe={"status": "failed", "raw": None, "p": None, "embedding": None},
                    )
                result["runtime_s"] = time.monotonic() - started
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if index % 30 == 0 or result["status"] == "failed" or index == 1080:
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "planned": 1080,
                                "status": result["status"],
                                "seconds": result["runtime_s"],
                            }
                        ),
                        flush=True,
                    )
    finally:
        hook.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--head", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze" and args.head:
        freeze(args.input, args.head, args.output)
    elif args.stage == "score":
        score(args.input, args.output)
    else:
        parser.error("freeze needs --head")
