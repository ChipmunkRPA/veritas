"""Score a frozen consumed-audio enhancement panel, without changing serving state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    import numpy as np
    import torch

    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import load_speech_activity, load_voice
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry
    from scripts.audio_calibration_v3_methods import score_views
    from scripts.audio_noise_view_v4 import METHOD, PINS, enhance, load

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = json.loads(args.manifest.read_text())
    if manifest["role"] != "consumed v3 development; not independent confirmation":
        raise ValueError("This development runner cannot score a fresh confirmation cohort")
    project = Path(__file__).resolve().parents[2]
    for name, expected in manifest["implementation_sha256"].items():
        if digest(project / name) != expected:
            raise ValueError("Development implementation changed after freeze: " + name)
    source = Path(manifest["source_root"])
    if (
        digest(source / "manifest.json") != manifest["source_manifest_file_sha256"]
        or digest(source / "predictions.json") != manifest["reference_predictions_file_sha256"]
    ):
        raise ValueError("Consumed source records changed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    registry = get_registry()
    verified = {}
    for model_id in ("pella.voice_v2", "silero.speech_activity"):
        model = registry.manifest.model(model_id)
        if model is None or model.licence.class_ != "commercial":
            raise ValueError("Missing commercial model permission record")
        files = ensure_verified(model, force=True)
        verified[model_id] = [
            {"sha256": digest(p), "bytes": p.stat().st_size, "filename": p.name} for p in files
        ]
    header = {
        "schema": "chipmunk.audio-noise-score/4",
        "manifest_sha256": digest(args.manifest),
        "implementation_sha256": manifest["implementation_sha256"],
        "method": METHOD,
        "weights": verified,
        "enhancer_files": PINS,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": "cpu",
        "threads": 4,
        "deployment_eligible": False,
    }
    completed = []
    if args.output.exists():
        with args.output.open() as stream:
            prior, *completed = [json.loads(line) for line in stream]
        if prior != header:
            raise ValueError("Cannot resume different inputs, method, weights or runtime")
        expected_ids = [r["id"] for r in manifest["variants"][: len(completed)]]
        if [r["id"] for r in completed] != expected_ids:
            raise ValueError("Scores are not the exact completed manifest prefix")
        if any(r["status"] not in ("ok", "not_applicable", "failed") for r in completed):
            raise ValueError("Invalid prior outcome")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")
    if len(completed) == len(manifest["variants"]):
        print("All planned development inputs already have terminal receipts.")
        return
    enhancer = load(args.enhancer)
    classifier = load_voice("pella.voice_v2", "cpu", "float32").obj
    activity = load_speech_activity("silero.speech_activity", "cpu", "float32").obj
    smokes = []
    for label in (0, 1):
        for condition in ("clean", "noise_0"):
            choices = [
                r
                for r in manifest["variants"]
                if r["label"] == label and r["condition"] == condition and r["raw"] is not None
            ]
            if not choices:
                raise ValueError("Missing source class/condition for original-path parity smoke")
            smokes.append(min(choices, key=lambda r: r["id"])["id"])
    with args.output.open("a") as stream:
        for n, row in enumerate(manifest["variants"][len(completed) :], len(completed) + 1):
            out = {
                "id": row["id"],
                "source_sha256": row["sha256"],
                "source_pcm_sha256": row["pcm_sha256"],
                "enhanced_raw": None,
                "enhanced_windows": [],
                "status": "not_applicable",
                "reason": "No original-view speech windows were assessable.",
            }
            started = time.monotonic()
            try:
                path = source / row["path"]
                if digest(path) != row["sha256"]:
                    raise ValueError("Consumed source bytes differ from their frozen hash")
                decoded = decode_audio(path, source_sha256=row["sha256"])
                if decoded.mapping["pcm_sha256"] != row["pcm_sha256"] or len(decoded.pcm) != row["samples"]:
                    raise ValueError("Production-decoded source PCM changed")
                if row["id"] in smokes:
                    replay = score_views(decoded.pcm, classifier, activity)
                    bounds = [(w["start_sample"], w["end_sample"]) for w in replay["windows"]]
                    expected = [(w["start_sample"], w["end_sample"]) for w in row["original_windows"]]
                    if (
                        bounds != expected
                        or replay["raw_unrounded"] is None
                        or abs(replay["raw_unrounded"] - row["raw"]) > 1e-7
                    ):
                        raise ValueError("Original-view scoring/gate no longer matches the stored baseline")
                    out["original_path_parity"] = True
                if row["raw"] is not None and row["original_windows"]:
                    transformed = enhance(decoded.pcm, enhancer)
                    chunks = [
                        transformed[w["start_sample"] : w["end_sample"]] for w in row["original_windows"]
                    ]
                    values = classifier.predict(chunks)
                    if len(values) != len(chunks) or not all(np.isfinite(v) and 0 <= v <= 1 for v in values):
                        raise ValueError("Enhancement view returned invalid classifier responses")
                    out.update(
                        status="ok",
                        reason=None,
                        enhanced_raw=float(np.mean(values)),
                        enhanced_pcm_sha256=hashlib.sha256(transformed.astype("<f4").tobytes()).hexdigest(),
                        enhanced_samples=len(transformed),
                        enhanced_windows=[
                            {**w, "response": float(v)}
                            for w, v in zip(row["original_windows"], values, strict=True)
                        ],
                    )
            except Exception as error:
                out.update(status="failed", reason=f"{type(error).__name__}: {error}")
            out["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(out, allow_nan=False) + "\n")
            stream.flush()
            if n % 10 == 0 or n == len(manifest["variants"]) or out["status"] == "failed":
                print(
                    json.dumps(
                        {
                            "completed": n,
                            "planned": len(manifest["variants"]),
                            "id": row["id"],
                            "status": out["status"],
                            "seconds": out["runtime_s"],
                            "reason": out["reason"],
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("enhancer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args)
