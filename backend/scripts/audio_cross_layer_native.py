"""Evaluate the frozen cross-layer heads on rights-reviewed, prepared native audio."""
from __future__ import annotations

import argparse
import hashlib
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_cross_layer_transfer as experiment

SAMPLES = 48000


def pcm(folder, row):
    path = folder / "pcm" / f"{int(row['id'])}.pcm"
    data = path.read_bytes()
    if len(data) != SAMPLES * 2 or hashlib.sha256(data).hexdigest() != row["pcm_sha256"]:
        raise ValueError("Prepared native voice samples changed")
    return np.frombuffer(data, "<i2").astype(np.float32) / 32768


def lock_for(folder):
    manifest = experiment.read(folder / "native-manifest.json")
    fitted = experiment.read(folder / "fitted.json")
    return {
        "schema": "chipmunk.audio-cross-layer-native/1",
        "native_code_sha256": experiment.digest(Path(__file__)),
        "fit_code_sha256": experiment.digest(Path(experiment.__file__)),
        "fitted_sha256": experiment.digest(folder / "fitted.json"),
        "manifest_sha256": experiment.digest(folder / "native-manifest.json"),
        "rows": len(manifest["rows"]),
    }, fitted, manifest


def run(folder, weights):
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.boundary import Deadline
    from scripts.audio_layer_features_v12 import LayerCapture
    from scripts.audio_noise_speech_gate_v256 import CountedVAD
    from scripts.run_audio_specialists_v15 import load_voice

    lock, fitted, manifest = lock_for(folder)
    if experiment.read(folder / "native-freeze.json") != lock:
        raise ValueError("Original fitted head and native manifest freeze required")
    catalogue = experiment.read(experiment.PANEL / experiment.SOURCE)
    forbidden = {row[key] for row in catalogue["sources"] if row["role"] in {"fit", "cal"}
                 for key in ("sha256", "prefix_pcm_sha256", "production_pcm_sha256")}
    forbidden_pcm = {row["pcm_sha256"] for row in catalogue["variants"]["development"] if row["role"] in {"fit", "cal"}}
    if any(r["source_sha256"] in forbidden or r["pcm_sha256"] in forbidden_pcm for r in manifest["rows"]):
        raise ValueError("Native transfer overlaps a fitting or calibration source commitment")
    calibration_path = experiment.ROOT / "backend/app/detectors/audio/calibrations/local-speech-v3.json"
    if experiment.digest(calibration_path) != manifest["calibration_sha256"]:
        raise ValueError("Current calibration bytes changed")
    model = load_voice({"weights": str(weights), "assets": manifest["baseline_assets"]})
    activity = SpeechActivity(weights / "silero_vad.jit")
    activity.model = gate = CountedVAD(activity.model)
    windows = PellaVoiceDetector()
    matrix = experiment.projection()
    rows = []
    calls = 0

    def counted(*_):
        nonlocal calls
        calls += 1

    handle = model.register_forward_pre_hook(counted)
    started = time.monotonic()
    try:
        with LayerCapture(model) as capture:
            for row in manifest["rows"]:
                wave = pcm(folder, row)
                spans = activity.spans(wave)

                def lease_activity(model_id, spans=spans):
                    if model_id != "silero.speech_activity":
                        raise ValueError("Unexpected authenticity lease during speech gate")
                    return SimpleNamespace(obj=SimpleNamespace(spans=lambda _: spans))

                ctx = SimpleNamespace(profile="full", lease=SimpleNamespace(get=lease_activity),
                                      outputs=SimpleNamespace(audio=AudioTrack("analysed", wave, len(wave))))
                admitted = windows._windows(ctx)
                result = {**row, "decision": "unassessed", "raw": None, "projected": None}
                if admitted:
                    if len(admitted) != 1 or admitted[0][0] != 0 or not np.array_equal(admitted[0][1], wave):
                        raise ValueError("Native speech gate admitted an unexpected window")
                    raw, = model.predict([wave])
                    evidence = capture.take()
                    features = experiment.project(evidence["layer_embeddings"], matrix)
                    capture.clear()
                    decision = calibration.evaluate(raw, wave, admitted, model, Deadline(120))
                    if not decision["scope_supported"]:
                        raise ValueError("Native window outside incumbent calibration")
                    result.update(decision=decision["decision"], raw=float(raw), projected=features.tolist(),
                                  exact_features_sha256=evidence["layer_embedding_sha256"],
                                  feature_frames=evidence["feature_frames"], incumbent_probe=decision["probe_status"])
                    capture.clear()  # The existing stability probe is separate from the primary readout.
                rows.append(result)
                if len(rows) % 10 == 0:
                    print(f"{len(rows)}/{len(manifest['rows'])} native cases, {calls} voice forwards", flush=True)
    finally:
        handle.remove()
    if gate.calls != len(rows) * ((SAMPLES + 511) // 512):
        raise ValueError("Measured speech-gate forward count differs")
    experiment.write(folder / "native-scores.json", {"rows": rows, "native_freeze_sha256": experiment.digest(folder / "native-freeze.json"),
                     "voice_forwards": calls, "speech_gate_calls": gate.calls,
                     "elapsed_s": time.monotonic()-started, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def readout(folder):
    lock, fitted, manifest = lock_for(folder)
    measurement = experiment.read(folder / "native-scores.json")
    freeze = folder / "native-freeze.json"
    if (experiment.read(freeze) != lock or measurement["native_freeze_sha256"] != experiment.digest(freeze)
            or len(measurement["rows"]) != len(manifest["rows"])):
        raise ValueError("Native transfer is incomplete or frozen sources differ")
    for recorded, expected in zip(measurement["rows"], manifest["rows"], strict=True):
        if {key: recorded[key] for key in expected} != expected:
            raise ValueError("Native output source bindings differ")
        pcm(folder, expected)
    rows = measurement["rows"]
    results = {arm: experiment.summarize(rows, experiment.score(rows, arm, fitted["heads"].get(arm)),
                                        fitted["points"][arm]["value"]) for arm in experiment.ARMS}
    experiment.write(folder / "native-readout.json", {"planned": len(rows), "arms": results,
                     "native_scores_sha256": experiment.digest(folder / "native-scores.json"),
                     "native_freeze_sha256": experiment.digest(freeze),
                     **{key: measurement[key] for key in ("voice_forwards", "speech_gate_calls", "elapsed_s", "peak_rss_bytes")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "score", "readout"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--weights", type=Path, default=experiment.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    if args.action == "freeze":
        lock, _, manifest = lock_for(args.folder)
        if not manifest["rows"] or [r["id"] for r in manifest["rows"]] != list(range(len(manifest["rows"]))):
            raise ValueError("Ordered unique native case IDs required")
        for row in manifest["rows"]:
            pcm(args.folder, row)
        experiment.write(args.folder / "native-freeze.json", lock)
    elif args.action == "score":
        run(args.folder, args.weights)
    else:
        readout(args.folder)


if __name__ == "__main__":
    main()
