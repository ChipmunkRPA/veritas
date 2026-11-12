"""Run the production Full voice adapter on the sealed call-processing panels."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.detectors.audio import calibration, decode
from app.detectors.video.adapters.audio import PellaVoiceDetector
from app.detectors.video.adapters.common import AudioTrack
from app.detectors.video.audio import voice_windows
from app.engine.detector import Profile
from app.engine.profiles import SAMPLING
from app.engine.stages.audio import AUDIO_MAX_WINDOWS
from scripts import audio_communication_replication as replication
from scripts import audio_communication_transfer as transfer

BASE, ROOT = replication.BASE, replication.ROOT
SCHEMA = "chipmunk.audio-communication-serving-grid/1"
PANELS = {
    "discovery": "audio-communication-transfer-2026-11-12",
    "confirmation": "audio-communication-replication-2026-11-12",
}
PARENT = BASE / "audio-remote-neighbor-replication-2026-11-12"
RUN = BASE / "audio-communication-serving-grid-2026-11-12"
WEIGHTS = ROOT / "models/weights/audio-upgrade"
RATE, SAMPLES = transfer.RATE, transfer.SAMPLES
GRID = ((0, 64000), (32000, 96000))
ARMS = transfer.ARMS[:2]
read, write, sha, digest, require = transfer.read, transfer.write, transfer.sha, transfer.digest, transfer.require


def public_receipt(name):
    return ROOT / "docs/evaluation-panels" / f"{PANELS[name]}.json"


def cases(name):
    folder = BASE / PANELS[name]
    public = read(public_receipt(name))
    require(all(sha(folder / f"{key}.json") == pin for key, pin in public["commitments"].items()),
            f"Public commitments do not match retained {name} panel")
    if name == "discovery":
        for row, original, _, delivered, path in transfer.panel(folder, PARENT):
            yield row, original, delivered, path
    else:
        yield from replication.panel(folder)


def freeze():
    source_files = (
        "backend/app/detectors/video/adapters/audio.py",
        "backend/app/detectors/video/audio.py",
        "backend/app/detectors/audio/upgraded.py",
        "backend/app/detectors/audio/content.py",
        "backend/app/detectors/audio/calibration.py",
        "backend/app/detectors/audio/calibrations/local-speech-v3.json",
        "backend/app/detectors/audio/decode.py",
        "backend/app/engine/profiles.py",
        "backend/app/engine/stages/audio.py",
        "backend/app/engine/stages/audio_screening.py",
        "backend/scripts/run_audio_specialists_v15.py",
    )
    points = {read(public_receipt(name))["fixed_point"] for name in PANELS}
    require(len(points) == 1, "Short-window research point changed between the source panels")
    measured_grid = [
        [round(start * RATE), round(start * RATE) + len(chunk)]
        for start, chunk in voice_windows(np.zeros(SAMPLES, np.float32), SAMPLING[Profile.FULL].voice_hop_s,
                                          max_windows=AUDIO_MAX_WINDOWS)
    ]
    require(measured_grid == [list(window) for window in GRID], "The actual production Full window plan changed")
    return {
        "schema": SCHEMA,
        "runner_sha256": sha(Path(__file__)),
        "implementation": {name: sha(ROOT / name) for name in source_files},
        "prior_receipts": {name: sha(public_receipt(name)) for name in PANELS},
        "model_assets": transfer.OLD.voice.ASSETS,
        "calibration_scope_seconds": [calibration.load()[key] for key in ("min_seconds", "max_seconds")],
        "profile": "full",
        "selection": "candidate",
        "sampling_rate": RATE,
        "samples_per_file": SAMPLES,
        "grid": measured_grid,
        "fixed_short_window_logit": points.pop(),
        "useful_min_iou": 0.30,
        "useful_min_edit_coverage": 0.50,
        "rules": {
            "paired": "Minimum score across the two deliveries of each same admitted range, then choose the best range.",
            "flag": "Maximum admitted window strictly above the unchanged experimental two-second logit.",
            "geometry": "Count any known-edit overlap, >=50% coverage, full coverage, and useful localization independently.",
            "fit": "No refit. No new source recruitment. No serving edits or customer confidence changes.",
        },
    }


def verify_call(pcm, path):
    decoded = decode.decode_audio(path, source_sha256=sha(path), max_seconds=7)
    require(decoded.mapping["timeline"] == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}]
            and (decoded.pcm * 32768).astype("<i2").tobytes() == pcm,
            "Sealed call delivery did not re-decode to its original samples and clock")


def serving_result(pcm, lease):
    values = np.frombuffer(pcm, "<i2").astype(np.float32) / 32768
    ctx = SimpleNamespace(
        profile=Profile.FULL,
        outputs=SimpleNamespace(audio=AudioTrack("analysed", values, samples=len(values), max_windows=AUDIO_MAX_WINDOWS)),
        lease=lease,
        exhibits=SimpleNamespace(series=lambda *args, **kwargs: "offline-native-audio-exhibit"),
        deadline=SimpleNamespace(check=lambda *args: None),
    )
    result = PellaVoiceDetector().run(ctx)
    require(result.status in {"ok", "not_applicable"}, "The native adapter did not return a valid assessment")
    details = result.details.get("audio_screening")
    if result.status == "ok":
        require(details and not details["scope_supported"] and details["decision"] == "inconclusive"
                and details["probability_ai"] is None, "Six-second production calibration scope changed")
    else:
        require(not details, "Inapplicable detector unexpectedly supplied screening")
    unit_scores = {
        (int(unit.quality["start_sample"]), int(unit.quality["end_sample"])): calibration.logit(unit.raw)
        for unit in result.unit_scores
    }
    require(set(unit_scores).issubset(GRID) and len(unit_scores) == len(result.unit_scores),
            "The production adapter scored a different grid")
    return {
        "status": result.status,
        "decision": details["decision"] if details else None,
        "scope_supported": details["scope_supported"] if details else None,
        "probability_ai": details["probability_ai"] if details else None,
        "pcm_sha256": digest(pcm),
        "float_pcm_sha256": digest(values.astype("<f4").tobytes()),
        "windows": [{"start": a, "end": b, "score": unit_scores.get((a, b)), "pcm_sha256": digest(pcm[2*a:2*b])}
                    for a, b in GRID],
    }


def run_score(run, weights):
    from app.core.config import settings
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    lock = read(run / "freeze.json")
    require(lock == freeze(), "Frozen production implementation, assets, inputs, or comparison changed")
    require(settings.audio_detector_mode == "candidate", "Set VERITAS_AUDIO_DETECTOR_MODE=candidate")
    transfer.OLD.spectra.incumbent.network_free()
    require(all(sha(weights / name) == pin for name, pin in lock["model_assets"].items()), "Pinned local weights changed")
    voice = load_voice({"weights": str(weights), "assets": lock["model_assets"]})
    activity = SpeechActivity(weights / "silero_vad.jit")
    resources = {
        PellaVoiceDetector.MODEL_ID: SimpleNamespace(obj=voice, revision=lock["model_assets"]["pellav2_detector.pt"]),
        "silero.speech_activity": SimpleNamespace(obj=activity, revision=lock["model_assets"]["silero_vad.jit"]),
    }
    lease = SimpleNamespace(get=resources.__getitem__)
    for name in PANELS:
        rows = []
        for index, (row, original, delivered, path) in enumerate(cases(name), 1):
            verify_call(delivered, path)
            rows.append({
                "id": row["id"],
                "views": {arm: serving_result(pcm, lease) for arm, pcm in zip(ARMS, (original, delivered), strict=True)},
            })
            if index % 13 == 0:
                print(f"{name}: scored {index}/52 exact native input pairs", flush=True)
        require(len(rows) == 52, "Expected 52 sealed pairs in each cohort")
        write(run / f"{name}-scores.json", {"freeze_sha256": sha(run / "freeze.json"), "records": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score"))
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    args = parser.parse_args()
    if args.phase == "prepare":
        for name in PANELS:
            require(sum(1 for _ in cases(name)) == 52, "Unexpected retained panel size")
        lock = freeze()
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        print(f"Locked the native Full grid, 104 source pairs and unchanged short-window point: {sha(args.run / 'freeze.json')}")
    else:
        run_score(args.run, args.weights)


if __name__ == "__main__":
    main()
