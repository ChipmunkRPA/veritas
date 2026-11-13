"""Score a fixed short-edit bank through the existing voice-window interfaces."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio import calibration, short_voice_review
from app.detectors.audio.content import window_in_speech
from app.detectors.video.audio import voice_windows
from app.engine.detector import Profile
from app.engine.profiles import SAMPLING
from app.evidence.audio_short_review_binding import GRID, REVIEW_POINT
from scripts import audio_noise_window_transfer as noise
from scripts import audio_odss_matched_room as audio
from scripts import audio_phoneme_short_edit_intake as intake
from scripts import replay_audio_odss_public as replay

SCHEMA = "chipmunk.audio-phoneme-short-edit-score/1"
FULL = ((0, 64000), (32000, 96000))
SHORT = tuple(GRID)
GRIDS = {"full": FULL, "short": SHORT, "reduced_short": SHORT}
METHOD = {
    "raw": "Compare the existing Full four-second window grid and the existing five two-second short-review windows. Each route uses the same pinned voice classifier, normalization, and independent per-view speech gate. Select the highest admitted raw logit; earlier window breaks ties. The production two-second point is reported as a diagnostic. The six-second source is outside whole-file confidence calibration.",
    "reduction": "A separate two-second arm applies the fixed previously published 512-point frequency power subtraction on the full mixture; no isolated reference or ground-truth edit enters the transform. Use the same speech-coverage rule on the processed view. It is a standalone arm, not an unvalidated addition to the primary rule.",
    "points": "Fit each arm's point at the strict maximum admitted score over ordinary development deliveries, including gain, genuine resampling, untouched, and background-only at all four noise settings. Freeze the points and complete development records before held inference.",
    "readout": "A strict localization must cover half the edit and achieve at least 0.30 interval intersection over union. A separate broad range covers half the edit irrespective of extra time. Also report availability, misplaced flags, exact-control false flags, stronger raw score than the matched ordinary replacement and best-response timing before operating points. Repeated mixtures share their source reader and masker.",
}


def file_inputs(args):
    payloads, keys, _, _ = intake.collect(args)
    alignments = intake.grid_inputs(args, keys, payloads)
    roster, _, _, _, _ = intake.select(args, payloads, keys, alignments)
    old, _ = audio.plan(args.odss_cache, False)
    annotation = args.ami / "ami_public_manual_1.6.2.zip"
    reserved = replay.manifest(annotation, old)
    _, backgrounds = intake._with_edit_overlap(annotation, args.ami, roster, reserved)
    source = audio.read(args.intake)
    audio.require(source["schema"] == intake.SCHEMA and source["runner_sha256"] == audio.file_digest(Path(intake.__file__))
                  and source["source_roster_sha256"] == audio.digest(audio.json_bytes(roster)), "Source intake or runner changed")
    inputs = {}
    for item in roster:
        case = item["case"]
        for level, kind, pcm, _ in intake.render(item, payloads, backgrounds[case]):
            inputs[case, level, kind] = pcm
    rows = source["deliveries"]
    audio.require(len(rows) == len(inputs) == 108 and all(audio.digest(
        inputs[row["case"], row["level"], row["kind"]].tobytes()) == row["pcm_sha256"] for row in rows),
        "Source deliveries do not match the frozen panel")
    return rows, inputs


def reduced(pcm):
    values = pcm.astype(np.float64) / 32768
    denoised = noise.reduce_noise(values)
    out = np.rint(denoised * 32768)
    audio.require(out.shape == pcm.shape and np.isfinite(out).all() and np.max(abs(out)) <= 32767,
                  "Reduced waveform has a different source clock or clips")
    return out.astype("<i2")


def windows(args):
    rows, inputs = file_inputs(args)
    bindings, unique = [], {}
    for index, row in enumerate(rows):
        original = inputs[row["case"], row["level"], row["kind"]]
        denoised = reduced(original)
        grid = tuple((round(start * audio.RATE), round(start * audio.RATE) + len(pcm))
                     for start, pcm in voice_windows(original, SAMPLING[Profile.FULL].voice_hop_s))
        audio.require(grid == FULL, "Current Full native window grid differs")
        view_map = {"full": original, "short": original, "reduced_short": denoised}
        item = {**row, "id": index, "input_windows": {}}
        for arm, pcm in view_map.items():
            pin = audio.digest(pcm.tobytes())
            unique.setdefault(pin, pcm)
            item["input_windows"][arm] = {"view_sha256": pin, "grid": [
                {"range": [a, b], "window_sha256": audio.digest(pcm[a:b].tobytes())} for a, b in GRIDS[arm]]}
        bindings.append(item)
    root = audio.ROOT
    code = (Path(__file__), Path(intake.__file__), Path(noise.__file__), Path(short_voice_review.__file__),
            Path(calibration.__file__), root / "backend/app/detectors/video/audio.py",
            root / "backend/app/detectors/audio/upgraded.py", root / "backend/app/detectors/audio/content.py",
            root / "backend/scripts/run_audio_specialists_v15.py")
    lock = {"schema": SCHEMA, "method": METHOD, "native_rate": audio.RATE, "grid": GRIDS, "production_short_point": REVIEW_POINT,
            "source_intake_sha256": audio.file_digest(args.intake), "source_windows_sha256": audio.digest(audio.json_bytes(bindings)),
            "code_sha256": {str(path.relative_to(root)): audio.file_digest(path) for path in code}, "weights": audio.models.ASSETS,
            "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "torch", "transformers")}}
    return lock, bindings, unique


def read_scores(model, activity, rows, payloads, role):
    arrays, speech, cache, result = {}, {}, {}, []
    counts = Counter()
    for row in rows:
        if row["role"] != role:
            continue
        output = {}
        for arm, item in row["input_windows"].items():
            pin = item["view_sha256"]
            if pin not in arrays:
                pcm = payloads[pin].astype(np.float32) / 32768
                arrays[pin] = pcm
                speech[pin] = activity.spans(pcm)
            output[arm] = []
            for interval in item["grid"]:
                a, b = interval["range"]
                value = None
                if window_in_speech(speech[pin], a / audio.RATE, b / audio.RATE):
                    key = interval["window_sha256"]
                    if key not in cache:
                        probability, = model.predict([arrays[pin][a:b]])
                        cache[key] = float(calibration.logit(probability))
                        counts["classifier_forwards"] += 1
                    value = cache[key]
                    audio.require(math.isfinite(value), "Voice model score is nonfinite")
                output[arm].append({**interval, "score": value})
        result.append({**row, "readings": output})
        if len(result) % 9 == 0:
            print(role, len(result), "deliveries", counts["classifier_forwards"], "classifier forwards", flush=True)
    return {"rows": result, "classifier_forwards": counts["classifier_forwards"], "vad_passes": len(speech)}


def points(development):
    return {arm: max(window["score"] for row in development["rows"] if not row["kind"].startswith("synthetic")
                     for window in row["readings"][arm] if window["score"] is not None) for arm in GRIDS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    for arg in ("cache", "old-cache", "record", "odss-cache", "ami", "intake", "run"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=audio.ROOT / "models/weights/audio-upgrade")
    args = parser.parse_args()
    lock, bindings, payloads = windows(args)
    frozen = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        audio.write(frozen, lock)
        audio.write(args.run / "windows.json", bindings)
        print(f"Froze {len(bindings)} deliveries and all model-input windows before scoring", flush=True)
        return
    audio.require(frozen.read_bytes() == audio.json_bytes(lock) and (args.run / "windows.json").read_bytes() ==
                  audio.json_bytes(bindings), "Pre-inference model inputs changed")
    if args.phase == "replay":
        print(f"Rebuilt all {len(bindings)} committed native and reduced window schedules")
        return

    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_strong_traffic_view import network_free
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    model = load_voice({"weights": str(args.weights), "assets": lock["weights"]})
    activity = SpeechActivity(args.weights / "silero_vad.jit")
    dev_path, point_path, final = (args.run / name for name in ("development.json", "points.json", "public.json"))
    audio.require(not final.exists(), "This assessment already has a receipt")
    if not dev_path.exists():
        development = read_scores(model, activity, bindings, payloads, "development")
        audio.write(dev_path, development)
    else:
        development = audio.read(dev_path)
    operating = {"freeze_sha256": audio.file_digest(frozen), "development_sha256": audio.file_digest(dev_path),
                 "points": points(development)}
    if not point_path.exists():
        audio.write(point_path, operating)
    audio.require(audio.read(point_path) == operating, "Development point changed")
    print("Sealed development-only points before assessment inference", flush=True)
    assessment = read_scores(model, activity, bindings, payloads, "assessment")
    audio.write(final, {"freeze": lock, "development": development, "operating_points": operating, "assessment": assessment})


if __name__ == "__main__":
    main()
