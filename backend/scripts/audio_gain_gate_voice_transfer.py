"""Test the downstream decision when gain admits an otherwise skipped voice window."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_paired_competing_speech as competing
from scripts import audio_paired_loud_background as io

SCHEMA = "chipmunk.audio-gain-gate-voice-transfer/1"
CALIBRATION = "912e1f25a6813fc610f6c93301263a77b36e01d40b6ba555c4421b57762719bf"
WEIGHTS = io.ROOT / "models/weights/audio-upgrade"
ENVIRONMENTAL = io.ROOT / "backend/data/audio-replays/persistence-confirmation"
DIRECTIONS = ("ai", "original", "inconclusive", "unassessed")
METHOD = {
    "inputs": "All 116 retained environmental-noise cases and all 77 retained competing-voice cases. Restore exact stored mono samples, original labels and prior primary results. These source files have prior project exposure.",
    "comparison": "Hold the three-second half-speech admission rule and the primary voice calibration frozen. For an originally unassessed window, try only the same speech-activity gate on a clip-free +6 dB copy. When that gate admits, score the ORIGINAL samples with the current voice detector and its stability check.",
    "advance": "At least two additional correct synthetic detections from separate source groups and named generator families, with no more than one new noisy genuine false AI flag, no new clean false AI flag and no AI flag on isolated background. A separate fresh test would still be required for adoption.",
    "confidence": "Record both alternative directions. Extrapolation of the incumbent calibration to these windows is diagnostic only.",
}


def plan(archive):
    groups = {}

    def group(bank, key):
        return groups.setdefault((bank, key), len(groups))

    rows = []
    for case, observed in io.parent():
        source = case["source"]
        rows.append({"bank": "environmental", "id": case["id"], "scene": case["scene"],
                     "label": {"bonafide": "genuine", "spoof": "synthetic", None: "background"}[
                         source["label"] if source else None],
                     "group": group("environmental", source["id"]) if source else None,
                     "generator_family": None, "before": observed["primary"],
                     "pcm_sha256": case["analysis_pcm_sha256"]})
    for case, observed in competing.rows_from_archive(archive):
        family = case["family"]
        rows.append({"bank": "competing", "id": case["id"], "scene": case["condition"],
                     "label": "synthetic" if family in competing.FAMILIES else family,
                     "group": group("competing", case["group"]) if family != "background" else None,
                     "generator_family": family if family in competing.FAMILIES else None,
                     "before": observed["decision"], "pcm_sha256": case["pcm_sha256"]})
    return rows


def lock(archive):
    return {"schema": SCHEMA, "method": METHOD, "runner_sha256": io.digest(Path(__file__)),
            "parents": {path.name: io.digest(path) for path in (io.PARENT, competing.PARENT)},
            "calibration_sha256": CALIBRATION, "rows": plan(archive)}


def measure(folder, environmental, archive):
    import numpy as np

    from app.core.config import settings
    from app.detectors.audio import calibration
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    frozen = io.read(folder / "freeze.json")
    io.check(frozen == lock(archive) and calibration.SHA256 == CALIBRATION, "A frozen input or method differs")
    network_free()
    settings.audio_detector_mode = "candidate"
    activity = SpeechActivity(WEIGHTS / "silero_vad.jit")
    voice = load_voice({"weights": str(WEIGHTS), "assets": ASSETS})
    env = {case["id"]: case for case, _ in io.parent()}
    observations = []
    for row in frozen["rows"]:
        case = row["id"]
        if row["bank"] == "environmental":
            original = io.waveform(environmental, env[case])
            checksum = hashlib.sha256(original.astype("<f4").tobytes()).hexdigest()
        else:
            raw = (archive / "pcm" / f"{case:03}.pcm").read_bytes()
            original = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
            checksum = hashlib.sha256(raw).hexdigest()
        io.check(original.shape == (48000,) and checksum == row["pcm_sha256"], "The original input changed")
        before = adapter_result(original, voice, activity)["decision"]
        io.check(before == row["before"], f"Changed primary decision for {row['bank']}/{case}")
        original_speech, gained_speech, alternative = None, None, None
        if before == "unassessed":
            original_speech = count_samples(activity.spans(original))
            io.check(original_speech < 24000, "Unassessed window contains enough detected speech")
            peak = max(float(np.max(np.abs(original))), 1e-9)
            multiplier = min(10**(6/20), (32767 / 32768) / peak)
            gained_spans = activity.spans(original * multiplier)
            gained_speech = count_samples(gained_spans)
            if gained_speech >= 24000:
                from types import SimpleNamespace

                def admitted_spans(pcm, expected=original, spans=gained_spans):
                    io.check(np.array_equal(pcm, expected), "The voice adapter must use original samples")
                    return spans

                alternative = adapter_result(original, voice, SimpleNamespace(spans=admitted_spans))["decision"]
                io.check(alternative != "unassessed", "The admitted source window was not scored")
        observations.append({"original_speech_samples": original_speech, "gain_speech_samples": gained_speech,
                             "after": alternative or before})
        if len(observations) % 24 == 0:
            print(f"Measured {len(observations)} of {len(frozen['rows'])} deliveries", flush=True)
    receipt = {"freeze": frozen, "observations": observations, "model_files": ASSETS}
    io.write(folder / "receipt.json", receipt | {"summary": summarize(receipt)})


def count_samples(spans):
    return sum(round(b * 16000) - round(a * 16000) for a, b in spans)


def summarize(receipt):
    frozen, measurements = (receipt[key] for key in ("freeze", "observations"))
    inputs = frozen["rows"]
    io.check(frozen["schema"] == SCHEMA and frozen["method"] == METHOD and frozen["calibration_sha256"] == CALIBRATION
             and frozen["runner_sha256"] == io.digest(Path(__file__)) and len(inputs) == len(measurements) == 193,
             "Incorrect method or coverage")
    io.check(frozen["parents"] == {path.name: io.digest(path) for path in (io.PARENT, competing.PARENT)},
             "The prior decision ledgers changed")
    counts = defaultdict(Counter)
    gained, families = set(), set()
    false_noisy = false_clean = false_background = 0
    for case, row in zip(inputs, measurements, strict=True):
        before, after = case["before"], row["after"]
        original, gain = row["original_speech_samples"], row["gain_speech_samples"]
        io.check(before in DIRECTIONS and after in DIRECTIONS, "Invalid voice direction")
        if before == "unassessed":
            io.check(type(original) is int and 0 <= original < 24000 and type(gain) is int and 0 <= gain <= 48000
                     and (after != "unassessed") == (gain >= 24000), "The gain-only admission does not replay")
        else:
            io.check(original is None and gain is None and before == after, "The incumbent treatment changed")
        label, scene = case["label"], case["scene"]
        count = counts[f"{case['bank']}/{scene}/{label}"]
        count["files"] += 1
        count[f"old_{before}"] += 1
        count[f"new_{after}"] += 1
        count["newly_scored"] += before == "unassessed" and after != "unassessed"
        if before != "ai" and after == "ai":
            if label == "synthetic":
                gained.add(case["group"])
                if case["generator_family"]:
                    families.add(case["generator_family"])
            elif label == "background":
                false_background += 1
            elif scene == "clean":
                false_clean += 1
            else:
                false_noisy += 1
    advance = len(gained) >= 2 and len(families) >= 2 and false_noisy <= 1 and false_clean == false_background == 0
    return {"by_condition": counts, "gained_synthetic_source_groups": len(gained), "gained_named_families": sorted(families),
            "added_false_noisy": false_noisy, "added_false_clean": false_clean, "added_false_background": false_background,
            "advance": advance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--environmental", default=ENVIRONMENTAL, type=Path)
    parser.add_argument("--archive", default=competing.ARCHIVE, type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.path.mkdir(mode=0o700)
        io.write(args.path / "freeze.json", lock(args.archive))
    elif args.phase == "measure":
        measure(args.path, args.environmental, args.archive)
    else:
        receipt = io.read(args.path)
        summary = summarize(receipt)
        io.check(summary == receipt["summary"], "Published results do not replay")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
