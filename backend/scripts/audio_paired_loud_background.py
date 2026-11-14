"""Replay a frozen second opinion on the English speech panel at −5 dB."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
PARENT = PANELS / "audio-background-persistence-2026-11-13.json"
FIT = PANELS / "audio-paired-consistency-complete-v223.json"
ENCODER = PANELS / "audio-wavlm-three-second-complete-v226.json"
SCHEMA = "chipmunk.audio-paired-loud-background/1"
BACKGROUND_POINT = 7.422796281843017
PAIRED_POINT = 0.6525272861441811
METHOD = "paired_wavlm_raw"
RECIPE = {
    "source": "All 116 held persistence deliveries: gain-matched clean, street and washroom at −5 dB, and isolated backgrounds.",
    "routes": "Compare primary, primary plus recorded background reduction, primary plus paired score, and their union.",
    "admission": "Score the exact 48,000 original samples with the paired model only when the primary is inconclusive and admitted the entire window.",
    "point": "Keep the prior four-second English point, without fitting. Both supplements can only add an AI flag. Preserve original and unassessed directions where there is no supplemental flag.",
    "advance": "For the union, require two new useful noisy synthetic flags from distinct files, at most one additional genuine source, no clean losses and no background-only flag. This comparison cannot establish generator independence or calibrated probabilities.",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def parent():
    source = read(PARENT)
    return list(zip(source["holdout"]["rows"], source["holdout_results"], strict=True))


def settings():
    directory = ROOT / "backend/data/independent-evaluation/audio-paired-consistency-v223"
    fitted, calibrated = directory / "fitted.json", directory / "calibration.json"
    published = read(FIT)
    check(digest(fitted) == published["private_fit_sha256"]
          and digest(calibrated) == published["private_calibration_sha256"], "The private fit or diagnostic point differs")
    check(read(calibrated)["points"][METHOD]["ai_above"] == PAIRED_POINT, "The published comparison point differs")
    return read(fitted)[METHOD]


def lock():
    return {"schema": SCHEMA, "recipe": RECIPE, "runner_sha256": digest(Path(__file__)),
            "parents": {p.name: digest(p) for p in (PARENT, FIT, ENCODER)}, "point": PAIRED_POINT,
            "background_point": BACKGROUND_POINT}


def waveform(directory, case):
    from scripts.audio_background_rate_diagnostic import decode

    path = directory / "wav" / f"{case['id']:03}.wav"
    check(digest(path) == case["wav_sha256"], "Frozen WAV bytes differ")
    pcm, checksum = decode(path, 16_000, case["wav_sha256"])
    check(checksum == case["analysis_pcm_sha256"], "Frozen production PCM differs")
    return pcm


def baseline(run, directory):
    from app.core.config import settings as config
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.audio_radar_transfer import adapter_result
    from scripts.audio_strong_traffic_view import network_free
    from scripts.prepare_audio_specialists_v15 import ASSETS
    from scripts.run_audio_specialists_v15 import load_voice

    network_free()
    config.audio_detector_mode = "candidate"
    weights = ROOT / "models/weights/audio-upgrade"
    voice = load_voice({"weights": str(weights), "assets": ASSETS})
    activity = SpeechActivity(weights / "silero_vad.jit")
    rows = []
    for case, expected in parent():
        pcm = waveform(directory, case)
        baseline_input = adapter_result(pcm, voice, activity)
        # Reopening the exact WAV also checks the recorded duration and timestamp extent.
        check(baseline_input["decision"] == expected["primary"], f"Primary direction changed on {case['id']}")
        check(not baseline_input["windows"] or baseline_input["windows"] == [[0, 48_000]], "Inexact primary clock")
        rows.append({"id": case["id"], **baseline_input})
        if len(rows) % 20 == 0:
            print(f"Replayed primary on {len(rows)} files", flush=True)
    write(run / "baseline.json", rows)


def score(run, directory):
    import torch

    from scripts import audio_paired_transfer_v230 as transfer
    from scripts import audio_wavlm_three_second_v226 as native
    from scripts.audio_strong_traffic_view import network_free
    from scripts.audio_wavlm_features_v13 import load

    network_free()
    head = settings()
    check(native.checkpoint() == read(native.PUBLIC)["checkpoint"], "The encoder or runtime differs")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = load(native.WEIGHTS)
    rows = []
    for (case, expected), primary in zip(parent(), read(run / "baseline.json"), strict=True):
        check(case["id"] == primary["id"] and expected["primary"] == primary["decision"], "Incorrect primary binding")
        candidate = None
        if primary["decision"] == "inconclusive":
            check(primary["windows"] == [[0, 48_000]] and primary["scope"] is True, "Incorrect paired window")
            feature = native.extract(waveform(directory, case), model)
            check(feature["input_pcm_sha256"] == case["analysis_pcm_sha256"], "Altered candidate input")
            candidate = transfer.paired_score(primary["raw"], feature, head)
        rows.append({"id": case["id"], "primary": primary["decision"], "raw": primary["raw"],
                     "paired": candidate})
        if len(rows) % 20 == 0:
            print(f"Compared paired decisions on {len(rows)} files", flush=True)
    result = {"lock": read(run / "freeze.json"), "baseline_sha256": digest(run / "baseline.json"), "rows": rows,
              "summary": summarize(rows)}
    write(run / "receipt.json", result)
    print(json.dumps(result["summary"], indent=2))


def summarize(rows):
    counts, noisy_additions = {}, Counter()
    added_sources = {"bonafide": set(), "spoof": set()}
    for (case, expected), row in zip(parent(), rows, strict=True):
        check(row["id"] == case["id"] and row["primary"] == expected["primary"], "Published row order or baseline differs")
        primary, raw, paired = row["primary"], row["raw"], row["paired"]
        check((raw is None) == (primary == "unassessed") and (raw is None or math.isfinite(raw) and 0 <= raw <= 1),
              "Primary raw response differs from its admission state")
        check((paired is not None) == (primary == "inconclusive")
              and (paired is None or math.isfinite(paired) and 0 <= paired <= 1), "Incorrect score coverage")
        source = case["source"]
        label = source["label"] if source else "background"
        group = counts.setdefault(f"{case['scene']}/{label}", Counter())
        group["planned"] += 1
        group[f"primary_{primary}"] += 1
        group["paired_assessed"] += paired is not None
        enhanced = expected["enhanced"] is not None and expected["enhanced"] > BACKGROUND_POINT
        second = paired is not None and paired > PAIRED_POINT
        check(not enhanced or primary in ("inconclusive", "unassessed"), "Background flag cannot override a primary direction")
        group["current_with_background_ai"] += primary == "ai" or enhanced
        group["paired_instead_ai"] += primary == "ai" or second
        group["combined_ai"] += primary == "ai" or enhanced or second
        group["both_supplements_flagged"] += enhanced and second
        if second and not enhanced and case["scene"] != "clean" and source:
            noisy_additions[label] += 1
            added_sources[label].add(source["id"])
    isolated = sum(row["combined_ai"] for key, row in counts.items() if key.endswith("/background"))
    distinct = {label: len(sources) for label, sources in added_sources.items()}
    qualifies = distinct["spoof"] >= 2 and distinct["bonafide"] <= 1 and isolated == 0
    return {"counts": counts, "new_noisy_flags": dict(noisy_additions), "distinct_noisy_sources": distinct,
            "isolated_background_flags": isolated, "advance_to_named_generator_confirmation": qualifies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "baseline", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT / "backend/data/audio-replays/persistence-confirmation")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.path)
        frozen = receipt["lock"]
        check(frozen == lock(), "Unknown frozen comparison")
        result = summarize(receipt["rows"])
        check(receipt["summary"] == result, "Counts differ from the saved ledger")
        print(json.dumps(result, indent=2))
    elif args.phase == "freeze":
        args.path.mkdir(mode=0o700)
        write(args.path / "freeze.json", lock())
    else:
        check(read(args.path / "freeze.json") == lock(), "The prospective lock changed")
        (baseline if args.phase == "baseline" else score)(args.path, args.source)


if __name__ == "__main__":
    main()
