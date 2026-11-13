# SPDX-License-Identifier: Apache-2.0
"""Check a frozen positive-only speech combination on unused RADAR recordings."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

from scripts import audio_radar_long_confirmation as radar_dev
from scripts import audio_radar_transfer as radar
from scripts import audio_selective_fusion_transfer as study
from scripts import review_audio_radar_transfer as native

ROOT = study.ROOT
SAMPLES = 64_000
WEIGHTS = radar.common.WEIGHTS
ENCODER = ROOT / "models/research/wavlm-v13"
BUNDLE = study.BASE / "audio-combined-serving-v97/serving-bundle.json"
SCHEMA = "chipmunk.audio-selective-radar-transfer/1"
PER_LABEL = 8
METHOD = {
    "sources": "Eight unused whole-label English FLAC sources per label per RADAR archive prefix, five to twelve seconds before cutting.",
    "span": [0, SAMPLES], "sample_rate": 16000,
    "input": "Exact first four continuous seconds from the independent decoder. No gain, resampling above the source profile, padding or augmentation.",
    "arms": ["current", "raw", "global"],
    "rule": "Preserve the original decision except an assessed inconclusive recording may lean AI at the original fitted point.",
    "advancement": "At least four additional spoof files including at least one from each split and at most one extra genuine flag. Generator, speaker and noise identities are unavailable; this alone cannot authorize adoption.",
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def source_bindings(source):
    names = ("archive-prefix.gz", "labels.txt", "languages.txt", "LICENSE.txt", *radar_dev.SOURCES)
    expected = {name: (radar.SOURCE_FILES | radar_dev.SOURCES)[name] for name in names}
    study.need({name: study.sha(source / name) for name in names} == expected, "Pinned RADAR publisher artifacts differ")
    return expected


def prior_inputs(source):
    excluded_ids, excluded_flac = set(), set()
    hashes = {}
    for relative in ("run/intake.json", "panel/intake.json", "confirmation/intake.json"):
        path = source / relative
        hashes[relative] = study.sha(path)
        for row in study.read(path):
            excluded_ids.add(row["id"])
            excluded_flac.add(row["flac_sha256"])
    return excluded_ids, excluded_flac, hashes


def extract_candidates(source, split, excluded_ids, excluded_flac):
    prefix, labels, languages, matcher = (("dev-prefix.gz", "dev-labels.txt", "dev-languages.txt", radar_dev.MEMBER) if split == "development"
                                          else ("archive-prefix.gz", "labels.txt", "languages.txt", radar.MEMBER))
    label_rows, language_rows = (native.table(source / name) for name in (labels, languages))
    candidates, seen = [], set()
    try:
        with tarfile.open(source / prefix, "r|gz") as stream:
            for member in stream:
                match = matcher.fullmatch(member.name)
                if not match or not member.isfile():
                    continue
                ident = match[1]
                study.need(ident not in seen and ident in label_rows and 0 < member.size <= 2 * 1024**2, "Invalid archive member")
                seen.add(ident)
                duration, label, *segments = label_rows[ident]
                if (ident in excluded_ids or language_rows[ident] != ["EN"] or not 5 <= Decimal(duration) <= 12
                        or label not in radar.ORIGINS or segments != [f"0.0000-{duration}-{label}"]):
                    continue
                with stream.extractfile(member) as file:
                    payload = file.read()
                study.need(len(payload) == member.size and payload[:4] == b"fLaC", "Incomplete selected FLAC")
                checksum = digest(payload)
                if checksum not in excluded_flac:
                    candidates.append((ident, label, duration, checksum, payload))
    except (EOFError, tarfile.ReadError) as error:
        study.need(isinstance(error, EOFError) or "end of data" in str(error), "Unexpected publisher prefix failure")
    return sorted(candidates, key=lambda row: digest(f"selective-radar/1/{row[0]}".encode()))


def check_pcm(path, duration, sha256):
    from app.detectors.audio.decode import decode_audio

    # The whole original is decoded first to retain and verify its entire time mapping.
    decoded = decode_audio(path, source_sha256=sha256, max_seconds=13)
    mapping = decoded.mapping
    expected = {"source_channels": 1, "source_sample_rate": 16000, "source_codec": "flac", "source_bit_depth": 16}
    study.need(all(mapping.get(key) == value for key, value in expected.items())
               and mapping["timeline"] == [{"start_sample": 0, "end_sample": len(decoded.pcm), "source_pts": 0}]
               and abs(Decimal(duration) * 16000 - len(decoded.pcm)) <= 3 and len(decoded.pcm) >= SAMPLES,
               "Original is outside the continuous publisher sample profile")
    return decoded.pcm[:SAMPLES]


def freeze(source, run, operating):
    source_files = source_bindings(source)
    excluded_ids, excluded_flac, excluded_sources = prior_inputs(source)
    locked = study.read(operating / "freeze.json")
    replay = study.read(operating / "receipt.json")
    study.need(locked["schema"] == replay["schema"] == study.SCHEMA
               and replay["freeze_sha256"] == study.sha(operating / "freeze.json")
               and replay["points"] == study.read(operating / "points.json"), "Frozen local operating points required")
    from app.detectors.audio import calibration, combined
    from app.detectors.audio.combined_adapter import BUNDLE_SHA256
    from scripts.prepare_audio_specialists_v15 import ASSETS

    run.mkdir(mode=0o700)
    inputs = run / "clips"
    inputs.mkdir()
    seen = set()
    roster = []
    for split in ("development", "evaluation"):
        counts = Counter()
        for ident, label, duration, checksum, data in extract_candidates(source, split, excluded_ids, excluded_flac):
            if counts[label] == PER_LABEL:
                continue
            path = inputs / f"{ident}.flac"
            path.write_bytes(data)
            try:
                pcm = check_pcm(path, duration, checksum)
            except ValueError:
                path.unlink()
                continue
            fingerprint = radar.pcm_hash(pcm)
            if fingerprint in seen:
                path.unlink()
                continue
            seen.add(fingerprint)
            roster.append({"id": ident, "split": split, "label": label, "duration": duration, "original_sha256": checksum,
                           "start_sample": 0, "end_sample": SAMPLES, "pcm_sha256": fingerprint})
            counts[label] += 1
            if counts.total() == 2 * PER_LABEL:
                break
        study.need(counts == {label: PER_LABEL for label in radar.ORIGINS}, "Publisher prefixes lack enough eligible sources")
    study.write(run / "roster.json", roster)
    lock = {"schema": SCHEMA, "method": METHOD, "runner_sha256": study.sha(Path(__file__)), "source_files": source_files,
            "excluded_intakes": excluded_sources, "roster_sha256": study.sha(run / "roster.json"),
            "points_sha256": study.sha(operating / "points.json"), "operating_freeze_sha256": study.sha(operating / "freeze.json"),
            "points": {name: replay["points"][name]["all"]["point"] for name in ("raw", "global")},
            "weights": ASSETS, "encoder": combined.ASSETS, "bundle": BUNDLE_SHA256, "incumbent_calibration": calibration.SHA256}
    study.write(run / "freeze.json", lock)
    print(f"Frozen {len(roster)} previously unused source recordings and their four-second windows.", flush=True)


def verify_native(run, roster):
    for row in roster:
        path = run / "clips" / f"{row['id']}.flac"
        study.need(study.sha(path) == row["original_sha256"], "Original carrier changed")
        samples, clock = native.decode(path, 13 * 16000)
        position = 0
        for _, tick, count in clock:
            study.need(tick == position, "Native original clock is discontinuous")
            position += count
        study.need(position == len(samples) and abs(Decimal(row["duration"]) * 16000 - position) <= 3
                   and native.pcm_sha(samples[:SAMPLES]) == row["pcm_sha256"], "Native source interval differs")


def summarize(rows):
    report = {}
    for split in ("development", "evaluation", "all"):
        chosen = [r for r in rows if split == "all" or r["split"] == split]
        report[split] = {}
        for arm in METHOD["arms"]:
            report[split][arm] = {label: dict(sorted(Counter(row["directions"][arm] for row in chosen if row["label"] == label).items()))
                                  for label in radar.ORIGINS}
    gained = {arm: sum(r["label"] == "spoof" and r["directions"][arm] == "ai" and r["directions"]["current"] != "ai" for r in rows)
              for arm in ("raw", "global")}
    report["advances"] = {arm: gained[arm] >= 4 and all(any(r["split"] == split and r["label"] == "spoof"
                          and r["directions"][arm] == "ai" and r["directions"]["current"] != "ai" for r in rows)
                          for split in ("development", "evaluation"))
                          and sum(r["label"] == "bonafide" and r["directions"][arm] == "ai" and r["directions"]["current"] != "ai" for r in rows) <= 1
                          for arm in gained}
    return report


def score(run, operating):
    from app.core.config import settings
    from app.detectors.audio import calibration, combined
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    lock, roster = (study.read(run / name) for name in ("freeze.json", "roster.json"))
    study.need(lock["schema"] == SCHEMA and lock["method"] == METHOD and lock["runner_sha256"] == study.sha(Path(__file__))
               and lock["roster_sha256"] == study.sha(run / "roster.json") and len(roster) == 4 * PER_LABEL
               and lock["points_sha256"] == study.sha(operating / "points.json")
               and lock["operating_freeze_sha256"] == study.sha(operating / "freeze.json")
               and lock["incumbent_calibration"] == calibration.SHA256 and settings.audio_detector_mode == "candidate", "Frozen study or current calibration differs")
    verify_native(run, roster)
    voice = load_voice({"weights": str(WEIGHTS), "assets": lock["weights"]})
    activity = SpeechActivity(WEIGHTS / "silero_vad.jit")
    bundle = combined.load_calibration(BUNDLE, lock["bundle"])
    encoder = None
    points = lock["points"]
    cases = []
    with (run / "scores.jsonl").open("x") as log:
        for index, row in enumerate(roster, 1):
            pcm = check_pcm(run / "clips" / f"{row['id']}.flac", row["duration"], row["original_sha256"])
            study.need(radar.pcm_hash(pcm) == row["pcm_sha256"], "Measured source samples differ")
            baseline = radar.adapter_result(pcm, voice, activity)
            raw, direction = baseline["raw"], baseline["decision"]
            estimate = None
            if direction == "inconclusive" and raw is not None:
                study.need(baseline["windows"] == [[0, SAMPLES]], "Current result used another sample extent")
                if encoder is None:
                    study.need(lock["encoder"] == combined.ASSETS, "Encoder declaration changed")
                    encoder = combined.load_encoder(ENCODER)
                estimate = bundle.predict(raw, combined.extract(pcm, encoder))
            scores = {"raw": raw if direction == "inconclusive" else None, "global": estimate}
            directions = {"current": direction} | {arm: "ai" if value is not None and value > points[arm] else direction for arm, value in scores.items()}
            record = {**row, "scores": scores, "directions": directions}
            cases.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if index % 8 == 0:
                print(f"Measured {index}/{len(roster)} frozen inputs.", flush=True)
    receipt = {"schema": SCHEMA, "freeze_sha256": study.sha(run / "freeze.json"), "method": METHOD, "points": points,
               "sources_sha256": lock["roster_sha256"], "independent_native_inputs": len(cases),
               "cases": [{key: row[key] for key in ("split", "label", "start_sample", "end_sample", "pcm_sha256", "scores", "directions")} for row in cases],
               "results": summarize(cases)}
    study.write(run / "receipt.json", receipt)
    print(json.dumps(receipt["results"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score", "audit"))
    parser.add_argument("source", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("operating", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.source, args.run, args.operating)
    elif args.mode == "score":
        score(args.run, args.operating)
    else:
        verify_native(args.run, study.read(args.run / "roster.json"))
        print("Independent native source clocks match.")
