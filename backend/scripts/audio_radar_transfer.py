# SPDX-License-Identifier: Apache-2.0
"""Measure fixed speech decisions on a bounded, published RADAR evaluation cohort."""
from __future__ import annotations

import argparse
import re
import shutil
import tarfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

from scripts import audio_vocal_event_mixtures as common

SOURCE_FILES = {
    "archive-prefix.gz": "d50b195fdccb37443c7a5e57169941c1646333afda57f97e72bca18c019c837c",
    "labels.txt": "dc51de1f7a6fcfb4e26ffd7fc458b5e0e290e6541bf6d2f0710250f67d29d3ba",
    "languages.txt": "ff11556268e4b404ac1fba048c7ef7cf9a0baaeb1a3889c5572ff6390d0e6735",
    "LICENSE.txt": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
    "traffic.ogg": "9702be26fec457ac8c54df61fbe62f120189119b5734377cda9e5ae442566efd",
    "cafe.ogg": "6360791fdcee1f64638f0fc9abd6d2c7a1aa75fd3d2b80033170a02d025099dc",
}
PUBLISHER = "https://doi.org/10.5281/zenodo.22110815"
MEMBER = re.compile(r"RADAR2026-eval/flac/(RADAR2026-EVAL\d{6})\.flac")
RATE = 16_000
SPAN = 3 * RATE
SIZE = 16
RANGES = {"short": (Decimal("2.5"), Decimal(4)), "long": (Decimal(6), Decimal(14))}
ORIGINS = ("bonafide", "spoof")
DIRECTIONS = {"ai", "original", "inconclusive", "unassessed"}
RULE = {"minimum_long_spoof_flags": 8, "maximum_long_bonafide_flags": 2}
SCENES = {"traffic": (2, 6, 10, 14), "cafe": (22, 96, 174, 262)}
MIX_SNR_DB = -5
require = common.require


def lock(intake):
    return {"schema": "chipmunk.audio-radar-transfer/1", "publisher": PUBLISHER, "source_files": SOURCE_FILES,
            "compressed_bytes": 32 * 1024**2, "intake_sha256": common.commitment(intake),
            "runner_sha256": common.sha(Path(__file__)), "voice_assets": common.assets(),
            "selection": "EN, one whole-file label, official 2.5–4 s or 6–14 s, 16 per class by salted SHA-256",
            "windows": "Disjoint 3 s at start/end, plus center for >=9 s, on exact decoded mono clock",
            "candidate": "At least two AI windows and zero original windows; otherwise no local corroboration",
            "noise": {"offsets_s": {key: list(values) for key, values in SCENES.items()}, "target_db": MIX_SNR_DB,
                      "selection": "Equal class assignment by sealed within-class rank modulo four",
                      "delivery": "Same peak-safe gain for the two mixtures and unoverlaid control, s16 PCM"},
            "minimum_usefulness": RULE}


def acquire(source):
    require({name: common.sha(source / name) for name in SOURCE_FILES} == SOURCE_FILES,
            "The official metadata or bounded archive prefix changed")
    labels = {name: rest for name, *rest in (line.split() for line in (source / "labels.txt").read_text().splitlines())}
    languages = dict(line.split() for line in (source / "languages.txt").read_text().splitlines())
    require(len(labels) == len(languages) == 102_725 and labels.keys() == languages.keys(), "Publisher tables disagree")
    records, seen = {}, set()
    try:
        with tarfile.open(source / "archive-prefix.gz", "r|gz") as archive:
            for member in archive:
                matched = MEMBER.fullmatch(member.name)
                if not matched or not member.isfile():
                    continue
                name = matched[1]
                require(name in labels and name not in seen, "Missing or repeated publisher identity")
                seen.add(name)
                duration, label, *segments = labels[name]
                require(0 < member.size < 2 * 1024**2, "Unexpected FLAC size")
                with archive.extractfile(member) as stream:
                    payload = stream.read()
                require(len(payload) == member.size and payload[:4] == b"fLaC", "Invalid FLAC member")
                if languages[name] != "EN" or segments != [f"0.0000-{duration}-{label}"]:
                    continue
                for stratum, (low, high) in RANGES.items():
                    if low <= Decimal(duration) <= high:
                        row = {"id": name, "duration": duration, "label": label, "stratum": stratum,
                               "flac_sha256": common.digest(payload), "flac_bytes": len(payload)}
                        records[name] = (row, payload)
    except (EOFError, tarfile.ReadError) as error:
        require("end of data" in str(error) or isinstance(error, EOFError), "Unexpected archive read error")
    chosen = []
    for stratum in RANGES:
        for label in ORIGINS:
            candidates = [(row, payload) for row, payload in records.values()
                          if (row["stratum"], row["label"]) == (stratum, label)]
            candidates.sort(key=lambda pair: common.digest(f"radar-transfer/1/{pair[0]['id']}".encode()))
            require(len(candidates) >= SIZE, f"The bounded publisher prefix lacks {stratum} {label} files")
            chosen.extend(candidates[:SIZE])
    return chosen


def freeze(source, run):
    selected = acquire(source)
    run.mkdir(mode=0o700)
    (run / "clips").mkdir(mode=0o700)
    for row, payload in selected:
        (run / "clips" / f"{row['id']}.flac").write_bytes(payload)
    for scene in SCENES:
        shutil.copyfile(source / f"{scene}.ogg", run / f"{scene}.ogg")
    intake = [row for row, _ in selected]
    common.save(run / "intake.json", intake)
    common.save(run / "freeze.json", lock(intake))


def spans(samples):
    if samples < 2 * SPAN:
        return []
    last = samples - SPAN
    starts = [0, last] if samples < 3 * SPAN else [0, last // 2, last]
    return [[start, start + SPAN] for start in starts]


def adapter_result(pcm, voice, activity, *, include_status=False):
    from types import SimpleNamespace

    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack
    from app.engine.detector import Profile

    objects = {PellaVoiceDetector.MODEL_ID: voice, "silero.speech_activity": activity}
    ctx = SimpleNamespace(profile=Profile.FULL, outputs=SimpleNamespace(audio=AudioTrack("analysed", pcm, samples=len(pcm))),
                          lease=SimpleNamespace(get=lambda key: SimpleNamespace(obj=objects[key], revision="pinned")),
                          exhibits=SimpleNamespace(series=lambda *args, **kwargs: "offline"),
                          deadline=SimpleNamespace(check=lambda *_: None))
    result = PellaVoiceDetector().run(ctx)
    if result.status == "not_applicable":
        record = {"decision": "unassessed", "scope": False, "windows": [], "raw": None}
    else:
        require(result.status == "ok", "Unexpected voice adapter result")
        screen = result.details["audio_screening"]
        record = {"decision": screen["decision"], "scope": screen["scope_supported"], "raw": result.aggregate.raw,
                  "windows": [[int(unit.quality["start_sample"]), int(unit.quality["end_sample"])] for unit in result.unit_scores]}
    if include_status:
        record["status"] = result.status
    return record


def pcm_hash(pcm):
    return common.digest(pcm.astype("<f4").tobytes())


def acoustic_views(pcm, backgrounds, rank):
    import numpy as np

    reference = pcm.astype(np.float64)
    signal_level = np.sqrt(np.mean(reference**2))
    require(signal_level > 0, "The benchmark foreground is silent")
    tracks = {"gain_control": reference}
    for scene, offsets in SCENES.items():
        start = offsets[rank % len(offsets)] * RATE
        clip = backgrounds[scene][start:start + len(pcm)].astype(np.float64)
        require(len(clip) == len(pcm), "Source background excerpt is too short")
        level = np.sqrt(np.mean(clip**2))
        require(level > 0, "Source background excerpt is silent")
        tracks[scene] = reference + clip * (signal_level / level) * 10**(-MIX_SNR_DB / 20)
    gain = min(1.0, .98 / max(float(np.max(np.abs(track))) for track in tracks.values()))
    return gain, {key: np.rint(values * gain * 32768).astype("<i2").astype(np.float32) / 32768
                  for key, values in tracks.items()}


def score(run, weights):
    from app.core.config import settings
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    intake, frozen = (common.read(run / name) for name in ("intake.json", "freeze.json"))
    require(frozen == lock(intake) and settings.audio_detector_mode == "candidate", "Serving scope or freeze changed")
    backgrounds = {}
    for scene, offsets in SCENES.items():
        key = f"{scene}.ogg"
        require(common.sha(run / key) == SOURCE_FILES[key], "Published background recording changed")
        backgrounds[scene] = decode_audio(run / key, source_sha256=SOURCE_FILES[key], max_seconds=max(offsets) + 5).pcm
    assets = {key: value for key, value in frozen["voice_assets"].items() if key not in common.SOUND_ASSETS}
    require(all(common.sha(weights / key) == value for key, value in assets.items()), "Serving weights changed")
    voice = load_voice({"weights": str(weights), "assets": assets})
    activity = SpeechActivity(weights / "silero_vad.jit")
    rows = []
    for index, row in enumerate(intake, 1):
        path = run / "clips" / f"{row['id']}.flac"
        require(common.sha(path) == row["flac_sha256"], "FLAC does not match frozen bytes")
        decoded = decode_audio(path, source_sha256=row["flac_sha256"], max_seconds=15)
        samples = len(decoded.pcm)
        require(abs(Decimal(row["duration"]) * RATE - samples) <= 3
                and decoded.mapping["timeline_time_base"] == "1/16000"
                and decoded.mapping["timeline"] == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}],
                "Publisher duration and continuous source PCM differ")
        full = adapter_result(decoded.pcm, voice, activity)
        portions, mixtures, gain = [], {}, None
        if row["stratum"] == "long":
            for start, end in spans(samples):
                pcm = decoded.pcm[start:end]
                portions.append({"interval": [start, end], "pcm_sha256": pcm_hash(pcm),
                                 "response": adapter_result(pcm, voice, activity)})
        else:
            gain, tracks = acoustic_views(decoded.pcm, backgrounds, (index - 1) % SIZE)
            for scene, pcm in tracks.items():
                digest = pcm_hash(pcm)
                result = full if digest == decoded.mapping["pcm_sha256"] else adapter_result(pcm, voice, activity)
                mixtures[scene] = {"pcm_sha256": digest, "response": result}
        rows.append({**row, "samples": samples, "pcm_sha256": decoded.mapping["pcm_sha256"], "full": full,
                     "portions": portions, "mixtures": mixtures, "matched_gain": gain})
        if index % 8 == 0:
            print(f"Measured {index}/{len(intake)} fixed RADAR sources.", flush=True)
    result = {"freeze": frozen, "intake": intake, "rows": rows, "rows_sha256": common.commitment(rows),
              "summary": summarize(rows, intake)}
    with (run / "result.json").open("xb") as output:
        output.write(common.encoded(result))


def summarize(rows, intake):
    totals, noise = {}, {}
    for row, source in zip(rows, intake, strict=True):
        require(all(row[key] == value for key, value in source.items()), "A scored record changed its source identity")
        stratum, label = row["stratum"], row["label"]
        n = row["samples"]
        full, portions = row["full"], row["portions"]
        responses = [portion["response"] for portion in portions]
        require(full["decision"] in DIRECTIONS and all(response["decision"] in DIRECTIONS for response in responses)
                and [portion["interval"] for portion in portions] == (spans(n) if stratum == "long" else []),
                "A response or source range is invalid")
        if stratum == "long":
            require(not full["scope"] and full["decision"] in {"inconclusive", "unassessed"}, "Long file received whole-file confidence")
        stats = totals.setdefault(f"{stratum}/{label}", Counter({state: 0 for state in DIRECTIONS}))
        stats["files"] += 1
        stats[full["decision"]] += 1
        if stratum == "long":
            directions = Counter(response["decision"] for response in responses)
            stats["two_portions_assessed"] += sum(v for k, v in directions.items() if k != "unassessed") >= 2
            stats["any_ai_portion"] += directions["ai"] >= 1
            stats["corroborated_ai_portions"] += directions["ai"] >= 2 and not directions["original"]
            require(not row["mixtures"] and row["matched_gain"] is None, "Long file has unexpected mixes")
        else:
            require(set(row["mixtures"]) == {"gain_control", *SCENES} and 0 < row["matched_gain"] <= 1,
                    "Short file is missing a matched mix")
            for scene, item in row["mixtures"].items():
                response = item["response"]["decision"]
                require(response in DIRECTIONS, "Invalid response to a matched scene")
                noise.setdefault(f"{scene}/{label}", Counter({state: 0 for state in DIRECTIONS}))[response] += 1
    require(set(totals) == {f"{stratum}/{label}" for stratum in RANGES for label in ORIGINS}
            and all(row["files"] == SIZE for row in totals.values()), "Incomplete fixed scoring panel")
    follow_up = (totals["long/spoof"]["corroborated_ai_portions"] >= RULE["minimum_long_spoof_flags"]
                 and totals["long/bonafide"]["corroborated_ai_portions"] <= RULE["maximum_long_bonafide_flags"])
    return {"counts": totals, "matched_short_scenes": noise, "follow_up_criterion_met": follow_up}


def review(path):
    result = common.read(path)
    rows, intake = result["rows"], result["intake"]
    require(result["freeze"] == lock(intake) and result["rows_sha256"] == common.commitment(rows)
            and result["summary"] == summarize(rows, intake), "The receipt differs from the fixed replay")
    print(result["summary"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path, default=common.WEIGHTS)
    args = parser.parse_args()
    if args.phase == "freeze":
        if args.source is None:
            parser.error("freeze requires --source")
        freeze(args.source, args.path)
    elif args.phase == "score":
        score(args.path, args.weights)
    else:
        review(args.path)


if __name__ == "__main__":
    main()
