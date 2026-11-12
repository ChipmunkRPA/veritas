# SPDX-License-Identifier: Apache-2.0
"""Confirm the fixed long-recording speech check on RADAR's other archive split."""
from __future__ import annotations

import argparse
import re
import tarfile
from collections import Counter
from decimal import Decimal
from pathlib import Path

from scripts import audio_radar_transfer as first
from scripts import review_audio_radar_transfer as native

common, require = first.common, first.require
SOURCES = {
    "dev-prefix.gz": "6e85fce6e8ee74784769f0bd76f77c379af44a1dab9848d61fc791a9ada05266",
    "dev-labels.txt": "33dbf6ef66df46c50716c419fab0c1c73edc160ca06996dbaf719bebe47204a3",
    "dev-languages.txt": "f28f2ebcad8ce58f0042f3f1fd2fa0c385284a9ca008e776f2d9ac9ae4dccd72",
}
MEMBER = re.compile(r"RADAR2026-dev/flac/(RADAR2026-DEV\d{6})\.flac")
PER_CLASS = 20
REQUIRED = {"spoof_supported": 10, "bonafide_false_max": 2}


def select(source):
    require({name: common.sha(source / name) for name in SOURCES} == SOURCES, "The official development inputs differ")
    labels, languages = (native.table(source / name) for name in ("dev-labels.txt", "dev-languages.txt"))
    require(len(labels) == len(languages) == 44_034 and labels.keys() == languages.keys(), "Publisher metadata differs")
    candidates, seen = [], set()
    try:
        with tarfile.open(source / "dev-prefix.gz", "r|gz") as archive:
            for member in archive:
                match = MEMBER.fullmatch(member.name)
                if match is None or not member.isfile():
                    continue
                name = match[1]
                require(name in labels and name not in seen and 0 < member.size < 2 * 1024**2, "Invalid archive member")
                seen.add(name)
                duration, label, *segments = labels[name]
                if languages[name] != ["EN"] or not 6 <= Decimal(duration) <= 14 or segments != [f"0.0000-{duration}-{label}"]:
                    continue
                with archive.extractfile(member) as payload:
                    raw = payload.read()
                require(len(raw) == member.size and raw[:4] == b"fLaC", "FLAC member differs")
                candidates.append(({"id": name, "duration": duration, "label": label, "flac_sha256": common.digest(raw)}, raw))
    except (EOFError, tarfile.ReadError) as error:
        require("end of data" in str(error) or isinstance(error, EOFError), "Unexpected archive read error")
    candidates.sort(key=lambda pair: common.digest(f"radar-long-confirmation/1/{pair[0]['id']}".encode()))
    selected = []
    for label in first.ORIGINS:
        rows = [pair for pair in candidates if pair[0]["label"] == label][:PER_CLASS]
        require(len(rows) == PER_CLASS, f"Missing {label} confirmation rows")
        selected.extend(rows)
    return selected


def lock(prior, intake):
    return {"schema": "chipmunk.audio-radar-long-confirmation/1", "runner": common.sha(Path(__file__)),
            "prior_freeze": common.commitment(prior["freeze"]), "source_files": SOURCES,
            "roster": common.commitment(intake), "native_profile": "16 kHz mono signed 16-bit FLAC, 6–14 s",
            "rule": prior["freeze"]["candidate"], "advance": REQUIRED}


def freeze(source, run, prior):
    chosen = select(source)
    run.mkdir(mode=0o700)
    for row, payload in chosen:
        (run / f"{row['id']}.flac").write_bytes(payload)
    prior_hashes = {row["pcm_sha256"] for row in prior["rows"]}
    intake = []
    for row, _ in chosen:
        samples, ticks = native.decode(run / f"{row['id']}.flac", 15 * first.RATE)
        digest, position = native.pcm_sha(samples), 0
        require(digest not in prior_hashes and abs(Decimal(row["duration"]) * first.RATE - len(samples)) <= 3,
                "Confirmation sample overlaps the initial trial or differs from published duration")
        prior_hashes.add(digest)
        for _, tick, count in ticks:
            require(tick == position, "Non-continuous decoded clock")
            position += count
        intake.append({**row, "pcm_sha256": digest, "samples": len(samples)})
    common.save(run / "intake.json", intake)
    common.save(run / "freeze.json", lock(prior, intake))


def summarize(rows, intake):
    output = {label: Counter(files=0, native_inconclusive=0, native_unassessed=0, two_assessed=0, one_ai=0, corroborated_ai=0)
              for label in first.ORIGINS}
    for row, source in zip(rows, intake, strict=True):
        require(all(row[key] == value for key, value in source.items()), "Confirmation row changed")
        require(row["full"]["decision"] in {"inconclusive", "unassessed"} and not row["full"]["scope"],
                "Long file received whole-file confidence")
        require([portion["interval"] for portion in row["portions"]] == first.spans(row["samples"]),
                "Confirmation ranges changed")
        directions = Counter(portion["response"]["decision"] for portion in row["portions"])
        require(set(directions) <= first.DIRECTIONS, "Unknown decision")
        counts = output[row["label"]]
        counts["files"] += 1
        counts[f"native_{row['full']['decision']}"] += 1
        counts["two_assessed"] += directions.total() - directions["unassessed"] >= 2
        counts["one_ai"] += directions["ai"] >= 1
        counts["corroborated_ai"] += directions["ai"] >= 2 and not directions["original"]
    require(all(row["files"] == PER_CLASS for row in output.values()), "Incomplete confirmation")
    passed = (output["spoof"]["corroborated_ai"] >= REQUIRED["spoof_supported"]
              and output["bonafide"]["corroborated_ai"] <= REQUIRED["bonafide_false_max"])
    return {"counts": output, "passes": passed}


def score(run, weights, prior):
    from app.core.config import settings
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import SpeechActivity
    from scripts.run_audio_specialists_v15 import load_voice

    intake, frozen = (common.read(run / name) for name in ("intake.json", "freeze.json"))
    require(frozen == lock(prior, intake) and settings.audio_detector_mode == "candidate", "Confirmation freeze differs")
    assets = {name: pin for name, pin in prior["freeze"]["voice_assets"].items() if name not in common.SOUND_ASSETS}
    voice = load_voice({"weights": str(weights), "assets": assets})
    activity = SpeechActivity(weights / "silero_vad.jit")
    rows = []
    for index, row in enumerate(intake, 1):
        path = run / f"{row['id']}.flac"
        require(common.sha(path) == row["flac_sha256"], "Frozen FLAC differs")
        decoded = decode_audio(path, source_sha256=row["flac_sha256"], max_seconds=15)
        mapping, pcm = decoded.mapping, decoded.pcm
        require(tuple(mapping[key] for key in ("source_codec", "source_bit_depth", "source_channels", "source_sample_rate",
                                               "source_container")) == ("flac", 16, 1, first.RATE, "flac")
                and mapping["timeline"] == [{"start_sample": 0, "end_sample": row["samples"], "source_pts": 0}]
                and mapping["pcm_sha256"] == row["pcm_sha256"], "Confirmation input is outside the frozen native profile")
        portions = []
        for start, end in first.spans(len(pcm)):
            chunk = pcm[start:end]
            portions.append({"interval": [start, end], "pcm_sha256": first.pcm_hash(chunk),
                             "response": first.adapter_result(chunk, voice, activity)})
        rows.append({**row, "full": first.adapter_result(pcm, voice, activity), "portions": portions})
        if index % 8 == 0:
            print(f"Measured {index}/{len(intake)} separately held long sources.", flush=True)
    result = {"freeze": frozen, "intake": intake, "rows": rows, "rows_sha256": common.commitment(rows),
              "summary": summarize(rows, intake)}
    with (run / "result.json").open("xb") as output:
        output.write(common.encoded(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--weights", type=Path, default=common.WEIGHTS)
    args = parser.parse_args()
    prior = common.read(args.prior)
    require(prior["summary"]["follow_up_criterion_met"], "The first trial did not advance")
    if args.phase == "freeze":
        if args.source is None:
            parser.error("freeze requires --source")
        freeze(args.source, args.path, prior)
    elif args.phase == "score":
        score(args.path, args.weights, prior)
    else:
        result = common.read(args.path)
        require(result["freeze"] == lock(prior, result["intake"]) and result["rows_sha256"] == common.commitment(result["rows"])
                and result["summary"] == summarize(result["rows"], result["intake"]), "Confirmation receipt differs")
        print(result["summary"])


if __name__ == "__main__":
    main()
