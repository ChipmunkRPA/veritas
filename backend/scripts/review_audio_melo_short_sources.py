"""Recount aligned PhonemeDF MeloTTS short-edit source candidates without inference."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts import audio_odss_matched_room as source
from scripts import audio_phoneme_short_edit_intake as intake
from scripts import audio_phonemedf_matched_transfer as archive

SCHEMA = "chipmunk.audio-melo-short-sources/1"
ROOT = Path(__file__).resolve().parents[2]
HISTORY = {
    "v9": ("audio-v9/source-plan.json", "f3d7a7f2e20a9d2aa9092c371f5af4d2a05ed9172e4652677049a8286f61605f"),
    "v172": ("audio-short-edit-cases-v172/development/private-plan.json",
             "803cf0e7adee70ab68d7f7b9501538e744b1ed92c2a9d56766567aa5cd504eaa"),
}
ORIGINS = ("real", "MeloTTS")
PREFIX_BYTES = 320 * 1024**2
MELO_ARCHIVE = 7_703_256_782
GENUINE_ARCHIVE = 9_420_947_621
PARENT = ROOT / "docs/evaluation-panels/audio-phoneme-short-edit-intake-2026-11-13.json"
PARENT_SHA = "ba7b61a274e89872e1049b9502ae55acb65e81994e0c8c62e6655f01ab8b5efe"


def parts(folder, total, old=None):
    paths = []
    for offset in range(0, PREFIX_BYTES, archive.CHUNK):
        path = (old if old is not None and offset < intake.PREVIOUS_PREFIX else folder) / f"{offset}.range"
        header = path.with_suffix(".range.headers")
        if not header.is_file():
            header = path.with_suffix(".headers")
        end = offset + archive.CHUNK - 1
        source.require(path.stat().st_size == archive.CHUNK and
                       f"content-range: bytes {offset}-{end}/{total}" in header.read_text().casefold().splitlines(),
                       "Publisher WAV range does not match the selected prefix")
        paths.append(path)
    return paths


def wavs(paths, pattern):
    import numpy as np

    payload = {}
    for path, raw in archive.archive_members(paths):
        if match := re.fullmatch(pattern, path):
            key = match[1]
            source.require(key not in payload, "Duplicate source")
            pcm = archive.pcm_bytes(raw)
            payload[key] = (source.digest(raw), source.digest(pcm), np.frombuffer(pcm, dtype="<i2"))
    return payload


def known_speakers(history):
    files = {}
    for role, (name, digest) in HISTORY.items():
        path = history / name
        source.require(source.file_digest(path) == digest, f"Saved source plan differs: {role}")
        files[role] = source.read(path)
    rosters = {"v9": files["v9"]["sources"], "v172": files["v172"]["source_manifest"]["foregrounds"]}
    return {role: {voice.split(":")[-1] for row in rows
                   if (voice := row.get("canonical_speaker") or row.get("speaker") or row.get("voice") or "")
                   .startswith("librispeech:")} for role, rows in rosters.items()}


def census(melo, cache, old, history):
    source.require(source.file_digest(PARENT) == PARENT_SHA, "Prior real and alignment commitments differ")
    parent = source.read(PARENT)
    genuine = parts(cache / "real", GENUINE_ARCHIVE, old / "real")
    source.require([source.file_digest(path) for path in genuine] == parent["bounded_wav_ranges"]["real"],
                   "Genuine archive prefix differs")
    synthetic = parts(melo, MELO_ARCHIVE)
    recordings = {"real": wavs(genuine, r"real/wav/(\d+-\d+-\d+)\.wav"),
                  "MeloTTS": wavs(synthetic, r"MeloTTS/(\d+-\d+-\d+)\.wav")}
    keys = sorted(set.intersection(*(set(bank) for bank in recordings.values())))
    alignments = defaultdict(dict)
    grids = intake.archive(cache, "textgrids", intake.TEXTGRID_BYTES, intake.TEXTGRID_BYTES,
                           complete_md5=intake.TEXTGRID_MD5)
    for path, raw in archive.archive_members(grids):
        match = re.fullmatch(r"textgrids/(real|MeloTTS)/(\d+-\d+-\d+)\.TextGrid", path)
        if match and match[2] in recordings["real"] and match[2] in recordings["MeloTTS"]:
            origin, key = match.groups()
            if spans := intake.words(raw, len(recordings[origin][key][2])):
                alignments[key][origin] = (spans, source.digest(raw))
    historical = known_speakers(history)
    counts = defaultdict(lambda: {"paired": 0, "matching_words": 0, "eligible": 0})
    for key in keys:
        speaker = key.split("-")[0]
        row = counts[speaker]
        row["paired"] += 1
        tiers = alignments[key]
        if set(tiers) != set(ORIGINS):
            continue
        if [word for _, _, word in tiers["real"][0]] != [word for _, _, word in tiers["MeloTTS"][0]]:
            continue
        row["matching_words"] += 1
        row["eligible"] += bool(intake.possibilities(tiers, recordings, key))
    readers = set(counts)
    rows = [{"reader_sha256": source.digest(f"{SCHEMA}:{speaker}".encode()),
             "recorded_in_v9": speaker in historical["v9"], "recorded_in_v172_development": speaker in historical["v172"],
             **counts[speaker]} for speaker in sorted(readers)]
    return {"schema": SCHEMA, "publisher_record": 19522979, "melo_archive_bytes": MELO_ARCHIVE,
            "melo_compressed_prefix_bytes": PREFIX_BYTES, "textgrid_archive_md5": intake.TEXTGRID_MD5,
            "melo_range_sha256": [source.file_digest(path) for path in synthetic], "previous_intake_sha256": PARENT_SHA,
            "local_ancestry_plan_sha256": {role: pin for role, (_, pin) in HISTORY.items()},
            "melo_native_wavs": len(recordings["MeloTTS"]), "paired_native_wavs": len(keys),
            "full_matching_word_tiers": sum(row["matching_words"] for row in rows),
            "eligible_short_edit_pairs": sum(row["eligible"] for row in rows), "paired_reader_groups": len(rows),
            "eligible_readers_outside_both_saved_plans": sum(row["eligible"] > 0 and not (
                row["recorded_in_v9"] or row["recorded_in_v172_development"]) for row in rows),
            "melo_builtin_voice_id_per_file": False, "model_forwards": 0, "groups": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("melo", "cache", "old-cache"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=ROOT / "backend/data/independent-evaluation")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    receipt = census(args.melo, args.cache, args.old_cache, args.history)
    if args.compare:
        source.require(receipt == source.read(args.compare), "Native source census differs")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
