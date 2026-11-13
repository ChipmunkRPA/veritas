"""Check the native WAV clocks and matched cases in the public SINE preview."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path

REVISION = "81be5d5e6a7d5d0f9bbbb613abed7b700f43d793"
PUBLISHER = "https://huggingface.co/datasets/PeacefulData/SINE"
PINS = {
    "sine-preview-audio.json": "b30b8eaa3f9263b40fee70d10ecbb1aaa504f9dad7bc03b2ce2a2c9d8b3598f5",
    "sine-preview-metadata.csv": "a35896baf6c0a0625c51b40fc0b5325706f3ee8989aa6e4fe23e73e545be10db",
}
RATE = 24_000
ROLE = {"real": "real", "edit": "infill", "cut_paste": "cut_paste"}
CASE = re.compile(r"dev_(real|edit|cut_paste)_medium-(100-emerald_city_librivox_64kb_mp3-emeraldcity_(\d{2})_baum_64kb_\d+)")
INTERVAL = re.compile(r"(\d+\.\d{2})-(\d+\.\d{2})-([TF])")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clock(label):
    return int(label.replace(".", "")) * (RATE // 100)


def label_clock(timestamp, samples, role):
    intervals = []
    last = 0
    for token in timestamp.split("/"):
        match = INTERVAL.fullmatch(token)
        need(match is not None, "Invalid native interval")
        start, end = map(clock, match.groups()[:2])
        need(start == last and start < end, "Gapped or reversed native intervals")
        intervals.append([start, end, match[3]])
        last = end
    kinds = [kind for _, _, kind in intervals]
    need(kinds == (["T"] if role == "real" else ["T", "F", "T"]), "Unexpected interval roles")
    need(abs(samples - last) <= RATE // 200, "Final timestamp differs by more than half a centisecond")
    return intervals, samples - last


def audit(directory):
    for filename, expected in PINS.items():
        need(digest((directory / filename).read_bytes()) == expected, "Publisher metadata changed")
    entries = json.loads((directory / "sine-preview-audio.json").read_bytes())
    files = {item["path"]: (item["size"], item["lfs"]["oid"]) for item in entries}
    need(len(entries) == len(files) == 30, "Missing publisher files")
    groups = defaultdict(dict)
    chapters = Counter()
    with (directory / "sine-preview-metadata.csv").open(newline="", encoding="utf-8") as table:
        for row in csv.DictReader(table):
            filename = row["filename"]
            case = CASE.fullmatch(filename)
            need(case is not None, "Unrecognized case ID")
            prefix, identifier, chapter = case.groups()
            role = ROLE[prefix]
            need(row["category"] == row["manipulation_type"] == prefix, "Publisher role mismatch")
            need(row["label"] == ("1" if role == "real" else "0"), "Publisher binary label mismatch")
            path = row["audio"]
            need(path == f"preview/audio/{filename}.wav" and path in files, "Publisher filename mismatch")
            size, sha = files.pop(path)
            body = (directory / f"{filename}.wav").read_bytes()
            need(len(body) == size and digest(body) == sha, "WAV differs from the pinned publisher LFS object")
            with wave.open(io.BytesIO(body)) as stream:
                fmt = stream.getframerate(), stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()
                need(fmt == (RATE, 1, 2, "NONE"), "Unexpected native WAV format")
                samples = stream.getnframes()
                pcm = stream.readframes(samples)
            need(2 * samples == len(pcm) and 6 * RATE < samples < 9 * RATE, "Invalid native PCM length")
            intervals, residual = label_clock(row["timestamp"], samples, role)
            need(role not in groups[identifier], "Duplicate case role")
            groups[identifier][role] = {
                "chapter": chapter,
                "filename": f"{filename}.wav",
                "wav_sha256": sha,
                "wav_bytes": size,
                "pcm_sha256": digest(pcm),
                "samples": samples,
                "publisher_intervals": intervals,
                "unrounded_end_minus_published_end_samples": residual,
            }
            if role == "real":
                chapters[chapter] += 1
    need(not files and len(groups) == 10, "Missing preview cases")
    edit_lengths = []
    for group in groups.values():
        need(set(group) == set(ROLE.values()), "Missing matched variant")
        infill = group["infill"]
        paste = group["cut_paste"]
        need(infill["samples"] == paste["samples"]
             and infill["publisher_intervals"] == paste["publisher_intervals"], "Edited variants disagree on truth")
        start, end, _ = infill["publisher_intervals"][1]
        need(end < infill["samples"], "Edit falls outside native samples")
        edit_lengths.append(end - start)
    rows = [row for cases in groups.values() for row in cases.values()]
    end_residuals = [row["unrounded_end_minus_published_end_samples"] for row in rows]
    need(len({row["pcm_sha256"] for row in rows}) == len(rows), "Duplicate waveforms in the matched cases")
    return {
        "schema": "chipmunk.audio-sine-preview/1",
        "publisher": PUBLISHER,
        "revision": REVISION,
        "publisher_inputs_sha256": PINS,
        "dataset_license": "Apache-2.0",
        "sample_rate": RATE,
        "channels": 1,
        "sample_width_bytes": 2,
        "wav_total_bytes": sum(row["wav_bytes"] for row in rows),
        "matched_triplets": len(groups),
        "source_reader_ids": ["100"],
        "triplets_by_chapter": dict(chapters),
        "edit_length_samples_min_max": [min(edit_lengths), max(edit_lengths)],
        "native_end_residual_samples_min_max": [min(end_residuals), max(end_residuals)],
        "cases": dict(groups),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write", type=Path)
    modes.add_argument("--compare", type=Path)
    args = parser.parse_args()
    report = audit(args.directory)
    if args.write:
        with args.write.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    else:
        need(json.loads(args.compare.read_bytes()) == report, "Native intake receipt differs")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, sort_keys=True))


if __name__ == "__main__":
    main()
