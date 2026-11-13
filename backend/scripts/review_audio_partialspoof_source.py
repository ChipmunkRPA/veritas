"""Audit score-blind speaker and timing metadata in the PartialSpoof v1.2 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

SCHEMA = "chipmunk.partialspoof-source-audit/1"
RELEASE = "https://zenodo.org/records/5766198"
PARTITIONS = {"train": "T", "dev": "D", "eval": "E"}
PINS = {
    "README_v1.2": ("4fec0742e2a606d1b1186067d79862d6", "5d0ea8337a8945dfa73ed709a6e308c9b5df30670d67e7edfb79606a5b735573"),
    "database_protocols.tar.gz": ("699d81f020e4b7fa8f33747010e1cba8", "00cdede56868570857b342e568f12172328dd68d08c8f8826c77d80075053715"),
    "database_vad.tar.gz": ("95e77e19a1bb4f2e79ed138fd35621ad", "2027efef64c2d3e27cab677454443137b00f8f607b23d774c805505f8b2eb118"),
}
PREFIX = {
    "train": (2027101056, "af7a439a0ae9f46b7d2f8e7d53dcd9664a77b2699b1b7cff06108f3024a7b9d9"),
    "dev": (2027972850, "e59e3bbfa1edc47b97a4c2bd32bd0e2f488f0e2445fb55c473b6605ea34bef2b"),
}
VAD_NAME = re.compile(r"database/vad/(train|dev|eval)/((?:CON|LA)_[TDE]_\d{7})\.vad")
UTTERANCE = re.compile(r"(CON|LA)_([TDE])_\d{7}")
SPEAKER = re.compile(r"LA_\d{4}")
ATTACK = re.compile(r"A(?:0[1-9]|1\d)")
TARGET = {
    "rule": "One contiguous spoof interval of 0.50–1.20 s within 4–9 s of annotation, with at least 0.40 s labeled bona fide speech on each side. Merge immediately adjacent equal-label intervals only. This census tests candidate duration, not a native media clock or sample selection.",
    "claims": "Only declared speaker groups and annotated origins are known for PartialSpoof rows. The CON code does not link to the donor recordings or attack system labels of ASVspoof2019.",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digests(path):
    with path.open("rb") as stream:
        first = hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
        stream.seek(0)
        second = hashlib.file_digest(stream, "sha256").hexdigest()
    return first, second


def cm_trials(archive, split, partition, *, partial):
    parent = "PartialSpoof" if partial else "ASVspoof2019"
    stem = f"database/protocols/{parent}_LA_cm_protocols/{parent}.LA.cm.{split}.trl.txt"
    data = archive.extractfile(stem).read()
    roster = {}
    for line in data.decode("ascii").splitlines():
        speaker, name, placeholder, system, label = line.split()
        identity = UTTERANCE.fullmatch(name)
        genuine = label == "bonafide"
        need(SPEAKER.fullmatch(speaker) and identity and identity[2] == partition and placeholder == "-"
             and label in {"bonafide", "spoof"} and name not in roster, "Invalid or repeated challenge trial")
        if genuine:
            need(system == "-" and identity[1] == "LA", "Genuine trial metadata differs")
        elif partial:
            need(system == "CON" and identity[1] == "CON", "Unexpected PartialSpoof system metadata")
        else:
            need(ATTACK.fullmatch(system) and identity[1] == "LA", "Invalid parent attack metadata")
        roster[name] = speaker, label, system
    return roster, hashlib.sha256(data).hexdigest()


def read_intervals(data):
    source = data.decode("ascii").splitlines()
    midpoint = len(source) // 2
    repeated = bool(source and source[:midpoint] == source[midpoint:])
    if repeated:
        source = source[:midpoint]
    prior_end = Decimal(0)
    intervals = []
    for row in source:
        start, end, label = row.split()
        start, end = Decimal(start), Decimal(end)
        need(start.is_finite() and end.is_finite() and prior_end <= start < end and label in {"0", "1", "2"},
             "Invalid publisher interval")
        if intervals and intervals[-1][1] == start and intervals[-1][2] == label:
            intervals[-1] = intervals[-1][0], end, label
        else:
            intervals.append((start, end, label))
        prior_end = end
    need(intervals, "Empty publisher interval file")
    return intervals, repeated


def short_edit(intervals):
    spoof = [(a, b) for a, b, label in intervals if label == "0"]
    if len(spoof) != 1 or not 4 <= intervals[-1][1] <= 9:
        return False
    start, end = spoof[0]
    before = sum(b - a for a, b, label in intervals if label == "1" and b <= start)
    after = sum(b - a for a, b, label in intervals if label == "1" and a >= end)
    return Decimal(".50") <= end - start <= Decimal("1.20") and min(before, after) >= Decimal(".40")


def inventory(directory):
    assets = {name: digests(directory / name) for name in PINS}
    need(assets == PINS, "Publisher metadata archive differs")
    record = json.loads((directory / "5766198.json").read_bytes())
    published = {item["key"]: item for item in record["files"]}
    need(record["id"] == 5766198 and record["metadata"]["license"]["id"] == "cc-by-4.0"
         and all(published[name]["checksum"] == f"md5:{PINS[name][0]}" for name in PINS),
         "Publisher rights or metadata identifiers differ")
    split_counts, parent_speakers, public_rosters = {}, {}, {}
    with tarfile.open(directory / "database_protocols.tar.gz", "r:gz") as protocols:
        for name, partition in PARTITIONS.items():
            rows, trials_sha = cm_trials(protocols, name, partition, partial=True)
            original, parent_sha = cm_trials(protocols, name, partition, partial=False)
            genuine = {key for key, (_, label, _) in rows.items() if label == "bonafide"}
            need(all(rows[key] == original[key] for key in genuine), "Genuine parent identifiers differ")
            public_rosters[name] = rows
            parent_speakers[name] = {row[0] for row in rows.values()}
            split_counts[name] = {
                "trials_sha256": trials_sha,
                "parent_trials_sha256": parent_sha,
                "declared_speakers": len(parent_speakers[name]),
                "partial_spoof_protocol_codes": dict(Counter(system for _, _, system in rows.values())),
                "parent_attack_ids": sorted({system for _, _, system in original.values()} - {"-"}),
                "parent_genuine_ids_linked": len(genuine),
                "partial_spoof_generator_ids_linked": 0,
            }
    counts, group_counts, seen = defaultdict(Counter), defaultdict(set), set()
    with tarfile.open(directory / "database_vad.tar.gz", "r|gz") as archive:
        for entry in archive:
            if not entry.isfile():
                continue
            match = VAD_NAME.fullmatch(entry.name)
            need(match and 0 < entry.size < 65536, "Unexpected timing archive member")
            split, name = match.groups()
            need(name in public_rosters[split] and name not in seen, "Unbound or duplicate timing record")
            seen.add(name)
            speaker, label, _ = public_rosters[split][name]
            intervals, repeated = read_intervals(archive.extractfile(entry).read())
            has_spoof = any(tag == "0" for _, _, tag in intervals)
            counts[split]["files"] += 1
            counts[split]["duplicated_identical_clock"] += repeated
            counts[split]["spoof_trial_no_spoof_interval"] += label == "spoof" and not has_spoof
            counts[split]["genuine_trial_has_spoof_interval"] += label == "bonafide" and has_spoof
            if label == "spoof" and short_edit(intervals):
                counts[split]["short_edits"] += 1
                group_counts[split].add(speaker)
    for split in PARTITIONS:
        split_counts[split]["timing"] = dict(counts[split])
        split_counts[split]["timing"]["short_edit_declared_speakers"] = len(group_counts[split])
        split_counts[split]["timing"]["missing_records"] = len(public_rosters[split]) - counts[split]["files"]
    splits = tuple(PARTITIONS)
    overlaps = {f"{a}/{b}": len(parent_speakers[a] & parent_speakers[b]) for i, a in enumerate(splits) for b in splits[i + 1:]}
    prefixes = {}
    for split, (length, sha) in PREFIX.items():
        path = directory / f"{split}-prefix"
        response = (directory / f"{split}-range.headers").read_text().casefold().splitlines()
        need(f"content-range: bytes 0-65535/{length}" in response and path.stat().st_size == 65536
             and digests(path)[1] == sha, "Range response or content differs")
        prefixes[split] = {"compressed_bytes": 65536, "complete_archive_bytes": length, "sha256": sha}
    return {
        "schema": SCHEMA, "release": RELEASE, "license": "CC BY 4.0", "parent_license": "ODC-By 1.0",
        "pinned_metadata": {name: {"md5": md5, "sha256": sha} for name, (md5, sha) in PINS.items()},
        "timing_census": TARGET, "partitions": split_counts, "declared_speaker_intersections": overlaps,
        "bounded_prefix_probes": prefixes, "media_evaluated": 0, "inference_calls": 0,
        "suitable_for_proposed_generator_disjoint_study": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Public metadata downloads and bounded HTTP responses")
    parser.add_argument("--compare", type=Path, help="Expected canonical audit receipt")
    args = parser.parse_args()
    receipt = json.dumps(inventory(args.source), sort_keys=True, indent=2) + "\n"
    if args.compare:
        need(args.compare.read_text() == receipt, "The source audit differs from the saved receipt")
    print(receipt, end="")


if __name__ == "__main__":
    main()
