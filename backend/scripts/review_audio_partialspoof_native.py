"""Replay score-blind native PartialSpoof timing and same-speaker controls."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import wave
from collections import Counter, defaultdict
from pathlib import Path

from scripts import review_audio_partialspoof_source as metadata_audit
from scripts.review_audio_partialspoof_source import (
    PARTITIONS,
    PINS,
    VAD_NAME,
    cm_trials,
    digests,
    need,
    read_intervals,
    short_edit,
)

SCHEMA = "chipmunk.partialspoof-native/1"
RATE = 16_000
CHUNK = 16 * 1024**2
PREFIX = 128 * 1024**2
TOTAL = {"train": 2_027_101_056, "dev": 2_027_972_850}
ARCHIVES = {"train": "c4853ddd831e8e96b0e279fc0a512e7e", "dev": "ddd4cd3221b7210ac879f67452fb209e"}
RULE = {
    "intake": "Count only complete regular WAV members in the first 128 MiB of each publisher archive.",
    "geometry": "Use the earlier metadata audit's 0.50–1.20 s single edit, 4–9 s extent and 0.40 s genuine speech on each side.",
    "clock": "Require contiguous integral 16 kHz intervals starting at zero and a final label no more than 15 samples before PCM end. The suffix is unlabelled.",
    "controls": "Match within the declared speaker and split, with at most one second native duration difference. Process positives in digest order and take the unused nearest duration, breaking ties by digest.",
}


class Prefix:
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def read(self, size):
        data = bytearray()
        while len(data) < size:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    break
                self.current = path.open("rb")
            chunk = self.current.read(size - len(data))
            if not chunk:
                self.close()
            data.extend(chunk)
        return bytes(data)

    def close(self):
        if self.current is not None:
            self.current.close()
            self.current = None


def archive_members(paths, split):
    source = Prefix(paths)
    pattern = re.compile(rf"database/{split}/con_wav/((?:CON|LA)_{PARTITIONS[split]}_\d{{7}})\.wav")
    try:
        with tarfile.open(fileobj=source, mode="r|gz") as archive:
            try:
                for item in archive:
                    if item.isdir():
                        continue
                    match = pattern.fullmatch(item.name)
                    need(item.isfile() and match and 0 < item.size < 2 * 1024**2, "Unexpected publisher audio member")
                    body = archive.extractfile(item).read()
                    need(len(body) == item.size, "Incomplete publisher audio member")
                    yield match[1], body
            except (EOFError, tarfile.ReadError) as error:
                # A bounded prefix may end inside the next archive member.
                need("end" in str(error).lower(), "Unexpected audio archive read error")
    finally:
        source.close()


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def rank(identifier):
    return hash_bytes(f"{SCHEMA}/{identifier}".encode("ascii"))


def get_prefix(directory, split):
    files, hashes = [], []
    prefix = hashlib.sha256()
    for offset in range(0, PREFIX, CHUNK):
        path = directory / f"{split}-{offset}.range"
        response = path.with_suffix(".headers").read_text().casefold().splitlines()
        expected = f"content-range: bytes {offset}-{offset + CHUNK - 1}/{TOTAL[split]}"
        need(expected in response and any(re.fullmatch(r"http/\S+ 206(?:\s.*)?", line) for line in response),
             "Publisher response did not confirm the exact requested range")
        need(path.stat().st_size == CHUNK, "Publisher range is incomplete")
        data = path.read_bytes()
        if offset == 0:
            need(hash_bytes(data[:65536]) == metadata_audit.PREFIX[split][1], "Publisher prefix differs from the source audit")
        hashes.append(hash_bytes(data))
        prefix.update(data)
        files.append(path)
    return files, {"chunk_sha256": hashes, "prefix_sha256": prefix.hexdigest(),
                   "prefix_bytes": PREFIX, "archive_bytes": TOTAL[split], "archive_md5": ARCHIVES[split]}


def source_labels(metadata, rosters):
    labels = defaultdict(dict)
    with tarfile.open(metadata / "database_vad.tar.gz", "r|gz") as archive:
        for item in archive:
            match = VAD_NAME.fullmatch(item.name)
            if not match or match[1] not in rosters:
                continue
            split, identifier = match.groups()
            need(item.isfile() and 0 < item.size < 65536, "Unexpected training or development label member")
            body = archive.extractfile(item).read()
            intervals, repeated = read_intervals(body)
            need(not repeated and identifier in rosters[split] and identifier not in labels[split],
                 "Unexpected training or development label")
            labels[split][identifier] = hash_bytes(body), intervals
    return labels


def native_clock(body, intervals):
    with wave.open(io.BytesIO(body)) as stream:
        params = stream.getparams()
        n = stream.getnframes()
        pcm = stream.readframes(n + 1)
    if (params[:3] != (1, 2, RATE) or params.comptype != "NONE" or len(pcm) != 2 * n):
        return None
    ranges = []
    prior = 0
    for a, b, label in intervals:
        start, end = a * RATE, b * RATE
        if start != prior or int(end) != end or not start < end <= n:
            return None
        ranges.append([int(start), int(end), int(label)])
        prior = int(end)
    if not 0 <= n - prior <= 15:
        return None
    return n, n - prior, ranges, hash_bytes(pcm)


def qualify(files, split, roster, labels):
    counts, suffix = Counter(), Counter()
    positives, controls, seen = {}, {}, set()
    for name, body in archive_members(files, split):
        need(name not in seen and name in roster and name in labels, "Duplicate or unbound native audio")
        seen.add(name)
        counts["wav_members"] += 1
        speaker, role, _ = roster[name]
        counts[role] += 1
        label_sha, intervals = labels[name]
        if role == "spoof" and not short_edit(intervals):
            continue
        if role == "bonafide" and (any(tag == "0" for _, _, tag in intervals)
                                  or not 4 <= intervals[-1][1] <= 9):
            continue
        counts[f"{role}_label_qualified"] += 1
        try:
            checked = native_clock(body, intervals)
        except (EOFError, wave.Error):
            checked = None
        if checked is None:
            counts[f"{role}_native_refused"] += 1
            continue
        n, tail, ranges, pcm_sha = checked
        counts[f"{role}_native_qualified"] += 1
        suffix[tail] += 1
        selected = positives if role == "spoof" else controls
        selected[name] = {"id": name, "declared_speaker": speaker, "samples": n, "unlabelled_tail_samples": tail,
                          "intervals": ranges, "wav_sha256": hash_bytes(body), "pcm_sha256": pcm_sha,
                          "label_sha256": label_sha}
    groups = []
    for name, positive in sorted(positives.items(), key=lambda item: rank(item[0])):
        matches = [control for control in controls.values()
                   if positive["declared_speaker"] == control["declared_speaker"]
                   and abs(positive["samples"] - control["samples"]) <= RATE]
        control = min(matches, key=lambda row: (abs(positive["samples"] - row["samples"]), rank(row["id"])), default=None)
        if control:
            del controls[control["id"]]
        groups.append({"positive": positive, "control": control})
    return {"counts": dict(counts), "native_tail_samples": dict(sorted(suffix.items())), "candidate_groups": groups,
            "paired_cases": sum(group["control"] is not None for group in groups),
            "paired_speakers": len({group["positive"]["declared_speaker"] for group in groups if group["control"] is not None})}


def review(metadata, ranges):
    need(all(digests(metadata / name) == pins for name, pins in PINS.items()), "Publisher metadata differs")
    record = json.loads((metadata / "5766198.json").read_text())
    published = {item["key"]: item for item in record["files"]}
    need(record["id"] == 5766198 and record["metadata"]["license"]["id"] == "cc-by-4.0",
         "Publisher release or terms differ")
    rosters = {}
    with tarfile.open(metadata / "database_protocols.tar.gz", "r:gz") as archive:
        for split in TOTAL:
            item = published[f"database_{split}.tar.gz"]
            need(item["size"] == TOTAL[split] and item["checksum"] == f"md5:{ARCHIVES[split]}",
                 "Published archive identity differs")
            rosters[split] = cm_trials(archive, split, PARTITIONS[split], partial=True)[0]
    labels = source_labels(metadata, rosters)
    partitions = {}
    for split in rosters:
        files, binding = get_prefix(ranges, split)
        partitions[split] = {"delivery": binding, **qualify(files, split, rosters[split], labels[split])}
    speakers = [{pair["positive"]["declared_speaker"] for pair in split["candidate_groups"]} for split in partitions.values()]
    return {"schema": SCHEMA, "sample_rate": RATE, "rules": RULE, "partitions": partitions,
            "candidate_declared_speaker_intersection": len(set.intersection(*speakers)),
            "per_edit_generator_lineage": False, "inference_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("ranges", type=Path)
    parser.add_argument("--compare", type=Path, help="Expected canonical native intake receipt")
    args = parser.parse_args()
    receipt = json.dumps(review(args.metadata, args.ranges), sort_keys=True, indent=2) + "\n"
    if args.compare:
        need(args.compare.read_text() == receipt, "Native intake differs from the saved receipt")
    print(receipt, end="")


if __name__ == "__main__":
    main()
