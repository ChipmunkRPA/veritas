"""Prepare unused public speech and licensed nuisance inputs before neural inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
SR, SAMPLES = 16000, 128000
SOURCE_FILES = [f"all/train.clean.360/{i:04d}.parquet" for i in range(8)]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable(text):
    return hashlib.sha256(("chipmunk-public-speech-watermark-v2\0" + text).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def pcm(data):
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            "pipe:0",
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=True,
        timeout=30,
    )
    return np.frombuffer(result.stdout, dtype="<i2").copy()


def exclusions(main):
    root = main / "backend/data/independent-evaluation"
    paths = sorted(
        [p for p in root.rglob("source-plan.json") if "audio" in str(p)]
        + [
            root / d / "manifest.json"
            for d in [
                "audio",
                "audio-v2",
                "audio-v3",
                "audio-v4",
                "audio-v5",
                "audio-v6",
                "audio-v7",
                "audio-v8",
            ]
            if (root / d / "manifest.json").is_file()
        ]
    )
    seen = set()
    for path in paths:
        seen.update(re.findall(r"(?:librispeech:|duplex-prompt:)([0-9]+)", path.read_text()))
    return seen, [{"path": str(p.relative_to(main)), "sha256": digest(p)} for p in paths]


def metadata(root):
    import fsspec
    import pyarrow.parquet as pq

    combined = []
    for name in SOURCE_FILES:
        path = root / "metadata" / Path(name).name
        path = path.with_suffix(".json")
        url = f"https://huggingface.co/datasets/openslr/librispeech_asr/resolve/{REVISION}/{name}"
        with fsspec.open(url, "rb", block_size=65536) as stream:
            reader = pq.ParquetFile(stream)
            rows = []
            for group in range(reader.num_row_groups):
                values = reader.read_row_group(
                    group, columns=["speaker_id", "chapter_id", "id", "text"]
                ).to_pylist()
                rows.extend(
                    {**row, "row_group": group, "group_row": i, "parquet": name}
                    for i, row in enumerate(values)
                )
            record = {"url": url, "rows": rows, "groups": reader.num_row_groups}
        write(path, record)
        combined.extend(rows)
        print(json.dumps({"metadata_file": name, "rows": len(rows), "model_forwards": 0}), flush=True)
    return combined


def prepare(main, root):
    import fsspec
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=False)
    seen, old = exclusions(main)
    rows = metadata(root)
    speakers = defaultdict(list)
    for row in rows:
        if str(row["speaker_id"]) not in seen:
            speakers[str(row["speaker_id"])].append(row)
    if len(speakers) < 32:
        raise ValueError("Fewer than32 unused known speaker identities; retain failed preflight")
    selected = sorted(speakers, key=stable)[:32]
    # Read each needed audio row group once. Selection uses identity and duration,
    # never a detector response or preferred assessment outcome.
    candidates = {s: sorted(speakers[s], key=lambda row: stable(row["id"])) for s in selected}
    wanted = {(r["parquet"], r["row_group"]) for group in candidates.values() for r in group}
    available = {}
    budget = 0
    for name, group in sorted(wanted):
        url = f"https://huggingface.co/datasets/openslr/librispeech_asr/resolve/{REVISION}/{name}"
        with fsspec.open(url, "rb", block_size=65536) as stream:
            reader = pq.ParquetFile(stream)
            compressed = sum(
                reader.metadata.row_group(group).column(i).total_compressed_size
                for i in range(reader.metadata.row_group(group).num_columns)
            )
            budget += compressed
            if budget > 3_000_000_000:
                raise ValueError("Bounded public range-read budget exceeded")
            values = reader.read_row_group(group, columns=["audio", "id"]).to_pylist()
        ids = {
            r["id"]
            for c in candidates.values()
            for r in c
            if r["parquet"] == name and r["row_group"] == group
        }
        for row in values:
            if row["id"] in ids:
                available[row["id"]] = row["audio"]["bytes"]
        del values
        print(json.dumps({"audio_group": name, "group": group, "compressed_bytes": compressed}), flush=True)
    fresh = []
    for speaker in selected:
        rejected = []
        for row in candidates[speaker]:
            data = available[row["id"]]
            signal = pcm(data)
            if len(signal) < SAMPLES:
                rejected.append({"id": row["id"], "samples": len(signal), "reason": "under8seconds"})
                continue
            source = root / "sources" / f"{stable(row['id'])}.flac"
            source.parent.mkdir(exist_ok=True)
            source.write_bytes(data)
            input_path = root / "inputs" / f"{stable(row['id'])}.npy"
            input_path.parent.mkdir(exist_ok=True)
            np.save(input_path, signal[:SAMPLES], allow_pickle=False)
            fresh.append(
                {
                    "id": row["id"],
                    "speaker": "librispeech:" + speaker,
                    "dataset": "openslr/librispeech_asr",
                    "revision": REVISION,
                    "parquet": row["parquet"],
                    "row_group": row["row_group"],
                    "group_row": row["group_row"],
                    "source_path": str(source.relative_to(root)),
                    "source_sha256": digest(source),
                    "source_samples": len(signal),
                    "input_path": str(input_path.relative_to(root)),
                    "input_sha256": digest(input_path),
                    "start_sample": 0,
                    "samples": SAMPLES,
                    "license": "CC-BY-4.0",
                    "license_url": "https://www.openslr.org/12/",
                    "attribution": "LibriSpeech, Panayotov et al.; LibriVox readers.",
                    "rejected_before_selection": rejected,
                }
            )
            break
        else:
            raise ValueError("No8second eligible source for selected speaker; retain preflight")
    del available
    write(
        root / "foregrounds.json",
        {
            "foregrounds": fresh,
            "prior_identity_sources": old,
            "excluded_known_speakers": len(seen),
            "compressed_audio_groups_read_bytes": budget,
        },
    )
    finish_backgrounds(main, root, fresh, old, seen, budget)


def production_pcm(row):
    from app.detectors.audio.decode import decode_audio

    decoded = decode_audio(Path(row["source_path"]), source_sha256=row["sha256"])
    if decoded.mapping["pcm_sha256"] != row["source_production_input"]["pcm_sha256"]:
        raise ValueError("Background production PCM identity changed")
    return np.rint(decoded.pcm * 32768).astype("<i2")


def finish_backgrounds(main, root, fresh, old, seen, budget):
    prior = main / "backend/data/independent-evaluation/audio-v9/source-plan.json"
    old_backgrounds = json.loads(prior.read_text())["backgrounds"]
    bgs = []
    for row in old_backgrounds:
        if row["full_samples"] < SAMPLES:
            continue
        path = Path(row["source_path"])
        if digest(path) != row["sha256"]:
            raise ValueError("Background source checksum changed")
        signal = production_pcm(row)
        if row["canonical_speaker"] in {r["speaker"] for r in fresh}:
            raise ValueError("Background speaker overlaps fresh foreground")
        bgs.append(row)
    trials = []
    for foreground in fresh:
        selected_bg = {}
        for kind in ["noise", "music", "speech"]:
            pool = sorted(
                [b for b in bgs if b["kind"] == kind],
                key=lambda b: stable(foreground["id"] + "\0" + b["sha256"]),
            )
            bg = pool[0]
            signal = production_pcm(bg)
            offset = int(stable(foreground["id"] + "\0offset\0" + bg["sha256"])[:16], 16) % (
                len(signal) - SAMPLES + 1
            )
            part = signal[offset : offset + SAMPLES]
            if not np.any(part):
                raise ValueError("Selected background is silent; retain failed preflight")
            target = root / "inputs" / f"{stable(foreground['id'] + kind)}.npy"
            np.save(target, part, allow_pickle=False)
            selected_bg[kind] = {
                **bg,
                "input_path": str(target.relative_to(root)),
                "input_sha256": digest(target),
                "start_sample": offset,
                "samples": SAMPLES,
                "project_consumed": True,
            }
        trials.append({"foreground": foreground, "backgrounds": selected_bg})
    result = {
        "schema": "chipmunk.public-speech-watermark-inputs/2",
        "trials": trials,
        "prior_identity_sources": old,
        "excluded_known_speakers": len(seen),
        "fresh_foreground_speakers": 32,
        "foreground_project_overlap": 0,
        "background_speaker_overlap": 0,
        "backgrounds_project_consumed": True,
        "external_pretraining_overlap": "unknown",
        "source_pool": SOURCE_FILES,
        "source_index_sha256": {p.name: digest(p) for p in sorted((root / "metadata").glob("*.json"))},
        "background_source_plan_sha256": digest(prior),
        "compressed_audio_groups_read_bytes": budget,
        "source_selection": "Salted identity order, first32 known-unused speakers; first hash-ordered utterance with at least8s; prefix0–8s.",
        "model_forwards": 0,
        "customer_inputs": 0,
    }
    write(root / "source-plan.json", result)
    print(
        json.dumps(
            {"prepared": len(trials), "fresh_speakers": 32, "backgrounds_reused": True, "model_forwards": 0}
        ),
        flush=True,
    )


def recover(main, root, failed_profile):
    """Recover pre-inference public source files; never rerun a failed native experiment."""
    profile = json.loads((failed_profile / "profile.json").read_text())
    if (
        profile["child_exit_code"] != 1
        or profile["guard"] is not None
        or (root / "source-plan.json").exists()
    ):
        raise ValueError("Exact retained failed source-preparation phase required")
    rows = []
    for p in sorted((root / "metadata").glob("*.json")):
        rows.extend(json.loads(p.read_text())["rows"])
    seen, old = exclusions(main)
    speakers = defaultdict(list)
    for row in rows:
        if str(row["speaker_id"]) not in seen:
            speakers[str(row["speaker_id"])].append(row)
    fresh = []
    for speaker in sorted(speakers, key=stable)[:32]:
        matches = [
            row for row in speakers[speaker] if (root / "sources" / (stable(row["id"]) + ".flac")).is_file()
        ]
        if len(matches) != 1:
            raise ValueError("One retained selected source for each planned speaker required")
        row = matches[0]
        source = root / "sources" / (stable(row["id"]) + ".flac")
        path = root / "inputs" / (stable(row["id"]) + ".npy")
        signal = pcm(source.read_bytes())
        saved = np.load(path, allow_pickle=False)
        if len(signal) < SAMPLES or saved.dtype != np.int16 or not np.array_equal(signal[:SAMPLES], saved):
            raise ValueError("Recovered exact source input differs")
        fresh.append(
            {
                "id": row["id"],
                "speaker": "librispeech:" + speaker,
                "dataset": "openslr/librispeech_asr",
                "revision": REVISION,
                "parquet": row["parquet"],
                "row_group": row["row_group"],
                "group_row": row["group_row"],
                "source_path": str(source.relative_to(root)),
                "source_sha256": digest(source),
                "source_samples": len(signal),
                "input_path": str(path.relative_to(root)),
                "input_sha256": digest(path),
                "start_sample": 0,
                "samples": SAMPLES,
                "license": "CC-BY-4.0",
                "license_url": "https://www.openslr.org/12/",
                "attribution": "LibriSpeech, Panayotov et al.; LibriVox readers.",
                "selection_receipt": "Recovered32 retained pre-inference sources from failed source preparation; earlier rejected short candidates were not retained.",
            }
        )
    if len(fresh) != 32 or len(list((root / "sources").glob("*.flac"))) != 32:
        raise ValueError("Complete32-speaker recovery required")
    # The original source-only log records compressed group sizes, before failure.
    events = [
        json.loads(line)
        for line in (failed_profile / "worker.log").read_text().splitlines()
        if line.startswith('{"audio_group"')
    ]
    budget = sum(e["compressed_bytes"] for e in events)
    write(
        root / "foreground-recovery.json",
        {
            "foregrounds": fresh,
            "failed_profile_sha256": digest(failed_profile / "profile.json"),
            "failed_code_sha256": digest(failed_profile / "preparer-attempt1.py"),
            "model_forwards": 0,
            "compressed_audio_groups_read_bytes": budget,
            "reason": "Compare the actual production float32 PCM checksum, rather than signed16 byte hashes; original foreground byte inputs are preserved.",
        },
    )
    finish_backgrounds(main, root, fresh, old, seen, budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("main", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--recover-failed-profile", type=Path)
    args = parser.parse_args()
    if args.recover_failed_profile:
        recover(args.main, args.root, args.recover_failed_profile)
    else:
        prepare(args.main, args.root)
