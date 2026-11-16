"""Freeze and acquire new, creator-disjoint environmental sound pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.audio.decode import decode_audio
from scripts import environmental_sound_transfer as previous
from scripts.audio_focused_context_transfer import digest, read, save
from scripts.environmental_embedding_transfer import PINS

SCHEMA = "chipmunk.environmental-component-sources/1"
SALT = b"chipmunk-environmental-component-2026-11-16/"
ROLE_COUNTS = {"fit": 36, "selection": 20, "held": 24}
FIT_GENERATORS = previous.DEVELOPMENT_GENERATORS
HELD_GENERATORS = (*previous.HELD_GENERATORS, "tangoflux")


def plan_sources(parent):
    lock = previous.bound(parent)
    metadata = previous.metadata_paths(parent)
    if digest((parent / "freeze.json").read_bytes()) != PINS["environment_freeze"]:
        raise ValueError("The earlier environmental sound roster changed")
    publisher = {row["wavename"]: row for row in previous.read_csv(metadata[1])}
    grants = defaultdict(list)
    for file in metadata[2:5]:
        for row in previous.read_csv(file, "latin1"):
            options = grants[row["file_name"]]
            if row not in options:
                options.append(row)
    by_original = defaultdict(dict)
    candidates = read(metadata[0])
    for index, row in enumerate(candidates):
        source = publisher[row["filename"]]
        if (source["source dataset"] != "Clotho" or row["dataset"] != "Clotho"
                or source["faketype"] != row["attack_type"]):
            raise ValueError("The test shard no longer agrees with the publisher crosswalk")
        if row["attack_type"] not in {"real", "tta"}:
            continue
        group = by_original[source["source wavename"]]
        lane = row["generative_model"]
        if lane in group:
            raise ValueError("A source has multiple members for one origin")
        group[lane] = [index, row["filename"]]

    seen_ids = {row["source_id"] for row in lock["groups"]}
    seen_creators = {row["creator"].strip().casefold() for row in lock["groups"]}
    eligible = []
    for source, parts in by_original.items():
        if len(grants[source]) != 1 or set(parts) != {"real", *FIT_GENERATORS, *HELD_GENERATORS}:
            continue
        rights, = grants[source]
        if rights["license"].replace("http:", "https:") != previous.CC0:
            continue
        creator, identifier, url = (rights[key].strip() for key in ("manufacturer", "sound_id", "sound_link"))
        if not all((creator, identifier, url)) or identifier in seen_ids or creator.casefold() in seen_creators:
            continue
        eligible.append((digest(SALT + identifier.encode()), source, parts, creator, identifier, url))
    groups = []
    roles = [role for role, size in ROLE_COUNTS.items() for _ in range(size)]
    for _, source, parts, creator, identifier, url in sorted(eligible):
        if creator.casefold() in seen_creators or identifier in seen_ids:
            continue
        seen_creators.add(creator.casefold())
        seen_ids.add(identifier)
        role = roles[len(groups)]
        lanes = HELD_GENERATORS if role == "held" else FIT_GENERATORS
        groups.append({"group": len(groups), "source": source, "source_id": identifier, "creator": creator,
                       "source_url": url, "license": previous.CC0, "role": role,
                       "parts": {name: parts[name] for name in ("real", *lanes)}})
        if len(groups) == len(roles):
            break
    if Counter(row["role"] for row in groups) != ROLE_COUNTS:
        raise ValueError("The pinned shard contains too few creator-disjoint sound groups")
    return {"schema": SCHEMA, "dataset_revision": previous.REVISION, "shard": previous.SHARD,
            "metadata_sha256": lock["inputs"], "earlier_freeze_sha256": PINS["environment_freeze"],
            "planning_code_sha256": digest(Path(__file__).read_bytes()), "eligible_sources": len(eligible),
            "shard_rows": len(candidates), "groups": groups}


def audio_samples(body):
    import numpy as np

    decoded = decode_audio(body, source_sha256=digest(body.read_bytes()), max_seconds=4).pcm
    if len(decoded) < previous.N:
        raise ValueError("Expected at least three seconds of decoded sound")
    start = (len(decoded) - previous.N) // 2
    clipped = np.clip(np.rint(decoded[start:start + previous.N] * 32768.0), -32768, 32767)
    return start, clipped.astype("<i2").tobytes()


def acquire(folder, parent):
    import fsspec
    import pyarrow.parquet as pq

    plan_path = folder / "environment-plan.json"
    plan = read(plan_path)
    if plan != plan_sources(parent):
        raise ValueError("The frozen plan or its source metadata changed")
    assets = folder / "environment"
    assets.mkdir(mode=0o700, exist_ok=True)
    _, reference = previous.intake(parent)
    seen_wav = {row["wav_sha256"] for row in reference.values()}
    seen_pcm = {row["pcm_sha256"] for row in reference.values()}
    wanted = {index: (group["group"], lane, filename) for group in plan["groups"]
              for lane, (index, filename) in group["parts"].items()}
    rows = []
    with fsspec.open(previous.URL, block_size=32768, cache_type="readahead") as stream:
        archive = pq.ParquetFile(stream)
        if archive.metadata.num_rows != plan["shard_rows"]:
            raise ValueError("The pinned shard has a different number of samples")
        offset = 0
        for number in range(archive.num_row_groups):
            limit = offset + archive.metadata.row_group(number).num_rows
            indices = sorted(index for index in wanted if offset <= index < limit)
            if indices:
                table = archive.read_row_group(number, columns=["audio", "filename"], use_threads=False)
                for index in indices:
                    item = table.slice(index - offset, 1).to_pylist()[0]
                    group, lane, filename = wanted[index]
                    raw = item["audio"]["bytes"]
                    if item["filename"] != filename or not raw or len(raw) > 2_000_000:
                        raise ValueError("A pinned publisher audio member differs")
                    wav = assets / f"{group:02d}-{lane}.wav"
                    wav.write_bytes(raw)
                    start, pcm = audio_samples(wav)
                    wav_sha, pcm_sha = digest(raw), digest(pcm)
                    if wav_sha in seen_wav or pcm_sha in seen_pcm:
                        raise ValueError("Duplicate environmental audio across groups or studies")
                    seen_wav.add(wav_sha)
                    seen_pcm.add(pcm_sha)
                    wav.with_suffix(".pcm").write_bytes(pcm)
                    rows.append({"group": group, "lane": lane, "index": index, "publisher_filename": filename,
                                 "wav_sha256": wav_sha, "pcm_sha256": pcm_sha, "start_sample": start,
                                 "wav_bytes": len(raw)})
                print(f"Shard row group {number}: {len(rows)}/{len(wanted)} exact source files", flush=True)
            offset = limit
    if len(rows) != len(wanted):
        raise ValueError("Frozen environmental input acquisition is incomplete")
    save(folder / "environment-acquisition.json", {"schema": SCHEMA,
                                                  "plan_sha256": digest(plan_path.read_bytes()), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "acquire"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    if args.phase == "plan":
        document = plan_sources(args.previous)
        args.folder.mkdir(parents=True, mode=0o700, exist_ok=True)
        save(args.folder / "environment-plan.json", document)
        print(f"Froze {len(document['groups'])} new environmental sound groups before inference.", flush=True)
    else:
        acquire(args.folder, args.previous)


if __name__ == "__main__":
    main()
