"""Freeze a source-bound synthetic-only speech panel without decoding media."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts.video_appearance_native_v108 import ARCHIVE, BASE, ROOT, read, require, sha, write

NAME = "audio-alania-independent-positive-v178"
SOURCE, TRIAL = ARCHIVE / NAME, BASE / NAME
PUBLIC = ROOT / "docs/evaluation-panels/audio-alania-roster-freeze-v178.json"
METHOD = ROOT / "docs/audio-alania-roster-v178-method.md"
REVISION = "277dfc8a2322014099ff72688bcaee12480c4605"
SHARD = "data/cc-by/train-00000.parquet"
SHARD_SIZE = 827_139_749
SHARD_SHA = "72820d81a4080d2b5ddf29eb61e1d2518275537ce9df37886df30d5feb718949"
GENERATOR = "openbmb/VoxCPM2@32279eff+nanovllm-voxcpm@c702631e"
PLANNED = 24
COLUMNS = (
    "id",
    "duration",
    "voice_id",
    "voice_type",
    "render_mode",
    "text_source",
    "text_doc_id",
    "generator",
    "postfilter",
    "license",
)


def eligible(row):
    return (
        row["text_source"] == "domain"
        and row["license"] == "CC-BY-4.0"
        and row["voice_type"] == "designed-identity"
        and row["render_mode"] in {"reference", "reference+instruction"}
        and row["generator"] == GENERATOR
        and row["postfilter"] == "resclip"
        and 2.5 <= row["duration"] <= 4.0
        and bool(row["id"] and row["voice_id"] and row["text_doc_id"])
    )


def select_rows(rows, limit=PLANNED):
    order = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"chipmunk.audio-alania/178:{SHARD_SHA}:{row['row_group']}:{row['row_offset']}".encode()
        ).digest(),
    )
    voices, documents, selected = set(), set(), []
    for row in order:
        if row["voice_id"] in voices or row["text_doc_id"] in documents:
            continue
        voices.add(row["voice_id"])
        documents.add(row["text_doc_id"])
        selected.append({"slot": f"positive-{len(selected):02d}", **row})
        if len(selected) == limit:
            break
    require(len(selected) == limit, "Insufficient distinct designed voices and text documents")
    return selected


def source_identity():
    revision = read(SOURCE / "revision.json")
    rows = read(SOURCE / "tree-page.json")
    [shard] = [item for item in rows if item["path"] == SHARD]
    path = SOURCE / Path(SHARD).name
    require(
        revision["sha"] == REVISION
        and revision["private"] is False
        and not revision["gated"]
        and shard["size"] == shard["lfs"]["size"] == SHARD_SIZE
        and shard["lfs"]["oid"] == SHARD_SHA
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == SHARD_SIZE
        and sha(path) == SHARD_SHA,
        "Pinned public publisher metadata and full shard bytes must agree",
    )
    return path


def plan():
    path = source_identity()
    candidates = []
    with pq.ParquetFile(path) as file:
        metadata_rows, row_groups = file.metadata.num_rows, file.num_row_groups
        for group in range(row_groups):
            table = file.read_row_group(group, columns=list(COLUMNS), use_threads=False)
            require(tuple(table.schema.names) == COLUMNS, "Only the exact metadata columns may be read")
            for index, row in enumerate(table.to_pylist()):
                if eligible(row):
                    candidates.append({**row, "row_group": group, "row_offset": index})
    require(len({row["id"] for row in candidates}) == len(candidates), "Source rows are not unique")
    selected = select_rows(candidates)
    aggregate = {
        "schema": "chipmunk.audio-alania-roster-freeze/178",
        "publisher_revision": REVISION,
        "shard": SHARD,
        "shard_bytes": SHARD_SIZE,
        "shard_sha256": SHARD_SHA,
        "publisher_records_sha256": {
            name: sha(SOURCE / name)
            for name in ("revision.json", "tree-page.json", "README.md", "LICENSE.md")
        },
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__), METHOD)},
        "shard_rows": metadata_rows,
        "shard_row_groups": row_groups,
        "eligible_rows": len(candidates),
        "eligible_voice_groups": len({row["voice_id"] for row in candidates}),
        "eligible_text_documents": len({row["text_doc_id"] for row in candidates}),
        "selected_rows": len(selected),
        "selected_voice_groups": len({row["voice_id"] for row in selected}),
        "selected_text_documents": len({row["text_doc_id"] for row in selected}),
        "selected_render_modes": dict(Counter(row["render_mode"] for row in selected)),
        "scope": "Positive-only Turkish synthetic speech transfer. Exact audio decode and scoring pending.",
        "decoded_audio_rows": 0,
        "model_calls": 0,
    }
    return selected, aggregate


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve the first source freeze")
    selected, aggregate = plan()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "private-roster.json", selected)
    aggregate["private_roster_sha256"] = sha(TRIAL / "private-roster.json")
    write(TRIAL / "freeze.json", aggregate)
    write(PUBLIC, aggregate)
    print(json.dumps(aggregate, indent=2))


def verify():
    selected, aggregate = plan()
    private = TRIAL / "private-roster.json"
    aggregate["private_roster_sha256"] = sha(private)
    require(
        read(private) == selected and read(PUBLIC) == read(TRIAL / "freeze.json") == aggregate,
        "Exact frozen metadata selection required",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify"))
    (freeze if parser.parse_args().command == "freeze" else verify)()
