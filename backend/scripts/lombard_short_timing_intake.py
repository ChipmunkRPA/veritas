# SPDX-License-Identifier: Apache-2.0
"""Freeze untouched Lombard source groups for a short unmarked timing test."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from scripts import lombard_mouth_source_intake as prior
from scripts.video_common_sources_v89 import read, require, sha, write

SEED = "chipmunk-lombard-short-timing-independent/1"
GROUPS = 12
ITEMS_PER_STYLE = 2


def order(code):
    return hashlib.sha256(f"{SEED}/{code}".encode()).digest()


def plan(existing, folder):
    intake = read(existing / "source-receipt.json")
    require(intake["freeze_sha256"] == sha(existing / "freeze.json"), "Prior source roster changed")
    exclude = {r["speaker"] for r in intake["rows"]}
    require(len(exclude) == 8, "Previous speaker exclusions differ")
    archives = prior.publisher(folder)
    roster = defaultdict(dict)
    with ZipFile(folder / "publisher-json.zip") as metadata, ZipFile(folder / "alignment.zip") as alignments:
        labels = prior.index_zip(alignments, ".json")
        for entry in prior.index_zip(metadata, ".json").values():
            for row in json.loads(metadata.read(entry)):
                speaker, style, utterance = (row[k] for k in ("SPKR", "COND", "UTTERANCE"))
                source = f"{speaker}_{style}_{utterance}"
                if (
                    speaker in exclude
                    or style not in {"p", "l"}
                    or row["STATUS"] != "CORRECT"
                    or source not in labels
                ):
                    continue
                raw = alignments.read(labels[source])
                if target := prior.target(json.loads(raw)[source]):
                    roster[speaker][source] = {
                        "condition": style,
                        "alignment_sha256": hashlib.sha256(raw).hexdigest(),
                        **target,
                    }
    with (
        ZipFile(prior.DirectoryReader(archives["front"])) as front,
        ZipFile(prior.DirectoryReader(archives["audio"])) as audio,
    ):
        indexed = {"front": prior.index_zip(front, ".mov"), "audio": prior.index_zip(audio, ".wav")}
    rows = []
    groups = 0
    for speaker in sorted(roster, key=order):
        eligible = [
            source
            for source in sorted(roster[speaker], key=order)
            if all(source in files for files in indexed.values())
        ]
        selected = [
            [s for s in eligible if roster[speaker][s]["condition"] == style][:ITEMS_PER_STYLE]
            for style in ("p", "l")
        ]
        if min(map(len, selected)) < ITEMS_PER_STYLE:
            continue
        groups += 1
        for source in (*selected[0], *selected[1]):
            rows.append(
                {
                    "ordinal": len(rows) + 1,
                    "group": groups,
                    "speaker": speaker,
                    "split": "held",
                    "source": source,
                    "members": {
                        role: {
                            "name": files[source].filename,
                            "bytes": files[source].file_size,
                            "crc32": files[source].CRC,
                        }
                        for role, files in indexed.items()
                    },
                    **roster[speaker][source],
                }
            )
        if groups == GROUPS:
            break
    require(groups == GROUPS, "Need complete fresh speaker groups")
    return {
        "schema": "chipmunk.lombard-short-timing-source-freeze/1",
        "archives": archives,
        "rows": rows,
        "prior_source_receipt_sha256": sha(existing / "source-receipt.json"),
        "worker_sha256": sha(Path(prior.__file__)),
        "selection_sha256": sha(Path(__file__)),
        "protocol": "Fixed speaker-hash order, two utterances per style, with prior eight speaker codes excluded. "
        "Publisher alignments are used only to qualify and author known event controls. The detector receives neither labels nor event times.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "acquire"))
    p.add_argument("existing", type=Path)
    p.add_argument("folder", type=Path)
    a = p.parse_args()
    if a.phase == "freeze":
        a.folder.mkdir(mode=0o700)
        for file in ("record.json", *prior.METADATA):
            shutil.copyfile(a.existing / file, a.folder / file)
        write(a.folder / "freeze.json", plan(a.existing, a.folder))
    else:
        lock = read(a.folder / "freeze.json")
        require(
            lock["selection_sha256"] == sha(Path(__file__))
            and lock["prior_source_receipt_sha256"] == sha(a.existing / "source-receipt.json"),
            "Fresh source plan differs from its commitments",
        )
        prior.acquire(a.folder)


if __name__ == "__main__":
    main()
