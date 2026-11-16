# SPDX-License-Identifier: Apache-2.0
"""Prepare historical Wikipedia edits for a source-separated writing evaluation."""

from __future__ import annotations

import argparse
import io
import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zipfile import ZipFile

from scripts.writing_language_window_transfer import read, sha, write
from scripts.writing_word_localization import WORD

SCHEMA = "chipmunk.writing-editorial-controls/1"
ARCHIVE_SHA256 = "caba18ddad6d24e56610091ba1d0203cb0b069b59d7a07ab30c8a78412645294"
ARCHIVE_MEMBERS = {"fit": "biased.word.train", "held": "biased.word.test"}
QUOTAS = {"fit": 180, "held": 120}
API = "https://en.wikipedia.org/w/api.php"


def sequence(text):
    return [match.group().casefold() for match in WORD.finditer(text)]


def changed(before, after):
    alignment = SequenceMatcher(None, sequence(before), sequence(after), autojunk=False)
    return [index for tag, _, _, a, b in alignment.get_opcodes() if tag != "equal" for index in range(a, b)]


def fingerprints(text, width=12):
    tokens = sequence(text)
    return set(zip(*(tokens[i:] for i in range(width))))


def qualify(revision, before, after):
    before, after = (" ".join(value.split()) for value in (before, after))
    if any(not 14 <= len(sequence(value)) <= 60 or not value.endswith(".")
           or any(symbol in value for symbol in ("http", "[[", "]]", "{{", "}}", "<", ">", "\ufffd"))
           for value in (before, after)):
        return None
    edits = changed(before, after)
    if not 1 <= len(edits) <= 6 or not 0.75 <= SequenceMatcher(None, sequence(before), sequence(after), autojunk=False).ratio() < 1:
        return None
    return {"revision": revision, "before": before, "after": after, "changed": edits}


def candidates(archive):
    digest = sha(archive.read_bytes())
    if digest != ARCHIVE_SHA256:
        raise ValueError("The publisher archive differs from the frozen source")
    pool = {}
    with ZipFile(archive) as source:
        for phase, member in ARCHIVE_MEMBERS.items():
            rows = {}
            with io.TextIOWrapper(source.open(f"bias_data/WNC/{member}"), encoding="utf-8", errors="strict") as lines:
                for raw in lines:
                    fields = raw.rstrip("\n").split("\t")
                    if len(fields) == 7 and fields[0].isascii() and fields[0].isdecimal():
                        revision = fields[0]
                        row = qualify(revision, fields[3], fields[4])
                        if row and (revision not in rows or sha(row["before"]) < sha(rows[revision]["before"])):
                            rows[revision] = row
            ranked = sorted(rows, key=lambda revision: sha(f"{SCHEMA}/{phase}/{revision}"))
            if len(ranked) < 2 * QUOTAS[phase]:
                raise ValueError(f"Too few eligible publisher revisions for {phase}")
            pool[phase] = [rows[revision] for revision in ranked[:2 * QUOTAS[phase]]]
    return {"schema": SCHEMA, "archive_sha256": digest, "pool": pool}


def metadata(ids, directory, *, fetch):
    directory.mkdir(exist_ok=True)
    saved = {}
    for offset in range(0, len(ids), 50):
        revisions = ids[offset:offset + 50]
        name = directory / f"{sha('|'.join(revisions))}.json"
        if fetch and not name.exists():
            query = urlencode({"action": "query", "prop": "revisions", "revids": "|".join(revisions),
                               "rvprop": "ids|timestamp", "format": "json", "formatversion": 2})
            request = Request(f"{API}?{query}", headers={
                "User-Agent": "ChipmunkVerifyLocalBenchmark/1.0 (https://github.com/ChipmunkRPA/veritas)",
                "Accept": "application/json",
            })
            with urlopen(request, timeout=22) as response:
                answer = json.load(response)
            write(name, answer)
            time.sleep(0.15)
        answer = read(name)
        if "error" in answer:
            raise ValueError("The public history query failed")
        for page in answer.get("query", {}).get("pages", []):
            if page.get("ns") != 0 or type(page.get("pageid")) is not int:
                continue
            for revision in page["revisions"]:
                ident = str(revision["revid"])
                if ident not in revisions or ident in saved:
                    raise ValueError("The public history returned an unexpected revision")
                parent, when = revision["parentid"], revision["timestamp"]
                if type(parent) is int and parent > 0 and "2001" <= when < "2020":
                    saved[ident] = {"page": page["pageid"], "parent": parent, "timestamp": when}
    return saved


def assemble(pool, directory, *, fetch):
    if pool.get("schema") != SCHEMA or pool.get("archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("The editorial source plan differs")
    revisions = sorted({r["revision"] for rows in pool["pool"].values() for r in rows})
    dates = metadata(revisions, directory, fetch=fetch)
    pages, seen, splits = set(), set(), {}
    for phase in ("fit", "held"):
        selected = []
        for row in pool["pool"][phase]:
            revision = row["revision"]
            reading = dates.get(revision)
            if not reading or reading["page"] in pages:
                continue
            marks = fingerprints(row["before"]) | fingerprints(row["after"])
            if not marks or marks & seen:
                continue
            selected.append(row | reading)
            pages.add(reading["page"])
            seen.update(marks)
            if len(selected) == QUOTAS[phase]:
                break
        if len(selected) != QUOTAS[phase]:
            raise ValueError(f"The separated {phase} group quota could not be met")
        splits[phase] = selected
    return {"schema": SCHEMA, "archive_sha256": ARCHIVE_SHA256, "splits": splits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    freeze = phases.add_parser("freeze")
    freeze.add_argument("archive", type=Path)
    freeze.add_argument("output", type=Path)
    for phase in ("verify", "fetch"):
        command = phases.add_parser(phase)
        for name in ("frozen", "metadata", "output"):
            command.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        write(args.output, candidates(args.archive))
    else:
        source = assemble(read(args.frozen), args.metadata, fetch=args.phase == "fetch")
        write(args.output, source)
        print({key: len(rows) for key, rows in source["splits"].items()})


if __name__ == "__main__":
    main()
