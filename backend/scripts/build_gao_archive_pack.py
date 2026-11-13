"""Build the local GAO highlights pack from the three pinned GovReport JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REVISION = "32feeaede49fed993aef070bc4da09263fd0429a"
SCHEMA = "chipmunk.gao-highlights-pack/1"
PUBLISHER = f"https://huggingface.co/datasets/launch/gov_report/tree/{REVISION}/data"
SPLITS = {
    "train": ("ee7a981a768c7f0c68d635e4a3cbfac03b95e45284181232d6c7021d511ea729", 11005),
    "valid": ("9cae7d5fecbcb507d1e84ecf8d476f26796d50a6690ac7c79fabe7b0e13a9b16", 612),
    "test": ("5285ee273ae430003c1e8fdb45ed7c6757349c114d2cae45160845ef60a49a94", 611),
}
REPORT_ID = re.compile(r"[A-Z0-9][A-Z0-9-]{2,39}")
MAX_ROW_BYTES = 8 * 1024 * 1024
MAX_SOURCE_CHARS = 20_000


def highlights(row):
    return " ".join(" ".join(paragraph.split()) for section in row["highlight"]
                    if section["section_title"] != "What GAO Recommends" for paragraph in section["paragraphs"]).strip()


def acquire(sources):
    reports = {}
    for split, (expected, expected_count) in SPLITS.items():
        digest = hashlib.sha256()
        count = 0
        with sources[split] as stream:
            while line := stream.readline(MAX_ROW_BYTES + 1):
                if len(line) > MAX_ROW_BYTES or count >= expected_count:
                    raise ValueError(f"GovReport {split} exceeds its pinned record limits")
                count += 1
                digest.update(line)
                row = json.loads(line)
                key = row["id"].upper()
                if not REPORT_ID.fullmatch(key) or key in reports:
                    raise ValueError("Unexpected or duplicated GAO report ID")
                text = highlights(row)
                if len(text) > MAX_SOURCE_CHARS:
                    raise ValueError("Unexpected highlights size")
                reports[key] = {"text": text, "split": split, "published_date": row["published_date"][:10],
                                "source_record_sha256": hashlib.sha256(line).hexdigest()}
        if count != expected_count or digest.hexdigest() != expected:
            raise ValueError(f"GovReport {split} differs from the pinned publisher release")
    return {"schema": SCHEMA, "revision": REVISION, "reports": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    for split in SPLITS:
        parser.add_argument(f"--{split}", required=True, help="Pinned local JSONL path or '-' for one streamed split")
    args = parser.parse_args()
    files = {split: getattr(args, split) for split in SPLITS}
    if list(files.values()).count("-") > 1:
        parser.error("Only one split can come from standard input")
    pack = acquire({split: sys.stdin.buffer if path == "-" else Path(path).open("rb") for split, path in files.items()})
    with args.output.open("xb") as output:
        data = (json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
        output.write(data)
    print(hashlib.sha256(data).hexdigest(), args.output, flush=True)


if __name__ == "__main__":
    main()
