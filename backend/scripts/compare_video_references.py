"""Create or replay a video comparison using two independently supplied originals."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from app.core.canonical import canonical_json
from app.evidence.video_reference import compare


def run(earlier: Path, submitted: Path, output: Path, *, verify: bool = False) -> dict:
    record, assets = compare(earlier, submitted)
    files = {**assets, "comparison.json": canonical_json(record) + b"\n"}
    if verify and output.is_file():
        exported = json.loads(output.read_bytes())
        if (canonical_json(exported["comparison"]) != canonical_json(record)
                or {name: base64.b64decode(data, validate=True) for name, data in exported["assets"].items()} != assets):
            raise ValueError("The exported JSON or media differs from the re-decoded sources.")
    elif verify:
        if {file.name for file in output.iterdir()} != files.keys():
            raise ValueError("The comparison directory is missing files or contains unexpected entries.")
        for name, content in files.items():
            if not (output / name).is_file() or (output / name).read_bytes() != content:
                raise ValueError(f"The comparison differs from the re-decoded sources at {name}.")
    else:
        output.mkdir(mode=0o700)
        for name, content in files.items():
            (output / name).write_bytes(content)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("submitted", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.earlier, args.submitted, args.output, verify=args.verify)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{'Verified' if args.verify else 'Saved'} comparison. Status {result['status']}.")


if __name__ == "__main__":
    main()
