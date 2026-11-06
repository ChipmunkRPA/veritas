"""Audit or summarize externally reproducible model evaluation receipts; no inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.bench.independent import audit_manifest, compare, freeze, summarize  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("manifest", type=Path)
    lock = commands.add_parser("freeze")
    lock.add_argument("manifest", type=Path)
    lock.add_argument("development", type=Path)
    lock.add_argument("thresholds", type=Path)
    lock.add_argument("output", type=Path)
    score = commands.add_parser("summarize")
    score.add_argument("manifest", type=Path)
    score.add_argument("confirmation", type=Path)
    score.add_argument("lock", type=Path)
    score.add_argument("output", type=Path)
    pair = commands.add_parser("compare")
    pair.add_argument("manifest", type=Path)
    pair.add_argument("baseline", type=Path)
    pair.add_argument("baseline_lock", type=Path)
    pair.add_argument("candidate", type=Path)
    pair.add_argument("candidate_lock", type=Path)
    pair.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = read(args.manifest)
    if args.command == "audit":
        result = audit_manifest(manifest)
    elif args.command == "freeze":
        result = freeze(manifest, read(args.development), thresholds=read(args.thresholds))
    elif args.command == "summarize":
        result = summarize(manifest, read(args.confirmation), read(args.lock))
    else:
        result = compare(manifest, read(args.baseline), read(args.baseline_lock),
                         read(args.candidate), read(args.candidate_lock))
    if args.command != "audit":
        with args.output.open("x") as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
