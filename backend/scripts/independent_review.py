"""Audit or summarize externally reproducible model evaluation receipts; no inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.bench.independent import audit_manifest, compare, digest, freeze, summarize  # noqa: E402


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, result: dict) -> None:
    with path.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


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
    lock.add_argument("--print-digest", action="store_true", help="Print only the canonical freeze digest")
    score = commands.add_parser("summarize")
    score.add_argument("manifest", type=Path)
    score.add_argument("confirmation", type=Path)
    score.add_argument("lock", type=Path)
    score.add_argument("output", type=Path)
    score.add_argument("--require-freeze-commitment", action="store_true", help="Require the confirmation receipt to commit to its exact freeze")
    pair = commands.add_parser("compare")
    pair.add_argument("manifest", type=Path)
    pair.add_argument("baseline", type=Path)
    pair.add_argument("baseline_lock", type=Path)
    pair.add_argument("candidate", type=Path)
    pair.add_argument("candidate_lock", type=Path)
    pair.add_argument("output", type=Path)
    pair.add_argument("--require-freeze-commitment", action="store_true", help="Require both confirmation receipts to commit to their exact freezes")
    example = commands.add_parser("demo", help="Create a fictional four-modality example with no media or weights")
    example.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        from scripts.independent_review_example import create

        create(args.output, write)
        print(args.output / "comparison.json")
        return
    manifest = read(args.manifest)
    if args.command == "audit":
        result = audit_manifest(manifest)
    elif args.command == "freeze":
        result = freeze(manifest, read(args.development), thresholds=read(args.thresholds))
    elif args.command == "summarize":
        result = summarize(manifest, read(args.confirmation), read(args.lock),
                           require_freeze_commitment=args.require_freeze_commitment)
    else:
        result = compare(manifest, read(args.baseline), read(args.baseline_lock),
                         read(args.candidate), read(args.candidate_lock),
                         require_freeze_commitment=args.require_freeze_commitment)
    if args.command != "audit":
        write(args.output, result)
    print(digest(result) if args.command == "freeze" and args.print_digest else json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
