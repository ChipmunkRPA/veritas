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
    locations = commands.add_parser("count-locations", help="Count source-coordinate overlap for all planned localization cases")
    locations.add_argument("manifest", type=Path)
    locations.add_argument("truth", type=Path)
    locations.add_argument("response", type=Path)
    locations.add_argument("output", type=Path)
    locations.add_argument("--image-mask-dir", type=Path, help="Local directory for explicitly declared binary PBM masks")
    inputs = commands.add_parser("verify-inputs", help="Rehash every planned input and parent source on disk")
    inputs.add_argument("manifest", type=Path)
    inputs.add_argument("file_index", type=Path)
    inputs.add_argument("output", type=Path)
    inputs.add_argument("--root", type=Path, required=True)
    repeat = commands.add_parser("replicate", help="Compare independently produced receipts from one frozen method")
    repeat.add_argument("manifest", type=Path)
    repeat.add_argument("reference", type=Path)
    repeat.add_argument("replication", type=Path)
    repeat.add_argument("lock", type=Path)
    repeat.add_argument("output", type=Path)
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
    elif args.command == "count-locations":
        from app.bench.independent_locations import count_locations

        result = count_locations(manifest, read(args.truth), read(args.response), mask_directory=args.image_mask_dir)
        write(args.output, result)
        print(digest(result))
        return
    elif args.command == "verify-inputs":
        from app.bench.independent_inputs import verify_inputs

        result = verify_inputs(manifest, read(args.file_index), args.root)
    elif args.command == "replicate":
        from app.bench.independent_replication import replicate

        result = replicate(manifest, read(args.reference), read(args.replication), read(args.lock))
    else:
        result = compare(manifest, read(args.baseline), read(args.baseline_lock),
                         read(args.candidate), read(args.candidate_lock),
                         require_freeze_commitment=args.require_freeze_commitment)
    if args.command != "audit":
        write(args.output, result)
    print(digest(result) if args.command == "freeze" and args.print_digest else json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
