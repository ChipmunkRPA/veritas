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
    pair.add_argument("--useful-location", nargs=2, type=float, metavar=("IOU", "RECALL"),
                      help="Count paired localization outcomes at the declared minimum IoU and recall")
    locations = commands.add_parser("count-locations", help="Count source-coordinate overlap for all planned localization cases")
    locations.add_argument("manifest", type=Path)
    locations.add_argument("truth", type=Path)
    locations.add_argument("response", type=Path)
    locations.add_argument("output", type=Path)
    locations.add_argument("--image-mask-dir", type=Path, help="Local directory for explicitly declared binary PBM masks")
    events = commands.add_parser("count-events", help="Count individual edit matches without disclosing private annotations")
    events.add_argument("manifest", type=Path)
    events.add_argument("truth", type=Path)
    events.add_argument("response", type=Path)
    events.add_argument("output", type=Path)
    events.add_argument("--match", nargs=2, type=float, required=True, metavar=("IOU", "RECALL"),
                        help="Declare the minimum per-event IoU and recall")
    events.add_argument("--image-mask-dir", type=Path, help="Local directory for explicitly declared binary PBM masks")
    paired_events = commands.add_parser("compare-events", help="Compare two localizers and their combined edit locations")
    for field in ("manifest", "truth", "baseline", "candidate", "output"):
        paired_events.add_argument(field, type=Path)
    paired_events.add_argument("--match", nargs=2, type=float, required=True, metavar=("IOU", "RECALL"),
                               help="Declare the minimum per-event IoU and recall")
    paired_events.add_argument("--image-mask-dir", type=Path, help="Local directory for explicitly declared binary PBM masks")
    inputs = commands.add_parser("verify-inputs", help="Rehash every planned input and parent source on disk")
    inputs.add_argument("manifest", type=Path)
    inputs.add_argument("file_index", type=Path)
    inputs.add_argument("output", type=Path)
    inputs.add_argument("--root", type=Path, required=True)
    positions = commands.add_parser("packet-positions", help="Read review coordinates from a signed app export and its original")
    for field in ("package", "original", "output"):
        positions.add_argument(field, type=Path)
    positions.add_argument("--public-key", type=Path, required=True, help="Instance public key obtained separately from the package")
    repeat = commands.add_parser("replicate", help="Compare independently produced receipts from one frozen method")
    repeat.add_argument("manifest", type=Path)
    repeat.add_argument("reference", type=Path)
    repeat.add_argument("replication", type=Path)
    repeat.add_argument("lock", type=Path)
    repeat.add_argument("output", type=Path)
    exact = commands.add_parser("replicate-locations", help="Compare the exact locations of two runs without sharing coordinates")
    exact.add_argument("manifest", type=Path)
    exact.add_argument("truth", type=Path)
    exact.add_argument("reference", type=Path)
    exact.add_argument("replication", type=Path)
    exact.add_argument("lock", type=Path)
    exact.add_argument("output", type=Path)
    exact.add_argument("--image-mask-dir", type=Path, help="Local directory containing the declared PBM masks")
    example = commands.add_parser("demo", help="Create a fictional four-modality example with no media or weights")
    example.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "demo":
        from scripts.independent_review_example import create

        create(args.output, write)
        print(args.output / "comparison.json")
        return
    if args.command == "packet-positions":
        from app.bench.independent_app_positions import packet_positions

        result = packet_positions(args.package, args.original, args.public_key)
        write(args.output, result)
        print(digest(result))
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
    elif args.command == "count-events":
        from app.bench.independent_locations import count_events

        result = count_events(manifest, read(args.truth), read(args.response), mask_directory=args.image_mask_dir,
                              minimum_iou=args.match[0], minimum_recall=args.match[1])
    elif args.command == "compare-events":
        from app.bench.independent_locations import compare_events

        result = compare_events(manifest, read(args.truth), read(args.baseline), read(args.candidate),
                                mask_directory=args.image_mask_dir, minimum_iou=args.match[0], minimum_recall=args.match[1])
    elif args.command == "replicate":
        from app.bench.independent_replication import replicate

        result = replicate(manifest, read(args.reference), read(args.replication), read(args.lock))
    elif args.command == "replicate-locations":
        from app.bench.independent_replication import replicate_locations

        result = replicate_locations(manifest, read(args.truth), read(args.reference), read(args.replication), read(args.lock),
                                     mask_directory=args.image_mask_dir)
    else:
        result = compare(manifest, read(args.baseline), read(args.baseline_lock),
                         read(args.candidate), read(args.candidate_lock),
                         require_freeze_commitment=args.require_freeze_commitment,
                         useful_location=tuple(args.useful_location) if args.useful_location else None)
    if args.command != "audit":
        write(args.output, result)
    print(digest(result) if args.command == "freeze" and args.print_digest else json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
