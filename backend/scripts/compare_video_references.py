"""Create or replay a video comparison using two independently supplied originals."""

from __future__ import annotations

import argparse
import base64
import json
from fractions import Fraction
from functools import partial
from pathlib import Path

from app.core.canonical import canonical_json
from app.evidence.video_reference import SCHEMA, SCHEMA_WITH_SHORT_TIMING, SCHEMA_WITH_TIMING, compare


def replay(compare, earlier: Path, submitted: Path, output: Path, *, verify: bool = False) -> dict:
    record, assets = compare(earlier, submitted)
    files = {**assets, "comparison.json": canonical_json(record) + b"\n"}
    if verify and output.is_file():
        exported = json.loads(output.read_bytes())
        if (not isinstance(exported, dict) or not isinstance(exported.get("assets"), dict)
                or canonical_json(exported.get("comparison")) != canonical_json(record)
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


def run(earlier: Path, submitted: Path, output: Path, *, verify: bool = False,
        event_seconds: Fraction | None = None, timing_model: Path | None = None) -> dict:
    compare_source = compare
    short_fallback = not verify
    if verify:
        exported = json.loads((output if output.is_file() else output / "comparison.json").read_bytes())
        record = exported.get("comparison", exported) if isinstance(exported, dict) else None
        event_seconds = None
        if isinstance(record, dict) and record.get("schema") == SCHEMA:
            compare_source = partial(compare, include_local=False)
        elif isinstance(record, dict) and record.get("schema") in (SCHEMA_WITH_TIMING, SCHEMA_WITH_SHORT_TIMING):
            event_seconds = Fraction(record["timing"]["requested_earlier_time_s"])
            short_fallback = record["schema"] == SCHEMA_WITH_SHORT_TIMING
    if event_seconds is not None:
        from app.core.config import settings

        compare_source = partial(compare, timing_event_s=event_seconds,
                                 timing_model=timing_model or settings.video_reference_timing_model_path,
                                 timing_short_fallback=short_fallback)
    return replay(compare_source, earlier, submitted, output, verify=verify)


def main(*, description=__doc__, runner=run, marked_timing=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("earlier", type=Path)
    parser.add_argument("submitted", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    if marked_timing:
        parser.add_argument("--event-seconds", type=Fraction, help="Marked mouth closure time in the earlier video.")
        parser.add_argument("--timing-model", type=Path, help="Path to the optional pinned local landmark task.")
    args = parser.parse_args()
    try:
        timing_options = {"event_seconds": args.event_seconds, "timing_model": args.timing_model} if marked_timing else {}
        result = runner(args.earlier, args.submitted, args.output, verify=args.verify, **timing_options)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{'Verified' if args.verify else 'Saved'} comparison. Status {result['status']}.")


if __name__ == "__main__":
    main(marked_timing=True)
