"""Create or replay a video comparison using two independently supplied originals."""

from __future__ import annotations

import argparse
import base64
import json
from fractions import Fraction
from functools import partial
from pathlib import Path

from app.core.canonical import canonical_json
from app.evidence.video_reference import (
    SCHEMA,
    SCHEMA_WITH_ADDITIONAL_SOUNDTRACK,
    SCHEMA_WITH_CHANNEL_TIMING,
    SCHEMA_WITH_MARKED_SOUND_FALLBACK,
    SCHEMA_WITH_SHORT_TIMING,
    SCHEMA_WITH_SOUNDTRACK,
    SCHEMA_WITH_TIMING,
    SCHEMA_WITH_TRIM,
    compare,
)
from app.evidence.video_reference_photometric import BASIS as LOCAL_AREA_BASIS
from app.evidence.video_reference_timing import SHORT_WAVEFORM_BASIS


def _same_json_value(actual, expected):
    # A browser serializes an integral float as an integer. Preserve strict
    # comparison for integer fields, including source ticks and sample counts.
    if type(expected) is float and type(actual) in (int, float):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same_json_value(actual[k], value) for k, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_json_value(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def replay(compare, earlier: Path, submitted: Path, output: Path, *, verify: bool = False) -> dict:
    record, assets = compare(earlier, submitted)
    files = {**assets, "comparison.json": canonical_json(record) + b"\n"}
    if verify and output.is_file():
        exported = json.loads(output.read_bytes())
        if (not isinstance(exported, dict) or not isinstance(exported.get("assets"), dict)
                or not _same_json_value(exported.get("comparison"), record)
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
    options = {"include_trimmed": not verify, "include_local_area": not verify}
    short_fallback = not verify
    local_sound = not verify
    if verify:
        exported = json.loads((output if output.is_file() else output / "comparison.json").read_bytes())
        record = exported.get("comparison", exported) if isinstance(exported, dict) else None
        event_seconds = None
        if isinstance(record, dict):
            schema = record.get("schema")
            options["include_trimmed"] = schema == SCHEMA_WITH_TRIM
            channel_timing = schema == SCHEMA_WITH_CHANNEL_TIMING
            timing = record.get("timing", {})
            local_sound = schema == SCHEMA_WITH_MARKED_SOUND_FALLBACK or (
                channel_timing and (timing.get("reason") == "profile" or "changed_audio_samples" in timing)
            )
            additional = schema == SCHEMA_WITH_ADDITIONAL_SOUNDTRACK
            composite = additional or local_sound or channel_timing
            local = record.get("picture", {}).get("local_comparison", {})
            options["include_local_area"] = local.get("basis") == LOCAL_AREA_BASIS
            options.update(include_local=("local_comparison" in record.get("picture", {}) if composite
                                          else schema != SCHEMA),
                           include_soundtrack=schema == SCHEMA_WITH_SOUNDTRACK
                           or (composite and "soundtrack_alignment" in record),
                           include_additional_soundtrack=additional or (composite and "additional_soundtrack" in record),
                           include_channel_timing=channel_timing)
            if "timing" in record and schema in (SCHEMA_WITH_TIMING, SCHEMA_WITH_SHORT_TIMING, SCHEMA_WITH_SOUNDTRACK,
                                                 SCHEMA_WITH_ADDITIONAL_SOUNDTRACK, SCHEMA_WITH_MARKED_SOUND_FALLBACK,
                                                 SCHEMA_WITH_CHANNEL_TIMING):
                event_seconds = Fraction(record["timing"]["requested_earlier_time_s"])
                short_fallback = (local_sound or schema == SCHEMA_WITH_SHORT_TIMING
                                  or record["timing"].get("waveform_basis") == SHORT_WAVEFORM_BASIS)
    if event_seconds is not None:
        from app.core.config import settings

        options.update(timing_event_s=event_seconds, timing_model=timing_model or settings.video_reference_timing_model_path,
                       timing_short_fallback=short_fallback, timing_local_sound=local_sound)
    return replay(partial(compare, **options), earlier, submitted, output, verify=verify)


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
