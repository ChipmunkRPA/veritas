"""Freeze and prepare a new VoxCPM2 source cohort from the pinned Alania shard."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from scripts import audio_alania_roster_v178 as roster
from scripts import audio_alania_transfer_v179 as transfer
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = transfer.ROOT, transfer.BASE, transfer.PANEL
TRIAL = BASE / "audio-alania-fresh-v213"
PUBLIC = PANEL / "audio-alania-fresh-freeze-v213.json"
MODULE = "scripts.audio_alania_fresh_v213"
METHOD = ROOT / "docs/audio-alania-fresh-v213-method.md"
PLANNED = 16
IDENTITIES = ("id", "voice_id", "text_doc_id")
FINGERPRINTS = ("source_sha256", "pcm_sha256", "wav_sha256")
read, write, sha, require = transfer.read, transfer.write, transfer.sha, transfer.require


def eligible_rows(rows, previous):
    excluded = {key: {row[key] for row in previous} for key in IDENTITIES}
    previous_locations = {(row["row_group"], row["row_offset"]) for row in previous}
    return [
        row
        for row in rows
        if roster.eligible(row)
        and row["duration"] >= 3.0
        and (row["row_group"], row["row_offset"]) not in previous_locations
        and all(row[key] not in excluded[key] for key in IDENTITIES)
    ]


def context():
    previous, publisher = transfer.roster_commitments()
    _, previous_prep = transfer.verified_preparation(previous)
    shard = roster.source_identity()
    with pq.ParquetFile(shard) as file:
        rows = []
        for group in range(file.num_row_groups):
            table = file.read_row_group(group, columns=list(roster.COLUMNS), use_threads=False)
            require(
                tuple(table.schema.names) == roster.COLUMNS, "Only committed metadata columns may be read"
            )
            rows.extend(
                {**row, "row_group": group, "row_offset": i} for i, row in enumerate(table.to_pylist())
            )
    eligible = eligible_rows(rows, previous)
    require(len({row["id"] for row in eligible}) == len(eligible), "Eligible row IDs must be unique")
    selected = roster.select_rows(eligible, limit=PLANNED)
    decoder = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(decoder.values()), "Production audio decoder tools required")
    files = (
        Path(__file__),
        ROOT / "backend/tests/test_audio_alania_fresh_v213.py",
        METHOD,
        Path(roster.__file__),
        Path(transfer.__file__),
        Path(transfer.runtime.__file__),
        ROOT / "backend/app/detectors/audio/decode.py",
        ROOT / "backend/app/detectors/video/decode.py",
    )
    return {
        "schema": "chipmunk.audio-alania-fresh-freeze/213",
        "publisher_revision": roster.REVISION,
        "shard_sha256": roster.SHARD_SHA,
        "publisher_records_sha256": publisher["publisher_records_sha256"],
        "prior_roster_freeze_sha256": sha(roster.PUBLIC),
        "prior_preparation_sha256": previous_prep["private_preparation_sha256"],
        "prior_preparation_profile_sha256": previous_prep["resource_profile_sha256"],
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "decoder_tool_sha256": {name: sha(Path(path).resolve()) for name, path in decoder.items()},
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "eligible_rows": len(eligible),
        "eligible_voice_groups": len({row["voice_id"] for row in eligible}),
        "eligible_text_documents": len({row["text_doc_id"] for row in eligible}),
        "excluded_previous_slots": len(previous),
        "planned_cases": PLANNED,
        "planned_by_render_mode": dict(Counter(row["render_mode"] for row in selected)),
        "language": "tr",
        "license": "CC-BY-4.0",
        "metadata_duration_bounds_seconds": [3, 4],
        "model_calls_permitted": False,
    }, selected


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve prior preparation attempts")
    frozen, selected = context()
    TRIAL.mkdir(mode=0o700)
    path = TRIAL / "private-roster.json"
    write(path, selected)
    frozen["private_roster_sha256"] = sha(path)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return frozen


def verify():
    frozen, selected = context()
    path = TRIAL / "private-roster.json"
    frozen["private_roster_sha256"] = sha(path)
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == frozen and read(path) == selected, "Roster changed"
    )
    return selected


def previous_fingerprints():
    previous, _ = transfer.roster_commitments()
    prior, _ = transfer.verified_preparation(previous)
    return {key: {row[key] for row in prior if row["status"] == "qualified"} for key in FINGERPRINTS}


def qualification_reason(row, fingerprints):
    if row["status"] != "qualified":
        return row["reason"]
    if row["samples"] < 3 * transfer.SAMPLE_RATE:
        return "shorter_than_three_seconds_after_decode"
    if any(row[key] in fingerprints[key] for key in FINGERPRINTS):
        return "overlap_with_prior_or_selected_audio"
    return None


def prepared_dir():
    return TRIAL / "bind-00/output/prepared"


def worker(output):
    fingerprints = previous_fingerprints()

    def prepare_case(row, embedded, scratch, destination):
        provisional = scratch.with_suffix(".wav")
        result = transfer.prepare_case(row, embedded, scratch, provisional)
        reason = qualification_reason(result, fingerprints)
        if reason is not None:
            provisional.unlink(missing_ok=True)
            return {
                "slot": row["slot"],
                "render_mode": row["render_mode"],
                "status": "unavailable",
                "reason": reason,
            }
        require(not destination.exists(), "Prepared waveform output already exists")
        provisional.rename(destination)
        for key in FINGERPRINTS:
            fingerprints[key].add(result[key])
        return result

    scoped(
        transfer.prepare_worker,
        {
            "exact_output": scoped(transfer.exact_output, {"TRIAL": TRIAL}),
            "verify": verify,
            "PUBLIC": PUBLIC,
            "prepare_case": prepare_case,
            "prepared_dir": prepared_dir,
        },
    )(output)


def run():
    scoped(transfer.run, {"TRIAL": TRIAL, "verify": verify, "MODULE": MODULE, "__file__": __file__})(
        "prepare"
    )


def verified_preparation():
    pcm = scoped(transfer.read_pcm, {"prepared_dir": prepared_dir})
    rows, preparation = scoped(
        transfer.verified_preparation,
        {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "__file__": __file__, "read_pcm": pcm},
    )(verify())
    fingerprints = previous_fingerprints()
    for row in rows:
        if row["status"] == "qualified":
            require(
                qualification_reason(row, fingerprints) is None, "Qualified cohort contains duplicate audio"
            )
            for key in FINGERPRINTS:
                fingerprints[key].add(row[key])
    return rows, {
        **preparation,
        "schema": "chipmunk.audio-alania-fresh-preparation/213",
        "qualified": sum(row["status"] == "qualified" for row in rows),
        "unavailable": sum(row["status"] == "unavailable" for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "prepare", "prepare-worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require(
        (args.command == "prepare-worker") == (args.output is not None),
        "Only the guarded worker takes an output",
    )
    result = {
        "preflight": lambda: context()[0],
        "freeze": freeze,
        "prepare": run,
        "prepare-worker": lambda: worker(args.output),
        "readout": lambda: verified_preparation()[1],
    }[args.command]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
