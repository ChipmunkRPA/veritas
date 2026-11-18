# SPDX-License-Identifier: Apache-2.0
"""Check transport of bounded earlier-copy waveform locations through Ogg Opus."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import audio_standalone_dual_view as source

PANELS = Path(__file__).resolve().parents[2] / "docs" / "evaluation-panels"
PLAN = PANELS / "audio-reference-waveform-opus-freeze-2026-11-18.json"
INPUTS = PANELS / "audio-standalone-dual-view-2026-11-18.json.gz"
SOURCE_PLAN = PANELS / "audio-standalone-dual-view-plan-2026-11-18.json"
SOURCES = PANELS / "audio-standalone-dual-view-sources-2026-11-18.json"
PLAN_SHA = "d084dae885c1b2c6577956c78b00d1c742842a709b7db1368f8a53c63c0bc8b2"
INPUT_SHA = "da1b3fc09b32a72ad958e3040f3fff5f6ffb6a46d26da2bae5859881b4a8ab42"
WAVEFORM_SHA = "8946b5247076b287f6b2badeba755791ca3a6eabe746b4a2939323974bc12675"
SCHEMA = "chipmunk.opus-waveform-transfer/1"
PROFILES = {"64-cbr": ("64k", "off"), "24-vbr": ("24k", "on")}
PRESENTED = 288_000
PACKET = 960
UNASSESSED = frozenset({"equal_duration_required", "insufficient_matching_waveform"})


def input_key(row):
    return row["reader"], row["level"], row["kind"]


def load_panel():
    if source.file_hash(PLAN) != PLAN_SHA or source.file_hash(INPUTS) != INPUT_SHA:
        raise ValueError("The frozen plan and prior input receipt must match exactly")
    _, selected = source.verify_plan(SOURCE_PLAN, SOURCES)
    previous = {input_key(row): row for row in source.read(INPUTS)["incumbent"]["rows"] if row["kind"] != "room_only"}
    if len(previous) != 96:
        raise ValueError("Expected 16 readers with both levels and all three paired roles")
    return selected, previous


def to_opus(samples, path, profile):
    bitrate, vbr = PROFILES[profile]
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
               "-i", "pipe:0", "-ar", "48000", "-threads", "1", "-c:a", "libopus", "-application", "voip",
               "-frame_duration", "20", "-b:a", bitrate, "-vbr", vbr, "-map_metadata", "-1", "-fflags", "+bitexact",
               str(path)]
    subprocess.run(command, input=samples.tobytes(), check=True, capture_output=True, timeout=15)


def measure(panel, previous, input_root, rooms, scratch):
    import numpy as np

    from app.detectors.audio.ogg_opus_reference import native_clock
    from scripts import audio_reference_waveform_aac_transfer as shared
    from scripts import audio_reference_waveform_transfer as waveform

    if source.file_hash(Path(waveform.waveform.__file__)) != WAVEFORM_SHA:
        raise ValueError("The waveform method must match the frozen production implementation")
    cleaned = {}
    for row in panel["pairs"]:
        if row["split"] == "held":
            path = input_root / "sine" / f'{row["id"]:03}-original.wav'
            cleaned[row["id"]] = source.crop(path, row["files"]["original"], row["native_start"]).astype("<i2")
    rows = []
    for metadata, samples in source.samples_for("held", input_root, panel, rooms):
        if metadata["kind"] == "room_only":
            continue
        earlier = cleaned[metadata["id"]]
        if source.digest(samples.tobytes()) != previous[input_key(metadata)]["input_sha256"]:
            raise ValueError("A generated mix differs from the exact held PCM")
        row = {**metadata, "lossless": waveform.measurement(earlier, samples), "opus": {}}
        path = scratch / "submitted.opus"
        for profile in PROFILES:
            to_opus(samples, path, profile)
            clock = native_clock(path)
            track = shared.track(path)[0]
            expected = {"relative_start": "0", "samples": source.SAMPLES, "source_pts": 0}
            if track[0] != "measured" or any(track[2][key] != number for key, number in expected.items()):
                raise ValueError("The comparison decoder did not reproduce the full presented clock")
            decoded = shared.pcm(track)
            if len(decoded) != source.SAMPLES or int(np.count_nonzero(decoded)) == 0:
                raise ValueError("The comparison decoder did not produce the bounded mono signal")
            row["opus"][profile] = {"clock": clock, "encoded_sha256": source.file_hash(path),
                                    "reading": waveform.measurement(earlier, decoded)}
        rows.append(row)
        if len(rows) % 16 == 0:
            print(f"Verified presented samples and compared {len(rows)}/96 source pairs", flush=True)
    return rows


def assessed(reading, target):
    selected = reading["selected"]
    if (reading["samples"] != [source.SAMPLES, source.SAMPLES]
            or (reading["status"] == "localized_difference") != bool(selected)):
        raise ValueError("Inconsistent waveform result")
    if selected is None:
        return False, False
    left, right = selected["span"]
    if not 0 <= left < right <= source.SAMPLES:
        raise ValueError("Waveform range falls outside the fixed clock")
    if not target:
        return True, False
    a, b = target
    overlap = max(0, min(right, b) - max(left, a))
    return True, 2 * overlap >= b - a and 10 * overlap >= 3 * (max(right, b) - min(left, a))


def review(rows, previous):
    original = dict(previous)
    cells = {profile: {} for profile in PROFILES}
    false_groups = {profile: set() for profile in PROFILES}
    for row in rows:
        known = original.pop(input_key(row))
        raw = row["lossless"]
        if (row["truth"] != known["truth"] or raw["sha256"][1] != known["input_sha256"]
                or sorted(row["opus"]) != sorted(PROFILES)):
            raise ValueError("The retained mix, edit label, or profile set changed")
        base_flag, base_match = assessed(raw, row["truth"])
        for profile, item in row["opus"].items():
            clock, coded = item["clock"], item["reading"]
            if (any(clock[key] != value for key, value in {"sample_rate": 48000, "channels": 1, "packets": 301,
                        "presentation_start_pts": 0, "presentation_end_exclusive_pts": PRESENTED}.items())
                    or clock["pre_skip"] + PRESENTED != clock["end_granule"]
                    or clock["end_granule"] + clock["end_trim"] != PACKET * clock["packets"]
                    or coded["sha256"][0] != raw["sha256"][0]):
                raise ValueError("Codec evidence does not retain the paired earlier bytes and verified native clock")
            flag, match = assessed(coded, row["truth"])
            cell = cells[profile].setdefault(f'{row["level"]}/{row["kind"]}', Counter())
            cell.update(total=1, lossless_matches=base_match, opus_matches=match, preserved=base_match and match,
                        lost=base_match and not match, gained=match and not base_match,
                        lossless_flags=base_flag, opus_flags=flag, new_flags=flag and not base_flag,
                        lossless_unassessed=raw["status"] in UNASSESSED, opus_unassessed=coded["status"] in UNASSESSED)
            if row["kind"] == "original" and flag and not base_flag:
                false_groups[profile].add(row["reader"])
    if original:
        raise ValueError("Results do not cover the complete panel")
    for profile, table in cells.items():
        edits = [table[f"{level}/edit"] for level in source.LEVELS]
        base = sum(row["lossless_matches"] for row in edits)
        preserved = sum(row["preserved"] for row in edits)
        ordinary = sum(table[f"{level}/original"]["new_flags"] for level in source.LEVELS)
        splice = sum(table[f"{level}/genuine_splice"]["new_flags"] for level in source.LEVELS)
        table["gate_passed"] = (base >= 8 and preserved * 100 >= 85 * base
                               and all(row["lossless_matches"] - row["opus_matches"] <= 2 for row in edits)
                               and ordinary <= 2 and len(false_groups[profile]) <= 1 and splice <= 2)
        table["ordinary_readers_newly_flagged"] = len(false_groups[profile])
        rates = [row["opus"][profile]["clock"]["packet_payload_bit_rate"] for row in rows]
        table["payload_rate_bounds"] = min(rates), max(rates)
    return cells


def prepare_fixtures(output, panel, previous, input_root, rooms):
    chosen_id = 18
    row, = [item for item in panel["pairs"] if item["id"] == chosen_id]
    path = input_root / "sine" / f"{chosen_id:03}-original.wav"
    earlier = source.crop(path, row["files"]["original"], row["native_start"]).astype("<i2")
    output.mkdir(mode=0o700)
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
               "-c:a", "flac", "-sample_fmt", "s16", "-map_metadata", "-1", "-fflags", "+bitexact", str(output / "earlier.flac")]
    subprocess.run(command, input=earlier.tobytes(), check=True, capture_output=True, timeout=15)
    roles = {"edit": "positive", "original": "ordinary"}
    for metadata, samples in source.samples_for("held", input_root, panel, rooms):
        if metadata.get("id") == chosen_id and metadata["level"] == -5 and metadata["kind"] in roles:
            if source.digest(samples.tobytes()) != previous[input_key(metadata)]["input_sha256"]:
                raise ValueError("The fixture differs from the exact held PCM")
            to_opus(samples, output / f'{roles[metadata["kind"]]}.opus', "64-cbr")
            if metadata["kind"] == "edit":
                to_opus(samples, output / "rejected.opus", "24-vbr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("run", "review"):
        command = modes.add_parser(name)
        command.add_argument("receipt", type=Path)
        if name == "run":
            command.add_argument("sources", type=Path)
            command.add_argument("rooms", type=Path)
    fixture = modes.add_parser("fixtures")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("sources", type=Path)
    fixture.add_argument("rooms", type=Path)
    args = parser.parse_args()
    panel, previous = load_panel()
    if args.mode == "fixtures":
        prepare_fixtures(args.output, panel, previous, args.sources, args.rooms)
        return
    commitments = {"schema": SCHEMA, "plan_sha256": PLAN_SHA, "previous_delivery_sha256": INPUT_SHA,
                   "waveform_implementation_sha256": WAVEFORM_SHA}
    if args.mode == "run":
        if args.receipt.exists():
            parser.error("Choose an unused receipt path")
        with TemporaryDirectory(prefix="chipmunk-opus-reference-") as temporary:
            rows = measure(panel, previous, args.sources, args.rooms, Path(temporary))
        summary = json.loads(json.dumps(review(rows, previous)))
        source.save(args.receipt, commitments | {"rows": rows, "summary": summary})
    else:
        receipt = source.read(args.receipt)
        summary = json.loads(json.dumps(review(receipt["rows"], previous)))
        if summary != receipt["summary"] or any(receipt[key] != expected for key, expected in commitments.items()):
            raise ValueError("The recount differs from the frozen receipt")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
