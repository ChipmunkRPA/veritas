# SPDX-License-Identifier: MIT
"""Independently check delivery clocks, RGB hashes and source cadence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts.video_face_temporal_fit import read, save


def _source_times(mapping):
    original = [Fraction(tick) * Fraction(timebase) for _, tick, timebase in mapping]
    assert len(original) == 100
    return [time - original[0] for time in original]


def cadence(manifest):
    initial = {(row["group"], row["family"]): row for row in manifest["rows"] if row.get("delivery") == "direct"}
    paired = {}
    for (group, family), row in initial.items():
        if family == "ordinary" or (group, "ordinary") not in initial:
            continue
        original = _source_times(initial[group, "ordinary"]["native_map"])
        swapped = _source_times(row["native_map"])
        delta = max(abs(a - b) for a, b in zip(original, swapped, strict=True))
        repeat_difference = sum((a == b) != (x == y)
                                for a, b, x, y in zip(original, original[1:], swapped, swapped[1:]))
        paired[group, family] = {"relative_time_difference_max": str(delta), "repetition_pattern_differences": repeat_difference,
                                 "comparable": delta <= Fraction(1, 250) and repeat_difference == 0}
    return paired


def _command(program, options, media):
    return subprocess.run([program, "-v", "error", "-protocol_whitelist", "file", *options, str(media)],
                          capture_output=True, check=True, timeout=60, text=True).stdout


def check_file(path, data):
    clock = json.loads(_command("ffprobe", ["-select_streams", "v:0", "-show_entries",
                                            "stream=time_base,width,height:frame=pts", "-of", "json"], path))
    stream, = clock["streams"]
    assert (stream["width"], stream["height"]) == (640, 360)
    timebase = Fraction(stream["time_base"])
    assert [Fraction(frame["pts"]) * timebase for frame in clock["frames"]] == [Fraction(i, 25) for i in range(100)], path
    # Produce RGB hashes without the app's decoder, cropping or evidence code.
    pixels = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(path),
                             "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-f", "framehash", "-hash", "sha256", "-"],
                            capture_output=True, check=True, text=True, timeout=60).stdout
    decoded = [line.split(",") for line in pixels.splitlines() if line and not line.startswith("#")]
    assert len(decoded) == 100 and all(int(row[4]) == 640 * 360 * 3 for row in decoded), path
    hashes = [line[-1].strip() for line in decoded]
    anchors = 0
    for window in data["windows"]:
        for index, mark, digest in zip(window["frames"], window["pts"], window["rgb_sha256"], strict=True):
            assert Fraction(mark) * Fraction(data["time_base"]) == Fraction(index, 25)
            assert hashes[index] == digest, (path.name, index)
            anchors += 1
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == data["sha256"]
    return anchors


def audit(media, development, held, destination):
    all_data = [read(development), read(held)]
    manifest = read(media / "manifest.json")
    assert len({data["manifest_sha256"] for data in all_data}) == 1
    rows = [row for data in all_data for row in data["rows"] if "file" in row]
    assert len({row["file"] for row in rows}) == len(rows) == sum("file" in row for row in manifest["rows"])
    with ThreadPoolExecutor(max_workers=3) as pool:
        found = list(pool.map(lambda row: check_file(media / row["file"], row), rows))
    comparison = cadence(manifest)
    subsets = Counter()
    for row in manifest["rows"]:
        if row.get("delivery") == "direct" and row["family"] != "ordinary":
            case = comparison.get((row["group"], row["family"]))
            subsets[row["role"], row["family"], "comparable" if case and case["comparable"] else "different"] += 1
    output = {"deliveries": len(rows), "picture_clocks": len(rows) * 100, "face_window_anchors": sum(found),
              "source_cadence": [{**row, "group": key[0], "family": key[1]} for key, row in sorted(comparison.items())],
              "source_cadence_summary": {"/".join(key): value for key, value in sorted(subsets.items())}}
    save(destination, output)
    print(f"Verified {len(rows)} deliveries, {len(rows) * 100} exact picture clocks and {sum(found)} face-window RGB hashes.")
    print(output["source_cadence_summary"])


def public_roster(private):
    return {key: value for key, value in private.items() if key != "groups"} | {
        "groups": [{key: row[key] for key in ("group", "role", "index")} | {
            "files": {family: {key: asset[key] for key in ("sha256", "size")} for family, asset in row["files"].items()},
        } for row in private["groups"]],
    }


def restore_roster(record, publisher, destination):
    data = read(publisher)
    assert data["sha"] == record["revision"]
    lookup = {(row["rfilename"].split("/")[0], row["lfs"]["sha256"], row["size"]): row["rfilename"]
              for row in data["siblings"] if row.get("lfs") and row["size"] == row["lfs"]["size"]
              and row["rfilename"].endswith(".mp4")}
    groups = []
    for group in record["groups"]:
        paths = {}
        for family, asset in group["files"].items():
            prefix = "original_videos" if family == "ordinary" else family
            paths[family] = asset | {"path": lookup[prefix, asset["sha256"], asset["size"]]}
        groups.append(group | {"files": paths})
    save(destination, record | {"groups": groups})


def release(result, roster, audited, output):
    data = read(result)
    for panel in ("development", "held"):
        for row in data[panel]["rows"]:
            for window in row.get("windows", ()):
                window.pop("boxes")
    for reference in data["reference"]:
        for sample in (reference["review"] or {}).get("samples", ()):
            sample.pop("box")
    save(output, data | {"acquisition": public_roster(read(roster)), "native_audit": read(audited)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="action", required=True)
    native = phases.add_parser("pixels")
    for name in ("deliveries", "development", "held", "output"):
        native.add_argument(name, type=Path)
    public = phases.add_parser("release")
    for name in ("result", "roster", "audited", "output"):
        public.add_argument(name, type=Path)
    replay = phases.add_parser("restore")
    for name in ("receipt", "publisher", "output"):
        replay.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "pixels":
        audit(args.deliveries, args.development, args.held, args.output)
    elif args.action == "release":
        release(args.result, args.roster, args.audited, args.output)
    else:
        restore_roster(read(args.receipt)["acquisition"], args.publisher, args.output)
