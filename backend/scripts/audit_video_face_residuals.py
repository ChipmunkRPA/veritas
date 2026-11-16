"""Independently verify acquired assets, native picture clocks and residual anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts.audit_video_face_temporal import _command, cadence, check_file
from scripts.video_face_residual_fit import journal, require, verify
from scripts.video_face_temporal_fit import read, save, sha


def asset_clock(asset, path, mapping):
    require(path.stat().st_size == asset["size"] and sha(path) == asset["sha256"], "Publisher asset changed")
    packet = json.loads(_command("ffprobe", ["-select_streams", "v:0", "-show_entries", "stream=time_base:frame=pts",
                                             "-of", "json"], path))
    time_base = Fraction(packet["streams"][0]["time_base"])
    native = [int(row["pts"]) for row in packet["frames"]]
    require(len(mapping) == 100 and all(a < b for a, b in zip(native, native[1:])), "Invalid native picture table")
    for number, (index, stamp, tick) in enumerate(mapping):
        expected = native[0] + Fraction(10 + number, 25) / time_base
        right = bisect_left(native, expected)
        require(0 < right < len(native), "Sample time is outside the acquired source")
        nearest = min((right - 1, right), key=lambda i: (abs(native[i] - expected), i))
        require(index == nearest and stamp == native[index] and time_base == Fraction(tick),
                "A delivered picture does not cite the nearest acquired source time")
    return len(mapping)


def delivery_receipt(folder, readings, *, pool):
    manifest = folder / "manifest.json"
    cases = {row["file"]: row for row in read(manifest)["rows"] if "file" in row and row["delivery"] != "middle"}
    header, observations = journal(readings)
    require(sha(manifest) == header["manifest_sha256"] and len(cases) == len(observations)
            and set(cases) == {row["file"] for row in observations}, "Native audit uses an incomplete delivery journal")

    def check(row):
        case = cases[row["file"]]
        require(case["sha256"] == row["sha256"], "Model readout and delivery digests differ")
        frames = row["frames"]
        data = {"time_base": row["time_base"], "sha256": row["sha256"], "windows": [{
            "frames": [spot["index"] for spot in frames], "pts": [spot["pts"] for spot in frames],
            "rgb_sha256": [spot["rgb_sha256"] for spot in frames],
        }]}
        return check_file(folder / row["file"], data)

    anchors = sum(pool.map(check, observations))
    return {"deliveries": len(cases), "picture_clocks": 100 * len(cases), "rgb_anchors": anchors,
            "manifest_sha256": sha(manifest)}


def audit(study, camera, output):
    roster = study / "fresh-roster.json"
    plans = read(roster)["groups"]
    manifest = read(study / "fresh/manifest.json")
    require(manifest["roster_sha256"] == sha(roster), "Delivery source roster changed")
    rendered = {(row["group"], row["family"], row["delivery"]): row for row in manifest["rows"]}
    require(len(rendered) == len(manifest["rows"]) == len(plans) * 6, "Missing or duplicate deliveries")
    source_jobs = []
    for plan in plans:
        for family, asset in plan["files"].items():
            direct, tandem = (rendered[plan["group"], family, delivery] for delivery in ("direct", "tandem"))
            require(direct["native_map"] == tandem["native_map"]
                    and direct["source_sha256"] == tandem["source_sha256"] == asset["sha256"],
                    "Re-encodes do not share exactly the same source picture mapping")
            source_jobs.append((asset, study / "fresh-sources" / f"{plan['index']:02d}-{family}.mp4", direct["native_map"]))
    source_cadence = Counter(family for (_, family), result in cadence(manifest).items() if result["comparable"])
    require(source_cadence == {"inswapper": len(plans), "uniface": len(plans)}, "A held swap has a different picture cadence")
    with ThreadPoolExecutor(max_workers=3) as pool:
        mappings = sum(pool.map(lambda job: asset_clock(*job), source_jobs))
        held_audit = delivery_receipt(study / "fresh", study / "held.jsonl", pool=pool)
        camera_audit = delivery_receipt(camera, study / "camera.jsonl", pool=pool)
    result = {"held": held_audit, "camera": camera_audit, "publisher_assets": len(source_jobs),
              "acquired_source_mappings": mappings, "cadence_matches": dict(source_cadence), "roster_sha256": sha(roster)}
    save(output, result)
    print(json.dumps(result, sort_keys=True))


def release(result_path, audit_path, destination):
    result = read(result_path)
    audit = read(audit_path)
    result["native_audit"] = audit
    for role, panel in result["panels"].items():
        checked = audit[role]
        require(checked["manifest_sha256"] == panel["header"]["manifest_sha256"]
                and checked["deliveries"] == len(panel["scores"])
                and checked["picture_clocks"] == 100 * len(panel["scores"])
                and checked["rgb_anchors"] == sum(len(row["frames"]) for row in panel["scores"]),
                "Native evidence audit covers a different delivery set")
    require(audit["roster_sha256"] == result["freeze"]["held_plan_sha256"], "Native audit covers another source roster")
    collections = [result["freeze"]["selection"], *(panel["scores"] for panel in result["panels"].values())]
    for rows in collections:
        for row in rows:
            for sample in (row["reference"] or {}).get("samples", ()):
                sample.pop("box", None)
    recap = verify(result)
    save(destination, result)
    # The release is deterministic and contains no media, crops or feature tensors.
    print(json.dumps(recap | {"release_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="phase", required=True)
    native = actions.add_parser("native")
    for name in ("study", "camera", "output"):
        native.add_argument(name, type=Path)
    public = actions.add_parser("release")
    for name in ("result", "native_audit", "output"):
        public.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.phase == "native":
        audit(args.study, args.camera, args.output)
    else:
        release(args.result, args.native_audit, args.output)
