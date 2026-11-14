# SPDX-License-Identifier: Apache-2.0
"""Evaluate a fixed 120-frame face proposal on distinct 720p30 H.264 deliveries."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts import review_video_brief_face_avc_confirmation as receipt
from scripts import video_brief_face_avc_confirmation as previous
from scripts import video_face_720p30_transfer as hd
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-brief-face-avc-clock-confirmation/1"
PRIOR_SHA = "71a506e5405b57811fe26c3891cdf54061e47e6803571d0fdb1fc2cc8b91d7c2"
PLACEMENTS = (62, 69, 76, 83, 90, 97, 104, 111)
RATES = (30, 30, Fraction(30000, 1001), Fraction(30000, 1001), 25, 25, 50, 50)


def eligible_rate(table):
    times = hd.relative(table)
    if len(times) < 2 or times[-1] < Fraction(119, 30):
        return None
    step = times[1]
    if step <= 0 or table.rotation or table.truncated or table.interpolated:
        return None
    if any(time != index * step for index, time in enumerate(times)):
        return None
    return 1 / step


def assignments(table):
    times = hd.relative(table)
    indices = []
    for output in range(hd.N):
        pos = Fraction(output, hd.RATE)
        bound = bisect.bisect_left(times, pos)
        before = max(0, bound - 1)
        after = min(len(times) - 1, bound)
        indices.append(min((before, after), key=lambda i: (abs(times[i] - pos), i)))
    return indices


def selection(prior_folders):
    from app.detectors.video.decode import FrameTable
    from scripts import video_face_short_boundary_confirmation as boundary
    from scripts.video_chroma_boundary_transfer import source_pin
    from scripts.video_native_sources_v92 import ARCHIVE

    catalog = boundary.metadata()
    qualification = {row["group"]: row["qualification"] for row in read(previous.QUALIFIED)["assets"]
                     if row["kind"] == "original_videos"}
    inventory = {item["index"]: item for path in sorted(previous.NATIVE.glob("group-*-complete.json"))
                 if (item := read(path))}
    commitments = {**previous.PRIOR, "second_avc": PRIOR_SHA}
    source_paths = (previous.DATA / "video-face-two-returns-hd-2026-11-12/freeze.json",
                    *(folder / "freeze.json" for folder in prior_folders))
    used_indices, urls, donors, components = set(), set(), set(), set()

    def mark(index):
        group = inventory[index]["group"]
        used_indices.add(index)
        urls.add(catalog[group]["url"])
        donors.update(catalog[group]["donors"])
        components.add(qualification[group]["component"])

    for (name, expected), path in zip(commitments.items(), source_paths, strict=True):
        require(sha(path) == expected, f"Prior {name} source plan differs")
        saved = read(path)
        for key in ("sources", "held", "development"):
            for row in saved.get(key, ()):
                mark(row["index"])

    candidates = sorted(inventory.values(), key=lambda item: hashlib.sha256(f"{SCHEMA}/{item['group']}".encode()).digest())
    selected = []
    for slot, (rate, start) in enumerate(zip(RATES, PLACEMENTS, strict=True)):
        family = "swap_a" if slot % 2 == 0 else "swap_b"
        for item in candidates:
            index, group = item["index"], item["group"]
            group_meta, qualified = catalog[group], qualification[group]
            if (index in used_indices or group_meta["url"] in urls or donors & group_meta["donors"]
                    or qualified["component"] in components or "child" in group_meta["tags"]
                    or not qualified["eligible"] or qualified["historical_rgb_connection"]):
                continue
            cases = {case["condition"]: case for case in item["cases"]}
            table = FrameTable.from_json(cases["original"]["frame_table"])
            if (table.width, table.height) not in ((1280, 720), (1920, 1080)) or eligible_rate(table) != rate:
                continue
            if FrameTable.from_json(cases[family]["frame_table"]) != table:
                continue
            native = assignments(table)
            if not set(native[start:start + 6]) <= set(cases[family]["actual_changed_frames"]):
                continue
            selected.append({**source_pin(index, ARCHIVE), "interval": [start, start + 6], "family": family,
                             "native_rate": str(rate), "assignments": native, "prior_project_role": item["role"]})
            mark(index)
            break
        require(len(selected) == slot + 1, f"No separated source for delivery slot {slot + 1} ({rate} fps)")
    return selected, commitments


def freeze(prior_folders):
    from unittest.mock import patch

    with patch.object(previous, "source_plan", return_value=()):
        lock = previous.freeze(prior_folders[:2])
    sources, commitments = selection(prior_folders)
    return {**lock, "schema": SCHEMA, "prior_sha256": commitments, "sources": sources,
            "placements": PLACEMENTS, "delivery_rate": hd.RATE, "delivery_geometry": hd.SIZE,
            "method_sha256": {**lock["method_sha256"], **{p.name: sha(p) for p in (Path(__file__), Path(receipt.__file__))}}}


def native(source, role, output_indices):
    import cv2

    from app.detectors.video.decode import decode_frames, probe_frames
    from scripts.video_native_sources_v92 import ARCHIVE

    record = read(previous.NATIVE / f"group-{source['index']:03d}-complete.json")
    case = next(row for row in record["cases"] if row["condition"] == role)
    path = ARCHIVE / "video-native-inserts-v95" / f"group-{source['index']:03d}" / case["file"]
    require(sha(path) == case["sha256"], "Native frame source differs")
    table = probe_frames(path, timeout_s=30)
    require(table.to_json() == case["frame_table"] and list(source["assignments"]) == assignments(table),
            "Native frame clock differs")
    positions = {i: source["assignments"][i] for i in output_indices}
    inputs = {}
    for frame in decode_frames(path, table, sorted(set(positions.values())), timeout_s=45):
        require(frame.pts_verified and frame.pts == table.pts[frame.index]
                and hashlib.sha256(frame.rgb).hexdigest() == case["expected_rgb_sha256"][frame.index],
                "Native frame at assigned clock differs")
        inputs[frame.index] = cv2.resize(frame.rgb, hd.SIZE, interpolation=cv2.INTER_AREA)
    return {output: inputs[index] for output, index in positions.items()}


def write_group(folder, source, other, group):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    left, right = source["interval"]
    positions = range(left, right)
    host = native(source, "original", range(hd.N))
    edited = {
        "swap": native(source, source["family"], positions),
        "lighting": {i: np.clip(host[i].astype(np.float32) * 1.15, 0, 255).astype(np.uint8) for i in positions},
        "same_recording_jump": {i: host[i - 18] for i in positions},
        "cross_recording_cut": native(other, "original", positions),
    }
    rows = []
    for role in previous.ROLES:
        path = folder / f"g{group:02}-{role}.mp4"
        replacements = edited.get(role, {})
        if role.startswith("tandem_"):
            hd.authored(path, source=folder / f"g{group:02}-{role.removeprefix('tandem_')}.mp4", crf=35)
        else:
            require(all(not np.array_equal(host[i], rgb) for i, rgb in replacements.items()), "Intervention equals the source")
            hd.authored(path, pictures=(replacements.get(i, host[i]) for i in range(hd.N)))
        row = hd.describe(path, group, role, [source["interval"]] if role in previous.POSITIVES else [])
        rows.append({**row, "authored_rgb_sha256": {str(i): hashlib.sha256(rgb).hexdigest()
                                                   for i, rgb in replacements.items()}})
    return rows


def author(folder, lock, jobs):
    media = folder / "media"
    media.mkdir(mode=0o700)
    sources = lock["sources"]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        authored = [pool.submit(write_group, media, source, sources[slot ^ 1], slot + 1)
                    for slot, source in enumerate(sources)]
        for slot, result in enumerate(authored, 1):
            rows.extend(result.result())
            print(f"Authored group {slot}", flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == len(sources) * len(previous.ROLES),
            "Incomplete or repeated deliveries")
    nbytes = sum(path.stat().st_size for path in media.iterdir())
    require(nbytes <= 160 * 1024**2, "Local media budget exceeded")
    return {"freeze_sha256": sha(folder / "freeze.json"), "rows": rows, "media_bytes": nbytes}


def score_section(folder):
    from scripts import screen_video_native_background_challenge as worker

    state = worker.configure(folder)
    os.environ.update(previous.BASE_FLAGS)
    previous.score(folder, read(folder / "freeze.json"), state)


def score(folder, jobs):
    manifest = read(folder / "media.json")
    commands, sections = [], []
    for job in range(jobs):
        section = folder / f"score-part-{job}"
        section.mkdir(mode=0o700)
        (section / "freeze.json").symlink_to(folder / "freeze.json")
        (section / "media").symlink_to(folder / "media")
        rows = [row for row in manifest["rows"] if (row["group"] - 1) % jobs == job]
        write(section / "media.json", {**manifest, "rows": rows})
        sections.append(section)
        commands.append([sys.executable, "-m", __spec__.name, "score-section", str(section)])
    children = [subprocess.Popen(cmd) for cmd in commands]
    try:
        require(not any(child.wait() for child in children), "A production comparison subprocess failed")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
    rows = {}
    for section in sections:
        for line in (section / "scored.jsonl").read_text().splitlines(keepends=True):
            row = json.loads(line)
            key = (row["group"], row["condition"])
            require(key not in rows, "Repeated measurement from scoring subprocess")
            rows[key] = line
    require(len(rows) == len(manifest["rows"]), "Incomplete scoring subprocess outputs")
    with (folder / "scored.jsonl").open("x") as output:
        for item in manifest["rows"]:
            output.write(rows[item["group"], item["condition"]])


def verify(folder):
    manifest = read(folder / "media.json")
    lock = read(folder / "freeze.json")
    rows = [json.loads(line) for line in (folder / "scored.jsonl").read_text().splitlines()]
    require(lock["schema"] == SCHEMA and lock["point"] == .10 and lock["gates"] == previous.GATES
            and lock["placements"] == list(PLACEMENTS) and manifest["freeze_sha256"] == sha(folder / "freeze.json"),
            "Scored method or delivery differs from the prospective protocol")
    outcome = receipt.summarize(rows, placements=PLACEMENTS)
    replayed_anchors = 0
    for row, recorded in zip(rows, manifest["rows"], strict=True):
        require(all(row[key] == recorded[key] for key in ("group", "condition", "truth", "sha256")), "Scored delivery differs")
        replayed_anchors += receipt.replay_native(folder / "media" / recorded["file"], row, recorded["rgb_sha256"])
    return {**outcome, "exact_anchor_replays": replayed_anchors, "replayed_frames": hd.N * len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score", "score-section", "verify"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--jobs", choices=(1, 2, 3, 4), type=int, default=1)
    defaults = ("chipmunk-eightsecond-face-20261112", "chipmunk-video-short-carrier-20261112-v2",
                "chipmunk-brief-avc-confirmation-final-20261113")
    parser.add_argument("--prior", nargs=3, type=Path, default=tuple(Path("/private/tmp") / name for name in defaults))
    args = parser.parse_args()
    args.run = args.run.resolve()
    if args.phase == "score-section":
        score_section(args.run)
        return
    if args.phase == "verify":
        print(json.dumps(verify(args.run), sort_keys=True), flush=True)
        return
    lock = freeze(args.prior)
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
        print([(item["index"], item["native_rate"], item["interval"]) for item in lock["sources"]], flush=True)
        return
    require(read(args.run / "freeze.json") == json.loads(json.dumps(lock)), "Frozen method or source plan differs")
    if args.phase == "prepare":
        write(args.run / "media.json", author(args.run, lock, args.jobs))
    else:
        score(args.run, args.jobs)


if __name__ == "__main__":
    main()
