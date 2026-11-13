# SPDX-License-Identifier: Apache-2.0
"""Freeze and test a standalone picture return on matched webcam deliveries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import scene_return
from app.detectors.video.decode import decode_frames, decode_stream, probe_frames
from app.detectors.video.shots import ShotDetector
from scripts import video_scene_only_cut_transfer as base
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-scene-return-transfer/1"
ROLES = ("unchanged", "layout_cut", "speaker_and_layout_cut", "speaker_only", "camera_shift", "grade", "full_camera")
INTERVALS = ((33, 40), (45, 56), (60, 77), (70, 93))


def freeze(prior: Path) -> dict:
    previous = read(prior)
    source_manifest = read(base.SOURCE_PLAN)
    excluded = {row["group"] for row in previous["sources"]}
    for path, field in zip(base.EXCLUDED, ("crema_originals", "sources"), strict=True):
        excluded.update(row["source_group"] for row in read(path)[field])
    native = {}
    for path in sorted(base.NATIVE.glob("batch-*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["source_family"] == "recorded_audiovisual_speech":
                    native[row["index"]] = row
    selected = []
    for role in ("confirmation", "selection", "confidence_cal", "fusion_fit"):
        for asset in source_manifest["assets"]:
            if (asset["source_family"] != "recorded_audiovisual_speech" or asset["provisional_role"] != role
                    or asset["group"] in excluded):
                continue
            row = native[asset["index"]]
            table = row["frame_table"]
            if len(table["pts"]) < 55:
                continue
            require(row["video_clock_qualified"] and asset["sha256"] == row["input_sha256"], "Source approval differs")
            selected.append({"file": str(Path(source_manifest["media_root"]) / asset["file"]), "sha256": asset["sha256"],
                             "group": asset["group"], "frame_table": table, "rgb_sha256": row["rgb_sha256"]})
    selected = selected[:8]
    require(len({row["group"] for row in selected}) == len({row["sha256"] for row in selected}) == 8,
            "Need eight distinct genuine source actors")
    return {"schema": SCHEMA, "bindings": bindings(), "source_plan_sha256": sha(base.SOURCE_PLAN),
            "prior_sha256": sha(prior), "excluded_source_groups": len(excluded), "roles": ROLES,
            "intervals": INTERVALS, "sources": selected, "minimum_layout_localizations": 6,
            "maximum_additional_control_flags": 1}


def bindings():
    return {Path(name).name: sha(Path(name)) for name in (__file__, base.__file__, scene_return.__file__)}


def surfaces():
    y, x = np.mgrid[:base.H, :base.W]
    first = ((x // 40 ^ y // 37) % 4) * 27 + 36
    second = ((x // 34 + 2 * (y // 45)) % 4) * 27 + 36
    return tuple(np.stack((v + 17, v + 21, v + 29), axis=-1).astype(np.uint8) for v in (first, second))


def pictures(source, guest, interval, role, backgrounds, direction):
    x, y, width, height = base.TILE
    first, second = backgrounds if direction == 0 else backgrounds[::-1]
    for i in range(base.N):
        active = interval[0] <= i < interval[1]
        actor = guest if active and role in {"speaker_only", "speaker_and_layout_cut"} else source
        native = actor[0][actor[1][i]]
        if role == "full_camera":
            yield cv2.resize(native, (base.W, base.H), interpolation=cv2.INTER_LINEAR)
            continue
        picture = (second if active and role in {"layout_cut", "speaker_and_layout_cut"} else first).copy()
        picture[y:y + height, x:x + width] = native
        if active and role == "camera_shift":
            picture = cv2.warpAffine(picture, np.array([[1, 0, 61], [0, 1, -42]], dtype=np.float32),
                                     (base.W, base.H), borderMode=cv2.BORDER_REFLECT_101)
        if active and role == "grade":
            picture = np.clip(picture.astype(np.float32) * .83 + 24, 0, 255).astype(np.uint8)
        yield picture


def prepare(folder, lock):
    media = folder / "media"
    media.mkdir(mode=0o700)
    background = surfaces()
    rows = []
    for pair, interval in enumerate(lock["intervals"]):
        sources = [base.source_pictures(row) for row in lock["sources"][2 * pair:2 * pair + 2]]
        for direction, host in enumerate(sources):
            guest = sources[1 - direction]
            commitments = {}
            for role in ROLES:
                path = media / f"{pair}-{direction}-{role}.mp4"
                exact = []
                base.write_video(path, base.commit_regions(pictures(host, guest, interval, role, background, direction), exact))
                commitments[role] = exact
                rows.append({"pair": pair, "direction": direction, "role": role, "truth": interval if "cut" in role else None,
                             "name": path.name, "sha256": sha(path)})
            unchanged = commitments["unchanged"]
            for role, sides in (("layout_cut", (False, True)), ("speaker_and_layout_cut", (True, True)),
                                ("speaker_only", (True, False))):
                for side, expected in enumerate(sides):
                    changed = [i for i, (old, new) in enumerate(zip(unchanged, commitments[role], strict=True)) if old[side] != new[side]]
                    require(changed == (list(range(*interval)) if expected else []), "Authored pixel scope differs")
            print("prepared", pair + 1, direction + 1, flush=True)
    write(folder / "media.json", {"schema": SCHEMA, "freeze_sha256": sha(folder / "freeze.json"), "rows": rows})


def score(folder, lock):
    roster = read(folder / "media.json")
    require(roster["freeze_sha256"] == sha(folder / "freeze.json") and len(roster["rows"]) == 56,
            "Held delivery roster differs from freeze")
    results = []
    for row in roster["rows"]:
        path = folder / "media" / row["name"]
        require(sha(path) == row["sha256"], "Delivered bytes differ")
        table = probe_frames(path, timeout_s=20)
        require(scene_return.supported(table), "Unsupported delivered source clock")
        incumbent = ShotDetector(table)

        found = scene_return.detect(observed_frames(path, table, incumbent), table)
        require(incumbent.frames_seen == len(table), "Decoded frame roster incomplete")
        frames = []
        if found is not None:
            for frame in decode_frames(path, table, found.anchors, timeout_s=20):
                require(frame.pts_verified, "Selected source timestamp was inferred")
                frames.append({"index": frame.index, "pts": frame.pts, "sha256": hashlib.sha256(frame.rgb).hexdigest()})
            require([frame["index"] for frame in frames] == list(found.anchors), "Incomplete selected source frame readback")
        results.append({**{key: row[key] for key in ("pair", "direction", "role", "truth", "sha256")},
                        "baseline_edges": [cut["frame_index"] for cut in incumbent.boundaries],
                        "candidate": [found.start, found.stop] if found is not None else None,
                        "tiles": list(found.tiles) if found is not None else [], "anchors": frames, "time_base": table.time_base,
                        "pts_sha256": hashlib.sha256(json.dumps(table.pts).encode()).hexdigest()})
        print("scored", row["name"], results[-1]["candidate"], flush=True)
    write(folder / "results.json", {"schema": SCHEMA, "freeze_sha256": sha(folder / "freeze.json"),
                                     "media_sha256": sha(folder / "media.json"), "bindings": lock["bindings"],
                                     "opencv": cv2.__version__, "rows": results})


def observed_frames(path, table, incumbent):
    for frame in decode_stream(path, table, timeout_s=20):
        incumbent.update(frame.index, frame.rgb)
        yield frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prior", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        require(args.prior is not None, "Supply the earlier private source freeze for actor exclusions")
        args.folder.mkdir(mode=0o700)
        write(args.folder / "freeze.json", freeze(args.prior))
    else:
        lock = read(args.folder / "freeze.json")
        require(lock["bindings"] == bindings(), "Code differs from the frozen comparison")
        (prepare if args.phase == "prepare" else score)(args.folder, lock)


if __name__ == "__main__":
    main()
