# SPDX-License-Identifier: Apache-2.0
"""Confirm the fixed brief face proposal on a separate 720p30 AVC panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts import video_brief_face_120_carrier as brief
from scripts import video_face_720p30_transfer as hd
from scripts import video_face_two_returns_hd as dual
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
NATIVE = DATA / "video-native-inserts-v95"
QUALIFIED = DATA / "video-native-sources-v92/qualified-plan.json"
PRIOR = {
    "four_second": "6eddf92858137633e64373f372d3764a42d1fca94d3679e689c92e66ec8cdb2a",
    "eight_second_hevc": "1ee86ff12dc7a3099b77072e4b558353b91ed7d8af410b9784d1f04319f9ff26",
    "eight_second_avc_vp9": "1aeaeee6c8dd7468e306fc0116678b8ffddc7d29f0800a14492e86e9ca7f5661",
}
ROLES = ("original", "tandem_original", "swap", "tandem_swap", "lighting", "same_recording_jump", "cross_recording_cut")
POSITIVES = {"swap", "tandem_swap"}
PLACEMENTS = (63, 72, 81, 90, 99, 108)
GROUPS = len(PLACEMENTS)
SCHEMA = "chipmunk.video-brief-face-avc-confirmation/1"
GATES = {
    "minimum_useful_additions": 4,
    "minimum_distinct_gained_groups": 3,
    "minimum_exact_additions": 2,
    "maximum_new_ordinary_flags": 1,
    "maximum_new_ordinary_cut_flags": 0,
    "maximum_new_wrong_edited_ranges": 1,
    "minimum_candidate_coverage": 0.75,
}
BASE_FLAGS = {
    "VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW": "true",
    "VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW": "true",
    "VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT": "true",
    "VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT": "true",
    "VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW": "false",
    "VERITAS_VIDEO_FACE_SHORT_WINDOW_REVIEW": "false",
    "VERITAS_VIDEO_FACE_SHORT_CARRIER_REVIEW": "false",
}


def source_plan(prior_dirs):
    from app.detectors.video.decode import FrameTable
    from scripts import video_face_short_boundary_confirmation as boundary
    from scripts.video_chroma_boundary_transfer import source_pin
    from scripts.video_native_sources_v92 import ARCHIVE

    metadata = boundary.metadata()
    original = {row["group"]: row for row in read(QUALIFIED)["assets"] if row["kind"] == "original_videos"}
    inventory = {row["index"]: row for path in sorted(NATIVE.glob("group-*-complete.json")) if (row := read(path))}
    older = set()
    files = [DATA / "video-face-two-returns-hd-2026-11-12/freeze.json", *[directory / "freeze.json" for directory in prior_dirs]]
    for (name, digest), path in zip(PRIOR.items(), files, strict=True):
        require(sha(path) == digest, f"Retained {name} source plan differs")
        old = read(path)
        for key in ("sources", "development", "held"):
            older.update(row["index"] for row in old.get(key, []))
    links = {metadata[inventory[i]["group"]]["url"] for i in older}
    donors = set.union(*(metadata[inventory[i]["group"]]["donors"] for i in older))
    components = {original[inventory[i]["group"]]["qualification"]["component"] for i in older}
    selected = []
    for row in sorted(inventory.values(), key=lambda row: hashlib.sha256(f"{SCHEMA}/{row['group']}".encode()).digest()):
        group = row["group"]
        meta = metadata[group]
        qualified = original[group]["qualification"]
        if row["index"] in older or meta["url"] in links or meta["donors"] & donors or qualified["component"] in components:
            continue
        if "child" in meta["tags"] or not qualified["eligible"] or qualified["historical_rgb_connection"]:
            continue
        cases = {case["condition"]: case for case in row["cases"]}
        table = FrameTable.from_json(cases["original"]["frame_table"])
        if (table.width, table.height) not in ((1280, 720), (1920, 1080)) or len(table) < 120:
            continue
        if table.rotation or table.truncated or table.interpolated or any(tick * 30 != i for i, tick in enumerate(hd.relative(table)[:120])):
            continue
        start = PLACEMENTS[len(selected)]
        family = "swap_a" if len(selected) % 2 == 0 else "swap_b"
        if not set(range(start, start + 6)) <= set(cases[family]["actual_changed_frames"]):
            continue
        pin = source_pin(row["index"], ARCHIVE)
        selected.append({**pin, "family": family, "interval": [start, start + 6], "prior_project_role": row["role"]})
        links.add(meta["url"])
        donors.update(meta["donors"])
        components.add(qualified["component"])
        if len(selected) == GROUPS:
            break
    require(len(selected) == GROUPS, f"Needed six source-separated groups, found {len(selected)}")
    return selected


def freeze(prior_dirs):
    import cv2

    from app.detectors.video import face_paired_edge, face_short_proposal
    from app.detectors.video.adapters import face_appearance
    from app.evidence import face_short_proposal_binding
    from scripts import screen_video_native_background_challenge as worker

    require(brief.DESIGN["point"] == face_short_proposal_binding.POINT, "Existing 0.10 point changed")
    files = [Path(__file__), *(Path(module.__file__) for module in (brief, hd, dual, face_paired_edge, face_short_proposal,
                                                                     face_appearance, face_short_proposal_binding, worker))]
    binaries = {name: subprocess.check_output([name, "-version"], text=True).splitlines()[0] for name in ("ffmpeg", "ffprobe")}
    return {"schema": SCHEMA, "prior_sha256": PRIOR, "qualified_sha256": sha(QUALIFIED), "sources": source_plan(prior_dirs),
            "method_sha256": {path.name: sha(path) for path in files}, "face_weights": brief.WEIGHTS, "flags": BASE_FLAGS,
            "opencv": cv2.__version__, "binaries": binaries, "gates": GATES, "point": brief.DESIGN["point"]}


def authored(folder, lock):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    media = folder / "media"
    media.mkdir(mode=0o700)
    sources = lock["sources"]
    matches = dual.closest_background(sources)
    rows = []
    for group, source in enumerate(sources, 1):
        start, end = source["interval"]
        interval = range(start, end)
        host = dual.native(source, "original", range(120))
        swap = dual.native(source, source["family"], interval)
        _, partner = matches[group - 1]
        require(group - 1 != -partner, "Genuine cut needs another source")
        guest = dual.native(sources[-partner], "original", interval)
        interventions = {
            "swap": swap,
            "lighting": {i: np.clip(host[i].astype(np.float32) * 1.15, 0, 255).astype(np.uint8) for i in interval},
            "same_recording_jump": {i: host[i - 18] for i in interval},
            "cross_recording_cut": guest,
        }
        for role in ROLES:
            path = media / f"g{group:02}-{role}.mp4"
            changes = interventions.get(role, {})
            require(all(not np.array_equal(host[i], rgb) for i, rgb in changes.items()), "Authored edit includes an unchanged picture")
            if role.startswith("tandem_"):
                hd.authored(path, source=media / f"g{group:02}-{role.removeprefix('tandem_')}.mp4", crf=35)
            else:
                hd.authored(path, pictures=(changes.get(i, host[i]) for i in range(120)))
            row = hd.describe(path, group, role, [source["interval"]] if role in POSITIVES else [])
            rows.append(row | {"authored_rgb_sha256": {str(i): hashlib.sha256(rgb).hexdigest() for i, rgb in changes.items()}})
        print("authored group", group, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == GROUPS * len(ROLES), "Incomplete or repeated authored deliveries")
    total = sum(path.stat().st_size for path in media.iterdir())
    require(total <= 160 * 1024**2, "Delivery storage bound exceeded")
    return {"freeze_sha256": sha(folder / "freeze.json"), "rows": rows, "media_bytes": total}


def score(folder, lock, state):
    import cv2

    from app.detectors.image.faces import load_yunet
    from app.detectors.video.adapters import face_appearance
    from app.detectors.video.decode import FrameTable
    from scripts import screen_video_native_background_challenge as worker
    from scripts import video_face_return_eligibility as ranges

    manifest = read(folder / "media.json")
    require(manifest["freeze_sha256"] == sha(folder / "freeze.json"), "Delivery manifest uses a different freeze")
    weights = ROOT / "models/weights"
    require(all(sha(weights / name) == digest for name, digest in lock["face_weights"].items()), "Pinned face weights differ")
    measure = worker.scorer(state)
    cv2.setNumThreads(1)
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    original_second = face_appearance._record_second_return
    with (folder / "scored.jsonl").open("x") as output:
        for row in manifest["rows"]:
            path = folder / "media" / row["file"]
            table = FrameTable.from_json(row["clock"])
            secondary = []

            def collect_second(details, *args, collected=secondary):
                original_second(details, *args)
                if record := details.get("face_return_secondary"):
                    collected.append(record)

            with patch.object(face_appearance, "_record_second_return", collect_second):
                result = measure(path, row)
            primary = [result[key] for key in ("coarse", "boundary_refinement", "short_range") if result[key]]
            baseline = sorted({tuple(ranges.interval(record, table.pts)) for record in (*primary, *secondary)})
            measurement, proposal = None, None
            eligible = result["appearance_status"] in {"ok", "not_applicable"} and result["short_range"] is None
            if eligible:
                select, measurement, _ = brief.scan(path, table, detector, recognizer, row["rgb_sha256"])
                proposal = select(baseline)
            saved = {key: row[key] for key in ("group", "condition", "sha256", "truth")}
            saved.update(prior=baseline, baseline_status=result["appearance_status"], eligible=eligible,
                         measurement=measurement, proposal=proposal, time_base=table.time_base, first_pts=table.pts[0])
            output.write(json.dumps(saved, allow_nan=False) + "\n")
            output.flush()
            print(row["group"], row["condition"], baseline, proposal["frames"] if proposal else None, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--prior-hevc", type=Path, required=True)
    parser.add_argument("--prior-avc-vp9", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "score":
        from scripts import screen_video_native_background_challenge as worker

        state = worker.configure(args.run)
        os.environ.update(BASE_FLAGS)
    lock = freeze((args.prior_hevc, args.prior_avc_vp9))
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(args.run / "freeze.json", lock)
    else:
        require(read(args.run / "freeze.json") == lock, "Frozen procedure or source plan differs")
        if args.phase == "prepare":
            write(args.run / "media.json", authored(args.run, lock))
        else:
            score(args.run, lock, state)


if __name__ == "__main__":
    main()
