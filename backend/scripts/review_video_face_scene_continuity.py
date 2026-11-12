# SPDX-License-Identifier: Apache-2.0
"""Replay native sources, exact evidence and both scene-suppression decisions."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.video import face_return
from scripts import review_video_face_six_codec_transfer as earlier
from scripts import video_face_scene_continuity_transfer as study

common, require, old = study.common, study.require, study.old


def reconstruct(folder, archive, locked, manifest):
    pair_receipt = common.read(folder / "source-pairs.json")
    require(pair_receipt["pairs"] == study.pairs(locked, archive), "Genuine-source pair selection differs")
    indexed = {(row["group"], row["condition"]): row for row in manifest["media"] if row["split"] == "held"}
    stats = Counter()
    with TemporaryDirectory(prefix="native-readback-", dir=folder) as scratch:
        root = Path(scratch)
        for pair in pair_receipt["pairs"]:
            if pair["role"] != "held":
                continue
            sources = {group: study.source_group(archive, locked["roster"][group], study.conditions(group))
                       for group in pair["groups"]}
            for group, other in (pair["groups"], pair["groups"][::-1]):
                plain = sources[group]["original"]
                alternate = sources[other]["original"]
                stats["verified_source_pictures"] += sum(map(len, sources[group].values()))
                original = root / f"g{group}-original.mp4"
                for condition in study.CASES:
                    row = indexed[group, condition]
                    path = root / row["file"]
                    frames = list(plain.values())
                    if condition == "recompressed":
                        old.encoded.encode(path, source=original, crf=36)
                    else:
                        interval = study.case_span(condition, group)
                        require(row["authored_interval"] == (list(interval) if interval else None), "Authored interval changed")
                        if interval:
                            a, b = interval
                            if row["truth"]:
                                require(row["truth"] == list(interval) and list(sources[group][condition]) == list(range(a,b)),
                                        "Publisher-swap support differs from exact authored interval")
                                frames[a:b] = sources[group][condition].values()
                            elif condition.startswith("cross_cut"):
                                require(row["partner"] == other and row["similar_background"] == pair["similar"], "Genuine cut pair differs")
                                frames[a:b] = [alternate[i] for i in range(a,b)]
                            elif condition == "jump_cut":
                                frames[a:b] = frames[a+22:b+22]
                            elif condition == "camera_punch":
                                frames[a:b] = [cv2.resize(rgb[8:-8, 8:-8], (640,360), interpolation=cv2.INTER_LINEAR) for rgb in frames[a:b]]
                            else:
                                require(condition == "light", "Unexpected ordinary control")
                                frames[a:b] = [np.rint(np.minimum(rgb.astype(np.float32)*1.20, 255)).astype(np.uint8) for rgb in frames[a:b]]
                            require(all(not np.array_equal(plain[i], frames[i]) for i in range(a,b)), "Authored pictures did not change")
                            stats["verified_authored_intervals"] += 1
                        old.encoded.encode(path, frames=frames, crf=23)
                    require(common.digest(path) == row["sha256"], "Independent authored media bytes differ")
                    stats["byte_identical_reconstructions"] += 1
                print("reconstructed", group, flush=True)
    require(stats["byte_identical_reconstructions"] == len(indexed) == 54, "Incomplete held authoring replay")
    return stats


def edges(path, media, row, detector):
    start, end = row["fallback"]["frames"]
    indices = start-2, start, end-2, end
    require(tuple(row["edges"]["indices"]) == indices, "Suppression is tied to the wrong native boundaries")
    table = study.prior.clock(path)
    scene = []
    for frame in common.decode_frames(path, table, indices, timeout_s=20):
        require(frame.pts_verified and frame.pts == table.pts[frame.index] and
                hashlib.sha256(frame.rgb.tobytes()).hexdigest() == media["rgb_sha256"][frame.index], "Boundary RGB or PTS differs")
        candidates = common.detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        scene.append(face_return.scene(frame.rgb, candidates[0] if len(candidates) == 1 else None))
    require(len(scene) == 4, "All four scene boundary pictures required")
    output = defaultdict(list)
    for (left, m), (right, n) in zip(scene[::2], scene[1::2], strict=True):
        keep = m & n
        require(keep.sum() >= face_return.MINIMUM_TILE_COUNT, "Scene support differs")
        # Linear interpolation of the independently sorted valid tile correlations.
        ordered = sorted(map(float, np.sum(left[keep] * right[keep], axis=1)))
        for kind, quantile in (("median", .5), ("lower_quartile", .25)):
            offset = (len(ordered)-1)*quantile
            i = int(offset)
            t = offset-i
            output[kind].append(ordered[i]*(1-t) + ordered[min(i+1, len(ordered)-1)]*t)
    for arm, values in output.items():
        require(all(abs(a-b) < 1e-7 for a,b in zip(values, row["edges"][arm], strict=True)), "Frozen edge value differs")
    return output


def audit(folder, archive, *, reuse_authoring=False):
    lock = study.frozen(folder, archive)
    manifest = common.read(folder / "media.json")
    pair_sha = common.digest(folder / "source-pairs.json")
    freeze_sha = common.digest(folder / "freeze.json")
    manifest_sha = common.digest(folder / "media.json")
    require(manifest["freeze_sha256"] == freeze_sha and manifest["pairs_sha256"] == pair_sha, "Held source bindings differ")
    aggregate = common.read(folder / "aggregate.json")
    require(aggregate["freeze_sha256"] == freeze_sha, "Reported protocol differs")
    if reuse_authoring:
        reused = common.read(folder / "independent-authoring.json")
        require(reused["media_sha256"] == manifest_sha and reused["worker_sha256"] == common.digest(Path(__file__)), "Authoring audit does not bind this media and code")
        stats = Counter(reused["counts"])
    else:
        stats = reconstruct(folder, archive, lock, manifest)
    held = [row for row in manifest["media"] if row["split"] == "held"]
    rows = study.journal(folder, "held")
    detector = common.load_yunet(common.FACE_FILE)
    schedules = list(range(0, study.N, 2)), common.schedule([Fraction(i, 25) for i in range(study.N)], 8)
    observed = defaultdict(Counter)
    examples = {}
    for row, media in zip(rows, held, strict=True):
        require(all(row[key] == media[key] for key in ("file", "group", "condition", "sha256", "truth", "partner", "similar_background")),
                "Scored source or treatment differs")
        path = folder / "media" / media["file"]
        coarse = row["coarse"]
        incumbent = [coarse["frames"]] if coarse and coarse["score"] > old.trial.prior.POINT else []
        if earlier.review_decision(row["eight_hz"], incumbent, schedules[1]):
            incumbent.append(row["eight_hz"]["frames"])
        dense = row["fallback"]
        used = earlier.review_decision(dense, incumbent, schedules[0])
        require(used == bool(row["edges"]) and (not dense or row["eight_hz"] and not row["eight_hz"]["selected"]), "Dense evidence eligibility differs")
        recorded = incumbent + ([dense["frames"]] if used else [])
        target, example = defaultdict(list), None
        kind = study.category(row)
        if used:
            values = edges(path, media, row, detector)
            stats["verified_scene_boundary_pictures"] += 4
            for role, anchor in dense["anchors"].items():
                target[anchor["index"]].append((role, anchor))
            if kind not in examples and kind in {"swap", "similar_cross_cut", "distinct_cross_cut", "jump_cut"}:
                example = folder / f"private-review-example-{kind}"
                example.mkdir(mode=0o700, exist_ok=True)
                examples[kind] = {"selected_s": [str(Fraction(frame, 25)) for frame in dense["frames"]],
                                  "authored_s": [str(Fraction(frame, 25)) for frame in media["authored_interval"]],
                                  "anchor_s": [str(Fraction(anchor["index"], 25)) for anchor in dense["anchors"].values()]}
        count, crops = earlier.reopen(path, media, target, example)
        stats["decoded_held_pictures"] += count
        stats["verified_face_crops"] += crops
        for arm in ("current", *study.POINTS):
            suppress = used and arm != "current" and min(values[arm]) < study.POINTS[arm]
            if used and arm != "current":
                require(suppress == (min(row["edges"][arm]) < study.POINTS[arm]), "Boundary arithmetic changed a suppression decision")
            retained = incumbent if suppress else recorded
            observed[arm][kind, "flagged"] += bool(retained)
            if row["truth"]:
                a,b = row["truth"]
                observed[arm][kind, "correct"] += any(2*max(0, min(b,d)-max(a,c)) >= max(b,d)-min(a,c) for c,d in retained)
            stats["verified_decisions"] += 1
    for arm, results in observed.items():
        for (kind, metric), count in results.items():
            require(aggregate["held"][arm]["counts"][kind][metric] == count, "Independent decision count differs")
    require(stats["decoded_held_pictures"] == 6480 and stats["verified_decisions"] == 162, "Incomplete held readback")
    return {"status": "verified", "aggregate_sha256": common.digest(folder / "aggregate.json"), "counts": dict(stats), "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=study.ARCHIVE)
    parser.add_argument("--stage", choices=("authoring", "decision", "both"), default="both")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == "authoring":
        lock = study.frozen(args.folder, args.archive)
        manifest = common.read(args.folder / "media.json")
        counts = reconstruct(args.folder, args.archive, lock, manifest)
        name = "independent-authoring.json"
        result = {"counts":dict(counts), "media_sha256": common.digest(args.folder / "media.json"),
                  "worker_sha256":common.digest(Path(__file__))}
    else:
        name = "independent-audit.json"
        result = audit(args.folder, args.archive, reuse_authoring=args.stage == "decision")
    common.write(args.folder / name, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
