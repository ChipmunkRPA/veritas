# SPDX-License-Identifier: Apache-2.0
"""Independently replay exact carriers, frame evidence and decisions for the codec transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.video import face_return
from scripts import review_video_face_six_codec_transfer as earlier
from scripts import video_face_codec_background_transfer as study

common, require = study.common, study.require


def reconstruct(run, archive, lock, manifest):
    expected = {(row["group"], row["codec"], row["condition"]): row for row in manifest}
    roster = {row["index"]: row for row in lock["sources"]}
    count = Counter()
    reader = study.matching.earlier.source_pictures
    for pair, (_, a, b) in enumerate(lock["pairs"]):
        originals = {key: list(reader(archive, roster[key], "original", range(study.FRAME_COUNT)).values())
                     for key in (a, b)}
        count["archive_picture_records"] += sum(map(len, originals.values()))
        for side, (host, guest) in enumerate(((a, b), (b, a))):
            group = 2 * pair + side
            input_pictures = originals[host]
            edited = {}
            for n, (start, end) in enumerate(study.SPANS):
                suffix = "ab"[n]
                swap = "swap_" + suffix
                native = reader(archive, roster[host], swap, range(start, end))
                edit = list(native.values())
                count["archive_picture_records"] += len(edit)
                edited[swap] = input_pictures[:start] + edit + input_pictures[end:]
                edited["cut_" + suffix] = input_pictures[:start] + originals[guest][start:end] + input_pictures[end:]
            start, end = study.SPANS[1]
            # round(6/5 * integer), with no half ties.
            illuminated = [np.minimum((6 * rgb.astype(np.uint16) + 2) // 5, 255).astype(np.uint8)
                           for rgb in input_pictures[start:end]]
            edited["light"] = input_pictures[:start] + illuminated + input_pictures[end:]
            with TemporaryDirectory(prefix="rebuild-", dir=run) as temporary:
                scratch = Path(temporary)
                for codec, recipe in study.codec_study.CODECS.items():
                    original = study.codec_study.filename(scratch, group, codec, "original")
                    for name in study.CASES:
                        path = study.codec_study.filename(scratch, group, codec, name)
                        options = ({"source": original} if name == "recompressed"
                                   else {"pictures": edited.get(name, input_pictures)})
                        quality = recipe["tandem" if name == "recompressed" else "primary"]
                        study.codec_study.encode(path, codec, quality, **options)
                        row = expected[group, codec, name]
                        actual = common.digest(path)
                        require(actual == row["sha256"], f"Separate authoring differs for source {group} {codec} {name}")
                        count["byte_identical_reconstructions"] += 1
                        count["authored_intervals"] += name in edited
            print("independently rebuilt source", group, flush=True)
    return dict(count)


def quartile(values):
    values = sorted(values)
    at = (len(values) - 1) / 4
    low, high = math.floor(at), math.ceil(at)
    return values[low] + (values[high] - values[low]) * (at - low)


def native_guard(path, media, evidence, detector):
    table = study.codec_study.clock(path)
    times = evidence["frames"]
    values = {}
    for frame in common.decode_frames(path, table, times, timeout_s=25):
        require(frame.pts_verified and frame.pts == table.pts[frame.index] and
                hashlib.sha256(frame.rgb.tobytes()).hexdigest() == media["rgb_sha256"][frame.index],
                "An edge picture lost its native input")
        faces = common.detect_faces(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        require(len(faces) <= 1, "The selected scan contains multiple faces")
        values[frame.index] = face_return.scene(frame.rgb, faces[0] if faces else None)
    require(list(values) == times, "Incomplete native edge roster")
    for a, b, expected in zip(times[::2], times[1::2], evidence["quartiles"], strict=True):
        x, m = values[a]
        y, n = values[b]
        products = [math.fsum(map(float, x[i] * y[i])) for i in range(len(m)) if m[i] and n[i]]
        require(len(products) >= 8 and abs(quartile(products) - expected) < 1e-6, "Background quartile differs")
    return len(times)


def read_rows(run, phase):
    journal = run / f"{phase}.jsonl"
    seal = common.read(run / f"{phase}-seal.json")
    require(common.digest(journal) == seal["journal_sha256"], "Scoring journal differs")
    return [json.loads(line) for line in journal.read_text().splitlines()]


def hits(span, truth):
    if not truth:
        return False
    x, y = span
    a, b = truth
    return 2 * max(0, min(b, y) - max(a, x)) >= max(b, y) - min(a, x)


def replay(run, phase, media, folder):
    detector = common.load_yunet(common.FACE_FILE)
    rows = read_rows(run, phase)
    stats = defaultdict(lambda: defaultdict(Counter))
    pictures, crops, edges, decisions = 0, 0, 0, 0
    dense = list(range(0, study.FRAME_COUNT, 2))
    for row in rows:
        record = media[row["file"]]
        path = folder / row["file"]
        require(common.digest(path) == record["sha256"] == row["sha256"] and record["truth"] == row["truth"],
                "Source or truth differs")
        current = [row["coarse"]["frames"]] if row["coarse"] and row["coarse"]["selected"] else []
        table = study.codec_study.clock(path)
        sparse = list(face_return.schedule(table)[1])
        checked = earlier.review_decision(row["eight_hz"], current, sparse)
        if checked:
            current.append(row["eight_hz"]["frames"])
        added = earlier.review_decision(row["fallback"], current, dense)
        guard = row["guard"]
        require(added == bool(guard), "Guard was applied to a range not selected by the frozen check")
        retained = False
        if guard:
            a, b = row["fallback"]["frames"]
            require(guard["frames"] == [a - 2, a, b - 2, b]
                    and guard["frame_sha256"] == [record["rgb_sha256"][i] for i in guard["frames"]],
                    "The guard used a different exact boundary")
            retained = min(guard["quartiles"]) >= study.POINT
            require(retained == guard["retained"], "Fixed guard decision differs")
            edges += native_guard(path, record, guard, detector)
        if phase == "held":
            target = defaultdict(list)
            for arm in ("eight_hz", "fallback"):
                result = row[arm]
                if result and result["selected"]:
                    for role, anchor in result["anchors"].items():
                        target[anchor["index"]].append((arm + "-" + role, anchor))
            n, c = earlier.reopen(path, record, target, None)
            pictures += n
            crops += c
        truth = row["truth"]
        kind = ("swap" if truth is not None else "cut" if row["condition"].startswith("cut_")
                or row["condition"] == "hard_cut" else "ordinary")
        observed = stats[row["codec"]][kind]
        observed["cases"] += 1
        observed["baseline_flags"] += bool(current)
        observed["fallback_measured"] += row["fallback"] is not None
        observed["dense_flags"] += added
        observed["guarded_dense_flags"] += retained

        before = any(hits(span, truth) for span in current)
        extra = added and hits(row["fallback"]["frames"], truth)
        observed["baseline_located"] += before
        observed["unfiltered_located"] += before or extra
        observed["guarded_located"] += before or (extra and retained)
        decisions += 3
    return ({codec: {k: dict(v) for k, v in counts.items()} for codec, counts in stats.items()},
            {"native_edge_pictures": edges, "native_delivered_pictures": pictures,
             "exact_native_crops": crops, "reviewed_decisions": decisions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("authoring", "decisions"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--archive", type=Path, default=study.ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run = args.run
    lock = study.frozen(run, args.archive)
    manifest = common.read(run / "media.json")
    require(manifest["freeze_sha256"] == common.digest(run / "freeze.json"), "Manifest freeze differs")
    if args.phase == "authoring":
        result = reconstruct(run, args.archive, lock, manifest["rows"])
    else:
        aggregate = common.read(run / "aggregate.json")
        result = {}
        for phase, root, rows in (
            ("held", run, manifest["rows"]),
            ("exposed", study.DEVELOPMENT, common.read(study.DEVELOPMENT / "media.json")["media"]),
        ):
            actual, counts = replay(run, phase, {row["file"]: row for row in rows}, root / "media")
            require(actual == {codec: row["counts"] for codec, row in aggregate[phase].items()}, "Independent aggregate differs")
            result[phase] = counts
    common.write(run / (args.phase + "-audit.json"), {"status": "verified", **result})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
