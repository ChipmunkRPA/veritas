# SPDX-License-Identifier: Apache-2.0
"""Test face-appearance returns on six-picture swaps in actual H.264 deliveries."""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import face_return
from app.detectors.video.decode import FrameTable
from scripts import video_short_return_codec_transfer as codec
from scripts import video_short_return_encoded_transfer as encoded

common, require, trial = encoded.common, encoded.require, encoded.trial
ARCHIVE = encoded.ARCHIVE
DATA = codec.DATA
DIMENSIONS = encoded.DIMENSIONS
N = 120
SPANS = ((54, 60), (67, 73))
ORDINARY = ("original", "recompressed", "ordinary_light", "ordinary_blur", "ordinary_time_step")
SWAPS = tuple(f"swap_{family}_{phase}" for family in ("a", "b") for phase in (0, 1))
ARMS = ("coarse", "eight_hz", "dense_rank", "consecutive")
SPEC = {
    "schema": "chipmunk.video-face-six-picture-transfer/1", "size": list(DIMENSIONS), "fps": 25, "clip_frames": N,
    "known_swap_intervals_frames": list(map(list, SPANS)), "short_core_samples": [3, 4], "short_flank_samples": 4,
    "point": face_return.POINT, "density_period_frames": 2, "reference_period_frames": "25/8",
    "primary_crf": 23, "ordinary_recompression_crf": 36,
    "gate": {"minimum_added": 6, "minimum_held_groups": 3, "maximum_new_ordinary_ranges": 2,
             "maximum_new_cut_ranges": 0, "minimum_coverage": .90},
    "method": "Compare current coarse plus 8 Hz with a 12.5 Hz version of the same rank scan and a 12.5 Hz short-window scan. "
              "The latter requires each observed center to clear the same within-clip separation point. Both require measured "
              "scene continuity at the edges, keep both existing results and suppress overlapping supplemental ranges.",
}


def bindings(archive):
    paths = [*codec.rosters(archive), DATA / "video-short-return-codec-2026-11-11/freeze.json"]
    return {"prior_freezes": [common.digest(path) for path in paths], "weights_tools": common.bindings(),
            "face_return": common.digest(Path(face_return.__file__)), "encode": common.digest(Path(encoded.__file__)),
            "method": common.digest(Path(__file__))}, paths


def roster(archive, paths):
    exposed = set.union(*(trial.previous_rtf_sources(common.read(path)) for path in paths))
    require(len(exposed) == 25, "All five preceding short-return rosters required")
    metadata = trial.prior.comparison.metadata()
    known = [metadata[common.read(common.DATA / f"group-{i:03d}-complete.json")["group"]] for i in exposed]
    urls, donors = {row["url"] for row in known}, set.union(*(row["donors"] for row in known))
    candidates = {"development": [], "held": []}
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        data = common.read(path)
        index, meta = data["index"], metadata[data["group"]]
        if index in exposed or meta["url"] in urls or meta["donors"] & donors or "child" in meta["tags"]:
            continue
        if "head rotation" not in meta["tags"]:
            continue
        original = next(case for case in data["cases"] if case["condition"] == "original")
        table = FrameTable.from_json(original["frame_table"])
        size = table.width, table.height
        if size not in encoded.PROFILES or common.relative_times(table) != encoded.CLOCK:
            continue
        split = "held" if data["role"] == "confirmation" else "development"
        candidates[split].append(("gesticulation" not in meta["tags"], sum(case["bytes"] for case in data["cases"]),
                                  index, meta, size))
    selected = []
    for split, target in (("development", 2), ("held", 4)):
        chosen = 0
        for _, _, index, meta, size in sorted(candidates[split]):
            if meta["url"] in urls or meta["donors"] & donors:
                continue
            selected.append({**encoded.prior.archive_io.source_pin(index, archive), "size": list(size), "split": split})
            urls.add(meta["url"])
            donors.update(meta["donors"])
            chosen += 1
            if chosen == target:
                break
        require(chosen == target, f"Insufficient source-disjoint {split} inputs")
    return selected


def freeze(folder, archive):
    identity, paths = bindings(archive)
    chosen = roster(archive, paths)
    folder.mkdir(parents=True, mode=0o700)
    common.write(folder / "freeze.json", {"spec": SPEC, "binding": identity, "roster": chosen})


def frozen(folder, archive):
    locked = common.read(folder / "freeze.json")
    require(locked["spec"] == SPEC and locked["binding"] == bindings(archive)[0], "Frozen inputs or method changed")
    return locked


def exact_rgb(path, table, n):
    require((table.width, table.height) == DIMENSIONS and not (table.rotation or table.truncated or table.interpolated)
            and common.relative_times(table) == [Fraction(i, 25) for i in range(n)], "H.264 output clock differs")
    return common.decode_frames(path, table, list(range(n)), timeout_s=40)


def describe(path, group, split, condition, truth):
    table = common.probe_frames(path, timeout_s=20)
    hashes = []
    for frame in exact_rgb(path, table, N):
        require(frame.index == len(hashes) and frame.pts_verified and frame.pts == table.pts[frame.index], "Unverified picture")
        hashes.append(common.hashlib.sha256(frame.rgb).hexdigest())
    require(len(hashes) == N, "Incomplete native output receipt")
    return {"file": path.name, "sha256": common.digest(path), "group": group, "split": split,
            "condition": condition, "truth": truth, "clock": table.to_json(), "rgb_sha256": hashes}


def prepare(folder, archive):
    plan = frozen(folder, archive)
    media = folder / "media"
    media.mkdir(mode=0o700)
    records, pair = [], []
    for group, pin in enumerate(plan["roster"]):
        index, split, size = pin["index"], pin["split"], tuple(pin["size"])
        require(encoded.prior.archive_io.source_pin(index, archive) == {k: pin[k] for k in ("index", "receipt_sha256", "review_sha256")},
                "Native source commitment differs")
        doc = common.read(common.DATA / f"group-{index:03d}-complete.json")
        cases = {row["condition"]: row for row in doc["cases"]}
        native = archive / "video-native-inserts-v95" / f"group-{index:03d}"
        plain = encoded.native_frames(native / cases["original"]["file"], cases["original"], size, range(N))
        base = media / f"g{group}-original.mp4"
        encoded.encode(base, frames=plain.values(), crf=23)
        records.append(describe(base, group, split, "original", None))
        recode = media / f"g{group}-recompressed.mp4"
        encoded.encode(recode, source=base, crf=36)
        records.append(describe(recode, group, split, "recompressed", None))
        for condition in (*ORDINARY[2:], *SWAPS):
            positive = condition in SWAPS
            name = condition[:-2] if positive else condition
            phase = int(condition[-1]) if positive else int(condition == "ordinary_blur")
            start, end = SPANS[phase]
            original, changed = cases["original"]["expected_rgb_sha256"], cases[name]["expected_rgb_sha256"]
            require(all(original[j] != changed[j] for j in range(start, end)), "Authored six pictures were not actually changed")
            inserted = encoded.native_frames(native / cases[name]["file"], cases[name], size, range(start, end))
            require(all(not np.array_equal(plain[i], inserted[i]) for i in range(start, end)), "Scaled edit lost a picture change")
            path = media / f"g{group}-{condition}.mp4"
            encoded.encode(path, frames=(inserted.get(i, rgb) for i, rgb in plain.items()), crf=23)
            records.append(describe(path, group, split, condition, [start, end] if positive else None))
        pair.append(plain)
        if len(pair) == 2:
            for phase, (base_frames, donor) in enumerate((pair, pair[::-1])):
                start, end = SPANS[phase]
                path = media / f"g{group-1+phase}-hard_cut.mp4"
                require(all(not np.array_equal(base_frames[i], donor[i]) for i in range(start, end)), "Whole-picture cut lacks source changes")
                encoded.encode(path, frames=(donor[i] if start <= i < end else rgb for i, rgb in base_frames.items()), crf=23)
                records.append(describe(path, group-1+phase, split, "hard_cut", None))
            pair.clear()
        print("prepared", split, group, flush=True)
    require(len(records) == len({row["sha256"] for row in records}) == 60, "Complete distinct delivery roster required")
    common.write(folder / "media.json", {"freeze_sha256": common.digest(folder / "freeze.json"), "media": records})


def supported_short(times, indices, rows):
    flank = SPEC["short_flank_samples"]
    usable = {i: rows[i] for i in indices if rows[i].vector is not None}
    if len(usable) < 11:
        return None
    order = {i: j for j, i in enumerate(usable)}
    vectors = np.stack([row.vector for row in usable.values()])
    d = 1 - np.clip(vectors @ vectors.T, -1, 1)
    adjacent = {(a, b): face_return.scene_correlation(rows[a], rows[b]) for a, b in zip(indices, indices[1:])}
    good_iou = {(a, b): a in usable and b in usable and common.iou(rows[a].face, rows[b].face) >= face_return.MINIMUM_FACE_IOU
                for a, b in zip(indices, indices[1:])}
    best = None
    for width in SPEC["short_core_samples"]:
        for start in range(flank, len(indices) - width - flank + 1):
            end = start + width
            support = (indices[start-flank:start], indices[start:end], indices[end:end+flank])
            joined = indices[start-flank:end+flank]
            if not all(good_iou[a,b] for a, b in zip(joined, joined[1:])):
                continue
            bounds = (indices[start-1], indices[start], indices[end-1], indices[end])
            entry, exit_ = adjacent[bounds[:2]], adjacent[bounds[2:]]
            if entry is None or exit_ is None or min(entry, exit_) < face_return.SCENE_CORRELATION_POINT:
                continue
            a, b, c = ([order[i] for i in part] for part in support)
            variation = [np.quantile(d[np.ix_(a, c)], .75)]
            for group in (a, b, c):
                variation.append(np.quantile(d[np.ix_(group, group)][np.triu_indices(len(group), k=1)], .75))
            before = np.quantile(d[np.ix_(b, a)], .25, axis=1)
            after = np.quantile(d[np.ix_(b, c)], .25, axis=1)
            margins = (np.minimum(before, after) - max(variation)).tolist()
            response = min(margins)
            if best is None or response > best[0].response:
                best = face_return.Range(indices[start], indices[end], response, (entry, exit_), support, bounds), margins
    return best


def compare(path, row, models):
    require(common.digest(path) == row["sha256"], "Scored source bytes differ")
    table = common.probe_frames(path, timeout_s=20)
    require(table.to_json() == row["clock"], "Scored source clock differs")
    y, s = models
    times, sparse = face_return.schedule(table)
    coarse = [i for i in common.schedule(times, 4) if 0 < i < N - 1]
    dense = tuple(range(0, N, SPEC["density_period_frames"]))
    wanted = {i + d for i in coarse for d in (-1, 0, 1)} | set(dense) | set(sparse)
    face, anchors, count, several = {}, {}, 0, set()
    for frame in common.decode_frames(path, table, sorted(wanted), timeout_s=65):
        index, rgb = frame.index, frame.rgb
        digest = common.hashlib.sha256(rgb).hexdigest()
        require(frame.pts_verified and digest == row["rgb_sha256"][index] and frame.pts == table.pts[index], "Native input differs")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        candidates = common.detect_faces(bgr, detector=y, min_size=48)
        head, vector = None, None
        if len(candidates) == 1:
            head = candidates[0]
            geometry = common.face_appearance.face_row(head.landmarks)
            geometry[0, :4], geometry[0, 14] = head.box(), head.score
            vector = common.face_appearance.unit(s.feature(s.alignCrop(bgr, geometry)))
            x, j, w, h = head.box()
            anchors[index] = {"index": index, "pts": frame.pts, "rgb_sha256": digest, "xywh": [x, j, w, h],
                              "crop_sha256": common.hashlib.sha256(rgb[j:j+h, x:x+w].tobytes()).hexdigest()}
        elif len(candidates) > 1:
            several.add(index)
        face[index] = face_return.Sample(head, vector, face_return.scene(rgb, head))
        count += 1
    require(set(face) == wanted, "Decoded sample roster differs")
    current = common.appearance(coarse, table, times, {i: {"appearance": f.vector} for i, f in face.items()})
    long = None
    if current and current["score"] > trial.prior.POINT:
        long = [int(Fraction(current[label])*25) for label in ("start", "end")]
    existing = [] if long is None else [long]

    def locate(result):
        if result is None:
            return None
        return {"frames": [result.start, result.end], "score": result.response,
                "selected": result.response > face_return.POINT and 0 < times[result.end]-times[result.start] < face_return.MAX_RANGE
                and all(iou([result.start, result.end], prior) < Fraction(1, 2) for prior in existing),
                "support": result.support, "scene_correlations": result.scene_correlations,
                "anchors": {label: anchors[i] for label, i in result.anchors().items()}}

    original = locate(None if several.intersection(sparse) else face_return.highest(times, sparse, face))
    if original and original["selected"]:
        existing.append(original["frames"])
    dense_rank = locate(None if several.intersection(dense) else face_return.highest(times, dense, face))
    stronger = None if several.intersection(dense) else supported_short(times, dense, face)
    consecutive = None if stronger is None else {**locate(stronger[0]), "minimum_each_center": stronger[1]}
    first = None if current is None else {"frames": [int(Fraction(current[k])*25) for k in ("start", "end")],
                                         "score": current["score"], "selected": long is not None}
    return {"coarse": first, "eight_hz": original, "dense_rank": dense_rank, "consecutive": consecutive}, count, len(anchors)


def iou(a, b):
    return Fraction(max(0, min(a[1], b[1])-max(a[0], b[0])), max(a[1], b[1])-min(a[0], b[0]))


def score(folder, archive, split):
    plan, delivery = frozen(folder, archive), common.read(folder / "media.json")
    require(delivery["freeze_sha256"] == common.digest(folder / "freeze.json"), "Prepared source commitment differs")
    if split == "held":
        require(common.read(folder / "development-seal.json")["journal_sha256"] == common.digest(folder / "development.jsonl"),
                "Complete development stage must precede held scores")
    cv2.setNumThreads(1)
    models = common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    started = time.monotonic()
    count = 0
    with (folder / f"{split}.jsonl").open("x") as stream:
        for row in delivery["media"]:
            if row["split"] != split:
                continue
            arms, decoded, supported = compare(folder / "media" / row["file"], row, models)
            stream.write(json.dumps({key: row[key] for key in ("file", "group", "condition", "sha256", "truth")}
                                    | {"arms": arms, "decoded": decoded, "supported": supported}, allow_nan=False) + "\n")
            stream.flush()
            count += 1
            print(split, row["group"], row["condition"], flush=True)
    group_count = sum(pin["split"] == split for pin in plan["roster"])
    require(count == group_count * (len(ORDINARY) + len(SWAPS) + 1), "Incomplete scored role")
    common.write(folder / f"{split}-seal.json", {"journal_sha256": common.digest(folder / f"{split}.jsonl"),
          "media_sha256": common.digest(folder / "media.json"), "elapsed_s": time.monotonic() - started,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def journal(folder, split):
    path = folder / f"{split}.jsonl"
    require(common.read(folder / f"{split}-seal.json")["journal_sha256"] == common.digest(path), "Scored journal differs")
    return list(map(json.loads, path.read_text().splitlines()))


def tally(rows):
    strata = {arm: defaultdict(Counter) for arm in (*ARMS, "current_combined")}
    groups = defaultdict(set)
    for row in rows:
        truth = row["truth"]
        kind = "swap" if truth else ("hard_cut" if row["condition"] == "hard_cut" else "ordinary")
        arms = row["arms"]
        now = [arms[key] for key in ("coarse", "eight_hz") if arms[key] and arms[key]["selected"]]
        current = bool(truth and any(iou(rec["frames"], truth) >= Fraction(1, 2) for rec in now))
        strata["current_combined"][kind].update({"views": 1, "correct": current, "flagged": bool(now),
             "assessed": any(arms[key] is not None for key in ("coarse", "eight_hz"))})
        for arm, result in arms.items():
            flag = result is not None and result["selected"]
            correct = bool(flag and truth and iou(result["frames"], truth) >= Fraction(1, 2))
            added = correct and not current
            strata[arm][kind].update({"views": 1, "assessed": result is not None, "flagged": flag, "correct": correct,
                                     "adds_correct": added, "wrong_range": bool(flag and truth and not correct)})
            if added:
                groups[arm].add(row["group"])
    decisions = {}
    for arm in ("dense_rank", "consecutive"):
        measured = strata[arm]
        gates = {"more_locations": measured["swap"]["adds_correct"] >= SPEC["gate"]["minimum_added"],
                 "groups": len(groups[arm]) >= SPEC["gate"]["minimum_held_groups"],
                 "ordinary": measured["ordinary"]["flagged"] <= SPEC["gate"]["maximum_new_ordinary_ranges"],
                 "cuts": measured["hard_cut"]["flagged"] <= SPEC["gate"]["maximum_new_cut_ranges"],
                 "coverage": all(measured[k]["assessed"] >= SPEC["gate"]["minimum_coverage"] * measured[k]["views"]
                                 for k in ("swap", "ordinary"))}
        decisions[arm] = {"criteria": gates, "advance": all(gates.values()), "added_groups": len(groups[arm])}
    return {"strata": strata, "decisions": decisions, "frames": sum(row["decoded"] for row in rows),
            "single_face_frames": sum(row["supported"] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "prepare", "development", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    folder, archive = args.folder, args.archive
    if args.stage == "freeze":
        freeze(folder, archive)
    elif args.stage == "prepare":
        cv2.setNumThreads(1)
        prepare(folder, archive)
    elif args.stage == "report":
        frozen(folder, archive)
        common.write(folder / "aggregate.json", {"spec": SPEC, "development": tally(journal(folder, "development")),
            "held": tally(journal(folder, "held")), "runtime": common.read(folder / "held-seal.json")})
    else:
        score(folder, archive, args.stage)


if __name__ == "__main__":
    main()
