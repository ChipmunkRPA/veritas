# SPDX-License-Identifier: Apache-2.0
"""Rebuild the exact duration-transfer inputs and replay native decisions and crops."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from PIL import Image

from scripts import video_face_six_duration_transfer as study

common, require = study.common, study.require


def overlap(a, b):
    return Fraction(max(0, min(a[1], b[1])-max(a[0], b[0])), max(a[1], b[1])-min(a[0], b[0]))


def review_decision(record, prior, count, schedule):
    if record is None:
        return False
    a, b = record["frames"]
    left, middle, right = record["support"]
    start, end = schedule.index(a), schedule.index(b)
    require(0 <= a < b < count and start >= 4 and end + 4 <= len(schedule) and end-start in (3,4,6)
            and min(len(left),len(right)) >= 3 and len(middle) >= (4 if end-start == 6 else 3)
            and set(left) <= set(schedule[start-4:start]) and set(middle) <= set(schedule[start:end])
            and set(right) <= set(schedule[end:end+4]) and min(record["scene"]) >= .55,
            "Guarded measured support differs")
    for role, index in zip(("before", "during", "after"), (left[-1], middle[len(middle)//2], right[0]), strict=True):
        require(record["evidence"][role]["frame"] == index, "Native before/during/after role differs")
    flag = record["margin"] > .10 and b-a < Fraction(25,2) and all(overlap([a,b], span) < Fraction(1,2) for span in prior)
    require(record["selected"] == flag, "Frozen margin or duplicate routing differs")
    return flag


def reopened(path, expected, targets, example):
    require(common.digest(path) == expected["sha256"], "Delivery bytes changed")
    table = common.probe_frames(path, timeout_s=25)
    codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
                                    "-of", "csv=p=0", str(path)], timeout=10, text=True).strip()
    require(codec == "h264,yuv420p" and table.to_json() == expected["table"] and (table.width,table.height)==(640,360),
            "Actual output dimensions, codec or frame table differ")
    crops, i = 0, 0
    for picture in common.decode_frames(path, table, range(expected["count"]), timeout_s=45):
        rgb = picture.rgb
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        require(picture.index == i and picture.pts_verified and picture.pts == table.pts[i]
                and (picture.pts-table.pts[0])*table.tb == Fraction(i,25) and digest == expected["rgb_sha256"][i],
                "Encoded native picture or source-clock binding differs")
        for role, crop in targets.get(i, ()):
            x,y,w,h = crop["xywh"]
            require(0 <= x < x+w <= table.width and 0 <= y < y+h <= table.height and picture.pts == crop["pts"]
                    and digest == crop["rgb_sha256"], "Selected evidence coordinates differ")
            section = rgb[y:y+h,x:x+w]
            require(hashlib.sha256(section.tobytes()).hexdigest() == crop["crop_sha256"], "Selected crop differs from source")
            crops += 1
            if example is not None:
                Image.fromarray(rgb).save(example / f"{role}-picture.png")
                Image.fromarray(section).save(example / f"{role}-crop.png")
        i += 1
    require(i == expected["count"], "A held output picture was omitted")
    return i, crops


def reconstruct(folder, archive, plan, media):
    """Re-encode all held cases from exact source RGB and independently compare file bytes."""
    lookup = {(r["group"],r["count"],r["condition"]):r for r in media}
    groups = [(i,row) for i,row in enumerate(plan["roster"]) if row["role"] == "held"]
    enc = study.earlier.encoded
    counts, first = Counter(), None
    with TemporaryDirectory(prefix="private-authoring-", dir=folder) as scratch:
        output = Path(scratch)
        for group, pin in groups:
            shape,cases,root = study.source_set(pin, archive)
            rec = cases["original"]
            plain = enc.native_frames(root / rec["file"], rec, shape, range(150))
            counts["native_source_pictures"] += len(plain)
            variants = {}
            for condition in ("ordinary_light", "ordinary_blur", "ordinary_time_step", "swap_a", "swap_b"):
                selected = set()
                for n in study.SPANS:
                    a,b = study.span(condition, group, n)
                    selected.update(range(a,b))
                variant = cases[condition]
                frames = enc.native_frames(root / variant["file"], variant, shape, sorted(selected))
                require(len(frames) == 12, "Two six-picture source intervals required")
                require(all(not (plain[i] == frame).all() for i,frame in frames.items()), "Known local authored picture is unchanged")
                variants[condition] = frames
                counts["native_source_pictures"] += len(frames)
            for n in study.SPANS:
                original = output / f"g{group}-{n}-original.mp4"
                enc.encode(original, frames=(plain[i] for i in range(n)), crf=23)
                for condition in study.CASE_NAMES[:-1]:
                    committed = lookup[group,n,condition]
                    truth = None
                    result = output / f"g{group}-{n}-{condition}.mp4"
                    if condition == "recompressed":
                        enc.encode(result, source=original, crf=36)
                    elif condition in variants:
                        a,b = study.span(condition, group, n)
                        pixels = dict(plain)
                        pixels.update({i: variants[condition][i] for i in range(a,b)})
                        enc.encode(result, frames=(pixels[i] for i in range(n)), crf=23)
                        if condition.startswith("swap_"):
                            truth = [a,b]
                    require(committed["truth"] == truth and common.digest(result) == committed["sha256"], "Exact encoded authored input differs")
                    counts["byte_identical_reconstructions"] += 1
                    counts["verified_ordinary_or_swap_intervals"] += condition in variants
            if first is None:
                first = plain
            else:
                for n,spans in study.SPANS.items():
                    for phase,(base,donor) in enumerate(((first,plain),(plain,first))):
                        a,b = spans[phase]
                        result = output / f"g{group-1+phase}-{n}-hard_cut.mp4"
                        enc.encode(result, frames=(donor[i] if a <= i < b else base[i] for i in range(n)), crf=23)
                        committed = lookup[group-1+phase,n,"hard_cut"]
                        require(committed["truth"] is None and common.digest(result) == committed["sha256"],
                                "Exact encoded genuine cut differs")
                        counts["byte_identical_reconstructions"] += 1
                        counts["verified_whole_picture_cuts"] += 1
                first = None
            print("reconstructed held source group", group, flush=True)
    return counts


def audit(folder, archive):
    lock = study.frozen(folder, archive)
    aggregate = common.read(folder / "aggregate.json")
    manifest = common.read(folder / "media.json")
    media = [row for row in manifest["media"] if row["role"] == "held"]
    rows = study.journal(folder, "held")
    require(len(rows) == len(media) == 64 and manifest["freeze_sha256"] == common.digest(folder / "freeze.json"),
            "Held source roster incomplete")
    counters = defaultdict(lambda: defaultdict(Counter))
    added_groups = defaultdict(set)
    sample_counts, crops, examples = Counter(),0,{}
    for observed,expected in zip(rows,media,strict=True):
        require(all(observed[k] == expected[k] for k in ("name", "sha256", "group", "condition", "count", "truth")),
                "Held input order or commitments differ")
        n,truth = expected["count"], expected["truth"]
        old,coarse = [],observed["coarse"]
        if coarse and coarse["score"] > study.earlier.trial.prior.POINT:
            old.append([int(Fraction(coarse[k])*25) for k in ("start", "end")])
        times = [Fraction(i,25) for i in range(n)]
        grid = common.schedule(times,8)
        existing = review_decision(observed["eight_hz"], old, n, grid)
        if existing:
            old.append(observed["eight_hz"]["frames"])
        dense = observed["fallback"]
        require(dense is None or (observed["eight_hz"] is not None and not existing), "Fallback ran after a selected legacy result")
        chosen = review_decision(dense, old, n, list(range(0,n,2)))
        baseline_located = bool(truth and any(overlap(span,truth) >= Fraction(1,2) for span in old))
        dense_located = bool(truth and chosen and overlap(dense["frames"],truth) >= Fraction(1,2))
        if dense_located and not baseline_located:
            added_groups[str(n)].add(expected["group"])
        kind = "positive" if truth else "cut" if expected["condition"] == "hard_cut" else "ordinary"
        counters[str(n)][kind].update({"cases":1,"current_located":baseline_located,"current_flagged":bool(old),
            "measured_fallback":dense is not None,"fallback_flagged":chosen,"added_locations":dense_located and not baseline_located,
            "combined_located":baseline_located or dense_located,"fallback_wrong_range":bool(chosen and truth and not dense_located)})
        target,exhibit = defaultdict(list),None
        if chosen:
            for role,crop in dense["evidence"].items():
                target[crop["frame"]].append((role,crop))
            if dense_located and str(n) not in examples:
                exhibit = folder / f"private-example-{n}"
                exhibit.mkdir(mode=0o700,exist_ok=True)
                examples[str(n)] = {"range_s":[str(Fraction(i,25)) for i in dense["frames"]],
                                    "truth_s":[str(Fraction(i,25)) for i in truth],
                                    "iou":str(overlap(dense["frames"],truth)),
                                    "anchor_s":[str(Fraction(dense["evidence"][role]["frame"],25))
                                                for role in ("before","during","after")]}
        samples,found = reopened(folder / "media" / expected["name"], expected, target, exhibit)
        sample_counts[n] += samples
        crops += found
    require(dict(counters) == {n:r["counts"] for n,r in aggregate["held"].items()}
            and sample_counts == {80:2560,150:4800}, "Independent routing, decision or picture counts differ")
    for n, result in aggregate["held"].items():
        observed = counters[n]
        gate = {"located":observed["positive"]["added_locations"] >= 4,"groups":len(added_groups[n]) >= 3,
                "ordinary":observed["ordinary"]["fallback_flagged"] <= 1,"cut":observed["cut"]["fallback_flagged"] == 0,
                "coverage":all(10 * observed[k]["measured_fallback"] >= 9 * observed[k]["cases"] for k in ("positive","ordinary"))}
        require(result["added_groups"] == len(added_groups[n]) and result["criteria"] == gate
                and result["advance"] == all(gate.values()), "Frozen advancement criteria differ")
    authorship = reconstruct(folder, archive, lock, media)
    require(authorship["byte_identical_reconstructions"] == 64 and authorship["verified_whole_picture_cuts"] == 8,
            "Incomplete independent authoring replay")
    return {"status":"verified","held_files":len(media),"decoded_h264_pictures":sum(sample_counts.values()),
            "checked_crops":crops, "timing_decisions_replayed":len(rows)*2, **authorship,
            "examples":examples,"aggregate_sha256":common.digest(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=study.archive_default)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = audit(args.folder, args.archive)
    common.write(args.folder / "independent-audit.json", result)
    print(result,flush=True)


if __name__ == "__main__":
    main()
