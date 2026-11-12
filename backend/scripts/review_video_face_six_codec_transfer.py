# SPDX-License-Identifier: Apache-2.0
"""Reconstruct both codec panels and independently verify their timed evidence."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
from PIL import Image

from scripts import review_video_face_six_duration_transfer as earlier
from scripts import video_face_six_codec_transfer as study

common, require = study.common, study.require


def review_decision(record, prior, schedule):
    if record is None:
        return False
    normalized = {"frames": record["frames"], "margin": record["score"], "selected": record["selected"],
                  "scene": record["scene_correlations"], "support": record["support"],
                  "evidence": {role:{"frame": anchor["index"]} for role,anchor in record["anchors"].items()}}
    return earlier.review_decision(normalized, prior, study.original.N, schedule)


def reopen(path, media, target, example):
    require(common.digest(path) == media["sha256"] and study.codec_of(path) == f"{media['codec']},yuv420p",
            "Actual carrier or file binding differs")
    with path.open("rb") as stream:
        header = stream.read(128)
    container_matches = (header.startswith(b"\x1aE\xdf\xa3") and b"\x42\x82\x84webm" in header
                         if media["codec"] == "vp9" else header[4:8] == b"ftyp")
    require(container_matches, "Actual WebM or MP4 file header differs")
    table = study.clock(path)
    require(table.to_json() == media["clock"], "Recorded frame table differs")
    n, crops = 0, 0
    for frame in common.decode_frames(path, table, range(study.original.N), timeout_s=45):
        rgb = frame.rgb
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        require(frame.pts_verified and frame.index == n and frame.pts == table.pts[n] and
                (frame.pts-table.pts[0])*table.tb == Fraction(n,25) and digest == media["rgb_sha256"][n],
                "Native media picture or presentation time differs")
        for role, item in target.get(n, ()):
            x,y,w,h = item["xywh"]
            require(0 <= x < x+w <= table.width and 0 <= y < y+h <= table.height
                    and item["pts"] == frame.pts and item["rgb_sha256"] == digest, "Evidence crop bounds or PTS differ")
            crop = rgb[y:y+h,x:x+w]
            require(hashlib.sha256(crop.tobytes()).hexdigest() == item["crop_sha256"], "Reopened crop bytes differ")
            crops += 1
            if example:
                Image.fromarray(rgb).save(example / f"{role}-picture.png")
                Image.fromarray(crop).save(example / f"{role}-crop.png")
        n += 1
    require(n == study.original.N, "Reopen every delivered picture")
    return n,crops


def reconstruct(folder, archive, lock, media):
    held = {(row["group"],row["codec"],row["condition"]):row for row in media}
    counts = Counter()
    previous = None
    with TemporaryDirectory(prefix="private-reconstruction-", dir=folder) as scratch:
        root = Path(scratch)
        for group,pin in enumerate(lock["roster"]):
            if pin["role"] != "held":
                continue
            plain,altered = study.source_frames(archive,pin,group)
            counts["native_source_pictures"] += len(plain) + sum(map(len,altered.values()))
            for codec,config in study.CODECS.items():
                base = study.filename(root,group,codec,"original")
                study.encode(base,codec,config["primary"],pictures=(plain[i] for i in range(study.original.N)))
                for condition in study.CASES[:-1]:
                    path = study.filename(root,group,codec,condition)
                    row = held[group,codec,condition]
                    truth = None
                    if condition == "recompressed":
                        study.encode(path,codec,config["tandem"],source=base)
                    elif condition in altered:
                        a,b = study.span(condition,group)
                        pixels = dict(plain)
                        require(sorted(altered[condition]) == list(range(a,b)), "Actual authored insert differs from the declared interval")
                        pixels.update(altered[condition])
                        study.encode(path,codec,config["primary"],pictures=(pixels[i] for i in range(study.original.N)))
                        if condition.startswith("swap_"):
                            truth = [a,b]
                        counts["verified_swap_or_ordinary_intervals"] += 1
                    require(row["truth"] == truth and common.digest(path) == row["sha256"], "Exact codec authoring replay differs")
                    counts["byte_identical_reconstructions"] += 1
            if previous is None:
                previous = plain
            else:
                for codec,config in study.CODECS.items():
                    for phase,(base,donor) in enumerate(((previous,plain),(plain,previous))):
                        a,b = study.original.SPANS[phase]
                        path = study.filename(root,group-1+phase,codec,"hard_cut")
                        pixels = [donor[i] if a <= i < b else base[i] for i in range(study.original.N)]
                        study.encode(path,codec,config["primary"],pictures=pixels)
                        row = held[group-1+phase,codec,"hard_cut"]
                        require(row["truth"] is None and common.digest(path) == row["sha256"], "Exact A/B/A ordinary cut replay differs")
                        counts["verified_cuts"] += 1
                        counts["byte_identical_reconstructions"] += 1
                previous = None
            print("reconstructed held source group",group,flush=True)
    return counts


def source_receipt(folder, archive, lock, manifest):
    require(manifest["freeze_sha256"] == common.digest(folder / "freeze.json"), "Source manifest differs")
    media = [row for row in manifest["media"] if row["role"] == "held"]
    require(len(media) == 64, "Complete held source manifest required")
    counts = reconstruct(folder,archive,lock,media)
    return {"status":"verified","media_sha256":common.digest(folder / "media.json"),
            "reviewer_sha256":common.digest(Path(__file__)),"counts":dict(counts)}


def audit(folder, archive, *, use_source_receipt=False):
    lock = study.frozen(folder,archive)
    aggregate = common.read(folder / "aggregate.json")
    manifest = common.read(folder / "media.json")
    media = [row for row in manifest["media"] if row["role"] == "held"]
    rows = study.journal(folder,"held")
    require(len(rows) == len(media) == 64 and manifest["freeze_sha256"] == common.digest(folder / "freeze.json"), "Held roster differs")
    matched = defaultdict(dict)
    counts = defaultdict(lambda:defaultdict(Counter))
    added_groups, picture_count = defaultdict(set),Counter()
    evidence_count, examples = 0,{}
    times = [Fraction(i,25) for i in range(study.original.N)]
    old_schedule = common.schedule(times,8)
    dense_schedule = list(range(0,study.original.N,2))
    for recorded,row in zip(rows,media,strict=True):
        require(all(recorded[k] == row[k] for k in ("file","group","condition","codec","sha256","truth")), "Held file binding differs")
        codec,truth = row["codec"],row["truth"]
        matched[row["group"],row["condition"]][codec] = truth
        existing = [recorded["coarse"]["frames"]] if recorded["coarse"] and recorded["coarse"]["selected"] else []
        if recorded["coarse"]:
            require(recorded["coarse"]["selected"] == (recorded["coarse"]["score"] > study.original.trial.prior.POINT),
                    "Coarse incumbent differs from frozen point")
        legacy = recorded["eight_hz"]
        old_range = review_decision(legacy,existing,old_schedule)
        if old_range:
            existing.append(legacy["frames"])
        fallback = recorded["fallback"]
        require(fallback is None or (legacy is not None and not old_range), "Fallback overrode the original short result")
        found = review_decision(fallback,existing,dense_schedule)
        old_hit = bool(truth and any(earlier.overlap(span,truth) >= Fraction(1,2) for span in existing))
        new_hit = bool(truth and found and earlier.overlap(fallback["frames"],truth) >= Fraction(1,2))
        kind = "swap" if truth else "cut" if row["condition"] == "hard_cut" else "ordinary"
        counts[codec][kind].update({"cases":1,"incumbent_flags":bool(existing),"incumbent_located":old_hit,
            "fallback_flags":found,"new_located":new_hit and not old_hit,"combined_located":old_hit or new_hit,
            "measured_fallback":fallback is not None,"misplaced_fallback":bool(truth and found and not new_hit)})
        if new_hit and not old_hit:
            added_groups[codec].add(row["group"])
        targets,example = defaultdict(list),None
        if found:
            for role,anchor in fallback["anchors"].items():
                targets[anchor["index"]].append((role,anchor))
        if new_hit and codec not in examples:
            example = folder / f"private-example-{codec}"
            example.mkdir(mode=0o700,exist_ok=True)
            examples[codec] = {"selected_s":[str(Fraction(i,25)) for i in fallback["frames"]],
                               "known_s":[str(Fraction(i,25)) for i in truth],"iou":str(earlier.overlap(fallback["frames"],truth)),
                               "anchor_s":[str(Fraction(fallback["anchors"][role]["index"],25)) for role in ("before","during","after")]}
        n,crops = reopen(folder / "media" / row["file"],row,targets,example)
        picture_count[codec] += n
        evidence_count += crops
    require(len(matched) == 32 and all(set(x) == set(study.CODECS) and x["vp9"] == x["hevc"] for x in matched.values()),
            "Cross-codec truth roles differ")
    for codec,published in aggregate["held"].items():
        r = counts[codec]
        criteria = {"swap_locations":r["swap"]["new_located"] >= 4,"source_groups":len(added_groups[codec]) >= 3,
                    "ordinary":r["ordinary"]["fallback_flags"] <= 1,"cut":r["cut"]["fallback_flags"] == 0,
                    "coverage":all(10*r[k]["measured_fallback"] >= 9*r[k]["cases"] for k in ("swap","ordinary"))}
        require(r == published["counts"] and published["added_groups"] == len(added_groups[codec])
                and criteria == published["criteria"] and all(criteria.values()) == published["advances"], "Held decision audit differs")
    source = common.read(folder / "source-audit.json") if use_source_receipt else source_receipt(folder,archive,lock,manifest)
    require(source["status"] == "verified" and source["media_sha256"] == common.digest(folder / "media.json")
            and source["reviewer_sha256"] == common.digest(Path(__file__)), "Independent source reconstruction commitment differs")
    authored = source["counts"]
    require(picture_count == {"vp9":3840,"hevc":3840} and authored["byte_identical_reconstructions"] == 64
            and authored["verified_swap_or_ordinary_intervals"] == 40 and authored["verified_cuts"] == 8,
            "Incomplete independent codec and source reconstruction")
    return {"status":"verified","held_files":len(media),"decoded_held_pictures":dict(picture_count),
            "exact_face_crops":evidence_count,"timing_decisions":2*len(rows),**authored,"examples":examples,
            "aggregate_sha256":common.digest(folder / "aggregate.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=study.ARCHIVE)
    parser.add_argument("--phase", choices=("all", "source", "held"), default="all")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    folder,archive = args.folder,args.archive
    if args.phase == "source":
        receipt = source_receipt(folder,archive,study.frozen(folder,archive),common.read(folder / "media.json"))
        common.write(folder / "source-audit.json",receipt)
    else:
        receipt = audit(folder,archive,use_source_receipt=args.phase == "held")
        common.write(folder / "independent-audit.json",receipt)
    print(receipt,flush=True)


if __name__ == "__main__":
    main()
