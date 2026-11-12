# SPDX-License-Identifier: Apache-2.0
"""Transfer the frozen six-picture fallback to VP9/WebM and HEVC/MP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import subprocess
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import face_return
from app.detectors.video.decode import FrameTable
from scripts import video_face_six_duration_transfer as durations
from scripts import video_face_six_picture_transfer as original

common, require, ARCHIVE = original.common, original.require, original.ARCHIVE
DATA = original.DATA
PRIOR = (original.DATA / "video-face-six-picture-2026-11-12-continuation",
         original.DATA / "video-face-six-duration-2026-11-12")
CODECS = {
    "vp9": {"suffix": "webm", "primary": 32, "tandem": 44,
            "options": ["-c:v", "libvpx-vp9", "-deadline", "good", "-cpu-used", "7", "-b:v", "0", "-g", "25",
                        "-row-mt", "1", "-lag-in-frames", "0"]},
    "hevc": {"suffix": "mp4", "primary": 26, "tandem": 38,
             "options": ["-c:v", "libx265", "-preset", "ultrafast", "-tag:v", "hvc1", "-video_track_timescale", "25000",
                         "-x265-params", "pools=1:frame-threads=1:keyint=25:min-keyint=25:scenecut=0:log-level=error"]},
}
ORDINARY = original.ORDINARY
CASES = (*ORDINARY, "swap_a", "swap_b", "hard_cut")
SPEC = {
    "schema": "chipmunk.video-face-six-codec-transfer/1", "dimensions": list(original.DIMENSIONS), "fps": 25,
    "frames": original.N, "spans": list(map(list, original.SPANS)), "codec_recipes": CODECS, "point": face_return.POINT,
    "dedupe_iou": str(face_return.DEDUPE_IOU), "methods": "Retain coarse and 8 Hz flags. Test every second native picture only after "
        "8 Hz measures a winner and emits no distinct range. Freeze the deployed rank, guards, point and duplicate suppression.",
    "gate_per_codec": {"added_correct": 4, "added_groups": 3, "ordinary_extra_max": 1, "cut_extra_max": 0, "coverage_min": .90},
}


def planned(archive):
    base, paths = original.bindings(archive)
    older = [common.read(folder / "freeze.json") for folder in PRIOR]
    for folder in PRIOR:
        audit = common.read(folder / "independent-audit.json")
        require(audit["status"] == "verified" and audit["aggregate_sha256"] == common.digest(folder / "aggregate.json"),
                "Prior exact codec and source study lacks its independent audit")
    exposed = set.union(*(original.trial.previous_rtf_sources(common.read(path)) for path in paths),
                        *({row["index"] for row in doc["roster"]} for doc in older))
    require(len(exposed) == 37, "Exact earlier localization roster required")
    metadata = original.trial.prior.comparison.metadata()
    known = [metadata[common.read(common.DATA / f"group-{index:03d}-complete.json")["group"]] for index in exposed]
    links, donors = {item["url"] for item in known}, set.union(*(item["donors"] for item in known))
    candidates = []
    for path in sorted(common.DATA.glob("group-*-complete.json")):
        data = common.read(path)
        index, meta = data["index"], metadata[data["group"]]
        if index in exposed or meta["url"] in links or meta["donors"] & donors or "child" in meta["tags"]:
            continue
        first = next(row for row in data["cases"] if row["condition"] == "original")
        table = FrameTable.from_json(first["frame_table"])
        shape = table.width, table.height
        if shape in original.encoded.PROFILES and common.relative_times(table) == original.encoded.CLOCK:
            candidates.append((sum(row["bytes"] for row in data["cases"]), index, meta, shape))
    require(len(candidates) == 4, "All four remaining source-disjoint exact-clock groups required")
    held = []
    for _, index, meta, shape in sorted(candidates):
        require(meta["url"] not in links and not meta["donors"] & donors, "Held donor or source overlap")
        held.append({**original.encoded.prior.archive_io.source_pin(index, archive), "shape": list(shape), "role": "held"})
        links.add(meta["url"])
        donors.update(meta["donors"])
    development = [row for row in older[1]["roster"] if row["role"] == "development"]
    require(len(development) == 2, "Use the two previously measured development groups")
    return {"spec": SPEC, "bindings": base, "worker_sha256": common.digest(Path(__file__)),
            "prior_freezes": [common.digest(folder / "freeze.json") for folder in PRIOR],
            "prior_audits": [common.digest(folder / "independent-audit.json") for folder in PRIOR],
            "roster": [*development, *held], "prior_development_groups": 2, "earlier_held_overlap_by_known_key": 0}


def frozen(folder, archive):
    lock = common.read(folder / "freeze.json")
    require(lock == planned(archive), "Prospective study inputs or method changed")
    return lock


def clock(path):
    table = common.probe_frames(path, timeout_s=20)
    require((table.width, table.height) == original.DIMENSIONS and not (table.rotation or table.truncated or table.interpolated)
            and common.relative_times(table) == [Fraction(i,25) for i in range(original.N)], "Delivery native clock differs")
    return table


def encode(path, codec, quality, *, pictures=None, source=None):
    inputs = ("-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "640x360", "-framerate", "25", "-i", "pipe:0") if pictures is not None else (
        "-threads", "1", "-i", str(source))
    argv = ("ffmpeg", "-nostdin", "-n", "-v", "error", *inputs, "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
            "-pix_fmt", "yuv420p", "-threads", "2", *CODECS[codec]["options"], "-crf", str(quality), "-fflags", "+bitexact", str(path))
    if pictures is None:
        subprocess.run(argv, check=True, timeout=60)
        return
    with subprocess.Popen(argv, stdin=subprocess.PIPE) as process:
        for image in pictures:
            process.stdin.write(image.tobytes())
        process.stdin.close()
        require(process.wait(timeout=60) == 0, "Codec output failed")


def codec_of(path):
    return subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
                                    "-of", "csv=p=0", str(path)], timeout=8, text=True).strip()


def describe(path, group, role, codec, condition, truth):
    table = clock(path)
    require(codec_of(path) == f"{codec},yuv420p", "Actual codec or pixel format differs")
    hashes = []
    for frame in common.decode_frames(path, table, range(original.N), timeout_s=40):
        require(frame.index == len(hashes) and frame.pts_verified and frame.pts == table.pts[frame.index], "Native output PTS unverified")
        hashes.append(hashlib.sha256(frame.rgb.tobytes()).hexdigest())
    require(len(hashes) == original.N, "Encoded output frame roster is incomplete")
    return {"file": path.name, "sha256": common.digest(path), "group": group, "role": role, "codec": codec,
            "condition": condition, "truth": truth, "clock": table.to_json(), "rgb_sha256": hashes}


def span(condition, group):
    phase = (group + (condition == "swap_b")) % 2 if condition.startswith("swap_") else int(condition == "ordinary_blur")
    return original.SPANS[phase]


def source_frames(archive, pin, group):
    shape, cases, root = durations.source_set(pin, archive)
    original_case = cases["original"]
    plain = original.encoded.native_frames(root / original_case["file"], original_case, shape, range(original.N))
    changed = {}
    for condition in (*ORDINARY[2:], "swap_a", "swap_b"):
        case = cases[condition]
        wanted = range(*span(condition, group))
        altered = original.encoded.native_frames(root / case["file"], case, shape, wanted)
        require(all(not np.array_equal(plain[i], altered[i]) for i in wanted), "An authored interval lost source picture changes")
        changed[condition] = altered
    return plain, changed


def filename(folder, group, codec, condition):
    return folder / f"g{group}-{codec}-{condition}.{CODECS[codec]['suffix']}"


def prepare(folder, archive):
    plan = frozen(folder, archive)
    media = folder / "media"
    media.mkdir(mode=0o700)
    pair, rows = [], []
    for group, pin in enumerate(plan["roster"]):
        plain, changed = source_frames(archive, pin, group)
        role = pin["role"]
        for codec, config in CODECS.items():
            base = filename(media, group, codec, "original")
            encode(base, codec, config["primary"], pictures=plain.values())
            rows.append(describe(base, group, role, codec, "original", None))
            tandem = filename(media, group, codec, "recompressed")
            encode(tandem, codec, config["tandem"], source=base)
            rows.append(describe(tandem, group, role, codec, "recompressed", None))
            for condition, edits in changed.items():
                path = filename(media, group, codec, condition)
                encode(path, codec, config["primary"], pictures=(edits.get(i, pixel) for i,pixel in plain.items()))
                truth = list(span(condition, group)) if condition.startswith("swap_") else None
                rows.append(describe(path, group, role, codec, condition, truth))
        pair.append(plain)
        if len(pair) == 2:
            for codec, config in CODECS.items():
                for phase, (base, donor) in enumerate((pair, pair[::-1])):
                    a,b = original.SPANS[phase]
                    require(all(not np.array_equal(base[i], donor[i]) for i in range(a,b)), "Cut altered no source pictures")
                    g = group-1+phase
                    cut = filename(media, g, codec, "hard_cut")
                    encode(cut, codec, config["primary"], pictures=(donor[i] if a <= i < b else base[i] for i in range(original.N)))
                    rows.append(describe(cut, g, role, codec, "hard_cut", None))
            pair.clear()
        print("prepared", role, group, flush=True)
    require(len(rows) == len({row["sha256"] for row in rows}) == 96, "Incomplete encoded panel")
    common.write(folder / "media.json", {"freeze_sha256": common.digest(folder / "freeze.json"), "media": rows})


def run(folder, archive, role):
    frozen(folder, archive)
    manifest = common.read(folder / "media.json")
    require(manifest["freeze_sha256"] == common.digest(folder / "freeze.json"), "Media manifest commitment differs")
    if role == "held":
        journal(folder, "development")
    cv2.setNumThreads(1)
    models = common.load_yunet(common.FACE_FILE), cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE), "")
    started = time.monotonic()
    count = 0
    with (folder / f"{role}.jsonl").open("x") as stream:
        for row in manifest["media"]:
            if row["role"] != role:
                continue
            arms, sampled, supported = original.compare(folder / "media" / row["file"], row, models)
            old = arms["eight_hz"]
            fallback = arms["dense_rank"] if old is not None and not old["selected"] else None
            stream.write(json.dumps({k: row[k] for k in ("file", "group", "codec", "condition", "sha256", "truth")}
                       | {"coarse": arms["coarse"], "eight_hz": old, "fallback": fallback, "sampled": sampled, "supported": supported},
                         allow_nan=False) + "\n")
            stream.flush()
            count += 1
            print(role, row["codec"], row["group"], row["condition"], flush=True)
    require(count == (32 if role == "development" else 64), "Complete role required")
    common.write(folder / f"{role}-seal.json", {"media_sha256": common.digest(folder / "media.json"),
          "journal_sha256": common.digest(folder / f"{role}.jsonl"), "elapsed_s": time.monotonic()-started,
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def journal(folder, role):
    path = folder / f"{role}.jsonl"
    seal = common.read(folder / f"{role}-seal.json")
    require(seal["journal_sha256"] == common.digest(path) and seal["media_sha256"] == common.digest(folder / "media.json"),
            "Scored or authored journal differs")
    return list(map(json.loads, path.read_text().splitlines()))


def totals(rows):
    records, additions = defaultdict(lambda: defaultdict(Counter)), defaultdict(set)
    for row in rows:
        codec, truth = row["codec"], row["truth"]
        kind = "swap" if truth else "cut" if row["condition"] == "hard_cut" else "ordinary"
        earlier = [row[k]["frames"] for k in ("coarse", "eight_hz") if row[k] is not None and row[k]["selected"]]
        located = bool(truth and any(original.iou(guess, truth) >= Fraction(1,2) for guess in earlier))
        flag = bool(row["fallback"] and row["fallback"]["selected"])
        correct = bool(truth and flag and original.iou(row["fallback"]["frames"], truth) >= Fraction(1,2))
        if correct and not located:
            additions[codec].add(row["group"])
        records[codec][kind].update({"cases":1,"incumbent_flags":bool(earlier),"incumbent_located":located,"fallback_flags":flag,
               "new_located":correct and not located,"combined_located":correct or located,"measured_fallback":row["fallback"] is not None,
               "misplaced_fallback":bool(truth and flag and not correct)})
    result = {}
    gate = SPEC["gate_per_codec"]
    for codec, rows in records.items():
        checks = {"swap_locations":rows["swap"]["new_located"] >= gate["added_correct"],"source_groups":len(additions[codec]) >= gate["added_groups"],
                  "ordinary":rows["ordinary"]["fallback_flags"] <= gate["ordinary_extra_max"],"cut":rows["cut"]["fallback_flags"] <= gate["cut_extra_max"],
                  "coverage":all(rows[k]["measured_fallback"] >= gate["coverage_min"]*rows[k]["cases"] for k in ("swap","ordinary"))}
        result[codec] = {"counts":rows,"criteria":checks,"advances":all(checks.values()),"added_groups":len(additions[codec])}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "development", "held", "report"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    folder,archive = args.folder,args.archive
    if args.phase == "freeze":
        lock = planned(archive)
        folder.mkdir(parents=True,mode=0o700)
        common.write(folder / "freeze.json", lock)
    elif args.phase == "prepare":
        cv2.setNumThreads(1)
        prepare(folder, archive)
    elif args.phase == "report":
        frozen(folder, archive)
        held = journal(folder,"held")
        common.write(folder / "aggregate.json", {"protocol":SPEC,"development":totals(journal(folder,"development")),"held":totals(held),
             "runtime":common.read(folder / "held-seal.json"),"sampled_native_pictures":sum(row["sampled"] for row in held)})
    else:
        run(folder, archive, args.phase)


if __name__ == "__main__":
    main()
