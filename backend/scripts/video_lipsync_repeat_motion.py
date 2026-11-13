# SPDX-License-Identifier: Apache-2.0
"""Test signed mouth-motion agreement on repeated short speech in a single video."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio, wav_bytes
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import lipsync
from app.detectors.video.decode import probe_frames
from app.evidence import audio_similarity
from scripts import video_lipsync_repeat_audio as prior
from scripts.video_common_sources_v89 import read, require, sha, write

SCHEMA = "chipmunk.video-lipsync-repeat-motion/1"
GRID = (.2, .35, .5, .65)
SPLITS = ("development", "retained", "transfer")
GATE = {"added": 6, "groups": 4, "lost": 0, "ordinary_added": 1, "synchronized": 0, "coverage": .75}
DESIGN = {
    "nomination": "Same active two-band repeat nomination as the retained static mouth study.",
    "motion": "Compare signed, central eye-aligned mouth picture differences on consecutive native pairs. Search one common small translation between sequences. Reject activity or geometry gaps, matching native mouth pixels, and closely matching wide pictures.",
    "point": "From four fixed distances, maximize correct development pairs with zero flagged development controls, at least one correct pair, and strict exceedance. Use the most conservative tied point. Use the largest predeclared point if none qualify.",
    "roles": "Rebuild the previous four development and eight retained source-disjoint qualified actors. Independently acquire the already archived source bytes from two disjoint recent paired picture studies and prepare the first exact eligible 2.52 s extent. Keep every planned source and report infeasible inputs.",
    "advance": "In both carriers on transfer only, add six correct locations over the static comparison across four new actors, keep its correct locations, add at most one ordinary alert and none on synchronized or whole-track controls, and assess at least 75% of cases. A location must overlap both authored witness ranges by at least half. No AI confidence weight.",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_roster(args):
    original = read(args.previous / "freeze.json")
    require(original["sources_sha256"] == sha(args.sources / "intake-receipt.json"), "Earlier source intake differs")
    rows = [{"bank": "repeat", "split": "development" if row["group"] <= 4 else "retained",
             "actor": row["actor"], "group": row["group"], "ordinal": row["ordinal"],
             "source": str(args.sources / row["file"]), "sha256": row["sha256"]}
            for row in original["source_rows"]]
    known = {row["actor"] for row in rows}
    for j, manifest in enumerate(args.transfer):
        panel = read(manifest)
        for i, row in enumerate(panel["candidate_sources"]):
            prefix, actor = row["source_group"].split(":")
            require(prefix == "CREMA-D" and int(actor) not in known, "Transfer actor is not source-disjoint")
            known.add(int(actor))
            rows.append({"bank": f"transfer-{j + 1}", "split": "transfer", "actor": int(actor),
                         "group": 101 + j * 8 + i, "ordinal": row["asset_index"],
                         "source": row["file"], "sha256": row["file_sha256"]})
    require(sum(row["split"] == "transfer" for row in rows) == 16, "Incomplete transfer intake")
    return {"schema": SCHEMA, "design": DESIGN, "gate": GATE, "grid": list(GRID), "rows": rows,
            "prior_freeze_sha256": sha(args.previous / "freeze.json"),
            "transfer_freeze_sha256": [sha(p) for p in args.transfer],
            "bindings": prior.binding() | {Path(__file__).name: sha(__file__)},
            "opencv": cv2.__version__, "numpy": np.__version__,
            "ffmpeg": prior.delivery.command("ffmpeg", "-version").decode().splitlines()[0]}


def consecutive_pairs(first, second, observed):
    step = prior.HOP
    pairs = [(i, (i * step + step // 2 + second[0] - first[0]) // step) for i in range(prior.N)
             if first[0] + step // 4 <= i * step + step // 2 < first[1] - step // 4]
    kept = [p for p in pairs if all(index in observed for index in p)]
    if len(kept) >= 5 and len(kept) * 3 >= 2 * len(pairs):
        return kept
    return []


def motion_distance(first, second, observed):
    pairs = consecutive_pairs(first, second, observed)
    if not pairs:
        return None
    left, right = [[observed[p[side]] for p in pairs] for side in (0, 1)]
    pose = np.linalg.norm(np.stack([a[2] - b[2] for a, b in zip(left, right, strict=True)]), axis=1)
    wide = [float(np.mean(np.abs(a[1] - b[1]))) for a, b in zip(left, right, strict=True)]
    if np.quantile(pose, .8) > 7 or np.median(wide) < 1.4:
        return None
    correlation = [prior.translation_correlation((a[5], a[6]), (b[5], b[6]), 12, 8, 5)
                   for a, b in zip(left, right, strict=True)]
    upper = [prior.translation_correlation((a[5], a[7]), (b[5], b[7]), 25, 15, 5)
             for a, b in zip(left, right, strict=True)]
    same_mouth = sum(m >= .90 or (m >= .82 and m >= u) for m, u in zip(correlation, upper, strict=True))
    if same_mouth * 3 >= len(pairs) * 2:
        return None
    return signed_distance(left, right, pairs)


def signed_distance(left, right, pairs):
    contiguous = [i for i, (a, b) in enumerate(zip(pairs, pairs[1:], strict=False)) if (b[0] - a[0], b[1] - a[1]) == (1, 1)]
    if len(contiguous) < 4:
        return None
    motion = [np.diff(np.stack([r[0] for r in side]), axis=0)[contiguous] for side in (left, right)]
    a, b = motion
    # Use a common offset for the entire repeated sequence. Separately fitting each image can remove real motion.
    central = a[:, 3:15, 4:26]
    if np.quantile(np.mean(np.abs(central), axis=(1, 2)), .65) < 1.5:
        return None
    if np.quantile(np.mean(np.abs(b[:, 3:15, 4:26]), axis=(1, 2)), .65) < 1.5:
        return None
    a_vector = central - central.mean(axis=(1, 2), keepdims=True)
    best = -1.0
    for y in range(-2, 3):
        for x in range(-3, 4):
            shifted = b[:, 3 + y:15 + y, 4 + x:26 + x]
            b_vector = shifted - shifted.mean(axis=(1, 2), keepdims=True)
            scale = float(np.linalg.norm(a_vector) * np.linalg.norm(b_vector))
            if scale > 0:
                best = max(best, float(np.clip(np.sum(a_vector * b_vector) / scale, -1, 1)))
    return round(1 - best, 9), pairs


def observe(path, detector):
    table, observed, patches, faces, pictures = prior.video_inputs(path, detector)
    for index, record in faces.items():
        gray, (x, y) = observed[index][5:7] if index in observed else (None, (0, 0))
        if gray is not None and 12 <= x <= gray.shape[1] - 12 and 8 <= y <= gray.shape[0] - 8:
            box = [x - 12, y - 8, x + 12, y + 8]
            x0, y0, x1, y1 = box
            pictures[index].update(native_mouth_xyxy=box, native_gray_sha256=digest(gray[y0:y1, x0:x1].tobytes()))
    return table, observed, patches, faces, pictures


def inspect(pcm, geometry, measure):
    table, observed, patches, faces, pictures = geometry
    data = np.asarray(pcm, dtype="<i2").tobytes()
    signal = pcm.astype(np.float32) / 32768
    speech = lipsync.vad(signal).speech
    acoustic = audio_similarity.find_similar_spans(data)
    old = prior.matched(pcm, observed, pictures)
    current = prior.baseline.current(faces, patches, signal,
                                    lipsync.tick_grid(table.times(), prior.N / 25, len(pcm), audio_start_s=0.0))
    choices = []
    for span in acoustic.spans:
        first, second = span.first, span.second
        if first[1] - first[0] < 6 * prior.HOP:
            continue
        if min(speech[a // prior.HOP:b // prior.HOP].mean() for a, b in (first, second)) < .4:
            continue
        if (observed_motion := measure(first, second, observed)) is not None:
            distance, pairs = observed_motion
            choices.append({"samples": [list(first), list(second)], "distance": distance,
                            "frame_pairs": pairs, "pcm_sha256": [prior.pcm_sha(pcm[a:b]) for a, b in (first, second)]})
    return {"current": {"measured": bool(current["measured_windows"]), "ranges": current["ranges"]}, "assessed": old["assessed"], "similar_audio": old["similar_audio"],
            "static": [item["samples"] for item in old["matches"]["native_mouth_guard"]], "motion": choices}


def assess(run, row, source, donor, detector, measure):
    frames, pcm, *_ = source
    i = row["ordinal"]
    sample = 14 + i % 3
    faces = [detect_faces(cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
             for k in (sample, sample + 5, sample + 10)]
    if any(len(f) != 1 for f in faces):
        return None
    mouth = np.median([item[0].landmarks[3:5] for item in faces], axis=0)
    width = float(np.linalg.norm(mouth[0] - mouth[1]))
    x, y = mouth.mean(axis=0)
    w, h = round(width * 2.25), round(width * 1.55)
    box = [round(x - w / 2), round(y - h / 2), w, h]
    require(box[0] > 0 and box[1] > 0 and box[0] + w < 480 and box[1] + h < 360, "Mouth crop differs")
    cases = []
    with TemporaryDirectory(prefix="media-", dir=run) as temp:
        folder = Path(temp)
        picture = folder / "unchanged.mov"
        prior.picture(frames, picture)
        videos = {"base": picture, "whole": folder / "picture-copy.mov", "mouth": folder / "mouth-only.mov"}
        geometries = {}
        for role, view, wave, truth in prior.edits(frames, pcm, donor[1], i, box):
            variant = "whole" if role in {"picture_copy", "matched_av"} else "mouth" if role == "matched_mouth_sound" else "base"
            if role in {"picture_copy", "matched_mouth_sound"}:
                prior.picture(view, videos[variant])
            wav = folder / "audio.wav"
            wav.write_bytes(wav_bytes(wave.astype(np.float32) / 32768))
            for carrier, (ext, options) in prior.CODECS.items():
                target = folder / f"delivery.{ext}"
                prior.delivery.remux(videos[variant], wav, target, options)
                source_hash = sha(target)
                decoded = decode_audio(target, source_sha256=source_hash, max_seconds=4)
                start, signal, _ = prior.clock.resample_clock(decoded)
                require(start == 0 and prior.N * prior.HOP <= len(signal) <= prior.N * prior.HOP + 1024,
                        "Delivered audio clock differs")
                require(carrier != "pcm" or np.array_equal(signal, wave), "Delivered PCM differs")
                if variant not in geometries:
                    geometries[variant] = observe(target, detector)
                table, _, _, _, pictures = geometry = geometries[variant]
                require(probe_frames(target, timeout_s=6) == table, "Remux changed picture clock")
                result = inspect(signal, geometry, measure)
                record = {"role": role, "carrier": carrier, "truth": truth, "sha256": source_hash,
                          "sample_count": len(signal), "pcm_sha256": prior.pcm_sha(signal), "time_base": table.time_base,
                          "pts_sha256": digest(str(table.pts).encode()), **result}
                flagged = [c for c in result["motion"] if c["distance"] > min(GRID)]
                if flagged:
                    anchor_frames = {pair[k] for c in flagged for pair in c["frame_pairs"] for k in (0, 1)}
                    fields = ("index", "pts", "rgb_sha256", "native_mouth_xyxy", "native_gray_sha256")
                    record["anchors"] = [{key: pictures[k][key] for key in fields} for k in sorted(anchor_frames)]
                    audit(target, record, table)
                cases.append(record)
                target.unlink()
    return {"group": row["group"], "bank": row["bank"], "source_sha256": row["sha256"], "cases": cases}


def audit(target, record, table):
    # Separate FFmpeg/FFprobe streams and independent audio slice hashing.
    command = prior.delivery.command
    data = json.loads(command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "frame=best_effort_timestamp:stream=time_base", "-of", "json", str(target)))
    pts = tuple(frame["best_effort_timestamp"] for frame in data["frames"])
    require(pts == table.pts and data["streams"][0]["time_base"] == record["time_base"], "Native audit clock differs")
    raw = command("ffmpeg", "-nostdin", "-v", "error", "-i", str(target), "-map", "0:a:0", "-ac", "1", "-ar",
                  "16000", "-f", "s16le", "-")
    require(len(raw) == 2 * record["sample_count"] and digest(raw) == record["pcm_sha256"], "Native audit audio differs")
    for choice in record["motion"]:
        require(all(digest(raw[2*a:2*b]) == h for (a, b), h in zip(choice["samples"], choice["pcm_sha256"], strict=True)),
                "Native audio excerpt differs")
    indices = [anchor["index"] for anchor in record["anchors"]]
    selection = "+".join(f"eq(n\\,{i})" for i in indices)
    rgb = command("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(target), "-an", "-vf",
                  f"select={selection}", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-")
    require(len(rgb) == len(indices) * 480 * 360 * 3, "Native audit frame count differs")
    for picture, image in zip(record["anchors"], np.frombuffer(rgb, np.uint8).reshape(-1, 360, 480, 3), strict=True):
        require(digest(image.tobytes()) == picture["rgb_sha256"], "Native audit source frame differs")
        x, y, x1, y1 = picture["native_mouth_xyxy"]
        crop = image[y:y1, x:x1]
        require(digest(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).tobytes()) == picture["native_gray_sha256"],
                "Native audit mouth crop differs")
        picture["native_mouth_sha256"] = digest(crop.tobytes())
    record["native_audit"] = {"timestamps": len(pts), "samples": len(raw) // 2, "crops": len(indices)}


def qualify(lock, phase):
    picks = {}
    for row in (r for r in lock["rows"] if r["split"] == phase):
        if row["group"] in picks:
            continue
        path = Path(row["source"])
        require(sha(path) == row["sha256"], "Frozen source bytes differ")
        if (original := prior.original(path, row["sha256"])) is not None:
            picks[row["group"]] = row, original
    return picks


def run_phase(run, lock, phase, measure):
    cv2.setNumThreads(1)
    detector = load_yunet(prior.WEIGHT)
    picks = qualify(lock, phase)
    absent = sorted({row["group"] for row in lock["rows"] if row["split"] == phase} - picks.keys())
    with (run / f"{phase}.jsonl").open("x") as stream:
        stream.write(json.dumps({"outside_shared_duration": absent}, sort_keys=True) + "\n")
        for i, (row, source) in enumerate(picks.values()):
            donor = list(picks.values())[(i + 1) % len(picks)][1]
            result = assess(run, row, source, donor, detector, measure)
            stream.write(json.dumps(result or {"unqualified_geometry": row["group"]}, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            print(phase, row["group"], len(result["cases"]) if result else "geometry decline", flush=True)


def main(*, freeze_roster=source_roster, measure=motion_distance):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", *SPLITS, "calibrate", "report"))
    p.add_argument("run", type=Path)
    p.add_argument("previous", type=Path)
    p.add_argument("sources", type=Path)
    p.add_argument("transfer", nargs=2, type=Path)
    args = p.parse_args()
    lock = freeze_roster(args)
    freeze = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(freeze, lock)
        return
    require(lock == read(freeze), "Frozen source or method differs")
    from scripts.review_video_lipsync_repeat_motion import calibrate, report

    if args.phase == "calibrate":
        rows = read_rows(args.run, "development")
        point = calibrate(rows, GRID)
        write(args.run / "calibration.json", {"freeze_sha256": sha(freeze), "point": point,
                                             "development_sha256": sha(args.run / "development.jsonl")})
        print(f"Development point {point}", flush=True)
        return
    if args.phase != "development":
        point = read(args.run / "calibration.json")
        require(point["freeze_sha256"] == sha(freeze) and point["development_sha256"] == sha(args.run / "development.jsonl"),
                "Calibrate on development before held inference")
    if args.phase == "report":
        rows = {phase: read_rows(args.run, phase) for phase in SPLITS}
        public = {key: val for key, val in lock.items() if key != "rows"}
        public.update(freeze_sha256=sha(freeze), point=point["point"], rows=rows)
        public["summary"] = report(public)
        write(args.run / "receipt.json", public)
        print(json.dumps(public["summary"], sort_keys=True), flush=True)
    else:
        run_phase(args.run, lock, args.phase, measure)


def read_rows(run, phase):
    return [json.loads(line) for line in (run / f"{phase}.jsonl").read_text().splitlines()]


if __name__ == "__main__":
    main()
