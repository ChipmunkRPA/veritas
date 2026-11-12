# SPDX-License-Identifier: Apache-2.0
"""Freeze separate native release observations and compare exact, local speech shifts."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
from scipy.signal import correlate

from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence.video_reference_timing import _mesh
from scripts import grid_fresh_release_labels as labels
from scripts import grid_paired_closure_panel as native
from scripts import grid_phone_alignment_v161 as words
from scripts import grid_reserve_neural_mouth as neural
from scripts import grid_reserve_neural_mouth_readout as current
from scripts import grid_reserve_word_clocks as grid
from scripts import grid_single_clip_release_timing as blind
from scripts import review_video_lipsync_local_delay_transfer as waveform
from scripts import video_lipsync_local_contrast_transfer as legacy
from scripts.video_common_sources_v89 import read, require, sha, write

RATE, TICK = 16000, 640
EXCLUDE = {"b", "p", "bin", "blue", "by", "place", "please"}
ROLES = {"original": 0, "gain": 0, "audio_early": -6, "audio_late": 6}
CONTRACT = {
    "roles": "Two each B/P and one lexical negative per held group, one each B/P per development group. Publisher word timing is rough context only. Previously planned members excluded. Exact selected footage is newly extracted, with prior speaker exposure.",
    "shift_pictures": ROLES,
    "known_edit": "240 ms ordinary sound copy, preserving all pictures, contiguous origin, and samples outside a half-open interval. Interval must contain the reference and moved waveform for both signs.",
    "oracle": "Match the original, independently annotated burst excerpt to analyzed sound by normalized correlation. This uses the supplied source waveform and chosen native picture event. It measures an upper bound with event identity, with no claim of single-file detection.",
    "review": "At least 120 ms change, two-frame maximum signed error and mapped acoustic anchor in the copied PCM range. Original and gain should never alert.",
}


def bindings():
    paths = (
        Path(__file__),
        Path(labels.__file__),
        Path(native.__file__),
        Path(current.__file__),
        Path(blind.__file__),
        Path(grid.__file__),
        Path(waveform.__file__),
        Path(legacy.baseline.__file__),
        native.geometry.WEIGHT,
    )
    return {path.name: sha(path) for path in paths}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def intake(run):
    source, identity = grid.plan()
    published = grid.publisher_record()["alignments.zip"]
    prior_members = {PurePosixPath(s["input"]["planned"]["member"]["name"]).stem for s in source.values()}
    with words.ALIGNMENTS.open("rb") as stream:
        checksum = hashlib.file_digest(stream, lambda: hashlib.md5(usedforsecurity=False)).hexdigest()
    require(
        published["checksum"] == f"md5:{checksum}" and published["size"] == words.ALIGNMENTS.stat().st_size,
        "Publisher word metadata differs",
    )
    run.mkdir(mode=0o700)
    assets = run / "sources"
    assets.mkdir(mode=0o700)
    output = []
    with ZipFile(words.ALIGNMENTS) as text_zip:
        directories = defaultdict(dict)
        for item in text_zip.infolist():
            path = PurePosixPath(item.filename)
            if (
                len(path.parts) == 3
                and path.parts[0] == "alignments"
                and path.suffix == ".align"
                and not path.name.startswith("._")
            ):
                require(path.stem not in directories[path.parent.name], "Duplicate alignment member")
                directories[path.parent.name][path.stem] = item
        for group in range(8, 16):
            item = next(s for s in source.values() if s["component_index"] == group)
            _, matched, media = grid.inventory(item, directories)
            archive = (
                grid.planning.ARCHIVE
                / "grid-acquisition-continuation-v135"
                / f"source-{item['source_slot']:02}"
                / "archive.zip"
            )
            candidates = defaultdict(list)
            for stem in sorted(
                media.keys() - prior_members,
                key=lambda s: digest(f"grid-native-release-v1/{group}/{s}".encode()),
            ):
                label_bytes = text_zip.read(directories[matched][stem])
                intervals, total = words.publisher_words(label_bytes)
                require(total == 74500, "Word alignment clock differs")
                types = {
                    "B": [(w, a, b) for w, a, b in intervals if w in {"blue", "by"} and 12500 <= a <= 45000],
                    "P": [(w, a, b) for w, a, b in intervals if w == "please" and 38500 <= a <= 57750],
                    "none": [(None, None, None)] if not any(w in EXCLUDE for w, _, _ in intervals) else [],
                }
                for role, targets in types.items():
                    if targets:
                        candidates[role].append((stem, targets[0], label_bytes))
            used = set()
            with ZipFile(archive) as video_zip:
                for role in ("B", "P") if group < 10 else ("B", "P", "none"):
                    count = 1 if group < 10 or role == "none" else 2
                    available = [x for x in candidates[role] if x[0] not in used]
                    require(len(available) >= count, "Fixed lexical roster unavailable")
                    for slot, (stem, target, aligned) in enumerate(available[:count], 1):
                        used.add(stem)
                        member = media[stem]
                        require(0 < member.file_size <= 1024**2, "Source MPEG entry exceeds fixed budget")
                        data = video_zip.read(member)
                        case = f"g{group:02}-{role}-{slot}"
                        filename = assets / f"{case}.mpg"
                        filename.write_bytes(data)
                        output.append(
                            {
                                "id": case,
                                "group": group,
                                "split": "development" if group < 10 else "held",
                                "kind": role,
                                "path": str(filename),
                                "source_sha256": digest(data),
                                "member": member.filename,
                                "source_archive": str(archive),
                                "member_crc32": member.CRC,
                                "word_alignment_sha256": digest(aligned),
                                "word": target[0],
                                "word_start_25k": target[1],
                                "word_end_25k": target[2],
                            }
                        )
            print("frozen source group", group, len(used), flush=True)
    require(
        Counter(item["split"] for item in output) == {"development": 4, "held": 30},
        "Incomplete frozen roster",
    )
    write(
        run / "freeze.json",
        {
            "schema": "chipmunk.grid-fresh-release-intake/1",
            "contract": CONTRACT,
            "label_rules": labels.RULES,
            "bindings": bindings(),
            "prior_plan_sha256": identity,
            "roster": output,
        },
    )


def roster(run):
    freeze = read(run / "freeze.json")
    require(
        freeze["bindings"] == bindings()
        and freeze["contract"] == CONTRACT
        and freeze["label_rules"] == labels.RULES,
        "Frozen code differs",
    )
    return freeze["roster"]


def check_source(item):
    path = Path(item["path"])
    require(sha(path) == item["source_sha256"], "Frozen original differs")
    return path


def audio_labels(run):
    cases = []
    for row in roster(run):
        path = check_source(row)
        pcm, origin, overlap = waveform.source_pcm(path, row["source_sha256"])
        require(not overlap and origin == 0 and len(pcm) == 47648, "Original PCM clock differs")
        word = round(Fraction(row["word_start_25k"] * RATE, 25000)) if row["word"] else None
        result = labels.audio_only(pcm, word) if word is not None else {"status": "lexical_negative"}
        if "range_samples" in result:
            a, b = result["range_samples"]
            result["burst_pcm_sha256"] = digest(pcm[a:b].tobytes())
        cases.append(
            {
                "id": row["id"],
                "origin_s": str(origin),
                "samples": len(pcm),
                "pcm_sha256": digest(pcm.tobytes()),
                **result,
            }
        )
    write(run / "audio-labels.json", {"freeze_sha256": sha(run / "freeze.json"), "cases": cases})
    print(Counter(r["status"] for r in cases), flush=True)


def visual_labels(run):
    cases = []
    module, task = neural.mp_task()
    with closing(task):
        for row in roster(run):
            path = check_source(row)
            table = probe_frames(path)
            require(
                not (table.rotation or table.truncated or table.interpolated)
                and (table.width, table.height) == (360, 288)
                and [p * table.tb for p in table.pts] == [Fraction(i, 25) for i in range(75)],
                "Native picture clock differs",
            )
            native_pictures, descriptors, boxes = [], [], []
            for frame in decode_frames(path, table, None, timeout_s=25):
                require(
                    frame.pts_verified and frame.pts == table.pts[frame.index], "Native picture proof differs"
                )
                result = _mesh(task, module, frame.rgb) if row["word"] else None
                descriptors.append(None if result is None else result[0])
                boxes.append(None if result is None else result[1])
                native_pictures.append(frame.rgb)
            word = round(Fraction(row["word_start_25k"] * RATE, 25000)) if row["word"] else None
            result = (
                labels.muted_video_only(descriptors, word)
                if word is not None
                else {"status": "lexical_negative"}
            )
            if "bracket_frames" in result:
                result["native_crops"] = []
                for index in result["bracket_frames"]:
                    x0, y0, x1, y1 = boxes[index]
                    rgb = native_pictures[index]
                    result["native_crops"].append(
                        {
                            "frame": index,
                            "pts": table.pts[index],
                            "time_base": table.time_base,
                            "xyxy": boxes[index],
                            "rgb_sha256": digest(rgb.tobytes()),
                            "crop_sha256": digest(rgb[y0:y1, x0:x1].tobytes()),
                        }
                    )
            cases.append(
                {
                    "id": row["id"],
                    "time_base": table.time_base,
                    "pts": list(table.pts),
                    "observed": sum(x is not None for x in descriptors),
                    **result,
                }
            )
    write(run / "visual-labels.json", {"freeze_sha256": sha(run / "freeze.json"), "cases": cases})
    print(Counter(r["status"] for r in cases), flush=True)


def seal_labels(run):
    a, v = (read(run / (name + "-labels.json")) for name in ("audio", "visual"))
    require(
        a["freeze_sha256"] == v["freeze_sha256"] == sha(run / "freeze.json"), "Native label ledgers differ"
    )
    views = []
    for source, audio, visual in zip(roster(run), a["cases"], v["cases"], strict=True):
        require(source["id"] == audio["id"] == visual["id"], "Label roster differs")
        row = {
            "id": source["id"],
            "group": source["group"],
            "split": source["split"],
            "kind": source["kind"],
            "audio_status": audio["status"],
            "visual_status": visual["status"],
            "status": "lexical_negative" if source["kind"] == "none" else "unsupported",
        }
        if "range_samples" in audio and "bracket_frames" in visual:
            sound = sum(audio["range_samples"]) // 2
            picture = sum(visual["bracket_frames"]) * TICK // 2
            first = max(6 * TICK, sound - 6 * TICK - 960)
            last = min(audio["samples"] - 6 * TICK, sound + 6 * TICK + 1280)
            if (
                abs(sound - picture) <= 1920
                and first <= sound - 6 * TICK - 480
                and last >= sound + 6 * TICK + 800
            ):
                row.update(
                    status="supported",
                    original_audio_center=sound,
                    original_picture_midpoint=picture,
                    span_samples=[first, last],
                    alternate_transients=audio["distractors"],
                )
            else:
                row["status"] = "unresolved_correspondence_or_bounds"
        views.append(row)
    write(
        run / "label-freeze.json",
        {
            "freeze_sha256": sha(run / "freeze.json"),
            "audio_sha256": sha(run / "audio-labels.json"),
            "visual_sha256": sha(run / "visual-labels.json"),
            "cases": views,
        },
    )
    print(Counter((x["split"], x["status"]) for x in views), flush=True)


def frozen_labels(run):
    rows = read(run / "label-freeze.json")
    require(
        rows["freeze_sha256"] == sha(run / "freeze.json")
        and rows["audio_sha256"] == sha(run / "audio-labels.json")
        and rows["visual_sha256"] == sha(run / "visual-labels.json"),
        "Label freeze differs",
    )
    return rows["cases"]


def author(run):
    folder = run / "media"
    folder.mkdir(mode=0o700)
    output = []
    for row, proof in zip(roster(run), frozen_labels(run), strict=True):
        if row["split"] != "held" or proof["status"] not in {"supported", "lexical_negative"}:
            continue
        path = check_source(row)
        table, pixels, pcm = native.video(path)
        hashes = [digest(x.tobytes()) for x in pixels]
        for role, delay in ROLES.items() if proof["status"] == "supported" else (("original", 0),):
            out, expected = path, pcm.copy()
            interval = proof.get("span_samples")
            if role != "original":
                start, end = interval
                expected[start:end] = (
                    pcm[start - delay * TICK : end - delay * TICK]
                    if delay
                    else np.rint(0.72 * pcm[start:end]).astype("<i2")
                )
                out = folder / f"{row['id']}-{role}.mkv"
                wav = out.with_suffix(".wav")
                native.save_pcm(wav, expected)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-i",
                        str(path),
                        "-i",
                        str(wav),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "pcm_s16le",
                        str(out),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=15,
                )
                wav.unlink()
                observed, after, exact = native.video(out)
                require(
                    [p * observed.tb for p in observed.pts] == [p * table.tb for p in table.pts]
                    and hashes == [digest(x.tobytes()) for x in after]
                    and np.array_equal(expected, exact),
                    "Exact authored delivery differs",
                )
                require(
                    np.array_equal(expected[:start], pcm[:start])
                    and np.array_equal(expected[end:], pcm[end:]),
                    "Edit escaped half-open interval",
                )
            output.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "role": role,
                    "shift": delay,
                    "file": str(out),
                    "file_sha256": sha(out),
                    "pcm_sha256": digest(expected.tobytes()),
                    "native_rgb": hashes,
                    "span_samples": interval,
                }
            )
    write(run / "media.json", {"labels_sha256": sha(run / "label-freeze.json"), "cases": output})
    print("authored views", len(output), flush=True)


def match_template(reference, submitted, center):
    start, stop = center - 320, center + 640
    template = reference[start:stop].astype(float)
    template -= template.mean()
    length = len(template)
    lo, hi = max(0, start - 6400), min(len(submitted) - length, start + 6400)
    target = submitted[lo : hi + length].astype(float)
    sums = np.r_[0, np.cumsum(target)]
    sq = np.r_[0, np.cumsum(target**2)]
    dev = np.sqrt(np.maximum(0, sq[length:] - sq[:-length] - (sums[length:] - sums[:-length]) ** 2 / length))
    values = np.clip(
        correlate(target, template, mode="valid", method="fft")
        / np.maximum(1e-12, np.linalg.norm(template) * dev),
        -1,
        1,
    )
    best = int(values.argmax())
    other = float(max(values[np.abs(np.arange(len(values)) - best) >= 640], default=-1))
    if values[best] < 0.94 or values[best] - other < 0.08:
        return None
    found = best + lo
    return {
        "offset_samples": found - start,
        "source_samples": [start, stop],
        "matched_samples": [found, found + length],
        "source_slice_sha256": digest(reference[start:stop].tobytes()),
        "matched_slice_sha256": digest(submitted[found : found + length].tobytes()),
        "correlation": float(values[best]),
        "margin": float(values[best] - other),
        "timing_difference": abs(found - start) >= 1920,
    }


def score(run):
    by_label = {r["id"]: r for r in frozen_labels(run)}
    inputs = read(run / "media.json")
    require(inputs["labels_sha256"] == sha(run / "label-freeze.json"), "Media and label freeze differ")
    cv2.setNumThreads(1)
    detector = load_yunet(native.geometry.WEIGHT)
    module, task = neural.mp_task()
    original, results = {}, []
    with closing(task):
        for row in inputs["cases"]:
            path = Path(row["file"])
            require(sha(path) == row["file_sha256"], "Scored media file differs")
            _, pixels, pcm = native.video(path)
            require(
                [digest(frame.tobytes()) for frame in pixels] == row["native_rgb"]
                and digest(pcm.tobytes()) == row["pcm_sha256"],
                "Native scored bytes differ",
            )
            label = by_label[row["id"]]
            if row["role"] == "original":
                exact, clock, faces, patches, _, _ = current.observe(path, row["file_sha256"], detector)
                require(np.array_equal(exact, pcm), "Current and candidate audio differs")
                desc = [_mesh(task, module, frame) for frame in pixels]
                original[row["id"]] = (
                    pcm,
                    clock,
                    faces,
                    patches,
                    [None if v is None else v[0] for v in desc],
                )
            ref, clock, faces, patches, geometry = original[row["id"]]
            incumbent = legacy.baseline.current(faces, patches, pcm.astype(np.float32) / 32768, clock)
            single = blind.measure(geometry, pcm)
            oracle = (
                match_template(ref, pcm, label["original_audio_center"])
                if label["status"] == "supported"
                else None
            )
            expected = row["shift"] * TICK
            span = row["span_samples"]
            located = bool(
                expected
                and oracle
                and oracle["timing_difference"]
                and abs(oracle["offset_samples"] - expected) <= 2 * TICK
                and span[0] <= oracle["matched_samples"][0] < oracle["matched_samples"][1] <= span[1]
            )
            blind_correct = bool(
                expected
                and single["status"] == "local_timing_change"
                and abs(single["audio_minus_picture_change_samples"] - expected) <= 2 * TICK
                and span[0] <= single["events"][1]["acoustic_anchor"] < span[1]
            )
            current_correct = bool(
                expected
                and any(
                    legacy.qualifies(flag, [Fraction(s, TICK) for s in span], row["shift"])
                    for flag in incumbent["ranges"]
                )
            )
            results.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "role": row["role"],
                    "shift_samples": expected,
                    "current": incumbent,
                    "blind": single,
                    "oracle": oracle,
                    "correct": [current_correct, blind_correct, located],
                }
            )
    write(run / "readout.json", {"media_sha256": sha(run / "media.json"), "rows": results})
    print(
        "scored",
        len(results),
        "correct",
        [sum(r["correct"][i] for r in results) for i in range(3)],
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = {
        "intake": intake,
        "audio": audio_labels,
        "visual": visual_labels,
        "seal": seal_labels,
        "author": author,
        "score": score,
    }
    parser.add_argument("phase", choices=phases)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    phases[args.phase](args.run)


if __name__ == "__main__":
    main()
