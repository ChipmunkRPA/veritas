# SPDX-License-Identifier: Apache-2.0
"""Compare whole-picture and local picture gates for four-second HD soundtrack pairs."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import wav_bytes
from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import audio_reference, video_reference_soundtrack
from app.evidence.video_reference_soundtrack_reencode import POINTS
from scripts import video_lipsync_repeat_public_speech as speech
from scripts import video_lipsync_window_schedule as earlier
from scripts.video_common_sources_v89 import read, require, sha, write
from scripts.video_lipsync_local_delay_transfer import shift
from scripts.video_reference_soundtrack_panel import remux

WIDTH, HEIGHT, FPS, PICTURES = 1280, 720, 30, 120
SR, SAMPLES = 16000, 64000
CASES = ("reencode", "shift_earlier", "shift_later", "gain", "filter", "constant", "synchronized_picture",
         "synchronized_mouth", "picture_only", "alternate_speech")
GRIDS = {"coarse": (80, 60), "fine": (160, 90)}
SCHEMA = "chipmunk.video-soundtrack-hd-transfer/1"


def lock(source_dir):
    sources = read(earlier.CONVERSATION)["source_plan"]["rows"]
    rows = [{"group": index, "name": item["name"], "sha256": item["source_sha256"], "offset": item["offset_s"] + 20,
             "split": "development" if index <= 3 else "held"} for index, item in enumerate(sources, 1)]
    public = read(speech.SOURCES)["sources"]
    for index, item in enumerate(public, 12):
        if item["id"] != "diego":
            rows.append({"group": index, "name": item["id"], "sha256": item["sha256"], "offset": item["offsets"][0] + 9,
                         "split": "held"})
    require(all(sha(source_dir / (row["name"] + ".webm")) == row["sha256"] for row in rows),
            "The video sources do not match the publisher commitments")
    return {"schema": SCHEMA, "sources": rows, "bindings": {Path(p).name: sha(p) for p in
            (__file__, earlier.CONVERSATION, speech.SOURCES, video_reference_soundtrack.__file__, earlier.FACE_FILE)},
            "runtime": {"ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                        "opencv": cv2.__version__, "numpy": np.__version__},
            "point": dict(POINTS), "scan_size": [WIDTH, HEIGHT], "cadence": FPS, "pictures": PICTURES,
            "gate": "No added synchronized, global, alternate speech or unshifted flags on held cases. At least half of "
                    "planned edits must be correctly located within 16 samples of the signed shift and at IoU at least 0.60. "
                    "Compare both grids using fixed bounds from the 480x360 study. Each carrier must meet the criterion."}


def execute(*args, data=None):
    return subprocess.run(args, input=data, check=True, capture_output=True, timeout=35).stdout


def excerpt(source, target, offset):
    execute("ffmpeg", "-nostdin", "-v", "error", "-ss", str(offset), "-i", str(source),
            "-map", "0:v:0", "-map", "0:a:0", "-t", "4",
            "-vf", "fps=30,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1",
            "-video_track_timescale", "48000", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(target))


def pictures(path):
    clock = probe_frames(path, timeout_s=10)
    require(len(clock) == PICTURES and not (clock.truncated or clock.interpolated or clock.rotation)
            and (clock.width, clock.height) == (WIDTH, HEIGHT)
            and all((pts - clock.pts[0]) * clock.tb == Fraction(i, FPS) for i, pts in enumerate(clock.pts)),
            "The comparison requires an exact, complete HD frame clock")
    result = np.empty((PICTURES, HEIGHT, WIDTH, 3), dtype=np.uint8)
    count = 0
    for frame in decode_frames(path, clock, list(range(PICTURES)), timeout_s=22):
        require(frame.index == count and frame.pts_verified and frame.pts == clock.pts[count],
                "A decoder returned an unverified picture")
        result[count] = frame.rgb
        count += 1
    require(count == PICTURES, "A delivered frame is missing")
    return result


def encode(path, pixels, crf):
    execute("ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-threads", "1",
            "-video_track_timescale", "48000", str(path), data=pixels.tobytes())


def _views(sequence):
    return {name: np.asarray([cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), size,
                               interpolation=cv2.INTER_AREA) for frame in sequence], dtype=np.float32)
            for name, size in GRIDS.items()}


def difference(first, second):
    result = {}
    for name, size in GRIDS.items():
        width, height = size
        values = np.abs(first[name] - second[name])
        mean = values.mean(axis=(1, 2))
        cells = values.reshape(PICTURES, height // 5, 5, width // 5, 5).mean(axis=(2, 4))
        result[name] = dict(whole_q65=round(float(np.quantile(mean, .65)), 6),
                            whole_q90=round(float(np.quantile(mean, .9)), 6),
                            local_five_frame_median=round(float(np.median(
                                np.lib.stride_tricks.sliding_window_view(cells, 5, axis=0), axis=-1).max()), 6))
    return result


def _mouth(pixels, detector, a, b, displacement):
    positions = [a, (a + b) // 2, b - 1]
    boxes = []
    for i in positions + [i - displacement for i in positions]:
        small = cv2.resize(pixels[i], (640, 360), interpolation=cv2.INTER_AREA)
        faces = detect_faces(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), detector=detector, min_size=38)
        if len(faces) != 1:
            return None
        points = np.asarray(faces[0].landmarks[3:5]) * 2.
        x, y = points.mean(axis=0)
        distance = float(np.linalg.norm(points[0] - points[1]))
        boxes.append((x - 1.1 * distance, y - .7 * distance, x + 1.1 * distance, y + .7 * distance))
    x0, y0 = np.floor(np.min(boxes, axis=0)[:2]).astype(int)
    x1, y1 = np.ceil(np.max(boxes, axis=0)[2:]).astype(int)
    return (x0, y0, x1, y1) if 0 <= x0 < x1 <= WIDTH and 0 <= y0 < y1 <= HEIGHT else None


def _track(path, digest):
    table = probe_frames(path, timeout_s=6)
    result = audio_reference.read_track(path, {"sha256": digest}, max_seconds=6, video=True, origin=table.pts[0] * table.tb)
    require(result[0] == "measured" and result[2]["relative_start"] == "0"
            and SAMPLES <= len(result[1]) <= SAMPLES + 1024, "Unsupported native soundtrack clock")
    return result[1], result[2]["pcm_sha256"]


def score_group(item, media, scratch, detector, donor):
    ref = scratch / "original.mov"
    excerpt(media / (item["name"] + ".webm"), ref, item["offset"])
    data = pictures(ref)
    first = _views(data)
    initial, _ = _track(ref, sha(ref))
    original = initial[:SAMPLES]
    a, b, off = 41 + item["group"] % 8, 68 + item["group"] % 8, 6
    start, end = a * SR // FPS, b * SR // FPS
    span = (start, end)
    movement = off * SR // FPS
    box = _mouth(data, detector, a, b, off)
    cases = []
    codecs = {"PCM": ("mov", ("-c:a", "pcm_s16le")), "AAC": ("mp4", ("-c:a", "aac", "-b:a", "64k"))}
    submitted_paths = {}
    picture_results = {}
    for style in ("reencode", "picture", "mouth"):
        if style == "mouth" and box is None:
            continue
        path = scratch / f"{style}.mov"
        if style == "picture":
            selected = slice(a, b)
            prior = data[selected].copy()
            data[selected] = data[a - off:b - off].copy()
        elif style == "mouth":
            x0, y0, x1, y1 = box
            selected = (slice(a, b), slice(y0, y1), slice(x0, x1))
            prior = data[selected].copy()
            data[selected] = data[a - off:b - off, y0:y1, x0:x1].copy()
        encode(path, data, 31 if item["group"] % 2 else 34)
        if style != "reencode":
            data[selected] = prior
        submitted_paths[style] = path
        picture_results[style] = difference(first, _views(pictures(path)))
    del data
    original_path = scratch / "wave.wav"
    encoded = {}
    original_path.write_bytes(wav_bytes(original))
    for carrier, (ext, codec) in codecs.items():
        name = scratch / f"earlier-{carrier}.{ext}"
        remux(ref, original_path, name, codec)
        encoded[carrier] = _track(name, sha(name))

    def mixed(wave):
        result = original.copy()
        weights = np.ones(end - start, dtype=np.float32)
        weights[:80] = np.arange(80) / 80
        weights[-80:] = weights[:80][::-1]
        result[start:end] += weights * (wave[start:end] - original[start:end])
        return result

    later = mixed(shift(original, movement))
    filtering = sosfiltfilt(butter(4, 2500, fs=SR, output="sos"), original).astype(np.float32)
    roles = {"reencode": ("reencode", original), "shift_earlier": ("reencode", mixed(shift(original, -movement))),
             "shift_later": ("reencode", later), "gain": ("reencode", mixed(original * .55)),
             "filter": ("reencode", mixed(filtering)), "constant": ("reencode", shift(original, movement)),
             "synchronized_picture": ("picture", later), "synchronized_mouth": ("mouth", later),
             "picture_only": ("picture", original), "alternate_speech": ("reencode", mixed(donor))}
    for name, (style, wave) in roles.items():
        for carrier, (ext, codec) in codecs.items():
            if style not in submitted_paths:
                cases.append({"role": name, "carrier": carrier, "unavailable": "mouth_geometry"})
                continue
            original_path.write_bytes(wav_bytes(wave))
            path = scratch / f"{name}-{carrier}.{ext}"
            remux(submitted_paths[style], original_path, path, codec)
            input_sha = sha(path)
            decoded, audio_sha = _track(path, input_sha)
            prior, prior_sha = encoded[carrier]
            require(len(decoded) == len(prior), "Delivered track has an incompatible sample count")
            match, count = video_reference_soundtrack.measure(prior, decoded)
            if match and (min(match["submitted_samples"][0], match["earlier_samples"][0]) < 0
                          or max(match["submitted_samples"][1], match["earlier_samples"][1]) * FPS > (PICTURES - 1) * SR):
                match = None
            cases.append({"role": name, "carrier": carrier, "picture": picture_results[style], "sound": match,
                          "matched_windows": count, "sha256": input_sha,
                          "earlier_pcm_sha256": prior_sha, "submitted_pcm_sha256": audio_sha})
    return {"group": item["group"], "sha256": item["sha256"], "cases": cases, "edit_samples": list(span),
            "offset_samples": movement, "mouth_box": None if box is None else list(map(int, box))}


def review(lock, rows):
    require(len(lock["sources"]) == len(rows), "Missing source results")
    summary = {}
    for item, row in zip(lock["sources"], rows, strict=True):
        require((item["group"], item["sha256"]) == (row["group"], row["sha256"]), "Result does not match source")
        require(len(row["cases"]) == len(CASES) * 2 and
                {(case["role"], case["carrier"]) for case in row["cases"]} == {
                    (role, carrier) for role in CASES for carrier in ("PCM", "AAC")}, "A matched case is missing")
        for case in row["cases"]:
            for method in ("coarse", "fine", "both"):
                stats = summary.setdefault(f'{item["split"]}/{case["carrier"]}/{method}', Counter())
                role = case["role"]
                edit = role.startswith("shift_")
                stats["planned_edits" if edit else "planned_controls"] += 1
                if "unavailable" in case:
                    stats["unavailable_controls"] += 1
                    continue
                grids = GRIDS if method == "both" else (method,)
                qualified = all(case["picture"][name][metric] <= point for name in grids for metric, point in POINTS.items())
                stats["qualified_edits" if edit else "qualified_controls"] += qualified
                match = case["sound"] if qualified else None
                if match:
                    if edit:
                        left, right = row["edit_samples"]
                        a, b = match["submitted_samples"]
                        correct = (5 * (min(right, b) - max(left, a)) >= 3 * (max(right, b) - min(left, a))
                                   and abs(match["shift_samples"] - row["offset_samples"] * (
                                       1 if role == "shift_later" else -1)) <= 16)
                        stats["useful" if correct else "misplaced"] += 1
                    else:
                        stats["control_flags"] += 1
                        stats[role] += 1
    return {key: dict(value) for key, value in summary.items()}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "score", "review"))
    cli.add_argument("sources", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    if args.phase == "review":
        result = read(args.output / "receipt.json")
        require(review(result["freeze"], result["rows"]) == result["outcome"], "Recorded counts differ")
        print(json.dumps(result["outcome"], sort_keys=True), flush=True)
        return
    current = lock(args.sources)
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", current)
        return
    require(current == read(args.output / "freeze.json") and sha(earlier.FACE_FILE) == earlier.FACE_SHA,
            "Source or implementation changed after freeze")
    cv2.setNumThreads(1)
    detector = load_yunet(earlier.FACE_FILE)
    rows = []
    for index, item in enumerate(current["sources"]):
        with TemporaryDirectory(prefix="soundtrack-hd-") as work:
            scratch = Path(work)
            donor_row = current["sources"][(index + 1) % len(current["sources"])]
            donor_file = scratch / "donor.mov"
            excerpt(args.sources / (donor_row["name"] + ".webm"), donor_file, donor_row["offset"])
            donor, _ = _track(donor_file, sha(donor_file))
            result = score_group(item, args.sources, scratch, detector, donor[:SAMPLES])
            print(item["group"], "mouth" if result["mouth_box"] is not None else "no observed mouth", flush=True)
            rows.append(result)
    result = {"freeze": current, "rows": rows, "outcome": review(current, rows)}
    write(args.output / "receipt.json", result)
    print(json.dumps(result["outcome"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
