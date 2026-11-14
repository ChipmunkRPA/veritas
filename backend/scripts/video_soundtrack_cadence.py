# SPDX-License-Identifier: Apache-2.0
"""Transfer the earlier-copy soundtrack check to exact 24 and 25 fps exports."""
from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import wav_bytes
from app.detectors.image.faces import load_yunet
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference_soundtrack as soundtrack
from app.evidence import video_reference_soundtrack_reencode as reencode
from scripts import review_video_soundtrack_cadence as rule
from scripts import video_soundtrack_hd_transfer as parent
from scripts.video_common_sources_v89 import read, require, sha, write

WIDTH, HEIGHT, RATE, SAMPLES = 1280, 720, 16000, 64000


def locate(directories, name, digest):
    for directory in directories:
        path = directory / (name + ".webm")
        if path.is_file() and sha(path) == digest:
            return path
    raise ValueError(f"Missing or mismatched publisher source: {name}")


def freeze(directories):
    conversation = read(parent.earlier.CONVERSATION)["source_plan"]["rows"]
    public = [row for row in read(parent.speech.SOURCES)["sources"] if row["id"] != "diego"]
    sources = [{"name": s["name"], "sha256": s["source_sha256"], "offset": s["offset_s"] + 26} for s in conversation]
    sources += [{"name": s["id"], "sha256": s["sha256"], "offset": s["offsets"][0] + 15} for s in public]
    for index, row in enumerate(sources, 1):
        row["group"] = index
        locate(directories, row["name"], row["sha256"])
    paths = (__file__, rule.__file__, parent.__file__, parent.earlier.CONVERSATION, parent.speech.SOURCES,
             soundtrack.__file__, parent.earlier.FACE_FILE)
    return {"schema": rule.SCHEMA, "design": rule.DESIGN, "points": rule.POINTS, "rates": list(rule.RATES),
            "sources": sources, "bindings": {Path(path).name: sha(path) for path in paths},
            "runtime": {"ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
                        "opencv": cv2.__version__, "numpy": np.__version__}}


def _clock(path, fps):
    table = probe_frames(path, timeout_s=10)
    require(len(table) == 4 * fps and (table.width, table.height) == (WIDTH, HEIGHT)
            and not (table.rotation or table.truncated or table.interpolated)
            and all((stamp - table.pts[0]) * table.tb == Fraction(i, fps) for i, stamp in enumerate(table.pts)),
            "A delivery failed the complete source-picture clock")
    return table


def _pictures(path, fps):
    clock = _clock(path, fps)
    result = np.empty((len(clock), HEIGHT, WIDTH, 3), dtype=np.uint8)
    count = 0
    with closing(decode_frames(path, clock, None, timeout_s=20)) as frames:
        for count, frame in enumerate(frames, 1):
            require(frame.index == count - 1 and frame.pts == clock.pts[frame.index] and frame.pts_verified,
                    "A delivered picture has no verified timestamp")
            result[frame.index] = frame.rgb
    require(count == len(clock), "A delivered picture is missing")
    return result


def _encode(path, pictures, fps, crf):
    parent.execute("ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(fps), "-i", "pipe:0",
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(crf), "-pix_fmt", "yuv420p",
                   "-threads", "1", "-video_track_timescale", "48000", str(path), data=pictures.tobytes())


def _view(path, fps):
    return parent._views(_pictures(path, fps))


def _remux_wave(picture, wave, output, carrier):
    raw = output.with_suffix(".wav")
    raw.write_bytes(wav_bytes(wave))
    codec = ("-c:a", "aac", "-b:a", "64k") if carrier == "AAC" else ("-c:a", "pcm_s16le")
    parent.remux(picture, raw, output, codec)


def _sound(path):
    decoded, digest = parent._track(path, sha(path))
    require(len(decoded) >= SAMPLES, "Missing decoded samples in the four-second delivery")
    return decoded, digest


def score_group(source, donor, source_paths, scratch, detector, fps):
    reference = scratch / "reference.mov"
    parent.execute("ffmpeg", "-nostdin", "-v", "error", "-ss", str(source["offset"]), "-i", str(source_paths[source["name"]]),
                   "-map", "0:v:0", "-map", "0:a:0", "-t", "4", "-vf",
                   f"fps={fps},scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
                   "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1",
                   "-video_track_timescale", "48000", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(reference))
    pixels = _pictures(reference, fps)
    base = parent._views(pixels)
    original = _sound(reference)[0][:SAMPLES]
    start, stop = rule.edit_interval(source["group"])
    low, high = [(sample * fps + RATE - 1) // RATE for sample in (start, stop)]
    shift = round(.2 * fps)
    box = parent._mouth(pixels, detector, low, high, shift)
    paths, picture = {}, {}
    for kind in ("unshifted", "picture", "mouth"):
        if kind == "mouth" and box is None:
            continue
        path = scratch / f"{kind}.mov"
        # A matched change at 25 fps uses five pictures exactly. At 24 fps,
        # test the closest native tick, five pictures or 208.3 ms.
        if kind != "unshifted":
            x0, y0, x1, y1 = box if kind == "mouth" else (0, 0, WIDTH, HEIGHT)
            selected = np.s_[low:high, y0:y1, x0:x1]
            saved = pixels[selected].copy()
            pixels[selected] = pixels[low - shift:high - shift, y0:y1, x0:x1].copy()
        _encode(path, pixels, fps, 31 if source["group"] % 2 else 34)
        if kind != "unshifted":
            pixels[selected] = saved
        paths[kind] = path
    del pixels
    for kind, path in paths.items():
        visual = _view(path, fps)
        picture[kind] = {name: reencode.summarize(np.abs(base[name] - visual[name]))["measurements"] for name in base}

    def mixed(sound):
        target = original.copy()
        weight = np.ones(stop - start, dtype=np.float32)
        weight[:80] = np.arange(80) / 80
        weight[-80:] = weight[:80][::-1]
        target[start:stop] += weight * (sound[start:stop] - original[start:stop])
        return target

    prior = parent.shift(original, -3200)
    delayed = parent.shift(original, 3200)
    require(np.array_equal(prior[start:stop], original[start + 3200:stop + 3200])
            and np.array_equal(delayed[start:stop], original[start - 3200:stop - 3200]), "The authored signed source ranges differ")
    later = mixed(delayed)
    synchronized = mixed(parent.shift(original, round(shift * RATE / fps)))
    filtered = sosfiltfilt(butter(4, 2500, fs=RATE, output="sos"), original)
    controls = {"earlier": ("unshifted", mixed(prior)),
                "later": ("unshifted", later), "untouched": ("unshifted", original),
                "gain": ("unshifted", mixed(original * .55)), "filter": ("unshifted", mixed(filtered)),
                "constant": ("unshifted", delayed), "picture_only": ("picture", original),
                "synchronized_picture": ("picture", synchronized), "synchronized_mouth": ("mouth", synchronized),
                "alternate_speech": ("unshifted", mixed(donor))}
    cases = []
    for carrier in rule.CARRIERS:
        ext = "mp4" if carrier == "AAC" else "mov"
        pinned = scratch / f"reference-{carrier}.{ext}"
        _remux_wave(reference, original, pinned, carrier)
        first, first_hash = _sound(pinned)
        for role, (style, wave) in controls.items():
            row = {"role": role, "carrier": carrier}
            cases.append(row)
            if style not in paths:
                row["unavailable"] = "mouth_geometry"
                continue
            delivered = scratch / f"{role}-{carrier}.{ext}"
            _remux_wave(paths[style], wave, delivered, carrier)
            _clock(delivered, fps)
            second, second_hash = _sound(delivered)
            require(len(first) == len(second), "Carrier changed the paired audio coverage")
            require(role not in rule.EDITS or first_hash != second_hash, "An edited delivery reuses the untouched soundtrack")
            best, count = soundtrack.measure(first, second)
            if best and max(best["submitted_samples"][1], best["earlier_samples"][1]) * fps > (4 * fps - 1) * RATE:
                best = None
            row.update(picture=picture[style], sound=best, matched_windows=count, delivery_sha256=sha(delivered),
                       earlier_pcm_sha256=first_hash, submitted_pcm_sha256=second_hash)
    return {"group": source["group"], "rate": fps, "pictures": 4 * fps, "source_sha256": source["sha256"],
            "edit_samples": [start, stop], "mouth_box": [int(i) for i in box] if box else None, "cases": cases}


def score(lock, directories):
    require(lock == freeze(directories), "Source plan or analysis changed after freeze")
    require(sha(parent.earlier.FACE_FILE) == parent.earlier.FACE_SHA, "Incorrect face geometry weights")
    require(reencode.POINTS == rule.POINTS, "Current picture operating points differ")
    cv2.setNumThreads(1)
    detector = load_yunet(parent.earlier.FACE_FILE)
    paths = {row["name"]: locate(directories, row["name"], row["sha256"]) for row in lock["sources"]}
    rows = []
    for index, source in enumerate(lock["sources"]):
        next_source = lock["sources"][(index + 1) % len(lock["sources"])]
        audio = parent.execute("ffmpeg", "-nostdin", "-v", "error", "-ss", str(next_source["offset"]),
                               "-i", str(paths[next_source["name"]]), "-map", "0:a:0", "-t", "4", "-ar", "16000", "-ac", "1",
                               "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1")
        donor = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.
        require(SAMPLES <= len(donor) <= SAMPLES + 1024, "The frozen competing speech excerpt is incomplete")
        donor = donor[:SAMPLES]
        for fps in rule.RATES:
            with TemporaryDirectory(prefix="soundtrack-cadence-") as tmp:
                rows.append(score_group(source, donor, paths, Path(tmp), detector, fps))
            print(f"verified {source['name']} at {fps} fps", flush=True)
    return {"freeze": lock, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score"))
    parser.add_argument("run", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    admitted = args.run / "freeze.json"
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        write(admitted, freeze(args.sources))
        return
    result = score(read(admitted), args.sources)
    result["outcome"] = rule.review(result)
    write(args.run / "receipt.json", result)
    print(json.dumps(result["outcome"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
