# SPDX-License-Identifier: Apache-2.0
"""Evaluate short signed soundtrack shifts with verified corresponding pictures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
RATE = 16000
FPS = 25
PCM = "PCM"
AAC = "AAC"
SOURCE_PLAN = (
    ("c/cd", "WIKITONGUES-_Gereon_speaking_English.webm", "Bogreudell",
     "32b20a20e84d46d17da126689572f9a431d01ddc23880b90240952875b2a3fe8"),
    ("5/57", "WIKITONGUES-_Imtimeren_speaking_Mongsen_and_English.webm", "Wikitongues",
     "0ccec42e6861332b6ad189ec6009f9ed6101f7422586a96b203e809e0620b0da"),
    ("a/a8", "WIKITONGUES-_Raluca_speaking_Romanian.webm", "Wikitongues and Nick Panzarella",
     "c4434b0b812ac0acbf2e3f4cbb19481232df1aa7c2a746d3286218cf412f4b90"),
    ("4/48", "WIKITONGUES-_Sam_speaking_English.webm", "Bogreudell",
     "fad57b91c051dbf389d238cdc42b9cd72aeb73a16e93afc9983b0a428ba91cf2"),
    ("4/4c", "WIKITONGUES-_William_speaking_English.webm", "Bogreudell",
     "f0c58a97344e56f5470243b629e1a43e62746c1487991598ede2fbe129b15366"),
)
OFFSETS = (23, 38)
EDITS = {"earlier": -3200, "later": 3200}
CONTROLS = ("unchanged", "gain", "filter", "constant", "synchronized_picture",
            "synchronized_mouth", "alternate_speech", "quiet", "repeat_bed")
SHORT_WINDOW = 1280
SHORT_STEP = 320
QUIET_RMS = .00075
POINTS = {"whole_q65": 2.5, "whole_q90": 4.0, "local_five_frame_median": 6.27}
DESIGN = {
    "method": "After the two existing passes, compare 80 ms windows every 20 ms at the existing quiet floor. "
              "Retain 0.86 correlation, the alternative-peak margin, four supporting shifted windows and two "
              "same-time windows per flank. Keep both existing picture grids and bounds.",
    "acceptance": "In each carrier add at least four useful locations on three distinct recordings. "
                  "Preserve earlier findings. Allow at most one new ordinary control flag per carrier and "
                  "zero additions on the constant offset and synchronized picture and mouth controls.",
    "scope": "Four-second 720p25 MP4/MOV pairs with identical clock. Edits of 320 or 480 ms on the native picture clock. "
             "Useful requires correct shift within 16 samples and temporal IoU of at least 0.50.",
}


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sources():
    return [{"file": name, "page": f"https://commons.wikimedia.org/wiki/File:{name}", "attribution": credit,
             "url": f"https://upload.wikimedia.org/wikipedia/commons/{prefix}/{name}", "sha256": digest,
             "license": "CC-BY-SA-4.0"} for prefix, name, credit, digest in SOURCE_PLAN]


def boundary(group, offset):
    # Exact frame edges at 25 fps with planned offsets around the middle of the clip.
    start_frame = (31 if offset == OFFSETS[0] else 55) + group % 4
    length = 8 if (group + OFFSETS.index(offset)) % 2 else 12
    return [start_frame * 640, (start_frame + length) * 640]


def qualified(picture):
    return (set(picture) == {"coarse_grid", "fine_grid"}
            and all(set(reading) == POINTS.keys() and all(0 <= value <= POINTS[key] for key, value in reading.items())
                    for reading in picture.values()))


def location(row, reading, truth):
    if reading is None:
        return False
    a, b = reading["submitted_samples"]
    c, d = reading["earlier_samples"]
    shift = reading["shift_samples"]
    if not (a - c == b - d == shift and 1920 <= abs(shift) <= 5760
            and 0 <= min(a, c) < max(b, d) <= 63360 and min(b - a, d - c) >= 1280
            and 4 <= reading["supporting_windows"] == len(reading["window_start_samples"])
            and .86 <= reading["minimum_correlation"] <= 1
            and set(reading["same_time_support"]) == {"before", "after"}
            and all(len(values) == 2 for values in reading["same_time_support"].values())):
        raise ValueError("Invalid signed range or source support")
    if row["role"] not in EDITS:
        return False
    low, high = truth
    return abs(shift - EDITS[row["role"]]) <= 16 and 2 * (min(b, high) - max(a, low)) >= max(b, high) - min(a, low)


def review(rows):
    expected = [(i, offset) for i in range(1, len(SOURCE_PLAN) + 1) for offset in OFFSETS]
    if [(r["group"], r["offset_s"]) for r in rows] != expected:
        raise ValueError("Incomplete source roster")
    totals = {carrier: {"edits": 0, "controls": 0, "unavailable": 0, "old_useful": 0, "new_useful": 0,
                        "added_wrong_edits": 0, "old_control_flags": 0, "new_control_flags": 0,
                        "new_timing_controls": 0, "blocked_picture_controls": 0} for carrier in (PCM, AAC)}
    groups = {carrier: set() for carrier in (PCM, AAC)}
    for row in rows:
        truth = boundary(row["group"], row["offset_s"])
        if row["samples"] != truth or row["sha256"] != SOURCE_PLAN[row["group"]-1][3]:
            raise ValueError("Edit truth or source does not match the frozen plan")
        if [(c["carrier"], c["role"]) for c in row["cases"]] != [
            (c, role) for c in (PCM, AAC) for role in (*EDITS, *CONTROLS)
        ]:
            raise ValueError("Incomplete role roster")
        for case in row["cases"]:
            carrier, role = case["carrier"], case["role"]
            counter = totals[carrier]
            edited = role in EDITS
            counter["edits" if edited else "controls"] += 1
            if case.get("unavailable"):
                if role != "synchronized_mouth" or case["unavailable"] != "mouth_geometry":
                    raise ValueError("Unexpected missing delivery")
                counter["unavailable"] += 1
                continue
            valid = qualified(case["picture"])
            if edited and case["earlier_pcm_sha256"] == case["submitted_pcm_sha256"]:
                raise ValueError("Edit reuses original audio")
            good_old = location(case, case["old"], truth)
            good_new = location(case, case["short"], truth)
            old, new = valid and case["old"] is not None, valid and case["short"] is not None
            added = new and not old
            if edited:
                counter["old_useful"] += valid and good_old
                counter["new_useful"] += valid and (good_old or (not old and good_new))
                counter["added_wrong_edits"] += added and not good_new
                if added and good_new:
                    groups[carrier].add(row["group"])
            else:
                counter["old_control_flags"] += old
                counter["new_control_flags"] += added
                counter["new_timing_controls"] += added and role in ("constant", "synchronized_picture", "synchronized_mouth")
                counter["blocked_picture_controls"] += not valid and role.startswith("synchronized")
    for carrier, counter in totals.items():
        counter["added_recordings"] = len(groups[carrier])
        counter["passes"] = (counter["new_useful"] - counter["old_useful"] >= 4
                             and counter["added_recordings"] >= 3
                             and counter["new_control_flags"] <= 1 and counter["new_timing_controls"] == 0)
    return totals


def run(bank, output):
    import cv2
    import numpy as np
    from scipy.signal import butter, sosfiltfilt

    from app.detectors.image.faces import load_yunet
    from app.evidence import video_reference_soundtrack as waveform
    from app.evidence import video_reference_soundtrack_reencode as picture
    from scripts import video_soundtrack_cadence as cadence

    for row in sources():
        if sha(bank / row["file"]) != row["sha256"]:
            raise ValueError("Fresh source differs from the published file")
    face_asset = cadence.parent.earlier.FACE_FILE
    if sha(face_asset) != cadence.parent.earlier.FACE_SHA or picture.POINTS != POINTS:
        raise ValueError("Pinned visual assets or bounds changed")
    cv2.setNumThreads(1)
    detector = load_yunet(face_asset)
    rows = []
    for group, source in enumerate(sources(), 1):
        for offset in OFFSETS:
            with TemporaryDirectory(prefix="short-soundtrack-") as temp:
                folder = Path(temp)
                reference = folder / "reference.mov"
                cadence.parent.execute(
                    "ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-ss", str(offset),
                    "-i", str(bank / source["file"]), "-map", "0:v:0", "-map", "0:a:0", "-t", "4", "-vf",
                    "fps=25,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "1",
                    "-video_track_timescale", "48000", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(reference),
                )
                pixels = cadence._pictures(reference, FPS)
                base = cadence.parent._views(pixels)
                audio = cadence._sound(reference)[0][:4*RATE]
                start, end = boundary(group, offset)
                a, b = start // 640, end // 640
                box = cadence.parent._mouth(pixels, detector, a, b, 5)
                picture_paths, reading = {}, {}
                for mode in ("unchanged", "synchronized_picture", "synchronized_mouth"):
                    if mode == "synchronized_mouth" and box is None:
                        continue
                    path = folder / f"{mode}.mov"
                    if mode == "unchanged":
                        cadence._encode(path, pixels, FPS, 31 + group % 2)
                    else:
                        x, y, right, bottom = box if mode.endswith("mouth") else (0, 0, 1280, 720)
                        segment = np.s_[a:b, y:bottom, x:right]
                        saved = pixels[segment].copy()
                        pixels[segment] = pixels[a-5:b-5, y:bottom, x:right].copy()
                        cadence._encode(path, pixels, FPS, 31 + group % 2)
                        pixels[segment] = saved
                    picture_paths[mode] = path
                    views = cadence._view(path, FPS)
                    grids = dict(zip(("coarse_grid", "fine_grid"), sorted(base, key=lambda key: base[key].shape[2]), strict=True))
                    reading[mode] = {
                        label: picture.summarize(np.abs(base[key] - views[key]))["measurements"]
                        for label, key in grids.items()
                    }
                del pixels
                weights = np.ones(end - start, dtype=np.float32)
                weights[:80] = np.arange(80) / 80
                weights[-80:] = weights[:80][::-1]

                def insert(sound, *, audio=audio, start=start, end=end, weights=weights):
                    result = audio.copy()
                    result[start:end] += weights * (sound[start:end] - audio[start:end])
                    return result

                def move(samples, *, audio=audio):
                    return cadence.parent.shift(audio, samples)

                # Distinct clean speech from later in the same recording, matched level.
                alternate = np.roll(audio, -13733)
                repeated = np.resize(audio[1000:3040], len(audio))
                waves = {"earlier": insert(move(-3200)), "later": insert(move(3200)), "unchanged": audio,
                         "gain": insert(audio * .45),
                         "filter": insert(sosfiltfilt(butter(4, 2000, fs=RATE, output="sos"), audio)),
                         "constant": move(3200), "synchronized_picture": insert(move(3200)),
                         "synchronized_mouth": insert(move(3200)), "alternate_speech": insert(alternate),
                         "quiet": insert(audio * .04), "repeat_bed": insert(repeated)}
                cases = []
                for carrier in (PCM, AAC):
                    suffix = "mp4" if carrier == AAC else "mov"
                    earlier = folder / f"reference-delivery.{suffix}"
                    cadence._remux_wave(reference, audio, earlier, carrier)
                    first, first_sha = cadence._sound(earlier)
                    for role, sound in waves.items():
                        mode = role if role.startswith("synchronized") else "unchanged"
                        case = {"carrier": carrier, "role": role}
                        cases.append(case)
                        if mode not in picture_paths:
                            case["unavailable"] = "mouth_geometry"
                            continue
                        path = folder / f"{role}-delivery.{suffix}"
                        cadence._remux_wave(picture_paths[mode], sound, path, carrier)
                        cadence._clock(path, FPS)
                        second, second_sha = cadence._sound(path)
                        if len(first) != len(second):
                            raise ValueError("Paired audio sample coverage differs")
                        standard, _ = waveform.measure(first, second)
                        quiet, _ = waveform.measure(first, second, min_rms=QUIET_RMS)
                        short, _ = waveform.measure(first, second, window=SHORT_WINDOW, step=SHORT_STEP, min_rms=QUIET_RMS)

                        def in_clock(value):
                            return value if value and max(value["submitted_samples"][1], value["earlier_samples"][1]) <= 63360 else None

                        case.update(picture=reading[mode], delivery_sha256=sha(path), earlier_pcm_sha256=first_sha,
                                    submitted_pcm_sha256=second_sha, old=in_clock(standard or quiet), short=in_clock(short))
                        if role == "earlier" and offset == OFFSETS[0] and carrier == AAC:
                            import shutil

                            shutil.copyfile(earlier, output.parent / f"g{group}-earlier.mp4")
                            shutil.copyfile(path, output.parent / f"g{group}-submitted.mp4")
                rows.append({"group": group, "offset_s": offset, "samples": [start, end], "sha256": source["sha256"], "cases": cases})
                print(f"Verified source {group}, excerpt {offset}", flush=True)
    receipt = {"schema": "chipmunk.video-short-soundtrack-transfer/1", "design": DESIGN, "sources": sources(),
               "implementation": sha(Path(__file__)), "rows": rows, "results": review(rows)}
    with output.open("x") as handle:
        print(json.dumps(receipt, indent=2, sort_keys=True), file=handle)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--bank", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        result = run(args.bank, args.receipt)
    else:
        result = json.loads(args.receipt.read_bytes())
        if (result["design"] != DESIGN or result["sources"] != sources()
                or result["results"] != review(result["rows"])):
            raise ValueError("Measured results do not match the public ledger")
    print(json.dumps(result["results"], indent=2, sort_keys=True))
