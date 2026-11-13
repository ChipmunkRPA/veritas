# SPDX-License-Identifier: Apache-2.0
"""Test paired soundtrack timing when otherwise matching pictures were re-encoded."""
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt

from app.detectors.audio.decode import wav_bytes
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import audio_reference, video_reference, video_reference_soundtrack
from scripts import review_video_soundtrack_reencode as review
from scripts import video_lipsync_repeat_audio as source
from scripts import video_lipsync_repeat_public_speech as speech
from scripts.video_common_sources_v89 import read, require, sha, write

ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
ACTORS = PANELS / "video-lipsync-repeat-fresh-actors-2026-11-13.json"
ACTOR_PLAN = PANELS / "video-lipsync-repeat-fresh-actors-sources-2026-11-13.json"
SPEECH = PANELS / "video-lipsync-repeat-public-speech-2026-11-13.json"
DESIGN = {
    "pictures": "Use separately re-encoded 480x360 H.264 25 fps clips on a verified exact relative clock. Compare 80x60 luma, "
                "a fixed 5x5 spatial block grid and the strongest five-frame median. Whole-picture mean thresholds: "
                "65th percentile <=2.5 and 90th <=4.0. The local point is the largest ordinary development transcode "
                "response plus .35, with a floor of 1.75, separately for PCM and AAC.",
    "sound": "Reuse the existing 16 kHz, 160 ms waveform matches with 40 ms hop and 360 ms signed search. "
             "Require continuous matching decoded clocks and four shifted windows bracketed by same-time sound.",
    "truth": "Match a local 640 ms or 880 ms sound move by at least 0.60 temporal IoU and recover signed lag within 16 samples. "
             "Exact paired picture and sound witnesses, with originals before, during and after.",
    "sources": "Use the prior public expressive and conversational source plans, preserving their source qualification. "
               "Expressive groups 201–204 are development. Remaining expressive groups and all conversations are transfer. "
               "Every source contributes the same 15 cases per carrier; H.264 CRF alternates 31 and 34 by source group.",
    "scope": "Measure a moved excerpt relative to a supplied earlier copy. No assertion of original audiovisual sync, "
             "AI involvement, or generation method.",
}


def raw_hash(array):
    return sha256(array.tobytes()).hexdigest()


def clock_commitment(table):
    return review.digest([table.time_base, list(table.pts)])


def plan(actor_dir, speech_dir):
    earlier, public = read(ACTORS), read(SPEECH)
    require(sha(actor_dir / "intake-freeze.json") == sha(ACTOR_PLAN), "Expressive source plan differs")
    require(sha(actor_dir / "intake-receipt.json") == earlier["freeze"]["acquisition_sha256"],
            "Prior source verification differs")
    roster = {row["sha256"]: row for row in read(actor_dir / "intake-receipt.json")["rows"]}
    actor_sources = {}
    for row in earlier["qualification"]:
        selected = row.get("selected")
        if selected:
            item = roster[selected]
            require(item["group"] == row["group"] and sha(actor_dir / item["file"]) == selected, "Actor source differs")
            actor_sources[row["group"]] = (actor_dir / item["file"], selected, None)
    public_sources = {}
    for row, src in zip(public["qualification"], read(speech.SOURCES)["sources"], strict=True):
        path = speech_dir / f'{src["id"]}.webm'
        require(src["sha256"] == row["source_sha256"] == sha(path), "Publisher conversation differs")
        public_sources[row["group"]] = (path, src["sha256"], row["selected"])
    bindings = (ACTORS, ACTOR_PLAN, SPEECH, speech.SOURCES, Path(__file__), Path(review.__file__),
                Path(source.__file__), Path(speech.__file__), Path(video_reference_soundtrack.__file__), source.WEIGHT)
    source_plan = actor_sources | public_sources
    lock = {"schema": review.SCHEMA, "design": DESIGN, "gate": review.GATE,
            "qualified_sources": [source_plan[group][1] if group in source_plan else None
                                  for group in [*range(201, 217), *range(301, 305)]],
            "bindings": {path.name: sha(path) for path in bindings},
            "runtime": {"opencv": cv2.__version__, "numpy": np.__version__,
                        "ffmpeg": source.delivery.command("ffmpeg", "-version").decode().splitlines()[0]}}
    return lock, source_plan


def prepare(item, folder):
    path, digest, offset = item
    prepared = source.original(path, digest) if offset is None else speech.prepare(path, folder / "adapted.mov", offset)
    require(prepared is not None, "Previously qualified source has changed")
    return prepared


def retime(frames, a, b, offset, box=None):
    altered = frames.copy()
    region = (slice(None), slice(None)) if box is None else (slice(box[1], box[3]), slice(box[0], box[2]))
    altered[(slice(a, b), *region)] = frames[(slice(a - offset, b - offset), *region)]
    return altered


def mouth_box(frames, first, last, shift, detector):
    indices = sorted({first, (first + last) // 2, last - 1, first - shift, (first + last) // 2 - shift,
                      last - 1 - shift})
    corners = []
    for index in indices:
        faces = source.detect_faces(cv2.cvtColor(frames[index], cv2.COLOR_RGB2BGR), detector=detector, min_size=48)
        require(len(faces) == 1, "Local synchronized mouth control needs one observed face")
        lips = np.asarray(faces[0].landmarks[3:5])
        width = float(np.linalg.norm(lips[0] - lips[1]))
        x, y = lips.mean(axis=0)
        corners.append((x - 1.15 * width, y - .8 * width, x + 1.15 * width, y + .8 * width))
    x0, y0 = np.floor(np.min(corners, axis=0)[:2]).astype(int)
    x1, y1 = np.ceil(np.max(corners, axis=0)[2:]).astype(int)
    require(0 <= x0 < x1 <= 480 and 0 <= y0 < y1 <= 360, "Mouth control outside native picture")
    return int(x0), int(y0), int(x1), int(y1)


def recode(frames, path, crf):
    command = ("ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "480x360", "-framerate", "25", "-i", "pipe:0", "-c:v", "libx264",
               "-threads", "1", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-video_track_timescale", "16000", str(path))
    subprocess.run(command, input=np.ascontiguousarray(frames).tobytes(), capture_output=True, check=True, timeout=20)


def picture_data(path):
    table = probe_frames(path, timeout_s=8)
    frames = list(decode_frames(path, table, list(range(review.PICTURES)), timeout_s=8))
    require(len(table) == len(frames) == review.PICTURES and not (table.rotation or table.interpolated or table.truncated)
            and (table.width, table.height) == (480, 360)
            and all(frame.pts_verified and frame.index == index and frame.pts == table.pts[index]
                    and (frame.pts - table.pts[0]) * table.tb * 25 == index for index, frame in enumerate(frames)),
            "Native picture alignment differs")
    return table, np.stack([frame.rgb for frame in frames])


def measure_picture(first, last):
    grids = [np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (80, 60), interpolation=cv2.INTER_AREA)
                       for f in side]).astype(np.float32) for side in (first, last)]
    diff = np.abs(grids[0] - grids[1])
    frame_errors = diff.mean(axis=(1, 2))
    cells = diff.reshape(len(diff), 12, 5, 16, 5).mean(axis=(2, 4))
    consecutive = np.lib.stride_tricks.sliding_window_view(cells, 5, axis=0)
    return {"same_rgb": bool(np.array_equal(first, last)), "q65": round(float(np.quantile(frame_errors, .65)), 6),
            "q90": round(float(np.quantile(frame_errors, .90)), 6),
            "local": round(float(np.median(consecutive, axis=-1).max()), 6)}


def input_digest(path, table, pcm, pixels):
    return {"sha256": sha(path), "rgb_sha256": raw_hash(pixels), "pts_sha256": clock_commitment(table),
            "pcm_sha256": pcm[2]["pcm_sha256"], "pictures": len(table), "samples": pcm[2]["samples"],
            "relative_start": pcm[2]["relative_start"]}


def measure_sound(path, table):
    track = audio_reference.read_track(path, {"sha256": sha(path)}, max_seconds=4, video=True,
                                       origin=table.pts[0] * table.tb)
    require(track[0] == "measured" and track[2]["relative_start"] == "0", "Continuous soundtrack clock needed")
    return track


def witness(paths, tables, inputs, tracks, match):
    if not match:
        return None
    start, stop = match["submitted_samples"]
    first, last = match["earlier_samples"]
    hop = review.HOP
    if min(first, start) < 0 or max(last, stop) > (review.PICTURES - 1) * hop:
        return None
    anchors = [max(0, start // hop - 1), (start + stop) // (2 * hop), (stop + hop - 1) // hop]
    frames, sounds = defaultdict(list), []
    for i, (path, table, pixels, track) in enumerate(zip(paths, tables, inputs, tracks, strict=True)):
        pts = source.delivery.command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      "frame=best_effort_timestamp", "-of", "csv=p=0", str(path)).decode()
        observed_pts = tuple(int(line.split(",")[0]) for line in pts.splitlines() if line.strip())
        require(observed_pts == table.pts, "Independent FFprobe timestamps differ")
        audio = source.delivery.command("ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
                                        "-ac", "1", "-ar", str(review.SR), "-f", "s16le", "-")
        a, b = (first, last) if i == 0 else (start, stop)
        local = np.asarray(track[1][a:b] * 32768, dtype="<i2")
        require(audio[2 * a:2 * b] == local.tobytes(), "Independent PCM witness differs")
        sounds.append({"samples": [a, b], "sha256": raw_hash(local)})
        select = "+".join(f"eq(n\\,{index})" for index in anchors)
        raw = source.delivery.command("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path),
                                      "-vf", f"select={select}", "-fps_mode", "passthrough", "-an", "-f", "rawvideo",
                                      "-pix_fmt", "rgb24", "-threads", "1", "-")
        require(raw == pixels[anchors].tobytes(), "Independent native RGB witness differs")
        for index in anchors:
            frames[index].append({"time_base": table.time_base, "pts": table.pts[index] - table.pts[0],
                                  "sha256": raw_hash(pixels[index])})
    return {"pictures": [{"frame": index, "paired": frames[index]} for index in anchors], "sound": sounds}


def data_for_roles(group, frames, pcm, other_pcm, detector):
    a = 21 + group % 3
    b = a + 16
    for role in review.POSITIVE:
        expected = review.truth(group, role)
        yield role, "same", None, source.delivery._mix(pcm, source.clock.shift(pcm, expected[1]), expected[0])
    selected = (a * review.HOP, b * review.HOP)
    yield "reencode", "same", None, pcm
    yield "gain", "same", None, source.delivery._mix(pcm, pcm * .65, selected)
    low = sosfiltfilt(butter(4, 2500, fs=review.SR, output="sos"), pcm)
    yield "lowpass", "same", None, source.delivery._mix(pcm, low, selected)
    yield "global", "same", None, source.clock.shift(pcm, 5 * review.HOP)
    retimed = retime(frames, a, b, 5)
    yield "picture_only", "picture_late", retimed, pcm
    for role, offset in (("late", 5), ("early", -5)):
        whole = retimed if offset > 0 else retime(frames, a, b, offset)
        moved = source.delivery._mix(pcm, source.clock.shift(pcm, offset * review.HOP), selected)
        yield f"matched_picture_{role}", f"picture_{role}", whole, moved
    for role, offset in (("late", 5), ("early", -5)):
        box = mouth_box(frames, a, b, offset, detector)
        moved = source.delivery._mix(pcm, source.clock.shift(pcm, offset * review.HOP), selected)
        yield f"matched_mouth_{role}", f"mouth_{role}", retime(frames, a, b, offset, box), moved
    yield "independent_sound", "same", None, source.delivery._mix(pcm, other_pcm, selected)
    yield "wrong_reference", "wrong", None, source.delivery._mix(pcm, source.clock.shift(pcm, 5 * review.HOP), selected)


def score_group(group, bank, source_plan, donor, detector, run):
    source_entry = source_plan.get(group)
    row = {"group": group, "bank": bank, "source_sha256": source_entry[1] if source_entry else None, "cases": []}
    if source_entry is None:
        return row
    with TemporaryDirectory(dir=run) as scratch:
        scratch = Path(scratch)
        earlier_picture = scratch / "original.mov"
        frames, pcm, *_ = prepare(source_entry, scratch)
        source.picture(frames, earlier_picture)
        del frames
        _, frames = picture_data(earlier_picture)
        donor_folder = scratch / "donor"
        donor_folder.mkdir()
        other, other_pcm, *_ = prepare(source_plan[donor], donor_folder)
        crf = 31 if group % 2 else 34
        recoded = {}
        for name, image in (("same", frames), ("wrong", other)):
            path = scratch / f"{name}.mov"
            recode(image, path, crf)
            table, pixels = picture_data(path)
            recoded[name] = (path, table, pixels, measure_picture(frames, pixels))
        del other
        wave_path = scratch / "input.wav"
        wave_path.write_bytes(wav_bytes(pcm.astype(np.float32) / 32768))
        pairs = defaultdict(dict)
        for carrier, (ext, options) in source.CODECS.items():
            path = scratch / f"earlier.{ext}"
            source.delivery.remux(earlier_picture, wave_path, path, options)
            table = probe_frames(path, timeout_s=6)
            track = measure_sound(path, table)
            pairs[carrier] = {"path": path, "table": table, "track": track, "digest": input_digest(path, table, track, frames)}
        by_carrier = defaultdict(list)
        for role, name, image, wave in data_for_roles(group, frames, pcm, other_pcm, detector):
            if name not in recoded:
                path = scratch / f"{name}.mov"
                recode(image, path, crf)
                table, pixels = picture_data(path)
                recoded[name] = (path, table, pixels, measure_picture(frames, pixels))
            path, _, pixels, picture = recoded[name]
            wave_path.write_bytes(wav_bytes(wave.astype(np.float32) / 32768))
            for carrier, (ext, options) in source.CODECS.items():
                delivered = scratch / f"submitted.{ext}"
                source.delivery.remux(path, wave_path, delivered, options)
                earlier = pairs[carrier]
                table = probe_frames(delivered, timeout_s=6)
                track = measure_sound(delivered, table)
                require(video_reference._comparison_issue((earlier["table"], table)) is None
                        and len(track[1]) == len(earlier["track"][1]), "Remux clock differs")
                match, support = video_reference_soundtrack.measure(earlier["track"][1], track[1])
                evidence = witness((earlier["path"], delivered), (earlier["table"], table),
                                   (frames, pixels), (earlier["track"], track), match)
                by_carrier[carrier].append({"carrier": carrier, "role": role, "truth": review.truth(group, role),
                                            "sources": [earlier["digest"], input_digest(delivered, table, track, pixels)],
                                            "picture": picture, "sound": {"match": match, "windows": support},
                                            "evidence": evidence})
                delivered.unlink()
        row["cases"] = [case for carrier in review.CARRIERS for case in by_carrier[carrier]]
    print(f"Measured {bank} group {group}", flush=True)
    return row


def score(run, source_plan, *, held):
    groups = list(range(205, 217)) + list(range(301, 305)) if held else list(range(201, 205))
    detector = source.load_yunet(source.WEIGHT)
    rows = []
    for group in groups:
        bank = ("conversational" if group >= 301 else "expressive") if held else "development"
        candidates = [item for item in groups if item != group and item in source_plan
                      and (item >= 301) == (group >= 301)]
        rows.append(score_group(group, bank, source_plan, candidates[0], detector, run))
    return rows


def with_actions(groups, points):
    for group in groups:
        for row in group["cases"]:
            row["actions"] = review.actions(row, points[row["carrier"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held"))
    parser.add_argument("run", type=Path)
    parser.add_argument("actors", type=Path)
    parser.add_argument("speech", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    lock, source_plan = plan(args.actors, args.speech)
    run = args.run
    if args.phase == "freeze":
        run.mkdir(mode=0o700)
        write(run / "freeze.json", lock)
        return
    require(read(run / "freeze.json") == lock, "Implementation or inputs differ from freeze")
    if args.phase == "development":
        development = score(run, source_plan, held=False)
        points = {carrier: review.point(development, carrier) for carrier in review.CARRIERS}
        with_actions(development, points)
        write(run / "development.json", development)
        write(run / "points.json", {"development_sha256": review.digest(development), "points": points})
        return
    points, development = read(run / "points.json"), read(run / "development.json")
    require(points["development_sha256"] == review.digest(development)
            and points["points"] == {c: review.point(development, c) for c in review.CARRIERS},
            "Development decisions differ")
    held = score(run, source_plan, held=True)
    with_actions(held, points["points"])
    receipt = {"freeze": lock, "freeze_sha256": review.digest(lock), "points": points,
               "point_sha256": review.digest(points), "groups": development + held}
    receipt["summary"] = review.summarize(receipt)
    write(run / "receipt.json", receipt)
    print(receipt["summary"], flush=True)


if __name__ == "__main__":
    main()
