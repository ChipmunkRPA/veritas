"""Freeze and test a longer native repeat search without changing the matching rule."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import subprocess
import time
from pathlib import Path

from app.detectors.video import picture_motion
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import picture_sequence
from scripts.video_lossy_sequence_transfer import hash_file, load, write
from scripts.video_motion_sequence_360p_transfer import CODECS, flip_yuv

WIDTH, HEIGHT, FPS, OPENING, EXTENDED = 640, 360, 25, 225, 450
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2
PUBLISHERS = {
    "cat": ("Cat_Play.webm", "NairobiPapel (Kamaa)", "CC BY-SA 4.0", "f/f2", 5),
    "rain": ("Water_flowing_in_a_channel_during_rainfall.webm", "Obed Makolo", "CC BY-SA 4.0", "0/05", 0),
    "train": ("Train_arriving_in_UlaanBaatar_.webm", "Ludovic Hirlimann", "CC BY-SA 4.0", "6/6d", 9),
    "grain": ("Video_2025-11-30_at_3.28.25_AM.webm", "Nolanus", "CC0 1.0", "d/d8", 3),
    "sprint": ("Was_passiert_bei_einem_Sprint.webm", "SRF/Einstein/Anatol Hug, Peter Höllrigl", "CC BY-SA 4.0", "6/62", 20),
    "training": ("New_York_Personal_Trainer_Joshua_Bailey_Circuit_Training.webm", "Dallas J Plummer",
                 "CC BY-SA 4.0", "a/ae", 20),
}
ASSEMBLIES = ((1, 5), (3, 6), (8, 11))
ROLES = ("late_cross", "late_both", "opening", "opening_after_gap", "ordinary", "cut", "hold", "reverse")
GATE = {"publisher_late_useful": 8, "publisher_useful_per_placement": 3, "camera_late_useful": 3,
        "max_added_ordinary": 1, "max_added_hold": 0, "preserve_opening_and_clock_gap": True,
        "minimum_iou_each_range": .5}


def checked_command(*arguments):
    return subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", *arguments], timeout=45)


def source(path, start, frames):
    raw = checked_command("-protocol_whitelist", "file", "-threads", "1", "-ss", str(start), "-i", f"file:{path}",
                          "-map", "0:v:0", "-an", "-filter_threads", "1", "-vf",
                          "fps=25,scale=640:360:force_original_aspect_ratio=increase:flags=area,crop=640:360,setsar=1",
                          "-frames:v", str(frames), "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1")
    if len(raw) != FRAME_BYTES * frames:
        raise ValueError("Frozen source interval is shorter than planned")
    return raw


def method_digest():
    sources = [inspect.getsource(fn) for fn in (picture_motion._pair, picture_motion._segments, picture_motion.descriptor)]
    return hashlib.sha256("".join(sources).encode()).hexdigest()


def freeze(run, publishers, archive):
    run.mkdir(mode=0o700)
    planned = []
    for key, (name, author, terms, directory, start) in PUBLISHERS.items():
        path = publishers / f"{key}.webm"
        planned.append({"name": key, "cohort": "publisher", "files": [{"file": str(path), "start": start, "frames": EXTENDED,
                        "sha256": hash_file(path), "page": "https://commons.wikimedia.org/wiki/File:" + name,
                        "download": f"https://upload.wikimedia.org/wikipedia/commons/{directory}/{name}",
                        "author": author, "terms": terms}]})
    for first, second in ASSEMBLIES:
        files = []
        for slot in (first, second):
            root = archive / f"slot-{slot:02}-attempt-01"
            path = root / "source.mkv"
            digest = hash_file(path)
            if digest != load(root / "qualification.json")["sha256"]:
                raise ValueError("Retained camera source differs from prior qualification")
            files.append({"file": str(path), "sha256": digest, "slot": slot, "start": 0, "frames": OPENING})
        planned.append({"name": f"camera-{first:02}-{second:02}", "cohort": "camera", "files": files})
    pins = {record["sha256"] for case in planned for record in case["files"]}
    if len(pins) != 12:
        raise ValueError("Source identity repeats within the frozen panel")
    write(run / "freeze.json", {"sources": planned, "geometry": [WIDTH, HEIGHT, FPS, OPENING, EXTENDED],
                               "roles": ROLES, "gate": GATE, "parameters": picture_sequence.MOTION_PARAMETERS,
                               "runner_sha256": hash_file(Path(__file__)), "method_sha256": method_digest()})


def placement(number, role):
    length = (23, 27, 31)[number % 3]
    if role.startswith("opening"):
        return 24 + number % 5, 151 + number, length
    if role == "late_both":
        return 240 + number, 382 + number, length
    return 42 + 4 * number, 288 + 3 * number, length


def encode(raw, path, number, role, codec):
    first, second, count = placement(number, role)
    view = memoryview(raw)
    frames = [view[i * FRAME_BYTES:(i + 1) * FRAME_BYTES] for i in range(EXTENDED)]
    if role.startswith(("late_", "opening")):
        frames[second:second + count] = frames[first:first + count]
    elif role == "reverse":
        frames[second:second + count] = frames[first:first + count][::-1]
    elif role == "hold":
        frames[second:second + count] = [frames[second]] * count
    elif role == "cut":
        frames[second:second + count] = [flip_yuv(frame) for frame in frames[second:second + count]]
    clock = "settb=expr=1/1000,setpts=N*40"
    if role == "opening_after_gap":
        clock += r"+8*gte(N\,310)"
    encoded = path.with_suffix(".mkv") if codec == "h264" else path
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", "640x360",
                    "-framerate", "25", "-i", "pipe:0", "-filter_threads", "1", "-vf", clock, "-fps_mode", "passthrough",
                    "-enc_time_base", "1:1000", "-threads", "1", "-pix_fmt", "yuv420p", *CODECS[codec], str(encoded)],
                   input=b"".join(frames), check=True, timeout=35)
    if encoded != path:
        checked_command("-i", str(encoded), "-map", "0:v:0", "-c", "copy", "-video_track_timescale", "1000", str(path))
        encoded.unlink()
    return [first, first + count - 1, second, second + count - 1] if role.startswith(("late_", "opening")) else None


def pair_positions(pair):
    return [row[2] for row in picture_sequence.endpoints(pair)] if pair is not None else None


def compare(path, role, codec):
    table = probe_frames(path)
    if (table.width, table.height, len(table)) != (WIDTH, HEIGHT, EXTENDED) or table.interpolated:
        raise ValueError("Delivery differs from the planned frame table")
    start = time.monotonic()
    values, observed = [], []
    for frame in decode_frames(path, table, range(EXTENDED), timeout_s=8):
        if not frame.pts_verified or frame.index != len(observed):
            raise ValueError("Delivery lacks its verified source frame")
        values.append(picture_motion.descriptor(frame.rgb))
        observed.append((frame.pts, hashlib.sha256(frame.rgb).hexdigest()))
    if len(observed) != EXTENDED:
        raise ValueError("Incomplete source decode")
    decode_s = time.monotonic() - start
    full_clock = all((b - a) * table.tb * FPS == 1 for a, b in zip(table.pts, table.pts[1:]))
    if full_clock == (role == "opening_after_gap"):
        raise ValueError("Delivery clock differs from the authored role")
    exact = picture_sequence.scan(observed, table.tb)["pair"]
    encoding = (codec, "yuv420p")
    start = time.monotonic()
    old = picture_motion.scan(observed[:OPENING], values[:OPENING], table.tb, encoding, exact,
                              source_profile=(FPS, OPENING))["pair"]
    extra = None
    if exact is None and old is None and full_clock:
        extra = picture_motion.scan(observed, values, table.tb, encoding, exact, source_profile=(FPS, EXTENDED))["pair"]
    return {"exact": pair_positions(exact), "opening": pair_positions(old), "extension": pair_positions(extra),
            "full_clock": full_clock, "pts": list(table.pts), "time_base": table.time_base,
            "rgb_sha256": [digest for _, digest in observed], "decode_and_describe_s": decode_s,
            "motion_comparison_s": time.monotonic() - start}


def run_cases(run):
    freeze_path = run / "freeze.json"
    plan = load(freeze_path)
    if plan["runner_sha256"] != hash_file(Path(__file__)) or plan["method_sha256"] != method_digest():
        raise ValueError("The frozen benchmark code or appearance comparison changed")
    media = run / "media"
    media.mkdir(mode=0o700)
    rows = []
    for i, group in enumerate(plan["sources"]):
        raw = b"".join(source(Path(item["file"]), item["start"], item["frames"]) for item in group["files"]
                       if hash_file(Path(item["file"])) == item["sha256"])
        if len(raw) != FRAME_BYTES * EXTENDED:
            raise ValueError("One or more frozen source files differ")
        for codec in CODECS:
            for role in ROLES:
                path = media / (f"{group['name']}-{role}.mp4" if codec == "h264" else f"{group['name']}-{role}.webm")
                truth = encode(raw, path, i, role, codec)
                rows.append({"source": group["name"], "cohort": group["cohort"], "role": role, "codec": codec, "truth": truth,
                             "file": path.name, "sha256": hash_file(path), **compare(path, role, codec)})
        print(group["name"], len(rows), flush=True)
    write(run / "native.json", {"freeze_sha256": hash_file(freeze_path), "rows": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--publishers", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.publishers and args.archive:
        freeze(args.run, args.publishers, args.archive)
    elif args.publishers or args.archive:
        parser.error("Freeze requires both publisher and camera source folders")
    else:
        run_cases(args.run)
