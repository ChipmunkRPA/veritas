"""Compare 2:1 picture reuse against the existing exact and lossy repeat checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from app.detectors.video import picture_motion, picture_speed
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import picture_sequence as binding

WIDTH, HEIGHT, RATE, LENGTH = 640, 360, 25, 225
CASES = (
    "later_fast", "earlier_fast", "later_slow", "earlier_slow", "original",
    "in_place_speed", "jump_cut", "held_picture", "same_speed", "reverse", "original_tandem", "later_fast_tandem",
)


def _sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _write(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def _input_pictures(path, offset):
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2", "-filter_threads", "1",
        "-ss", str(offset), "-i", str(path), "-map", "0:v:0", "-an",
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
               f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={RATE}",
        "-frames:v", str(LENGTH + 24), "-pix_fmt", "yuv420p", "-f", "rawvideo", "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True, timeout=45).stdout
    bytes_per_frame = WIDTH * HEIGHT * 3 // 2
    if len(raw) != (LENGTH + 24) * bytes_per_frame:
        raise ValueError(f"{path.name} did not have enough decodable pictures")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, bytes_per_frame)


def _delivery(pictures, early, late, role):
    output = pictures[:LENGTH].copy()
    first, second = (late, early) if role.startswith("earlier") else (early, late)
    if role in {"later_fast", "earlier_fast"}:
        output[second:second + 21] = pictures[first:first + 41:2]
        truth = sorted(((first, first + 40, 2), (second, second + 20, 1)))
    elif role in {"later_slow", "earlier_slow"}:
        output[second:second + 42] = np.repeat(pictures[first:first + 21], 2, axis=0)
        truth = sorted(((first, first + 20, 1), (second, second + 40, 2)))
    else:
        truth = None
        if role == "in_place_speed":
            output = np.concatenate((pictures[:early], pictures[early:early + 42:2], pictures[early + 42:]))[:LENGTH]
        if role == "jump_cut":
            output = np.concatenate((pictures[:early], pictures[early + 21:]))[:LENGTH]
        if role == "held_picture":
            output[late:late + 21] = pictures[early]
        if role == "same_speed":
            output[late:late + 21] = pictures[early:early + 21]
        if role == "reverse":
            output[late:late + 21] = pictures[early:early + 21][::-1]
    return output, truth


def _encode(path, pictures):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(RATE), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-threads", "2", "-g", "50", "-preset", "veryfast", "-crf", "25", str(path),
    ], input=memoryview(pictures), capture_output=True, check=True, timeout=25)


def build(output, sources):
    output.mkdir()
    planned = []
    for source in sources:
        path = Path(source["path"])
        digest = _sha(path)
        if source.get("sha256", digest) != digest:
            raise ValueError(f"Source hash differs for {source['id']}")
        pictures = _input_pictures(path, source["offset_s"])
        early, late = 25 + int(digest[:2], 16) % 16, 132 + int(digest[2:4], 16) % 16
        for role in CASES:
            name = f"{source['id']}-{role}.mp4"
            destination = output / name
            original_role = role.removesuffix("_tandem")
            if original_role == role:
                frames, truth = _delivery(pictures, early, late, role)
                _encode(destination, frames)
            else:
                prior = next(row for row in planned if row["source"] == source["id"] and row["role"] == original_role)
                truth = prior["truth"]
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "2",
                    "-i", str(output / prior["delivery"]), "-an", "-c:v", "libx264", "-threads", "2",
                    "-g", "40", "-preset", "veryfast", "-crf", "33", str(destination),
                ], capture_output=True, check=True, timeout=25)
            planned.append({"source": source["id"], "role": role, "delivery": name,
                            "sha256": _sha(destination), "truth": truth})
        print(f"prepared {source['id']}", flush=True)
    _write(output / "plan.json", {"sources": [{k: v for k, v in source.items() if k != "path"}
                                              for source in sources], "cases": planned})


def _current(observations, descriptors, table):
    if not all(tick is not None and tick == table.pts[i] for i, (tick, _) in enumerate(observations)):
        raise ValueError("A delivery picture did not reproduce its native timestamp")
    result = []
    for reverse, schema in ((False, binding.MOTION_SCHEMA), (True, binding.MOTION_REVERSE_SCHEMA)):
        exact = binding.scan(observations, table.tb, reverse=reverse)
        lossy = picture_motion.scan(observations, descriptors, table.tb, ("h264", "yuv420p"), exact["pair"],
                                    source_profile=(RATE, LENGTH), schema=schema)
        if lossy["reason"] == "source_clock":
            raise ValueError("A delivery did not have the expected 25 fps presentation clock")
        if pair := exact["pair"] or lossy["pair"]:
            result.append([(pair[side]["start_frame_index"], pair[side]["end_frame_index"], 1)
                           for side in ("first", "second")])
    return result


def _predicted(found):
    return None if found is None else [(start, start + step * (found[1] - 1), step) for start, step in found[0]]


def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1) / (max(a[1], b[1]) - min(a[0], b[0]) + 1)


def useful(predicted, truth):
    return bool(predicted and truth and all(a[2] == b[2] and _overlap(a, b) >= .5
                                           for a, b in zip(predicted, truth, strict=True)))


def score(run, report):
    plan = json.loads((run / "plan.json").read_text())
    cases = []
    for row in plan["cases"]:
        path = run / row["delivery"]
        if _sha(path) != row["sha256"]:
            raise ValueError(f"Delivery hash differs for {path.name}")
        table = probe_frames(path)
        observations, descriptors = [], []
        with closing(decode_frames(path, table, range(LENGTH))) as frames:
            for frame in frames:
                if frame.index != len(observations) or not frame.pts_verified:
                    raise ValueError(f"Unverified source picture in {path.name}")
                observations.append((frame.pts, hashlib.sha256(frame.rgb).hexdigest()))
                descriptors.append(picture_motion.descriptor(frame.rgb))
        if len(table) != len(observations) or len(observations) != LENGTH:
            raise ValueError(f"Unexpected picture count in {path.name}")
        current = _current(observations, descriptors, table)
        before = time.perf_counter()
        found = picture_speed.find(descriptors, RATE)
        elapsed = time.perf_counter() - before
        candidate = _predicted(found)
        adopted = None if current else candidate
        anchors = [[observations[index] for index in interval[:2]] for interval in adopted] if adopted else []
        cases.append(row | {
            "current": current, "candidate_alone": candidate, "added": adopted,
            "useful": useful(adopted, row["truth"]), "support": found[2] if adopted else [],
            "candidate_ms": round(elapsed * 1000, 3), "time_base": table.time_base, "anchors": anchors,
        })
        print(path.stem, current, adopted, cases[-1]["useful"], flush=True)
    _write(report, plan | {"cases": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("run", type=Path)
    builder.add_argument("sources", type=Path)
    scorer = commands.add_parser("score")
    scorer.add_argument("run", type=Path)
    scorer.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        build(args.run, json.loads(args.sources.read_text()))
    else:
        score(args.run, args.report)


if __name__ == "__main__":
    main()
