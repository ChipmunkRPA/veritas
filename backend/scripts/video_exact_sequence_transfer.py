"""Author and replay matched repeated-picture deliveries without model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.video import picture_hold
from app.detectors.video.decode import probe_frames
from scripts.video_extended_picture_hold import _frame_hashes

WIDTH, HEIGHT, FPS, FRAMES = 320, 180, 20, 180
FIRST, SECOND, LENGTH = 20, 130, 22
KINDS = ("original", "ordinary_cut", "static_hold", "exact_copy", "different_spacing", "lossy_copy")
ENCODINGS = {"h264": ("-c:v", "libx264rgb", "-crf", "0", "-preset", "ultrafast"),
             "vp9": ("-c:v", "libvpx-vp9", "-lossless", "1", "-deadline", "realtime", "-cpu-used", "5", "-pix_fmt", "yuv420p"),
             "lossy": ("-c:v", "libx264", "-crf", "23", "-g", "18", "-preset", "fast", "-pix_fmt", "yuv420p")}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_source(path):
    if path is None:
        y, x = np.mgrid[:HEIGHT, :WIDTH]
        return b"".join(np.stack(((x + 13 * i) % 256, (y * 5 + i) % 256, (x + y + 3 * i) % 256), axis=-1).astype("u1").tobytes()
                        for i in range(FRAMES))
    return subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", f"file:{path}",
                                    "-map", "0:v:0", "-an", "-vf", f"fps={FPS},scale={WIDTH}:{HEIGHT}:flags=area",
                                    "-frames:v", str(FRAMES), "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"], timeout=30)


def deliver(raw, target, kind, codec):
    size = WIDTH * HEIGHT * 3
    if len(raw) != size * FRAMES:
        raise ValueError("The source does not provide the full nine-second frame raster")
    source = memoryview(raw)
    frames = [source[i * size:(i + 1) * size] for i in range(FRAMES)]
    if kind in {"exact_copy", "different_spacing", "lossy_copy"}:
        frames[SECOND:SECOND + LENGTH] = frames[FIRST:FIRST + LENGTH]
    elif kind == "static_hold":
        frames[SECOND:SECOND + LENGTH] = [frames[SECOND]] * LENGTH
    elif kind == "ordinary_cut":
        changed = np.frombuffer(raw, dtype="u1").reshape(FRAMES, HEIGHT, WIDTH, 3)
        frames[SECOND:SECOND + LENGTH] = [frame[:, ::-1].tobytes() for frame in changed[SECOND:SECOND + LENGTH]]
    clock = r"settb=expr=1/1000,setpts=N*50+10*mod(N\,2)*between(N\,130\,151)" if kind == "different_spacing" else "settb=expr=1/1000,setpts=N*50"
    encoder = ENCODINGS["lossy" if kind == "lossy_copy" else codec]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}",
                    "-framerate", str(FPS), "-i", "pipe:0", "-vf", clock, "-fps_mode", "passthrough", "-enc_time_base", "1:1000",
                    *encoder, str(target)], input=b"".join(frames), check=True, timeout=30)


def oracle(pts, hashes, tb):
    """Separate starting-position enumeration over external FFmpeg framehash rows."""
    by_digest = defaultdict(list)
    for index, digest in enumerate(hashes):
        by_digest[digest].append(index)
    winner = None
    for i, digest in enumerate(hashes):
        for j in by_digest[digest]:
            if j - i < 8:
                continue
            n, distinct = 0, set()
            while i + n < j and j + n < len(hashes) and hashes[i + n] == hashes[j + n]:
                if n:
                    delta = pts[i + n] - pts[i + n - 1]
                    if not 0 < delta * tb <= .125 or delta != pts[j + n] - pts[j + n - 1]:
                        break
                distinct.add(hashes[i + n])
                n += 1
            span = pts[i + n - 1] - pts[i]
            if len(distinct) >= 8 and span * tb >= .5:
                candidate = (span, -i, -j, n)
                if winner is None or candidate > winner:
                    winner = candidate
    if winner is None:
        return None
    _, first, second, n = winner
    return [-first, -first + n - 1, -second, -second + n - 1]


def assess(target):
    table = probe_frames(target)
    started = time.monotonic()
    record = picture_hold.inspect(target, table, origin=0., sequence_review=True)
    elapsed = time.monotonic() - started
    cover = record["coverage"]
    if cover["frames_scanned"] != FRAMES:
        raise ValueError("The exact comparison did not finish the matched native prefix")
    hashes = _frame_hashes(target, table)
    expected = oracle(table.pts, hashes, table.tb)
    pair = cover["sequence_review"]["pair"]
    actual = [pair[side][f"{end}_frame_index"] for side in ("first", "second") for end in ("start", "end")] if pair else None
    if actual != expected:
        raise ValueError("The independent search found a different pair on the native clock")
    return {"matched_frames": actual, "checked_frames": len(hashes), "clock": table.time_base,
            "held_picture": record["match"] is not None, "seconds": round(elapsed, 4)}


def run(output, webcams):
    output.mkdir(mode=0o700)
    files = webcams or [None]
    inputs = [read_source(path) for path in files]
    plan = [{"group": i + 1, "source_rgb_sha256": sha(raw), "encoding": "h264" if i % 2 == 0 else "vp9"}
            for i, raw in enumerate(inputs)]
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    with (output / "outcomes.jsonl").open("x") as journal:
        for source, raw in zip(plan, inputs, strict=True):
            for kind in KINDS:
                path = output / f"{source['group']}-{kind}.mkv"
                deliver(raw, path, kind, source["encoding"])
                row = {"group": source["group"], "delivery": kind, "encoding": "lossy" if kind == "lossy_copy" else source["encoding"],
                       **assess(path)}
                if kind == "exact_copy" and row["matched_frames"] != [FIRST, FIRST + LENGTH - 1, SECOND, SECOND + LENGTH - 1]:
                    raise ValueError("The exact authored repeat did not produce its intended evidence bounds")
                journal.write(json.dumps(row, sort_keys=True) + "\n")
                journal.flush()
                rows.append(row)
                path.unlink()
    summary = {"schema": "chipmunk.video-exact-sequence-transfer/1", "groups": len(files), "webcam_groups": len(webcams),
               "source_rgb_commitments": plan, "independent_frame_hashes": sum(row["checked_frames"] for row in rows),
               "matches": dict(Counter(row["delivery"] for row in rows if row["matched_frames"] is not None)), "outcomes": rows}
    (output / "aggregate.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Replayed {len(rows)} exact native deliveries: {summary['matches']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--webcam", type=Path, action="append", default=[])
    args = parser.parse_args()
    run(args.output, args.webcam)
