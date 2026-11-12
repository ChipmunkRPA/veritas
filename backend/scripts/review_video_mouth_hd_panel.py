"""Replay the HD mouth-hold outcome arithmetic and optional native evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

POSITIVE = {"mouth_early", "mouth_late"}
CONTROLS = {"ordinary_pause", "tandem", "whole_hold", "cut", "mouth_blur", "off_mouth_hold", "tail_only"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def commit(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def portable(record):
    plan = {row["name"]: row for row in record["source_plan"]["rows"] if row["split"] == "held"}
    rows = record["rows"]
    require(len(rows) == len(plan) and set(plan) == {row["name"] for row in rows}, "Held source roster differs")
    require(sum(not row["control_only"] for row in plan.values()) >= 5, "Too few distinct positive-source groups")
    counts, measured, useful, misplaced, control = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in rows:
        expected = {"ordinary_pause", "tandem"} if plan[row["name"]]["control_only"] else POSITIVE | CONTROLS
        require(len(row["cases"]) == len(expected) and {case["role"] for case in row["cases"]} == expected, "Case roster differs")
        for case in row["cases"]:
            role, found, starts = case["role"], case["choice"], case["assessed_starts"]
            counts[role] += 1
            require(starts == sorted(set(starts)) and set(starts) <= {0, 30, 60} and
                    case["status"] == ("matched" if found else "no_match" if starts else "no_stable_face"), "Coverage state differs")
            measured[role] += bool(starts)
            if not found:
                continue
            start, stop = found["frames"]
            require(0 <= start < stop < 120 and found["window_start"] in starts and
                    found["anchors"] == [start-1, (start+stop)//2, stop], "Selected range differs from checked prefix")
            if role not in POSITIVE:
                control[role] += 1
                continue
            before, after = case["authored_frames"]
            x, y, w, h = found["box"]
            a, b, u, v = case["authored_box"]
            common = max(0, min(x+w, a+u) - max(x, a)) * max(0, min(y+h, b+v) - max(y, b))
            overlap = max(0, min(after, stop) - max(before, start))
            (useful if 2 * overlap >= max(after, stop) - min(before, start) and 4 * common >= 3 * w * h else misplaced)[role] += 1
    positives = sum(counts[k] for k in POSITIVE)
    assessed_positive = sum(measured[k] for k in POSITIVE)
    negatives = sum(counts.values()) - positives
    gate = (4 * sum(useful.values()) >= 3 * positives and useful["mouth_late"] >= 4 and 5 * assessed_positive >= 4 * positives
            and 5 * (sum(measured.values()) - assessed_positive) >= 4 * negatives and sum(control.values()) <= 1
            and sum(control[k] for k in ("whole_hold", "cut", "tail_only")) == 0)
    result = {"admitted": gate, "cases": dict(counts), "assessed": dict(measured), "useful": dict(useful),
              "misplaced": dict(misplaced), "control_alerts": dict(control)}
    require(result == record["summary"], "Public summary differs from the case arithmetic")
    return result


def command(*args):
    return subprocess.run(args, timeout=30, stdin=subprocess.DEVNULL, check=True, capture_output=True).stdout


def native(record, run, sources):
    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import mouth_local_hold as method
    from scripts.video_mouth_local_hold_panel import WEIGHTS

    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHTS)
    totals = Counter()
    for row in record["source_plan"]["rows"]:
        require(sha_file(sources / (row["name"] + ".webm")) == row["source_sha256"], "Publisher file hash differs")
    for row in record["rows"]:
        folder = run / row["name"]
        require(sha_file(folder / "base.mp4") == row["base_sha256"], "Normalized reference bytes differ")
        for case in row["cases"]:
            path = folder / (case["role"] + ".mp4")
            require(sha_file(path) == case["sha256"], "Delivered file bytes differ")
            probe = json.loads(command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      "stream=codec_name,pix_fmt,time_base,width,height:frame=best_effort_timestamp", "-of", "json", str(path)))
            (stream,) = probe["streams"]
            pts = [int(item["best_effort_timestamp"]) for item in probe["frames"]]
            base = Fraction(stream["time_base"])
            require((stream["width"], stream["height"], stream["codec_name"], stream["pix_fmt"]) == (1280, 720, "h264", "yuv420p") and
                    len(pts) == case["source_frames"] and all((tick - pts[0]) * base == Fraction(i, 30) for i, tick in enumerate(pts)),
                    "Native delivery clock differs")
            pixels = command("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path), "-an", "-map", "0:v:0",
                             "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-threads", "1", "-frames:v", "120", "-f", "rawvideo", "-")
            frames = np.frombuffer(pixels, dtype=np.uint8).reshape(120, 720, 1280, 3)
            hashes = [digest(frame.tobytes()) for frame in frames]
            require(commit(hashes) == case["prefix_rgb_sha256"] and commit(pts[:120]) == case["prefix_pts_sha256"],
                    "Native prefix pixels or timestamp commitments differ")
            state, choice, starts = method.inspect_hd(frames, detector)
            require((state, choice, starts) == (case["status"], case["choice"], case["assessed_starts"]), "Window decisions differ")
            totals.update(deliveries=1, native_pictures=120)
            if choice:
                method.verify_measurements(frames, choice, hd=True)
                x, y, w, h = choice["box"]
                require([ref["index"] for ref in case["anchors"]] == choice["anchors"], "Evidence roles differ")
                for ref in case["anchors"]:
                    i = ref["index"]
                    require(ref["pts"] == pts[i] and Fraction(ref["time_base"]) == base and ref["frame_sha256"] == hashes[i] and
                            ref["crop_sha256"] == digest(frames[i, y:y+h, x:x+w].tobytes()), "Exact source crop differs")
                totals.update(crops=3, transitions=choice["total_edges"])
            del pixels, frames
        print("replayed", row["name"], flush=True)
    return dict(totals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--sources", type=Path)
    args = parser.parse_args()
    document = json.loads(args.receipt.read_bytes())
    require(document["schema"] == "chipmunk.video-mouth-hd-panel/1", "Unrecognized HD video study")
    print(json.dumps(portable(document), sort_keys=True), flush=True)
    if args.run is not None:
        require(args.sources is not None and sha_file(args.run / "freeze.json") == document["freeze_sha256"], "Frozen method or sources differ")
        print(json.dumps(native(document, args.run, args.sources), sort_keys=True))


if __name__ == "__main__":
    main()
