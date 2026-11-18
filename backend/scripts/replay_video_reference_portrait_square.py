# SPDX-License-Identifier: Apache-2.0
"""Rebuild the matched portrait and square video panel from pinned originals."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import video_reference
from app.evidence.video_reference_photometric import SHORT30_BASIS
from scripts.video_common_sources_v89 import require, sha

SCHEMA = "chipmunk.video-reference-portrait-square/1"
SIZES = {"portrait": (360, 640), "square": (480, 480)}
EDIT_SIZES = {"large_tone": (136, 104), "small_tone": (58, 44), "texture_copy": (88, 74)}
LEAF = re.compile(r"[0-9]{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.flv\Z")


def native_pictures(source, count, size):
    table = probe_frames(source, cap=100, timeout_s=12)
    require(len(table) >= count and not (table.rotation or table.interpolated), "Source picture clock differs")
    pictures = []
    with closing(decode_frames(source, table, range(count), timeout_s=15)) as sequence:
        for index, frame in enumerate(sequence):
            require(frame.index == index and frame.pts_verified, "Source picture lacks an exact timestamp")
            height, width = frame.rgb.shape[:2]
            crop_width = round(height * size[0] / size[1])
            require(0 < crop_width <= width, "Source cannot supply the centered delivery crop")
            left = (width - crop_width) // 2
            pictures.append(cv2.resize(frame.rgb[:, left:left + crop_width], size, interpolation=cv2.INTER_LINEAR))
    require(len(pictures) == count, "Source decoder did not return every picture")
    return np.stack(pictures)


def encode(path, pictures, quality):
    height, width = pictures[0].shape[:2]
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{width}x{height}", "-framerate", "30", "-i", "pipe:0", "-threads", "1",
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-crf", str(quality),
         "-g", "30", "-video_track_timescale", "30000", str(path)],
        input=pictures.tobytes(), check=True, timeout=35,
    )


def edit(pictures, index, kind):
    if kind == "global_tone":
        return np.clip(pictures.astype(np.float32) * 1.1 - 12, 0, 255).astype(np.uint8), None
    if kind not in EDIT_SIZES:
        return pictures, None
    _, height, width, _ = pictures.shape
    sx, sy = [(74, 110), (88, 162), (100, 218), (59, 270)][index % 4]
    x, y = sx * width // 360, sy * height // 480
    w, h = EDIT_SIZES[kind]
    start = 13 + 2 * (index % 5)
    stop = start + (12, 16, 20)[index % 3]
    region = np.s_[start:stop, y:y + h, x:x + w]
    original = pictures[region]
    if kind == "texture_copy":
        copy = pictures[start:stop, y:y + h, x + 60:x + w + 60]
        weight = np.minimum(1, np.minimum(np.arange(w), np.arange(w)[::-1]) / 5)[None, None, :, None]
        patch = np.rint(original * (1 - weight) + copy * weight).astype(np.uint8)
    else:
        gain, offset = ((.70, 59) if index % 2 else (1.24, -52))
        patch = np.clip(original.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
    altered = pictures.copy()
    altered[region] = patch
    mask = np.any(original != patch, axis=-1)
    require(mask.any(axis=(1, 2)).all(), "Rebuilt edit does not change every picture in its interval")
    return altered, (x, y, start, stop, mask)


def verify_location(record, expected, truth):
    local = record["picture"]["local_comparison"]
    require(local["basis"] == SHORT30_BASIS and local["status"] == expected["status"],
            "The rebuilt comparison status differs")
    span = local.get("range")
    require(bool(span) == ("range" in expected), "The presence of a comparison range differs")
    if not span:
        return
    start, end = span["start_frame"], span["end_frame_exclusive"]
    require(([start, end], span["tile"]) == (expected["range"], expected["box"]), "The localization differs")
    for role, anchor in span["previews"].items():
        retained = expected["anchors"][role]
        require(anchor["frame"] == retained["frame"] and
                [anchor[f"{side}_pts"] for side in ("earlier", "submitted")] == retained["source_pts"] and
                [anchor[f"{side}_rgb_sha256"] for side in ("earlier", "submitted")] == retained["source_rgb"],
                "The rebuilt native picture anchor differs")
    if truth is None:
        return
    x, y, original_start, original_stop, mask = truth
    require([original_start, original_stop] == expected["target"], "The authored interval differs")
    shared = max(0, min(end, original_stop) - max(start, original_start))
    time_iou = shared / (max(end, original_stop) - min(start, original_start))
    spatial_iou = 0
    middle = span["previews"]["during"]["frame"]
    if original_start <= middle < original_stop:
        target = mask[middle - original_start]
        box = span["tile"]
        height, width = target.shape
        left, top = max(box["x"] - x, 0), max(box["y"] - y, 0)
        right, bottom = min(box["x"] + box["w"] - x, width), min(box["y"] + box["h"] - y, height)
        overlap = int(target[top:bottom, left:right].sum()) if right > left and bottom > top else 0
        spatial_iou = overlap / (int(target.sum()) + box["w"] * box["h"] - overlap)
    require((time_iou, spatial_iou) == (expected["time_iou"], expected["pixel_iou"]),
            "The rebuilt temporal or authored-pixel overlap differs")


def replay(receipt, originals, output):
    require(receipt["schema"] == SCHEMA, "Unexpected video comparison receipt")
    paths = {}
    for source in receipt["sources"]:
        leaf = source["publisher_leaf"]
        require(LEAF.fullmatch(leaf) is not None, "Invalid publisher filename")
        path = originals / leaf
        require(sha(path) == source["sha256"], f"Publisher original differs for {source['group']}")
        paths[source["group"]] = path
    readings = {(row["profile"], row["group"], row["role"]): row for row in receipt["rows"]}
    output.mkdir(mode=0o700)
    completed = 0
    for profile, size in SIZES.items():
        for index, source in enumerate(receipt["sources"]):
            group, count = source["group"], source["frames"]
            require(re.fullmatch(r"camera-0[1-8]", group) is not None, "Invalid camera group")
            pictures = native_pictures(paths[group], count, size)
            earlier = output / f"{profile}-{group}-earlier.mp4"
            encode(earlier, pictures, 23)
            for role in (*EDIT_SIZES, "identical", "recompressed", "global_tone", "wrong"):
                row = readings[profile, group, role]
                submitted = output / f"{profile}-{group}-{role}.mp4"
                altered, truth = edit(pictures, index, role)
                if role == "wrong":
                    altered = native_pictures(paths[row["comparison_source_group"]], count, size)
                if role == "identical":
                    shutil.copyfile(earlier, submitted)
                else:
                    encode(submitted, altered, 28 if truth is not None else 32)
                require([earlier.name, submitted.name] == row["filename_pair"] and
                        [sha(earlier), sha(submitted)] == row["input_sha256"],
                        f"Encoded deliveries differ for {profile} {group} {role}")
                record, _ = video_reference.compare(earlier, submitted, include_short30_local_area=True)
                require(record["status"] == "measured" and record["affects_overall_confidence"] is False,
                        "Unexpected rebuilt comparison disposition")
                verify_location(record, row, truth)
                completed += 1
            print(f"Verified {profile} {group}", flush=True)
    require(completed == len(readings) == len(receipt["rows"]) == 112, "Incomplete comparison replay")
    return {"comparison_pairs": completed, "status": "verified", "deliveries": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, help="Published compressed study receipt")
    parser.add_argument("originals", type=Path, help="Directory containing the pinned CREMA-D source names")
    parser.add_argument("output", type=Path, help="New directory in which to retain the rebuilt clips")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(replay(json.loads(gzip.decompress(args.receipt.read_bytes())), args.originals, args.output)))


if __name__ == "__main__":
    main()
