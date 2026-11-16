# SPDX-License-Identifier: MIT
"""Build reproducible direct and tandem encodes of the paired face-swap panel."""
from __future__ import annotations

import argparse
import subprocess
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts import video_common_sources_v89 as sources
from scripts.video_face_temporal_panel import RECIPE

FPS = RECIPE["fps"]
WIDTH, HEIGHT = RECIPE["size"]
FRAME_COUNT = FPS * RECIPE["duration"]
START = Fraction(RECIPE["source_sample_start_seconds"])
MAX_SOURCE_ERROR = Fraction(1, 24)
MEDIA = ("ordinary", "inswapper", "uniface")


def _picture(rgb):
    import cv2
    import numpy as np

    height, width = rgb.shape[:2]
    ratio = min(WIDTH / width, HEIGHT / height)
    w, h = round(width * ratio), round(height * ratio)
    crop = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR)
    x, y = (WIDTH - w) // 2, (HEIGHT - h) // 2
    result = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    result[y:y+h, x:x+w] = crop
    return result


def sample(path):
    """Select the nearest observed picture at each delivery time, with earlier ties."""
    from app.detectors.video.decode import decode_frames, probe_frames

    table = probe_frames(path, cap=900)
    if table.interpolated or len(table) < 2 or any(a >= b for a, b in zip(table.pts, table.pts[1:])):
        raise ValueError("Native picture timestamps are incomplete or out of order")
    tick = table.tb
    targets = [table.pts[0] + (START + Fraction(i, FPS)) / tick for i in range(FRAME_COUNT)]
    if targets[-1] > table.pts[-1]:
        raise ValueError("The source lacks the fixed four-second delivery window")
    selected = []
    for wanted in targets:
        right = bisect_left(table.pts, wanted)
        index = min((right - 1, right), key=lambda i: (abs(table.pts[i] - wanted), i))
        if abs((table.pts[index] - wanted) * tick) > MAX_SOURCE_ERROR:
            raise ValueError("A native picture is too far from its delivery time")
        selected.append(index)
    decoded = {frame.index: _picture(frame.rgb)
               for frame in decode_frames(path, table, selected, max_side=WIDTH) if frame.pts_verified}
    if set(decoded) != set(selected):
        raise ValueError("Nearest native pictures lack verified decode clocks")
    return [decoded[index] for index in selected], [[index, table.pts[index], table.time_base] for index in selected]


def _encoder(crf):
    return ["-map", "0:v:0", "-an", "-map_metadata", "-1", "-frames:v", str(FRAME_COUNT), "-c:v", "libx264",
            "-threads", "1", "-preset", "medium", "-crf", str(crf), "-g", str(FPS * 2), "-pix_fmt", "yuv420p",
            "-video_track_timescale", "25000", "-movflags", "+faststart", "-fps_mode", "passthrough"]


def encode(pictures, path):
    if path.exists():
        raise FileExistsError(path)
    command = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0", *_encoder(23), str(path)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL) as process:
        with process.stdin as stream:
            for picture in pictures:
                stream.write(memoryview(picture))
        if process.wait(timeout=120):
            raise RuntimeError(f"Direct encoding failed for {path.name}")


def tandem(original, path):
    if path.exists():
        raise FileExistsError(path)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-i", str(original),
                    *_encoder(36), str(path)], capture_output=True, check=True, timeout=120)


def build(plan_path, inputs, destination):
    import cv2

    cv2.setNumThreads(1)
    plan = sources.read(plan_path)
    assert plan["recipe"] == RECIPE
    destination.mkdir(mode=0o700)

    def build_group(group):
        measured = []
        for family in MEDIA:
            tag = f"{group['index']:02d}-{family}"
            original = inputs / f"{tag}.mp4"
            assert sources.sha(original) == group["files"][family]["sha256"]
            try:
                pictures, mapping = sample(original)
            except (ValueError, OSError) as error:
                measured.append({"group": group["group"], "role": group["role"], "family": family, "omission": str(error)})
                continue
            standard = destination / f"{tag}-direct.mp4"
            strong = destination / f"{tag}-tandem.mp4"
            encode(pictures, standard)
            tandem(standard, strong)
            for delivery in (standard, strong):
                measured.append({"group": group["group"], "role": group["role"], "family": family,
                                 "source_sha256": group["files"][family]["sha256"], "file": delivery.name,
                                 "delivery": delivery.stem.rsplit("-", 1)[1], "sha256": sources.sha(delivery),
                                 "native_map": mapping})
        print("Prepared group", group["index"], len([row for row in measured if "file" in row]), "deliveries", flush=True)
        return measured

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = [row for group in pool.map(build_group, plan["groups"]) for row in group]
    sources.write(destination / "manifest.json", {"roster_sha256": sources.sha(plan_path), "recipe": RECIPE, "rows": records})
    print("Prepared", sum("file" in row for row in records), "deliveries.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.roster, args.inputs, args.output)
