"""Build and verify paired face-swap controls with exact frame edit boundaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

from prepare_video_upgrade_eval import sha

FPS = 24
POLICY = "chipmunk-face-interval/1"
FFMPEG = ["ffmpeg", "-v", "error", "-nostdin", "-y"]
ENCODE = ["-c:v", "libx264", "-qp", "0", "-threads", "2"]
FRAME_BYTES = 640 * 480 * 3 // 2


def run(args: list[str]) -> bytes:
    return subprocess.check_output([*FFMPEG, *args], timeout=60)


def hashes(path: Path) -> list[str]:
    lines = run(["-i", str(path), "-map", "0:v:0", "-f", "framemd5", "pipe:1"]).decode().splitlines()
    return [line.rsplit(",", 1)[-1].strip() for line in lines if not line.startswith("#")]


def prepare(rows: list[dict], root: Path, output: Path, *, development: bool, hard_controls: bool = False) -> list[dict]:
    paths, digests = [], []
    for row in rows:
        item = row["matched"]
        source = root / item["path"]
        if sha(source) != item["sha256"]:
            raise ValueError(f"source differs from acquisition receipt: {row['id']}")
        path = output / "normalized" / f"{row['id']}.mp4"
        run(["-i", str(source), "-map", "0:v:0", "-map_metadata", "-1", "-vf",
             "fps=24,scale=640:480:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p",
             *ENCODE, str(path)])
        paths.append(path)
        digests.append(hashes(path))
    count = min(map(len, digests))
    # Keep at least 0.75 s of original context on either side of each edit.
    if count < 90:
        return [{"id": row["id"], "group": row["group"], "family": row["family"], "status": "short",
                 "split": "development" if development else "test", "label": row["label"]} for row in rows]
    choice = hashlib.sha256(rows[0]["group"].encode()).digest()
    duration = (30, 36, 48)[choice[0] % 3]
    start = 18 + choice[1] % (count - duration - 35)
    end = start + duration
    original = run(["-i", str(paths[0]), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"])
    result = []
    for row, insert, pixels in zip(rows, paths, digests, strict=True):
        target = output / f"{row['id']}.mp4"
        altered = (original if insert == paths[0] else run(
            ["-i", str(insert), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]))
        a, b, limit = (index * FRAME_BYTES for index in (start, end, count))
        source = b"".join((memoryview(original)[:a], memoryview(altered)[a:b], memoryview(original)[b:limit]))
        subprocess.run([*FFMPEG, "-f", "rawvideo", "-pixel_format", "yuv420p", "-video_size", "640x480",
                        "-framerate", str(FPS), "-i", "pipe:0", *ENCODE, str(target)],
                       input=source, check=True, timeout=60)
        expected = digests[0][:start] + pixels[start:end] + digests[0][end:count]
        if hashes(target) != expected:
            raise ValueError(f"could not verify edited frames: {row['id']}")
        unchanged = row["label"] and pixels[start:end] == digests[0][start:end]
        result.append({"id": row["id"], "group": row["group"], "family": row["family"], "status": "ready",
                       "split": "development" if development else "test", "label": row["label"], "path": target.name,
                       "sha256": sha(target), "frame_count": count, "fps": FPS, "inserted_frames": [start, end],
                       "edit_s": [start / FPS, end / FPS] if row["label"] else None, "exact_frames_verified": True})
        if unchanged:
            result[-1]["status"] = "unchanged-insert"
    if hard_controls:
        original_case = result[0]
        target = output / f"{rows[0]['id']}-ordinary.mp4"
        benign = ("jump-cut", "lighting-change", "recompression")[choice[2] % 3]
        if benign == "jump-cut":
            transform = ["-vf", f"select='not(between(n,{start},{start+7}))',setpts=N/({FPS}*TB)", *ENCODE]
        elif benign == "lighting-change":
            transform = ["-vf", f"eq=brightness=0.14:enable='between(n,{start},{end-1})'", *ENCODE]
        else:
            transform = ["-c:v", "libx264", "-crf", "38", "-threads", "2"]
        run(["-i", str(output / original_case["path"]), *transform, str(target)])
        control_frames = hashes(target)
        if control_frames == digests[0][:count]:
            raise ValueError(f"ordinary control was not modified: {target.name}")
        if benign == "jump-cut":
            expected = digests[0][:start] + digests[0][start+8:count]
            if [value for value, _ in groupby(control_frames)] != [value for value, _ in groupby(expected)]:
                raise ValueError(f"ordinary cut does not preserve the expected source content: {target.name}")
        result.append({**original_case, "id": f"{rows[0]['id']}-ordinary", "path": target.name,
                       "sha256": sha(target), "family": benign, "frame_count": len(control_frames),
                       "inserted_frames": None, "exact_frames_verified": benign == "jump-cut"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--development-groups", type=int, default=8)
    parser.add_argument("--hard-controls", action="store_true")
    args = parser.parse_args()
    lock = args.output / "manifest.json"
    if lock.exists():
        parser.error("manifest already frozen")
    (args.output / "normalized").mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in json.loads(args.sources.read_text())["items"]:
        groups[row["group"]].append(row)
    items = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(prepare, rows, args.sources.parent, args.output, development=i < args.development_groups,
                               hard_controls=args.hard_controls)
                   for i, rows in enumerate(groups.values())]
        for future in futures:
            group = future.result()
            items.extend(group)
            print(group[0]["group"], group[0]["status"], flush=True)
    with lock.open("x") as f:
        json.dump({"policy": POLICY, "source_manifest_sha256": sha(args.sources), "items": items}, f, indent=2)
    print("manifest SHA-256", sha(lock), flush=True)


if __name__ == "__main__":
    main()
