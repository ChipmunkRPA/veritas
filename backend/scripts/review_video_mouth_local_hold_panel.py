"""Review held mouth-patch outcomes, clocks and exact source crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

POSITIVE = {"mouth_early", "mouth_late"}
CONTROL = {"original", "tandem", "whole_hold", "whole_cut", "mouth_blur", "off_mouth_hold"}
GROUPS = {"grid": {10, 11, 12, 13, 14, 15}, "crema": {5, 6, 7, 9, 11, 12, 14, 15}}


def assert_that(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def portable(record):
    totals = {}
    for bank, groups in GROUPS.items():
        assessed, cases, locations, wrong = Counter(), Counter(), Counter(), Counter()
        selected = set()
        rows = [row for row in record["rows"] if row["bank"] == bank]
        assert_that(len(rows) == len(groups) and {row["group"] for row in rows} == groups, "Held source roster differs")
        for row in rows:
            assert_that(len(row["cases"]) == len(POSITIVE | CONTROL) and
                        {case["role"] for case in row["cases"]} == POSITIVE | CONTROL, "Held control roster differs")
            for case in row["cases"]:
                role, status, choice = case["role"], case["state"], case["choice"]
                cases[role] += 1
                assert_that(status in {"no_stable_face", "no_match", "matched"} and bool(choice) == (status == "matched"),
                            "Evidence selection differs from its declared status")
                assessed[role] += status != "no_stable_face"
                if not choice:
                    continue
                if role in POSITIVE:
                    x, y, w, h = choice["box"]
                    a, b, u, v = case["authored_box"]
                    intersection = max(0, min(x+w, a+u) - max(x, a)) * max(0, min(y+h, b+v) - max(y, b))
                    l, r = choice["frames"]
                    p, q = case["authored_frames"]
                    if 4 * intersection >= 3 * w * h and 2 * max(0, min(r, q) - max(l, p)) >= max(r, q) - min(l, p):
                        selected.add(row["group"])
                        locations[role] += 1
                        continue
                wrong[role] += 1
        minimum, sources = (8, 4) if bank == "grid" else (10, 5)
        admitted = (sum(locations.values()) >= minimum and len(selected) >= sources and
                    sum(wrong[role] for role in CONTROL) <= 1 and not (wrong["whole_cut"] or wrong["whole_hold"]) and
                    all(5 * sum(assessed[role] for role in roles) >= 4 * sum(cases[role] for role in roles)
                        for roles in (POSITIVE, CONTROL)))
        totals[bank] = {"admitted": admitted, "groups": len(selected), "true_locations": dict(locations),
                        "assessed": dict(assessed), "cases": dict(cases), "false_or_mislocated": dict(wrong)}
    assert_that(totals == record["results"], "Reported outcomes differ from held measurements")
    return totals


def command(*args):
    return subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=16).stdout


def replay_group(run, row):
    import cv2
    import numpy as np

    from app.detectors.image.faces import load_yunet
    from app.detectors.video import mouth_local_hold as method
    from scripts.video_mouth_local_hold_panel import WEIGHTS

    cv2.setNumThreads(1)
    detector = load_yunet(WEIGHTS)
    counters = Counter()
    folder = run / f"{row['bank']}-{row['group']:02d}"
    for case in row["cases"]:
        path = folder / f"{case['role']}.mp4"
        assert_that(digest(path.read_bytes()) == case["sha256"], "Delivered media differs")
        probe = json.loads(command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "frame=best_effort_timestamp:stream=codec_name,pix_fmt,width,height,time_base", "-of", "json", str(path)))
        stream, = probe["streams"]
        width, height, base = stream["width"], stream["height"], Fraction(stream["time_base"])
        pts = [int(item["best_effort_timestamp"]) for item in probe["frames"]]
        assert_that((stream["codec_name"], stream["pix_fmt"]) == ("h264", "yuv420p") and len(pts) == 50 and
                    all((point - pts[0]) * base == Fraction(i, 25) for i, point in enumerate(pts)), "Native picture clock differs")
        pixels = command("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(path), "-an", "-map", "0:v:0",
                         "-vf", "format=rgb24", "-fps_mode", "passthrough", "-threads", "1", "-f", "rawvideo", "-")
        frames = np.frombuffer(pixels, dtype=np.uint8).reshape(50, height, width, 3)
        hashes = [digest(frame.tobytes()) for frame in frames]
        assert_that(canonical(hashes) == case["native_rgb_sha256"] and canonical(pts) == case["native_picture_pts_sha256"],
                    "Native picture commitments differ")
        status, choice = method.inspect(frames, detector)
        assert_that((status, choice) == (case["state"], case["choice"]), "Timing or box selection differs")
        counters.update(deliveries=1, frames=len(pts))
        if not choice:
            continue
        x, y, w, h = choice["box"]
        start, stop = choice["frames"]
        gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]).astype(np.int16)
        changes = np.abs(np.diff(gray, axis=0))
        core = changes[start:stop - 1, y:y + h, x:x + w]
        values = changes[:, y:y + h, x:x + w].mean(axis=(1, 2))
        regions = [changes[start:stop - 1, cy:cy + ch, cx:cx + cw].mean(axis=(1, 2))
                   for cx, cy, cw, ch in choice["context_boxes"]]
        motion = np.count_nonzero(np.count_nonzero(np.asarray(regions) >= .75, axis=0) >= 2)
        flank = min(values[start - 4:start].max(), values[stop - 1:stop + 3].max())
        assert_that(float(core.mean(axis=(1, 2)).max()) == choice["maximum_patch_mean_difference"] and
                    float((core <= 1).mean(axis=(1, 2)).min()) == choice["minimum_patch_near_equal_fraction"] and
                    motion == choice["moving_context_edges"] and flank == choice["minimum_flank_peak"], "Native local statistics differ")
        assert_that([ref["index"] for ref in case["anchors"]] == [start - 1, (start + stop) // 2, stop], "Context anchor roles differ")
        for ref in case["anchors"]:
            i = ref["index"]
            assert_that(ref["pts"] == pts[i] and Fraction(ref["time_base"]) == base and ref["frame_sha256"] == hashes[i] and
                        ref["crop_sha256"] == digest(frames[i, y:y+h, x:x+w].tobytes()), "Exact context image or crop differs")
        counters.update(crops=len(case["anchors"]), measured_edges=stop - start - 1)
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    document = json.loads(args.receipt.read_bytes())
    assert_that(document["schema"] == "chipmunk.video-mouth-local-hold-panel/1", "Unexpected evidence schema")
    print(json.dumps(portable(document), sort_keys=True), flush=True)
    if args.run:
        assert_that(digest((args.run / "freeze.json").read_bytes()) == document["freeze_sha256"], "Source and method freeze differs")
        totals = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row, result in zip(document["rows"], pool.map(lambda item: replay_group(args.run, item), document["rows"]), strict=True):
                totals.update(result)
                print(f"Verified {row['bank']} {row['group']:02d}", flush=True)
        print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    main()
