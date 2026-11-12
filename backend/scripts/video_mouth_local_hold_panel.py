"""Frozen native-pixel evaluation of a local mouth-patch hold in one video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.image.faces import load_yunet
from app.detectors.video import mouth_local_hold as method
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.video_reference_soundtrack_panel import digest, read, require, save

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "backend/data/independent-evaluation"
GRID = DATA / "grid-fresh-release-qualification-2026-11-12"
CREMA = DATA / "video-lipsync-local-delay-recovery-2026-11-11"
WEIGHTS = ROOT / "models/weights/face_detection_yunet_2023mar.onnx"
POSITIVES = ("mouth_early", "mouth_late")
CONTROLS = ("original", "tandem", "whole_hold", "whole_cut", "mouth_blur", "off_mouth_hold")
GATE = {"grid_minimum": 8, "crema_minimum": 10, "grid_groups": 4, "crema_groups": 5,
        "maximum_alerts_per_bank": 1, "maximum_cut_or_whole_hold_alerts": 0, "time_iou": "1/2",
        "minimum_region_precision": "3/4", "minimum_assessed_per_bank": "4/5"}


def bindings():
    return {str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__), Path(method.__file__), WEIGHTS, GRID / "freeze.json", CREMA / "intake-receipt.json")}


def freeze(run):
    run.mkdir(mode=0o700, exist_ok=False)
    grid = [dict(bank="grid", group=row["group"], split=row["split"], path=row["path"], sha256=row["source_sha256"])
            for row in read(GRID / "freeze.json")["roster"] if row["id"].endswith("B-1")]
    crema = [dict(bank="crema", group=row["ordinal"], split=row["split"],
                  path=str((CREMA / row["file"]).relative_to(ROOT)), sha256=row["sha256"])
             for row in read(CREMA / "intake-receipt.json")["rows"] if row["ordinal"] not in {8, 10, 13}]
    require(len(grid) == 8 and len(crema) == 12, "Prespecified roster incomplete")
    # Only the first verified 1.96 seconds are used. The three exclusions do not have that much original picture support.
    save(run / "freeze.json", {"schema": "chipmunk.video-mouth-local-hold-freeze/1", "rows": grid + crema,
                              "bindings": bindings(), "positive": POSITIVES, "controls": CONTROLS, "gate": GATE,
                              "delivery": "H.264 yuv420p CRF 21, 50 pictures at 25 fps, 360x288 or 480x360"})


def source(row):
    path = ROOT / row["path"]
    require(digest(path) == row["sha256"], "Retained qualified source differs")
    table = probe_frames(path, cap=140, timeout_s=10)
    clocks = [(point - table.pts[0]) * table.tb for point in table.pts]
    require(clocks[-1] >= (method.FRAME_COUNT - 1) * method.STEP and not table.interpolated and not table.rotation,
            "Original lacks verified coverage of the fixed native-derived clock")
    selected = []
    for i in range(method.FRAME_COUNT):
        target = i * method.STEP
        after = bisect_left(clocks, target)
        selected.append(min(range(max(0, after - 1), after + 1), key=lambda j: (abs(clocks[j] - target), j)))
    available = {frame.index: frame for frame in decode_frames(path, table, sorted(set(selected)), timeout_s=8)}
    require(all(i in available and available[i].pts_verified and available[i].pts == table.pts[i] for i in selected),
            "Could not verify the source frame mapping")
    return [available[i].rgb for i in selected], [table.pts[i] for i in selected]


def encode(path, frames, crf):
    h, w = frames[0].shape[:2]
    command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
               "-framerate", "25", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-threads", "1", str(path)]
    subprocess.run(command, input=np.stack(frames).tobytes(), capture_output=True, check=True, timeout=16)


def _patch(frames, interval, geometry, role):
    a, b = interval
    h, w = frames[0].shape[:2]
    fw, fh = round(geometry.width * 1.40), round(geometry.width * .98)
    offset_y = round(geometry.width * 1.22) if role == "off_mouth_hold" else 0
    x = round(geometry.x - fw / 2)
    y = round(geometry.y - fh / 2) - offset_y
    require(0 <= x < x + fw <= w and 0 <= y < y + fh <= h and fw > 10 and fh > 10, "Unusable patch geometry")
    col, row = np.arange(fw), np.arange(fh)
    fade = np.minimum(np.minimum(col + 1, fw - col)[None, :], np.minimum(row + 1, fh - row)[:, None])
    alpha = np.clip(fade / 5, 0, 1)[..., None]
    sample = frames[a - 3][y:y + fh, x:x + fw]
    output = frames.copy()
    changes = []
    for i in range(a, b):
        target = frames[i][y:y + fh, x:x + fw]
        replacement = cv2.GaussianBlur(target, (9, 9), 2.3) if role == "mouth_blur" else sample
        value = np.rint((1 - alpha) * target + alpha * replacement).astype(np.uint8)
        changes.append(int(np.count_nonzero(np.any(value != target, axis=2))))
        output[i] = frames[i].copy()
        output[i][y:y + fh, x:x + fw] = value
    require(min(changes) >= 8, "Expected actual RGB changes on every authored picture")
    return output, [x + 4, y + 4, fw - 8, fh - 8], changes


def content(frames, donor, detector, group):
    shift = group % 3
    for role in POSITIVES + CONTROLS:
        start = 12 + shift if role == "mouth_early" else 25 + shift
        interval = start, start + (8 if role == "mouth_early" else 10)
        a, b = interval
        if role in POSITIVES or role in {"mouth_blur", "off_mouth_hold"}:
            geometry, count = method.source_geometry(frames[(a + b) // 2], detector)
            require(count == 1, "Authored local edit needs one independently measured mouth")
            clips, box, changed = _patch(frames, interval, geometry, role)
            yield role, clips, interval, box, changed
        elif role in {"whole_hold", "whole_cut"}:
            samples = [frames[a - 3]] * (b - a) if role == "whole_hold" else donor[a:b]
            output = frames[:a] + samples + frames[b:]
            yield role, output, interval, None, None
        else:
            yield role, frames, None, None, None


def _records(table, frames, choice):
    if choice is None:
        return []
    x, y, w, h = choice["box"]
    return [{"index": i, "pts": table.pts[i], "time_base": table.time_base,
             "frame_sha256": hashlib.sha256(frames[i].tobytes()).hexdigest(),
             "crop_sha256": hashlib.sha256(frames[i][y:y+h, x:x+w].tobytes()).hexdigest()}
            for i in choice["anchors"]]


def score(run, row, donor):
    detector = load_yunet(WEIGHTS)
    folder = run / f"{row['bank']}-{row['group']:02d}"
    folder.mkdir(mode=0o700, exist_ok=False)
    frames, mapped_pts = source(row)
    alternative, _ = source(donor)
    cases = []
    for role, raw, truth, box, changed in content(frames, alternative, detector, row["group"]):
        path = folder / f"{role}.mp4"
        if role == "tandem":
            raw = method.native(folder / "original.mp4")[1]
        encode(path, raw, 34 if role == "tandem" else 21)
        table, delivered = method.native(path)
        state, choice = method.inspect(delivered, detector)
        cases.append({"role": role, "state": state, "authored_frames": truth, "authored_box": box,
                      "authored_changes": changed, "choice": choice, "anchors": _records(table, delivered, choice),
                      "sha256": digest(path), "native_rgb_sha256": canonical_hash([
                          hashlib.sha256(frame.tobytes()).hexdigest() for frame in delivered]),
                      "native_picture_pts_sha256": canonical_hash(list(table.pts))})
    value = {"bank": row["bank"], "group": row["group"], "source_sha256": row["sha256"], "mapped_pts_sha256": canonical_hash(mapped_pts),
             "donor_group": donor["group"], "cases": cases}
    save(folder / "results.json", value)
    print(f"Verified {row['bank']} {row['group']:02d} with {len(cases)} deliveries", flush=True)


def intersection(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def useful(case):
    a, b = case["choice"]["frames"]
    start, end = case["authored_frames"]
    if Fraction(intersection(a, b, start, end), max(b, end) - min(a, start)) < Fraction(GATE["time_iou"]):
        return False
    x, y, w, h = case["choice"]["box"]
    ax, ay, aw, ah = case["authored_box"]
    area = intersection(x, x + w, ax, ax + aw) * intersection(y, y + h, ay, ay + ah)
    return Fraction(area, w * h) >= Fraction(GATE["minimum_region_precision"])


def aggregate(rows):
    output = {}
    for bank in ("grid", "crema"):
        groups, counts, assessed, false, found = set(), Counter(), Counter(), Counter(), Counter()
        for row in (row for row in rows if row["bank"] == bank):
            for case in row["cases"]:
                role = case["role"]
                counts[role] += 1
                assessed[role] += case["state"] != "no_stable_face"
                if case["choice"]:
                    if role in POSITIVES and useful(case):
                        found[role] += 1
                        groups.add(row["group"])
                    else:
                        false[role] += 1
        coverage = {kind: Fraction(sum(assessed[role] for role in roles), max(1, sum(counts[role] for role in roles)))
                    for kind, roles in (("positives", POSITIVES), ("controls", CONTROLS))}
        output[bank] = {"admitted": sum(found.values()) >= GATE[bank + "_minimum"] and len(groups) >= GATE[bank + "_groups"]
                        and sum(false[role] for role in CONTROLS) <= GATE["maximum_alerts_per_bank"]
                        and false["whole_cut"] + false["whole_hold"] <= GATE["maximum_cut_or_whole_hold_alerts"]
                        and min(coverage.values()) >= Fraction(GATE["minimum_assessed_per_bank"]), "groups": len(groups),
                        "true_locations": dict(found), "assessed": dict(assessed), "cases": dict(counts), "false_or_mislocated": dict(false)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "development", "held", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        freeze(args.run)
        return
    plan = read(args.run / "freeze.json")
    require(plan["bindings"] == bindings(), "Method or source lock differs")
    if args.phase == "report":
        held = [row for row in plan["rows"] if row["split"] == "held"]
        rows = [read(args.run / f"{row['bank']}-{row['group']:02d}" / "results.json") for row in held]
        report = {"schema": "chipmunk.video-mouth-local-hold-panel/1", "freeze_sha256": digest(args.run / "freeze.json"),
                  "gate": GATE, "results": aggregate(rows), "rows": rows}
        save(args.run / "receipt.json", report)
        print(json.dumps(report["results"], sort_keys=True))
        return
    rows = [row for row in plan["rows"] if row["split"] == args.phase]
    pairs = []
    for bank in ("grid", "crema"):
        bank_rows = [row for row in rows if row["bank"] == bank]
        pairs.extend(zip(bank_rows, bank_rows[1:] + bank_rows[:1], strict=True))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda pair: score(args.run, *pair), pairs))


if __name__ == "__main__":
    main()
