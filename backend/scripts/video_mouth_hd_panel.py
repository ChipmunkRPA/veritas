"""Freeze and compare mouth-region hold evidence on licensed HD speech excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from app.core.canonical import canonical_hash
from app.detectors.image.faces import load_yunet
from app.detectors.video import mouth_local_hold as method
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.video_face_720p30_transfer import authored
from scripts.video_mouth_local_hold_panel import WEIGHTS, _patch, _records, useful
from scripts.video_reference_soundtrack_panel import digest, read, require, save

SIZE = (1280, 720)
RATE = 30
LONG = 240
NORMALIZE = "fps=30,scale=1280:720:flags=area,setsar=1"
OFFSETS = (8, 16, 24, 32, 40)
FIRST = (35, 44)
LATE = (99, 111)
TAIL = (158, 170)
CASES = {"mouth_early": (120, FIRST), "mouth_late": (LONG, LATE), "ordinary_pause": (LONG, None),
         "tandem": (LONG, None), "whole_hold": (LONG, LATE), "cut": (LONG, LATE),
         "mouth_blur": (LONG, LATE), "off_mouth_hold": (LONG, LATE), "tail_only": (LONG, TAIL)}
POSITIVE = {"mouth_early", "mouth_late"}
DEVELOPMENT = {"piotr", "priscilla"}
GATE = {"useful_share": "3/4", "minimum_late_groups": 4, "assessed_share": "4/5", "max_control_alerts": 1,
        "max_cut_whole_or_tail_alerts": 0, "minimum_fresh_held_speakers": 5}
COMMONS = {
    "diego": ("Diego speaking Portuguese, English, Spanish, French, Italian, and German", "Wikitongues, Diego Martins", "by-sa/4.0"),
    "ioanna": ("Ioanna speaking Greek", "Wikitongues", "by/3.0"),
    "james": ("James speaking Filipino, English, and Spanish", "Wikitongues, James Habitan", "by-sa/4.0"),
    "paison": ("Paison speaking Kiga and English", "Wikitongues, Chabota Kanguya", "by-sa/4.0"),
    "piotr": ("Piotr speaking Silesian and English", "Wikitongues, Piotr Usiek", "by-sa/4.0"),
    "priscilla": ("Priscilla speaking Nakanamanga", "Wikitongues", "by/3.0"),
    "regina": ("Regina speaking Spanish", "Wikitongues", "by/3.0"),
    "simon": ("Simon speaking Cumbrian", "Wikitongues", "by/3.0"),
    "jan": ("Jan speaking Brabants", "Wikitongues, Fiel Sahir", "by-sa/4.0"),
    "mckensie": ("Mckensie speaking English", "Bogreudell", "by-sa/4.0"),
    "sara": ("Sara speaking Portuguese", "Wikitongues, Guillem Belmar", "by-sa/4.0"),
}


def command(*args):
    return subprocess.run(args, timeout=35, stdin=subprocess.DEVNULL, check=True, capture_output=True).stdout


def ffmpeg(*args):
    return command("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", *map(str, args))


def pause(path, offset):
    audio = ffmpeg("-ss", offset, "-i", path, "-map", "0:a:0", "-vn", "-t", 4, "-ac", 1, "-ar", 16000, "-f", "s16le", "-")
    signal = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768
    if len(signal) < 64000:
        return None
    rms = np.sqrt((signal[:64000].reshape(-1, 320) ** 2).mean(axis=1))
    candidates = []
    for first in range(15, 175):
        last = first + 10
        near = float(min(rms[first - 15:first].max(), rms[last:last + 15].max()))
        if near >= .012:
            relative = float(np.percentile(rms[first:last], 75)) / near
            if relative <= .30:
                candidates.append((relative, first, last))
    if candidates:
        ratio, first, last = min(candidates)
        return {"sample_range": [first * 320, last * 320], "quiet_to_flank_peak": round(ratio, 6)}
    return None


def face_preflight(path, offset, detector, *, control_only):
    indices = (12, 39, 67, 95, 104, 128, 164, 198, 224)
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    frames = ffmpeg("-ss", offset, "-i", path, "-map", "0:v:0", "-an", "-t", 8, "-vf", f"{NORMALIZE},select={select}",
                    "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-threads", 1, "-f", "rawvideo", "-")
    if len(frames) != len(indices) * math.prod(SIZE) * 3:
        return False
    rgb = np.frombuffer(frames, dtype=np.uint8).reshape(len(indices), SIZE[1], SIZE[0], 3)
    geometry = [method.source_geometry(frame, detector) for frame in rgb]
    return sum(count == 1 for _, count in geometry) >= (7 if control_only else 9) and all(count <= 1 for _, count in geometry)


def freeze(run, source_dir):
    run.mkdir(mode=0o700, exist_ok=False)
    detector = load_yunet(WEIGHTS)
    rows, excluded = [], []
    for name, (title, credit, license_id) in sorted(COMMONS.items()):
        path = source_dir / f"{name}.webm"
        require(path.is_file(), f"Missing planned source {name}")
        details = json.loads(command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=avg_frame_rate,width,height", "-of", "json", str(path)))["streams"][0]
        low_rate = Fraction(details["avg_frame_rate"]) < Fraction("23.9")
        possible = [(offset, quiet) for offset in OFFSETS if (quiet := pause(path, offset)) is not None]
        selected = next(((offset, quiet) for offset, quiet in possible if face_preflight(path, offset, detector, control_only=low_rate)), None)
        page = "https://commons.wikimedia.org/wiki/File:" + quote("WIKITONGUES-_" + title.replace(" ", "_") + ".webm", safe="_")
        facts = {"name": name, "source_sha256": digest(path), "source_dimensions": [details["width"], details["height"]],
                 "source_rate": details["avg_frame_rate"], "commons": page, "credit": credit,
                 "license": f"https://creativecommons.org/licenses/{license_id}/", "control_only": low_rate}
        if selected is None:
            excluded.append(facts | {"reason": "No qualifying quiet interval and one-face geometry at the fixed source offsets"})
        else:
            offset, quiet = selected
            rows.append(facts | {"split": "development" if name in DEVELOPMENT else "held", "offset_s": offset, "natural_pause": quiet})
        print("preflight", name, selected, low_rate, flush=True)
    require(len([r for r in rows if r["split"] == "held" and not r["control_only"]]) >= GATE["minimum_fresh_held_speakers"],
            "Too few held positive speakers pass source-only preparation")
    require(any(r["split"] == "development" for r in rows), "No qualified development source")
    save(run / "freeze.json", {"schema": "chipmunk.video-mouth-hd-freeze/1", "rows": rows, "excluded": excluded, "cases": CASES, "gate": GATE,
                              "script_sha256": digest(Path(__file__)), "method_sha256": digest(Path(method.__file__)), "weights_sha256": digest(WEIGHTS)})


def prepare(run, source_dir, plan):
    for row in plan["rows"]:
        source = source_dir / (row["name"] + ".webm")
        require(digest(source) == row["source_sha256"], "Publisher source bytes differ from freeze")
        directory = run / row["name"]
        directory.mkdir(mode=0o700)
        path = directory / "base.mp4"
        ffmpeg("-ss", row["offset_s"], "-i", source, "-map", "0:v:0", "-an", "-frames:v", LONG, "-vf", NORMALIZE,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", 18, "-pix_fmt", "yuv420p", "-threads", 1, "-video_track_timescale", 30000, path)
        table = native_table(path)
        require(len(table) == LONG, "Normalized preparation is incomplete")


def native_table(path):
    table = probe_frames(path, cap=LONG, max_pixels=math.prod(SIZE), timeout_s=12)
    require(not (table.truncated or table.interpolated or table.rotation) and (table.width, table.height) == SIZE and
            all((tick - table.pts[0]) * table.tb == Fraction(i, RATE) for i, tick in enumerate(table.pts)),
            "Delivery failed native clock qualification")
    return table


def native_rgb(path, table, indices):
    indices = list(indices)
    frames = list(decode_frames(path, table, indices, timeout_s=16, max_pixels=math.prod(SIZE)))
    require(len(frames) == len(indices) and all(frame.index == i and frame.pts == table.pts[i] and frame.pts_verified
                for frame, i in zip(frames, indices, strict=True)), "Decoded native frame coverage differs")
    return [frame.rgb for frame in frames]


def score_group(run, row, donor):
    directory = run / row["name"]
    base = directory / "base.mp4"
    table = native_table(base)
    inputs = native_rgb(base, table, range(LONG))
    detector = load_yunet(WEIGHTS)
    cases = []
    roles = ("ordinary_pause", "tandem") if row["control_only"] else CASES
    for role in roles:
        count, truth = CASES[role]
        path = directory / (role + ".mp4")
        frames, box, changed = inputs, None, None
        if truth:
            start, end = truth
            if role in {"whole_hold", "cut"}:
                other = run / donor["name"] / "base.mp4"
                fill = [inputs[start - 3]] * (end - start) if role == "whole_hold" else native_rgb(other, native_table(other), range(start, end))
                frames = inputs[:start] + fill + inputs[end:]
            else:
                geo, detected = method.source_geometry(inputs[(start + end) // 2], detector)
                require(detected == 1, f"Local authored edit needs one visible mouth on {row['name']} / {role}")
                frames, box, changed = _patch(inputs, truth, geo, role)
        if role == "tandem":
            authored(path, source=directory / "ordinary_pause.mp4", crf=34)
        else:
            authored(path, pictures=frames[:count], crf=23)
        delivered = native_table(path)
        require(len(delivered) == count, "Delivered length differs from the fixed placement")
        started = time.monotonic()
        pixels = native_rgb(path, delivered, range(method.HD_COUNT))
        status, choice, windows = method.inspect_hd(pixels, detector)
        cases.append({"role": role, "authored_frames": truth, "authored_box": box, "authored_changes": changed,
                      "source_frames": len(delivered), "sha256": digest(path), "status": status, "assessed_starts": windows,
                      "choice": choice, "anchors": _records(delivered, pixels, choice), "decode_and_measure_s": time.monotonic() - started,
                      "prefix_rgb_sha256": canonical_hash([hashlib.sha256(frame.tobytes()).hexdigest() for frame in pixels]),
                      "prefix_pts_sha256": canonical_hash(list(delivered.pts[:method.HD_COUNT]))})
        del pixels
    save(directory / "results.json", {"name": row["name"], "donor": donor["name"], "base_sha256": digest(base), "cases": cases})
    print("verified", row["name"], len(cases), flush=True)


def summarize(rows):
    counts, assessed, good, extra, misplaced = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in rows:
        for case in row["cases"]:
            role = case["role"]
            counts[role] += 1
            assessed[role] += bool(case["assessed_starts"])
            if case["choice"]:
                if role in POSITIVE:
                    (good if useful(case) else misplaced)[role] += 1
                else:
                    extra[role] += 1
    positive = sum(counts[key] for key in POSITIVE)
    controls = sum(counts.values()) - positive
    measured = sum(assessed[key] for key in POSITIVE)
    admitted = (positive > 0 and Fraction(sum(good.values()), positive) >= Fraction(GATE["useful_share"])
                and good["mouth_late"] >= GATE["minimum_late_groups"]
                and Fraction(measured, positive) >= Fraction(GATE["assessed_share"])
                and Fraction(sum(assessed.values()) - measured, max(controls, 1)) >= Fraction(GATE["assessed_share"])
                and sum(extra.values()) <= GATE["max_control_alerts"] and sum(extra[k] for k in ("cut", "whole_hold", "tail_only")) == 0)
    return {"admitted": admitted, "cases": dict(counts), "assessed": dict(assessed), "useful": dict(good),
            "misplaced": dict(misplaced), "control_alerts": dict(extra)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "prepare", "development", "held", "report"))
    parser.add_argument("run", type=Path)
    parser.add_argument("sources", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase == "freeze":
        freeze(args.run, args.sources)
        return
    plan = read(args.run / "freeze.json")
    require(plan["method_sha256"] == digest(Path(method.__file__)) and plan["script_sha256"] == digest(Path(__file__)) and
            plan["weights_sha256"] == digest(WEIGHTS), "Frozen method or asset differs")
    if args.phase == "prepare":
        prepare(args.run, args.sources, plan)
    elif args.phase == "report":
        rows = [read(args.run / row["name"] / "results.json") for row in plan["rows"] if row["split"] == "held"]
        report = {"schema": "chipmunk.video-mouth-hd-panel/1", "freeze_sha256": digest(args.run / "freeze.json"), "source_plan": plan,
                  "summary": summarize(rows), "rows": rows}
        save(args.run / "receipt.json", report)
        print(json.dumps(report["summary"], sort_keys=True))
    else:
        selected = [row for row in plan["rows"] if row["split"] == args.phase]
        for row, donor in zip(selected, selected[1:] + selected[:1], strict=True):
            score_group(args.run, row, donor)


if __name__ == "__main__":
    main()
