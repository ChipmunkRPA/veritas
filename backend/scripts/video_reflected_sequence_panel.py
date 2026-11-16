"""Build and score source-matched controls for the mirrored footage check."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from app.detectors.video import picture_hold
from app.detectors.video.decode import probe_frames
from app.evidence import picture_sequence as binding
from scripts.video_picture_speed_panel import (
    HEIGHT,
    LENGTH,
    RATE,
    WIDTH,
    _input_pictures,
    _overlap,
    _sha,
    _write,
)

ENCODINGS = {"h264": ("mp4", "libx264", "24", "33"), "vp9": ("webm", "libvpx-vp9", "28", "37")}
CASES = ("later_mirrored", "earlier_mirrored", "original", "in_place_mirror", "held_mirror",
         "same_speed", "reverse", "later_mirrored_tandem", "original_tandem")
SCHEMA = binding.MOTION_REFLECTED_SCHEMA
KEY = binding.MOTION_REFLECTED_KEY
BASELINE_KEYS = ("sequence_review", binding.REVERSE_KEY, "motion_sequence_review",
                 binding.MOTION_REVERSE_KEY, binding.MOTION_RETIMED_KEY)


def mirror(pictures, *, dimensions=(WIDTH, HEIGHT)):
    output = np.empty_like(pictures)
    start = 0
    w, h = dimensions
    for width, height in ((w, h), (w // 2, h // 2), (w // 2, h // 2)):
        end = start + width * height
        plane = pictures[:, start:end].reshape(len(pictures), height, width)
        output[:, start:end] = plane[:, :, ::-1].reshape(len(pictures), -1)
        start = end
    return output


def delivery(pictures, early, late, width, role):
    output = pictures[:LENGTH].copy()
    if role in {"later_mirrored", "earlier_mirrored"}:
        source, target = (early, late) if role == "later_mirrored" else (late, early)
        output[target:target + width] = mirror(pictures[source:source + width])
        return output, [[start, start + width - 1] for start in (early, late)]
    if role == "in_place_mirror":
        output[late:late + width] = mirror(pictures[late:late + width])
    if role == "held_mirror":
        output[late:late + width] = mirror(pictures[early:early + 1])
    if role == "same_speed":
        output[late:late + width] = pictures[early:early + width]
    if role == "reverse":
        output[late:late + width] = pictures[early:early + width][::-1]
    return output, None


def _encode(destination, codec, *, pictures=None, prior=None):
    _, encoder, direct, tandem = ENCODINGS[codec]
    source = ["-i", str(prior)] if prior else ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s:v", f"{WIDTH}x{HEIGHT}",
                                               "-r", str(RATE), "-i", "pipe:0"]
    options = ["-preset", "veryfast"] if codec == "h264" else ["-b:v", "0", "-deadline", "good", "-cpu-used", "4"]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", *source, "-an", "-c:v", encoder,
               "-threads", "2", "-g", "48", "-crf", tandem if prior else direct, *options, str(destination)]
    subprocess.run(command, input=memoryview(pictures) if prior is None else None, capture_output=True, check=True, timeout=60)


def build(run, sources):
    run.mkdir()
    if len({row["sha256"] for row in sources}) != len(sources):
        raise ValueError("Duplicate source hashes")
    cases = []
    for source in sources:
        path = Path(source["path"])
        digest = _sha(path)
        if digest != source["sha256"]:
            raise ValueError(f"Source hash differs for {source['id']}")
        pictures = _input_pictures(path, source["offset_s"])
        early = 24 + int(digest[:2], 16) % 18
        late = 134 + int(digest[2:4], 16) % 18
        for codec, (extension, *_) in ENCODINGS.items():
            priors = {}
            for role in CASES:
                name = f"{source['id']}-{role}.{extension}"
                base = role.removesuffix("_tandem")
                if role == base:
                    width = (17 + int(digest[4:6], 16) % 5) if base == "later_mirrored" else (26 + int(digest[6:8], 16) % 7)
                    frames, truth = delivery(pictures, early, late, width, role)
                    _encode(run / name, codec, pictures=frames)
                else:
                    prior, truth = priors[base]
                    _encode(run / name, codec, prior=run / prior)
                priors[role] = name, truth
                cases.append({"source": source["id"], "role": role, "codec": codec, "delivery": name,
                              "sha256": _sha(run / name), "truth": truth})
        print(f"Prepared {source['id']}", flush=True)
    _write(run / "plan.json", {
        "method": {"schema": SCHEMA, "parameters": binding.motion_parameters(SCHEMA, RATE),
                   "minimum_iou": .5, "candidate_source_sha256": _sha(Path(picture_hold.picture_motion.__file__))},
        "sources": [{k: v for k, v in source.items() if k != "path"} for source in sources],
        "cases": cases,
    })


def _ranges(pair):
    return [[pair[side]["start_frame_index"], pair[side]["end_frame_index"]] for side in ("first", "second")]


def score(run, report):
    plan = json.loads((run / "plan.json").read_text())
    if plan["method"]["candidate_source_sha256"] != _sha(Path(picture_hold.picture_motion.__file__)):
        raise ValueError("The frozen candidate has changed")
    cases = []
    for row in plan["cases"]:
        path = run / row["delivery"]
        if _sha(path) != row["sha256"]:
            raise ValueError(f"Delivery hash differs for {path.name}")
        table = probe_frames(path)
        if len(table) != LENGTH or table.interpolated:
            raise ValueError(f"Unexpected presentation clock in {path.name}")
        before = time.perf_counter()
        cover = picture_hold.inspect(path, table, origin=0., sequence_review=True, retimed_review=True,
                                     motion_encoding=(row["codec"], "yuv420p"), reflected_review=True)["coverage"]
        review = cover.get(KEY)
        if review is None or review["status"] == "unavailable":
            raise ValueError(f"Unassessed reflected sequence in {path.name}")
        baseline = {key: _ranges(record["pair"]) for key in BASELINE_KEYS if (record := cover.get(key)) and record["pair"]}
        found = review["pair"]
        reflected = _ranges(found) if found else None
        truth = row["truth"]
        useful = bool(reflected and truth and all(_overlap(predicted, actual) >= plan["method"]["minimum_iou"]
                                                for predicted, actual in zip(reflected, truth, strict=True)))
        cases.append(row | {"baseline": baseline, "reflected": reflected, "useful": useful, "exact": reflected is not None and reflected == truth,
                            "time_base": table.time_base, "scan_ms": round((time.perf_counter() - before) * 1000, 2),
                            "anchors": list(binding.endpoints(found)) if found else [],
                            "supporting_edge_offsets": found["supporting_edge_offsets"] if found else []})
        print(path.name, reflected, useful, flush=True)
    _write(report, plan | {"cases": cases})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action, name in (("build", "sources"), ("score", "report")):
        command = commands.add_parser(action)
        command.add_argument("run", type=Path)
        command.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == "build":
        build(args.run, json.loads(args.sources.read_text()))
    else:
        score(args.run, args.report)


if __name__ == "__main__":
    main()
