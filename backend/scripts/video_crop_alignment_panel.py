"""Compare the fixed crop search with centered and landmark-based subpixel alignment."""
from __future__ import annotations

import argparse
import json
from contextlib import closing
from pathlib import Path
from time import monotonic

import cv2

from app.detectors.video import picture_crop, picture_crop_aligned, picture_motion
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import picture_sequence as binding
from scripts.video_cropped_reuse_panel import _correct, _sha, _write

SCHEMA = "chipmunk.video-crop-alignment-panel/1"
ENCODING = "h264", "yuv420p"
METHOD = {
    "rate": 25,
    "extent": "The first 225 native pictures at 640 × 360 or 200 at 1280 × 720.",
    "comparison": "Run the existing fixed crop search, centered subpixel crops and landmark-estimated crops independently. "
                  "Keep the existing match when it exists, then prefer the aligned result with the most matched frames. "
                  "Ties use more supporting changes, then the centered geometry.",
    "localization": "Require at least one-half intersection over union in each nonoverlapping source range. "
                    "Record maximum normalized edge error separately from temporal localization.",
}


def lock(folder, freeze):
    planfile = folder / "plan.json"
    plan = json.loads(planfile.read_text())
    if plan["candidate_sha256"] != _sha(Path(picture_crop.__file__)):
        raise ValueError("The current crop search differs from the source plan")
    _write(freeze, {"schema": SCHEMA, "method": METHOD, "input_sha256": _sha(planfile),
                    "candidate_sha256": _sha(Path(picture_crop_aligned.__file__)), "opencv": cv2.__version__})


def data(clip):
    table = probe_frames(clip)
    profile = picture_motion.profile(table, ENCODING, schema=binding.MOTION_CROP_SCHEMA)
    if not profile or profile[0] != METHOD["rate"]:
        raise ValueError(f"{clip.name} does not have the supported source clock")
    with closing(decode_frames(clip, table, range(profile[1]), timeout_s=20)) as frames:
        pictures = [picture_crop.luminance(frame.rgb) for frame in frames if frame.pts_verified]
    if len(pictures) != profile[1]:
        raise ValueError(f"{clip.name} could not be decoded with all its native clocks")
    return pictures, table


def record(match, table, unit, truth):
    if match is None:
        return None
    return match._asdict() | {
        "unit": unit, "pts": [[table.pts[i], table.pts[j]] for i, j in match.ranges],
        "useful": _correct(match.ranges, truth, .5),
    }


def select(arms):
    if arms["fixed"] is not None:
        return "fixed"
    available = [name for name in ("centered", "landmarks") if arms[name] is not None]
    if not available:
        return None
    return max(available, key=lambda name: (arms[name]["ranges"][0][1] - arms[name]["ranges"][0][0],
                                           len(arms[name]["support"])))


def score(folder, freeze, output):
    cv2.setNumThreads(1)
    expected = json.loads(freeze.read_text())
    planfile = folder / "plan.json"
    if (expected != {"schema": SCHEMA, "method": METHOD, "input_sha256": _sha(planfile),
                     "candidate_sha256": _sha(Path(picture_crop_aligned.__file__)), "opencv": cv2.__version__}):
        raise ValueError("The alignment comparison differs from the frozen configuration")
    plan = json.loads(planfile.read_text())
    if plan["candidate_sha256"] != _sha(Path(picture_crop.__file__)):
        raise ValueError("The current crop search differs from the source plan")
    rows = []
    for row in plan["cases"]:
        clip = folder / row["delivery"]
        if _sha(clip) != row["sha256"]:
            raise ValueError(f"Input differs from its frozen source bytes: {clip.name}")
        pixels, table = data(clip)
        start = monotonic()
        original = picture_crop.find(pixels, METHOD["rate"])
        fixed_seconds = monotonic() - start
        predictions = picture_crop_aligned.find(pixels)
        elapsed = monotonic() - start - fixed_seconds
        arms = {"fixed": record(original, table, 1, row["truth"])}
        arms.update({name: record(match, table, picture_crop_aligned.UNIT, row["truth"]) for name, match in predictions.items()})
        chosen = select(arms)
        rows.append(row | {"results": arms, "selected": chosen, "time_base": table.time_base,
                           "fixed_s": round(fixed_seconds, 4), "aligned_s": round(elapsed, 4)})
        print(row["source"], row["role"], chosen, {name: bool(match and match["useful"]) for name, match in arms.items()},
              round(elapsed, 3), flush=True)
    _write(output, expected | {"plan": plan | {"cases": rows}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("lock")
    prepare.add_argument("folder", type=Path)
    prepare.add_argument("freeze", type=Path)
    run = commands.add_parser("score")
    run.add_argument("folder", type=Path)
    run.add_argument("freeze", type=Path)
    run.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "lock":
        lock(args.folder, args.freeze)
    else:
        score(args.folder, args.freeze, args.output)
