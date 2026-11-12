# SPDX-License-Identifier: Apache-2.0
"""Check frozen two-range decisions and reopen native source pictures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import video_face_two_returns as study
from scripts.video_common_sources_v89 import read, require, sha, write


def overlap(selected, truth):
    a, b = selected
    x, y = truth
    return 2 * (min(b, y) - max(a, x)) >= max(b, y) - min(a, x)


def evaluate(row, media):
    pts = media["clock"]["pts"]
    before = [(pts.index(record["start_pts"]), pts.index(record["end_pts"])) for record in
              (row["before"]["coarse"], row["before"]["boundary_refinement"], row["before"]["short_range"]) if record]
    extra = row["additional"]["candidate"]
    truth = row["truth"]
    initial = [span for span in truth if any(overlap(first, span) for first in before)]
    useful = [span for span in truth if span not in initial and extra and overlap(extra["frames"], span)]
    return before, initial, useful


def original_pixels(path, media, candidate, target):
    from PIL import Image

    from app.detectors.video import face_return
    from app.detectors.video.decode import decode_frames, probe_frames
    from app.evidence.face_interval_binding import face_context_box

    table = probe_frames(path, timeout_s=20)
    require(sha(path) == media["sha256"] and table.to_json() == media["clock"]
            and (table.width, table.height) == (640, 360), "Native original geometry or file hash differs")
    codec = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                     "stream=codec_name,pix_fmt", "-of", "csv=p=0", str(path)], timeout=10, text=True).strip()
    require(codec == "h264,yuv420p", "Original codec is outside the measured scope")
    anchors = {}
    if candidate:
        start, end = candidate["frames"]
        left, core, right = candidate["support"]
        require(0 < start < end < len(table) and 0 < (end-start)*Fraction(1,25) < Fraction(1,2)
                and len(left) >= 3 and len(core) >= 3 and len(right) >= 3
                and max(left) < start <= min(core) <= max(core) < end <= min(right)
                and all(i % 2 == 0 for i in [start, end, *left, *core, *right])
                and list(map(Fraction, candidate["clock_s"])) == [table.pts[i]*table.tb for i in (start,end)]
                and min(candidate["scene_correlations"]) >= face_return.SCENE_CORRELATION_POINT
                and min(candidate["scene_lower_quartiles"]) >= study.SPEC["minimum_scene_lower_quartile"]
                and float.fromhex(candidate["response_hex"]) > face_return.POINT, "Selected source evidence differs")
        for role, index in zip(("before", "during", "after"), (left[-1], core[len(core)//2], right[0]), strict=True):
            item = candidate["anchors"][role]
            require(item["index"] == index, "An evidence crop has the wrong temporal role")
            anchors[index] = role, item
    seen, crops = 0, {}
    for frame in decode_frames(path, table, range(120), timeout_s=40):
        digest = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        require(frame.pts_verified and frame.index == seen and frame.pts == table.pts[seen]
                and (frame.pts-table.pts[0])*table.tb == Fraction(seen,25) and digest == media["rgb_sha256"][seen],
                "Reopened source RGB bytes or presentation time differ")
        if seen in anchors:
            role, item = anchors[seen]
            require(item["pts"] == frame.pts and item["rgb_sha256"] == digest, "Evidence anchor differs from its source")
            native_box = dict(zip(("x", "y", "w", "h"), item["box"], strict=True))
            box = face_context_box(native_box, {"width": table.width, "height": table.height})
            x, y, w, h = (box[key] for key in ("x", "y", "w", "h"))
            require(0 <= x < x+w <= table.width and 0 <= y < y+h <= table.height,
                    "Evidence crop is outside the original picture")
            crop = frame.rgb[y:y+h, x:x+w]
            crops[role] = {"frame_index": seen, "source_s": str(frame.pts*table.tb), "frame_sha256": digest,
                           "crop_xywh": [x,y,w,h], "crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest()}
            if target is not None:
                Image.fromarray(frame.rgb).save(target/f"{role}-picture.png")
                Image.fromarray(crop).save(target/f"{role}-crop.png")
        seen += 1
    require(seen == 120 and len(crops) == len(anchors), "Native picture or crop count differs")
    return seen, crops


def replay_authoring(folder, frozen, media):
    from scripts import video_face_six_picture_transfer as panel

    reviewed = 0
    with TemporaryDirectory(prefix="authoring-replay-", dir=folder) as temp:
        for group, pin in enumerate(frozen["sources"], 1):
            host = study.source(pin, "original", range(120))
            first = study.source(pin, "swap_a", study.FIRST)
            second = study.source(pin, "swap_b", study.SECOND)
            guest = study.source(frozen["sources"][(group-1) ^ 1], "original", tuple(study.FIRST)+tuple(study.SECOND))
            for condition, replacement in (("dual_swap", {**first, **second}), ("dual_picture_cuts", guest)):
                receipt = media[group, condition]
                expected = {str(i): hashlib.sha256(pixels.tobytes()).hexdigest() for i, pixels in replacement.items()}
                require(expected == receipt["authored_rgb_sha256"] and all(hashlib.sha256(host[i].tobytes()).hexdigest() != digest
                        for i, digest in ((int(i), digest) for i, digest in expected.items())), "Authored picture truth differs")
                output = Path(temp) / receipt["file"]
                panel.encoded.encode(output, frames=(replacement.get(i, host[i]) for i in range(120)), crf=23)
                require(sha(output) == receipt["sha256"], "Independent authoring replay differs from the submitted video")
                output.unlink()
                reviewed += 1
            print("replayed authoring group", group, flush=True)
    return reviewed


def audit(folder):
    import cv2

    cv2.setNumThreads(1)
    frozen = read(folder / "freeze.json")
    manifest, aggregate, runtime = (read(folder / name) for name in ("media.json", "aggregate.json", "runtime.json"))
    require(frozen == study.plan() and manifest["freeze_sha256"] == aggregate["freeze_sha256"] == sha(folder/"freeze.json")
            and runtime["held_sha256"] == sha(folder/"held.jsonl"), "Review inputs differ from frozen measurements")
    source_map = {(row["group"], row["condition"]): row for row in manifest["rows"]}
    matched, examples, count, crops = set(), {}, 0, 0
    dual_completed = Counter()
    results = {role: Counter() for role in study.ROLES}
    rows = [json.loads(line) for line in (folder / "held.jsonl").read_text().splitlines()]
    for row in rows:
        role = row["condition"]
        key = row["group"], role
        media = source_map[key]
        require(key not in matched and all(row[name] == media[name] for name in ("file", "sha256", "truth")), "Source roster differs")
        matched.add(key)
        before, found, novel = evaluate(row, media)
        result = row["additional"]
        candidate = result["candidate"]
        if candidate:
            start, end = candidate["frames"]
            require(result["status"] == "measured" and result["excluded_frames"] and before
                    and all(end+2 <= a or b+2 <= start for a,b in result["excluded_frames"]),
                    "Secondary interval is not disjoint from the earlier source ranges")
        if role == "dual_swap":
            dual_completed.update(before=len(found)==2, after=len(found)+len(novel)==2)
        results[role].update(files=1, truth_ranges=len(row["truth"]), before_useful=len(found), after_useful=len(found)+len(novel),
                             before_files_flagged=bool(before), after_files_flagged=bool(before or candidate),
                             second_measured=result["status"]=="measured", added_wrong=bool(candidate and not novel))
        target = None
        if novel and role not in examples:
            target = folder / f"private-example-{role}"
            target.mkdir(mode=0o700)
        checked, source_anchors = original_pixels(folder/"media"/media["file"], media, candidate, target)
        count += checked
        crops += len(source_anchors)
        if target is not None:
            write(target / "source-anchors.json", source_anchors)
            examples[role] = {"observed_s": candidate["clock_s"], "truth_s": [str(Fraction(i,25)) for i in novel[0]],
                              "anchors_s": [row["source_s"] for row in source_anchors.values()]}
    require(matched == set(source_map) and len(matched) == 64 and count == 7680 and results == aggregate["strata"],
            "Independent outcomes or source count differ")
    authored = replay_authoring(folder, frozen, source_map)
    require(authored == 16, "Required matched dual and cut authoring could not be reproduced")
    receipt = {"status": "verified", "video_files": len(rows), "exact_native_pictures": count,
               "exact_source_crops": crops, "exact_encoded_authoring_replays": authored, "decision_rows": len(rows),
               "dual_swap_both_localized": dual_completed, "examples": examples,
               "aggregate_sha256": sha(folder/"aggregate.json")}
    write(folder / "independent-audit.json", receipt)
    print(receipt, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder)


if __name__ == "__main__":
    main()
