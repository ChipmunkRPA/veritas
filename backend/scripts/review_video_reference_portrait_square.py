# SPDX-License-Identifier: Apache-2.0
"""Recount the frozen portrait/square comparison without importing the app.

The default uses Python's standard library. Optional --deliveries also needs
ffmpeg/ffprobe and replays picture/crop commitments from the retained MP4s.
Neither mode reconstructs the authored, pre-encoding spatial truth mask.
"""

import argparse
import gzip
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-reference-portrait-square/1"
PROFILES = {"portrait": [360, 640], "square": [480, 480]}
KINDS = ("large_tone", "small_tone", "texture_copy", "identical", "recompressed", "global_tone", "wrong")
EDITED = set(KINDS[:3])
ORDINARY = set(KINDS[3:6])
POINT = 1.5712549522005277
REJECTED = {"unverified_photometric_support", "insufficient_shared_picture_content"}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def useful(row):
    return (row["role"] in EDITED and row["status"] == "local_difference"
            and row["time_iou"] >= .5 and row["pixel_iou"] >= .1)


def private_path(value):
    if isinstance(value, str):
        return value.startswith(("/Users/", "/private/", "/Volumes/", "file://"))
    if isinstance(value, dict):
        return any(private_path(k) or private_path(v) for k, v in value.items())
    if isinstance(value, list):
        return any(map(private_path, value))
    return False


def summary(study):
    result = {}
    for profile in PROFILES:
        rows = [r for r in study["rows"] if r["profile"] == profile]
        edits = [r for r in rows if r["role"] in EDITED]
        ordinary = [r for r in rows if r["role"] in ORDINARY]
        wrong = [r for r in rows if r["role"] == "wrong"]
        counts = {
            "pairs": len(rows), "source_groups": len({r["group"] for r in rows}),
            "baseline_local_assessed": sum(r["baseline_local_assessed"] for r in rows),
            "candidate_outcomes": dict(sorted(Counter(r["status"] for r in rows).items())),
            "edited_pairs": len(edits),
            "edited_pairs_flagged": sum(r["status"] == "local_difference" for r in edits),
            "edits_located": sum(map(useful, edits)),
            "edits_not_located": sum(not useful(r) for r in edits),
            "located_by_role": {kind: sum(useful(r) for r in edits if r["role"] == kind) for kind in KINDS[:3]},
            "ordinary_pairs": len(ordinary),
            "ordinary_flags": sum(r["status"] == "local_difference" for r in ordinary),
            "wrong_source_pairs": len(wrong),
            "wrong_source_admissions": sum(r["status"] not in REJECTED for r in wrong),
            "flagged_pairs_with_six_picture_and_six_crop_commitments": sum(r["status"] == "local_difference" for r in rows),
        }
        gates = study["plan"]["gates_per_profile"]
        counts["frozen_gate_passed"] = (
            counts["edits_located"] >= gates["minimum_edit_locations"]
            and min(counts["located_by_role"].values()) >= gates["minimum_each_edit_role"]
            and counts["ordinary_flags"] <= gates["maximum_ordinary_flags"]
            and counts["wrong_source_admissions"] == gates["wrong_sources_admitted"]
        )
        result[profile] = counts
    flags = sum(r["status"] == "local_difference" for r in study["rows"])
    return {"profiles": result, "pairs": len(study["rows"]), "independent_source_actor_groups": len(study["sources"]),
            "profile_copies_share_source_groups": True, "flagged_pairs": flags,
            "full_picture_commitments": 6 * flags, "crop_commitments": 6 * flags,
            "every_frozen_gate_passed": all(r["frozen_gate_passed"] for r in result.values())}


def review(study):
    require(not private_path(study), "Portable receipt contains a local path")
    require(study["schema"] == SCHEMA and study["plan"]["profiles"] == PROFILES, "Unexpected profile receipt")
    require(study["plan"]["rate"] == 30 and tuple(study["plan"]["kinds"]) == KINDS, "Frozen schedule differs")
    require(study["plan"]["review_point"] == POINT, "Inherited point differs")
    require(study["plan"]["event_match"] == {"minimum_tiou": .5, "minimum_middle_picture_iou": .1}, "Matching differs")
    require(study["plan"]["gates_per_profile"] == {
        "minimum_edit_locations": 16, "minimum_each_edit_role": 3, "maximum_ordinary_flags": 2,
        "wrong_sources_admitted": 0, "every_flag_replay_verified": True}, "Frozen gates differ")
    sources = {r["group"]: r for r in study["sources"]}
    require(len(sources) == len(study["sources"]) == 8, "Expected eight source groups")
    require(len({r["publisher_leaf"][:4] for r in sources.values()}) == 8, "Source actors repeat")
    require(len({r["sha256"] for r in sources.values()}) == 8, "Source bytes repeat")
    for i, (group, source) in enumerate(sources.items(), 1):
        require(group == f"camera-{i:02d}" and sha(source["sha256"]), "Source identity differs")
        require(re.fullmatch(r"[0-9]{4}_[A-Z]{3}_[A-Z]{3}_[A-Z]{2}\.flv", source["publisher_leaf"]), "Publisher identity differs")
        require(source["source_url"] == "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/1658cd342dff90010aa843eaeebd53610a08b1dc/VideoFlash/"+source["publisher_leaf"], "Publisher source link differs")
        require(65 <= source["frames"] <= 91 and source["frames"] <= source["native_length_available"], "Source extent differs")
    require(len(study["rows"]) == 112 and {(r["profile"], r["group"], r["role"]) for r in study["rows"]} == {
        (p, g, k) for p in PROFILES for g in sources for k in KINDS}, "Frozen 112-pair roster differs")
    require(digest(canonical(study["rows"])) == study["rows_sha256"], "Portable rows changed")
    earlier = {}
    for row in study["rows"]:
        source = sources[row["group"]]
        number = list(sources).index(row["group"])
        require(row["frames"] == source["frames"] and row["picture_clock"] == "1/30000", "Picture clock differs")
        require(row["baseline_local_assessed"] is False, "Historical baseline was outside local-area scope")
        require(len(row["input_sha256"]) == 2 and all(map(sha, row["input_sha256"])) and sha(row["frame_pairs_sha256"]), "Input commitments incomplete")
        key = row["profile"], row["group"]
        require(row["filename_pair"] == [f'{row["profile"]}-{row["group"]}-earlier.mp4',
                f'{row["profile"]}-{row["group"]}-{row["role"]}.mp4'], "Relative fixture names differ")
        earlier.setdefault(key, row["input_sha256"][0])
        require(earlier[key] == row["input_sha256"][0], "Earlier copy changes within a group/profile")
        require((row["input_sha256"][0] == row["input_sha256"][1]) == (row["role"] == "identical"), "Identical-byte control differs")
        if row["role"] == "wrong":
            alternatives = [g for g, s in sources.items() if g != row["group"] and s["native_length_available"] >= row["frames"]]
            require(row["comparison_source_group"] == alternatives[number % len(alternatives)], "Wrong-source identity differs")
        else:
            require(row["comparison_source_group"] == row["group"], "Matched-source identity differs")
        target = [13 + 2 * (number % 5), 13 + 2 * (number % 5) + (12, 16, 20)[number % 3]] if row["role"] in EDITED else None
        require(row["target"] == target, "Authored temporal boundary differs")
        if row["status"] != "local_difference":
            require(row["status"] in {"no_local_range", *REJECTED}, "Unexpected outcome")
            require(not any(k in row for k in ("range", "box", "score", "anchors", "time_iou", "pixel_iou")), "Unflagged pair carries a finding")
            continue
        a, b = row["range"]
        require(type(a) is int and type(b) is int and b-a in (8, 12, 16, 20) and 5 <= a < b <= row["frames"]-5, "Invalid measured range")
        require(finite(row["score"]) and row["score"] > POINT, "Below-point local finding")
        width, height = PROFILES[row["profile"]]
        x, y, w, h = (row["box"][k] for k in ("x", "y", "w", "h"))
        require(all(type(v) is int for v in (x, y, w, h)) and min(x, y) >= 0 and min(w, h) > 0 and x+w <= width and y+h <= height, "Finding box outside picture")
        require(set(row["anchors"]) == {"before", "during", "after"}, "Three source-paired anchors required")
        for role, frame in (("before", a-1), ("during", (a+b)//2), ("after", b)):
            anchor = row["anchors"][role]
            require(anchor["frame"] == frame and anchor["source_pts"] == [1000*frame]*2, "Anchor timestamps differ")
            require(all(len(anchor[k]) == 2 and all(map(sha, anchor[k])) for k in ("source_rgb", "crop_sha256")), "Six picture/crop commitments incomplete")
        for key in ("time_iou", "pixel_iou"):
            require(finite(row[key]) and 0 <= row[key] <= 1, "Invalid overlap measurement")
        if target:
            intersection = max(0, min(b, target[1])-max(a, target[0]))
            expected = intersection / (max(b, target[1])-min(a, target[0])) if intersection else 0
            require(abs(row["time_iou"]-expected) <= 1e-12, "Temporal IoU differs")
        else:
            require(row["time_iou"] == row["pixel_iou"] == 0, "Ordinary finding carries edit overlap")
    result = summary(study)
    require(result == study["results"], "Published totals differ from frozen rows")
    return result


def replay_pictures(study, directory):
    """Verify retained MP4s and six anchors; this does not recreate the truth mask."""
    checks = pictures = crops = 0
    verified_files = set()
    for row in study["rows"]:
        names = [f'{row["profile"]}-{row["group"]}-earlier.mp4', f'{row["profile"]}-{row["group"]}-{row["role"]}.mp4']
        paths = [directory / name for name in names]
        for path, expected in zip(paths, row["input_sha256"]):
            if path not in verified_files:
                require(digest(path.read_bytes()) == expected, "Retained input bytes differ")
                verified_files.add(path)
        if row["status"] != "local_difference":
            continue
        width, height = PROFILES[row["profile"]]
        stride, frame_bytes = width*3, width*height*3
        anchors = sorted(row["anchors"].values(), key=lambda r: r["frame"])
        select = "+".join(f'eq(n\\,{r["frame"]})' for r in anchors)
        for side, path in enumerate(paths):
            probe = subprocess.run(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,time_base,codec_name,pix_fmt:frame=pts", "-of", "json", "file:"+str(path.resolve())],
                capture_output=True, check=True, timeout=20)
            metadata = json.loads(probe.stdout)
            stream, = metadata["streams"]
            require((stream["width"], stream["height"], stream["codec_name"], stream["pix_fmt"]) == (width, height, "h264", "yuv420p"), "Retained carrier differs")
            require(Fraction(stream["time_base"]) == Fraction(1, 30000) and [r["pts"] for r in metadata["frames"]] == list(range(0, row["frames"]*1000, 1000)), "Retained source clock differs")
            decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-copyts", "-i", "file:"+str(path.resolve()),
                "-vf", "select="+select, "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"],
                capture_output=True, check=True, timeout=20).stdout
            require(len(decoded) == 3*frame_bytes, "Retained anchor decode differs")
            x, y, w, h = (row["box"][k] for k in ("x", "y", "w", "h"))
            for index, anchor in enumerate(anchors):
                frame = decoded[index*frame_bytes:(index+1)*frame_bytes]
                crop = b"".join(frame[r*stride+x*3:r*stride+(x+w)*3] for r in range(y, y+h))
                require(digest(frame) == anchor["source_rgb"][side] and digest(crop) == anchor["crop_sha256"][side], "Retained RGB/crop commitment differs")
                pictures += 1
                crops += 1
        checks += 1
    return {"input_files_reverified": len(verified_files), "flagged_pairs_replayed": checks,
            "native_pictures_reverified": pictures, "native_crops_reverified": crops,
            "authored_spatial_mask_replayed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="?", default=Path(__file__).resolve().parents[2]/"docs/evaluation-panels/video-reference-portrait-square-2026-11-18.json.gz")
    parser.add_argument("--deliveries", type=Path, help="Optional retained MP4 directory; needs ffmpeg/ffprobe")
    args = parser.parse_args()
    raw = args.receipt.read_bytes()
    study = json.loads(gzip.decompress(raw) if args.receipt.suffix == ".gz" else raw)
    result = review(study)
    if args.deliveries:
        result["native_picture_replay"] = replay_pictures(study, args.deliveries)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
