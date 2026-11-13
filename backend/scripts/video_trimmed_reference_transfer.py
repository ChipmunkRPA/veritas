"""Evaluate alignment of a shortened lossless delivery against an earlier video."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from itertools import groupby, pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence.video_reference import _comparison_issue

ROOT = Path(__file__).resolve().parents[2]
MIN_SHARED = 16
MIN_SIDE = 3
ROLES = ("original", "same_time_neural", "same_time_resample", "delayed_pixels")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def roster(source):
    freeze, rendered, audit = (read(source / name) for name in ("freeze.json", "render.json", "audit.json"))
    if rendered["freeze_sha256"] != digest(source / "freeze.json") or audit["render_sha256"] != digest(source / "render.json"):
        raise ValueError("Mouth-redraw fixture is missing its source proof")
    targets = {(row["ordinal"], row["role"]): row for row in rendered["cases"] if row["role"] in ROLES}
    rows = []
    for index, group in enumerate(freeze["sources"]):
        ordinal = group["source"]["ordinal"]
        if any((ordinal, role) not in targets for role in ROLES):
            raise ValueError("Incomplete frozen video roles")
        rows.append({"group": index, "trim_lead": 4 + index % 3, "trim_tail": 3 + index % 4,
                     "roles": {role: {key: targets[ordinal, role][key] for key in ("token", "file_sha256")} for role in ROLES},
                     "truth": group["span"]})
    if len(rows) != 6:
        raise ValueError("Expected the complete six-source test")
    return {"schema": "chipmunk.trimmed-video-reference-freeze/1", "code_sha256": digest(Path(__file__)),
            "fixture_sha256": {name: digest(source / name) for name in ("freeze.json", "render.json", "audit.json")},
            "rows": rows, "minimum_shared_unique": MIN_SHARED, "minimum_per_flank": MIN_SIDE,
            "minimum_shared_fraction": "1/2", "maximum_different_ranges": 3}


def native(path):
    table = probe_frames(path, cap=96, max_pixels=1280 * 720, timeout_s=15)
    if table.truncated or table.interpolated or table.rotation or any(a >= b for a, b in pairwise(table.pts)):
        raise ValueError("Cannot compare an unverified native video clock")
    result = []
    for frame in decode_frames(path, table, None, timeout_s=18):
        if not frame.pts_verified or frame.index != len(result):
            raise ValueError("Decoded frame is missing its native timestamp")
        result.append(hashlib.sha256(frame.rgb).hexdigest())
    if len(result) != len(table):
        raise ValueError("Incomplete decoded pictures")
    return table, result


def align(earlier, submitted):
    """Unique unchanged moving pictures support the same offset on both flanks."""
    old_table, old = earlier
    new_table, new = submitted
    if (old_table.width, old_table.height) != (new_table.width, new_table.height) or not 32 <= len(new) < len(old):
        return {"status": "unsupported_geometry_or_duration"}
    old_count, new_count = Counter(old), Counter(new)
    positions = {digest: index for index, digest in enumerate(old) if old_count[digest] == 1}
    votes = Counter(positions[digest] - index for index, digest in enumerate(new)
                    if new_count[digest] == 1 and digest in positions)
    qualified = [(offset, count) for offset, count in votes.items()
                 if 0 <= offset <= len(old) - len(new) and count >= max(MIN_SHARED, (len(new) + 1) // 2)]
    if len(qualified) != 1:
        return {"status": "no_unique_offset"}
    offset, support = qualified[0]
    elapsed_old = [(pts - old_table.pts[offset]) * old_table.tb for pts in old_table.pts[offset:offset + len(new)]]
    elapsed_new = [(pts - new_table.pts[0]) * new_table.tb for pts in new_table.pts]
    if elapsed_old != elapsed_new:
        return {"status": "native_clocks_differ"}
    matches = [a == b for a, b in zip(old[offset:offset + len(new)], new, strict=True)]
    flank = len(new) // 4
    unique = [same and old_count[digest] == new_count[digest] == 1 for digest, same in zip(new, matches, strict=True)]
    if min(sum(unique[:flank]), sum(unique[-flank:])) < MIN_SIDE:
        return {"status": "insufficient_flank_support"}
    ranges = []
    for same, group in groupby(enumerate(matches), key=lambda item: item[1]):
        if same:
            continue
        indices = [index for index, _ in group]
        start, end = indices[0], indices[-1] + 1
        if start == 0 or end == len(new) or not unique[start - 1] or not unique[end]:
            return {"status": "difference_lacks_unchanged_flanks"}
        anchors = (start - 1, (start + end) // 2, end)
        ranges.append({"earlier_frames": [start + offset, end + offset], "submitted_frames": [start, end],
                       "earlier_seconds": [str(old_table.pts[index + offset] * old_table.tb) for index in (start, end)],
                       "submitted_seconds": [str(new_table.pts[index] * new_table.tb) for index in (start, end)],
                       "anchors": [{"earlier_frame": index + offset, "earlier_rgb_sha256": old[index + offset],
                                    "submitted_frame": index, "submitted_rgb_sha256": new[index]} for index in anchors]})
    if len(ranges) > 3:
        return {"status": "too_many_disjoint_differences"}
    return {"status": "measured", "earlier_overlap_frames": [offset, offset + len(new)],
            "submitted_overlap_frames": [0, len(new)], "shared_unique": support, "ranges": ranges}


def trim(path, target, table, begin, end, *, lossy=False, retime=False):
    base = Fraction(table.time_base)
    shift = r"+if(gte(N\,14)\,1\,0)" if retime else ""
    command = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an",
               "-vf", f"trim=start_frame={begin}:end_frame={end},settb={base},setpts=PTS-STARTPTS{shift}",
               "-fps_mode", "passthrough", "-frames:v", str(end - begin), "-enc_time_base", str(base),
               "-video_track_timescale", str(base.denominator), "-use_editlist", "0",
               "-threads", "1", "-c:v", "libx264rgb", "-crf", "23" if lossy else "0", "-preset", "ultrafast", str(target)]
    result = subprocess.run(command, capture_output=True, timeout=18)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace"))


def score(source, plan):
    rows = []
    groups = plan["rows"]
    for group in groups:
        paths = {role: source / "deliveries" / f"{asset['token']}.mov" for role, asset in group["roles"].items()}
        if any(digest(path) != group["roles"][role]["file_sha256"] for role, path in paths.items()):
            raise ValueError("Video source differs from its prior native proof")
        earlier = native(paths["original"])
        cut_begin = group["trim_lead"]
        cut_end = len(earlier[0]) - group["trim_tail"]
        with TemporaryDirectory() as directory:
            for role in (*ROLES, "wrong_source", "lossy", "retimed"):
                target = Path(directory) / f"{role}.mov"
                if role == "wrong_source":
                    other = groups[(group["group"] + 1) % len(groups)]["roles"]["original"]
                    wrong = source / "deliveries" / f"{other['token']}.mov"
                    if digest(wrong) != other["file_sha256"]:
                        raise ValueError("Wrong-source control differs from its native fixture")
                    other_table = probe_frames(wrong, cap=96, timeout_s=10)
                    trim(wrong, target, other_table, cut_begin, len(other_table) - group["trim_tail"])
                else:
                    trim(paths.get(role, paths["original"]), target, earlier[0], cut_begin, cut_end,
                         lossy=role == "lossy", retime=role == "retimed")
                later = native(target)
                result = align(earlier, later)
                if role in ROLES or role == "retimed":
                    original = earlier if role not in ROLES or role == "original" else native(paths[role])
                    if later[1] != original[1][cut_begin:cut_end]:
                        raise ValueError("Lossless trim changed delivered pixels")
                rows.append({"group": group["group"], "role": role, "file_sha256": digest(target),
                             "earlier_expected_overlap": [cut_begin, cut_end] if role != "wrong_source" else None,
                             "truth": group["truth"] if role in ROLES[1:] else None,
                             "current": _comparison_issue((earlier[0], later[0])), "result": result})
        print(f"Replayed trimmed source {group['group'] + 1}/{len(groups)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("source", type=Path)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    plan = roster(args.source)
    if args.stage == "freeze":
        args.work.mkdir(mode=0o700)
        write(args.work / "freeze.json", plan)
        return
    if read(args.work / "freeze.json") != plan:
        raise ValueError("Use the exact previously frozen sources and code")
    rows = score(args.source, plan)
    if args.stage == "run":
        write(args.work / "results.json", rows)
    elif read(args.work / "results.json") != rows:
        raise ValueError("Trimmed video native replay changed its results")


if __name__ == "__main__":
    main()
