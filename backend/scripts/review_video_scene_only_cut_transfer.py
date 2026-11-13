# SPDX-License-Identifier: Apache-2.0
"""Independently replay the ordinary-cut matrix and native evidence anchors."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-scene-only-cut-transfer/1"
ROLES = ("unchanged", "speaker_only", "scene_only", "speaker_and_scene")
POINT = 0.30
FULL_SCHEMA = "chipmunk.video-scene-only-cut-full/1"


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def totals(rows):
    by_role, pairs = {}, {}
    selected = 0
    for row in rows:
        role, span, shot, paired = (row[k] for k in ("role", "span", "shot_frames", "paired"))
        a, b = span
        result = bool(paired and paired["score"] >= POINT)
        paired_span = paired["interval"] if result else None
        required = {a - 1, (a + b) // 2, b}
        if paired_span:
            p, q = paired_span
            required.update((p - 1, (p + q) // 2, q))
        ids = [frame["frame"] for frame in row["selected"]]
        time_base = Fraction(row["time_base"])
        require(row["frame_count"] == 120 and ids == sorted(required)
                and all((f["pts"] - row["first_pts"]) * time_base == Fraction(f["frame"], 30)
                        and len(f["rgb_sha256"]) == 64 for f in row["selected"]), "Native anchors differ")
        selected += len(ids)
        by_role.setdefault(role, Counter()).update({
            "files": 1, "shot_entry_exact": a in shot, "shot_return_exact": b in shot,
            "shot_both_exact": a in shot and b in shot, "shot_any": bool(shot),
            "paired_measured": row["face_observations"] >= 2,
            "paired_reportable": result, "paired_exact": paired_span == span,
            "paired_wrong_range": bool(paired_span and paired_span != span),
        })
        key = row["pair"], row["direction"]
        siblings = pairs.setdefault(key, set())
        require(role in ROLES and role not in siblings, "Unexpected repeated matrix cell")
        siblings.add(role)
    require(len(pairs) == 8 and all(v == set(ROLES) for v in pairs.values()), "Incomplete matched bank")
    return {role: dict(count) for role, count in by_role.items()}, selected


def audit_media(path, row):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,codec_name,pix_fmt,codec_tag_string,time_base:frame=best_effort_timestamp",
        "-of", "json", str(path),
    ], timeout=20))
    stream, = data["streams"]
    pts = [int(f["best_effort_timestamp"]) for f in data["frames"]]
    require([stream[k] for k in ("width", "height", "codec_name", "pix_fmt", "codec_tag_string")] ==
            [1280, 720, "hevc", "yuv420p", "hvc1"] and stream["time_base"] == row["time_base"]
            and len(pts) == row["frame_count"] and pts[0] == row["first_pts"]
            and all((p - pts[0]) * Fraction(stream["time_base"]) == Fraction(i, 30) for i, p in enumerate(pts)),
            "Independent carrier clock or profile differs")
    selected = row["selected"]
    expression = "+".join(f"eq(n,{f['frame']})" for f in selected)
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-protocol_whitelist", "file", "-noautorotate",
        "-i", str(path), "-map", "0:v:0", "-vf", "select=" + expression.replace(",", r"\,"),
        "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ], timeout=30)
    frame_size = 1280 * 720 * 3
    require(len(raw) == len(selected) * frame_size, "Missing representative pictures")
    for i, anchor in enumerate(selected):
        require(anchor["pts"] == pts[anchor["frame"]] and anchor["rgb_sha256"] ==
                hashlib.sha256(raw[i * frame_size:(i + 1) * frame_size]).hexdigest(),
                "Representative picture bytes or timestamps differ")
    return len(pts)


def assemble(work):
    freeze, media, scored = (work / n for n in ("freeze.json", "media.json", "results.json"))
    lock, planned, observed = map(read, (freeze, media, scored))
    require(planned["freeze_sha256"] == sha(freeze) and observed["media_sha256"] == sha(media)
            and planned["raw_factorial_frames_verified"] == 3840 and len(observed["rows"]) == 32,
            "Source preparation differs")
    rows, clocks = [], 0
    for expected, row in zip(planned["rows"], observed["rows"], strict=True):
        require(all(row[k] == v for k, v in expected.items()), "Measured matrix differs")
        path = work / "media" / row["name"]
        require(sha(path) == row["sha256"], "Carrier bytes differ")
        clocks += audit_media(path, row)
        rows.append({k: v for k, v in row.items() if k not in {"sha256", "name"}})
    counts, selected = totals(rows)
    return {"schema": SCHEMA, "protocol": lock["protocol"], "freeze_sha256": sha(freeze), "media_sha256": sha(media),
            "results_sha256": sha(scored), "source_groups": len(lock["sources"]), "prior_actor_exclusions": 20,
            "raw_factorial_frames_verified": 3840, "reopened_carrier_clocks": clocks,
            "reopened_native_pictures": selected, "totals": counts, "rows": rows}


def review_full(reference, result):
    require(result["schema"] == FULL_SCHEMA and reference["media_sha256"] == result["media_sha256"]
            and result["isolated_production_full_face_pipeline"] and result["unpaid_jobs"] == 0
            and result["simulated_paid_jobs"] == len(result["rows"]) == len(reference["rows"]) == 32,
            "Full worker did not bind the same paid fixture bank")
    totals = {}
    for base, row in zip(reference["rows"], result["rows"], strict=True):
        require(all(row[k] == base[k] for k in ("pair", "direction", "role", "span"))
                and row["authored_time"] == [str(Fraction(i, 30)) for i in base["span"]],
                "Full delivery ordering or clock differs")
        coarse, brief, cut = (row[k] for k in ("coarse", "brief", "cut"))
        if coarse is None:
            route = "no_coarse"
        elif brief:
            route = "brief_precedence"
        elif not 5 <= coarse[1] - coarse[0] <= 42:
            route = "coarse_duration_outside_scope"
        else:
            route = "published" if cut else "eligible_without_published_cut"
        require(route == row["route"] and bool(cut) == (route == "published"), "Full route differs")
        if cut:
            a, b = cut
            anchors = row["anchors"]
            require([x["frame"] for x in anchors] == [a - 1, (a + b) // 2, b]
                    and all(Fraction(x["time"]) == Fraction(x["frame"], 30) and len(x["rgb_sha256"]) == 64
                            for x in anchors), "Published cut anchors differ")
            known = {x["frame"]: x["rgb_sha256"] for x in base["selected"]}
            require(all(x["frame"] not in known or x["rgb_sha256"] == known[x["frame"]] for x in anchors),
                    "The shared Full and pre-marked frames differ")
        else:
            require(row["anchors"] == [], "A nonpublished range carries cut anchors")
        counts = totals.setdefault(row["role"], Counter())
        counts.update({"files": 1, route: 1, "exact_published": cut == row["span"],
                       "shot_boundaries": len(row["shot_boundaries"]), "verified_cut_frames": len(row["anchors"])})
    return {k: dict(v) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--assemble", type=Path)
    parser.add_argument("--full", type=Path)
    args = parser.parse_args()
    if args.assemble:
        result = assemble(args.assemble)
        with args.receipt.open("x") as output:
            json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    result = read(args.receipt)
    counts, selected = totals(result["rows"])
    require(result["schema"] == SCHEMA and counts == result["totals"] and len(result["rows"]) == 32
            and selected == result["reopened_native_pictures"] and result["reopened_carrier_clocks"] == 3840
            and result["raw_factorial_frames_verified"] == 3840, "Published results differ")
    print(json.dumps(counts, sort_keys=True))
    print("verified clocks", result["reopened_carrier_clocks"], "verified representative pictures", selected)
    if args.full:
        print("Full worker", json.dumps(review_full(result, read(args.full)), sort_keys=True))


if __name__ == "__main__":
    main()
