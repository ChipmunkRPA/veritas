"""Independently replay the 360p motion sequence trial and verify its public counts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

from scripts.review_video_lossy_sequence_transfer import agreement, approximate, ffmpeg, read, sha_file

SCHEMA = "chipmunk.video-motion-sequence-360p-transfer/1"
CODECS = ("h264", "vp9")
CAMERA_CASES = ("copy", "unchanged", "cut", "hold", "reverse", "spacing")
ARMS = ("exact", "motion", "combined")


def verify_public(data):
    if (data["schema"], data["geometry"], data["point"]) != (SCHEMA, [640, 360, 25, 225], [1.6, .4, .65]):
        raise ValueError("Unexpected source or method scope")
    rows = data["rows"]
    case_plan = {"camera": CAMERA_CASES, "commons": ("copy", "published")}
    source_groups = {role: {r["source"] for r in rows if r["cohort"] == role} for role in case_plan}
    if len(source_groups["camera"]) != 6 or len(source_groups["commons"]) != 8:
        raise ValueError("Incomplete source panel")
    if source_groups["camera"] & source_groups["commons"]:
        raise ValueError("Source identity overlap")
    plan = {(role, group, codec, kind) for role, kinds in case_plan.items() for group in source_groups[role]
            for codec in CODECS for kind in kinds}
    if len(rows) != len(plan) or {(r["cohort"], r["source"], r["codec"], r["kind"]) for r in rows} != plan:
        raise ValueError("Incomplete matched comparisons")
    for row in rows:
        for key in (*ARMS, "truth"):
            pair = row[key]
            if pair is not None:
                a, b, c, d = pair
                if not (0 <= a < b < c < d < 225 and b - a == d - c):
                    raise ValueError("Invalid frame interval pair")
        if (row["truth"] is None) == (row["kind"] == "copy") or row["verified_cfr"] == (row["kind"] == "spacing"):
            raise ValueError("A public row mislabels its source conditions")
        if row["combined"] != (row["exact"] or row["motion"]) or (not row["verified_cfr"] and row["motion"]):
            raise ValueError("A public row changes the fallback scope")
    results = {}
    for codec in CODECS:
        positives = [r for r in rows if r["codec"] == codec and r["kind"] == "copy"]
        controls = [r for r in rows if r["codec"] == codec and r["kind"] != "copy"]
        counts = {role: Counter(arm for row in positives for arm in ARMS
                                if row["cohort"] == role and agreement(row[arm], row["truth"])) for role in case_plan}
        added = [r for r in controls if r["exact"] is None and r["combined"] is not None]
        results[codec] = {
            "copy_cases": {role: sum(r["cohort"] == role for r in positives) for role in case_plan},
            "useful": {role: {arm: counts[role][arm] for arm in ARMS} for role in case_plan},
            "flags_by_kind": {arm: dict(Counter(r["kind"] for r in controls if r[arm] is not None)) for arm in ARMS},
            "controls": len(controls), "additional_controls": len(added),
            "new_hold_or_spacing": sum(r["kind"] in ("hold", "spacing") for r in added),
            "lost_exact": sum(agreement(r["exact"], r["truth"]) and not agreement(r["combined"], r["truth"]) for r in positives),
        }
    passes = all(s["useful"]["camera"]["combined"] >= 4 and s["useful"]["commons"]["combined"] >= 5
                 and s["additional_controls"] <= 1 and s["new_hold_or_spacing"] == 0 and s["lost_exact"] == 0
                 for s in results.values())
    if data["metrics"] != results or data["passes"] != passes:
        raise ValueError("The published summary differs from the case rows")
    print("Public arithmetic verified across 104 held deliveries.")


def replay_native(study, row):
    import numpy as np

    from scripts.video_exact_sequence_transfer import oracle

    path = study / f"{row['source']}-{row['codec']}-{row['kind']}.mkv"
    if row["sha256"] != sha_file(path):
        raise ValueError("Committed delivery changed")
    info = ffmpeg(path, ["-enc_time_base", row["time_base"], "-c:v", "rawvideo", "-f", "framehash", "-hash", "sha256"],
                  text=True).splitlines()
    table = [line.split(",") for line in info if not line.startswith("#")]
    pts = [int(fields[2]) for fields in table]
    digests = [fields[5].strip() for fields in table]
    if ("#tb 0: 1/1000" not in info or pts != row["pts"] or digests != row["rgb_sha256"]
            or any(int(fields[4]) != 640 * 360 * 3 for fields in table)):
        raise ValueError("Independent decoder disagrees with delivered frame provenance")
    cfr = all(b - a == 40 for a, b in pairwise(pts))
    if cfr != row["verified_cfr"] or oracle(pts, digests, Fraction(1, 1000)) != row["exact"]:
        raise ValueError("Independent clock or exact-search outcome differs")
    if cfr:
        raw = ffmpeg(path, ["-c:v", "rawvideo", "-f", "rawvideo"])
        raster = np.frombuffer(raw, dtype="u1").reshape(225, 30, 12, 40, 16, 3)
        sums = raster.sum(axis=(2, 4)).astype(np.float64)
        luma = (77 * sums[..., 0] + 150 * sums[..., 1] + 29 * sums[..., 2]) / (192 * 256)
        motion = approximate(pts, luma.reshape(225, -1))["motion"]
        if motion != row["motion"]:
            raise ValueError(f"Separate comparison disagrees on {path.name}")
    return len(pts)


def replay(study, artifact):
    freeze, results = read(study / "freeze.json"), read(study / "native.json")
    if freeze["runner_sha256"] != sha_file(Path(__file__).with_name("video_motion_sequence_360p_transfer.py")):
        raise ValueError("Scoring runner differs from its original freeze")
    native_frames = 0
    for i, row in enumerate(results["rows"], 1):
        native_frames += replay_native(study, row)
        if i % 12 == 0:
            print(f"Verified {i} deliveries and {native_frames} pictures.", flush=True)
    keys = ("cohort", "source", "codec", "kind", "truth", "verified_cfr", *ARMS)
    data = {"schema": SCHEMA, **{key: freeze[key] for key in ("geometry", "point", "gate", "runner_sha256")},
            "rows": [{key: r[key] for key in keys} for r in results["rows"]],
            "metrics": results["metrics"], "passes": results["passes"],
            "verification": {"deliveries": len(results["rows"]), "native_frames": native_frames, "separate_native_replay": True}}
    verify_public(data)
    with artifact.open("x") as output:
        output.write(json.dumps(data, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        replay(args.replay, args.artifact)
    else:
        verify_public(read(args.artifact))
