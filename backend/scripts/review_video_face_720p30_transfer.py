"""Verify the published 720p30 face-range comparison and optionally replay native evidence."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_lossy_sequence_transfer import ffmpeg, read, sha_file


def agrees(pair, truth):
    if pair is None or truth is None:
        return False
    a, b = pair
    c, d = truth
    return 2 * (min(b, d) - max(a, c)) >= max(b, d) - min(a, c)


def verify_public(data):
    if data["schema"] != "chipmunk.video-face-720p30-transfer/1":
        raise ValueError("Unexpected report format")
    rows = data["rows"]
    roles = {"original", "recompressed", "light", "jump", "cut", "swap_a", "swap_b"}
    expected = {(group, role) for group in range(1, 5) for role in roles} | {(group, "camera") for group in range(5, 8)}
    if len(rows) != len(expected) or {(row["group"], row["condition"]) for row in rows} != expected:
        raise ValueError("Incomplete case roster")
    for row in rows:
        if row["truth"] != {"swap_a": [72, 78], "swap_b": [87, 97]}.get(row["condition"]):
            raise ValueError("Authored range changed")
        for arm in ("baseline", "sparse", "dense"):
            if row[arm] is not None and not 0 <= row[arm][0] < row[arm][1] < 120:
                raise ValueError("Invalid half-open frame range")
        if row["dense"] and (row["sparse"] is not None or row["sparse_status"] != "no_distinct_range"):
            raise ValueError("Invalid eligibility for dense fallback")
    positives = [row for row in rows if row["truth"]]
    controls = [row for row in rows if not row["truth"]]
    baseline = {i for i, row in enumerate(positives) if agrees(row["baseline"], row["truth"])}
    reports = {}
    for arm in ("sparse", "dense"):
        def shown(row, arm=arm):
            return row["sparse"] or (row["dense"] if arm == "dense" else None)

        new = {i for i, row in enumerate(positives) if agrees(shown(row), row["truth"])} - baseline
        measured_swap, measured_ordinary = (sum(r["sparse_status"] in {"no_distinct_range", "localized"} for r in family)
                                            for family in (positives, [r for r in controls if r["condition"] != "cut"]))
        added_control = Counter("cut" if r["condition"] == "cut" else "ordinary" for r in controls
                                if shown(r) is not None and r["baseline"] is None)
        groups = len({positives[i]["group"] for i in new})
        arm_result = {"baseline_useful": len(baseline), "combined_useful": len(baseline | new), "new_groups": groups,
                      "positives": 8, "ordinary": 19, "cuts": 4, "new_ordinary": added_control["ordinary"],
                      "new_cut": added_control["cut"], "sparse_measured_swap": measured_swap,
                      "sparse_measured_ordinary": measured_ordinary,
                      "baseline_control_flags": dict(Counter(r["condition"] for r in controls if r["baseline"] is not None))}
        arm_result["passes"] = (len(new) >= 3 and groups >= 2 and added_control["ordinary"] <= 1 and not added_control["cut"]
                                and Fraction(measured_swap, 8) >= Fraction(4, 5) and Fraction(measured_ordinary, 19) >= Fraction(4, 5))
        if arm_result != data["results"][arm]:
            raise ValueError(f"Public arithmetic differs for {arm}")
        reports[arm] = dict(Counter(row["condition"] for row in controls if shown(row) is not None))
    print(f"Arithmetic verified on 31 cases. Added control ranges by condition: {reports}.", flush=True)
    return reports


def verify_native(study, data):
    from app.detectors.video.decode import FrameTable, decode_frames
    from app.reports.verify_video_source import _verify_aligned_face, _verify_guarded_scene

    freeze, media, outcomes = (read(study / f"{name}.json") for name in ("freeze", "media", "private"))
    runner = Path(__file__).with_name("video_face_720p30_transfer.py")
    if freeze["runner_sha256"] != sha_file(runner) or media["freeze_sha256"] != sha_file(study / "freeze.json"):
        raise ValueError("The implementation or source freeze differs")
    count, inputs, aligned = 0, 0, 0
    for row, outcome, decision in zip(media["rows"], outcomes, data["rows"], strict=True):
        path = study / row["file"]
        if row["file"] != outcome["file"] or row["sha256"] != sha_file(path):
            raise ValueError("Changed native delivered file")
        clock = row["clock"]["time_base"]
        info = ffmpeg(path, ["-enc_time_base", clock, "-c:v", "rawvideo", "-f", "framehash", "-hash", "sha256"],
                      text=True).splitlines()
        if f"#tb 0: {clock}" not in info:
            raise ValueError("Independent decoder returned a different clock")
        table = [line.split(",") for line in info if not line.startswith("#")]
        ticks = [int(fields[2]) for fields in table]
        digests = [fields[5].strip() for fields in table]
        if (ticks != row["clock"]["pts"] or digests != row["rgb_sha256"] or len(ticks) != 120
                or any(int(fields[4]) != 1280 * 720 * 3 for fields in table)):
            raise ValueError("Independent frame hash and clock replay disagrees")
        count += len(ticks)
        baseline = outcome["baseline"]["coarse"]
        if decision["baseline"] != ([ticks.index(baseline[key]) for key in ("start_pts", "end_pts")] if baseline else None):
            raise ValueError("Public baseline range differs from actual Full measurement")
        for arm in ("sparse", "dense"):
            record = outcome["additions"][arm]
            scope = outcome["additions"][f"{arm}_coverage"]
            if decision[f"{arm}_status"] != scope["status"] or decision[arm] != (
                    [record[key] for key in ("start_frame_index", "end_frame_index")] if record else None):
                raise ValueError("Public candidate differs from its measured response")
            if record is None:
                continue
            if not (record["response"] > .10 and all(
                    record["anchors"][key] in record["support_indices"][key] for key in ("before", "during", "after"))
                    and (arm != "dense" or min(record["scene_lower_quartiles"]) >= .60)
                    and not agrees(decision[arm], decision["baseline"])):
                raise ValueError("Candidate fails the fixed response, support or scene gate")
            for frame in record["frame_inputs"] + record["scene_frame_inputs"]:
                i = frame["frame_index"]
                if frame["pts"] != ticks[i] or frame["decoded_rgb_sha256"] != digests[i]:
                    raise ValueError("Candidate evidence uses another native frame")
                inputs += 1
            indices = set(record["anchors"].values()) | {frame["frame_index"] for frame in record["scene_frame_inputs"]}
            frames = {frame.index: frame.rgb for frame in decode_frames(path, FrameTable.from_json(row["clock"]), sorted(indices))
                      if frame.pts_verified}
            if frames.keys() != indices:
                raise ValueError("Could not decode candidate evidence anchors")
            for frame in record["frame_inputs"]:
                if frame["frame_index"] in record["anchors"].values():
                    _verify_aligned_face(frames[frame["frame_index"]], frame)
                    aligned += 1
            if arm == "dense":
                _verify_guarded_scene(record, frames)
    print(f"Native replay verified {count} decoded pictures, {inputs} candidate frame bindings and {aligned} aligned face crops.", flush=True)
    return {"deliveries": len(media["rows"]), "native_frames": count, "candidate_frame_bindings": inputs,
            "aligned_anchor_crops": aligned}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = read(args.panel)
    result["additional_control_ranges"] = verify_public(result)
    if args.study:
        result["native_replay"] = verify_native(args.study, result)
    if args.output:
        with args.output.open("x") as output:
            output.write(json.dumps(result, separators=(",", ":")) + "\n")
