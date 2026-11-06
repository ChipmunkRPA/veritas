"""Replay exact pixels, edit masks, clocks and soundtrack correspondence, no detector."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.detectors.audio.decode import decode_audio
from app.detectors.video.decode import decode_frames, probe_frames
from scripts.liveportrait_mouth_benchmark import digest


def audit(manifest_path, output):
    m = json.loads(manifest_path.read_text())
    groups = defaultdict(list)
    for row in m["items"]:
        groups[row["group"]].append(row)
    records = []
    for cases in groups.values():
        original = next(r for r in cases if r["condition"] == "original")
        path = manifest_path.parent / original["path"]
        table = probe_frames(path)
        baseline = [f.rgb for f in decode_frames(path, table, range(len(table)))]
        audio = decode_audio(path, source_sha256=original["sha256"]).pcm
        for case in cases:
            result = {
                "id": case["id"],
                "group": case["group"],
                "condition": case["condition"],
                "status": "failed",
            }
            try:
                if case["status"] != "ready":
                    raise ValueError("A planned case was not prepared")
                path = manifest_path.parent / case["path"]
                if digest(path) != case["sha256"]:
                    raise ValueError("Prepared video bytes changed")
                actual = probe_frames(path)
                if (
                    actual.pts != table.pts
                    or actual.time_base != table.time_base
                    or actual.width != table.width
                    or actual.height != table.height
                ):
                    raise ValueError("Case does not share the original's exact frame clock or dimensions")
                masks = {r["frame_index"]: r for r in case["masks"]}
                changed_pixels, compared = 0, 0
                for f in decode_frames(path, actual, range(len(actual))):
                    compared += 1
                    changed = np.any(f.rgb != baseline[f.index], axis=2)
                    if f.index in masks:
                        mask = masks[f.index]
                        p = manifest_path.parent / mask["path"]
                        if digest(p) != mask["sha256"]:
                            raise ValueError("Ground-truth mask bytes changed")
                        truth = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                        if truth is None or not np.array_equal(changed, truth > 0):
                            raise ValueError("Changed pixels do not match the declared mask exactly")
                        if int(np.count_nonzero(changed)) != mask["changed_pixels"] or not changed.any():
                            raise ValueError("Recorded nonempty changed-pixel count is incorrect")
                        changed_pixels += mask["changed_pixels"]
                    elif changed.any():
                        raise ValueError("Unedited source frame changed")
                if compared != len(table):
                    raise ValueError("Incomplete frame replay")
                pcm = decode_audio(path, source_sha256=case["sha256"]).pcm
                expected = (
                    np.r_[np.zeros(5120, np.float32), audio][: len(audio)]
                    if case["condition"] == "constant-audio-delay"
                    else audio
                )
                if not np.array_equal(pcm, expected):
                    raise ValueError("Audio changed outside its declared control transformation")
                result.update(
                    status="ok",
                    source_frames=compared,
                    changed_frames=len(masks),
                    exact_changed_pixels=changed_pixels,
                    audio_samples=len(pcm),
                    frame_pts_verified=True,
                    exact_before_during_after=True,
                    soundtrack_verified=True,
                )
            except Exception as error:
                result["reason"] = f"{type(error).__name__}: {error}"
            records.append(result)
    complete = {
        "schema": "chipmunk.mouth-edit-source-audit/4",
        "manifest_sha256": digest(manifest_path),
        "implementation_sha256": digest(Path(__file__)),
        "planned_cases": len(m["items"]),
        "passed": sum(r["status"] == "ok" for r in records),
        "source_groups": len(groups),
        "scope": "Local exact-input audit; not outside review, accuracy validation or proof of unknown customer edits",
        "cases": records,
    }
    with output.open("x") as stream:
        json.dump(complete, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in complete.items() if k != "cases"}))
    if complete["passed"] != complete["planned_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    audit(a.manifest, a.output)
