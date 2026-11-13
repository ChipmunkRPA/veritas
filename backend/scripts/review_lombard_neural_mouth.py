# SPDX-License-Identifier: Apache-2.0
"""Verify native Lombard redraws and re-observe mouth releases without case labels."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence.video_reference_timing import _mesh
from scripts import build_lombard_neural_mouth as build
from scripts import qualify_lombard_mouth_events as original
from scripts.video_common_sources_v89 import read, require, sha, write


def partitions(run, split, include_neural=True):
    rows = []
    for stage in ("controls", "neural") if include_neural else ("controls",):
        path = run / f"{stage}-{split}.json"
        receipt = read(path)
        require(
            receipt["freeze_sha256"] == sha(run / "freeze.json"), "Construction is from a different freeze"
        )
        rows.extend(receipt["cases"])
    return rows


def prepare(run, split):
    lock = read(run / "freeze.json")
    by_id = {r["source"]["ordinal"]: r["source"] for r in lock["sources"]}
    cases = partitions(run, split)
    require(len(cases) == (8 if split == "development" else 44), "Incomplete frozen construction roles")
    rows = [
        {
            "token": r["token"],
            "file_sha256": r["file_sha256"],
            "coarse_hint_s": by_id[r["ordinal"]]["aligned_following_phone_s"],
        }
        for r in cases
        if r["status"] == "authored"
    ]
    require(len({r["token"] for r in rows}) == len(rows), "Duplicate blinded fixture")
    write(run / f"blind-plan-{split}.json", sorted(rows, key=lambda row: row["token"]))


def observe(run, split):
    plan = run / f"blind-plan-{split}.json"
    cv2.setNumThreads(1)
    module, task = build.retained.mp_task()
    output = []
    with closing(task):
        for item in read(plan):
            path = run / "deliveries" / f"{item['token']}.mov"
            require(sha(path) == item["file_sha256"], "Blinded delivery bytes differ")
            table = probe_frames(path, timeout_s=15)
            times = [v * table.tb for v in table.pts]
            step = times[1] - times[0]
            hint = Fraction(item["coarse_hint_s"])
            late_hint = hint + build.DELAY_FRAMES * step
            indices = [
                i for i, t in enumerate(times) if hint - Fraction(1, 2) <= t <= late_hint + Fraction(1, 2)
            ]
            apertures = [None] * len(times)
            anchors = {}
            for frame in decode_frames(path, table, indices, timeout_s=30):
                require(
                    frame.pts_verified and frame.pts == table.pts[frame.index], "Blind picture clock differs"
                )
                found = _mesh(task, module, frame.rgb)
                if found is None:
                    continue
                gaps, box = found
                x0, y0, x1, y1 = box
                apertures[frame.index] = float(np.median(gaps[:3]))
                anchors[frame.index] = {
                    "pts": frame.pts,
                    "box": box,
                    "rgb_sha256": original.digest(frame.rgb.tobytes()),
                    "crop_sha256": original.digest(frame.rgb[y0:y1, x0:x1].tobytes()),
                }
            releases = {
                label: original.visual_releases(apertures, times, center)
                for label, center in (("source_neighborhood", hint), ("later_neighborhood", late_hint))
            }
            for event in releases.values():
                if "frames" in event:
                    event["anchors"] = [anchors[i] for i in event["frames"]]
            output.append(
                {
                    "token": item["token"],
                    "file_sha256": item["file_sha256"],
                    "measured": len(anchors),
                    "time_base": table.time_base,
                    "releases": releases,
                }
            )
            print("Observed blind", split, item["token"], len(anchors), flush=True)
    write(run / f"blind-observations-{split}.json", {"plan_sha256": sha(plan), "rows": output})


def exact_delivery(run, case, row, table, source):
    path = run / "deliveries" / f"{case['token']}.mov"
    require(sha(path) == case["file_sha256"], "Authored delivery differs")
    delivered, _ = build.table_for(path, row)
    scheduled = {item["index"]: item for item in case["edits"]}
    actual = {}
    for frame in decode_frames(path, delivered, None, timeout_s=30):
        i = frame.index
        require(frame.pts_verified and frame.pts == table.pts[i], "Reopened picture lacks exact PTS")
        mask = np.any(frame.rgb != source[i], axis=2).astype(np.uint8) * 255
        if mask.any():
            expected = scheduled[i]
            require(
                expected["pts"] == frame.pts
                and expected["driver_pts"] == table.pts[expected["driver_index"]]
                and expected["mask_sha256"] == original.digest(mask.tobytes())
                and expected["rgb_sha256"] == original.digest(frame.rgb.tobytes()),
                "Reopened native edit bytes differ",
            )
            actual[i] = int(np.count_nonzero(mask))
    require(set(actual) == set(scheduled), "Actual altered pictures differ from the authored range")
    return {
        "native_pictures": len(delivered.pts),
        "float_pcm_samples": build.verified_audio(path, row),
        "changed_pictures": len(actual),
        "changed_pixels": sum(actual.values()),
    }


def release_status(event, row, table, *, delayed=False):
    if event["status"] != "observed_reopening":
        return {"status": event["status"]}
    original_frames = row["visual"]["frames"]
    result = {"frames": event["frames"], "seconds": event["seconds"], "anchors": event["anchors"]}
    require(
        event["seconds"] == [str(table.pts[i] * table.tb) for i in event["frames"]],
        "Blind observation clocks differ from delivery",
    )
    start, stop = row["event"]["native_edit_frames"]
    middle = sum(map(Fraction, event["seconds"])) / 2
    burst = sum(map(Fraction, row["event"]["audio_range_s"])) / 2
    shift = event["frames"][0] - original_frames[0]
    result.update(visual_picture_shift=shift, acoustic_bracket_midpoint_offset_s=str(middle - burst))
    if delayed:
        supported = (
            abs(shift - build.DELAY_FRAMES) <= 1
            and start <= min(event["frames"]) < max(event["frames"]) < stop
        )
        return result | {"status": "observed_delayed_control" if supported else "different_observed_motion"}
    supported = (
        abs(shift) <= 1
        and abs(middle - burst) <= Fraction(2, 25)
        and start <= min(event["frames"]) < max(event["frames"]) < stop
    )
    return result | {"status": "source_event_retained" if supported else "different_observed_motion"}


def audit(sources, run, split):
    lock = read(run / "freeze.json")
    require(lock == build.freeze(sources), "Native replay does not match the original freeze")
    plan = run / f"blind-plan-{split}.json"
    observations_path = run / f"blind-observations-{split}.json"
    observations = read(observations_path)
    require(observations["plan_sha256"] == sha(plan), "Blinded observations do not match the routed plan")
    by_token = {r["token"]: r for r in observations["rows"]}
    require(len(by_token) == len(observations["rows"]) == len(read(plan)), "Blinded result roster differs")
    all_cases = partitions(run, split)
    output = []
    for row in lock["sources"]:
        item = row["source"]
        k = item["ordinal"]
        if item["split"] != split:
            continue
        path = original.source(sources, item, "front")
        table, _ = build.table_for(path, row)
        pictures = build.read_pictures(path, table)
        cases = {c["role"]: c for c in all_cases if c["ordinal"] == k}
        require(set(cases) == {*build.CONTROL_ROLES, build.NEURAL}, "One frozen case is missing")
        baseline_geometry = None
        for role, case in cases.items():
            evidence = {
                "ordinal": k,
                "group": item["group"],
                "condition": item["condition"],
                "role": role,
                "status": case["status"],
            }
            if case["status"] == "authored":
                expected_indices = (
                    list(range(*row["event"]["native_edit_frames"])) if role != "original" else []
                )
                require([e["index"] for e in case["edits"]] == expected_indices, "Case span differs")
                geometry = [edit["alpha_sha256"] for edit in case["edits"]]
                if role != "original":
                    if baseline_geometry is None:
                        baseline_geometry = geometry
                    require(geometry == baseline_geometry, "Matched controls use different mouth geometry")
                observation = by_token[case["token"]]
                require(observation["file_sha256"] == case["file_sha256"], "Blind source identity differs")
                delayed = role == "delayed_pixels_ordinary"
                event = observation["releases"]["later_neighborhood" if delayed else "source_neighborhood"]
                evidence.update(
                    exact_delivery(run, case, row, table, pictures),
                    file_sha256=case["file_sha256"],
                    observation=release_status(event, row, table, delayed=delayed),
                    authored_frames=row["event"]["native_edit_frames"] if expected_indices else None,
                    authored_seconds=row["event"]["native_edit_range_s"] if expected_indices else None,
                )
                if role == "original":
                    require(
                        event.get("frames") == row["visual"]["frames"],
                        "Reopened original observation differs",
                    )
            output.append(evidence)
            print(
                "Audited",
                split,
                k,
                role,
                evidence.get("observation", {}).get("status", case["status"]),
                flush=True,
            )
    counts = Counter((r["role"], r.get("observation", {}).get("status", r["status"])) for r in output)
    write(
        run / f"review-{split}.json",
        {
            "freeze_sha256": sha(run / "freeze.json"),
            "blind_observations_sha256": sha(observations_path),
            "cases": output,
            "counts": [[*k, v] for k, v in sorted(counts.items())],
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "observe", "audit"))
    p.add_argument("sources", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("--split", choices=("development", "held"), default="development")
    a = p.parse_args()
    if a.phase == "audit":
        audit(a.sources, a.run, a.split)
    else:
        (prepare if a.phase == "prepare" else observe)(a.run, a.split)


if __name__ == "__main__":
    main()
