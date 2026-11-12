# SPDX-License-Identifier: Apache-2.0
"""Compare two acoustic template lengths for the marked earlier-copy timing route."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.evidence import video_reference, video_reference_timing
from scripts import grid_fresh_release_qualification as prior
from scripts import grid_reserve_neural_mouth as geometry
from scripts.review_grid_single_clip_release_timing import RGB_SIZE, digest, native, read, sha
from scripts.video_common_sources_v89 import write

SCHEMA = "chipmunk.grid-marked-release-transfer/1"
POLICY = {
    "target": "Revisit every scored view from the exposed, source-bound fresh-release panel.",
    "marker": "Earlier last-closed native picture from the muted-video bracket; fixed frame 37 for lexical negatives.",
    "candidate": "One 60 ms template from the marked picture, using samples 0 through 959 relative to its clock.",
    "search": "Reuse the verified reference experiment's unchanged 0.94 correlation, 0.08 margin and ±400 ms search.",
    "qualify": "Signed lag within two native pictures, with the whole selected submitted template inside the shifted range.",
    "controls": "Report every alert on unchanged or ordinary gain-only views.",
}


def bindings(panel):
    return {
        "input_ledgers": {
            f: digest(panel / f)
            for f in (
                "freeze.json",
                "visual-labels.json",
                "media.json",
                "readout.json",
                "independent-audit.json",
            )
        },
        "code": {
            path.name: digest(path)
            for path in map(
                Path, (__file__, prior.__file__, video_reference.__file__, video_reference_timing.__file__)
            )
        },
        "timing_weight_sha256": video_reference_timing.MODEL_SHA256,
    }


def freeze(panel, run):
    audit = read(panel / "independent-audit.json")
    assert audit["verified"] and audit["readout_sha256"] == digest(panel / "readout.json")
    assert audit["freeze_sha256"] == digest(panel / "freeze.json") and audit["scored_views"] == 58
    by_id = {row["id"]: row for row in read(panel / "visual-labels.json")["cases"]}
    references = {row["id"]: row for row in read(panel / "freeze.json")["roster"]}
    cases = []
    for row in read(panel / "media.json")["cases"]:
        label, ref = by_id[row["id"]], references[row["id"]]
        marker = label["bracket_frames"][0] if ref["kind"] != "none" else 37
        assert 2 <= marker <= 70
        cases.append(
            {
                **{key: row[key] for key in ("id", "role", "shift", "span_samples", "file", "file_sha256")},
                "reference_file": ref["path"],
                "reference_sha256": ref["source_sha256"],
                "mark_frame": marker,
            }
        )
    assert len(cases) == 58 and len({row["id"] for row in cases}) == 19
    run.mkdir(mode=0o700)
    write(
        run / "freeze.json", {"schema": SCHEMA, "policy": POLICY, "bindings": bindings(panel), "cases": cases}
    )


def short_audio(original, submitted, mark):
    # The center is a fixed 20 ms after the marked picture. No acoustic event is supplied.
    result = prior.match_template(original, submitted, mark * 640 + 320)
    if result is None:
        return None
    return {
        "source_samples": result["source_samples"],
        "submitted_samples": result["matched_samples"],
        "offset_samples": result["offset_samples"],
        "correlation": result["correlation"],
        "correlation_margin": result["margin"],
        "source_pcm_sha256": result["source_slice_sha256"],
        "submitted_pcm_sha256": result["matched_slice_sha256"],
    }


def check_assets(timing, assets):
    if timing["status"] == "unassessed":
        return
    for side in ("earlier", "submitted"):
        for kind in ("crop", "speech"):
            row = timing[side][kind]
            assert sha(assets[row["file"]]) == row["sha256"]


def score(panel, run):
    lock = read(run / "freeze.json")
    assert lock["bindings"] == bindings(panel) and lock["policy"] == POLICY
    sys.path.insert(0, str(geometry.runtime.SITE))
    model = geometry.ASSETS / "permissions/face_landmarker.task"
    assert video_reference_timing.available(model)
    rows = []
    for index, row in enumerate(lock["cases"], 1):
        paths = Path(row["reference_file"]), Path(row["file"])
        assert [digest(path) for path in paths] == [row["reference_sha256"], row["file_sha256"]]
        options = {
            "include_local": False,
            "timing_event_s": Fraction(row["mark_frame"], 25),
            "timing_model": model,
        }
        current, assets = video_reference.compare(*paths, **options)
        check_assets(current["timing"], assets)
        with patch.object(video_reference_timing, "audio_shift", short_audio):
            candidate, assets = video_reference.compare(*paths, **options)
        check_assets(candidate["timing"], assets)
        assert current["status"] == candidate["status"] == "measured"
        assert current["picture"]["matching_frames"] == candidate["picture"]["matching_frames"] == 75
        assert current["audio"] == candidate["audio"]
        rows.append(
            {
                "id": row["id"],
                "role": row["role"],
                "current": current["timing"],
                "candidate": candidate["timing"],
            }
        )
        if index % 12 == 0:
            print(f"Compared {index}/{len(lock['cases'])} marked deliveries", flush=True)
    write(run / "readout.json", {"freeze_sha256": digest(run / "freeze.json"), "rows": rows})


def review(panel, run, public):
    lock = read(run / "freeze.json")
    outcomes = read(run / "readout.json")
    assert lock["bindings"] == bindings(panel) and outcomes["freeze_sha256"] == digest(run / "freeze.json")
    rows, totals = defaultdict(Counter), Counter()
    for case, result in zip(lock["cases"], outcomes["rows"], strict=True):
        assert (case["id"], case["role"]) == (result["id"], result["role"])
        sources = tuple(Path(case[key]) for key in ("reference_file", "file"))
        expected = -case["shift"] * 640
        role = rows[case["role"]]
        role["views"] += 1
        decoded = tuple(native(path) for path in sources)
        for arm in ("current", "candidate"):
            timing = result[arm]
            assessed = timing["status"] != "unassessed"
            alert = timing["status"] == "timing_difference"
            useful = exact = False
            if assessed:
                earlier, submitted = timing["earlier"], timing["submitted"]
                assert earlier["frame_index"] == case["mark_frame"]
                assert not timing["affects_overall_confidence"]
                clocks = []
                for proof, (pts, time_base, pictures, pcm) in zip((earlier, submitted), decoded, strict=True):
                    k = proof["frame_index"]
                    frame_bytes = pictures[k * RGB_SIZE : (k + 1) * RGB_SIZE]
                    x0, y0, x1, y1 = proof["crop_xyxy"]
                    pixels = np.frombuffer(frame_bytes, np.uint8).reshape(288, 360, 3)
                    start, end = proof["pcm_samples"]
                    assert (
                        Fraction(proof["picture_time_base"]) == time_base and proof["picture_pts"] == pts[k]
                    )
                    assert pts[k] * time_base == Fraction(k, 25)
                    assert sha(frame_bytes) == proof["native_picture_rgb_sha256"]
                    assert sha(pixels[y0:y1, x0:x1].tobytes()) == proof["native_crop_rgb_sha256"]
                    assert sha(pcm[start:end].tobytes()) == proof["pcm_s16_sha256"]
                    assert 0 <= start < end <= len(pcm) and end - start == (
                        960 if arm == "candidate" else 2560
                    )
                    clocks.append((k * 640, start))
                    totals["native_picture_crop_and_pcm_records"] += 1
                (left_frame, left_sound), (right_frame, right_sound) = clocks
                lag = right_frame - left_frame - (right_sound - left_sound)
                assert abs(lag - timing["offset_ms"] * 16) < 1e-7
                assert alert == (abs(lag) >= 1920)
                if arm == "candidate":
                    assert left_sound == left_frame
                if expected and alert:
                    start, end = submitted["pcm_samples"]
                    lo, hi = case["span_samples"]
                    useful = lo <= start < end <= hi and abs(lag - expected) <= 1280
                    exact = useful and lag == expected
            role[arm + "_assessed"] += assessed
            role[arm + "_alerts"] += alert
            role[arm + "_localized"] += useful
            role[arm + "_exact"] += exact
            totals["replayed_dispositions"] += 1
    result = {
        "schema": SCHEMA,
        "policy": POLICY,
        "freeze_sha256": digest(run / "freeze.json"),
        "readout_sha256": digest(run / "readout.json"),
        "exposed_panel": True,
        "roles": dict(rows),
        "audit": dict(totals),
    }
    write(public, result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "review"))
    parser.add_argument("panel", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--public", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        assert args.public is not None
        review(args.panel, args.run, args.public)
    else:
        {"freeze": freeze, "score": score}[args.phase](args.panel, args.run)


if __name__ == "__main__":
    main()
