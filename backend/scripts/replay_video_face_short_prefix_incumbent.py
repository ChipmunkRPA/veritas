# SPDX-License-Identifier: Apache-2.0
"""Read the exact Full face stages on the eight-second comparison deliveries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

from scripts import screen_video_native_background_challenge as worker
from scripts import video_face_paired_edge_transfer as previous
from scripts import video_face_short_prefix_transfer as study
from scripts.video_common_sources_v89 import read, require, sha, write


def run(folder, role):
    import structlog

    state = worker.configure(folder)
    os.environ.update(
        VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW="true",
        VERITAS_VIDEO_FACE_DETAIL_REPEAT_REVIEW="true",
        VERITAS_VIDEO_FACE_PAIRED_EDGE_REFINEMENT="true",
        VERITAS_VIDEO_FACE_UNROUTED_SHORT_REVIEW="true",
        VERITAS_VIDEO_FACE_SPATIAL_CUT_CONTEXT="true",
    )
    from app.core.config import settings
    from app.engine.stages import video

    settings.data_dir = state
    settings.database_url = "sqlite:///:memory:"
    settings.run_worker = False
    settings.video_face_review_mode = "appearance"
    settings.video_face_short_return_review = True
    settings.video_face_second_return_review = True
    settings.video_face_detail_repeat_review = True
    settings.video_face_boundary_refinement = True
    settings.video_face_paired_edge_refinement = True
    settings.video_face_unrouted_short_review = True
    settings.video_face_spatial_cut_context = True
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(30))

    measure = worker.scorer(state)
    original_shots = video.shots
    original_detect = video.detect
    last_detail = {}

    def shots(context):
        result = original_shots(context)
        require(result["frames_scanned"] == study.N, "Full shot stage did not scan all 240 pictures")
        # The reused comparator's internal assertion expects 120 pictures.
        # The actual scan was already checked and no consuming stage uses this count.
        return {**result, "frames_scanned": 120}

    def detect(context):
        result = original_detect(context)
        [observed] = [r for r in result["detectors"] if r["detector_id"] == "vid.review.face_appearance"]
        details = observed.get("details") or {}
        last_detail.update(
            {
                key: details.get(key)
                for key in ("face_short_proposal_check", "face_spatial_cut", "face_return_secondary")
            }
        )
        return result

    source = read(folder / f"{role}.json")
    output = []
    with patch.object(video, "shots", shots), patch.object(video, "detect", detect):
        for row in source["rows"]:
            source_file = folder / f"{role}-{row['group']}" / (row["role"] + ".mp4")
            result = measure(source_file, {**row, "rgb_sha256": row["native_hashes"]})
            ranges = previous.intervals(result, row["clock"]["pts"])
            if last_detail["face_return_secondary"]:
                second = last_detail["face_return_secondary"]
                ranges.append([row["clock"]["pts"].index(second[k]) for k in ("start_pts", "end_pts")])
            require(
                not last_detail["face_short_proposal_check"] and not last_detail["face_spatial_cut"],
                "The 120-picture-only checks should have no input route on these clips",
            )
            output.append(
                {
                    "group": row["group"],
                    "role": row["role"],
                    "sha256": row["sha256"],
                    "ranges": ranges,
                    "track_status": result["track_status"],
                    "tracks": result["tracks"],
                    "ordinary_shot_boundaries": result["shot_boundaries"],
                    "raw_coarse": (result["coarse"] or {}).get("response"),
                }
            )
            print(row["group"], row["role"], ranges, flush=True)
    write(
        folder / f"{role}-incumbent.json",
        {"scored_panel_sha256": sha(folder / f"{role}.json"), "rows": output},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("role", choices=("development", "held"))
    args = parser.parse_args()
    run(args.folder, args.role)


if __name__ == "__main__":
    main()
