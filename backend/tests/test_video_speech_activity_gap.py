# SPDX-License-Identifier: Apache-2.0

import numpy as np

from app.bench.video_speech_activity_gap import BLOCK, scan
from app.evidence.video_viseme_binding import activity_profile


def test_the_joint_reading_uses_one_interval_and_original_picture_brackets():
    original = np.full(38, .9)
    enhanced = np.full(38, .8)
    original[12:23] = .1
    enhanced[12:23] = .2
    frames = [{"index": i, "sample": i * BLOCK, "gap_units": 200_000 * (i % 2)} for i in range(38)]

    reading = scan(original, enhanced, frames)["both"]["reading"]

    start, end = reading["span"]
    assert 12 * BLOCK <= start < end <= 23 * BLOCK
    assert reading["margin_millionths"] == 600_000
    assert reading["activity_millionths"] == [[900_000, 800_000], [100_000, 200_000], [900_000, 800_000]]
    indexes = reading["picture_frames"]
    assert indexes["before"] * BLOCK < start <= indexes["during"] * BLOCK < end <= indexes["after"] * BLOCK


def test_drop_is_unassessed_without_mouth_motion_or_both_speech_flanks():
    scores = np.full(38, .9)
    scores[12:23] = .05
    constant = [{"index": i, "sample": i * BLOCK, "gap_units": 0} for i in range(38)]
    moving = [{**frame, "gap_units": 200_000 * (i % 2)} for i, frame in enumerate(constant)]

    assert scan(scores, scores, constant)["both"] == {"assessed": False, "reading": None}

    scores[23:] = .05
    assert scan(scores, scores, moving)["both"] == {"assessed": False, "reading": None}


def test_stereo_pcm_profile_uses_the_decoder_clock_without_aac_container_fields():
    ticks = range(0, 60_000, 1_000)
    mapping = {"source_channels": 2, "source_codec": "pcm_s16le", "source_time_base": "1/16000",
               "source_sample_rate": 16000, "samples": 32000, "source_start_pts": 0, "timeline_time_base": "1/16000",
               "timeline": [{"start_sample": 0, "end_sample": 32_000, "source_pts": 0}]}

    assert activity_profile(ticks, "1/30000", mapping) == "pcm"
    assert activity_profile(ticks, "1/30000", {**mapping, "source_start_pts": 1000}) is None
    assert activity_profile(ticks, "1/30000", {**mapping, "audio_stream_indices": [1, 2]}) is None
    assert activity_profile(ticks, "1/24000", mapping) is None
