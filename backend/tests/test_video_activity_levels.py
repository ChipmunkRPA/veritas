# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from app.bench.video_speech_activity_gap import BLOCK, scan
from app.evidence import video_activity_levels as levels


def test_equal_and_opposite_channels_produce_distinct_exact_sound_levels():
    tape = np.empty((22 * BLOCK, 2), dtype=np.int16)
    tape[:] = [16_384, 16_384]
    tape[6 * BLOCK:14 * BLOCK] = [16_384, -16_384]
    tape[15 * BLOCK:20 * BLOCK] = [0, 16_384]

    trace = levels.squared_samples(tape)
    levels.validate(trace, len(tape))
    values = levels.measure(trace, [6 * BLOCK, 14 * BLOCK])

    assert values == [
        {"source_samples": [0, 5 * BLOCK], "rms_dbfs": [-6.02, -6.02, None]},
        {"source_samples": [6 * BLOCK, 14 * BLOCK], "rms_dbfs": [-6.02, -6.02, -6.02]},
        {"source_samples": [15 * BLOCK, 20 * BLOCK], "rms_dbfs": [None, -6.02, -12.04]},
    ]
    assert levels.measurements(values)[0]["value"] == "-6.02 → -6.02 → Silence"


def test_stereo_measurement_rejects_clock_length_and_power_outside_signed_pcm():
    trace = [[0, 0, 0]] * 10
    with pytest.raises(ValueError, match="signed sample range"):
        levels.validate(trace, BLOCK * 11)
    trace[0] = [32_768**2 * BLOCK + 1, 0, 0]
    with pytest.raises(ValueError, match="signed sample range"):
        levels.validate(trace, BLOCK * 10)


def test_public_speech_activity_replay_matches_the_frozen_two_channel_scan():
    random = np.random.default_rng(101)
    pictures = [{"index": index, "sample": index * 533, "gap_units": 40_000 + 120_000 * (index % 3)}
                for index in range(120)]
    observed = []
    for start in (13, 22, 36, 54, 71):
        signals = random.uniform(.55, 1.0, (2, 120))
        signals[:, start:start + 13] = random.uniform(0, .3, (2, 13))
        original, enhanced = signals.tolist()
        old = scan(original, enhanced, pictures)["both"]
        assert levels.scan_both(original, enhanced, pictures) == old
        observed.append(old["reading"]["span"])
    assert len({tuple(span) for span in observed}) == 5
