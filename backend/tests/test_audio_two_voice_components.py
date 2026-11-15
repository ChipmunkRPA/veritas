# SPDX-License-Identifier: MIT
"""Guard the separate estimate and time axes in the two-voice evaluation."""

from scripts import audio_two_voice_components as study


def test_directional_readout_keeps_time_and_estimate_indices_distinct():
    first = [None] * len(study.prior.parent.GRID)
    second = first.copy()
    first[8] = 9.0
    second[6] = 8.5
    second[8] = 6.0
    row = {"streams": [{"scores": first}, {"scores": second}]}

    assert study.strongest(row, "directional") == (9.0, -64000, 0)
    assert study.strongest(row, "directional_lead") == (9.0, -64000, 0)

    first[8] = 6.1
    assert study.strongest(row, "directional") == (8.5, -48000, 1)
    assert study.strongest(row, "directional_lead") is None
