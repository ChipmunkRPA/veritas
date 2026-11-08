"""Watermark evidence intervals preserve samples and never become AI confidence."""

import numpy as np
import pytest

from scripts import public_audio_watermark_v1 as method


def test_partial_interval_and_tail_are_source_aligned():
    values = np.zeros(128013, np.float32)
    values[32000:96000] = 0.9
    result = method.summarize(values)
    assert result["intervals"] == [{"start_sample": 32000, "end_sample": 96000, "start_s": 2.0, "end_s": 6.0}]
    assert result["bins"][-1]["end_sample"] == 128013
    assert (
        result["provider_authenticated"]
        is result["used_in_confidence"]
        is result["early_stop_eligible"]
        is False
    )


def test_equality_and_exact_half_bin_do_not_select_an_interval():
    values = np.full(8000, 0.5, np.float32)
    values[:2000] = 0.9
    result = method.summarize(values)
    assert result["intervals"] == [] and result["positive_sample_fraction"] == 0.25


@pytest.mark.parametrize(
    "values",
    [
        np.array([], np.float32),
        np.array([np.nan], np.float32),
        np.array([np.inf], np.float32),
        np.array([1.01], np.float32),
        np.array([0.5], np.float64),
    ],
)
def test_unavailable_or_invalid_responses_are_not_originality_evidence(values):
    with pytest.raises(ValueError):
        method.summarize(values)
