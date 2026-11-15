from app.evidence.video_viseme_timing import LAGS
from scripts.video_lip_timing_block_support import corroborated, support


def test_signed_block_support_requires_evidence_on_both_sides_of_each_split():
    measurement = {"span": [1280, 3840], "baseline_lag": 0, "change": -1920}
    events = []
    for sample in range(0, 5120, 640):
        scores = [0] * len(LAGS)
        scores[LAGS.index(-1920)] = 7 if 1280 <= sample < 3840 else -7
        events.append((sample, "test", scores, []))

    both = support(events, measurement, 640)
    assert both == [[[1, 7], [2, 14], [1, 7]], [[1, 7], [2, 14], [1, 7]]]
    assert corroborated(both)
    assert not corroborated(support(events[:-1], measurement, 640))
    assert corroborated(support(events[:-1], measurement))


def test_positive_total_does_not_override_a_region_with_the_wrong_sign():
    blocks = [[[1, 7], [2, 14], [1, -1]], [[1, 7], [2, 14], [1, 7]]]

    assert not corroborated(blocks)
