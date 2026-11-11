from copy import deepcopy
from fractions import Fraction

import pytest

from app.core.canonical import canonical_json
from app.evidence import whole_face_binding as review


def test_native_sampling_uses_the_source_clock_and_earlier_ties():
    assert review.sample_indices([3, 4, 5, 6, 7, 8, 9, 10]) == (1, 2, 3, 4, 5, 6)
    assert review.sample_indices([0, 1, 3, 4, 6, 8, 10, 12, 14]) == (1, 3, 4, 5, 6, 7)
    assert review.sample_indices([0, 1, 1, 3, 4, 5, 6, 7]) == ()


def test_paired_measurement_requires_support_from_each_arm():
    reference = [0.05] * 6
    assert review.paired_score(reference, [2.0] * 3 + [None] * 3) is None
    assert review.paired_score(reference, [2.0] * 4 + [None] * 2) > review.REVIEW_POINT


@pytest.fixture
def packet_inputs():
    pts = list(range(30_000, 60_000, 311))
    source = {"width": 1920, "height": 1080, "time_base": "1/30000", "frame_count": len(pts),
              "timeline_origin_s": 1.0000004}
    indices = review.sample_indices(pts)
    box = {"x": 50, "y": 50, "w": 72, "h": 72}
    first, second = [0.05, 0.1, 0.2, 0.15, 0.02, None], [2.0, None, 2.5, 3.0, 1.2, None]
    score = review.paired_score(first, second)
    record = {"schema": review.METHOD, "time_base": source["time_base"], "score_hex": score.hex(),
              "flagged": score > review.REVIEW_POINT,
              "samples": [{"frame_index": i, "pts": pts[i], "box": box if a is not None else None,
                           "responses_hex": [x.hex() if x is not None else None for x in (a, b)]}
                          for i, a, b in zip(indices, first, second, strict=True)]}
    start, end = pts[indices[0]], pts[indices[-1]]
    def seconds(point):
        return float(point * Fraction(source["time_base"]))

    packet = {"whole_face_review": record, "semantics": review.METHOD, "claim_scope": "sampled_window",
              "affects_overall_confidence": False, "signal_refs": [{"id": "check-123"}],
              "window": {"frame_indices": list(indices), "time_base": source["time_base"],
                         "start_pts": start, "end_pts": end, "start_stream_s": seconds(start),
                         "end_stream_s": seconds(end), "start_playback_s": seconds(start) - source["timeline_origin_s"],
                         "end_playback_s": seconds(end) - source["timeline_origin_s"]},
              "frames": [{"anchor": {"frame_index": row["frame_index"], "pts": row["pts"], "time_base": source["time_base"],
                                     "stream_time_s": seconds(row["pts"]),
                                     "playback_time_s": seconds(row["pts"]) - source["timeline_origin_s"]},
                          "context_region": row["box"], "artifact_refs": {"original": "original", "crop": "crop"}}
                         for row in review.representatives(record)]}
    return packet, source, pts


def test_source_and_public_round_trip(packet_inputs):
    import json

    packet, source, pts = packet_inputs
    packet, source = json.loads(canonical_json([packet, source]))
    review.validate_packet(packet, source, pts=pts)
    review.validate_packet(packet, source)
    assert [f["anchor"]["frame_index"] for f in packet["frames"]] == [
        packet["window"]["frame_indices"][i] for i in (0, 3, 4)
    ]


@pytest.mark.parametrize("path,value", [
    (("affects_overall_confidence",), True),
    (("whole_face_review", "flagged"), False),
    (("whole_face_review", "samples", 2, "pts"), 40001),
    (("whole_face_review", "samples", 2, "responses_hex", 1), 99.0.hex()),
    (("window", "start_playback_s"), 100.0),
    (("frames", 1, "anchor", "stream_time_s"), 99.0),
    (("frames", 1, "context_region"), {"x": 30, "y": 30, "w": 72, "h": 72}),
])
def test_measurement_and_source_tampering_fails(packet_inputs, path, value):
    original, source, pts = packet_inputs
    packet = deepcopy(original)
    target = packet
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        review.validate_packet(packet, source, pts=pts)
