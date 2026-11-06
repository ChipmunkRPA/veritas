from copy import deepcopy

import pytest

from app.detectors.video.decode import FrameTable
from app.evidence import face_interval_binding, face_intervals

TABLE = FrameTable("1/1000", tuple(range(0, 2400, 250)), (), "I" * 10, 160, 120, 160, 120)
BOX = {"x": 20, "y": 12, "w": 64, "h": 72}


def tracked_scores(scores, *, include_ambiguous_track=False):
    points = [{"frame_index": i, "score": score} for i, score in enumerate(scores)]
    frames = [{"frame_index": i, "pts": TABLE.pts[i], "bbox": BOX} for i in range(len(scores))]
    exhibits = [{"kind": "regions", "created_by": "vid.faces.yunet", "regions": [
        {"region": {"kind": "bbox", "frame_pts": row["pts"], "bbox": BOX}} for row in frames]},
        {"kind": "face_track", "track": {"track_id": "track:0"}, "frames": frames}]
    if include_ambiguous_track:
        exhibits.append({"kind": "face_track", "track": {"track_id": "track:1"}, "frames": frames})
    return face_intervals.bind_tracks(points, exhibits, TABLE)


def test_sustained_face_change_binds_exact_before_during_after_frames():
    tracks = tracked_scores([.17, .19, .75, .89, .78, .11, .15])
    interval, = face_intervals.localize(tracks)
    measured = face_intervals.record(interval, TABLE)
    assert (measured["start_pts"], measured["end_pts"]) == (500, 1000)
    assert (interval.before[-1].frame_index, interval.peak.frame_index, interval.after[0].frame_index) == (1, 3, 5)
    packet = {"semantics": face_interval_binding.METHOD, "face_interval": measured, "frames": [
        {"anchor": {"frame_index": sample.frame_index, "pts": TABLE.pts[sample.frame_index]},
         "interval_role": role, "score": sample.score, "context_region": BOX, "is_peak": role == "during"}
        for role, sample in (("before", interval.before[-1]), ("during", interval.peak), ("after", interval.after[0]))]}
    source = {"time_base": TABLE.time_base, "frame_count": len(TABLE)}
    face_interval_binding.validate_packet(packet, source, table_pts=TABLE.pts)
    bad = deepcopy(packet)
    bad["frames"][2]["anchor"]["pts"] += 1
    with pytest.raises(ValueError, match="measured role"):
        face_interval_binding.validate_packet(bad, source, table_pts=TABLE.pts)


@pytest.mark.parametrize("scores", [[.1, .2, .11, .93, .12, .1, .09], [.85] * 7])
def test_isolated_peak_and_whole_track_score_do_not_create_a_local_interval(scores):
    assert face_intervals.localize(tracked_scores(scores)) == []


def test_ambiguous_faces_are_not_linked_to_an_interval():
    assert tracked_scores([.1, .2, .8, .9, .8, .1, .1], include_ambiguous_track=True) == {}
