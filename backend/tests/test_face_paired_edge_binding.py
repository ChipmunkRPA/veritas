from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import pytest

from app.core.config import settings
from app.detectors.video.decode import FrameTable
from app.evidence import (
    face_boundary_binding,
    face_return_binding,
    face_short_proposal_binding,
    scene_return_binding,
)
from app.evidence import face_paired_edge_binding as paired
from app.evidence.face_interval_binding import APPEARANCE_METHOD
from app.evidence.video_packet_versions import BOUNDARY_VERSION, PAIRED_EDGE_VERSION, SHORT_PROPOSAL_VERSION


def record():
    origin = 150
    pts = tuple(origin + i * 1000 for i in range(120))
    table = FrameTable(
        time_base="1/30000",
        pts=pts,
        key_frames=(),
        pict_types="",
        width=1280,
        height=720,
        coded_width=1280,
        coded_height=720,
    )
    source = {"width": 1280, "height": 720, "frame_count": 120, "time_base": table.time_base}
    coarse = {"method": APPEARANCE_METHOD, "track_id": "track:0", "start_pts": pts[79], "end_pts": pts[95]}
    planned = paired.schedule(79, 95)
    start, stop = 87, 97

    def frame(i):
        return {"frame_index": i, "pts": pts[i], "decoded_rgb_sha256": f"{i:064x}"}

    matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    face = {
        "bbox": {"x": 600, "y": 320, "w": 110, "h": 110},
        "alignment_matrix": matrix,
        "alignment_matrix_hex": [[v.hex() for v in row] for row in matrix],
        "alignment_side": 112,
        "aligned_size": 112,
        "aligned_rgb_sha256": "a" * 64,
        "landmarks": [[610.0, 330.0], [690.0, 330.0], [640.0, 350.0], [620.0, 370.0], [680.0, 370.0]],
    }
    scenes = set(paired.motion_inputs(start, set(planned))) | set(paired.motion_inputs(stop, set(planned)))
    value = {
        "method": paired.METHOD,
        "track_id": "track:0",
        "time_base": table.time_base,
        "source_first_pts": origin,
        "start_frame_index": start,
        "end_frame_index": stop,
        "start_pts": pts[start],
        "end_pts": pts[stop],
        "start_stream_s": float(pts[start] * table.tb),
        "end_stream_s": float(pts[stop] * table.tb),
        "source_encoding": list(paired.ENCODING),
        "source_carrier": "mp4",
        "scheduled_frame_indices": planned,
        "frame_inputs": [{**frame(i), **deepcopy(face)} for i in planned],
        "scene_frame_inputs": [frame(i) for i in sorted(scenes)],
        "edges": {
            key: {
                "frame_indices": list(range(i - 3, i + 3)),
                "separation_hex": (0.40).hex(),
                "dispersion_hex": (0.02).hex(),
                "central_hex": (8.0).hex(),
                "sector_ids": [1, 2, 3, 4],
                "sectors_hex": [(2.0).hex()] * 4,
            }
            for key, i in (("entry", start), ("exit", stop))
        },
        "return_hex": [(0.35).hex(), (0.45).hex(), (0.04).hex(), (0.03).hex()],
        "anchors": {"before": 86, "during": 92, "after": 97},
    }
    return table, source, coarse, value


def test_paired_edge_enforces_eligible_native_clock_and_new_version():
    table, source, coarse, value = record()
    assert type(settings).model_fields["video_face_paired_edge_refinement"].default is False
    assert paired.supported(table, paired.ENCODING, "mp4")
    assert not paired.supported(table, ("hevc", "yuv420p", "hev1"), "mp4")
    assert len(value["scheduled_frame_indices"]) == paired.MAX_READS
    assert Fraction(value["start_pts"] - value["source_first_pts"], 30000) == Fraction(87, 30)
    face_boundary_binding.validate_record(value, coarse, source, table_pts=table.pts)
    packets = [{"face_boundary": value}]
    face_boundary_binding.validate_set({"version": PAIRED_EDGE_VERSION, "packets": packets})
    face_boundary_binding.validate_set({"version": BOUNDARY_VERSION, "packets": []})
    with pytest.raises(ValueError, match="version disagrees"):
        face_boundary_binding.validate_set({"version": BOUNDARY_VERSION, "packets": packets})
    with pytest.raises(ValueError, match="version disagrees"):
        face_boundary_binding.validate_set({"version": PAIRED_EDGE_VERSION, "packets": []})


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda r: r["anchors"].update(during=93), "three native roles"),
        (lambda r: r.update(start_pts=r["start_pts"] + 1), "native timestamp"),
        (lambda r: r["scene_frame_inputs"].pop(0), "omits native inputs"),
        (lambda r: r["scene_frame_inputs"][0].update(decoded_rgb_sha256="f" * 64), "omits native inputs"),
        (lambda r: r["edges"]["exit"].update(sectors_hex=[(6.0).hex()] * 4), "Near-head motion"),
        (lambda r: r["edges"]["entry"].update(separation_hex="0x1.0p+9999999"), "outside its range"),
        (lambda r: r.update(return_hex=[(0.01).hex()] * 4), "fixed support point"),
    ],
)
def test_paired_edge_rejects_altered_proof(change, reason):
    table, source, coarse, value = record()
    change(value)
    with pytest.raises(ValueError, match=reason):
        face_boundary_binding.validate_record(value, coarse, source, table_pts=table.pts)


@pytest.mark.parametrize("carrier", ["hevc", "webm", "avc", "avc_position"])
def test_short_proposal_requires_exact_schedule_pixel_inputs_and_arithmetic(carrier):
    webm = carrier == "webm"
    table, source, _, value = record()
    if webm:
        ticks = tuple(420 + scene_return_binding.webm_tick(i) for i in range(120))
        table = replace(table, time_base="1/1000", pts=ticks)
        source.update(time_base=table.time_base)
        assert face_short_proposal_binding.webm_source(table, scene_return_binding.WEBM_ENCODING, "webm")
        assert not face_short_proposal_binding.webm_source(table, scene_return_binding.WEBM_ENCODING, "mp4")
    if carrier == "avc":
        assert face_short_proposal_binding.avc_source(table, face_short_proposal_binding.AVC_ENCODING, "mp4")
        assert not face_short_proposal_binding.avc_source(table, paired.ENCODING, "mp4")
        assert not face_short_proposal_binding.avc_source(table, face_short_proposal_binding.AVC_ENCODING, "webm")
        irregular = replace(table, pts=(*table.pts[:70], table.pts[70] + 1, *table.pts[71:]))
        assert not face_short_proposal_binding.avc_source(irregular, face_short_proposal_binding.AVC_ENCODING, "mp4")
    method = {"hevc": face_short_proposal_binding.METHOD, "webm": face_short_proposal_binding.WEBM_METHOD,
              "avc": face_short_proposal_binding.AVC_METHOD,
              "avc_position": face_short_proposal_binding.POSITION_TRACK_AVC_METHOD}[carrier]
    encoding, container = face_short_proposal_binding.SOURCE_PROFILES[method]
    old_faces = {row["frame_index"]: row for row in value["frame_inputs"]}
    sample = list(face_short_proposal_binding.ANCHOR_SAMPLES)
    proposed = [87, 93]
    planned = face_short_proposal_binding.face_schedule(proposed)

    def frame(i):
        return {"frame_index": i, "pts": table.pts[i], "decoded_rgb_sha256": f"{i:064x}"}

    edges = deepcopy(value["edges"])
    edges["exit"]["frame_indices"] = face_short_proposal_binding.edge_frames(93)
    value.pop("track_id")
    value.update(
        method=method,
        time_base=table.time_base,
        source_first_pts=table.pts[0],
        source_encoding=list(encoding),
        source_carrier=container,
        start_frame_index=87,
        end_frame_index=93,
        start_pts=table.pts[87],
        start_stream_s=table.seconds(87),
        end_pts=table.pts[93],
        end_stream_s=table.seconds(93),
        shortlist_anchor_indices=sample,
        shortlist_edges=proposed,
        scheduled_frame_indices=planned,
        anchors={"before": 86, "during": 90, "after": 93},
        frame_inputs=[{**old_faces[87], **frame(i)} for i in planned],
        scene_frame_inputs=[frame(i) for i in range(120)],
        edges=edges,
    )
    if carrier == "avc_position":
        tracked = [600, 320, 110, 110]
        value["face_detection_boxes"] = [
            {"frame_index": i, "boxes": [tracked, [950, 35, 100 if 74 <= i <= 104 else 48, 48]]}
            for i in planned
        ]
    response = paired.edge_response(value, {row["frame_index"]: row for row in value["frame_inputs"]}, 87, 93)
    value.update(response=response, response_hex=response.hex())
    assert type(settings).model_fields["video_face_unrouted_short_review"].default is False
    assert type(settings).model_fields["video_face_short_avc_review"].default is False
    face_return_binding.validate_record(value, source, table_pts=table.pts)
    assert [(item["label"], item["value"]) for item in face_return_binding.proposal_boundary_measurements(value)] == [
        ("Start face change above nearby variation", 0.38),
        ("Start nearby regions with substantial motion", "0 of 4"),
        ("End face change above nearby variation", 0.38),
        ("End nearby regions with substantial motion", "0 of 4"),
    ]
    if webm:
        bad = deepcopy(value)
        bad["scene_frame_inputs"][70]["pts"] += 1
        with pytest.raises(ValueError, match="source clock"):
            face_return_binding.validate_record(bad, source, table_pts=table.pts)
        with pytest.raises(ValueError, match="measured carrier"):
            face_return_binding.validate_record({**value, "source_carrier": "mp4"}, source, table_pts=table.pts)
    if carrier == "avc":
        with pytest.raises(ValueError, match="measured carrier"):
            face_return_binding.validate_record({**value, "source_encoding": list(paired.ENCODING)}, source, table_pts=table.pts)
    for key, changed, reason in (
        ("shortlist_edges", [87, 93, 93], "edge shortlist"),
        ("scheduled_frame_indices", planned[:-1], "face sample budget"),
        ("shortlist_anchor_indices", value["shortlist_anchor_indices"][:-1], "shortlist anchors"),
        ("scene_frame_inputs", value["scene_frame_inputs"][:-1], "omits native pixel inputs"),
        ("response_hex", (0.11).hex(), "fixed support point"),
    ):
        with pytest.raises(ValueError, match=reason):
            face_return_binding.validate_record({**value, key: changed}, source, table_pts=table.pts)


def test_bounded_proposal_cannot_duplicate_the_final_paired_boundary():
    boundary = {"method": paired.METHOD, "time_base": "1/30000", "start_pts": 45000, "end_pts": 51000}
    proposed = {**boundary, "method": face_short_proposal_binding.METHOD}
    document = {"version": SHORT_PROPOSAL_VERSION, "packets": [
        {"face_boundary": boundary}, {"face_return": proposed, "semantics": proposed["method"]},
    ]}
    with pytest.raises(ValueError, match="duplicates the incumbent range"):
        face_return_binding.validate_set(document)
    proposed.update(start_pts=53000, end_pts=59000)
    face_return_binding.validate_set(document)


def test_moving_shortlist_interpolates_centers_and_preserves_rank_budget(monkeypatch):
    from app.detectors.video import face_paired_edge, face_short_proposal

    observed = {}

    def motion(index, reduced, box):
        observed[index] = box["x"]
        return {"central_hex": float(index).hex()}

    monkeypatch.setattr(face_paired_edge, "motion", motion)
    anchors = {i: {"x": i, "y": 0, "w": 100, "h": 100} for i in (14, 34, 64, 104)}
    edges = face_short_proposal.moving_shortlist({}, anchors, lambda _: None)
    assert observed[6] == 14 and observed[39] == 39 and observed[114] == 104
    assert edges == list(range(114, 98, -1))
    assert len(face_short_proposal_binding.face_schedule(edges)) <= 84
    face_indices = set(range(59, 73))
    assert face_short_proposal.has_measurable_pair([62, 70], face_indices, (7, 8))
    assert not face_short_proposal.has_measurable_pair([62, 70], face_indices - {70}, (7, 8))


def test_dominant_face_position_rejects_competing_or_discontinuous_detections():
    from app.evidence import face_primary_track_binding as track

    assert type(settings).model_fields["video_face_primary_track_avc_review"].default is False
    main, inset, peer = (10, 30, 120, 150), (430, 40, 48, 50), (350, 40, 110, 145)
    slots = [4, 14, 24, 34]
    reference = track.anchors(dict.fromkeys(slots, [main, inset]))
    assert track.follow(18, [inset, main], reference) == main
    assert track.follow(18, [main, peer], reference) is None
    assert track.follow(18, [inset], reference) is None
    assert track.follow(45, [main], reference) is None
    assert track.anchors({**dict.fromkeys(slots, [main]), 44: [peer]}) == {}
    planned = [*slots, 35]
    record = {"face_detection_boxes": [{"frame_index": i, "boxes": [list(main), list(inset)]} for i in planned]}
    faces = {35: {"bbox": dict(zip(("x", "y", "w", "h"), main, strict=True))}}
    track.validate(record, {"width": 1280, "height": 720}, planned, faces, slots)
    record["face_detection_boxes"][-1]["boxes"].append(list(peer))
    with pytest.raises(ValueError, match="does not follow"):
        track.validate(record, {"width": 1280, "height": 720}, planned, faces, slots)


def test_face_position_tracks_a_smaller_face_only_between_matching_anchors():
    from app.evidence import face_primary_track_binding as track

    assert type(settings).model_fields["video_face_position_track_avc_review"].default is False
    main, inset, peer = (10, 30, 120, 150), (430, 40, 48, 50), (350, 40, 110, 145)
    slots = [4, 14, 24, 34, 44, 54, 64, 74]
    detections = {i: [main, peer] if 24 < i < 74 else [main, inset] for i in slots}
    assert track.anchors(detections) == {}
    reference = track.anchors(detections, position_only=True)
    assert track.follow(54, [peer, main], reference, position_only=True) == main
    assert track.follow(54, [main, (22, 24, 110, 145)], reference, position_only=True) is None
    assert track.follow(81, [main, peer], reference, position_only=True) is None
    record = {"face_detection_boxes": [{"frame_index": i, "boxes": [list(box) for box in detections[i]]}
                                       for i in slots]}
    source = {"width": 1280, "height": 720}
    faces = {54: {"bbox": dict(zip(("x", "y", "w", "h"), main, strict=True))}}
    track.validate(record, source, slots, faces, slots, position_only=True)
    faces[54]["bbox"] = dict(zip(("x", "y", "w", "h"), peer, strict=True))
    with pytest.raises(ValueError, match="does not follow"):
        track.validate(record, source, slots, faces, slots, position_only=True)
