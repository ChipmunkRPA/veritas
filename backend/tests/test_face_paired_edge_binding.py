from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest

from app.core.config import settings
from app.detectors.video.decode import FrameTable
from app.evidence import face_boundary_binding
from app.evidence import face_paired_edge_binding as paired
from app.evidence.face_interval_binding import APPEARANCE_METHOD
from app.evidence.video_packet_versions import BOUNDARY_VERSION, PAIRED_EDGE_VERSION


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
