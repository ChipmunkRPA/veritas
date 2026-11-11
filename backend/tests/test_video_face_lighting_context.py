import numpy as np
import pytest

from app.evidence import face_lighting_context as lighting
from app.evidence.face_interval_binding import APPEARANCE_METHOD


def display_frames(boxes):
    return {
        "face_sequence": {"method": APPEARANCE_METHOD},
        "frames": [
            {
                "interval_role": role,
                "anchor": {"frame_index": index, "decoded_rgb_sha256": str(index) * 64},
                "context_region": dict(zip(("x", "y", "w", "h"), box, strict=True)),
            }
            for index, (role, box) in enumerate(zip(lighting.ROLES, boxes, strict=True))
        ],
    }


def test_compares_the_same_surrounding_area_when_face_boxes_move():
    source = {"width": 40, "height": 30}
    boxes = ((2, 3, 8, 10), (5, 3, 8, 10), (11, 3, 8, 10))
    packet = display_frames(boxes)
    frames = {}
    for index, (x, y, w, h) in enumerate(boxes):
        rgb = np.full((30, 40, 3), 180 if index == 1 else 80, dtype=np.uint8)
        rgb[y : y + h, x : x + w] = 230 if index == 1 else 120
        frames[index] = rgb
    context = lighting.record(packet, source, frames)
    assert context["status"] == "measured"
    assert [row["scene_sites"] for row in context["frames"]] == [1030] * 3
    assert [row["face_sites"] for row in context["frames"]] == [80] * 3
    assert [row["scene"] for row in context["frames"]] == [31.37, 70.59, 31.37]
    assert "both brightened" in lighting.explanation(context)
    packet.update(face_lighting=context, measurements=lighting.measurements(context))
    lighting.validate(packet, source)
    context["frames"][1]["scene_sites"] -= 1
    with pytest.raises(ValueError, match="displayed frame binding"):
        lighting.validate(packet, source)


def test_small_face_context_is_unassessed():
    source = {"width": 40, "height": 30}
    packet = display_frames([(0, 0, 2, 2)] * 3)
    context = lighting.record(packet, source, {})
    assert context["status"] == "insufficient_coverage"
    assert lighting.measurements(context) == []
    assert "insufficient" in lighting.explanation(context)
    packet.update(face_lighting=context, measurements=[])
    lighting.validate(packet, source)
