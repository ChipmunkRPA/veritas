from types import SimpleNamespace

import numpy as np

from app.detectors.image.faces import Face
from app.detectors.video import face_detail_inputs
from app.evidence.face_detail_binding import SCHEDULE


def test_face_detail_track_uses_source_order_with_cached_frames(monkeypatch):
    rgb = np.zeros((80, 80, 3), dtype=np.uint8)
    eyes = [(28., 30.), (52., 30.)]
    face = Face(12, 12, 56, 56, .99, eyes)
    monkeypatch.setattr(face_detail_inputs, "detect_faces", lambda *args, **kwargs: [face])
    monkeypatch.setattr(face_detail_inputs, "align", lambda *args: rgb)
    cached_then_decoded = SCHEDULE[::2] + SCHEDULE[1::2]
    store = SimpleNamespace(table=SimpleNamespace(pts=range(100)), native_pts_verified=lambda index: True,
                            get_many=lambda indices: {index: rgb for index in cached_then_decoded})
    samples = face_detail_inputs.tracked_faces(store, None)
    assert tuple(sample[0] for sample in samples) == SCHEDULE
