"""An explanation must depict the same face that produced the aggregate score."""

from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.image.adapters import learned
from app.engine.detector import Profile


@pytest.mark.parametrize(
    ("scores", "expected", "whole"),
    [([0.55, 0.95], 1, False), ([0.95, 0.55], 0, False),
     ([0.8, 0.8], 0, False), ([0.55, 0.95], 1, True)],
)
def test_face_attribution_uses_aggregate_input(monkeypatch, scores, expected, whole):
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    crops = [np.full((20, 20, 3), 30 + i, dtype=np.uint8) for i in range(2)]
    boxes = [(0, 0, 20, 20), (40, 30, 60, 50)]
    captured = {}
    classifier = SimpleNamespace(
        id2label={0: "fake"}, pos_idx=0,
        predict=lambda images: [{"p_ai": score} for score in scores],
    )
    monkeypatch.setattr(learned.HFImageClassifier, "from_parts", lambda *args: classifier)
    monkeypatch.setattr(learned, "analysis_rgb", lambda ctx: rgb)
    monkeypatch.setattr(learned, "face_boxes", lambda ctx: [0, 1])
    monkeypatch.setattr(learned, "face_input", lambda pixels, face: (crops[face], whole, False))
    monkeypatch.setattr(learned, "crop_box", lambda shape, face: boxes[face])
    monkeypatch.setattr(learned, "coordinate_space", lambda ctx: {
        "width": 100, "height": 80, "scale_to_original": 1.0,
    })

    def gradcam(clf, pixels):
        captured["pixels"] = pixels.copy()
        return np.ones((4, 4), dtype=np.float32)

    def heatmap(values, **kwargs):
        captured.update(kwargs)
        return "attribution-artifact"

    monkeypatch.setattr(learned, "vit_gradcam_map", gradcam)
    ctx = SimpleNamespace(
        lease=SimpleNamespace(get=lambda model: SimpleNamespace(obj=(None, None), revision="test")),
        device="cpu", profile=Profile.FULL,
        deadline=SimpleNamespace(check=lambda stage: None),
        exhibits=SimpleNamespace(heatmap=heatmap),
    )
    result = learned.Dima806Detector().run(ctx)
    assert result.aggregate.raw == max(scores)
    np.testing.assert_array_equal(captured["pixels"], crops[expected])
    assert captured["semantics"] == "attribution"
    assert result.details["attribution"]["input_scope_ref"] == f"face:{expected}"
    assert result.details["attribution"]["input_score"] == max(scores)
    assert result.details["attribution"]["coverage_kind"] == "input_footprint"
    assert result.details["attribution"]["crop_box_xyxy"] == (None if whole else list(boxes[expected]))
    if whole:
        assert captured["mask"] is None
    else:
        # The face crop footprint, rather than another face, is the attributed coverage.
        ys, xs = np.where(captured["mask"])
        assert xs.min() == int(boxes[expected][0] * 2.56)
        assert ys.min() == int(boxes[expected][1] * 2.56)
