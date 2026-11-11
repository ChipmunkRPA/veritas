from __future__ import annotations

import cv2
import numpy as np

from scripts.video_registered_light_context import ROLES, contexts, illumination, scene_transform


def test_directional_exposure_requires_registered_scene_and_face_with_both_flanks():
    quadrants = [(y, x) for y in (0, 1) for x in (0, 1) for _ in range(6)]
    base = np.linspace(3., 4., len(quadrants))
    face = np.linspace(3., 4., 20)
    result = illumination(np.array([base, base + .1, base]), quadrants, np.array([face, face + .1, face]))
    assert result["context_ambiguous"] and result["supporting_quadrants"] == 4
    local_only = illumination(np.array([base, base, base]), quadrants, np.array([face, face + .1, face]))
    assert not local_only["context_ambiguous"]
    sustained = illumination(np.array([base, base + .1, base + .1]), quadrants, np.array([face, face + .1, face + .1]))
    assert not sustained["context_ambiguous"]
    discordant = illumination(np.array([base, base + .1, base]), quadrants, np.array([face, face - .1, face]))
    assert not discordant["context_ambiguous"]


def test_scene_tracks_small_camera_motion_and_detects_exposure_without_face_only_flags():
    rng = np.random.default_rng(203)
    base = cv2.GaussianBlur(rng.integers(18, 190, size=(180, 320), dtype=np.uint8), (3, 3), 0)
    mask = np.full(base.shape, 255, np.uint8)
    mask[45:125, 108:207] = 0
    shift = np.array([[1., 0., 3.], [0., 1., 2.]])
    moved = cv2.warpAffine(base, shift, (320, 180))
    matrix = scene_transform(base, moved, mask, mask)
    assert matrix is not None and np.allclose(matrix[:, 2], [-3, -2], atol=.25)
    assert scene_transform(np.zeros_like(base), moved, mask, mask) is None

    def frame(role, dx, *, broad):
        transform = np.array([[1., 0., float(dx)], [0., 1., 0.]])
        image = base.copy()
        if role == "during" and broad:
            image = np.round((image.astype(float) + 16) * 1.16 - 16).astype(np.uint8)
        moved = cv2.warpAffine(image, transform, (320, 180))
        moved_mask = cv2.warpAffine(mask, transform, (320, 180), flags=cv2.INTER_NEAREST)
        face = np.full(200, 117 if role == "during" else 100, np.uint8)
        return {"scene": moved, "outside": moved_mask, "face": face}

    for broad in (True, False):
        result = contexts({role: [frame(role, dx, broad=broad) for dx in (-2, 0, 3)] for role in ROLES})
        assert result["status"] == "measured" and result["context_ambiguous"] is broad
