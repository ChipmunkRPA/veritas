# SPDX-License-Identifier: Apache-2.0
"""Background support for a single face transition with sparse coarse texture."""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.detectors.image.faces import Face, iou
from app.detectors.video import face_return, face_return_fine_scene
from app.evidence import face_single_boundary_binding as binding

WIDTH, HEIGHT = 160, 90
Y, X = np.meshgrid(np.arange(HEIGHT) * binding.HEIGHT / HEIGHT,
                   np.arange(WIDTH) * binding.WIDTH / WIDTH, indexing="ij")


def neighbor_pairs(left: int, right: int):
    step = right - left
    for offset in (1, 2, 3):
        yield left - offset * step, left - (offset - 1) * step
    for offset in (1, 2, 3):
        yield right + (offset - 1) * step, right + offset * step


def _evidence_edges(record):
    coarse, exact = record["coarse"], record["boundary"]
    sources, methods = record["source_frames"], record["scene_context"]
    return ((coarse - 2, coarse, methods["coarse_boxes"], methods["coarse"], "scene"),
            (exact - 1, exact, [sources[key]["box"] for key in ("before", "transition")], methods["boundary"], "fine_scene"))


def required_frames(record):
    indices = set()
    for left, right, _, basis, _ in _evidence_edges(record):
        indices.update((left, right))
        if basis == binding.MOTION_CONTEXT:
            for first, second in neighbor_pairs(left, right):
                indices.update((first, second))
    return indices


def verify(record, pictures):
    """Recompute the two background witnesses using only the original pictures and bound geometry."""
    background = Background(pictures)
    arm = record["scene_context"]["arm"]
    for left, right, boxes, basis, name in _evidence_edges(record):
        first, second = (Face(*(box[key] for key in "xywh"), 0.) for box in boxes)
        if iou(first, second) < binding.MINIMUM_FACE_IOU:
            raise ValueError("The face geometry differs across the recorded background comparison")
        samples = [face_return.Sample(face, None, face_return.scene(pictures[index], face))
                   for index, face in ((left, first), (right, second))]
        measured_basis, value = background.readings(left, right, *samples, (arm,)).get(arm, (None, None))
        if (measured_basis != basis or value is None
                or not math.isclose(value, binding.reading(record, name), abs_tol=1e-8, rel_tol=0)):
            raise ValueError("The original video does not reproduce the recorded sparse background comparison")


def _region(box_a, box_b):
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    mid = (xa + wa / 2 + xb + wb / 2) / 2
    w, h = (wa + wb) / 2, (ha + hb) / 2
    bottom_face = max(ya + ha, yb + hb)
    return (max(0, math.floor((mid - 1.05 * w) * WIDTH / binding.WIDTH)),
            max(0, math.floor((bottom_face + .12 * h) * HEIGHT / binding.HEIGHT)),
            min(WIDTH, math.ceil((mid + 1.05 * w) * WIDTH / binding.WIDTH)),
            min(HEIGHT, math.ceil((bottom_face + .78 * h) * HEIGHT / binding.HEIGHT)))


def _motion_pixels(rgb):
    reduced = cv2.resize(rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(reduced, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    softened = cv2.GaussianBlur(yuv, (3, 3), 0)
    gray = softened[:, :, 0]
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, scale=.25), cv2.Sobel(gray, cv2.CV_32F, 0, 1, scale=.25))
    return softened, gradient


def _difference(first, second):
    delta = np.abs(first - second)
    return delta[:, :, 0] + .5 * np.maximum(delta[:, :, 1], delta[:, :, 2])


class Background:
    """Reusable pixel measurements on a bounded native frame sequence."""

    def __init__(self, frames):
        self.frames = frames
        self.motion = {}
        self.tiles = {}

    def _motion_at(self, index):
        if index not in self.motion:
            self.motion[index] = _motion_pixels(self.frames[index])
        return self.motion[index]

    def _tiles_at(self, index, box):
        key = index, box
        if key not in self.tiles:
            self.tiles[key] = face_return_fine_scene.frame(self.frames[index], box)
        return self.tiles[key]

    def _motion_score(self, left, right, boxes):
        x, y, end_x, end_y = _region(*boxes)
        if end_x - x < 40 or end_y - y < 12:
            return None
        area = np.s_[y:end_y, x:end_x]
        first, ga = self._motion_at(left)
        second, gb = self._motion_at(right)
        shared = (ga[area] >= 5.) & (gb[area] >= 5.)
        if np.count_nonzero(shared) < 32 or sum(np.count_nonzero(part) >= 8 for part in np.array_split(shared, 3, axis=1)) < 2:
            return None
        a, b = first[area][:, :, 0], second[area][:, :, 0]
        a, b = a - a.mean(), b - b.mean()
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        similarity = float(np.clip((a * b).sum() / denominator, -1, 1)) if denominator > 0 else 0.
        if similarity < .90:
            return None
        difference = _difference(first, second)
        outside = np.ones((HEIGHT, WIDTH), dtype=bool)
        for bx, by, bw, bh in boxes:
            outside &= (bx - .12 * bw > X) | (bx + 1.12 * bw <= X) | (by - .12 * bh > Y) | (by + 1.12 * bh <= Y)
        change = float(difference[area].mean())
        if change > 9. or not outside.any() or np.quantile(difference[outside], .85) > 8.:
            return None
        neighbors = [float(_difference(self._motion_at(i)[0][area], self._motion_at(j)[0][area]).mean())
                     for i, j in neighbor_pairs(left, right) if i in self.frames and j in self.frames]
        if len(neighbors) < 4 or change / max(1.5, float(np.median(neighbors))) > 2.5:
            return None
        return similarity

    def readings(self, left, right, before, after, methods=binding.ALTERNATE_CONTEXTS):
        """Try alternate witnesses only when the coarse scene calculation lacks texture."""
        coarse = face_return.scene_lower_quartile(before, after)
        if coarse is not None:
            if coarse < binding.MIN_SCENE:
                return {}
            return dict.fromkeys(methods, (binding.COARSE_CONTEXT, round(coarse, 9)))
        boxes = before.face.box(), after.face.box()
        readings = {}
        if binding.FINE_CONTEXT in methods:
            count = int(np.count_nonzero(before.scene[1] & after.scene[1]))
            value, _ = face_return_fine_scene.edge(self._tiles_at(left, boxes[0]), self._tiles_at(right, boxes[1]), count)
            if value is not None:
                readings[binding.FINE_CONTEXT] = binding.FINE_CONTEXT, round(value, 9)
        if binding.MOTION_CONTEXT in methods:
            value = self._motion_score(left, right, boxes)
            if value is not None:
                readings[binding.MOTION_CONTEXT] = binding.MOTION_CONTEXT, round(value, 9)
        return readings
