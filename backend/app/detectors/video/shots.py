"""Shot boundaries on the low-resolution stream (ARCHITECTURE §3.1 `shots`; VF §8.1).

Two measures between consecutive frames, both on a 64 px wide HSV thumbnail (OpenCV HSV, H in 0–179):
- **content:** mean absolute HSV difference on the 0–255 scale, the measure of PySceneDetect's ContentDetector (whose
  default threshold is 27);
- **histogram:** 1 − intersection of normalised 8 × 4 × 4 HSV histograms.

A cut needs both measures over their thresholds and at least 0.5 s (pts time) since the previous cut. A fast pan or
motion raises the content difference but keeps the colour histogram; a flash frame is suppressed by the minimum length.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np

from app.detectors.video.decode import DecodedFrame, FrameTable

METHOD = "hsv-content-histogram-1"
CONTENT_THRESHOLD = 27.0
HISTOGRAM_THRESHOLD = 0.35
MIN_SHOT_S = 0.5
THUMB_W = 64


def _features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    thumb = cv2.resize(np.ascontiguousarray(rgb), (THUMB_W, max(1, round(THUMB_W * h / w))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    return hsv.astype(np.float32), hist / max(float(hist.sum()), 1.0)


class ShotDetector:
    def __init__(self, table: FrameTable, *, content_threshold: float = CONTENT_THRESHOLD,
                 histogram_threshold: float = HISTOGRAM_THRESHOLD, min_shot_s: float = MIN_SHOT_S):
        self.times = table.times()
        self.n = len(table)
        self.content_threshold, self.histogram_threshold, self.min_shot_s = content_threshold, histogram_threshold, min_shot_s
        self.boundaries: list[dict[str, Any]] = []
        self.frames_seen = 0
        self._prev: tuple[np.ndarray, np.ndarray] | None = None
        self._shot_start = 0

    def update(self, index: int, rgb: np.ndarray) -> bool:
        """Feed frames in table order; True when `index` starts a new shot."""
        hsv, hist = _features(rgb)
        prev, self._prev = self._prev, (hsv, hist)
        self.frames_seen += 1
        if prev is None or prev[0].shape != hsv.shape:
            return False
        content = float(np.abs(hsv - prev[0]).mean())
        histogram = 1.0 - float(np.minimum(hist, prev[1]).sum())
        long_enough = self.times[index] - self.times[self._shot_start] >= self.min_shot_s
        if content >= self.content_threshold and histogram >= self.histogram_threshold and long_enough:
            self.boundaries.append({"frame_index": index, "time_s": round(float(self.times[index]), 6),
                                    "content": round(content, 3), "histogram": round(histogram, 4)})
            self._shot_start = index
            return True
        return False

    def ranges(self) -> list[tuple[int, int]]:
        starts = [0, *(b["frame_index"] for b in self.boundaries)]
        return [(a, b) for a, b in zip(starts, [*starts[1:], self.n], strict=True) if b > a] if self.n else []

    def as_json(self) -> dict[str, Any]:
        return {"method": METHOD, "thresholds": {"content": self.content_threshold, "histogram": self.histogram_threshold,
                                                 "min_shot_s": self.min_shot_s, "thumbnail_width_px": THUMB_W},
                "frames_scanned": self.frames_seen, "boundaries": self.boundaries, "ranges": [list(r) for r in self.ranges()]}


def detect_shots(frames: Iterable[DecodedFrame], table: FrameTable, **kw: Any) -> ShotDetector:
    detector = ShotDetector(table, **kw)
    for frame in frames:
        detector.update(frame.index, frame.rgb)
    return detector
