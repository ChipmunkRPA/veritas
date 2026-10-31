"""Multi-face tracking and identity clustering (ARCHITECTURE §3.1 `tracks`; VF §8.1; PRD DC-03).

- **Detections:** YuNet on every 2nd low-resolution frame; boxes and landmarks in original pixels.
- **Size:** a face whose short side is ≥ 64 px (original pixels) starts a track; a matched face ≥ 0.8 × 64 px keeps it,
  so a face near the threshold does not fragment its track.
- **Matching:** per detection frame, a Hungarian assignment (`scipy.optimize.linear_sum_assignment`) on the cost
  0.5 × (1 − IoU) + 0.5 × cosine distance of the appearance embeddings. A pair is admissible when IoU ≥ 0.1 and the
  cosine distance to the track's mean embedding is ≤ 0.5 (IoU alone when there are no embeddings).
- **Gaps:** a track survives 3 detection frames without a match (6 frames when detection runs on every 2nd frame).
- **Shots:** every track ends at a shot boundary. Tracks with fewer than 3 detections are dropped.
- **Interpolation:** frames between two detections of a track get linearly interpolated boxes and landmarks, with
  `det_score` None.
- **Identity:** average-linkage agglomerative clustering of the tracks' mean embeddings (cosine distance ≤ threshold),
  with a cannot-link between tracks whose frame spans overlap. Labels A, B, … follow first appearance. This is appearance
  similarity from a general-purpose image embedding, not face recognition.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.detectors.video.decode import FrameTable

MIN_TRACK_PX = 64.0
KEEP_RATIO = 0.8
MAX_MISSED = 3
IOU_GATE = 0.1
EMBED_GATE = 0.5
MIN_DETECTIONS = 3
# Measured on 13 sample faces with DINOv2-small (2026-09-13): same face under ¼ scale, darkening, JPEG q60 or a wider
# margin, cosine distance median 0.027, max 0.277; different people (78 pairs) min 0.433, median 0.755.
IDENTITY_THRESHOLD = 0.35
_BLOCKED = 1e6


@dataclass(frozen=True)
class Detection:
    frame_index: int
    bbox: tuple[float, float, float, float]  # x, y, w, h in original pixels
    score: float
    landmarks: tuple[tuple[float, float], ...] = ()
    embedding: np.ndarray | None = None  # L2-normalised


@dataclass
class Track:
    id: int
    shot: int
    detections: list[Detection] = field(default_factory=list)
    missed: int = 0
    _embedding_sum: np.ndarray | None = None

    @property
    def last(self) -> Detection:
        return self.detections[-1]

    @property
    def first_frame(self) -> int:
        return self.detections[0].frame_index

    @property
    def last_frame(self) -> int:
        return self.detections[-1].frame_index

    def add(self, detection: Detection) -> None:
        self.detections.append(detection)
        self.missed = 0
        if detection.embedding is not None:
            e = np.asarray(detection.embedding, dtype=np.float64)
            self._embedding_sum = e if self._embedding_sum is None else self._embedding_sum + e

    def mean_embedding(self) -> np.ndarray | None:
        if self._embedding_sum is None:
            return None
        return self._embedding_sum / max(float(np.linalg.norm(self._embedding_sum)), 1e-12)

    def face_px(self) -> list[float]:
        return [min(d.bbox[2], d.bbox[3]) for d in self.detections]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(1.0 - np.dot(np.asarray(a, np.float64), np.asarray(b, np.float64)))


class Tracker:
    def __init__(self, *, min_px: float = MIN_TRACK_PX, keep_ratio: float = KEEP_RATIO, max_missed: int = MAX_MISSED,
                 iou_gate: float = IOU_GATE, embed_gate: float = EMBED_GATE, min_detections: int = MIN_DETECTIONS):
        self.min_px, self.keep_px, self.max_missed = min_px, keep_ratio * min_px, max_missed
        self.iou_gate, self.embed_gate, self.min_detections = iou_gate, embed_gate, min_detections
        self.active: list[Track] = []
        self.finished: list[Track] = []
        self.dropped = 0
        self.shot = 0
        self._next_id = 0

    def _close(self, track: Track) -> None:
        if len(track.detections) >= self.min_detections:
            self.finished.append(track)
        else:
            self.dropped += 1

    def new_shot(self, shot: int) -> None:
        for track in self.active:
            self._close(track)
        self.active, self.shot = [], shot

    def _cost(self, track: Track, detection: Detection) -> float:
        overlap = iou(track.last.bbox, detection.bbox)
        if overlap < self.iou_gate:
            return _BLOCKED
        distance = cosine_distance(track.mean_embedding(), detection.embedding)
        if distance is None:
            return 1.0 - overlap
        if distance > self.embed_gate:
            return _BLOCKED
        return 0.5 * (1.0 - overlap) + 0.5 * distance

    def step(self, detections: Sequence[Detection]) -> None:
        """One detection frame (every detection frame must be stepped, including frames without faces)."""
        candidates = [d for d in detections if min(d.bbox[2], d.bbox[3]) >= self.keep_px]
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if self.active and candidates:
            cost = np.asarray([[self._cost(t, d) for d in candidates] for t in self.active])
            for i, j in zip(*linear_sum_assignment(cost), strict=True):
                if cost[i, j] < _BLOCKED:
                    self.active[i].add(candidates[j])
                    matched_tracks.add(int(i))
                    matched_dets.add(int(j))
        survivors: list[Track] = []
        for i, track in enumerate(self.active):
            if i not in matched_tracks:
                track.missed += 1
                if track.missed > self.max_missed:
                    self._close(track)
                    continue
            survivors.append(track)
        for j, detection in enumerate(candidates):
            if j not in matched_dets and min(detection.bbox[2], detection.bbox[3]) >= self.min_px:
                track = Track(self._next_id, self.shot)
                track.add(detection)
                self._next_id += 1
                survivors.append(track)
        self.active = survivors

    def finish(self) -> list[Track]:
        self.new_shot(self.shot)
        return sorted(self.finished, key=lambda t: (t.first_frame, t.id))


def _lerp(a: Sequence[float], b: Sequence[float], w: float) -> list[float]:
    return [float(x + (y - x) * w) for x, y in zip(a, b, strict=True)]


def _frame(table: FrameTable, index: int, bbox: Sequence[float], landmarks: Sequence[Sequence[float]] | None,
           score: float | None) -> dict[str, Any]:
    x, y, w, h = (round(float(v), 2) for v in bbox)
    return {"pts": table.pts[index], "frame_index": index, "bbox": {"x": x, "y": y, "w": w, "h": h},
            "landmarks": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in landmarks] if landmarks else None,
            "det_score": None if score is None else round(float(score), 4)}


def track_frames(track: Track, table: FrameTable) -> list[dict[str, Any]]:
    """Detected frames plus linearly interpolated frames between consecutive detections (`det_score` None)."""
    out: list[dict[str, Any]] = []
    for a, b in pairwise(track.detections):
        out.append(_frame(table, a.frame_index, a.bbox, a.landmarks, a.score))
        for k in range(a.frame_index + 1, b.frame_index):
            w = (k - a.frame_index) / (b.frame_index - a.frame_index)
            landmarks = ([_lerp(p, q, w) for p, q in zip(a.landmarks, b.landmarks, strict=True)]
                         if len(a.landmarks) == len(b.landmarks) and a.landmarks else None)
            out.append(_frame(table, k, _lerp(a.bbox, b.bbox, w), landmarks, None))
    last = track.last
    out.append(_frame(table, last.frame_index, last.bbox, last.landmarks, last.score))
    return out


def identity_letters(n: int) -> list[str]:
    out = []
    for k in range(n):
        label, k1 = "", k + 1
        while k1:
            k1, r = divmod(k1 - 1, 26)
            label = chr(65 + r) + label
        out.append(label)
    return out


def cluster_identities(tracks: Sequence[Track], *, threshold: float = IDENTITY_THRESHOLD) -> list[str | None]:
    """Identity label per track (None for a track without embeddings)."""
    embeddings = [t.mean_embedding() for t in tracks]
    clusters: list[list[int]] = [[i] for i, e in enumerate(embeddings) if e is not None]

    def overlap(i: int, j: int) -> bool:
        return tracks[i].first_frame <= tracks[j].last_frame and tracks[j].first_frame <= tracks[i].last_frame

    while len(clusters) > 1:
        best: tuple[float, int, int] | None = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if any(overlap(i, j) for i in clusters[a] for j in clusters[b]):
                    continue
                distance = float(np.mean([cosine_distance(embeddings[i], embeddings[j]) for i in clusters[a] for j in clusters[b]]))
                if distance <= threshold and (best is None or distance < best[0]):
                    best = (distance, a, b)
        if best is None:
            break
        _, a, b = best
        clusters[a] = sorted(clusters[a] + clusters[b])
        del clusters[b]
    clusters.sort(key=lambda c: min((tracks[i].first_frame, tracks[i].id) for i in c))
    labels: list[str | None] = [None] * len(tracks)
    for cluster, letter in zip(clusters, identity_letters(len(clusters)), strict=True):
        for i in cluster:
            labels[i] = letter
    return labels


class DinoEmbedder:
    """DINOv2-small appearance embedding of face crops: 224 × 224 bicubic, ImageNet mean and std, `pooler_output`
    (the layer-normed class token), L2-normalised."""

    SIZE = 224
    MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)
    STD = np.asarray((0.229, 0.224, 0.225), np.float32)

    def __init__(self, model: Any, device: Any):
        self.model, self.device = model, device

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        import cv2
        import torch

        if not crops:
            return np.zeros((0, 384), np.float32)
        batch = np.stack([cv2.resize(np.ascontiguousarray(c), (self.SIZE, self.SIZE), interpolation=cv2.INTER_CUBIC)
                          for c in crops]).astype(np.float32) / 255.0
        batch = (batch - self.MEAN) / self.STD
        dtype = next(self.model.parameters()).dtype
        x = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).to(self.device, dtype)
        with torch.inference_mode():
            out = self.model(pixel_values=x).pooler_output.float().cpu().numpy()
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
