"""Face detection with OpenCV YuNet (ONNX). Returns boxes, 5-point landmarks and confidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from app.core.config import settings
from app.detectors.common.download import fetch
from app.detectors.common.registry import registry

YUNET_TOP_K = 50
# YuNet mutates native input buffers. Serialize every caller, including v1 and concurrent analyses.
# Separate YuNet instances also serialize; the critical section excludes image resizing and result conversion.
_YUNET_LOCK = Lock()


@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float
    landmarks: list[tuple[float, float]] = field(default_factory=list)  # right eye, left eye, nose, mouth R, mouth L

    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "score": round(self.score, 3), "landmarks": self.landmarks}


def load_yunet(path: Path) -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), score_threshold=0.7, nms_threshold=0.3, top_k=YUNET_TOP_K)


def _load():
    return load_yunet(fetch(settings.yunet_url, "face_detection_yunet_2023mar.onnx"))


def detect_faces(bgr: np.ndarray, max_side: int = 1280, *, detector: cv2.FaceDetectorYN | None = None,
                 min_size: int | None = None) -> list[Face]:
    """Detect faces in a BGR image. Downscales large images for speed, rescales boxes back.

    v1 uses the shared registry detector and `settings.face_min_size`; v2 adapters pass both explicitly."""
    if detector is None:
        with registry.borrow("yunet", _load) as borrowed:
            try:
                return detect_faces(bgr, max_side, detector=borrowed, min_size=min_size)
            finally:
                del borrowed
    det = detector
    min_size = settings.face_min_size if min_size is None else min_size
    h, w = bgr.shape[:2]
    scale = 1.0
    img = bgr
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    with _YUNET_LOCK:
        det.setInputSize((img.shape[1], img.shape[0]))
        _, out = det.detect(img)
        # Own the result before another request can overwrite native output storage.
        if out is not None:
            out = out.copy()
    faces: list[Face] = []
    if out is None:
        return faces
    for row in out:
        x, y, fw, fh = (row[:4] / scale).tolist()
        lm = [(float(row[4 + 2 * i] / scale), float(row[5 + 2 * i] / scale)) for i in range(5)]
        x, y = max(0, int(x)), max(0, int(y))
        fw, fh = int(min(fw, w - x)), int(min(fh, h - y))
        if fw >= min_size and fh >= min_size:
            faces.append(Face(x, y, fw, fh, float(row[14]), lm))
    faces.sort(key=lambda f: f.w * f.h, reverse=True)
    return faces


def crop_box(shape: tuple[int, ...], face: Face, margin: float = 0.3) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the square crop around the face with margin, clipped to the image."""
    h, w = shape[:2]
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    side = max(face.w, face.h) * (1 + 2 * margin)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return x0, y0, x1, y1


def crop_face(rgb: np.ndarray, face: Face, margin: float = 0.3) -> np.ndarray:
    """Square crop around the face with margin (deepfake detectors are trained on such crops)."""
    x0, y0, x1, y1 = crop_box(rgb.shape, face, margin)
    return rgb[y0:y1, x0:x1]


def iou(a: Face, b: Face) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


PORTRAIT_AREA = 0.30  # face box covering >= 30 % of the image => tight portrait; feed the whole image
MODEL_INPUT = 224


def face_input(rgb: np.ndarray, face: Face) -> tuple[np.ndarray, bool, bool]:
    """Input for face-oriented classifiers.

    Returns (array, whole_image_used, upscaled). GAN/face detectors are trained on tightly framed
    portraits; re-cropping a portrait and resampling it destroys the low-level fingerprint they rely
    on, so portraits are passed whole. Small faces are cropped and the caller down-weights them when
    the crop is smaller than the model input (it will be upscaled).
    """
    h, w = rgb.shape[:2]
    if face.w * face.h >= PORTRAIT_AREA * h * w:
        return rgb, True, min(h, w) < MODEL_INPUT
    crop = crop_face(rgb, face)
    return crop, False, min(crop.shape[:2]) < MODEL_INPUT
