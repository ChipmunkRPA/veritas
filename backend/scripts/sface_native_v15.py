# SPDX-License-Identifier: Apache-2.0
"""Research-only SFace input contract; no identity or AI-origin classification.

Alignment math is adapted from OpenCV's Apache-2.0 face_recognize.cpp at
40738fb16ceddb5fb3fea747585f7ce6abb0605b. OpenCV copyright and license apply
to that reference: https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/LICENSE
This adaptation changes C++ into an explicit Python transcription and rejects
degenerate inputs. Full license: docs/third-party/opencv-sface-v15-Apache-2.0.txt.
Actual native pixel/feature comparisons are required before project inference.
"""
from __future__ import annotations

import cv2
import numpy as np

TARGET = np.asarray([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def validate(landmarks):
    source = np.asarray(landmarks, dtype=np.float32)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        raise ValueError('Five finite image-coordinate landmarks required')
    if np.linalg.matrix_rank(source.astype(np.float64) - source.mean(0)) < 2:
        raise ValueError('Degenerate landmarks are unassessed, never model evidence')
    return source


def cpp_alignment(landmarks):
    """Separate explicit transcription preserving the upstream float/double stages."""
    source = validate(landmarks)
    mean = np.asarray([np.float32(sum_sequential(source[:, i]) / np.float32(5)) for i in range(2)], np.float32)
    # Upstream deliberately stores rounded mean constants, not np.mean(TARGET).
    target_mean = np.asarray([56.0262, 71.9008], np.float32)
    src = source - mean
    dst = TARGET - target_mean
    covariance = np.empty((2, 2), np.float64)
    for i in range(2):
        for j in range(2):
            covariance[i, j] = sum(float(np.float32(dst[k, i] * src[k, j])) for k in range(5)) / 5
    singular, u, vt = cv2.SVDecomp(covariance)
    s = singular.reshape(-1)
    d = np.asarray([1., -1. if np.linalg.det(covariance) < 0 else 1.])
    rank = int(np.sum(s > max(s) * 2 * np.finfo(np.float32).tiny))
    if rank < 2:
        raise ValueError('Degenerate alignment covariance is unassessed')
    rotation = u @ (np.diag(d) @ vt)
    variance = sum(sum(float(np.float32(src[k, j] * src[k, j])) for k in range(5)) / 5 for j in range(2))
    scale = (1. / variance) * (s[0] * d[0] + s[1] * d[1])
    matrix = np.empty((2, 3), np.float64)
    matrix[:, :2] = rotation * scale
    matrix[:, 2] = target_mean.astype(np.float64) - scale * (rotation @ mean.astype(np.float64))
    return matrix


def sum_sequential(values):
    total = np.float32(values[0])
    for value in values[1:]:
        total = np.float32(total + value)
    return total


def manual_blob(aligned_bgr):
    """SFace API accepts BGR uint8; native DNN swaps to RGB with no outer scaling."""
    image = np.asarray(aligned_bgr)
    if image.shape != (112, 112, 3) or image.dtype != np.uint8:
        raise ValueError('Exact 112x112 BGR uint8 aligned crop required')
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], np.float32)


def face_row(landmarks):
    points = validate(landmarks)
    row = np.zeros((1, 15), np.float32)
    row[0, :4] = [0, 0, 112, 112]
    row[0, 4:14] = points.reshape(-1)
    row[0, 14] = 1
    return row
