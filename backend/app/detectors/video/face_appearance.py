# SPDX-License-Identifier: Apache-2.0
"""Qualified same-clip appearance transitions; never AI or identity probabilities.

Alignment transcription adapted from Apache-2.0 OpenCV face_recognize.cpp at
40738fb16ceddb5fb3fea747585f7ce6abb0605b. OpenCV copyright and license apply:
https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/LICENSE
See docs/third-party/opencv-sface-v15-Apache-2.0.txt. Alignment and window math
are copied unchanged from the qualified v15/v16 research implementation.
"""
from __future__ import annotations

import cv2
import numpy as np

METHOD = "face-appearance-change/1"
REVIEW_POINT = 0.11243900750879454
FLANK_PENALTY = .35

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

def face_row(landmarks):
    points = validate(landmarks)
    row = np.zeros((1, 15), np.float32)
    row[0, :4] = [0, 0, 112, 112]
    row[0, 4:14] = points.reshape(-1)
    row[0, 14] = 1
    return row

def unit(vector):
    value = np.asarray(vector, np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if value.shape != (128,) or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError('A finite nonzero native 128-value feature is required')
    return value / norm

def distance(a, b):
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(an, bn) <= 1e-12:
        raise ValueError('Degenerate pooled face appearance cannot support a window')
    return float(1 - np.clip(np.dot(a, b) / (an * bn), -1., 1.))

def windows(centers, tracks, features, table):
    """Same source triplets, track membership, cadence and windows as the current cue."""
    normalized = {i: unit(value) for i, value in features.items()}
    result = []
    for track_id, samples in tracks.items():
        tracked = {sample.frame_index for sample in samples}
        blocks = []
        for i in sorted(centers):
            group = (i - 1, i, i + 1)
            if all(k in tracked and k in normalized for k in group) and table.seconds(i + 1) - table.seconds(i - 1) <= .13:
                blocks.append((i, np.mean([normalized[k] for k in group], axis=0)))
        times = np.asarray([table.seconds(i) for i, _ in blocks], np.float64)
        for start in range(2, len(blocks) - 4):
            for end in range(start + 3, len(blocks) - 1):
                span = times[end - 1] - times[start]
                if span > 2.6:
                    break
                if span < .52 or np.max(np.diff(times[start - 2:end + 2])) > .55:
                    continue
                left, middle, right = [np.mean([value for _, value in part], axis=0) for part in
                    (blocks[start - 2:start], blocks[start:end], blocks[end:end + 2])]
                l, r, flanks = distance(middle, left), distance(middle, right), distance(left, right)
                response = min(l, r) - FLANK_PENALTY * flanks
                result.append({'track_id': track_id, 'start_s': float(times[start]), 'end_s': float(times[end - 1]),
                    'response': response, 'middle_to_before': l, 'middle_to_after': r, 'flank_change': flanks,
                    'centers': [i for i, _ in blocks[start - 2:end + 2]]})
    return sorted(result, key=lambda w: (-w['response'], w['start_s'], w['track_id']))[:12]
