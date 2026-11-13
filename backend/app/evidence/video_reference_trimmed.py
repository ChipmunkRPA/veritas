"""Bounded frame alignment for two separately encoded, trimmed copies."""

from __future__ import annotations

import subprocess
from contextlib import closing
from fractions import Fraction
from itertools import pairwise

import cv2
import numpy as np

from app.detectors.video.decode import DecodeError, decode_frames
from app.evidence import video_reference_local as local

MIN_LENGTH = 40
MAX_LENGTH = 100
CADENCE = frozenset({Fraction(1, 24), Fraction(1, 25)})
LAGS = (2, 4)
SCOPE = ("Compares a shorter, separately encoded video against corresponding pictures from an earlier copy. "
         "The check requires matching verified frame timing and distinctive motion on both sides. "
         "It measures local picture differences after accounting for overall brightness and contrast. "
         "Ordinary editing can produce a difference. The soundtrack is outside this check.")
BASIS = "paired_h264_trim_motion/1"


def eligible(paths, tables):
    earlier, submitted = tables
    if not MIN_LENGTH <= len(submitted) < len(earlier) <= MAX_LENGTH or any(
        table.rotation or table.truncated or table.interpolated or (table.width, table.height) != (720, 480)
        for table in tables
    ):
        return False
    step = (submitted.pts[1] - submitted.pts[0]) * submitted.tb
    if step not in CADENCE or any((b - a) * table.tb != step for table in tables for a, b in pairwise(table.pts)):
        return False
    try:
        return all(local._carrier(path) == [("h264", "yuv420p")] for path in paths)
    except (DecodeError, OSError, ValueError, subprocess.SubprocessError):
        return False


def motion(pictures, lag):
    changes = pictures[lag:] - pictures[:-lag]
    vectors = changes.reshape(len(changes), -1)
    vectors -= np.median(vectors, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(vectors * vectors, axis=1))
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6), rms


def profile(earlier, submitted):
    votes = []
    motion_pairs = [(motion(earlier, lag), motion(submitted, lag)) for lag in LAGS]
    for offset in range(len(earlier) - len(submitted) + 1):
        flanks = [[], []]
        for (first, first_rms), (second, second_rms) in motion_pairs:
            flank = len(second) // 4
            old = first[offset:offset + len(second)]
            active = np.minimum(first_rms[offset:offset + len(second)], second_rms) >= .35
            similarity = np.sum(old * second, axis=1)
            for band, span in zip(flanks, (slice(None, flank), slice(-flank, None)), strict=True):
                matches = similarity[span][active[span]]
                band.append({"active": len(matches),
                             "similarity": float(np.median(matches)) if len(matches) >= 4 else -1.})
        votes.append({"offset": offset, "flanks": flanks})
    return votes


def decide(votes):
    def means(vote):
        return tuple(sum(row["similarity"] for row in band) / len(band) for band in vote["flanks"])

    ranks = sorted(((sum(scores) / len(scores), vote["offset"], scores, vote["flanks"])
                    for vote in votes if (scores := means(vote))), key=lambda row: row[0], reverse=True)
    first, second = ranks[:2]
    score, offset, scores, flanks = first
    lead = score - second[0]
    if (score < .67 or lead < .045 or min(scores) < .60
            or any(row[2][index] - value > .03 for index, value in enumerate(scores) for row in ranks[1:])):
        return None
    return {"earlier_start": offset, "lead": round(lead, 6), "flanks": [
        {"moving_pairs": [row["active"] for row in band], "mean_similarity": round(value, 6)}
        for band, value in zip(flanks, scores, strict=True)]}


def read_pictures(path, table):
    images = []
    with closing(decode_frames(path, table, None, timeout_s=40, max_pixels=1280 * 720)) as decoded:
        for frame in decoded:
            if frame.index != len(images) or not frame.pts_verified:
                raise DecodeError("Video could not be decoded with verified picture timestamps.")
            gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
            images.append(cv2.resize(gray, (48, 32), interpolation=cv2.INTER_AREA))
    if len(images) != len(table):
        raise DecodeError("Video could not be decoded with every recorded picture.")
    return np.asarray(images, dtype=np.float32)
