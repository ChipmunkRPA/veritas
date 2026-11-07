"""Candidate sampling with explicit native clock support; no model inference.

Keep the publisher's index/stride/center-crop arithmetic, while requiring eight
actual frames and recording decoded timestamps. Zero padding and guessed FPS
are not evidence from the source and cause abstention instead. Independent
pixel/clock replay remains required before predictive evaluation is accepted.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


class UnsupportedSource(ValueError):
    """Missing/invalid input support, never evidence of an original file."""


def sample_plan(fps, total):
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0
            or isinstance(total, bool) or type(total) is not int or total < 1):
        raise UnsupportedSource("Finite positive native FPS and complete frame count required")
    duration = total / fps
    target = 2 if duration >= 4 else math.ceil(8 / max(duration, 1.))
    stride = max(1, int(round(fps / target)))
    sampled = list(range(0, total, stride))[:32]
    if len(sampled) < 8:
        raise UnsupportedSource("Fewer than eight actual sampled frames; zero padding refused")
    start = (len(sampled) - 8) // 2
    return {"fps": fps, "metadata_duration_s": duration, "stride": stride,
            "sampled_indices": sampled, "window_start_in_samples": start,
            "selected_indices": sampled[start:start + 8]}


def crop_box(height, width):
    if height < 1 or width < 1:
        raise UnsupportedSource("Nonempty decoded RGB frame required")
    if width > height:
        trim = int(width * .10)
        return (trim, 0, width - trim, height)
    trim = int(height * .10)
    return (0, trim, width, height - trim)


def preprocess(frame_bgr):
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise UnsupportedSource("Original uint8 BGR frame required")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = crop_box(height, width)
    resized = cv2.resize(rgb[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_LINEAR)
    mean = np.array([.485, .456, .406], np.float32) * 255.
    std = np.array([.229, .224, .225], np.float32) * 255.
    tensor = ((resized.astype(np.float32) - mean) / std).transpose(2, 0, 1)
    return np.ascontiguousarray(tensor), {
        "native_size": [width, height], "crop_box_xyxy": [x0, y0, x1, y1],
        "native_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "resized_rgb_sha256": hashlib.sha256(resized.tobytes()).hexdigest(),
        "normalized_tensor_sha256": hashlib.sha256(tensor.astype("<f4").tobytes()).hexdigest(),
    }


def native_clock(path, total):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
                             "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)],
                            capture_output=True, text=True, timeout=120, check=True)
    frames = json.loads(result.stdout)["frames"]
    if len(frames) != total:
        raise UnsupportedSource("Decoded clock and native frame-count correspondence differs")
    try:
        clock = [float(row["best_effort_timestamp_time"]) for row in frames]
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedSource("Missing decoded frame timestamps") from exc
    if any(not math.isfinite(t) for t in clock) or any(b <= a for a, b in zip(clock, clock[1:])):
        raise UnsupportedSource("Finite strictly increasing decoded timestamps required")
    return clock


def read_native(path: Path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise UnsupportedSource("Source video cannot be opened")
        fps, count = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count) or count != int(count):
            raise UnsupportedSource("Integral native frame count required")
        plan = sample_plan(fps, int(count))
        clock = native_clock(path, int(count))
        selected = set(plan["selected_indices"])
        tensors, frames = [], []
        for index in range(max(selected) + 1):
            ok, bgr = capture.read()
            if not ok:
                raise UnsupportedSource("Native decode ended before a selected frame")
            if index in selected:
                tensor, trace = preprocess(bgr)
                tensors.append(tensor)
                frames.append({"native_index": index, "pts_s": clock[index],
                               "relative_to_first_decoded_pts_s": clock[index] - clock[0], **trace})
        if len(tensors) != 8:
            raise UnsupportedSource("All eight native frames are required")
        batch = np.ascontiguousarray(np.stack(tensors)[None], dtype=np.float32)
        return batch, {"schema": "chipmunk.video-demamba-native-input/13", "plan": plan,
            "source_frame_count": int(count), "decoded_clock_entries": len(clock),
            "first_decoded_pts_s": clock[0], "frames": frames,
            "synthetic_padding_frames": 0, "tensor_shape": list(batch.shape),
            "tensor_sha256": hashlib.sha256(batch.astype("<f4").tobytes()).hexdigest(),
            "scope": "Eight sampled pictures in one center window within the first32 sampled frames. PTS association is by decoder ordinal; independent pixel/clock replay is still required. This is not full-clip coverage, causal pixel evidence or an inferred edit boundary."}
    finally:
        capture.release()
