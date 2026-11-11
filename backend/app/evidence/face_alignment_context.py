"""Describe the landmark inputs already bound to an appearance review range."""
from __future__ import annotations

import math
from statistics import median

from app.core.canonical import FLOAT_DP
from app.evidence.face_interval_binding import APPEARANCE_METHOD

LABELS = {role: f"Median eye spacing {role} the range" for role in ("before", "during", "after")}


def measure(sequence: dict) -> list[dict]:
    by_index = {frame["frame_index"]: frame for frame in sequence["frame_inputs"]}
    measurements = []
    for role, label in LABELS.items():
        indices = {center + offset for center in sequence["centers"][role] for offset in (-1, 0, 1)}
        spacing = []
        for index in indices:
            frame = by_index[index]
            # Use the coordinates that survive canonical evidence serialization.
            left, right = [[round(value, FLOAT_DP) for value in eye] for eye in frame["landmarks"][:2]]
            distance = math.hypot(right[0] - left[0], right[1] - left[1])
            width = round(frame["bbox"]["w"], FLOAT_DP)
            if width <= 0:
                raise ValueError("Face tracking context requires a recorded positive face width")
            spacing.append(distance / width)
        measurements.append({"label": label, "value": round(100 * median(spacing), 2), "unit": "% of face-box width"})
    return measurements


def validate(packet: dict) -> None:
    sequence = packet.get("face_sequence") or {}
    expected = measure(sequence) if sequence.get("method") == APPEARANCE_METHOD else []
    measurements = packet.get("measurements", [])
    if not isinstance(measurements, list) or not all(isinstance(row, dict) for row in measurements):
        raise ValueError("Video measurements must be a list of recorded values")
    recorded = [row for row in measurements if row.get("label") in LABELS.values()]
    if recorded != expected:
        raise ValueError("Face tracking context differs from the recorded landmark inputs")
