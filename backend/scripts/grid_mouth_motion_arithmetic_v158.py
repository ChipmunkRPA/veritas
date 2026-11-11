"""Shared-support geometry readout for the finished GRID mouth redraws."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

STEP = Fraction(1, 25)
LAGS = tuple(range(-12, 13))
EARLIER, SOURCE = -8, 0
TARGETS = (EARLIER, SOURCE)
EDIT_FRAMES, MIN_FRAMES, MIN_PAIRS = 20, 16, 12


@dataclass(frozen=True)
class MouthShape:
    """Four normalized lip shape coordinates and native eye separation."""

    q: tuple[float, float, float, float]
    eye_px: float

    def __post_init__(self):
        if len(self.q) != 4 or not all(math.isfinite(x) and x >= 0 for x in self.q):
            raise ValueError("Four finite nonnegative lip coordinates required")
        if not math.isfinite(self.eye_px) or self.eye_px < 12:
            raise ValueError("Finite native eye separation of at least twelve pixels required")


Trajectory = Mapping[Fraction, MouthShape | None]


def edit_times(times: Sequence[Fraction]) -> tuple[Fraction, ...]:
    result = tuple(times)
    if len(result) != EDIT_FRAMES or any(type(t) is not Fraction for t in result):
        raise ValueError("Twenty exact rational edit positions required")
    if any(b - a != STEP for a, b in zip(result, result[1:])):
        raise ValueError("Edit positions must retain the observed 25 Hz clock")
    return result


def adjacent(times):
    return [(i, i + 1) for i, (a, b) in enumerate(zip(times, times[1:])) if b - a == STEP]


def supported(variant: Trajectory, original: Trajectory, times, lags):
    return tuple(
        t for t in times
        if variant.get(t) is not None and all(original.get(t + lag * STEP) is not None for lag in lags)
    )


def rms(a):
    return float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))


def geometry(trajectory, times):
    return np.asarray([trajectory[t].q for t in times], dtype=np.float64)


def centered(a):
    return a - np.mean(a, axis=0, dtype=np.float64)


def readout(variant, original, support, lags):
    pairs = adjacent(support)
    result = {"supported_positions": len(support), "adjacent_pairs": len(pairs), "measured": False}
    if len(support) < MIN_FRAMES or len(pairs) < MIN_PAIRS:
        return result
    y = geometry(variant, support)
    yc = centered(y)
    first, second = np.asarray(pairs).T
    vy = (y[second] - y[first]) / float(STEP)
    references = {k: geometry(original, [t + k * STEP for t in support]) for k in lags}
    rc = {k: centered(r) for k, r in references.items()}
    resolution = 1.0 / float(np.median([original[t].eye_px for t in support]))
    movement = rms(yc)
    target_movement = {k: rms(rc[k]) for k in TARGETS}
    target_separation = rms(rc[SOURCE] - rc[EARLIER])
    distances = {
        k: {
            "centered_shape_rms": rms(yc - rc[k]),
            "absolute_shape_rms": rms(y - r),
            "velocity_rms_per_second": rms(vy - (r[second] - r[first]) / float(STEP)),
        }
        for k, r in references.items()
    }
    preference = distances[SOURCE]["centered_shape_rms"] - distances[EARLIER]["centered_shape_rms"]
    return {
        **result,
        "measured": True,
        "one_pixel_dead_band": resolution,
        "output_movement_rms": movement,
        "target_movement_rms": target_movement,
        "target_separation_rms": target_separation,
        "weak_reference": min(*target_movement.values(), target_separation) <= resolution,
        "earlier_preference": preference,
        "lags": distances,
    }


def classify(curve):
    if not curve["measured"]:
        return {"status": "unavailable", "near_best_lags": []}
    tolerance = curve["one_pixel_dead_band"]
    c = {k: v["centered_shape_rms"] for k, v in curve["lags"].items()}
    near = [k for k in LAGS if c[k] <= min(c.values()) + tolerance]
    outcome = {"status": "other_or_poor_match", "near_best_lags": near}
    if curve["weak_reference"]:
        return {**outcome, "status": "weak_reference"}
    if curve["output_movement_rms"] <= tolerance:
        return {**outcome, "status": "static_output"}
    for k, other, name in ((SOURCE, EARLIER, "source_time"), (EARLIER, SOURCE, "earlier")):
        if (
            all(abs(k - best) <= 1 for best in near)
            and c[other] - c[k] > tolerance
            and c[k] + tolerance < curve["target_movement_rms"][k]
        ):
            return {**outcome, "status": name + "_compatible"}
    if near[0] == LAGS[0] or near[-1] == LAGS[-1]:
        return {**outcome, "status": "search_boundary"}
    if near[-1] - near[0] > 2 or abs(c[SOURCE] - c[EARLIER]) <= tolerance:
        return {**outcome, "status": "ambiguous"}
    return outcome


def boundary_velocity(trajectory: Trajectory, start: Fraction, end: Fraction):
    if end - start != STEP:
        raise ValueError("Boundary transition must be adjacent on the native clock")
    a, b = trajectory.get(start), trajectory.get(end)
    if a is None or b is None:
        return None
    return math.dist(a.q, b.q) / float(end - start)


def measure(variant: Trajectory, original: Trajectory, times: Sequence[Fraction]):
    edit = edit_times(times)
    full_support = supported(variant, original, edit, LAGS)
    target_support = supported(variant, original, edit, TARGETS)
    curve = readout(variant, original, full_support, LAGS)
    target_curve = readout(variant, original, target_support, TARGETS)
    return {
        **classify(curve),
        "full_support_times": [str(t) for t in full_support],
        "target_support_times": [str(t) for t in target_support],
        "curve": curve,
        "target_only_contrast": target_curve,
        "entry_velocity_norm": boundary_velocity(variant, edit[0] - STEP, edit[0]),
        "exit_velocity_norm": boundary_velocity(variant, edit[-1], edit[-1] + STEP),
    }


def paired_preference(earlier: Trajectory, comparator: Trajectory, original: Trajectory, times):
    edit = edit_times(times)
    a, b = set(supported(earlier, original, edit, LAGS)), set(supported(comparator, original, edit, LAGS))
    common = tuple(t for t in edit if t in a and t in b)
    left, right = (readout(arm, original, common, LAGS) for arm in (earlier, comparator))
    result = {"supported_positions": len(common), "adjacent_pairs": len(adjacent(common)), "status": "unavailable"}
    if not left["measured"] or not right["measured"]:
        return result
    delta = left["one_pixel_dead_band"]
    difference = left["earlier_preference"] - right["earlier_preference"]
    return {
        **result,
        "status": "gain" if difference > delta else "loss" if difference < -delta else "within_dead_band",
        "shared_support_times": [str(t) for t in common],
        "weak_reference": left["weak_reference"],
        "one_pixel_dead_band": delta,
        "earlier_preference_difference": difference,
    }
