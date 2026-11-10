"""Study-only calibration, blind native windows and common-input current check."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import fit_native_mouth_heads_v125 as fitted

f = fitted.f
require = fitted.require
CAL_METHOD = "positive-slope-balanced-study-logit/126"
BUDGETS = (0, 1, 2)
PRIMARY_BUDGET = 2


def balanced_weights(rows, role):
    require(
        role == "confidence_cal"
        and rows
        and all(r["study_role"] == role and type(r["label"]) is int and r["label"] in (0, 1) for r in rows),
        "Only explicit calibration-role binary labels may calibrate",
    )
    counts = Counter((r["source_group"], r["label"], r["condition"]) for r in rows)
    groups = {y: {g for g, label, _ in counts if label == y} for y in (0, 1)}
    require(all(groups.values()), "Both observed calibration classes required")
    conditions = defaultdict(set)
    for g, y, c in counts:
        conditions[g, y].add(c)
    weights = np.asarray(
        [
            1
            / (
                2
                * len(groups[r["label"]])
                * len(conditions[r["source_group"], r["label"]])
                * counts[r["source_group"], r["label"], r["condition"]]
            )
            for r in rows
        ],
        np.float64,
    )
    require(np.isclose(weights.sum(), 1), "Complete balanced calibration mass required")
    return weights


def calibrate(margins, rows):
    from scipy.optimize import minimize
    from scipy.special import expit

    weights = balanced_weights(rows, "confidence_cal")
    x = np.asarray(margins, np.float64)
    require(
        x.shape == (len(rows),) and np.isfinite(x).all(),
        "One finite raw calibration margin per target required",
    )
    y = np.asarray([r["label"] for r in rows], np.float64)
    centre = float(weights @ x)
    scale = max(float(np.sqrt(weights @ ((x - centre) ** 2))), 0.05)
    standardized = (x - centre) / scale

    def objective(parameters):
        a, b = parameters
        z = a * standardized + b
        residual = weights * (expit(z) - y)
        value = weights @ (np.logaddexp(0, z) - y * z) + 0.01 * (a * a + b * b)
        gradient = np.asarray([residual @ standardized + 0.02 * a, residual.sum() + 0.02 * b])
        return float(value), gradient

    result = minimize(
        objective,
        np.asarray([1.0, 0.0]),
        jac=True,
        method="L-BFGS-B",
        bounds=((0.0, 50.0), (-50.0, 50.0)),
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    require(
        result.success and np.isfinite(result.x).all(),
        "Frozen monotone calibration did not converge; no retry",
    )
    return {
        "method": CAL_METHOD,
        "centre": centre,
        "scale": scale,
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "iterations": int(result.nit),
        "calibration_targets": len(rows),
        "scope": "Class/source/condition-balanced study probability; not customer or population accuracy.",
        "deployment_eligible": False,
    }


def probability(values, calibration):
    from scipy.special import expit

    require(
        calibration["method"] == CAL_METHOD
        and 0 <= calibration["slope"] <= 50
        and calibration["scale"] >= 0.05,
        "Exact finite monotone study calibration required",
    )
    x = np.asarray(values, np.float64)
    require(
        np.isfinite(x).all()
        and all(math.isfinite(calibration[k]) for k in ("centre", "scale", "slope", "intercept")),
        "Finite study calibration inputs required",
    )
    return expit(
        calibration["slope"] * ((x - calibration["centre"]) / calibration["scale"]) + calibration["intercept"]
    )


def windows(indices, values, table, terminal_ms):
    """Three consecutive scored centres, five actual native input pictures."""
    require(
        len(indices) == len(values)
        and len(set(indices)) == len(indices)
        and indices == sorted(indices)
        and np.isfinite(values).all(),
        "Ordered distinct complete native margins required",
    )
    by_index = dict(zip(indices, map(float, values), strict=True))
    result = []
    for i in indices:
        if not all(j in by_index for j in (i - 1, i, i + 1)):
            continue
        require(2 <= i < len(table) - 2, "Complete native five-picture visual support required")
        start, stop = i - 2, i + 3
        times = [Fraction(table.pts[j]) * table.tb for j in range(start, stop)]
        if max(b - a for a, b in zip(times, times[1:])) > Fraction(2, 25):
            continue
        end = (
            Fraction(table.pts[stop]) * table.tb
            if stop < len(table)
            else times[-1] + Fraction(terminal_ms, 1000)
        )
        result.append(
            {
                "centre_frame_index": i,
                "support_frames_half_open": [start, stop],
                "start_seconds": str(times[0]),
                "end_seconds": str(end),
                "response": float(np.mean([by_index[j] for j in (i - 1, i, i + 1)])),
                "before_frame_index": max(0, start - 1),
                "during_frame_index": i,
                "after_frame_index": min(len(table) - 1, stop),
            }
        )
    return sorted(result, key=lambda r: (-r["response"], r["centre_frame_index"]))


def operating_points(cases):
    """Frozen control-group budgets; internal research settings, not SaaS claims."""
    controls = defaultdict(list)
    for row in cases:
        require(row["study_role"] == "confidence_cal", "Only calibration groups choose research points")
        if not row["label"] and row["window_score"] is not None:
            controls[row["source_group"]].append(row["window_score"])
    require(len(controls) == 10, "All ten calibration groups need observed genuine-control windows")
    maxima = sorted((max(v) for v in controls.values()), reverse=True)
    return {
        str(budget): {
            "raw_margin": math.nextafter(maxima[budget], math.inf),
            "comparison": "strict_greater_than",
            "calibration_group_budget": budget,
            "calibration_genuine_groups": len(controls),
        }
        for budget in BUDGETS
    }


def score_case(row, learned, constructed):
    from app.detectors.video.decode import FrameTable

    indices, visual, audio = fitted.point_features(row)
    metadata = {k: row[k] for k in ("asset_index", "source_group", "study_role", "condition")}
    metadata.update(input_status=row["status"], label=int(row["condition"].startswith("ai_")))
    if not indices:
        return {
            **metadata,
            "eligible_centres": [],
            "arms": {arm: {"point_margins": [], "windows": [], "window_score": None} for arm in fitted.ARMS},
        }
    table = FrameTable.from_json(row["table"])
    representations = fitted.core.representations(visual, audio, learned["reference"])
    arms = {}
    for arm in fitted.ARMS:
        margins = fitted.margins(representations[arm], learned["heads"][arm])
        grouped = windows(
            indices, margins, table, constructed["expected"]["constructed_last_picture_duration_ms"]
        )
        arms[arm] = {
            "point_margins": margins.tolist(),
            "windows": grouped,
            "window_score": grouped[0]["response"] if grouped else None,
        }
    return {**metadata, "eligible_centres": indices, "arms": arms}


class ExhibitSink:
    def __init__(self):
        self.records = []

    def series(self, data, **metadata):
        return self._save("series", data, metadata)

    def segments(self, data, **metadata):
        return self._save("segments", data, metadata)

    def _save(self, kind, data, metadata):
        self.records.append({"kind": kind, "data": data, "metadata": metadata})
        return f"private-research-exhibit:{len(self.records)}"


class VerifiedFrameStore:
    def __init__(self, constructed, table):
        self.constructed, self.table, self.anchors = constructed, table, {}

    def iter_many(self, indices):
        for frame in f.verified_frames(self.constructed, self.table, indices):
            self.anchors[frame.index] = {
                "frame_index": frame.index,
                "pts": frame.pts,
                "time_base": self.table.time_base,
                "decoded_rgb_sha256": f.clips.inputs.speech.speech.fingerprint(frame.rgb),
            }
            yield frame.index, frame.rgb

    def native_anchors(self):
        return [self.anchors[i] for i in sorted(self.anchors)]


class NoModelLease:
    def get(self, *args, **kwargs):
        raise AssertionError("Current-check replay may not load a model")


def current_check(row, constructed):
    """Call the actual current adapter with common recorded native face inputs."""
    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.adapters.lipsync import AVInputs
    from app.detectors.video.adapters.lipsync_xcorr import LandmarkXcorrDetector
    from app.detectors.video.decode import FrameTable

    if row["status"] == "unavailable":
        return {
            "status": "unavailable",
            "reason": row["reason"],
            "assessed": False,
            "flagged": False,
            "segments": [],
            "details": {},
            "metrics": {},
        }
    require(
        f.sha(Path(constructed["file"])) == constructed["file_sha256"],
        "Exact common-input baseline clip required",
    )
    table = FrameTable.from_json(row["table"])
    frames = []
    for frame in row["frames"]:
        if frame["status"] != "ok":
            continue
        x, y, w, h = frame["face"]["box"]
        frames.append(
            {
                "frame_index": frame["frame_index"],
                "pts": frame["pts"],
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "landmarks": frame["face"]["landmarks"],
            }
        )
    decoded = decode_audio(
        Path(constructed["file"]), source_sha256=constructed["file_sha256"], max_seconds=30
    )
    sink = ExhibitSink()
    inputs = AVInputs(
        table,
        ({"track_id": "track:0", "identity": "A"},),
        {"track:0": frames},
        SimpleNamespace(status="analysed", pcm=decoded.pcm),
        VerifiedFrameStore(constructed, table),
        decoded.mapping,
    )
    ctx = SimpleNamespace(
        outputs=SimpleNamespace(av=inputs), exhibits=sink, deadline=None, lease=NoModelLease()
    )
    detector = LandmarkXcorrDetector()
    applicability = detector.applicable(ctx)
    require(applicability.ok, "Constructed inputs must reach the actual current check")
    result = detector.run(ctx)
    require(result.status in {"ok", "not_applicable"}, "Unexpected failed current check; preserve stage")
    segments = [r for item in sink.records if item["kind"] == "segments" for r in item["data"]]
    return {
        "status": result.status,
        "reason": result.reason or result.observed,
        "assessed": result.metrics.get("windows", 0) > 0,
        "flagged": result.aggregate is not None,
        "segments": segments,
        "metrics": result.metrics,
        "details": result.details,
        "unit_scores": [asdict(u) for u in result.unit_scores],
        "exhibits": sink.records,
        "aggregate_raw": None if result.aggregate is None else result.aggregate.raw,
        "scope": "Actual correlation adapter; common recorded dense native faces, not serving tracker/full-pipeline parity. Informational/no-window outputs are not original-file verdicts.",
    }


def merged(intervals):
    result = []
    for left, right in sorted((Fraction(a), Fraction(b)) for a, b in intervals):
        require(left < right, "Positive source interval required")
        if result and left <= result[-1][1]:
            result[-1][1] = max(result[-1][1], right)
        else:
            result.append([left, right])
    return result


def interval_metrics(intervals, truth):
    spans = merged(intervals)
    length = sum((b - a for a, b in spans), Fraction())
    intersection = Fraction()
    if truth is not None:
        a, b = map(Fraction, truth)
        require(a < b, "Positive known source edit interval required")
        intersection = sum((max(Fraction(), min(y, b) - max(x, a)) for x, y in spans), Fraction())
        union = length + (b - a) - intersection
        iou = float(intersection / union)
    else:
        iou = 0.0
    return {
        "flagged_duration_seconds": float(length),
        "edit_overlap_seconds": float(intersection),
        "outside_edit_flagged_seconds": float(length - intersection),
        "temporal_union_iou": iou,
        "localized_iou_0_2": truth is not None and iou >= 0.2,
        "merged_source_intervals": [[str(a), str(b)] for a, b in spans],
    }


def reliability(probabilities, labels, weights=None):
    p, y = np.asarray(probabilities, np.float64), np.asarray(labels, np.float64)
    require(
        p.shape == y.shape
        and p.ndim == 1
        and np.isfinite(p).all()
        and np.isfinite(y).all()
        and ((p >= 0) & (p <= 1)).all()
        and np.isin(y, [0, 1]).all(),
        "Complete study probabilities and labels required",
    )
    if not len(p):
        return {"n": 0, "brier": None, "log_loss": None, "ece_10": None}
    w = np.ones(len(p)) / len(p) if weights is None else np.asarray(weights, np.float64)
    require(
        w.shape == p.shape and np.isfinite(w).all() and (w > 0).all(),
        "Positive finite reliability weights required",
    )
    w = w / w.sum()
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.0
    for b in range(10):
        keep = bins == b
        if keep.any():
            ece += abs(float(w[keep] @ (p[keep] - y[keep])))
    return {
        "n": len(p),
        "brier": float(w @ ((p - y) ** 2)),
        "log_loss": float(-w @ (y * np.log(safe) + (1 - y) * np.log1p(-safe))),
        "ece_10": ece,
    }
