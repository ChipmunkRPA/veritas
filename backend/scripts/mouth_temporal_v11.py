"""Fixed49-variable video-window classifier; raw evidence, never probability."""

from __future__ import annotations

import hashlib
import math

import cv2
import numpy as np

from scripts import mouth_context_v10 as context
from scripts.mouth_probe_v6 import source_weights

METHOD = "mouth-temporal-linear/11.0"
MESH_POINTS = (13, 14, 78, 308, 33, 263, 1)
GEOMETRY = ("aperture", "mouth_eye_ratio", "yaw_proxy", "pitch_proxy", "roll", "face_scale")
QUALITY = ("gray_mean", "gray_std", "laplacian_variance")
HEAD_STATS = (
    "middle",
    "flank",
    "relative",
    "flank_difference",
    "middle_std",
    "relative_standardized",
    "std_ratio",
)
FEATURE_NAMES = tuple(
    [f"{h}.{s}" for h in ("v8", "v9") for s in HEAD_STATS]
    + [f"{s}.{v}" for s in GEOMETRY for v in ("middle", "middle_delta", "flank_delta", "middle_std")]
    + [f"{s}.{v}" for s in QUALITY for v in ("middle", "middle_delta", "middle_std")]
    + ["duration_s", "middle_centers"]
)


def geometry_values(points, width, height):
    p = np.asarray(points, np.float64)
    if p.shape != (7, 3) or not np.isfinite(p).all() or width <= 0 or height <= 0:
        raise ValueError("Seven finite native mesh points required")
    xy = p[:, :2] * [width, height]
    upper, lower, left, right, eye1, eye2, nose = xy
    eye = eye2 - eye1
    distance = float(np.linalg.norm(eye))
    mouth = float(np.linalg.norm(right - left))
    if min(distance, mouth) < 1:
        raise ValueError("Insufficient measured eye/mouth resolution")
    axis = eye / distance
    offset = nose - (eye1 + eye2) / 2
    result = [
        np.linalg.norm(upper - lower) / mouth,
        mouth / distance,
        offset @ axis / distance,
        offset @ [-axis[1], axis[0]] / distance,
        math.atan2(eye[1], eye[0]),
        distance / width,
    ]
    return [float(v) for v in result]


def quality_values(chip):
    gray = cv2.cvtColor(chip, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return [
        float(gray.mean() / 255),
        float(gray.std() / 255),
        float(cv2.Laplacian(gray, cv2.CV_64F).var() / (255**2)),
    ]


def window_key(window):
    return (window["track_id"], window["start_s"], window["end_s"], tuple(window["centers"]))


def window_features(heads, measurements):
    """Consume only measured video evidence; case labels/metadata are absent."""
    indexes = {h: {b["frame_index"]: b["margin"] for b in heads[h]["model_inputs"]} for h in ("v8", "v9")}
    windows = heads["v8"]["context_windows_all"]
    if {window_key(w) for w in windows} != {window_key(w) for w in heads["v9"]["context_windows_all"]}:
        raise ValueError("Both frozen heads must enumerate identical complete windows")
    by_index = {m["frame_index"]: m for m in measurements}
    if (
        len(by_index) != len(measurements)
        or set(indexes["v8"]) != set(indexes["v9"])
        or not set(indexes["v8"]) <= set(by_index)
    ):
        raise ValueError("Missing, duplicate or different native measurements")
    result = []
    for w in sorted(windows, key=lambda w: (w["start_s"], w["end_s"], w["track_id"], w["centers"])):
        centers = w["centers"]
        vector = []
        for h in ("v8", "v9"):
            values = np.array([np.mean([indexes[h][i + k] for k in (-1, 0, 1)]) for i in centers])
            left, middle, right = values[:2], values[2:-2], values[-2:]
            l, m, r = float(left.mean()), float(middle.mean()), float(right.mean())
            flank = np.r_[left, right]
            relative = min(m - l, m - r) - 0.25 * abs(l - r)
            sd = float(middle.std())
            vector.extend(
                [
                    m,
                    float(flank.mean()),
                    relative,
                    abs(l - r),
                    sd,
                    relative / (0.1 + sd),
                    sd / (0.1 + float(flank.std())),
                ]
            )
        measured = np.array(
            [np.mean([by_index[i + k]["values"] for k in (-1, 0, 1)], axis=0) for i in centers]
        )
        if measured.shape != (len(centers), 9) or not np.isfinite(measured).all():
            raise ValueError("Invalid source geometry/quality traces")
        for j in range(9):
            left, mid, right = measured[:2, j], measured[2:-2, j], measured[-2:, j]
            flank = np.r_[left, right]
            vector.extend([float(mid.mean()), abs(float(mid.mean() - flank.mean()))])
            if j < 6:
                vector.append(abs(float(left.mean() - right.mean())))
            vector.append(float(mid.std()))
        vector.extend([w["end_s"] - w["start_s"], len(centers) - 4])
        if len(vector) != 49 or not np.isfinite(vector).all():
            raise ValueError("Finite49-variable feature vector required")
        values = np.asarray(vector, np.float64)
        result.append(
            {
                **{k: w[k] for k in ("track_id", "start_s", "end_s", "centers")},
                "features": values.tolist(),
                "features_float64_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            }
        )
    return result


def train(rows):
    from sklearn.linear_model import LogisticRegression

    if not rows or any(r["role"] != "fit" for r in rows) or len({r["group"] for r in rows}) != 16:
        raise ValueError("All and only sixteen fitting actors required")
    for group in {r["group"] for r in rows}:
        positives = {r["condition"] for r in rows if r["group"] == group and r["label"]}
        if positives != {"ai-redraw", "ai-retime", "muse-ai-aligned", "muse-ai-earlier"}:
            raise ValueError("Every AI fitting condition needs an eligible positive window")
    x = np.asarray([r["features"] for r in rows], np.float64)
    if x.shape != (len(rows), 49) or not np.isfinite(x).all():
        raise ValueError("Invalid complete fitting matrix")
    weights = source_weights(rows)
    mean = np.average(x, weights=weights, axis=0)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, weights=weights, axis=0)), 1e-3)
    model = LogisticRegression(C=0.1, solver="lbfgs", tol=1e-8, max_iter=2000, random_state=2718)
    model.fit((x - mean) / scale, [r["label"] for r in rows], sample_weight=weights)
    if int(model.n_iter_.max()) >= 2000:
        raise ValueError("Fixed temporal learner did not converge")
    return {
        "method": METHOD,
        "feature_names": FEATURE_NAMES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "fit_groups": sorted({r["group"] for r in rows}),
        "fit_windows": len(rows),
        "iterations": int(model.n_iter_.max()),
        "response_kind": "raw signed margin; not probability",
        "deployment_eligible": False,
    }


def margins(windows, head):
    if list(head["feature_names"]) != list(FEATURE_NAMES):
        raise ValueError("Feature order changed")
    x = np.asarray([w["features"] for w in windows], np.float64)
    if x.shape != (len(windows), 49) or not np.isfinite(x).all():
        raise ValueError("Missing/invalid inference features")
    return ((x - np.asarray(head["mean"])) / np.asarray(head["scale"])) @ np.asarray(head["coef"]) + head[
        "intercept"
    ]


def selected_signal(absolute, absolute_point, temporal, temporal_point):
    return context.selected_signal(absolute, absolute_point, temporal, temporal_point)
