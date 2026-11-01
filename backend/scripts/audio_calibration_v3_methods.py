"""Isolated audio research methods; never imported by serving code.

The speech-band view is a sensitivity probe, not a denoised original or a second
truth label. Only the unmodified, production-decoded view has a calibrated
probability. An unstable low AI score cannot clear a recording as original.
All fit/calibration callers must enforce independent identity splits externally.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt
from scipy.special import expit

SR = 16000
METHOD = {
    "version": "audio-noise-sensitivity/3.0.0",
    "sample_rate": SR,
    "source_decode": "production mono 16000 Hz signed16 PCM, converted to float32",
    "filter": "whole-clip zero-phase Butterworth bandpass",
    "filter_order": 4,
    "filter_band_hz": [100.0, 3800.0],
    "windows": "original-view Silero gate; production full-profile windows and arithmetic mean",
    "filtered_gate": False,
    "aggregate_score_decimal_places": 6,
    "logit_clip": 1e-6,
    "original_veto_log_odds_delta": math.log(2),
    "veto_direction": "original_only",
    "probability_source": "fit-only monotone Platt mapping of original-view raw aggregate",
    "lean_threshold": 0.8,
    "class_set_alpha": 0.1,
    "deployment_eligible": False,
}


def valid_probability(value: Any) -> bool:
    return isinstance(value, (float, int, np.number)) and not isinstance(value, bool) and bool(
        np.isfinite(value) and 0 <= value <= 1
    )


def clipped_logit(value: float) -> float:
    if not valid_probability(value):
        raise ValueError("A finite probability in [0, 1] is required")
    p = float(np.clip(value, METHOD["logit_clip"], 1 - METHOD["logit_clip"]))
    return math.log(p) - math.log1p(-p)


def band_view(pcm: np.ndarray) -> np.ndarray:
    """Fixed probe applied after production decoding, with no source/SNR input.

    No re-quantization, loudness normalization or clipping is added here. The
    frozen publisher's window preprocessing still applies within predict().
    """
    values = np.asarray(pcm, dtype=np.float32)
    if values.ndim != 1 or len(values) < SR or not np.isfinite(values).all():
        raise ValueError("At least one second of finite mono 16 kHz PCM is required")
    sos = butter(METHOD["filter_order"], METHOD["filter_band_hz"],
                 btype="bandpass", fs=SR, output="sos")
    return sosfiltfilt(sos, values).astype(np.float32)


def sensitivity(raw: float | None, band: float | None) -> dict:
    """Absolute mean-score log-odds change, not a noise or SNR estimate."""
    valid = valid_probability(raw) and valid_probability(band)
    delta = abs(clipped_logit(raw) - clipped_logit(band)) if valid else None
    return {
        "available": bool(valid),
        "absolute_log_odds_delta": delta,
        "odds_change_factor": math.exp(delta) if valid else None,
        "original_clearance_stable": bool(valid and delta <= METHOD["original_veto_log_odds_delta"]),
    }


def _aggregate(scores: list[float]) -> float | None:
    if not scores:
        return None
    if not all(valid_probability(p) for p in scores):
        raise ValueError("Classifier returned a nonfinite or out-of-range score")
    return round(float(np.mean(scores)), METHOD["aggregate_score_decimal_places"])


def score_views(pcm: np.ndarray, voice_model: Any, speech_activity: Any) -> dict:
    """Score both views with one original-source gate and identical windows.

    Pass PCM returned by app.detectors.audio.decode.decode_audio; this function
    does not decode files or fit parameters. Models remain on the CPU. The raw
    result follows PellaVoiceDetector's full-profile windows/mean exactly.
    Missing speech windows abstain in both views; a failed probe cannot clear
    an original. PANNS sound tags are descriptive and are not used here.
    """
    import functools

    from app.detectors.audio.content import window_in_speech
    from app.detectors.video.audio import MAX_VOICE_WINDOWS, VOICE_WINDOW_S, voice_windows
    from app.engine.detector import Profile
    from app.engine.profiles import SAMPLING

    values = np.asarray(pcm, dtype=np.float32)
    filtered = band_view(values)
    spans = speech_activity.spans(values)
    params = SAMPLING[Profile("full")]
    windows = voice_windows(values, params.voice_hop_s or VOICE_WINDOW_S,
                            max_windows=params.audio_windows or MAX_VOICE_WINDOWS,
                            keep=functools.partial(window_in_speech, spans))
    intervals = [(round(start * SR), round(start * SR) + len(chunk)) for start, chunk in windows]
    raw_scores = [float(p) for p in voice_model.predict([chunk for _, chunk in windows])] if windows else []
    if len(raw_scores) != len(windows):
        raise ValueError("Original prediction count differs from selected source windows")
    raw = _aggregate(raw_scores)
    probe_error = None
    try:
        band_scores = [float(p) for p in voice_model.predict([filtered[a:b] for a, b in intervals])] if windows else []
        if len(band_scores) != len(windows):
            raise ValueError("Probe prediction count differs from selected source windows")
        band = _aggregate(band_scores)
    except Exception as error:
        band_scores, band = [], None
        probe_error = f"{type(error).__name__}: {error}"
    return {
        "method": METHOD["version"], "device": "cpu", "sample_rate": SR,
        "source_pcm_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest(),
        "band_pcm_sha256": hashlib.sha256(filtered.astype("<f4").tobytes()).hexdigest(),
        "status": "ok" if windows else "not_applicable", "probe_error": probe_error,
        "speech_spans_s": [list(s) for s in spans],
        "raw": raw, "band": band,
        "raw_unrounded": float(np.mean(raw_scores)) if raw_scores else None,
        "band_unrounded": float(np.mean(band_scores)) if band_scores else None,
        "windows": [{"start_sample": a, "end_sample": b,
                     "raw": raw_scores[i], "band": band_scores[i] if band_scores else None}
                    for i, (a, b) in enumerate(intervals)],
        "sensitivity": sensitivity(raw, band),
    }


def fit_platt(rows: list[dict], score_key: str = "raw") -> dict:
    """Monotone source- and class-balanced fit; no calibration/test rows allowed.

    Required keys: split='fit', base_id, label (0/1), and score_key. Conditions
    of each source share its weight; the two classes each receive half weight.
    Platt target smoothing counts distinct assessed original sources, not
    augmented variants. The map is a 50:50 research-reference probability.
    """
    if any(row.get("split") != "fit" for row in rows):
        raise ValueError("Probability fitting accepts fit rows only")
    selected = [row for row in rows if valid_probability(row.get(score_key))]
    source_classes: dict[str, int] = {}
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in selected:
        label, source = row["label"], row["base_id"]
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("Labels must be integer 0 or 1")
        if source in source_classes and source_classes[source] != label:
            raise ValueError("One original source cannot have conflicting labels")
        source_classes[source] = label
        counts[(label, source)] += 1
    sizes = [sum(label == c for label in source_classes.values()) for c in (0, 1)]
    if min(sizes) < 2:
        raise ValueError("At least two assessed original sources per class are required")
    x = np.array([clipped_logit(r[score_key]) for r in selected])
    y = np.array([r["label"] for r in selected])
    target = np.where(y == 1, (sizes[1] + 1) / (sizes[1] + 2), 1 / (sizes[0] + 2))
    weights = np.array([.5 / sizes[r["label"]] / counts[(r["label"], r["base_id"])] for r in selected])

    def objective(parameters):
        z = parameters[0] * x + parameters[1]
        residual = weights * (expit(z) - target)
        return float(np.sum(weights * (np.logaddexp(0, z) - target * z))), np.array([
            np.sum(residual * x), np.sum(residual)])

    result = minimize(objective, [1., 0.], method="L-BFGS-B", jac=True,
                      bounds=[(0., None), (None, None)], options={"ftol": 1e-12, "gtol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Probability fit did not converge: {result.message}")
    return {"slope": float(result.x[0]), "intercept": float(result.x[1]),
            "source_n_by_class": sizes, "assessed_rows": len(selected),
            "objective": float(result.fun), "iterations": int(result.nit),
            "score_key": score_key, "reference_class_prior": .5,
            "method": "source-and-class-balanced monotone Platt with source-count target smoothing"}


def probability(raw: float | None, parameters: dict) -> float | None:
    if not valid_probability(raw):
        return None
    slope, intercept = float(parameters["slope"]), float(parameters["intercept"])
    if slope < 0 or not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Invalid monotone calibration parameters")
    return float(expit(slope * clipped_logit(raw) + intercept))


def calibrate_class_set(rows: list[dict], probability_key: str = "p") -> dict:
    """Worst nonconformity per independent identity, all its sources/variants.

    This uses cal only. Missing original scores retain infinity, represented by
    None, and are never dropped. Exchangeability and adequate calibration
    identities are requirements, not facts guaranteed by this function.
    """
    groups: dict[int, dict[str, list]] = {0: defaultdict(list), 1: defaultdict(list)}
    identities: dict[str, int] = {}
    for row in rows:
        if row.get("split") != "cal":
            raise ValueError("Class-set calibration accepts cal rows only")
        label, identity = row["label"], row["speaker"]
        if label not in (0, 1) or isinstance(label, bool):
            raise ValueError("Labels must be integer 0 or 1")
        if identity in identities and identities[identity] != label:
            raise ValueError("An identity cannot have conflicting labels")
        identities[identity] = label
        p = row.get(probability_key)
        groups[label][identity].append(None if not valid_probability(p) else p if label == 0 else 1 - p)
    return {str(label): [{"speaker": identity,
                          "maximum": None if any(x is None for x in values) else max(values),
                          "rows": len(values)} for identity, values in sorted(by_identity.items())]
            for label, by_identity in groups.items()}


def class_set(p: float | None, guard: dict) -> dict:
    if not valid_probability(p) or min(len(guard.get("0", [])), len(guard.get("1", []))) < 9:
        return {"included": [0, 1], "p_values": {"0": None, "1": None}, "adequate": False}
    p_values = {}
    for label in (0, 1):
        records = guard[str(label)]
        nonconformity = p if label == 0 else 1 - p
        p_values[str(label)] = (1 + sum(r["maximum"] is None or r["maximum"] >= nonconformity for r in records)) / (len(records) + 1)
    return {"included": [label for label in (0, 1) if p_values[str(label)] > METHOD["class_set_alpha"]],
            "p_values": p_values, "adequate": True}


def decide(p: float | None, raw: float | None, band: float | None, guard: dict) -> dict:
    """Return a baseline conclusion and the fixed, original-only safety veto.

    Probability and conformal class-set p-values are distinct quantities. The
    sensitivity probe can only remove an original conclusion; it cannot turn
    one class into the other or modify the probability used by proper scores.
    """
    confidence = METHOD["lean_threshold"]
    lean = ("ai" if p >= confidence else "original" if p <= 1 - confidence + 1e-12 else "inconclusive") if valid_probability(p) else "inconclusive"
    classes = class_set(p, guard)
    expected = [1] if lean == "ai" else [0]
    baseline = lean if lean != "inconclusive" and classes["included"] == expected else "inconclusive"
    stability = sensitivity(raw, band)
    veto = baseline == "original" and not stability["original_clearance_stable"]
    return {"probability_ai": p, "probability_only_decision": lean,
            "identity_guard_decision": baseline,
            "decision": "inconclusive" if veto else baseline,
            "original_stability_veto": veto,
            "class_set": classes, "sensitivity": stability}
