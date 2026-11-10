"""Score meanings and packet selection; no model or private calibration imports.

These checks validate recorded claims. They do not recalculate model confidence.
"""

from __future__ import annotations

import math
import re

REFERENCE_SHA256 = "e7de202c42b80fc81d8ab7faa0ce3538c6af5a4c28a9830e4b9dfbcaa6b63641"
SEMANTICS = "speech-window-responses/2"


def reference_claim(probability: float) -> dict:
    return {
        "kind": "balanced-reference-ai-estimate",
        "calibration_sha256": REFERENCE_SHA256,
        "reference_prior_ai": 0.5,
        "scope": "complete-4s-speech-window",
        "direction": "ai" if probability > 0.5 else "original" if probability < 0.5 else "inconclusive",
    }


def observation(window: dict) -> dict:
    reference = window.get("response_reference")
    return {
        "label": "Speech AI reference estimate" if reference else "Voice-check window response",
        "value": window["score"],
        "unit": "calibrated AI reference estimate" if reference else "uncalibrated score",
        **{key: window[key] for key in ("signal_refs", "unit_ref", "raw_output_sha256")},
        **({"response_reference": reference} if reference else {}),
    }


def candidates(windows: list[dict]) -> list[tuple[tuple[int, int], list[dict]]]:
    grouped: dict[tuple[int, int], list[dict]] = {}
    for window in windows:
        grouped.setdefault((window["start_sample"], window["end_sample"]), []).append(window)
    # Calibrated original-leaning and inconclusive windows also need inspectable
    # support. Keep the historical raw-score selection unchanged.
    selected = [
        (span, values)
        for span, values in grouped.items()
        if any(v.get("response_reference") or v["score"] >= 0.5 for v in values)
    ]
    return sorted(selected, key=lambda item: (-max(v["score"] for v in item[1]), item[0]))


def validate(content: dict, *, required: bool = False) -> None:
    coverage = content.get("coverage")
    if coverage is None:
        return
    declared = coverage.get("response_semantics")
    windows = coverage["windows"]
    has_reference = any("response_reference" in w for w in windows)
    if not (required or declared is not None or has_reference):
        return  # Historical evidence keeps its original format and bytes.
    if declared != SEMANTICS:
        raise ValueError("Speech window response meanings are missing or unsupported")
    source = content["source"]
    for window in windows:
        if (
            type(window.get("score")) not in (int, float)
            or not math.isfinite(window["score"])
            or not 0 <= window["score"] <= 1
            or type(window.get("start_sample")) is not int
            or type(window.get("end_sample")) is not int
            or not 0 <= window["start_sample"] < window["end_sample"] <= source["samples"]
            or not re.fullmatch(r"[a-f0-9]{64}", str(window.get("raw_output_sha256", "")))
            or not window.get("signal_refs")
            or not isinstance(window.get("unit_ref"), str)
        ):
            raise ValueError("Invalid speech window response or source coordinates")
        if "response_reference" not in window:
            continue
        reference = window["response_reference"]
        timeline = source.get("timeline")
        if (
            not isinstance(reference, dict)
            or reference != reference_claim(window["score"])
            or type(reference.get("reference_prior_ai")) not in (int, float)
            or source.get("sample_rate") != 16000
            or source.get("samples") != 64000
            or window["start_sample"] != 0
            or window["end_sample"] != 64000
            or not isinstance(timeline, list)
            or len(timeline) != 1
            or timeline[0].get("start_sample") != 0
            or timeline[0].get("end_sample") != 64000
            or type(timeline[0].get("source_pts")) is not int
            or source.get("timeline_time_base") != "1/16000"
        ):
            raise ValueError("Calibrated speech response is outside its complete source-bound window")
    selected = candidates(windows)
    if len(content["packets"]) != min(len(selected), 6) or content["omitted_windows"] != max(
        0, len(selected) - 6
    ):
        raise ValueError("Speech packet selection differs from its response meanings")
    for packet, (span, values) in zip(content["packets"], selected[:6], strict=True):
        if packet["interval"] != {"start_sample": span[0], "end_sample": span[1]} or packet[
            "observations"
        ] != [observation(w) for w in values]:
            raise ValueError("Speech observation labels differ from their recorded response meanings")
