# SPDX-License-Identifier: Apache-2.0
"""Separate waveform and muted-picture observations for a narrow stop-release pilot."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

RATE, TICK, HOP = 16000, 640, 80
RULES = {
    "audio": "A short high-band transient with a preceding attenuation and sustained following speech. Inspect a fixed broad publisher word neighborhood. Multiple comparable candidates remain ambiguous. Report a 15 ms sample bracket and retain nearby alternate events.",
    "visual": "Last native closed picture and first crossing before sustained reopening. Require at least two near-closed pictures and a .032 eye-normalized reopening. Use publisher word context only. Retain alternate closures separately.",
    "origin_agreement_ms": 120,
}


def audio_only(pcm, word_start):
    """Use original mono samples and coarse lexical context only."""
    signal = pcm.astype(float) / 32768
    band = sosfiltfilt(butter(3, (1250, 7200), fs=RATE, btype="bandpass", output="sos"), signal)
    usable = len(pcm) // HOP * HOP
    powers = np.mean(signal[:usable].reshape(-1, HOP) ** 2, axis=1)
    bright = np.mean(band[:usable].reshape(-1, HOP) ** 2, axis=1)
    floor = max(float(np.percentile(bright, 80)) * 0.004, 1e-9)
    speech_floor = float(np.percentile(powers, 85)) * 0.08
    ranked = []
    for i in range(max(8, (word_start - 6400) // HOP), min(len(powers) - 16, (word_start + 6400) // HOP)):
        if bright[i] < bright[i + 1] or bright[i] <= bright[i - 1]:
            continue
        high_pre = float(np.median(bright[i - 7 : i - 2]))
        power_pre = float(np.median(powers[i - 7 : i - 2]))
        high_rise = float(bright[i]) / max(floor, high_pre)
        after = float(np.median(powers[i + 3 : i + 14]))
        after_ratio = after / max(speech_floor * 0.20, power_pre, 1e-9)
        if high_rise < 3.2 or after < speech_floor or after_ratio < 1.8 or bright[i] < 4 * floor:
            continue
        score = float(np.log(high_rise) + 0.45 * np.log(after_ratio))
        ranked.append(
            {
                "center_sample": i * HOP + HOP // 2,
                "salience": score,
                "band_rise": high_rise,
                "following_to_preceding_power": after_ratio,
            }
        )
    distinct = []
    for row in sorted(ranked, key=lambda r: (-r["salience"], r["center_sample"])):
        if all(abs(row["center_sample"] - item["center_sample"]) >= 560 for item in distinct):
            distinct.append(row)
    local = [row for row in distinct if word_start - 1280 <= row["center_sample"] <= word_start + 1760]
    if not local:
        return {"status": "no_clear_burst", "nearby": distinct}
    best = local[0]
    if len(local) > 1 and best["salience"] - local[1]["salience"] < 0.65:
        return {"status": "ambiguous_bursts", "nearby": distinct}
    center = best["center_sample"]
    return {
        "status": "signal_supported_burst",
        "range_samples": [center - 120, center + 120],
        "nearby": distinct,
        "distractors": sum(abs(item["center_sample"] - center) >= 560 for item in distinct),
    }


def muted_video_only(geometry, word_start):
    """Use verified ordered picture observations and coarse lexical context only."""
    aperture = [None if v is None else float(np.median(v[:3])) for v in geometry]
    candidates = []
    for i in range(1, len(aperture) - 4):
        near = aperture[i - 1 : i + 5]
        if any(value is None for value in near):
            continue
        if not (
            aperture[i] <= 0.055 < aperture[i + 1]
            and aperture[i - 1] <= 0.075
            and max(aperture[i + 1 : i + 5]) - aperture[i] >= 0.032
        ):
            continue
        if abs((i * TICK + TICK // 2) - word_start) <= 6400:
            candidates.append([i, i + 1])
    context = [
        item for item in candidates if word_start - 1920 <= item[0] * TICK + TICK // 2 <= word_start + 2560
    ]
    if not context:
        return {"status": "no_clear_reopening", "nearby_frames": candidates}
    if len(context) > 1:
        return {"status": "ambiguous_reopening", "nearby_frames": candidates}
    return {"status": "observed_reopening", "bracket_frames": context[0], "nearby_frames": candidates}
