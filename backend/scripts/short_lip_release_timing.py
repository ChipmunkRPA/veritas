# SPDX-License-Identifier: Apache-2.0
"""Match independent mouth-release and acoustic-onset events without a marked time."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

RATE = 16000
HOP = 80
MATCH = 0.072
STEP_S = 0.04
SHIFT_S = (0.16, 0.20, 0.24, 0.28, 0.32)
MAX_EVENTS = 16
SPEC = {
    "audio": "Full-clip short high-band transient after attenuation, with sustained following speech. Use no transcript or timing hint.",
    "video": "Full-clip stateless native lip reopening after a near-closed run. Require exact picture ticks and report both source pictures. Use no target hint.",
    "correspondence": "Ordered one-to-one matching with event strength and sub-picture residual. Search a shared constant offset, then at most one alternate offset over a short bounded run with matched events on both sides. Preserve ambiguous assignments.",
    "maximum_clip_s": 3.4,
    "output": "Evidence of localized relative timing only. Good alignment is compatible with edited content.",
}


@dataclass(frozen=True)
class Event:
    lo: int
    hi: int
    t: float
    strength: float


def acoustic(pcm):
    n = len(pcm) // HOP * HOP
    signal = pcm[:n].astype(float)
    band = sosfiltfilt(butter(3, (1250, 7200), fs=RATE, btype="bandpass", output="sos"), signal)
    powers = np.mean(signal.reshape(-1, HOP) ** 2, axis=1)
    high = np.mean(band.reshape(-1, HOP) ** 2, axis=1)
    floor = max(float(np.percentile(high, 80)) * 0.004, 1e-9)
    speech_floor = float(np.percentile(powers, 85)) * 0.08
    rows = []
    for i in range(8, len(powers) - 16):
        if high[i] < high[i + 1] or high[i] <= high[i - 1]:
            continue
        high_pre = float(np.median(high[i - 7 : i - 2]))
        pre = float(np.median(powers[i - 7 : i - 2]))
        rise = float(high[i]) / max(floor, high_pre)
        after = float(np.median(powers[i + 3 : i + 14]))
        contrast = after / max(speech_floor * 0.20, pre, 1e-9)
        if rise < 3.2 or after < speech_floor or contrast < 1.8 or high[i] < 4 * floor:
            continue
        center = i * HOP + HOP // 2
        salience = float(np.log(rise) + 0.45 * np.log(contrast))
        rows.append(Event(center - 120, center + 120, center / RATE, min(1.0, salience / 7)))
    chosen = []
    for event in sorted(rows, key=lambda r: (-r.strength, r.t)):
        if all(abs(other.t - event.t) >= 0.035 for other in chosen):
            chosen.append(event)
        if len(chosen) >= MAX_EVENTS:
            break
    return sorted(chosen, key=lambda e: e.t)


def visual(apertures, times):
    result = []
    for i in range(1, len(times) - 4):
        near = apertures[i - 1 : i + 5]
        if any(v is None for v in near):
            continue
        gain = max(apertures[i + 1 : i + 5]) - apertures[i]
        if apertures[i - 1] <= 0.075 and apertures[i] <= 0.055 < apertures[i + 1] and gain >= 0.032:
            strength = min(1.0, gain / 0.10) * 0.55 + min(1.0, max(0.0, 0.06 - apertures[i]) / 0.06) * 0.45
            result.append(Event(i, i + 1, (times[i] + times[i + 1]) / 2, strength))
    return result[:MAX_EVENTS]


def pair_ordered(mouth, sound, expected):
    # Mismatched or absent events can be skipped, but every used event is unique and ordered.
    @cache
    def best(i, j):
        if i == len(mouth) or j == len(sound):
            return 0.0, ()
        options = [best(i + 1, j), best(i, j + 1)]
        residual = abs(sound[j].t - mouth[i].t - expected[i])
        if residual <= MATCH:
            future, other = best(i + 1, j + 1)
            gain = 0.85 + 0.25 * (mouth[i].strength + sound[j].strength) - 0.55 * residual / MATCH
            options.append((future + gain, ((i, j), *other)))
        return max(options, key=lambda r: (r[0], len(r[1]), r[1]))

    return best(0, 0)


def assess(mouth, sound):
    if len(mouth) < 3 or len(sound) < 3:
        return {"measured": False, "reason": "fewer_than_three_events", "candidate": None}
    # The baseline spans ordinary whole-clip delays. No clip's known marker restricts the search.
    baselines = [i * STEP_S for i in range(-8, 9)]
    normal = [pair_ordered(mouth, sound, (lag,) * len(mouth)) for lag in baselines]
    strongest = max(s for s, _ in normal)
    rows = []
    for base, (score, baseline_pairs) in zip(baselines, normal, strict=True):
        if len(baseline_pairs) < 2 or score < strongest - 0.55:
            continue
        for first in range(1, len(mouth) - 1):
            for end in range(first + 1, len(mouth)):
                if mouth[end - 1].t - mouth[first].t > 0.72:
                    break
                for delta in (*SHIFT_S, *(-x for x in SHIFT_S)):
                    expected = (
                        (base,) * first + (base + delta,) * (end - first) + (base,) * (len(mouth) - end)
                    )
                    changed, pairs = pair_ordered(mouth, sound, expected)
                    before = [(i, j) for i, j in pairs if i < first]
                    during = [(i, j) for i, j in pairs if first <= i < end]
                    after = [(i, j) for i, j in pairs if i >= end]
                    if not (before and during and after) or changed <= strongest:
                        continue
                    if not all(abs(sound[j].t - mouth[i].t - base) >= 0.12 for i, j in during):
                        continue
                    inside = [sound[j].t - mouth[i].t for i, j in during]
                    flank = [sound[j].t - mouth[i].t for i, j in (*before, *after)]
                    actual = float(np.median(inside) - np.median(flank))
                    if abs(actual - delta) > MATCH or abs(actual) < 0.12:
                        continue
                    rows.append(
                        {
                            "score": changed - strongest,
                            "shift_s": actual,
                            "pairs": pairs,
                            "inside": during,
                            "outside": [before[-1], after[0]],
                            "event_seconds": [
                                float(np.median([mouth[i].t for i, _ in during])),
                                float(np.median([sound[j].t for _, j in during])),
                            ],
                        }
                    )
    if not rows:
        return {"measured": True, "reason": "no_local_pairing", "candidate": None}
    selected = max(rows, key=lambda r: (r["score"], len(r["inside"]), -abs(r["shift_s"])))
    alternatives = [
        r["score"]
        for r in rows
        if r["inside"] != selected["inside"] or r["shift_s"] * selected["shift_s"] < 0
    ]
    selected["alternative_margin"] = selected["score"] - max(alternatives, default=0.0)
    return {"measured": True, "reason": "local_pairing", "candidate": selected}
