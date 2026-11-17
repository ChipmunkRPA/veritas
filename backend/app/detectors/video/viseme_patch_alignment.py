"""Locate audible speech with weaker visual support than its two flanks."""

from __future__ import annotations

import numpy as np

GROUPS = ("closure", "open", "rounded", "spread", "lower_lip")
ARMS = ("relative", "alternative")
LAGS = np.arange(-2240, 2241, 320)
WIDTHS = (5760, 7040, 8320, 9600, 10880, 12160)
STEP = 640
FLANK = 11520


def support_matrix(readings, sample_times, kept, events):
    mass = np.asarray([row[2] for row in events])
    clock = np.asarray([row[1] for row in events])
    labels = np.asarray([GROUPS.index(row[0]) for row in events])
    candidates = clock[:, None] - LAGS[None, :]
    after = np.searchsorted(sample_times, candidates).clip(1, len(sample_times) - 1)
    before = after - 1
    valid = ((candidates >= sample_times[0]) & (candidates <= sample_times[-1]) & kept[before] & kept[after])
    share = ((candidates - sample_times[before]) / (sample_times[after] - sample_times[before])).clip(0, 1)
    values = readings[before] + share[:, :, None] * (readings[after] - readings[before])
    nearest = np.where(share <= .5, before, after)
    group_mass = np.take_along_axis(values, labels[:, None, None], axis=2)[..., 0]
    peers = values[..., :len(GROUPS)].copy()
    np.put_along_axis(peers, labels[:, None, None], 0., axis=2)
    prior = np.median(readings[kept, :len(GROUPS)], axis=0)[labels]
    relative = np.log((group_mass + .05) / (prior[:, None] + .05))
    alternative = np.log((group_mass + .05) / (peers.max(axis=2) + .05))
    # Low acoustic support contributes less to either score.
    reliability = np.sqrt(mass[:, None])
    return clock, labels, nearest, valid, values, (relative * reliability, alternative * reliability)


def observed_context(kept):
    result = np.zeros_like(kept)
    if len(kept) >= 5:
        result[2:-2] = np.lib.stride_tricks.sliding_window_view(kept, 5).all(axis=1)
    return result


def locate(readings, sample_times, kept, events, samples):
    events = [row for row in events if row[0] in GROUPS and row[2] >= .35]
    kept = observed_context(kept)
    if (len(events) < 7 or kept.sum() < 28 or len(sample_times) != len(kept)
            or not np.all(np.diff(sample_times) > 0)):
        return {}
    times, labels, nearest, valid, probabilities, scores = support_matrix(readings, sample_times, kept, events)
    outcomes = {}
    for name, energies in zip(ARMS, scores, strict=True):
        possible = []
        any_lag = valid.sum(axis=0)
        stable = any_lag >= max(6, np.ceil(.72 * len(events)))
        if not np.any(stable):
            continue
        all_means = np.where(stable, (energies * valid).sum(axis=0) / any_lag.clip(1), -np.inf)
        for width in WIDTHS:
            for start in range(4480, samples - width - 3839, STEP):
                end = start + width
                left = (times < start) & (times >= start - FLANK)
                middle = (times >= start) & (times < end)
                right = (times >= end) & (times < end + FLANK)
                sections = np.array((left, middle, right))[:, :, None] & valid
                counts = sections.sum(axis=1)
                supported = stable & (counts.min(axis=0) >= 2)
                if not supported.any():
                    continue
                before, inside, after = (energies * sections).sum(axis=1) / counts.clip(1)
                flank = np.minimum(before, after)
                lane = max(np.flatnonzero(supported), key=lambda j: (flank[j], all_means[j], -abs(int(LAGS[j]))))
                neighbors = slice(max(0, lane - 2), lane + 3)
                inner = max(inside[neighbors][counts[1, neighbors] >= 2])
                if flank[lane] < (-.25 if name == "alternative" else -.10) or inner >= -.07:
                    continue
                stronger = np.flatnonzero(sections[1, :, lane] & (energies[:, lane] <= -.05))
                if not len(stronger):
                    continue
                during = min(stronger, key=lambda j: (energies[j, lane], times[j]))
                left_event = np.flatnonzero(sections[0, :, lane])[-1]
                right_event = np.flatnonzero(sections[2, :, lane])[0]
                brackets = [int(nearest[i, lane]) for i in (left_event, during, right_event)]
                if not brackets[0] < brackets[1] < brackets[2]:
                    continue
                compatibility = float(flank[lane] - inner)
                item = {"score": round(compatibility, 7), "span": [start, end], "frames": brackets,
                        "lag_samples": int(LAGS[lane]), "checked_events": list(map(int, counts[:, lane])),
                        "speech_sample": int(times[during]), "heard": GROUPS[labels[during]],
                        "strongest_visual_group": GROUPS[int(probabilities[during, lane, :len(GROUPS)].argmax())],
                        "local_support": round(float(inner), 7), "flank_support": round(float(flank[lane]), 7)}
                possible.append(item)
        if possible:
            outcomes[name] = min(possible, key=lambda row: (-row["score"], row["span"]))
    return outcomes
