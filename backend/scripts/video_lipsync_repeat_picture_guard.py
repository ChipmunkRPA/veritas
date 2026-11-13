# SPDX-License-Identifier: Apache-2.0
"""Compare native picture context guards for repeated speech in one short upload."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from scripts import video_lipsync_repeat_motion as base

DESIGN = {
    "nomination": "Freeze the same two-band acoustic, speech, geometry and motion requirements as repeat-motion.",
    "changes": "Compare a fixed whole-picture amplitude gate with two direct native picture context guards. The replacement rejects repeated non-mouth pixels. It preserves the existing native mouth guard unless both measured motion sequences disagree strongly and multiple observed non-mouth picture regions disagree.",
    "context": "Compare four 96x72-resolution picture quadrants. Mask both measured mouth rectangles, leaving independent upper, lower and lateral picture evidence. At least two textured quadrants must have <.975 normalized correlation and mean absolute difference >=1.1 for at least half the paired pictures. Never use an edit location or truth for either arm.",
    "point": "Same frozen grid and zero-ordinary development choice as the parent. The signed-motion corroboration for overriding static mouth similarity is fixed at .90.",
    "assessment": "Retained actors are now diagnostic. The transfer bank contains new sound/picture operations on 16 separate archived actors with known decoded source bytes. The same sources and operations are in the parent study, so this is a comparative transfer even though the new context measurements are first opened after freezing.",
    "advance": "Same held requirements as the parent. Any passing result on these consumed transfer sources would still require separate confirmation before serving. Neither result carries AI confidence.",
}


def context_diff(left, right):
    first, second = [cv2.resize(r[5], (96, 72), interpolation=cv2.INTER_AREA).astype(np.float32) for r in (left, right)]
    valid = np.ones((72, 96), bool)
    for x, y in (left[6], right[6]):
        x, y = x // 5, y // 5
        valid[max(0, y-7):y+8, max(0, x-11):x+12] = False
    disagree = 0
    for y in (slice(0, 36), slice(36, 72)):
        for x in (slice(0, 48), slice(48, 96)):
            mask = valid[y, x]
            a, b = first[y, x][mask], second[y, x][mask]
            if len(a) < 800 or min(a.std(), b.std()) < 6:
                continue
            if np.mean(np.abs(a - b)) < 1.1:
                continue
            a, b = a - a.mean(), b - b.mean()
            if np.dot(a, b) < .975 * np.linalg.norm(a) * np.linalg.norm(b):
                disagree += 1
    return disagree >= 2


def measurement(first, second, observed):
    pairs = base.consecutive_pairs(first, second, observed)
    if not pairs:
        return None
    left, right = [[observed[p[side]] for p in pairs] for side in (0, 1)]
    pose = np.linalg.norm(np.stack([a[2] - b[2] for a, b in zip(left, right, strict=True)]), axis=1)
    if np.quantile(pose, .8) > 7 or 2 * sum(context_diff(a, b) for a, b in zip(left, right, strict=True)) < len(pairs):
        return None
    result = base.signed_distance(left, right, pairs)
    if result is None or result[0] > .90:
        return result
    static = [base.prior.translation_correlation((a[5], a[6]), (b[5], b[6]), 12, 8, 5)
              for a, b in zip(left, right, strict=True)]
    upper = [base.prior.translation_correlation((a[5], a[7]), (b[5], b[7]), 25, 15, 5)
             for a, b in zip(left, right, strict=True)]
    similar = sum(m >= .90 or (m >= .82 and m >= u) for m, u in zip(static, upper, strict=True))
    return result if 3 * similar < 2 * len(pairs) else None


def source_roster(args):
    original = base.source_roster(args)
    return original | {"design": DESIGN,
                       "bindings": original["bindings"] | {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}


def main():
    base.main(freeze_roster=source_roster, measure=measurement)


if __name__ == "__main__":
    main()
