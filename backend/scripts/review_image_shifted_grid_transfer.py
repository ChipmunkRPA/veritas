# SPDX-License-Identifier: Apache-2.0
"""Recount the anonymous image-grid experiment without an image runtime."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path

CARRIERS = ("png", "jpeg90")
ARMS = ("single_grid", "same_support", "shifted_minimum")
PUBLISHER_ROLES = {"base", "blur", "tone", "target", "confined"}
ORDINARY_ROLES = {"ordinary_picture", "ordinary_blur", "ordinary_tone"}


def review(receipt, payload):
    if sha256(payload).hexdigest() != receipt["case_archive_sha256"]:
        raise ValueError("Compressed held cases differ from their recorded hash")
    rows = [json.loads(line) for line in gzip.decompress(payload).splitlines()]
    calibration = receipt["calibration"]
    expected = {
        (carrier, arm, f"{kind}:{group}", role)
        for carrier in CARRIERS for arm in ARMS
        for kind, size, roles in (("publisher", 11, PUBLISHER_ROLES), ("ordinary", 12, ORDINARY_ROLES))
        for group in range(size) for role in roles
    }
    seen = set()
    sums = {}
    for row in rows:
        carrier, arm, group, role = (row[field] for field in ("carrier", "arm", "group", "role"))
        key = carrier, arm, group, role
        if key in seen or key not in expected:
            raise ValueError("A held case was repeated or outside the frozen cohort")
        seen.add(key)
        point = calibration[f"{carrier}:{arm}"]
        values = point["family_maxima"]
        if len(values) != 8 or point["point"] != sorted(values)[6] or row["point"] != point["point"]:
            raise ValueError("A held case differs from its calibrated point")
        left, top, right, bottom = row["region"]
        area, label, overlap = (row[field] for field in ("area", "label", "overlap"))
        if (not (0 <= left < right and 0 <= top < bottom) or area != (right-left)*(bottom-top)
                or not 0 <= overlap <= min(area, label) or (label > 0) != (role in {"target", "confined"})
                or not math.isfinite(row["score"])):
            raise ValueError("A recorded region has invalid geometry")
        eligible = label > 0 and 2*overlap >= label and 10*overlap >= 3*(area+label-overlap)
        alerts = row["score"] > row["point"]
        total = sums.setdefault(f"{carrier}:{arm}:{role}", Counter())
        total.update(deliveries=1, useful=int(eligible and alerts), rank_useful=int(eligible),
                     alerts=int(alerts), wrong=int(bool(label) and alerts and not eligible))
    if seen != expected or sums != receipt["closure"]["counts"] or len(seen) != receipt["finite_geometries_verified"]:
        raise ValueError("Held counts differ from the closed evaluation")
    decisions = {}
    for carrier in CARRIERS:
        by_arm = {}
        for arm in ARMS:
            target = sums[f"{carrier}:{arm}:target"]
            by_arm[arm] = {"direct_useful": target["useful"], "direct_wrong": target["wrong"],
                           "ordinary_alerts": sum(sums[f"{carrier}:{arm}:{role}"]["alerts"] for role in ORDINARY_ROLES)}
        baseline, support, shifted = (by_arm[arm] for arm in ARMS)
        by_arm["gate"] = (shifted["direct_useful"] >= 3
                          and shifted["direct_useful"] >= max(baseline["direct_useful"], support["direct_useful"])+2
                          and shifted["ordinary_alerts"] <= baseline["ordinary_alerts"]+1
                          and shifted["direct_wrong"] <= baseline["direct_wrong"])
        decisions[carrier] = by_arm
    if decisions != receipt["closure"]["decisions"]:
        raise ValueError("The held gate does not agree with the recorded counts")
    return {"verified_rows": len(seen), "decisions": decisions}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes()), args.cases.read_bytes()), indent=2, sort_keys=True))
