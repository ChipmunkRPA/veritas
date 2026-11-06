"""Source-separated residual learner/guard; local research, never serving."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict

from scripts import audio_calibration_v3_methods as guard_method
from scripts import audio_residual_head_v6 as learner
from scripts.review_audio_noise_v4 import valid

METHOD = "audio-source-separated-guard/7.0"
CONDITIONS = ("clean", "noise_0", "noise_5", "noise_15", "music_5", "music_15")


def source_roles(bases):
    parent = {r["id"]: r["id"] for r in bases}
    if len(parent) != len(bases):
        raise ValueError("Duplicate original source identity")

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for row in bases:
        keys = [("voice", row["cluster_id"]), ("source", row["source_group"])]
        if row.get("transcript"):
            keys.append(("text", " ".join(re.findall(r"\w+", row["transcript"].casefold()))))
        if row["dataset"] == "isjwdu/DFADD" and row["label"] == 1:
            match = re.search(r"p\d+_(\d+)(?:_|$)", row["id"])
            if match:
                keys.append(("dfadd_prompt", match.group(1)))
        for key in keys:
            if key in owners:
                parent[find(row["id"])] = find(owners[key])
            else:
                owners[key] = row["id"]
    groups = defaultdict(list)
    for row in bases:
        groups[find(row["id"])].append(row)
    assignments, records = {}, []
    for members in groups.values():
        roles = {r["split"] for r in members}
        if len(roles) != 1:
            raise ValueError("Connected voice/transcript source group crosses roles")
        role = {"fit": "fit", "cal": "cal", "test": "development"}[roles.pop()]
        identifiers = sorted(r["id"] for r in members)
        key = (
            "connected:" + hashlib.sha256(json.dumps(identifiers, separators=(",", ":")).encode()).hexdigest()
        )
        records.append(
            {
                "group": key,
                "role": role,
                "sources": identifiers,
                "class_counts": [sum(r["label"] == c for r in members) for c in (0, 1)],
            }
        )
        for row in members:
            assignments[row["id"]] = {"group": key, "role": role}
    return assignments, sorted(records, key=lambda r: r["group"])


def initial(p, guards):
    classes = guard_method.class_set(p, guards)
    if not valid(p):
        return "inconclusive"
    lean = "ai" if p >= 0.8 else "original" if p <= 0.2 + 1e-12 else "inconclusive"
    expected = [1] if lean == "ai" else [0]
    return lean if lean != "inconclusive" and classes["included"] == expected else "inconclusive"


def final(p, band_p, guards):
    proposed = initial(p, guards)
    delta = None
    decision = proposed
    if proposed == "original":
        if valid(band_p):
            delta = abs(
                guard_method.clipped_logit(round(p, 6)) - guard_method.clipped_logit(round(band_p, 6))
            )
        if delta is None or delta > math.log(2):
            decision = "inconclusive"
    return {
        "proposed": proposed,
        "decision": decision,
        "stability_log_odds_change": delta,
        "original_stability_veto": proposed == "original" and decision == "inconclusive",
    }


def fit(rows, head_direction):
    if not rows or any(r["role"] != "fit" for r in rows):
        raise ValueError("Only fitting rows may learn parameters")
    if len({r["base_id"] for r in rows}) != 40 or {r["condition"] for r in rows} != set(CONDITIONS):
        raise ValueError("All forty fitting sources and conditions required")
    return learner.fit(rows, head_direction)


def calibrate(rows, fitted):
    if not rows or any(r["role"] != "cal" for r in rows) or len({r["base_id"] for r in rows}) != 80:
        raise ValueError("Only all eighty independent calibration sources may set guards")
    records = []
    for row in rows:
        records.append(
            {
                "split": "cal",
                "speaker": row["group"],
                "label": row["label"],
                "p": learner.predict(row, fitted),
            }
        )
    result = guard_method.calibrate_class_set(records)
    if min(len(result[str(c)]) for c in (0, 1)) < 9:
        raise ValueError("Insufficient independent class calibration groups")
    return result
