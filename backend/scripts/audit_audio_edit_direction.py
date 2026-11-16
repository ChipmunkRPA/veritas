# SPDX-License-Identifier: MIT
"""Independently recount voice edit coverage and replay fresh PCM controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_clean_teacher_transfer import read


def useful(interval, labels):
    if interval is None:
        return False
    start, end = interval
    for left, right in labels:
        common = min(end, right) - max(start, left)
        if common >= (right - left + 1) // 2 and 10 * common >= 3 * (end - start + right - left - common):
            return True
    return False


def trials(rows, field, cutoff, retain):
    flags, attempts, before = [], [], []
    for row in rows:
        current = row["current"]
        if field is None:
            proposed = interval = current["range"]
            assessed = current["assessed"]
        else:
            candidate = row["candidates"][field]
            assessed = candidate is not None
            proposed = candidate["range"] if candidate else None
            interval = proposed if candidate and candidate["score"] > cutoff else None
            if proposed:
                a, b = proposed
                assert 0 <= a < b <= 96000 and a % 1600 == 0 and b - a in (8000, 12000, 16000)
            if retain and current["range"] is not None:
                interval, assessed = current["range"], True
        flags.append(interval)
        attempts.append(assessed)
        before.append(proposed)

    controls = defaultdict(list)
    for row, interval in zip(rows, flags, strict=True):
        if row["kind"] != "background" and not row["truth"]:
            controls[row["group"]].append(interval)
    counts = Counter(dict.fromkeys(("edits", "useful", "outside", "missed", "ordinary", "ordinary_alerts", "isolated",
                                   "isolated_alerts", "unassessed", "useful_before_cutoff", "clear_useful"), 0))
    for row, interval, assessed, proposal in zip(rows, flags, attempts, before, strict=True):
        truth = row["truth"]
        assert all(0 <= left < right <= 96000 for left, right in truth)
        counts["unassessed"] += not assessed
        if truth:
            counts["edits"] += 1
            hit = useful(interval, truth)
            counts["useful"] += hit
            counts["outside"] += interval is not None and not hit
            counts["missed"] += not hit
            counts["useful_before_cutoff"] += field is not None and useful(proposal, truth)
            counts["clear_useful"] += hit and not any(useful(item, truth) for item in controls[row["group"]])
        else:
            label = "isolated" if row["kind"] == "background" else "ordinary"
            counts[label] += 1
            counts[f"{label}_alerts"] += interval is not None
    return counts


def recount(path):
    packet = read(path)
    panels = defaultdict(list)
    for row in packet["rows"]:
        panels[row["panel"]].append(row)
    checked = 0
    for head in packet["heads"]:
        for rule, rows in head["calibration"].items():
            assert not any(row["truth"] for row in rows)
            kind = {"background": [], "foreground": []}
            for row in rows:
                if row["best"] is not None:
                    kind["background" if row["kind"] == "background" else "foreground"].append(row["best"]["score"])
            foreground = sorted(kind["foreground"])
            background = max(kind["background"])
            point = {"strict": max(foreground[-1], background),
                     "wider": max(foreground[(19 * len(foreground) + 19) // 20 - 1], background)}
            assert point == head["points"][rule]
            fit, held = head["trained_family"], head["held_family"]
            arm = f"{head['feature']}_{rule}"
            for panel, cases in panels.items():
                cohort = [case for case in cases if not (panel == "phoneme" and case["kind"].endswith(f"/{fit}"))]
                actual = packet["comparison"][panel][fit]
                assert fit != held
                assert trials(cohort, None, None, False) == actual["current"]
                for setting, threshold in point.items():
                    for use_fallback, role in ((False, "alone"), (True, "fallback")):
                        assert trials(cohort, f"{fit}/{arm}", threshold, use_fallback) == actual[arm][setting][role]
                        checked += 1
    return {"submissions": len(packet["rows"]), "recounted_outcome_cells": checked}


def samples(path, digest):
    raw = (path / "pcm" / f"{digest}.pcm").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest and len(raw) == 192000
    result = array("h", raw)
    if sys.byteorder == "big":
        result.byteswap()
    return result


def check_fresh(media, receipt):
    data = read(receipt)
    manifest = read(media / "freeze.json")
    assert manifest == data["source_manifest"] and len(manifest["cases"]) == len(data["rows"])
    groups = defaultdict(dict)
    for saved, observed in zip(manifest["cases"], data["rows"], strict=True):
        assert saved == {key: observed[key] for key in saved}
        groups[observed["panel"], observed["group"]][observed["kind"]] = observed
    readers = manifest["pairs"]
    assert len(readers) == len({row["reader"] for row in readers})
    assert len(groups) == 3 * len(readers)
    for (panel, group), roles in groups.items():
        assert roles["edited/sine"]["truth"] == [readers[group]["truth"]]
        start, stop = readers[group]["truth"]
        original = samples(media, roles["untouched"]["pcm_sha256"])
        natural = samples(media, roles["resampled"]["pcm_sha256"])
        changed = samples(media, roles["edited/sine"]["pcm_sha256"])
        assert original[:start] == natural[:start] and original[stop:] == natural[stop:]
        assert original[start:stop] != natural[start:stop] and original[start:stop] != changed[start:stop]
        backgrounds = {row["shared_background_sha256"] for row in roles.values()}
        if panel == "new_dry":
            assert len(roles) == 3 and backgrounds == {None}
            continue
        assert len(roles) == 4 and backgrounds == {roles["background"]["pcm_sha256"]}
        background = samples(media, roles["background"]["pcm_sha256"])
        level = int(panel.removeprefix("new_").removesuffix("dB"))
        real_energy = sum((a - b) ** 2 for a, b in zip(original, background, strict=True))
        background_energy = sum(value * value for value in background)
        assert abs(real_energy / background_energy - 10 ** (level / 10)) < 0.0003
    return {"verified_audio_files": len(data["rows"]), "reader_groups": len(readers), "source_pairs": len(groups)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--media", type=Path)
    args = parser.parse_args()
    print(json.dumps(recount(args.receipt) | (check_fresh(args.media, args.receipt) if args.media else {}), sort_keys=True))
