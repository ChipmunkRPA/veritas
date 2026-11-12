"""Rebuild the removed-sound waveforms and decisions without model inference."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from scripts import audio_removed_source_transfer as study
from scripts import review_audio_spatial_interference as original


def energy(values):
    return float(np.sqrt(np.dot(values, values)))


def exact(args, row):
    identity = f"{row['id']:03}"
    folder = args.archived if row["role"] == "archived" else args.parent
    wave_path = folder / "wav" / f"{identity}.wav"
    original_raw = original.wave_pcm(wave_path, 2, whole=True)
    mono = (args.parent / "mono" / f"{identity}.pcm").read_bytes()
    original.require(original.digest(original_raw) == row["stereo_sha256"] and original.digest(mono) == row["mono_sha256"],
                     "Held stereo or mono commitment differs")
    return np.frombuffer(original_raw, "<i2").astype(float).reshape(study.spatial.COUNT, 2), np.frombuffer(mono, "<i2").astype(float)


def reconstruct(args, row, ref, obs):
    stereo, mono = exact(args, row)
    if ref["treatment"] != "conditioned" or ref["conditioned"]["raw"] is None:
        original.require(all(obs[key] is None for key in ("removed_pcm_sha256", "removed_rms_fraction", "removed_score")),
                         "Unexpected result without a projected view")
        return False, None, False
    weights = original.fitted(stereo)
    projected = original.condition(stereo, weights)
    tie = original.verify_conditioned(stereo, weights, projected, ref["conditioned_sha256"], row["id"])
    mean = (stereo[:, 0] + stereo[:, 1]) / 2
    removed = mean - projected
    power, starting = energy(removed), energy(mono)
    ratio = power / starting if starting else 0.
    original.require(math.isclose(ratio, obs["removed_rms_fraction"], rel_tol=0, abs_tol=1e-12), "Removed support differs")
    audible = ratio >= .1 and power > 0
    if audible:
        gain = min(6., starting / power, 31129 / max(abs(removed)))
        actual = original.digest(original.pcm(removed * gain).tobytes())
        original.require(actual == obs["removed_pcm_sha256"], "Rebuilt removed waveform differs")
    else:
        original.require(obs["removed_pcm_sha256"] is None and obs["removed_score"] is None, "Unreliable removed view admitted")
    leak = None
    if row["foreground_sha256"] is not None:
        raw = (args.parent / "foreground" / f"{row['id']:03}.pcm").read_bytes()
        original.require(original.digest(raw) == row["foreground_sha256"], "Source decomposition foreground differs")
        target = np.frombuffer(raw, "<i2").astype(float).reshape(study.spatial.COUNT, 2)
        known = target.mean(axis=1)
        target_projection = original.condition(target, weights)
        interference = stereo - target
        noise_projection = original.condition(interference, weights)
        original.require(np.max(abs(projected - target_projection - noise_projection)) < 1e-7,
                         "Projected sound does not recombine from known foreground and background")
        leak = energy(known - target_projection) / energy(known)
    return audible, leak, tie


def score_readout(ref, obs, bundle):
    current = ref["baseline"]["decision"]
    filtered, target = original.decision(ref["conditioned"]["raw"], bundle)
    removed, removed_logit = original.decision(obs["removed_score"], bundle)
    original.require((filtered, removed) == (ref["conditioned"]["decision"], obs["removed_direction"]), "Raw readout differs")
    baseline_logit = original.decision(ref["baseline"]["raw"], bundle)[1]
    if current == "inconclusive" and filtered == "ai":
        return current, "ai", {"rescue": target - removed_logit} if removed_logit is not None else {}
    if current == "ai":
        projection = "inconclusive" if target is not None and filtered != "ai" and baseline_logit - target > math.log(2) else "ai"
        return current, projection, {"suppress": removed_logit - target} if removed == "ai" else {}
    return current, current, {}


def verify(args):
    lock, all_rows, _ = study.evidence(args)
    run = args.output
    read, require, digest = study.read, study.require, study.digest
    frozen, fit, result = (read(run / f"{name}.json") for name in ("freeze", "points", "result"))
    require(lock == frozen and result["freeze_sha256"] == fit["freeze_sha256"] == digest(run / "freeze.json")
            and result["points_sha256"] == digest(run / "points.json"), "Study, operating points or report differ")
    bundle = read(study.calibration.PATH)
    panels, bounds, leakage, total = {}, {}, {}, Counter()
    for panel, role in study.PANELS.items():
        path = run / f"{panel}.json"
        ledger = read(path)
        require(digest(path) == result["panels_sha256"][panel] and ledger["freeze_sha256"] == result["freeze_sha256"], "Ledger changed")
        if panel == "selection":
            require(digest(path) == fit["selection_sha256"], "Source point used a different selection ledger")
        records = []
        for (row, ref), obs in zip((pair for pair in all_rows if pair[0]["role"] == role), ledger["rows"], strict=True):
            require((row["id"], row["mono_sha256"]) == (obs["id"], obs["input_sha256"]), "Audio and response are unrelated")
            audible, leak, tie = reconstruct(args, row, ref, obs)
            total.update({"original_views": 1, "removed_waveforms": audible, "projection_ties": tie})
            if leak is not None:
                leakage.setdefault(f"{panel}/{row['placement']}", []).append(leak)
            records.append((row, obs, score_readout(ref, obs, bundle)))
        require(sum(obs["removed_score"] is not None for _, obs, _ in records) == ledger["forwards"], "Forward count differs")
        panels[panel] = records
    for action, family in (("rescue", True), ("suppress", False)):
        samples = [(votes[action], row["group"]) for row, _, (_, _, votes) in panels["selection"] if
                   (row["family"] == "genuine") == family and action in votes]
        bounds[action] = {"point": max([math.log(2), *(value for value, _ in samples)]), "protected_views": len(samples),
                          "protected_groups": len({group for _, group in samples})}
    for action in bounds:
        require(bounds[action]["protected_views"] == fit["points"][action]["protected_views"]
                and bounds[action]["protected_groups"] == fit["points"][action]["protected_groups"]
                and math.isclose(bounds[action]["point"], fit["points"][action]["point"], abs_tol=1e-12),
                "Independently reconstructed source point differs")
    outcomes = 0
    for panel, observations in panels.items():
        cells = result["results"][panel]["cells"]
        for placement, recorded in cells.items():
            selected = [(row, obs, votes) for row, obs, votes in observations if row["placement"] == placement]
            require(sum(obs["removed_score"] is not None for _, obs, _ in selected) == recorded["assessed_removed"], "Admission tally differs")
            for arm, idx in (("current", 0), ("projection_only", 1), ("removed_source", 2)):
                events = Counter()
                for row, _, (current, projected, scores) in selected:
                    family = "genuine" if row["family"] == "genuine" else "synthetic"
                    value = current
                    if scores.get("rescue", -math.inf) > bounds["rescue"]["point"] + 1e-12:
                        value = "ai"
                    if scores.get("suppress", -math.inf) > bounds["suppress"]["point"] + 1e-12:
                        value = "inconclusive"
                    chosen = (current, projected, value)[idx]
                    events.update({f"{family}_{arm}_{chosen}": 1, f"{family}_{arm}_added": chosen == "ai" and current != "ai",
                                   f"{family}_{arm}_lost": chosen != "ai" and current == "ai"})
                require({k: v for k, v in recorded.items() if f"_{arm}_" in k} == dict(events), "Outcome tally differs")
                outcomes += 1
    require(total["original_views"] == 217 and outcomes == 21, "Incomplete stereo replay")
    held = result["results"]["confirmation"]["cells"]
    center, offset = held["center"], held["offset"]
    criteria = {
        "center_true_gains": center["synthetic_removed_source_added"] >= 2,
        "center_false_flags_cleared": center["genuine_removed_source_lost"] >= 2,
        "center_limits": (center["synthetic_removed_source_added"] >= center["synthetic_removed_source_lost"]
                          and max(center["genuine_removed_source_added"], center["synthetic_removed_source_lost"]) <= 1),
        "native_and_clean_no_added_false": held["clean"]["genuine_removed_source_added"] == held["native"]["genuine_removed_source_added"] == 0,
        "offset_limits": max(offset["genuine_removed_source_added"], offset["synthetic_removed_source_lost"]) <= 1,
    }
    require(result["qualification"] == criteria and result["advances"] == all(criteria.values()), "Independent advancement differs")
    summary = {"panels": {key: len(value) for key, value in panels.items()}, "outcome_cells": outcomes,
               "result_sha256": digest(run / "result.json"), "qualification": criteria, **total,
               "foreground_leakage_rms_fraction": {k: {"views": len(v), "median": median(v), "maximum": max(v)} for k, v in leakage.items()}}
    study.write(run / "audit.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("output", "parent", "archived"):
        cli.add_argument(field, type=Path)
    verify(cli.parse_args())
