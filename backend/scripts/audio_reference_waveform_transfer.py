"""Replay the earlier-copy waveform check on pinned speech and noise comparisons."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.evidence import audio_reference_waveform as waveform
from scripts import audio_aligned_speech_transfer as prior

BASELINE = prior.io.PUBLIC / "audio-aligned-speech-app-transfer-2026-11-14.json"
STOCK = prior.io.PUBLIC / "audio-random-clock-stock-2026-11-13.json"
STRESS_ROLES = ("varied_noise", "speed_and_noise", "padded_speed_and_noise", "gain_and_noise", "eq_and_noise", "only_noise")
SCHEMA = "chipmunk.earlier-copy-waveform-transfer/1"


def spec():
    return {"schema": SCHEMA, "baseline_sha256": prior.io.sha(BASELINE), "stock_sha256": prior.io.sha(STOCK),
            "method": waveform.BASIS, "implementation_sha256": prior.io.sha(Path(waveform.__file__))}


def previous_spans(row):
    spans = row["current"]["spans"]
    aligned = row["alignment"]["selected"]
    residual = row["alignment"]["processing_residual"]
    if (aligned and aligned["score"] > prior.POINT and aligned["earlier_span"][0] < aligned["earlier_span"][1]
            and (residual is None or residual > prior.RESIDUAL)):
        return [*spans, aligned["span"]]
    return spans


def summarize(rows, stress):
    baselines = {(r["panel"], r["id"], r["scene"], r["role"]): r for r in prior.io.read(BASELINE)["rows"]}
    table = {}
    stress_counts = {}
    useful = prior.original.common.useful
    for row in rows:
        key = tuple(row[field] for field in ("panel", "id", "scene", "role"))
        base = baselines.pop(key)
        if base["sha256"] != row["sha256"] or base["samples"] != row["samples"]:
            raise ValueError("An input does not match the existing benchmark")
        span = validate(row)
        spans = previous_spans(base)
        baseline = any(useful(item, truth) for truth in base["truth"] for item in spans)
        new = bool(span and any(useful(span, truth) for truth in base["truth"]))
        role = "ordinary" if row["role"] in prior.original.CONTROLS else row["role"]
        cell = table.setdefault("/".join((row["scene"], role)), Counter())
        cell.update(total=1, baseline_useful=baseline, combined_useful=baseline or new, candidate_useful=new,
                    baseline_flags=bool(spans), candidate_flags=bool(span), combined_flags=bool(spans or span),
                    candidate_unassessed=row["status"] in {"equal_duration_required", "insufficient_matching_waveform"})
    if baselines:
        raise ValueError("Benchmark cases are missing")
    stock_groups = {row["group"] for row in stress}
    expected = {(group, scene, role) for group in stock_groups for scene in ("rain", "train") for role in STRESS_ROLES}
    for row in stress:
        key = tuple(row[field] for field in ("group", "scene", "role"))
        expected.remove(key)
        span = validate(row)
        stress_counts.setdefault(row["role"], Counter()).update(
            total=1, flags=bool(span), unassessed=row["status"] in {"equal_duration_required", "insufficient_matching_waveform"})
    if len(stock_groups) != 8 or expected:
        raise ValueError("Stress cases are missing")
    return {"transfer": table, "stress": stress_counts}


def validate(row):
    status = row["status"]
    span = row["selected"]
    if (status not in {"equal_duration_required", "insufficient_matching_waveform", "no_local_difference",
                       "broad_processing_difference", "localized_difference"}
            or (status == "localized_difference") != bool(span)):
        raise ValueError("The local result is inconsistent")
    if not span:
        return None
    a, b = row["samples"]
    start, end = span["earlier_span"]
    left, right = span["lags"]
    if not (0 <= start < end <= a and 0 <= start + left < end + right <= b
            and span["span"] == [start + left, end + right] and abs(left - right) >= waveform.MIN_DELAY_CHANGE
            and 0 <= span["matching_ratio"] <= waveform.MAX_RETAINED_SHARE and min(span["supports"]) >= 3):
        raise ValueError("A selected range differs from its supporting evidence")
    return span["span"]


def measurement(first, latter):
    status, anchors, selected = waveform.locate(first, latter)
    return {"sha256": [prior.io.digest(pcm.tobytes()) for pcm in (first, latter)], "samples": [len(first), len(latter)],
            "status": status, "stable_match_windows": anchors, "selected": selected}


def stress_pairs(stock, noises):
    import numpy as np

    for index, (group, first, _) in enumerate(prior.window.source_groups(stock, "held")):
        processing = prior.original.matched_controls(first, first, rate_ratio=(83, 84))
        speed = processing["resampled"]
        profile = np.ones(first.size)
        profile[2 * waveform.RATE:4 * waveform.RATE] += 2 * np.hanning(2 * waveform.RATE)
        for scene, tracks in noises.items():
            track = tracks[index % len(tracks)]
            boosted = np.clip(np.rint(track * profile), -32768, 32767).astype("<i2")
            padded = np.pad(speed, (0, len(first) - len(speed)))
            mix = prior.sine.mix
            controls = (mix(first, first, boosted)[0], mix(speed, speed, track[:len(speed)])[0],
                        mix(first, padded, track)[1], mix(first, processing["gain"], track)[1],
                        mix(first, processing["equalized"], track)[1], track)
            for role, submitted in zip(STRESS_ROLES, controls, strict=True):
                yield {"group": group, "scene": scene, "role": role} | measurement(first, submitted)


def run(folders, stock, sounds):
    noises = prior.window.read_noise(sounds)
    rows = []
    for name, source, folder in zip(prior.PANELS, prior.sources(), folders, strict=True):
        for item, scene, role, _, first, latter in prior.cases(prior.io.read(source), folder, noises):
            rows.append({"panel": name, "id": item["id"], "scene": scene, "role": role} | measurement(first, latter))
        print(f"Measured {name}", flush=True)
    stress = list(stress_pairs(stock, noises))
    return spec() | {"rows": rows, "stress": stress, "results": summarize(rows, stress)}


def review(receipt):
    if any(receipt[key] != value for key, value in spec().items()):
        raise ValueError("The pinned comparison procedure differs")
    results = summarize(receipt["rows"], receipt["stress"])
    if results != receipt["results"]:
        raise ValueError("The reported results do not match the evidence")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("receipt", type=Path)
    for name in ("full", "short", "stock", "sounds"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        if args.receipt.exists() or not all((args.full, args.short, args.stock, args.sounds)):
            parser.error("Native replay needs new output and all four inputs.")
        result = run((args.full, args.short), args.stock, args.sounds)
        prior.io.save(args.receipt, result)
    else:
        result = prior.io.read(args.receipt)
    print(json.dumps(review(result), sort_keys=True))
