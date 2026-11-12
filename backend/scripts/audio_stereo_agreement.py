"""Replay the paired room panel with two fixed physical-channel agreement rules."""

from __future__ import annotations

import argparse
import platform
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_paired_stereo_overlap as paired

sha, read, write = paired.sha, paired.read, paired.write
PANEL_HASH = "b2498395d65efc976df030f1be65c73fbbe6bf8dd1879ca1f83e4960cbb5388a"
CHANNELS = ("ch0", "ch1")
POLICY = {
    "prerequisites": "mono AI, two distinct physical channels, both with frozen speech admission",
    "neither": "withhold to inconclusive if neither physical channel leans AI",
    "either": "withhold to inconclusive if at least one physical channel does not lean AI",
    "minimum_false_flags_withheld_by_mode": 2,
    "maximum_synthetic_flags_withheld": 2,
    "maximum_clean_or_native_flags_withheld": 0,
}


def inputs(source):
    lock, roster = paired.frozen(source)
    summary, gates, scores = (read(source / f"{name}.json") for name in ("aggregate", "gate", "scores"))
    parents = {
        "aggregate_sha256": sha(source / "aggregate.json"),
        "freeze_sha256": sha(source / "freeze.json"),
        "roster_sha256": sha(source / "roster.json"),
        "gate_sha256": sha(source / "gate.json"),
        "scores_sha256": sha(source / "scores.json"),
    }
    if (parents["aggregate_sha256"] != PANEL_HASH
            or any(summary[name] != parents[name] for name in ("freeze_sha256", "gate_sha256", "scores_sha256"))
            or scores["gate_sha256"] != parents["gate_sha256"]
            or any(record["freeze_sha256"] != parents["freeze_sha256"] for record in (gates, scores))):
        raise ValueError("The prior paired-room evidence commitments differ")
    for row, gate, score in zip(roster, gates["rows"], scores["rows"], strict=True):
        if not row["id"] == gate["id"] == score["id"]:
            raise ValueError("The prior paired-room evidence no longer follows the roster")
    return lock, roster, gates["rows"], scores["rows"], parents


def plan(row, gate, score):
    if score.get("mono", {}).get("decision") != "ai":
        return "no_mono_ai"
    if row["channels_sha256"][0] == row["channels_sha256"][1]:
        return "identical_channels"
    if not all(gate["admitted"][name] for name in CHANNELS):
        return "speech_not_admitted_in_both"
    return "compare"


def freeze(source, output):
    lock, rows, gates, scores, parents = inputs(source)
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "chipmunk.audio-stereo-agreement-freeze/1",
        "source": parents,
        "runner_sha256": sha(Path(__file__)),
        "calibration_sha256": lock["calibration_sha256"],
        "incumbent_assets": lock["incumbent_assets"],
        "policy": POLICY,
        "planned": dict(Counter(plan(row, gate, score) for row, gate, score in zip(rows, gates, scores, strict=True))),
    }
    write(output / "freeze.json", record)
    print(record["planned"], flush=True)


def binding(source, output):
    lock, rows, gates, scores, parents = inputs(source)
    frozen = read(output / "freeze.json")
    if (frozen["runner_sha256"] != sha(Path(__file__)) or frozen["source"] != parents or frozen["policy"] != POLICY
            or frozen["incumbent_assets"] != lock["incumbent_assets"]
            or frozen["calibration_sha256"] != lock["calibration_sha256"]):
        raise ValueError("The channel agreement inputs or method changed after freezing")
    return frozen, rows, gates, scores


def run(source, output):
    import torch

    from app.detectors.audio import calibration
    from scripts.run_audio_specialists_v15 import load_voice

    frozen, rows, gates, scores = binding(source, output)
    paired.serving.incumbent.network_free()
    bundle = calibration.load()
    voice = load_voice({"weights": str(paired.serving.incumbent.WEIGHTS), "assets": frozen["incumbent_assets"]})
    torch.set_num_threads(1)
    count = 0

    def count_forward(*_):
        nonlocal count
        count += 1

    hook = voice.register_forward_pre_hook(count_forward)
    measurements = []
    start = time.monotonic()
    try:
        for row, gate, score in zip(rows, gates, scores, strict=True):
            pcm, views = paired.pcm_views(source, row)
            if paired.stereo.observation(pcm, gate["spans"])["admitted"] != gate["admitted"]:
                raise ValueError("The saved speech admission does not match the exact physical samples")
            status = plan(row, gate, score)
            measured = {"id": row["id"], "status": status, "responses": []}
            if status == "compare":
                raw_scores = voice.predict([views[name] for name in CHANNELS])
                for name, digest, raw in zip(CHANNELS, row["channels_sha256"], raw_scores, strict=True):
                    if not calibration.valid(raw):
                        raise ValueError("Invalid physical-channel classifier response")
                    direction = calibration.initial_decision(calibration.probability(raw, bundle), bundle)
                    measured["responses"].append({"channel": name, "pcm_sha256": digest, "raw": raw,
                                                  "supports_ai": direction == "ai"})
                if count % 20 == 0:
                    print(f"Completed {count} physical-channel forwards", flush=True)
            measurements.append(measured)
    finally:
        hook.remove()
    if count != 2 * frozen["planned"]["compare"]:
        raise ValueError("Physical-channel inference count differs")
    write(output / "measurements.json", {
        "freeze_sha256": sha(output / "freeze.json"), "rows": measurements, "forwards": count,
        "elapsed_seconds": time.monotonic() - start, "peak_rss_native": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "platform": platform.system(),
    })
    print({"forwards": count, "elapsed_seconds": round(time.monotonic() - start, 3)}, flush=True)


def readout(source, output):
    from app.detectors.audio import calibration

    frozen, rows, gates, scores = binding(source, output)
    captured = read(output / "measurements.json")
    bundle = calibration.load()
    if captured["freeze_sha256"] != sha(output / "freeze.json"):
        raise ValueError("Measurements use another freeze")
    cells, withheld = defaultdict(Counter), defaultdict(Counter)
    for row, gate, score, measurement in zip(rows, gates, scores, captured["rows"], strict=True):
        mode, treatment, family = row.get("mode"), row["treatment"], row["family"]
        cell = cells[(treatment, row.get("room"), mode, family)]
        status, responses = measurement["status"], measurement["responses"]
        baseline = score.get("mono", {}).get("decision", "unassessed")
        if measurement["id"] != row["id"] or status != plan(row, gate, score):
            raise ValueError("A physical-channel result used the wrong source or eligibility")
        cell.update(planned=1, baseline_ai=baseline == "ai", assessed=baseline != "unassessed")
        expected = list(zip(CHANNELS, row["channels_sha256"], strict=True)) if status == "compare" else []
        if [(item["channel"], item["pcm_sha256"]) for item in responses] != expected:
            raise ValueError("Physical-channel measurement used different bytes")
        if not responses:
            continue
        support = []
        for response in responses:
            raw = response["raw"]
            supports_ai = calibration.initial_decision(calibration.probability(raw, bundle), bundle) == "ai"
            if not calibration.valid(raw) or supports_ai != response["supports_ai"]:
                raise ValueError("Replayed voice points differ")
            support.append(supports_ai)
        cell.update(compared=1, **{f"ai_channels_{sum(support)}": 1})
        for policy, suppress in (("neither", not any(support)), ("either", not all(support))):
            if suppress:
                cell[f"{policy}_withheld"] += 1
                withheld[policy][(treatment, mode, family == "genuine")] += 1

    results = {}
    for policy in ("neither", "either"):
        counts = withheld[policy]
        false = {mode: counts[("mixed", mode, True)] for mode in paired.MODES}
        missed = {mode: counts[("mixed", mode, False)] for mode in paired.MODES}
        controls = sum(n for (treatment, _, _), n in counts.items() if treatment != "mixed")
        results[policy] = {
            "false_flags_withheld": false, "synthetic_flags_withheld": missed, "controls_withheld": controls,
            "advance": (all(n >= POLICY["minimum_false_flags_withheld_by_mode"] for n in false.values())
                        and sum(missed.values()) <= POLICY["maximum_synthetic_flags_withheld"]
                        and controls <= POLICY["maximum_clean_or_native_flags_withheld"]),
        }
    forwards = sum(2 * cell["compared"] for cell in cells.values())
    if captured["forwards"] != forwards or forwards != 2 * frozen["planned"]["compare"]:
        raise ValueError("Replayed physical-channel inference count differs")
    result = {
        "schema": "chipmunk.audio-stereo-agreement-result/1", "freeze": frozen,
        "freeze_sha256": sha(output / "freeze.json"), "measurements_sha256": sha(output / "measurements.json"),
        "forwards": forwards, "results": results,
        "cells": [{"treatment": t, "room": r, "mode": m, "family": f, **values} for (t, r, m, f), values in cells.items()],
        **{key: captured[key] for key in ("elapsed_seconds", "peak_rss_native", "platform")},
    }
    write(output / "aggregate.json", result)
    print(results, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "readout"))
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "readout": readout}[args.stage](args.source, args.output)
