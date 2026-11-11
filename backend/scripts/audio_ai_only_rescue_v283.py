"""Replay the fixed AI-only rescue policy on the saved stereo development cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from scripts import audio_native_stereo_authenticity_v282 as prior
from scripts.video_appearance_corrected_v112 import scoped

ROOT, PANEL = prior.ROOT, prior.PANEL
PUBLIC = PANEL / "audio-ai-only-rescue-freeze-v283.json"
PREVIOUS = PANEL / "audio-native-stereo-authenticity-complete-v282.json"
PREVIOUS_SHA = "4f40ca216a1c2a2ef7491c1a962e32fdd01d61afe1ae6a1556baec6c7d096269"
PREVIOUS_FREEZE_SHA = "0ba2fc44772795aa67756df6df747415e5c45e1d506f89850d72075f1aaa4e23"
CODE = (
    "backend/scripts/audio_ai_only_rescue_v283.py",
    "backend/tests/test_audio_ai_only_rescue_v283.py",
    "docs/audio-ai-only-rescue-v283-method.md",
)
DIRECTIONS = ("ai", "original", "inconclusive", "not_assessed")
POLICIES = ("mono", "selected_only", "ai_only_rescue")
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def apply_ai_only(mono, selected):
    """Add an AI direction only where the mono assessment supplies no direction."""
    require(mono in DIRECTIONS and selected in DIRECTIONS, "Unknown speech decision")
    if mono in ("not_assessed", "inconclusive") and selected == "ai":
        return "ai"
    return mono


def compare(cases, observed):
    prior.downstream.validate(cases, observed)
    scores = {row["case"]: row["decision"] for row in observed}
    views = {row["case"]: row for row in cases}
    counts, transitions = defaultdict(Counter), defaultdict(Counter)
    additional_groups = defaultdict(set)
    for case in cases:
        if case["view"] != "mono":
            continue
        selected_id = f"{case['source_case']}-selected"
        selected_view = views[selected_id]
        require(
            selected_view["speech_gate_admitted"]
            and all(selected_view[key] == case[key] for key in ("arm", "condition", "cohort", "group")),
            "Exact paired admitted selected view required",
        )
        mono = scores[case["case"]] if case["speech_gate_admitted"] else "not_assessed"
        selected = scores[selected_id]
        rescued = apply_ai_only(mono, selected)
        arm = case["arm"]
        for policy, outcome in zip(POLICIES, (mono, selected, rescued), strict=True):
            counts[arm, policy][outcome] += 1
        for policy, outcome in (("selected_only", selected), ("ai_only_rescue", rescued)):
            transitions[arm, policy][f"{mono}_to_{outcome}"] += 1
        if rescued != mono:
            additional_groups[arm].add(case["group"])

    result = {}
    for arm in prior.EXPECTED_CLEAN:
        result[arm] = {
            "paired_cases": counts[arm, "mono"].total(),
            "by_policy": {
                policy: {direction: counts[arm, policy][direction] for direction in DIRECTIONS}
                for policy in POLICIES
            },
            "transitions_from_mono": {policy: dict(transitions[arm, policy]) for policy in POLICIES[1:]},
            "source_groups_with_added_ai_direction": len(additional_groups[arm]),
        }
    return result


def protocol():
    require(
        sha(PREVIOUS) == PREVIOUS_SHA and sha(prior.PUBLIC) == PREVIOUS_FREEZE_SHA,
        "Exact v282 public comparison and freeze required",
    )
    return {
        "schema": "chipmunk.audio-ai-only-rescue-freeze/283",
        "source_comparison_sha256": PREVIOUS_SHA,
        "source_freeze_sha256": PREVIOUS_FREEZE_SHA,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "rule": "Preserve mono. Add selected-channel AI only when mono is inconclusive or not assessed.",
        "selected_only_comparator": "The selected channel alone on the v282 eligible forced-imbalance cases.",
        "planned_paired_cases": 33,
        "holdout_access_permitted": False,
        "model_calls_permitted": False,
        "serving_change_permitted": False,
    }


def readout():
    require(read(PUBLIC) == protocol(), "Exact prospective policy replay required")
    cases = prior.verify()
    previous = scoped(prior.readout, {"verify": lambda: cases})()
    require(previous == read(PREVIOUS), "The full v282 guarded comparison must replay exactly")
    journal = prior.OUTPUT / "private-observations.json"
    require(sha(journal) == previous["private_observation_journal_sha256"], "Exact score journal required")
    arms = compare(cases, read(journal))
    require(
        {arm: metrics["paired_cases"] for arm, metrics in arms.items()} == {"genuine": 13, "expressive": 20},
        "Every predetermined noisy comparison pair is required",
    )
    return {
        "schema": "chipmunk.audio-ai-only-rescue-result/283",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "source_comparison_sha256": PREVIOUS_SHA,
        "by_family": arms,
        "new_neural_calls": 0,
        "holdout_cases_opened": 0,
        "serving_changed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "readout"))
    if parser.parse_args().command == "freeze":
        write(PUBLIC, protocol())
    else:
        print(json.dumps(readout(), indent=2, allow_nan=False))
