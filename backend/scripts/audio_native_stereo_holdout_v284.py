"""Evaluate a fixed stereo AI-only rescue on every sealed v277 holdout case."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from types import SimpleNamespace

from scripts import audio_native_stereo_authenticity_v282 as previous
from scripts import audio_native_stereo_gate_v279 as stereo
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = previous.ROOT, previous.BASE, previous.PANEL
retry, downstream, pairs = previous.retry, previous.downstream, stereo.pairs
MODULE = "scripts.audio_native_stereo_holdout_v284"
TRIAL = BASE / "audio-native-stereo-holdout-v284"
PUBLIC = PANEL / "audio-native-stereo-holdout-freeze-v284.json"
GATE = TRIAL / "gate/authored"
SCORE = TRIAL / "classifier/authored"
CLASSIFIER_PLAN = TRIAL / "classifier-plan.json"
V277 = PANEL / "audio-native-stereo-pair-holdout-complete-v277.json"
V283_FREEZE = PANEL / "audio-ai-only-rescue-freeze-v283.json"
RECEIPTS = {
    **previous.RECEIPTS,
    V277: "40bbdc73ad3110857926bd345952f3a918910b56cdcd49c45ef5fd99318d8b44",
    PANEL / "audio-native-stereo-pair-freeze-v277.json": "6a31cdf0b033330cec0d317f6c4d3dd99381f2f2d00eab85731535585ead9f0e",
    PANEL / "audio-native-stereo-gate-complete-v279.json": previous.GATE_SHA,
    previous.PUBLIC: "0ba2fc44772795aa67756df6df747415e5c45e1d506f89850d72075f1aaa4e23",
    PANEL / "audio-native-stereo-authenticity-complete-v282.json": "4f40ca216a1c2a2ef7491c1a962e32fdd01d61afe1ae6a1556baec6c7d096269",
    V283_FREEZE: "49e16c82ffb35bd79bbb5f44052303ed9fec1053c44f54e08e1869023431a56f",
    PANEL / "audio-ai-only-rescue-complete-v283.json": "043ee77b8585aa1ea7e79e709e76e84d29062ccf2f9caaca9c0ef7db8b8a8430",
}
CODE = (
    "backend/scripts/audio_native_stereo_holdout_v284.py",
    "backend/tests/test_audio_native_stereo_holdout_v284.py",
    "docs/audio-native-stereo-holdout-v284-method.md",
    "backend/scripts/audio_ai_only_rescue_v283.py",
    "backend/scripts/audio_native_stereo_gate_v279.py",
    "backend/scripts/audio_native_stereo_authenticity_v282.py",
    "backend/scripts/audio_noise_speech_rescue_downstream_v270.py",
)
PLANNED_CASES = 200
MAX_VIEWS = 2 * PLANNED_CASES
DIRECTIONS = ("ai", "original", "inconclusive", "not_assessed")
ROUTES = ("mono", "ai_only_rescue", "channel_priority")
read, write, sha, require = previous.read, previous.write, previous.sha, previous.require
digest = previous.digest


def source_metadata():
    require(all(sha(path) == value for path, value in RECEIPTS.items()), "Exact sealed source and development receipts required")
    plan, pair_freeze, _ = pairs.verify("holdout")
    result, profile, journal = pairs.guarded_result("holdout")
    public = read(V277)
    require(public == {**result, "resource_profile_sha256": sha(profile), "raw_case_replay_count": PLANNED_CASES}
            and result["case_outcomes"] == {"prepared": PLANNED_CASES} and result["planned_cases"] == PLANNED_CASES
            and sha(pairs.FREEZE) == RECEIPTS[PANEL / "audio-native-stereo-pair-freeze-v277.json"],
            "Exact completed 200-case v277 holdout preparation required")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    return stereo.bind_cases(plan, pair_freeze["conditions"]["holdout"], rows), public


runtime_identity = scoped(downstream.identity, {"RECEIPTS": RECEIPTS, "CODE": CODE, "EXPECTED": {}, "CONDITIONS": ()})


def identity(cases, source):
    inherited = runtime_identity([])
    policy_file = "backend/scripts/audio_ai_only_rescue_v283.py"
    require(inherited["runtime"] == read(previous.PUBLIC)["runtime"] and sha(stereo.gate.VAD) == stereo.gate.VAD_SHA
            and inherited["code_sha256"][policy_file] == read(V283_FREEZE)["code_sha256"][policy_file],
            "Exact v282 CPU voice runtime, v279 activity checkpoint and v283 policy code required")
    return {
        **{key: inherited[key] for key in ("source_receipts_sha256", "serving_calibration_sha256", "serving_asset_sha256",
                                           "inherited_v261_code_sha256", "code_sha256", "runtime", "sample_rate", "samples_per_window")},
        "schema": "chipmunk.audio-native-stereo-holdout-freeze/284",
        "role": "holdout",
        "source_private_cases_sha256": source["private_cases_sha256"],
        "source_profile_sha256": source["resource_profile_sha256"],
        "planned_cases": len(cases),
        "conditions": list(dict.fromkeys(case["condition"] for case in cases)),
        "foregrounds_by_family": dict(Counter(row["lane"] for row in cases if row["condition"] == "matched_clean")),
        "connected_source_groups": len({(row["lane"], row["source_group"]) for row in cases}),
        "vad_frame_forwards": len(cases) * len(stereo.VIEWS) * stereo.FORWARDS_PER_VIEW,
        "max_classifier_views": MAX_VIEWS,
        "max_classifier_forwards": 2 * MAX_VIEWS,
        "routes": list(ROUTES),
        "gate_and_selection": "v279 without changes",
        "policy": "Keep directional mono. Only admit selected-channel AI if mono is inconclusive or unassessed.",
        "channel_priority_diagnostic": "Use eligible selected-channel direction when available, otherwise keep mono.",
    }


def preflight():
    return {**identity(*source_metadata()), "status": "ready_for_freeze", "media_opened": 0, "model_calls": 0}


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve prior attempts")
    locked = identity(*source_metadata())
    TRIAL.mkdir(mode=0o700)
    GATE.parent.mkdir(mode=0o700)
    SCORE.parent.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "media_opened": 0, "model_calls": 0}


def verify():
    cases, source = source_metadata()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == identity(cases, source), "Exact prospective v284 freeze required")
    return cases


def case_path(row):
    path = pairs.MEDIA / "holdout" / f"{row['case']}.wav"
    require(path.is_file() and path.resolve(strict=True) == path and sha(path) == row["wav_sha256"],
            "Exact v277 holdout stereo WAV required")
    return path


read_native = scoped(stereo.read_native, {"case_path": case_path})


def gate_result(cases, observations, journal):
    totals = stereo.summarize(cases, observations)
    return {**totals, "schema": "chipmunk.audio-native-stereo-holdout-gate-result/284", "status": "complete",
            "freeze_sha256": sha(PUBLIC), "private_observations_sha256": sha(journal),
            "new_neural_calls": totals["vad_frame_forwards"], "holdout_cases_opened": len(cases)}


GATE_BINDINGS = {"verify": verify, "STAGE": GATE, "OUTPUT": GATE / "output", "read_native": read_native, "result": gate_result}
gate_worker = scoped(stereo.worker, GATE_BINDINGS)
gate_readout = scoped(stereo.readout, {**GATE_BINDINGS, "__file__": __file__})


def views_for(cases, observed):
    variants = []
    for case, row in zip(cases, observed, strict=True):
        require(case["case"] == row["case"] and row["swap_invariant"] and row["selected"] in (None, 0, 1),
                "Exact replayed v279 selection required")
        selected = row["selected"]
        views = []
        if row["admitted"]["mono"]:
            views.append(("mono", None, case["mixed_pcm_sha256"]))
        if selected is not None and row["admitted"][f"ch{selected}"]:
            views.append(("selected", selected, case["channel_pcm_sha256"][selected]))
        for name, channel, commitment in views:
            variants.append({"case": f"{case['case']}-{name}", "source_case": case["case"], "view": name,
                             "channel": channel, "pcm_sha256": commitment, "condition": case["condition"],
                             "group": (case["lane"], case["source_group"])})
    require(len(variants) <= MAX_VIEWS and len({row["case"] for row in variants}) == len(variants), "Bounded unique views required")
    return variants


def score_roster(replay_gate=False):
    cases = verify()
    folder = GATE / "output"
    saved, journal = folder / "result.json", folder / "private-observations.jsonl"
    gate_saved = read(saved)
    if replay_gate:
        replayed = scoped(gate_readout, {"verify": lambda: cases})()
        require(all(replayed[key] == value for key, value in gate_saved.items()), "Exact guarded VAD result required")
    require(gate_saved["freeze_sha256"] == sha(PUBLIC) and gate_saved["holdout_cases_opened"] == PLANNED_CASES
            and gate_saved["private_observations_sha256"] == sha(journal), "Complete sealed holdout gate required")
    observed = [json.loads(line) for line in journal.read_text().splitlines()]
    views = views_for(cases, observed)
    binding = {"schema": "chipmunk.audio-native-stereo-holdout-classifier-plan/284", "freeze_sha256": sha(PUBLIC),
               "gate_result_sha256": sha(saved), "gate_profile_sha256": sha(GATE / "profile.json"),
               "gate_observations_sha256": sha(journal), "planned_cases": len(cases),
               "eligible_views": previous.plan(views), "planned_views": len(views), "max_classifier_forwards": 2 * len(views)}
    return cases, views, binding


def seal_classifier():
    require(not CLASSIFIER_PLAN.exists(), "Preserve the first classifier plan")
    _, _, binding = score_roster(replay_gate=True)
    write(CLASSIFIER_PLAN, binding)
    return {"status": "frozen", "classifier_plan_sha256": sha(CLASSIFIER_PLAN), "planned_cases": binding["planned_cases"],
            "planned_views": binding["planned_views"], "max_classifier_forwards": binding["max_classifier_forwards"], "model_calls": 0}


def score_verify(replay_gate=False):
    cases, variants, binding = score_roster(replay_gate)
    require(read(CLASSIFIER_PLAN) == binding, "Exact prospective classifier view plan required")
    return cases, variants


def score_inputs():
    cases, variants = score_verify()
    metadata = {row["case"]: row for row in cases}
    current, result = None, []
    for variant in variants:
        case = metadata[variant["source_case"]]
        if current != case["case"]:
            path, pcm = read_native(case)
            current = case["case"]
        channel = variant["channel"]
        body = previous.mono_bytes(case, path) if channel is None else pcm[:, channel].tobytes()
        require(len(body) == 2 * stereo.SAMPLES and digest(body) == variant["pcm_sha256"], "Exact admitted view PCM required")
        result.append({**variant, "payload": body})
    return result


def score_result(variants, journal):
    audited = downstream.validate(variants, read(journal))
    return {"schema": "chipmunk.audio-native-stereo-holdout-classifier-result/284", "status": "complete",
            "freeze_sha256": sha(PUBLIC), "classifier_plan_sha256": sha(CLASSIFIER_PLAN),
            "private_score_journal_sha256": sha(journal), "assessed_views": audited["assessed_variants"],
            "stability_probe_counts": audited["stability_probe_counts"], "new_neural_calls": audited["new_neural_calls"], "vad_forwards": 0}


SCORE_BINDINGS = {"verify": score_inputs, "STAGE": SCORE, "OUTPUT": SCORE / "output", "result": score_result}
score_worker = scoped(retry.worker, {**SCORE_BINDINGS, "classify": downstream.classify})


def run(phase):
    stage, check = (GATE, verify) if phase == "gate" else (SCORE, score_verify)

    def launch(argv, **kwargs):
        require(argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(stage / "output")],
                "Only the frozen v284 worker may run")
        return subprocess.Popen([*argv[:3], MODULE, f"{phase}-worker"], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(runtime.run, {"TRIAL": stage.parent, "verify": check, "prerequisites": lambda *_: True,
                         "subprocess": proxy, "__file__": __file__})("authored")


def decision_report(cases, variants, observed):
    from scripts.audio_ai_only_rescue_v283 import apply_ai_only

    downstream.validate(variants, observed)
    scores = {(row["source_case"], row["view"]): score["decision"] for row, score in zip(variants, observed, strict=True)}
    counts, source_groups, changed = defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    added_groups = defaultdict(set)
    for case in cases:
        condition, family = case["condition"], case["lane"]
        treatment = "clean" if condition == "matched_clean" else "forced" if condition.endswith("_forced") else "equal_gain"
        mono = scores.get((case["case"], "mono"), "not_assessed")
        selected = scores.get((case["case"], "selected"), "not_assessed")
        ai_only = apply_ai_only(mono, selected)
        channel_priority = selected if selected != "not_assessed" else mono
        truth = "original" if family == "genuine" else "ai"
        strata = (("all", "all"), ("condition", condition), ("treatment", treatment), ("family", family),
                  ("family_condition", f"{family}/{condition}"))
        for stratum in strata:
            source_groups[stratum].add((family, case["source_group"]))
            if ai_only != mono:
                changed[stratum][f"{mono}_to_ai"] += 1
                added_groups[stratum, truth].add((family, case["source_group"]))
            for route, decision in zip(ROUTES, (mono, ai_only, channel_priority), strict=True):
                counts[stratum, route].update({decision: 1, "planned": 1,
                    "wrong_ai": int(truth == "original" and decision == "ai"),
                    "wrong_original": int(truth == "ai" and decision == "original"),
                    "correct_ai": int(truth == "ai" and decision == "ai"),
                    "correct_original": int(truth == "original" and decision == "original"),
                    "synthetic_without_ai": int(truth == "ai" and decision != "ai")})
    require(len(scores) == len(variants), "Every eligible view requires a unique terminal score")
    fields = (*DIRECTIONS, "planned", "wrong_ai", "wrong_original", "correct_ai", "correct_original", "synthetic_without_ai")
    return {kind: {name: {"connected_source_groups": len(groups), "ai_only_changes": dict(changed[kind, name]),
                         "groups_with_added_correct_ai": len(added_groups[(kind, name), "ai"]),
                         "groups_with_added_false_ai": len(added_groups[(kind, name), "original"]),
                         **{route: {field: counts[(kind, name), route][field] for field in fields} for route in ROUTES}}
                   for (group_kind, name), groups in source_groups.items() if group_kind == kind}
            for kind in ("all", "condition", "treatment", "family", "family_condition")}


def readout():
    cases, variants = score_verify(replay_gate=True)
    score_readout = scoped(retry.readout, {**SCORE_BINDINGS, "verify": lambda: variants, "__file__": __file__})()
    journal = SCORE / "output/private-observations.json"
    outcomes = decision_report(cases, variants, read(journal))
    require(all(outcomes["all"]["all"][route]["planned"] == PLANNED_CASES for route in ROUTES), "Every case needs each route disposition")
    gate_receipt = read(GATE / "output/result.json")
    return {"schema": "chipmunk.audio-native-stereo-holdout-complete/284", "status": "complete", "freeze_sha256": sha(PUBLIC),
            "role": "holdout", "gate_result_sha256": sha(GATE / "output/result.json"),
            "gate_profile_sha256": sha(GATE / "profile.json"), "vad_frame_forwards": gate_receipt["vad_frame_forwards"],
            "gate_coverage_by_condition": gate_receipt["by_condition"], "classifier": score_readout, "outcomes": outcomes,
            "serving_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run-gate": lambda: run("gate"), "gate-worker": gate_worker,
                "gate-readout": gate_readout, "seal-classifier": seal_classifier, "run-classifier": lambda: run("score"),
                "score-worker": score_worker, "readout": readout}
    parser.add_argument("action", choices=commands)
    result = commands[parser.parse_args().action]()
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
