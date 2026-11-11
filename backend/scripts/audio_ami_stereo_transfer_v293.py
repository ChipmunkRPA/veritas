"""Transfer the frozen stereo decision rules to all 24 genuine AMI windows."""

from __future__ import annotations

import argparse
import json

from scripts import audio_ami_native_stereo_v291 as ami
from scripts import audio_native_stereo_holdout_v284 as previous
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = previous.ROOT, previous.BASE, previous.PANEL
stereo, retry, downstream = previous.stereo, previous.retry, previous.downstream
MODULE = "scripts.audio_ami_stereo_transfer_v293"
TRIAL = BASE / "audio-ami-stereo-transfer-v293"
PUBLIC = PANEL / "audio-ami-stereo-transfer-freeze-v293.json"
GATE, SCORE = TRIAL / "gate/authored", TRIAL / "classifier/authored"
CLASSIFIER_PLAN = TRIAL / "classifier-plan.json"
V291 = PANEL / "audio-ami-native-stereo-complete-v291.json"
RECEIPTS = {
    V291: "70b603a680b7fb6756ff0e9f9d0d1bb0ea83118e0f4f98a1a388235dfbc34312",
    previous.PUBLIC: "b0091e894198242e2ccf86ed779b5181964416491f5a0da36aa8c25bdfc59b23",
    PANEL / "audio-native-stereo-holdout-complete-v284.json": "7cba9085e038a459383cf023e91146eedfdde46a196b8e318a6574996ea7294f",
}
CODE = (*previous.CODE, "backend/scripts/audio_ami_native_stereo_v291.py",
        "backend/scripts/audio_ami_stereo_transfer_v293.py", "backend/tests/test_audio_ami_stereo_transfer_v293.py",
        "docs/audio-ami-stereo-transfer-v293-method.md")
PLANNED, MAX_VIEWS = 24, 48
read, sha, require = previous.read, previous.sha, previous.require


def bind_rows(rows):
    planned = [(session, index, start) for session in ami.SESSIONS for index, start in enumerate(ami.WINDOWS)]
    require(len(rows) == len(planned) == PLANNED, "All 24 exact AMI windows are required")
    bound = []
    for (session, index, start), row in zip(planned, rows, strict=True):
        end = start + ami.SAMPLES
        require(row["session"] == session and row["case"] == f"{session}-{index:02d}"
                and row["source_range"] == {"start_sample": start, "end_sample": end, "time_base": f"1/{ami.RATE}"}
                and row["source_samples"] >= end and row["source_timeline"] == stereo.TIMELINE
                and len(row["channel_pcm_sha256"]) == 2,
                "Exact ordered AMI source intervals and clock are required")
        bound.append({**row, "lane": "genuine", "condition": session, "source_group": session,
                      "wav_sha256": row["stereo_wav_sha256"]})
    return bound


def source_metadata():
    require(all(sha(path) == expected for path, expected in RECEIPTS.items()), "Exact v291 and v284 published receipts required")
    ami.verify()
    result, private = read(V291), read(ami.RESULT)
    require(result["status"] == "complete" and result["planned_cases"] == result["cases_replayed"] == PLANNED
            and result["qualified_by_session"] == {session: len(ami.WINDOWS) for session in ami.SESSIONS}
            and result["qualification_sha256"] == sha(ami.RESULT)
            and result["freeze_sha256"] == private["freeze_sha256"] == sha(ami.FREEZE)
            and result["acquisition_sha256"] == private["acquisition_sha256"] == sha(ami.ACQUISITION)
            and result["model_calls"] == private["model_calls"] == 0,
            "Exact completed model-free AMI source qualification required")
    return bind_rows(private["cases"]), result


runtime_identity = scoped(downstream.identity, {"RECEIPTS": RECEIPTS, "CODE": CODE, "EXPECTED": {}, "CONDITIONS": ()})


def identity(cases, source):
    inherited, v284 = runtime_identity([]), read(previous.PUBLIC)
    fields = ("serving_calibration_sha256", "serving_asset_sha256", "inherited_v261_code_sha256", "runtime", "sample_rate", "samples_per_window")
    require(all(inherited[key] == v284[key] for key in fields)
            and all(inherited["code_sha256"][name] == value for name, value in v284["code_sha256"].items())
            and sha(stereo.gate.VAD) == stereo.gate.VAD_SHA,
            "Exact v279 activity, v283 policy and v284 classifier implementation and runtime required")
    return {**{key: inherited[key] for key in fields}, "schema": "chipmunk.audio-ami-stereo-transfer-freeze/293",
            "source_receipts_sha256": inherited["source_receipts_sha256"], "code_sha256": inherited["code_sha256"],
            "v291_qualification_sha256": source["qualification_sha256"], "planned_cases": len(cases),
            "sessions": source["qualified_by_session"], "sample_truth": "genuine_only", "vad_frame_forwards": PLANNED * 3 * stereo.FORWARDS_PER_VIEW,
            "max_classifier_views": MAX_VIEWS, "max_classifier_forwards": 2 * MAX_VIEWS,
            "gate_and_selection": "v279 without changes", "policy": "v283 AI-only rescue without changes"}


FREEZE_BINDINGS = {"source_metadata": source_metadata, "identity": identity, "PUBLIC": PUBLIC, "TRIAL": TRIAL, "GATE": GATE, "SCORE": SCORE}
preflight = scoped(previous.preflight, FREEZE_BINDINGS)
freeze = scoped(previous.freeze, FREEZE_BINDINGS)
verify = scoped(previous.verify, FREEZE_BINDINGS)


def case_path(row):
    path = ami.MEDIA / "qualified" / f"{row['case']}.wav"
    require(path.is_file() and path.resolve(strict=True) == path and sha(path) == row["wav_sha256"],
            "Exact sealed v291 AMI WAV required")
    return path


read_native = scoped(stereo.read_native, {"case_path": case_path})


def gate_result(cases, observed, journal):
    totals = stereo.summarize(cases, observed)
    return {**totals, "schema": "chipmunk.audio-ami-stereo-transfer-gate-result/293", "status": "complete",
            "freeze_sha256": sha(PUBLIC), "private_observations_sha256": sha(journal),
            "new_neural_calls": totals["vad_frame_forwards"], "holdout_cases_opened": len(cases)}


GATE_BINDINGS = {"verify": verify, "STAGE": GATE, "OUTPUT": GATE / "output", "read_native": read_native, "result": gate_result}
gate_worker = scoped(stereo.worker, GATE_BINDINGS)
gate_readout = scoped(stereo.readout, {**GATE_BINDINGS, "__file__": __file__})
views_for = scoped(previous.views_for, {"MAX_VIEWS": MAX_VIEWS})
inherited_score_roster = scoped(previous.score_roster, {"verify": verify, "gate_readout": gate_readout, "views_for": views_for,
                                                       "GATE": GATE, "PUBLIC": PUBLIC, "PLANNED_CASES": PLANNED})


def score_roster(replay_gate=False):
    cases, variants, binding = inherited_score_roster(replay_gate)
    return cases, variants, {**binding, "schema": "chipmunk.audio-ami-stereo-transfer-classifier-plan/293"}


seal_classifier = scoped(previous.seal_classifier, {"CLASSIFIER_PLAN": CLASSIFIER_PLAN, "score_roster": score_roster})
score_verify = scoped(previous.score_verify, {"CLASSIFIER_PLAN": CLASSIFIER_PLAN, "score_roster": score_roster})
score_inputs = scoped(previous.score_inputs, {"score_verify": score_verify, "read_native": read_native})
inherited_score_result = scoped(previous.score_result, {"PUBLIC": PUBLIC, "CLASSIFIER_PLAN": CLASSIFIER_PLAN})


def score_result(variants, journal):
    return {**inherited_score_result(variants, journal), "schema": "chipmunk.audio-ami-stereo-transfer-classifier-result/293"}


SCORE_BINDINGS = {"verify": score_inputs, "STAGE": SCORE, "OUTPUT": SCORE / "output", "result": score_result}
score_worker = scoped(retry.worker, {**SCORE_BINDINGS, "classify": downstream.classify})
run = scoped(previous.run, {"GATE": GATE, "SCORE": SCORE, "verify": verify, "score_verify": score_verify,
                            "MODULE": MODULE, "__file__": __file__})


def genuine_summary(group):
    summary = {}
    for name in ("mono", "ai_only_rescue"):
        row = group[name]
        summary[name] = {"planned": row["planned"], "assessed": row["planned"] - row["not_assessed"],
                         "false_ai": row["wrong_ai"], "original": row["original"], "inconclusive": row["inconclusive"],
                         "unassessed": row["not_assessed"]}
    return summary


def readout():
    cases, variants = score_verify(replay_gate=True)
    classifier = scoped(retry.readout, {**SCORE_BINDINGS, "verify": lambda: variants, "__file__": __file__})()
    outcomes = previous.decision_report(cases, variants, read(SCORE / "output/private-observations.json"))
    by_session = {session: genuine_summary(outcomes["condition"][session]) for session in ami.SESSIONS}
    overall = genuine_summary(outcomes["all"]["all"])
    require(all(sum(group[route]["planned"] for group in by_session.values()) == overall[route]["planned"] == PLANNED
                for route in overall), "Every AMI window must receive both decisions")
    gate_path = GATE / "output/result.json"
    gate = read(gate_path)
    coverage = ("planned", "mono_admitted", "selection_available", "selected_admitted", "additional_admission",
                "mono_or_selected_admitted", "swap_invariant")
    return {"schema": "chipmunk.audio-ami-stereo-transfer-complete/293", "status": "complete", "freeze_sha256": sha(PUBLIC),
            "source": "v291 AMI genuine-only natural stereo", "sessions": len(by_session), "planned_cases": PLANNED,
            "gate_result_sha256": sha(gate_path), "gate_profile_sha256": sha(GATE / "profile.json"),
            "vad_frame_forwards": gate["vad_frame_forwards"],
            "gate_by_session": {session: {key: gate["by_condition"][session][key] for key in coverage} for session in ami.SESSIONS},
            "overall": overall, "by_session": by_session, "classifier": classifier, "serving_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run-gate": lambda: run("gate"), "gate-worker": gate_worker,
                "gate-readout": gate_readout, "seal-classifier": seal_classifier, "run-classifier": lambda: run("score"),
                "score-worker": score_worker, "readout": readout}
    parser.add_argument("action", choices=commands)
    result = commands[parser.parse_args().action]()
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
