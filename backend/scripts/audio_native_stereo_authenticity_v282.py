"""Compare exact forced-stereo rescue views with admitted controls and matched clean."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from scripts import audio_native_stereo_gate_v279 as stereo
from scripts import audio_noise_speech_rescue_downstream_v270 as downstream
from scripts.video_appearance_corrected_v112 import scoped

retry, gate = downstream.retry, downstream.gate
ROOT, BASE, PANEL = retry.ROOT, retry.BASE, retry.PANEL
MODULE = "scripts.audio_native_stereo_authenticity_v282"
TRIAL = BASE / "audio-native-stereo-authenticity-v282"
STAGE, OUTPUT = TRIAL / "authored", TRIAL / "authored/output"
PUBLIC = PANEL / "audio-native-stereo-authenticity-freeze-v282.json"
GATE_SHA = "a37dabab269d0fc9b22a8f2f8f364b44ff5b9ba3428ea101e390a714cb5a0976"
RECEIPTS = {
    stereo.PUBLIC: "6a80567c7767ef208450c70e92d4d06719a9dc1899d363c3affe6cfc5aa9ed1c",
    retry.PUBLIC: downstream.RECEIPTS[retry.PUBLIC],
    downstream.RETRY_RESULT: downstream.RECEIPTS[downstream.RETRY_RESULT],
    downstream.PUBLIC: "88e2a68eb60b6b9c88298ea7f800496fb946508887641f9ec238cd271d3c459c",
}
CODE = (
    "backend/scripts/audio_native_stereo_authenticity_v282.py",
    "backend/tests/test_audio_native_stereo_authenticity_v282.py",
    "docs/audio-native-stereo-authenticity-v282-method.md",
    "backend/scripts/audio_native_stereo_gate_v279.py",
    "backend/scripts/audio_noise_speech_rescue_downstream_v270.py",
)
CLEAN = "matched_clean"
FORCED = ("river_forced", "washing_room_forced")
EXPECTED = {
    ("genuine", "rescued", FORCED[0]): 3,
    ("genuine", "admitted", FORCED[0]): 4,
    ("genuine", "admitted", FORCED[1]): 6,
    ("expressive", "rescued", FORCED[0]): 3,
    ("expressive", "rescued", FORCED[1]): 1,
    ("expressive", "admitted", FORCED[0]): 7,
    ("expressive", "admitted", FORCED[1]): 9,
}
EXPECTED_CLEAN = {"genuine": 7, "expressive": 11}
read, write, sha, require = retry.read, retry.write, retry.sha, retry.require
digest = stereo.pairs.stereo.digest


def select(cases, observed):
    indexed = {}
    for case, row in zip(cases, observed, strict=True):
        key = (case["source_slot"], case["condition"])
        require(row["case"] == case["case"] and key not in indexed and row["swap_invariant"],
                "Exact saved activity case binding required")
        indexed[key] = (case, row)
    counts, clean_cases, paired = Counter(), {}, []
    for (slot, condition), (case, row) in indexed.items():
        index = row["selected"]
        if index is None or not row["admitted"][f"ch{index}"]:
            continue
        mono_admitted = row["admitted"]["mono"]
        target = case["lane"] in EXPECTED_CLEAN and condition in FORCED
        require(mono_admitted or target, "Every v279 rescue must enter the forced-stereo comparison")
        if not target:
            continue
        cohort = "admitted" if mono_admitted else "rescued"
        require((slot, CLEAN) in indexed, "Exact matched clean for every included recording required")
        clean, eligibility = indexed[slot, CLEAN]
        require(eligibility["admitted"]["mono"] and clean["source_group"] == case["source_group"]
                and clean["lane"] == case["lane"] and clean["common_gain"] == case["common_gain"],
                "Matched clean must be gate-admitted and preserve source, family and common gain")
        clean_cases[slot] = clean
        counts[case["lane"], cohort, condition] += 1
        paired.append((case, clean, index, cohort, mono_admitted))
    require(counts == EXPECTED and Counter(row["lane"] for row in clean_cases.values()) == EXPECTED_CLEAN,
            "Every rescue, deterministic same-family control and unique matched clean is required")
    return paired, list(clean_cases.values())


def view(case, clean, kind, body, expected, admitted, cohort=None, index=None):
    require(len(body) == 2 * gate.SAMPLES and digest(body) == expected, "Exact v277 view PCM required")
    return {"case": f"{case['case']}-{kind}", "source_case": case["case"], "clean_case": f"{clean['case']}-clean",
            "view": kind, "condition": case["condition"], "arm": case["lane"], "cohort": cohort,
            "group": (case["lane"], case["source_group"]), "selected_channel": index,
            "speech_gate_admitted": admitted, "pcm_sha256": expected, "payload": body}


def mono_bytes(case, path):
    return (stereo.mono_pcm(case, path) * 32768).astype("<i2").tobytes()


def inputs():
    require(all(sha(path) == value for path, value in RECEIPTS.items()), "Exact predecessor receipts required")
    cases = stereo.verify()
    gate_result = scoped(stereo.readout, {"verify": lambda: cases})()
    encoded = (json.dumps(gate_result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    require(digest(encoded) == GATE_SHA and gate_result["overall"]["additional_admission"] == 7
            and gate_result["overall"]["planned"] == 185 and gate_result["vad_frame_forwards"] == 52170,
            "Exact v279 classifier-independent eligibility receipt required")
    journal = stereo.OUTPUT / "private-observations.jsonl"
    require(sha(journal) == gate_result["private_observations_sha256"], "Exact replayed activity journal required")
    paired, clean_cases = select(cases, [json.loads(line) for line in journal.read_text().splitlines()])
    output = []
    for case, clean, index, cohort, mono_admitted in paired:
        path, pcm = stereo.read_native(case)
        output.extend((
            view(case, clean, "mono", mono_bytes(case, path), case["mixed_pcm_sha256"], mono_admitted, cohort),
            view(case, clean, "selected", pcm[:, index].tobytes(), case["channel_pcm_sha256"][index], True, cohort, index),
        ))
    for clean in clean_cases:
        path, _ = stereo.read_native(clean)
        output.append(view(clean, clean, "clean", mono_bytes(clean, path), clean["mixed_pcm_sha256"], True))
    require(len(output) == len({row["case"] for row in output}) == 84, "Exact 84-view comparison required")
    return output


def plan(cases):
    return [{**{key: value for key, value in case.items() if key not in {"payload", "group"}},
             "group": list(case["group"])} for case in cases]


inherited_identity = scoped(downstream.identity, {"RECEIPTS": RECEIPTS, "CODE": CODE, "EXPECTED": {}, "CONDITIONS": FORCED})


def identity(cases):
    inherited = inherited_identity(cases)
    require(inherited["runtime"] == read(downstream.PUBLIC)["runtime"], "Exact v270 CPU classifier runtime required")
    inherited.pop("planned_source_groups_by_arm_and_cohort")
    return {**inherited, "schema": "chipmunk.audio-native-stereo-authenticity-freeze/282",
            "v279_gate_readout_sha256": GATE_SHA, "role": "development", "unique_clean_by_family": EXPECTED_CLEAN,
            "planned_pairs_by_family_cohort_condition": [
                {"family": family, "cohort": cohort, "condition": condition, "pairs": count}
                for (family, cohort, condition), count in EXPECTED.items()
            ]}


def summarize(cases, rows):
    audited = downstream.validate(cases, rows)
    scores = {row["case"]: row["decision"] for row in rows}
    metrics, unique_clean = {}, {}
    for case in cases:
        if case["view"] == "clean":
            unique_clean.setdefault(case["arm"], Counter())[scores[case["case"]]] += 1
            continue
        if case["view"] != "mono":
            continue
        key = (case["arm"], case["cohort"], case["condition"])
        group = metrics.setdefault(key, {name: Counter() for name in ("mono_gate_respecting", "mono_counterfactual", "selected", "matched_clean")})
        mono = scores[case["case"]]
        group["mono_gate_respecting"][mono if case["speech_gate_admitted"] else "not_assessed"] += 1
        if not case["speech_gate_admitted"]:
            group["mono_counterfactual"][mono] += 1
        group["selected"][scores[f"{case['source_case']}-selected"]] += 1
        group["matched_clean"][scores[case["clean_case"]]] += 1
    return {
        "assessed_variants": len(rows),
        "paired_direction_counts": [
            {"family": family, "cohort": cohort, "condition": condition, **{name: dict(count) for name, count in values.items()}}
            for (family, cohort, condition), values in metrics.items()
        ],
        "unique_clean_direction_counts": {name: dict(count) for name, count in unique_clean.items()},
        "stability_probe_counts": audited["stability_probe_counts"], "new_neural_calls": audited["new_neural_calls"],
        "vad_forwards": 0, "second_opinion_scored": False, "holdout_scored": False, "serving_changed": False,
    }


def result(cases, journal):
    return {"schema": "chipmunk.audio-native-stereo-authenticity-result/282", "status": "complete", "freeze_sha256": sha(PUBLIC),
            "v279_gate_readout_sha256": GATE_SHA, "private_observation_journal_sha256": sha(journal), **summarize(cases, read(journal))}


FREEZE_BINDINGS = {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "inputs": inputs, "plan": plan, "identity": identity}
preflight = scoped(retry.prior.preflight, FREEZE_BINDINGS)
freeze = scoped(retry.prior.freeze, FREEZE_BINDINGS)
verify = scoped(retry.prior.verify, FREEZE_BINDINGS)
WORKER_BINDINGS = {"verify": verify, "STAGE": STAGE, "OUTPUT": OUTPUT, "result": result}
worker = scoped(retry.worker, {**WORKER_BINDINGS, "classify": downstream.classify})
run = scoped(retry.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__})
readout = scoped(retry.readout, {**WORKER_BINDINGS, "__file__": __file__})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("action", choices=commands)
    output = commands[parser.parse_args().action]()
    if output is not None:
        print(json.dumps(output, indent=2, allow_nan=False))
