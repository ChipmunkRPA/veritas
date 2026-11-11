"""Test the fixed speech-admission fallback on different sources and tape excerpts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from types import SimpleNamespace

from scripts import audio_fresh_shared_noise_v217 as sources
from scripts import audio_noise_speech_gate_v256 as gate
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = gate.ROOT, gate.BASE, gate.PANEL
MODULE = "scripts.audio_noise_speech_gate_transfer_v267"
TRIAL = BASE / "audio-noise-speech-gate-transfer-v267"
STAGE, OUTPUT = TRIAL / "authored", TRIAL / "authored/output"
PUBLIC = PANEL / "audio-noise-speech-gate-transfer-freeze-v267.json"
V217 = PANEL / "audio-fresh-shared-noise-complete-v217.json"
V248 = PANEL / "audio-matched-background-panel-complete-v248.json"
V256 = PANEL / "audio-noise-speech-gate-complete-v256.json"
V248_JOURNAL = BASE / "audio-matched-background-panel-v248/bind-00/output/private-cases.jsonl"
RECEIPTS = {
    sources.PUBLIC: "6217e14c017101a4ad7862c52f11daf2ccc65e62b0fa808ccb37beb2f8c38ecb",
    V217: "5fb45d9333ec42aee1221905f48d2e613f6c67a3dca277ad0959a80d35c270d2",
    V248: "af7b558a456d4bbb0e97dde96bf0fb8a9e402652cd514d1c061d32c99fa9a792",
    gate.PUBLIC: "a9da45b9baec98d3228bda6e765eddbe7e641ffc93292f8a34208938adc5804a",
    V256: "7150fa192232b946caa541e4c6896d4160810a3c10b0d54868fb96b5ea4413b3",
}
CODE = (
    "backend/scripts/audio_noise_speech_gate_transfer_v267.py",
    "backend/tests/test_audio_noise_speech_gate_transfer_v267.py",
    "docs/audio-noise-speech-gate-transfer-v267-method.md",
)
ARMS, TAPES, CONDITIONS = sources.shared.ARMS, sources.shared.TAPES, sources.CONDITIONS
RATE, SAMPLES, FRAME = gate.RATE, gate.SAMPLES, gate.FRAME
TRIPLETS, FIRST, STEP = 16, 99 * RATE, 6 * RATE
read, write, sha, require = gate.read, gate.write, gate.sha, gate.require


def digest(pcm):
    return hashlib.sha256(pcm).hexdigest()


def offsets(index):
    require(type(index) is int and 0 <= index < TRIPLETS, "One frozen triplet position required")
    return FIRST + index * STEP


def control_rows(pcms, mix_starts):
    """Use all complete, nonoverlapping late-grid cuts outside the mixture intervals."""
    rows = []
    used_mixes = {digest(gate.excerpt(pcm, start)) for pcm in pcms.values() for start in mix_starts}
    for name, pcm in sorted(pcms.items()):
        for start in range(FIRST, len(pcm) // 2 - SAMPLES + 1, SAMPLES):
            if any(start < mixed + SAMPLES and mixed < start + SAMPLES for mixed in mix_starts):
                continue
            fingerprint = digest(gate.excerpt(pcm, start))
            require(fingerprint not in used_mixes, "Background control duplicates a mixture excerpt")
            rows.append({"slot": name, "environment": name, "start_sample": start, "sha256": fingerprint})
    require(len({row["sha256"] for row in rows}) == len(rows), "Repeated background-only waveform")
    return rows


def verified_history():
    require(all(sha(path) == expected for path, expected in RECEIPTS.items()), "Published lineage differs")
    rule = read(gate.PUBLIC)
    require(
        rule["rescue_single_span_minimum_samples"] == RATE
        and rule["sample_rate"] == RATE
        and rule["samples_per_window"] == SAMPLES
        and rule["vad_frame_samples"] == FRAME
        and rule["vad_sha256"] == gate.VAD_SHA == sha(gate.VAD)
        and all(
            importlib.metadata.version(name) == version for name, version in rule["runtime_versions"].items()
        )
        and all(sha(ROOT / path) == expected for path, expected in rule["code_sha256"].items())
        and read(V256)["freeze_sha256"] == sha(gate.PUBLIC),
        "The exact v256 serving comparison and fallback must be retained",
    )
    prior = read(V248)
    require(sha(V248_JOURNAL) == prior["private_cases_sha256"], "Exact consumed v248 case journal required")
    compared = [json.loads(line) for line in V248_JOURNAL.read_text().splitlines()]
    require(len(compared) == 360, "The complete v248 roster is required for exposure checks")
    return rule, compared


def inputs():
    rule, compared = verified_history()
    previous = read(sources.PUBLIC)
    rosters = sources.verify()
    require(sources.readout() == read(V217), "The earlier v217 preparation must replay")
    require(
        sources.offsets(TRIPLETS - 1) + SAMPLES <= FIRST
        and all(sha(ROOT / path) == expected for path, expected in previous["code_sha256"].items()),
        "Later tape intervals must be disjoint from the prior grid and use the same mixer",
    )
    old_voices = {row["group"] for row in compared if row["arm"] == "alania"}
    new_voices = {row["voice_id"] for row in sources.alania.verify()}
    require(
        len(old_voices) == 24 and len(new_voices) == TRIPLETS and old_voices.isdisjoint(new_voices),
        "Alania catalog voice groups must be disjoint",
    )
    readouts = sources.source_readers()
    publisher = sources.publisher_assets()
    pcms = {name: sources.tape(name, publisher) for name in TAPES}
    mix_starts = [offsets(i) for i in range(TRIPLETS)]
    controls = control_rows(pcms, mix_starts)
    old_noise = {
        digest(gate.excerpt(pcm, sources.offsets(i))) for pcm in pcms.values() for i in range(TRIPLETS)
    }
    new_noise = {row["sha256"] for row in controls} | {
        digest(gate.excerpt(pcm, start)) for pcm in pcms.values() for start in mix_starts
    }
    require(new_noise.isdisjoint(old_noise), "An evaluation excerpt repeats a prior tape window")
    prepared = list(scoped(sources.cases, {"offsets": offsets})(rosters, pcms, readouts))
    cases = [{**row, "payload": pcm} for row, pcm in prepared]
    require(len(cases) == TRIPLETS * len(ARMS) * len(CONDITIONS), "Every planned foreground row is required")
    clean = [row for row in cases if row["condition"] == "matched_clean" and row["payload"] is not None]
    for key in ("source_wav_sha256", "source_prefix_sha256"):
        new = {row[key] for row in clean}
        old = {row[key] for row in compared if row["status"] == "prepared"}
        require(
            len(new) == len(clean) and new.isdisjoint(old),
            "A transfer foreground repeats v248 or another arm",
        )
    groups = {
        (arm, f"triplet-{i:02d}"): row["voice"] if arm == "drifting" else i
        for arm, rows in rosters.items()
        for i, row in enumerate(rows)
    }
    require(all(len(pcm) % 2 == 0 for pcm in pcms.values()), "Even signed16 byte lengths are required")
    sources.model_free()
    return rule, cases, groups, controls, pcms


def plan(cases, groups, controls):
    return {
        "matched": [
            {
                **{key: value for key, value in row.items() if key != "payload"},
                "source_group": groups[row["arm"], row["triplet"]],
            }
            for row in cases
        ],
        "unscreened_background": controls,
    }


def identity(cases, groups, controls, rule):
    return {
        "schema": "chipmunk.audio-noise-speech-gate-transfer-freeze/267",
        "source_receipts_sha256": {path.name: expected for path, expected in RECEIPTS.items()},
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "inherited_v256_code_sha256": rule["code_sha256"],
        "vad_sha256": rule["vad_sha256"],
        "runtime_versions": rule["runtime_versions"],
        "demand_asset_record_sha256": sources.ASSET_RECORD_SHA,
        "foreground_planned_by_arm": dict(
            Counter(row["arm"] for row in cases if row["condition"] == "matched_clean")
        ),
        "prepared_foreground_groups_by_arm": {
            arm: len(
                {
                    groups[arm, row["triplet"]]
                    for row in cases
                    if row["arm"] == arm
                    and row["condition"] == "matched_clean"
                    and row["payload"] is not None
                }
            )
            for arm in ARMS
        },
        "matched_planned": len(cases),
        "matched_prepared": sum(row["payload"] is not None for row in cases),
        "background_controls_by_environment": dict(Counter(row["environment"] for row in controls)),
        "mixture_offset_rule": "3s excerpts at 99 + 6 * ordinal seconds",
        "background_rule": "every complete 3s cut starting at 99s, excluding mixture intervals",
        "rescue_single_span_minimum_samples": rule["rescue_single_span_minimum_samples"],
        "speech_free_background_verified": False,
        "authenticity_detector_calls_permitted": False,
        "serving_change_permitted": False,
    }


def preflight():
    rule, cases, groups, controls, _ = inputs()
    return {**identity(cases, groups, controls, rule), "status": "ready_for_freeze", "model_calls": 0}


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    rule, cases, groups, controls, _ = inputs()
    locked = identity(cases, groups, controls, rule)
    TRIAL.mkdir(mode=0o700)
    path = TRIAL / "private-plan.json"
    write(path, plan(cases, groups, controls))
    locked["private_plan_sha256"] = sha(path)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return {"status": "frozen", "freeze_sha256": sha(PUBLIC), "model_calls": 0}


def verify():
    rule, cases, groups, controls, pcms = inputs()
    path = TRIAL / "private-plan.json"
    locked = {**identity(cases, groups, controls, rule), "private_plan_sha256": sha(path)}
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == locked and read(path) == plan(cases, groups, controls),
        "Exact prospective input freeze required",
    )
    return locked, cases, groups, controls, pcms


def summarize(cases, groups, controls, observed):
    require(
        set(observed) == {"matched", "unscreened_background"}
        and len(cases) == len(observed["matched"])
        and len(controls) == len(observed["unscreened_background"]),
        "Exact complete observation roster required",
    )
    grouped, background = defaultdict(list), defaultdict(list)
    rescued = defaultdict(set)
    for case, row in zip(cases, observed["matched"], strict=True):
        spans = row["spans"]
        require(
            row["case"] == case["case"] and (spans is None) == (case["payload"] is None),
            "Observation differs from the exact transfer input",
        )
        disposition = None if spans is None else gate.coverage(spans)
        key = (case["arm"], case["condition"])
        grouped[key].append(disposition)
        if disposition is not None and disposition["rescue"] and not disposition["baseline"]:
            rescued[key].add(groups[case["arm"], case["triplet"]])
    for case, row in zip(controls, observed["unscreened_background"], strict=True):
        require(row["control"] == case, "Background observation differs from its exact input")
        background[case["environment"]].append(gate.coverage(row["spans"]))
    return {
        "by_arm_and_condition": {
            arm: {
                condition: {
                    **gate.counts(grouped[arm, condition]),
                    "source_groups_with_additional_admissions": len(rescued[arm, condition]),
                }
                for condition in CONDITIONS
            }
            for arm in ARMS
        },
        "background_controls_by_environment": {
            name: gate.counts(rows) for name, rows in sorted(background.items())
        },
        "background_recordings": len({row["slot"] for row in controls}),
        "vad_frame_forwards": (sum(row["payload"] is not None for row in cases) + len(controls))
        * math.ceil(SAMPLES / FRAME),
        "speech_free_background_verified": False,
        "authenticity_detector_calls": 0,
        "serving_changed": False,
    }


def result(totals, journal):
    return {
        "schema": "chipmunk.audio-noise-speech-gate-transfer-result/267",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "previous_rule_result_sha256": RECEIPTS[V256],
        "private_observation_journal_sha256": sha(journal),
        "new_neural_calls": totals["vad_frame_forwards"],
        **totals,
    }


def mixture_waveform(row):
    require(digest(row["payload"]) == row["pcm_sha256"], "Exact signed16 mixture bytes required")
    return gate.prior.float_input(row["payload"])


worker = scoped(
    gate.worker,
    {
        "verify": verify,
        "STAGE": STAGE,
        "OUTPUT": OUTPUT,
        "summarize": summarize,
        "result": result,
        "prior": SimpleNamespace(candidate_waveform=mixture_waveform, float_input=gate.prior.float_input),
    },
)
run = scoped(
    gate.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__}
)
readout = scoped(
    gate.readout,
    {
        "verify": verify,
        "STAGE": STAGE,
        "OUTPUT": OUTPUT,
        "summarize": summarize,
        "result": result,
        "__file__": __file__,
    },
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("command", choices=commands)
    output = commands[parser.parse_args().command]()
    if output is not None:
        print(json.dumps(output, indent=2, allow_nan=False))
