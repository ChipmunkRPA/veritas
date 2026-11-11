"""Score every v269 rescue and same-arm admitted control with matched clean audio."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import platform
import wave
from collections import Counter
from types import SimpleNamespace

from scripts import audio_noise_speech_gate_strong_noise_v269 as stress
from scripts import audio_rescued_speech_retry_v261 as retry
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL, gate = retry.ROOT, retry.BASE, retry.PANEL, retry.gate
MODULE = "scripts.audio_noise_speech_rescue_downstream_v270"
TRIAL = BASE / "audio-noise-speech-rescue-downstream-v270"
STAGE, OUTPUT = TRIAL / "authored", TRIAL / "authored/output"
PUBLIC = PANEL / "audio-noise-speech-rescue-downstream-freeze-v270.json"
STRESS_RESULT = PANEL / "audio-noise-speech-gate-strong-noise-complete-v269.json"
RETRY_RESULT = PANEL / "audio-rescued-speech-retry-complete-v261.json"
RECEIPTS = {
    stress.PUBLIC: "b7559aaa1c11c56eb92fbd4699ca1ab83f4f2648fe274625840c70adbb80cea5",
    STRESS_RESULT: "40d7b661032db949d00b4d3cd524d53ac12bb7d7db9c9276b46ec35b3a2c8139",
    retry.PUBLIC: "a6e0f56f392f79b445d39117e68ba7c9421bbccaeba5a049bc464481d3c3c3c9",
    RETRY_RESULT: "89656b3f98035b6429795094dac8d29c12fb00d58be14f2e32709d3b6a2c3bcc",
}
CODE = (
    "backend/scripts/audio_noise_speech_rescue_downstream_v270.py",
    "backend/tests/test_audio_noise_speech_rescue_downstream_v270.py",
    "docs/audio-noise-speech-rescue-downstream-v270-method.md",
)
CLEAN, NOISE = "matched_clean", "traffic_-10db"
CONDITIONS = (CLEAN, NOISE)
EXPECTED = {"genuine": {"rescued": 1, "admitted": 7}, "alania": {"rescued": 2, "admitted": 13}}
PILOT = gate.prior.baseline.serving.prior.pilot
read, write, sha, require = retry.read, retry.write, retry.sha, retry.require


def select(cases, groups, observed):
    indexed = {}
    for row, saved in zip(cases, observed, strict=True):
        spans = saved["spans"]
        key = (row["triplet"], row["arm"], row["condition"])
        require(
            row["case"] == saved["case"]
            and key not in indexed
            and ((spans is None) == (row["status"] == "unavailable")),
            "Exact unique saved eligibility roster required",
        )
        indexed[key] = (row, None if spans is None else gate.coverage(spans))
    selected, counts, seen = [], Counter(), set()
    for (triplet, arm, condition), (row, eligibility) in indexed.items():
        if condition != NOISE or arm not in EXPECTED or eligibility is None or not eligibility["rescue"]:
            continue
        cohort = "admitted" if eligibility["baseline"] else "rescued"
        require((triplet, arm, CLEAN) in indexed, "Every selected noisy group requires a matched clean row")
        clean, clean_eligibility = indexed[triplet, arm, CLEAN]
        group = (arm, groups[arm, triplet])
        require(
            group not in seen
            and clean_eligibility is not None
            and clean_eligibility["baseline"]
            and all(clean[key] == row[key] for key in ("source_wav_sha256", "source_prefix_sha256")),
            "Every selected noisy group requires its own baseline-admitted matched clean waveform",
        )
        seen.add(group)
        counts[arm, cohort] += 1
        selected.extend(
            {**case, "cohort": cohort, "group": group, "original_gate_admitted": state["baseline"]}
            for case, state in ((clean, clean_eligibility), (row, eligibility))
        )
    expected = {
        (arm, cohort): count for arm, cohorts in EXPECTED.items() for cohort, count in cohorts.items()
    }
    require(counts == expected, "Every rescued and baseline-admitted same-arm control must be present")
    return selected


def inputs():
    require(
        all(sha(path) == expected for path, expected in RECEIPTS.items()), "Exact published receipts required"
    )
    verified = stress.verify()
    frozen, cases, groups, background, _ = verified
    result = scoped(stress.readout, {"verify": lambda: verified})()
    require(
        not background
        and result == read(STRESS_RESULT)
        and result["status"] == "complete"
        and frozen["matched_prepared"] == 195
        and result["new_neural_calls"] == 18330,
        "Exact v269 resource and eligibility receipts must replay",
    )
    journal = stress.OUTPUT / "private-observations.json"
    return select(cases, groups, read(journal)["matched"])


def plan(cases):
    return [
        {
            **{
                key: case[key]
                for key in ("case", "cohort", "arm", "condition", "pcm_sha256", "original_gate_admitted")
            },
            "group": list(case["group"]),
        }
        for case in cases
    ]


def identity(cases):
    recipe = read(retry.PUBLIC)
    calibration_sha = sha(retry.prior.CALIBRATION)
    require(
        all(sha(ROOT / name) == digest for name, digest in recipe["code_sha256"].items())
        and calibration_sha == retry.prior.CALIBRATION_SHA == recipe["serving_calibration_sha256"]
        and all(sha(PILOT.WEIGHTS / name) == digest for name, digest in PILOT.ASSETS.items()),
        "Unchanged local voice scorer, calibration, weights and runtime code required",
    )
    stress.prior.sources.model_free()
    return {
        "schema": "chipmunk.audio-noise-speech-rescue-downstream-freeze/270",
        "source_receipts_sha256": {path.name: digest for path, digest in RECEIPTS.items()},
        "serving_calibration_sha256": calibration_sha,
        "serving_asset_sha256": PILOT.ASSETS,
        "inherited_v261_code_sha256": recipe["code_sha256"],
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "transformers")
            },
        },
        "planned_variants": len(cases),
        "planned_source_groups_by_arm_and_cohort": EXPECTED,
        "conditions": list(CONDITIONS),
        "sample_rate": gate.RATE,
        "samples_per_window": gate.SAMPLES,
        "maximum_classifier_forwards": 2 * len(cases),
        "vad_forwards_permitted": False,
        "second_opinion_permitted": False,
        "serving_change_permitted": False,
    }


def wav_bytes(case):
    payload = case["payload"]
    require(
        isinstance(payload, bytes)
        and len(payload) == 2 * gate.SAMPLES
        and stress.prior.digest(payload) == case["pcm_sha256"],
        "Exact frozen signed16 waveform required",
    )
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setparams((1, 2, gate.RATE, gate.SAMPLES, "NONE", "not compressed"))
        writer.writeframes(payload)
    return stream.getvalue()


validate = scoped(retry.prior.summarize, {"PLANNED": ()})


def summarize(cases, rows):
    audited = validate(cases, rows)
    groups = {
        arm: {cohort: {condition: Counter() for condition in CONDITIONS} for cohort in cohort_counts}
        for arm, cohort_counts in EXPECTED.items()
    }
    for case, row in zip(cases, rows, strict=True):
        groups[case["arm"]][case["cohort"]][case["condition"]][row["decision"]] += 1
    return {
        "assessed_variants": len(rows),
        "by_arm_cohort_and_condition": {
            arm: {
                cohort: {
                    condition: {
                        "assessed": values.total(),
                        "directions": {direction: values[direction] for direction in retry.prior.DIRECTIONS},
                    }
                    for condition, values in by_condition.items()
                }
                for cohort, by_condition in by_cohort.items()
            }
            for arm, by_cohort in groups.items()
        },
        "stability_probe_counts": audited["stability_probe_counts"],
        "new_neural_calls": audited["new_neural_calls"],
        "vad_forwards": 0,
        "second_opinion_scored": False,
        "serving_changed": False,
    }


def result(cases, journal):
    return {
        "schema": "chipmunk.audio-noise-speech-rescue-downstream-result/270",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "gate_result_sha256": RECEIPTS[STRESS_RESULT],
        "private_observation_journal_sha256": sha(journal),
        **summarize(cases, read(journal)),
    }


FREEZE_BINDINGS = {"TRIAL": TRIAL, "PUBLIC": PUBLIC, "inputs": inputs, "plan": plan, "identity": identity}
preflight = scoped(retry.prior.preflight, FREEZE_BINDINGS)
freeze = scoped(retry.prior.freeze, FREEZE_BINDINGS)
verify = scoped(retry.prior.verify, FREEZE_BINDINGS)
classify = scoped(
    retry.classify,
    {
        "gate": SimpleNamespace(
            prior=SimpleNamespace(baseline=gate.prior.baseline, baseline_waveform=wav_bytes)
        )
    },
)
WORKER_BINDINGS = {"verify": verify, "STAGE": STAGE, "OUTPUT": OUTPUT, "result": result}
worker = scoped(retry.worker, {**WORKER_BINDINGS, "classify": classify})
run = scoped(
    retry.run, {"TRIAL": TRIAL, "OUTPUT": OUTPUT, "verify": verify, "MODULE": MODULE, "__file__": __file__}
)
readout = scoped(retry.readout, {**WORKER_BINDINGS, "__file__": __file__})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"preflight": preflight, "freeze": freeze, "run": run, "worker": worker, "readout": readout}
    parser.add_argument("command", choices=commands)
    output = commands[parser.parse_args().command]()
    if output is not None:
        print(json.dumps(output, indent=2, allow_nan=False))
