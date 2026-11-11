"""Measure the unchanged speech route under two recorded environmental noises."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import wave
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

from scripts import audio_voxforge_tr_transfer_v201 as prior
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
TRIAL = BASE / "audio-demand-transfer-v208"
INPUT = BASE / "audio-demand-controls-v204/bind-00"
PUBLIC = PANEL / "audio-demand-transfer-freeze-v208.json"
PREP = PANEL / "audio-demand-controls-complete-v204.json"
PREP_FREEZE = PANEL / "audio-demand-controls-freeze-v204.json"
GENUINE = PANEL / "audio-voxforge-tr-transfer-complete-v201.json"
MODULE = "scripts.audio_demand_transfer_v208"
CONDITIONS = (
    "source_clean",
    "matched_clean",
    "washing_room_0db",
    "washing_room_5db",
    "river_0db",
    "river_5db",
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def waveform(row):
    filename = INPUT / "output/prepared" / f"{row['case']}.wav"
    require(filename.resolve() == filename and filename.is_file(), "Exact direct prepared waveform required")
    body = filename.read_bytes()
    require(hashlib.sha256(body).hexdigest() == row["wav_sha256"], "Saved waveform changed")
    with wave.open(io.BytesIO(body), "rb") as reader:
        require(
            (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes())
            == (1, 2, prior.pilot.RATE, row["samples"])
            and reader.getcomptype() == "NONE",
            "Exact mono signed16 production sample extent required",
        )
        pcm = reader.readframes(reader.getnframes())
    require(
        len(pcm) == 2 * row["samples"] and hashlib.sha256(pcm).hexdigest() == row["pcm_sha256"],
        "Signed production samples changed",
    )
    return body


def cases():
    result, frozen = read(PREP), read(PREP_FREEZE)
    journal = INPUT / "output/private-cases.jsonl"
    worker = INPUT / "output/result.json"
    code = frozen["code_sha256"]["backend/scripts/audio_demand_controls_v204.py"]
    prior.verify_data_guard(read(INPUT / "profile.json"), INPUT, worker, code)
    require(
        result["schema"] == "chipmunk.audio-demand-controls-result/204"
        and result["status"] == "complete"
        and all(result[k] == v for k, v in read(worker).items())
        and result["profile_sha256"] == sha(INPUT / "profile.json")
        and result["freeze_sha256"] == sha(PREP_FREEZE)
        and result["private_cases_sha256"] == sha(journal)
        and all(sha(ROOT / path) == expected for path, expected in frozen["code_sha256"].items())
        and tuple(frozen["conditions"]) == CONDITIONS,
        "Exact data-only source preparation required",
    )
    rows = records(journal)
    require(len(rows) == 48 * len(CONDITIONS), "Complete planned source and condition roster required")
    require(Counter(row["status"] for row in rows) == result["outcomes"], "Prepared dispositions differ")
    for i in range(48):
        source = f"source-{i:02d}"
        group = rows[i * len(CONDITIONS) : (i + 1) * len(CONDITIONS)]
        require(
            [row["case"] for row in group] == [f"{source}-{condition}" for condition in CONDITIONS]
            and tuple(row["condition"] for row in group) == CONDITIONS
            and all(
                row["source"] == source and row["origin"] == ("genuine" if i < 24 else "synthetic")
                for row in group
            ),
            "Exact ordered paired foreground design required",
        )
        for row in group:
            if row["status"] == "prepared":
                waveform(row)
            else:
                require(row["status"] == "unavailable", "Unrecognized preparation disposition")
    return rows


def earlier():
    references = []
    for stage, path in (
        (prior.TRIAL / "authored", GENUINE),
        (prior.pilot.ALANIA_STAGE, prior.pilot.ALANIA_RESULT),
    ):
        result = read(path)
        journal = stage / "output/private-observations.jsonl"
        guarded = prior.pilot.speech.qualification_profile(
            stage,
            result["new_neural_calls"],
            {},
            "audio_voxforge_tr_transfer_v201.py" if path == GENUINE else "audio_alania_transfer_v179.py",
        )
        require(
            guarded["private_observations_sha256"] == result["private_observations_sha256"] == sha(journal),
            "Exact prior clean speech reference required",
        )
        rows = records(journal)
        require(len(rows) == 24, "All prior planned source slots required")
        references.extend(rows)
    return references


def identity():
    prior.pilot.require_model_free()
    rows = cases()
    earlier()
    prior.verify()
    prior.pilot.require_model_free()
    files = (
        Path(__file__),
        ROOT / "backend/tests/test_audio_demand_transfer_v208.py",
        ROOT / "docs/audio-demand-transfer-v208-method.md",
        ROOT / "backend/scripts/video_appearance_runtime_v108.py",
    )
    return {
        "schema": "chipmunk.audio-demand-transfer-freeze/208",
        "preparation_result_sha256": sha(PREP),
        "serving_freeze_sha256": sha(prior.PUBLIC),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "planned_cases": len(rows),
        "conditions": list(CONDITIONS),
        "route_reference_language": "en",
        "calibration_change_permitted": False,
    }, rows


def freeze():
    require(not PUBLIC.exists() and not TRIAL.exists(), "Preserve any earlier transfer attempt")
    locked, _ = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    return locked


def verify():
    locked, rows = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == locked, "Exact guarded evaluation freeze required")
    return rows


def check_clean(observed, expected):
    require(
        all(observed[k] == expected[k] for k in ("route_status", "decision", "windows", "stability_probe")),
        "The unchanged source-clean route differs from the earlier result",
    )
    for key in ("raw", "english_reference_estimate_ai", "stability_log_odds_change"):
        a, b = observed.get(key), expected.get(key)
        require(
            (a is b is None)
            or (a is not None and b is not None and math.isclose(a, b, abs_tol=1e-6, rel_tol=0)),
            "Source-clean response failed numerical replay",
        )


def audit_case(row, case):
    require(
        all(row[k] == case[k] for k in ("case", "source", "condition", "origin")), "Outcome roster differs"
    )
    prior.pilot.check_case(
        {**row, "slot": case["case"]},
        {
            **case,
            "slot": case["case"],
            "status": "qualified" if case["status"] == "prepared" else "unavailable",
        },
    )


def start(stack, count):
    from app.detectors.audio.decode import read_wav
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.audio_alania_transfer_v179 import score_voice
    from scripts.run_audio_specialists_v15 import load_voice

    model = load_voice({"weights": str(prior.pilot.WEIGHTS), "assets": prior.pilot.ASSETS})
    activity = SpeechActivity(prior.pilot.WEIGHTS / "silero_vad.jit")
    stack.callback(model.register_forward_pre_hook(count).remove)

    def score(body):
        return score_voice(read_wav(body), model, activity, Deadline(120))

    return score


def worker(batch, output):
    rows = verify()
    stage = TRIAL / f"extract-{batch:02d}"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact owned fresh offline scoring worker required",
    )
    selected = rows[batch :: len(CONDITIONS)]
    clean = earlier() if batch == 0 else [None] * len(selected)
    prior.pilot.require_model_free()
    output.mkdir(mode=0o700)
    journal = output / "private-observations.jsonl"
    forwards = 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    with ExitStack() as stack, journal.open("x") as stream:
        score = start(stack, count)
        for case, reference in zip(selected, clean, strict=True):
            before = forwards
            outcome = score(waveform(case)) if case["status"] == "prepared" else prior.pilot.unavailable()
            if reference is not None:
                check_clean(outcome, reference)
            row = {k: case[k] for k in ("case", "source", "condition", "origin")}
            row.update(outcome, pella_forwards=forwards - before)
            audit_case(row, case)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    write(
        output / "result.json",
        {
            "status": "complete",
            "batch": batch,
            "condition": CONDITIONS[batch],
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": forwards,
            "source_clean_replays": len(selected) if batch == 0 else 0,
        },
    )


def run(batch):
    verify()

    def launch(argv, **kwargs):
        output = TRIAL / f"extract-{batch:02d}/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(output), str(batch)],
            "Exact owned scoring command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(batch), str(output)], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
        },
    )("extract", batch)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["origin"], row["condition"]].append(row)
    result = {}
    for (origin, condition), group in grouped.items():
        counts = Counter(row["decision"] for row in group)
        reference = "source_clean" if condition == "matched_clean" else "matched_clean"
        earlier = {row["source"]: row for row in grouped.get((origin, reference), [])}
        transitions = (
            Counter(f"{earlier[row['source']]['decision']}->{row['decision']}" for row in group)
            if condition != "source_clean"
            else {}
        )
        result.setdefault(origin, {})[condition] = {
            "planned": len(group),
            "scored": sum(row["route_status"] == "ok" for row in group),
            "speech_gate_rejected": sum(row["route_status"] == "not_applicable" for row in group),
            "source_preparation_unavailable": sum(
                row["route_status"] == "preparation_unavailable" for row in group
            ),
            "directions": {direction: counts[direction] for direction in prior.pilot.DIRECTIONS},
            "transitions_from": reference if condition != "source_clean" else None,
            "paired_transitions": transitions,
        }
    return result


def readout():
    cases = verify()
    rows, profiles, calls = [], [], 0
    clean = earlier()
    for batch, condition in enumerate(CONDITIONS):
        stage = TRIAL / f"extract-{batch:02d}"
        journal = stage / "output/private-observations.jsonl"
        result = read(stage / "output/result.json")
        selected = records(journal)
        count = sum(row["pella_forwards"] for row in selected)
        prior.pilot.speech.qualification_profile(stage, count, {}, Path(__file__).name, "extract", batch)
        require(
            result["batch"] == batch
            and result["condition"] == condition
            and result["private_observations_sha256"] == sha(journal)
            and result["freeze_sha256"] == sha(PUBLIC),
            "Exact complete scored batch required",
        )
        for i, (row, case) in enumerate(zip(selected, cases[batch :: len(CONDITIONS)], strict=True)):
            audit_case(row, case)
            if batch == 0:
                check_clean(row, clean[i])
        profiles.append(sha(stage / "profile.json"))
        calls += count
        rows.extend(selected)
    return {
        "schema": "chipmunk.audio-demand-transfer-result/208",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": calls,
        "profile_sha256": profiles,
        "source_clean_replays": 48,
        "results": summarize(rows),
        "scope": "Matched environmental perturbation on fixed Turkish foregrounds and two recordings. No new calibration or population estimate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    sub.add_parser("readout")
    for action in ("run", "_worker"):
        command = sub.add_parser(action)
        command.add_argument("batch", type=int, choices=range(len(CONDITIONS)))
        if action == "_worker":
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args.batch)
    elif args.action == "_worker":
        worker(args.batch, args.output)
    else:
        print(json.dumps({"freeze": freeze, "readout": readout}[args.action](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
