"""Prepare prospective paired Turkish speech on shared early/late river excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_demand_controls_v204 as prior
from scripts import audio_short_edit_controls as controls
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE = prior.ROOT, prior.BASE
TRIAL = BASE / "audio-demand-crossover-v210"
STAGE, OUTPUT = TRIAL / "bind-00", TRIAL / "bind-00/output"
PUBLIC = ROOT / "docs/evaluation-panels/audio-demand-crossover-freeze-v210.json"
PREP = ROOT / "docs/evaluation-panels/audio-demand-controls-complete-v204.json"
TRANSFER = ROOT / "docs/evaluation-panels/audio-demand-transfer-complete-v208.json"
METHOD = ROOT / "docs/audio-demand-crossover-v210-method.md"
MODULE = "scripts.audio_demand_crossover_v210"
CONDITIONS = ("matched_clean", "river_early_0db", "river_late_0db")
CODE = (
    "backend/scripts/audio_demand_crossover_v210.py",
    "backend/tests/test_audio_demand_crossover_v210.py",
    "docs/audio-demand-crossover-v210-method.md",
    "backend/scripts/audio_demand_controls_v204.py",
    "backend/scripts/audio_short_edit_controls.py",
    "backend/scripts/video_appearance_runtime_v108.py",
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def paired_sources(sources):
    require(
        len(sources) == 48
        and all(origin == "genuine" for origin, _ in sources[:24])
        and all(origin == "synthetic" for origin, _ in sources[24:]),
        "Exact ordered v204 roster required",
    )
    return [(sources[i][1], sources[i + 24][1]) for i in range(24)]


def offsets(index):
    require(type(index) is int and 0 <= index < 24, "A fixed roster position is required")
    return {
        "early": (prior.START_SECONDS + prior.STEP_SECONDS * index) * prior.RATE,
        "late": (prior.START_SECONDS + prior.STEP_SECONDS * (index + 24)) * prior.RATE,
    }


def source_context():
    """Verify v204 preparation and river publisher bytes without reading v208 scores."""
    model_modules = {"app.detectors.audio.upgraded", "torch", "transformers", "onnxruntime"}
    require(not model_modules & sys.modules.keys(), "Model-free preparation process required")
    frozen, local, published = read(prior.PUBLIC), read(prior.TRIAL / "freeze.json"), read(PREP)
    stage, worker, journal = prior.STAGE, prior.OUTPUT / "result.json", prior.OUTPUT / "private-cases.jsonl"
    prior.genuine_route.verify_data_guard(
        read(stage / "profile.json"),
        stage,
        worker,
        frozen["code_sha256"]["backend/scripts/audio_demand_controls_v204.py"],
    )
    result = read(worker)
    require(
        frozen == local
        and published["schema"] == "chipmunk.audio-demand-controls-result/204"
        and published["status"] == "complete"
        and published["new_neural_calls"] == 0
        and all(published[key] == value for key, value in result.items())
        and published["freeze_sha256"] == sha(prior.PUBLIC)
        and published["profile_sha256"] == sha(stage / "profile.json")
        and published["private_cases_sha256"] == sha(journal)
        and all(sha(ROOT / name) == digest for name, digest in frozen["code_sha256"].items()),
        "Exact completed v204 preparation required",
    )
    asset = prior.SOURCES / prior.TAPES["river"]
    expected = frozen["publisher_assets"][asset.name]
    require(
        asset.is_file()
        and not asset.is_symlink()
        and asset.resolve(strict=True) == asset
        and asset.stat().st_size == expected["bytes"]
        and sha(asset) == expected["sha256"],
        "Exact frozen publisher river archive required",
    )
    with asset.open("rb") as stream:
        require(
            hashlib.file_digest(stream, "md5").hexdigest()
            == expected["publisher_md5"]
            == prior.MD5[asset.name],
            "Publisher river checksum differs",
        )
    sources = prior.foregrounds()
    paired_sources(sources)
    require(not model_modules & sys.modules.keys(), "Model-free source verification required")
    return frozen, sources


def identity():
    frozen, sources = source_context()
    return {
        "schema": "chipmunk.audio-demand-crossover-freeze/210",
        "planned_pairs": 24,
        "pairing": "same fixed v204 roster ordinal; no score-based selection",
        "conditions": list(CONDITIONS),
        "classes": ["genuine", "synthetic"],
        "sample_rate": prior.RATE,
        "prefix_rule": "minimum of the two original sample counts, starting at zero",
        "early_offset": "4 + 6 * ordinal seconds",
        "late_offset": "4 + 6 * (ordinal + 24) seconds",
        "snr_db": 0,
        "unavailable_rule": "both members of an incomplete pair remain unavailable",
        "v204_freeze_sha256": sha(prior.PUBLIC),
        "v204_result_sha256": sha(PREP),
        "v208_public_result_sha256": sha(TRANSFER),
        "river": frozen["publisher_assets"][prior.TAPES["river"]],
        "foregrounds": dict(Counter(f"{origin}_{row['status']}" for origin, row in sources)),
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "model_calls_permitted": False,
        "holdout_access_permitted": False,
    }, sources


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every earlier attempt")
    frozen, _ = identity()
    prior.tape("river")
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return frozen


def verify():
    frozen, sources = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == frozen, "Exact frozen crossover required")
    return sources


def mix_foreground(clean, excerpts):
    count = len(clean) // 2
    require(
        prior.MIN_SAMPLES <= count <= prior.MAX_SAMPLES
        and len(clean) == 2 * count
        and tuple(excerpts) == ("early", "late")
        and all(len(body) == len(clean) for body in excerpts.values()),
        "One common prefix and two exact river excerpts are required",
    )
    sampler = scoped(controls._samples, {"SAMPLES": count})
    mix = scoped(controls.mix_paired_noise, {"_samples": sampler})
    sources = {"source": clean}
    gain = min(
        mix(sources, reference="source", noise_pcm_s16le=noise).common_gain for noise in excerpts.values()
    )
    prepared = {}
    for name, noise in excerpts.items():
        treatment = mix(sources, reference="source", noise_pcm_s16le=noise, common_gain=gain)
        arm = treatment.variants["source"]
        clean_record = (arm.clean_s16le, None, None, gain)
        require(
            prepared.setdefault("matched_clean", clean_record) == clean_record,
            "Both river windows must share the same gain-matched clean control",
        )
        payload, residual, measured = arm.mixed_s16le[0], treatment.noise_s16le[0], arm.measured_snr_db[0]
        actual = np.frombuffer(payload, dtype="<i2").astype(np.int32) - np.frombuffer(
            arm.clean_s16le, dtype="<i2"
        ).astype(np.int32)
        require(
            np.array_equal(actual, np.frombuffer(residual, dtype="<i2")) and abs(measured) <= 0.1,
            "Exact quantized residual and measured 0 dB SNR required",
        )
        prepared[f"river_{name}_0db"] = (payload, residual, measured, gain)
    require(tuple(prepared) == CONDITIONS, "Fixed crossover conditions required")
    return prepared


def cases(sources, river):
    for index, (genuine, synthetic) in enumerate(paired_sources(sources)):
        members = {"genuine": genuine, "synthetic": synthetic}
        unavailable = [origin for origin, row in members.items() if row["status"] != "qualified"]
        mix = None
        if not unavailable:
            raw = {
                "genuine": prior.signed_pcm(prior.genuine.read_prepared_pcm(genuine)),
                "synthetic": prior.signed_pcm(prior.synthetic.read_pcm(synthetic)),
            }
            count = min(len(raw["genuine"]), len(raw["synthetic"])) // 2
            starts = offsets(index)
            excerpts = {key: river[2 * start : 2 * (start + count)] for key, start in starts.items()}
            require(
                all(len(body) == 2 * count for body in excerpts.values()),
                "River excerpt exceeds the frozen tape",
            )
            try:
                mix = {origin: mix_foreground(pcm[: 2 * count], excerpts) for origin, pcm in raw.items()}
            except controls.UnsupportedControl:
                unavailable = ["mix"]
        for origin, row in members.items():
            for condition in CONDITIONS:
                record = {
                    "case": f"pair-{index:02d}-{origin}-{condition}",
                    "pair": f"pair-{index:02d}",
                    "origin": origin,
                    "condition": condition,
                }
                if unavailable:
                    yield (
                        {
                            **record,
                            "status": "unavailable",
                            "reason": "pair_unavailable",
                            "unavailable_inputs": unavailable,
                        },
                        None,
                    )
                    continue
                payload, residual, measured, gain = mix[origin][condition]
                which = (
                    "early"
                    if condition == "river_early_0db"
                    else "late"
                    if condition == "river_late_0db"
                    else None
                )
                yield (
                    {
                        **record,
                        "status": "prepared",
                        "samples": count,
                        "foreground_wav_sha256": row["wav_sha256"],
                        "foreground_prefix_pcm_sha256": hashlib.sha256(raw[origin][: 2 * count]).hexdigest(),
                        "pcm_sha256": hashlib.sha256(payload).hexdigest(),
                        "gain": gain,
                        "background_sample_offset": starts[which] if which else None,
                        "background_pcm_sha256": hashlib.sha256(excerpts[which]).hexdigest()
                        if which
                        else None,
                        "background_residual_sha256": hashlib.sha256(residual).hexdigest()
                        if residual is not None
                        else None,
                        "measured_snr_db": measured,
                    },
                    payload,
                )


def worker():
    sources = verify()
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and not (STAGE / "profile.json").exists(),
        "Use the owned v108 data-only guard",
    )
    OUTPUT.mkdir(mode=0o700)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    river, counts = prior.tape("river"), Counter()
    journal = OUTPUT / "private-cases.jsonl"
    with journal.open("x") as stream:
        for row, payload in cases(sources, river):
            if payload is not None:
                row["wav_sha256"] = prior.save_wave(prepared / f"{row['case']}.wav", payload)
            counts[row["status"]] += 1
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-demand-crossover-result/210",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_cases_sha256": sha(journal),
            "outcomes": dict(counts),
            "planned_pairs": 24,
            "planned_cases": 144,
            "new_neural_calls": 0,
        },
    )


def run():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Exact guarded data-only worker required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker"], **kwargs)

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
    )("bind", 0)


def readout():
    sources = verify()
    journal, result_path = OUTPUT / "private-cases.jsonl", OUTPUT / "result.json"
    prior.genuine_route.verify_data_guard(
        read(STAGE / "profile.json"), STAGE, result_path, sha(Path(__file__))
    )
    result, rows = read(result_path), [json.loads(line) for line in journal.read_text().splitlines()]
    require(
        result["schema"] == "chipmunk.audio-demand-crossover-result/210"
        and result["status"] == "complete"
        and result["new_neural_calls"] == 0
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_cases_sha256"] == sha(journal)
        and result["planned_pairs"] == 24
        and result["planned_cases"] == len(rows) == 144,
        "Exact guarded crossover receipt required",
    )
    counts = {}
    for (expected, payload), row in zip(cases(sources, prior.tape("river")), rows, strict=True):
        if payload is not None:
            body = (OUTPUT / "prepared" / f"{expected['case']}.wav").read_bytes()
            require(prior.read_wave(body) == payload, "Saved crossover waveform differs")
            expected["wav_sha256"] = hashlib.sha256(body).hexdigest()
        require(row == expected, "Independent paired construction differs from the journal")
        counts.setdefault(row["origin"], {}).setdefault(row["condition"], Counter())[row["status"]] += 1
    require(
        result["outcomes"] == dict(Counter(row["status"] for row in rows)), "Crossover case coverage differs"
    )
    return {
        **result,
        "by_origin_and_condition": counts,
        "profile_sha256": sha(STAGE / "profile.json"),
        "scope": "Consumed Turkish development foregrounds and one river tape; no new detector outputs or population claim.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "readout", "_worker"))
    args = parser.parse_args()
    result = {"freeze": freeze, "run": run, "readout": readout, "_worker": worker}[args.action]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
