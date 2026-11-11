"""Qualify native three-second feature extraction with the unchanged v13 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import wave
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_fresh_shared_noise_score_v219 as prior
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require
PILOT = prior.serving.prior.pilot
TRIAL = BASE / "audio-wavlm-three-second-v226"
PUBLIC = PANEL / "audio-wavlm-three-second-freeze-v226.json"
PRIOR_RESULT = PANEL / "audio-fresh-shared-noise-score-complete-v219.json"
V13_FREEZE = PANEL / "audio-v13-wavlm-freeze.json"
V13_LOCAL_LOCK = BASE / "audio-v13-wavlm/lock.json"
WEIGHTS = ROOT / "models/research/wavlm-v13"
MODULE = "scripts.audio_wavlm_three_second_v226"
V13_CODE = "backend/scripts/audio_wavlm_features_v13.py"
CONDITION = "traffic_0db"
ARMS = prior.ARMS
RATE, SAMPLES, FRAMES, WIDTH, DIMENSIONS = 16000, 48000, 149, 768, 4608
LAYERS = (4, 8, 12)
REPEAT_ARM = ARMS[0]
FORWARDS = len(ARMS) + 1
CODE = (
    "backend/scripts/audio_wavlm_three_second_v226.py",
    "backend/tests/test_audio_wavlm_three_second_v226.py",
    "docs/audio-wavlm-three-second-v226-method.md",
    "backend/scripts/video_appearance_runtime_v108.py",
)
INSTALLED = {
    "transformers.models.wavlm.modeling_wavlm": (
        "transformers",
        "transformers/models/wavlm/modeling_wavlm.py",
    ),
    "transformers.models.wavlm.configuration_wavlm": (
        "transformers",
        "transformers/models/wavlm/configuration_wavlm.py",
    ),
    "torch.nn.functional": ("torch", "torch/nn/functional.py"),
}


def convolution_clock(samples, kernels, strides):
    require(len(kernels) == len(strides) and bool(kernels), "Complete convolution definition required")
    frames, receptive, step = samples, 1, 1
    for kernel, stride in zip(kernels, strides, strict=True):
        require(
            type(kernel) is type(stride) is int and kernel > 0 and stride > 0,
            "Positive convolution geometry required",
        )
        frames = (frames - kernel) // stride + 1
        receptive += (kernel - 1) * step
        step *= stride
    require(frames > 0, "No complete convolution frame")
    return {
        "frames": frames,
        "stride_samples": step,
        "receptive_field_samples": receptive,
        "last_receptive_range": [(frames - 1) * step, (frames - 1) * step + receptive],
    }


def checkpoint():
    published = read(V13_FREEZE)
    lock = published["lock"]
    backbone = lock["method"]["backbone"]
    require(
        published["schema"] == "chipmunk.audio-wavlm-public-freeze/13"
        and published["private_lock_sha256"] == sha(V13_LOCAL_LOCK)
        and sha(ROOT / V13_CODE) == lock["implementation_sha256"][V13_CODE]
        and backbone["revision"] == "4c66d4806a428f2e922ccfa1a962776e232d487b"
        and "CC-BY-SA-3.0" in backbone["weight_terms"]
        and backbone["layer_ids"] == list(LAYERS)
        and backbone["width"] == WIDTH
        and backbone["states"] == 13
        and backbone["feature_clock"]
        == {"frames": 199, "stride_samples": 320, "receptive_field_samples": 400},
        "Exact commercial-permission v13 checkpoint and extractor required",
    )
    for name, commitment in lock["pretrained_assets"].items():
        path = WEIGHTS / name
        require(
            path.resolve() == path
            and path.is_file()
            and path.stat().st_size == commitment["bytes"]
            and sha(path) == commitment["sha256"] == backbone["assets_sha256"][name],
            "Pinned WavLM source, weights, preprocessing or license bytes differ",
        )
    config, preprocessing = read(WEIGHTS / "config.json"), read(WEIGHTS / "preprocessor_config.json")
    clock = convolution_clock(SAMPLES, config["conv_kernel"], config["conv_stride"])
    require(
        preprocessing["sampling_rate"] == RATE
        and preprocessing["do_normalize"] is False
        and clock
        == {
            "frames": FRAMES,
            "stride_samples": 320,
            "receptive_field_samples": 400,
            "last_receptive_range": [47360, 47760],
        },
        "Exact unpadded native three-second WavLM clock required",
    )
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    implementations = {
        name: sha(Path(importlib.metadata.distribution(package).locate_file(path)))
        for name, (package, path) in INSTALLED.items()
    }
    original = lock["runtime"]
    require(
        versions
        == {
            "torch": original["torch"],
            "numpy": original["numpy"],
            "transformers": original["packages"]["transformers"],
        }
        and implementations == {name: lock["installed_implementation_sha256"][name] for name in INSTALLED}
        and platform.python_version() == original["python"]
        and platform.system() == original["platform"]
        and platform.machine() == original["machine"]
        and (original["device"], original["dtype"], original["threads"]) == ("cpu", "float32", 4),
        "Exact v13 CPU package and implementation lineage required",
    )
    return {
        "v13_freeze_sha256": sha(V13_FREEZE),
        "revision": backbone["revision"],
        "assets": lock["pretrained_assets"],
        "runtime_packages": versions,
        "installed_implementation_sha256": implementations,
        "clock": clock,
    }


def choose(prepared, observed):
    selected = {}
    for case, result in zip(prepared, observed, strict=True):
        require(
            all(case[key] == result[key] for key in ("case", "triplet", "arm", "condition")),
            "Recorded case identity differs",
        )
        if case["condition"] == CONDITION and result["route_status"] == "ok":
            require(
                case["status"] == "prepared"
                and case["samples"] == SAMPLES
                and result["windows"] == [[0, SAMPLES]],
                "Exact observed-admitted three-second support required",
            )
            selected.setdefault(case["triplet"], {})[case["arm"]] = case
    triplet = next((rows for rows in selected.values() if set(rows) == set(ARMS)), None)
    require(triplet is not None, "A fully admitted traffic-noise triplet is required")
    result = [triplet[arm] for arm in ARMS]
    require(
        len({row["pcm_sha256"] for row in result}) == len(ARMS)
        and len({row["noise_residual_sha256"] for row in result}) == 1,
        "Distinct foreground waveforms with the same exact background required",
    )
    return result


def waveform(case):
    body = prior.waveform(case)
    with wave.open(io.BytesIO(body), "rb") as stream:
        pcm = np.frombuffer(stream.readframes(SAMPLES), dtype="<i2").astype(np.float32) / np.float32(32768)
    require(
        pcm.shape == (SAMPLES,) and np.isfinite(pcm).all(), "Exact finite unpadded production PCM required"
    )
    return pcm


def input_context():
    PILOT.require_model_free()
    # The existing v217 admission checks the complete guard/metadata receipt.
    # Reopen only the three waveforms selected from that immutable roster below.
    prepared = scoped(prior.cases, {"waveform": lambda _case: None})()
    locked = read(prior.PUBLIC)
    require(
        locked == read(prior.TRIAL / "freeze.json")
        and locked["schema"] == "chipmunk.audio-fresh-shared-noise-score-freeze/219"
        and locked["v217_preparation_sha256"] == sha(prior.PREP)
        and locked["v217_freeze_sha256"] == sha(prior.PREP_FREEZE)
        and locked["route_freeze_sha256"] == sha(prior.serving.prior.PUBLIC)
        and all(sha(ROOT / name) == checksum for name, checksum in locked["code_sha256"].items())
        and tuple(locked["conditions"]) == prior.CONDITIONS
        and locked["holdout_access_permitted"] is locked["calibration_change_permitted"] is False,
        "Exact completed v219 evaluation identity required",
    )
    batches = [prior.completed_batch(index, prepared) for index in range(len(prior.CONDITIONS))]
    aggregate = scoped(
        prior.readout,
        {"verify": lambda: prepared, "completed_batch": lambda batch, _rows: batches[batch]},
    )()
    require(
        aggregate == read(PRIOR_RESULT)
        and aggregate["schema"] == "chipmunk.audio-fresh-shared-noise-score-result/219"
        and aggregate["status"] == "complete",
        "Exact published v219 aggregate and guarded case journals required",
    )
    index = prior.CONDITIONS.index(CONDITION)
    rows, profile, _calls = batches[index]
    picked = choose(prepared[index :: len(prior.CONDITIONS)], rows)
    for case in picked:
        waveform(case)
    return picked, {
        "v217_freeze_sha256": sha(prior.PREP_FREEZE),
        "v217_result_sha256": sha(prior.PREP),
        "v219_freeze_sha256": sha(prior.PUBLIC),
        "v219_result_sha256": sha(PRIOR_RESULT),
        "v219_condition_profile_sha256": profile,
        "v219_condition_journal_sha256": sha(
            prior.TRIAL / f"extract-{index:02d}/output/private-observations.jsonl"
        ),
    }


def encoded(value):
    return json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"


def identity():
    sources, bindings = input_context()
    model = checkpoint()
    PILOT.require_model_free()
    return {
        "schema": "chipmunk.audio-wavlm-three-second-freeze/226",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "checkpoint": model,
        **bindings,
        "private_plan_sha256": hashlib.sha256(encoded(sources)).hexdigest(),
        "condition": CONDITION,
        "arms": list(ARMS),
        "repeated_arm": REPEAT_ARM,
        "new_encoder_forwards": FORWARDS,
        "input_samples": SAMPLES,
        "sample_rate": RATE,
        "pool_layers": list(LAYERS),
        "output_dimensions": DIMENSIONS,
        "candidate_calls_permitted": False,
        "fitting_permitted": False,
    }, sources


def preflight():
    lock, sources = identity()
    return {
        "status": "ready_for_freeze",
        "selected_inputs": len(sources),
        "condition": lock["condition"],
        "input_samples": lock["input_samples"],
        "expected_frames": lock["checkpoint"]["clock"]["frames"],
        "model_calls": 0,
    }


def verify():
    lock, sources = identity()
    require(
        lock == read(PUBLIC) == read(TRIAL / "freeze.json")
        and (TRIAL / "private-plan.json").read_bytes() == encoded(sources),
        "Exact prospectively frozen v226 source and runtime required",
    )
    return sources


def freeze():
    require(not PUBLIC.exists() and not TRIAL.exists(), "Preserve every earlier v226 attempt")
    lock, sources = identity()
    TRIAL.mkdir(mode=0o700)
    with (TRIAL / "private-plan.json").open("xb") as stream:
        stream.write(encoded(sources))
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return {"status": "frozen", "selected_inputs": len(sources), "model_calls": 0}


def float_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def extract(pcm, model):
    import torch

    from scripts.audio_wavlm_features_v13 import pool

    require(
        pcm.dtype == np.float32 and pcm.shape == (SAMPLES,) and np.isfinite(pcm).all(),
        "Exact finite three-second input required",
    )
    values = torch.from_numpy(np.ascontiguousarray(pcm).copy())[None]
    original_sha = float_sha(pcm)
    with torch.inference_mode():
        result = model(
            values,
            attention_mask=torch.ones_like(values, dtype=torch.long),
            output_hidden_states=True,
            return_dict=True,
        )
        states = result.hidden_states
        require(
            result.last_hidden_state.shape == (1, FRAMES, WIDTH)
            and len(states) == 13
            and all(
                isinstance(state, torch.Tensor)
                and state.device.type == "cpu"
                and state.dtype == torch.float32
                and state.shape == (1, FRAMES, WIDTH)
                and torch.isfinite(state).all().item()
                for state in states
            ),
            "Exact finite native 149-frame CPU feature clock required",
        )
        record = pool(states, torch.ones((1, FRAMES), dtype=torch.long))
        state_sha = [float_sha(states[i].detach().numpy()) for i in LAYERS]
    require(
        original_sha == float_sha(values.detach().numpy()) == float_sha(pcm),
        "Encoder changed original input bytes",
    )
    return {**record, "input_pcm_sha256": original_sha, "pooled_state_sha256": state_sha}


def check_observation(row, case, repeat):
    features = np.asarray(row["wavlm"], dtype="<f4")
    require(
        row["case"] == case["case"]
        and row["arm"] == case["arm"]
        and row["repeat"] is repeat
        and row["feature_frames"] == row["valid_feature_frames"] == FRAMES
        and features.shape == (DIMENSIONS,)
        and np.isfinite(features).all()
        and features.tolist() == row["wavlm"]
        and row["wavlm_sha256"] == float_sha(features)
        and row["input_pcm_sha256"] == float_sha(waveform(case))
        and len(row["pooled_state_sha256"]) == len(LAYERS)
        and all(
            isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
            for digest in row["pooled_state_sha256"]
        )
        and row["encoder_forwards"] == 1,
        "Exact private three-second feature and source commitments required",
    )


def check_repeat(rows):
    first, repeated = rows[0], rows[-1]
    require(
        first["arm"] == repeated["arm"] == REPEAT_ARM
        and first["repeat"] is False
        and repeated["repeat"] is True
        and {k: v for k, v in first.items() if k != "repeat"}
        == {k: v for k, v in repeated.items() if k != "repeat"},
        "Repeated exact waveform must reproduce the selected hidden and pooled features",
    )


def worker(output):
    stage = TRIAL / "authored"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact guarded offline qualification required",
    )
    sources = verify()
    PILOT.require_model_free()
    import torch

    from scripts.audio_wavlm_features_v13 import load

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = load(WEIGHTS)
    output.mkdir(mode=0o700)
    journal = output / "private-features.jsonl"
    rows, calls = [], 0

    def count(_module, _args):
        nonlocal calls
        calls += 1

    with ExitStack() as stack, journal.open("x") as stream:
        stack.callback(model.register_forward_pre_hook(count).remove)
        for i, case in enumerate([*sources, sources[0]]):
            before = calls
            result = extract(waveform(case), model)
            row = {
                "case": case["case"],
                "arm": case["arm"],
                "repeat": i == len(sources),
                **result,
                "encoder_forwards": calls - before,
            }
            check_observation(row, case, i == len(sources))
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            rows.append(row)
    check_repeat(rows)
    require(calls == FORWARDS, "Expected exactly three native inputs and one repeat")
    write(
        output / "result.json",
        {
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_features_sha256": sha(journal),
            "new_neural_calls": calls,
            "repeated_exact": True,
        },
    )


def run():
    def launch(argv, **kwargs):
        output = TRIAL / "authored/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)],
            "Exact guarded v226 invocation required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", str(output)], **kwargs)

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
    )("authored")


def readout():
    cases = verify()
    stage = TRIAL / "authored"
    journal = stage / "output/private-features.jsonl"
    rows = prior.serving.records(journal)
    receipt = PILOT.speech.qualification_profile(stage, FORWARDS, {}, Path(__file__).name)
    require(
        receipt["freeze_sha256"] == sha(PUBLIC)
        and receipt["private_features_sha256"] == sha(journal)
        and receipt["repeated_exact"] is True,
        "Exact completed v226 feature receipt required",
    )
    for i, (row, case) in enumerate(zip(rows, [*cases, cases[0]], strict=True)):
        check_observation(row, case, i == len(cases))
    check_repeat(rows)
    profile = read(stage / "profile.json")
    return {
        "schema": "chipmunk.audio-wavlm-three-second-result/226",
        **receipt,
        "resource_profile_sha256": sha(stage / "profile.json"),
        "worker_duration_s": profile["duration_s"],
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "condition": CONDITION,
        "arms_qualified": list(ARMS),
        "input_samples": SAMPLES,
        "feature_frames": FRAMES,
        "feature_dimensions": DIMENSIONS,
        "candidate_scores": 0,
        "new_fitting": False,
        "defaults_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "freeze", "run", "readout", "_worker"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require(
        (args.command == "_worker") == (args.output is not None),
        "Only the guarded worker takes an output path",
    )
    if args.command == "_worker":
        worker(args.output)
    elif args.command == "run":
        run()
    else:
        actions = {"preflight": preflight, "freeze": freeze, "readout": readout}
        print(json.dumps(actions[args.command](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
