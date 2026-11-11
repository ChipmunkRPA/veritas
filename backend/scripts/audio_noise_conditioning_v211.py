"""Compare one clock-preserving noise-reduced view against recorded v208 speech results."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import subprocess
import wave
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import audio_demand_transfer_v208 as prior
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = prior.ROOT, prior.BASE, prior.PANEL
read, write, require, sha = prior.read, prior.write, prior.require, prior.sha
PILOT = prior.prior.pilot
TRIAL = BASE / "audio-noise-conditioning-v211"
PUBLIC = PANEL / "audio-noise-conditioning-freeze-v211.json"
PRIOR_RESULT = PANEL / "audio-demand-transfer-complete-v208.json"
MODULE = "scripts.audio_noise_conditioning_v211"
CONDITIONS = ("matched_clean", "washing_room_0db", "river_0db", "river_5db")
ORIGINS = ("genuine", "synthetic")
DIRECTIONS = PILOT.DIRECTIONS
RATE, MIN_SAMPLES, MAX_SAMPLES = 16000, 40000, 64000
FFT, HOP, QUANTILE, MIN_GAIN, POWER_EPSILON = 512, 128, 0.20, 0.5, 1e-12
SILENCE_RMS = 1e-6
WINDOW = np.hanning(FFT + 1)[:-1]
CASE_KEYS = ("case", "source", "condition", "origin")
METHOD = {
    "sample_rate": RATE,
    "sample_bounds": [MIN_SAMPLES, MAX_SAMPLES],
    "fft": FFT,
    "hop": HOP,
    "window": "periodic_hann",
    "boundary": "centered_reflection_with_right_hop_completion",
    "noise_quantile": QUANTILE,
    "noise_quantile_interpolation": "linear",
    "minimum_gain": MIN_GAIN,
    "power_epsilon": POWER_EPSILON,
    "silence_rms": SILENCE_RMS,
    "phase": "unmodified",
    "output_dtype": "float32",
    "post_normalization": False,
    "post_clipping": False,
    "gate_scope": "unchanged_recorded_v208",
    "new_calibration": False,
}
CODE = (
    "backend/scripts/audio_noise_conditioning_v211.py",
    "backend/tests/test_audio_noise_conditioning_v211.py",
    "docs/audio-noise-conditioning-v211-method.md",
    "backend/scripts/video_appearance_runtime_v108.py",
)


@dataclass(frozen=True)
class View:
    status: str
    reason: str | None = None
    pcm: np.ndarray | None = None
    mask_sha256: str | None = None


def pcm_sha(values):
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def spectrum(pcm):
    half = FFT // 2
    right_extra = (-len(pcm)) % HOP
    padded = np.pad(pcm.astype(np.float64), (half, half + right_extra), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, FFT)[::HOP]
    return np.fft.rfft(frames * WINDOW, axis=1), len(padded)


def reconstruct(bins, padded_size, samples):
    frames = np.fft.irfft(bins, n=FFT, axis=1)
    numerator, denominator = np.zeros(padded_size), np.zeros(padded_size)
    for i, frame in enumerate(frames):
        start = i * HOP
        numerator[start : start + FFT] += frame * WINDOW
        denominator[start : start + FFT] += WINDOW**2
    center = slice(FFT // 2, FFT // 2 + samples)
    require(
        len(denominator[center]) == samples and np.all(denominator[center] > POWER_EPSILON),
        "Overlap-add does not fully support every original sample",
    )
    return (numerator[center] / denominator[center]).astype(np.float32)


def spectral_gain(power):
    floor = np.quantile(power, QUANTILE, axis=0, method="linear")
    return np.sqrt(np.maximum(MIN_GAIN**2, 1.0 - floor / (power + POWER_EPSILON)))


def condition(pcm):
    require(
        isinstance(pcm, np.ndarray)
        and pcm.dtype == np.float32
        and pcm.ndim == 1
        and MIN_SAMPLES <= len(pcm) <= MAX_SAMPLES
        and np.isfinite(pcm).all()
        and float(np.max(np.abs(pcm))) <= 1.0,
        "Expected finite canonical short mono PCM",
    )
    if np.ptp(pcm) == 0 or np.sqrt(np.mean(np.square(pcm, dtype=np.float64))) <= SILENCE_RMS:
        return View("unavailable", "silent_or_constant")
    bins, padded = spectrum(pcm)
    gain = spectral_gain(np.square(np.abs(bins)))
    result = reconstruct(bins * gain, padded, len(pcm))
    require(result.shape == pcm.shape and np.isfinite(result).all(), "Conditioned PCM lost its sample clock")
    if np.max(np.abs(result)) > 1:
        return View("unavailable", "conditioned_clipping")
    if np.array_equal(result, pcm):
        return View("unavailable", "identical_view")
    return View("ready", pcm=result, mask_sha256=hashlib.sha256(gain.astype("<f8").tobytes()).hexdigest())


def waveform(case):
    body = prior.waveform(case)
    with wave.open(io.BytesIO(body), "rb") as stream:
        pcm = np.frombuffer(stream.readframes(case["samples"]), dtype="<i2").astype(np.float32)
    return pcm / np.float32(32768)


def source_context():
    PILOT.require_model_free()
    cases = prior.verify()
    aggregate = scoped(prior.readout, {"verify": lambda: cases})()
    require(
        read(PRIOR_RESULT) == aggregate
        and aggregate["schema"] == "chipmunk.audio-demand-transfer-result/208"
        and aggregate["status"] == "complete",
        "Exact completed guarded v208 public result required",
    )
    conditions = {}
    dependencies = {}
    for selected in CONDITIONS:
        batch = prior.CONDITIONS.index(selected)
        stage = prior.TRIAL / f"extract-{batch:02d}"
        journal = stage / "output/private-observations.jsonl"
        rows = prior.records(journal)
        dependencies[selected] = {
            "private_journal_sha256": sha(journal),
            "profile_sha256": sha(stage / "profile.json"),
        }
        conditions[selected] = list(zip(cases[batch :: len(prior.CONDITIONS)], rows, strict=True))
        require(
            len(conditions[selected]) == 48
            and Counter(row["origin"] for row in rows) == {origin: 24 for origin in ORIGINS},
            "Preserve all 48 planned source slots in each selected condition",
        )
    PILOT.require_model_free()
    return conditions, dependencies


def identity():
    conditions, dependencies = source_context()
    return {
        "schema": "chipmunk.audio-noise-conditioning-freeze/211",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "v208_freeze_sha256": sha(prior.PUBLIC),
        "v208_result_sha256": sha(PRIOR_RESULT),
        "v204_preparation_sha256": sha(prior.PREP),
        "selected_v208_receipts": dependencies,
        "assets": PILOT.ASSETS,
        "numpy_version": importlib.metadata.version("numpy"),
        "method": METHOD,
        "conditions": list(CONDITIONS),
        "planned_cases": sum(len(pairs) for pairs in conditions.values()),
        "new_fitting_permitted": False,
        "holdout_access_permitted": False,
    }, conditions


def preflight():
    lock, conditions = identity()
    return {
        "status": "ready_for_freeze",
        "planned_cases": lock["planned_cases"],
        "original_gate_admitted": {
            condition: sum(row["route_status"] == "ok" for _, row in pairs)
            for condition, pairs in conditions.items()
        },
        "model_calls": 0,
    }


def verify():
    lock, conditions = identity()
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == lock,
        "Exact prospectively frozen v211 procedure required",
    )
    return conditions


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every previous study attempt")
    lock, _ = identity()
    require(
        all(sha(PILOT.WEIGHTS / name) == value for name, value in PILOT.ASSETS.items()),
        "Exact local voice weights required",
    )
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return {"status": "frozen", "planned_cases": lock["planned_cases"], "model_calls": 0}


def rescue(recorded, candidate):
    return "ai" if recorded == "inconclusive" and candidate == "ai" else recorded


def parity_cases(pairs):
    return {
        next(case["case"] for case, row in pairs if row["origin"] == origin and row["route_status"] == "ok")
        for origin in ORIGINS
    }


def unavailable_view(baseline):
    return View("unavailable", baseline["route_status"])


def case_view(case, baseline):
    if baseline["route_status"] != "ok":
        return None, unavailable_view(baseline)
    original = waveform(case)
    require(
        baseline["windows"] == [[0, len(original)]], "Only the original admitted complete window is scored"
    )
    return original, condition(original)


def calibrated_direction(raw, calibration, bundle):
    return calibration.initial_decision(calibration.probability(raw, bundle), bundle)


def start(stack, count):
    from app.detectors.audio import calibration
    from scripts.run_audio_specialists_v15 import load_voice

    bundle = calibration.load()
    model = load_voice({"weights": str(PILOT.WEIGHTS), "assets": PILOT.ASSETS})
    stack.callback(model.register_forward_pre_hook(count).remove)

    def score(pcm):
        (raw,) = model.predict([pcm])
        require(calibration.valid(raw), "Expected one finite voice score")
        return float(raw)

    return score, lambda raw: calibrated_direction(raw, calibration, bundle)


def build_case(case, baseline, parity, score, initial):
    original, view = case_view(case, baseline)
    baseline_raw = baseline.get("raw")
    baseline_initial = initial(baseline_raw) if original is not None else "unavailable"
    parity_raw = score(original) if parity else None
    if parity:
        require(
            math.isclose(parity_raw, baseline_raw, abs_tol=1e-6, rel_tol=0),
            "Original-view numerical replay failed",
        )
    conditioned_raw = score(view.pcm) if view.status == "ready" else None
    conditioned_direction = initial(conditioned_raw) if view.status == "ready" else "unavailable"
    return {
        **{key: case[key] for key in CASE_KEYS},
        "recorded_route_status": baseline["route_status"],
        "recorded_direction": baseline["decision"],
        "recorded_raw": baseline_raw,
        "recorded_initial_direction": baseline_initial,
        "parity_raw": parity_raw,
        "window": baseline["windows"],
        "conditioned_status": view.status,
        "conditioned_reason": view.reason,
        "conditioned_pcm_sha256": pcm_sha(view.pcm) if view.pcm is not None else None,
        "conditioned_gain_sha256": view.mask_sha256,
        "conditioned_raw": conditioned_raw,
        "conditioned_diagnostic_direction": conditioned_direction,
        "one_way_ai_rescue": rescue(baseline["decision"], conditioned_direction),
    }


def audit_case(row, case, baseline, parity, initial):
    original, view = case_view(case, baseline)
    recorded = row["recorded_raw"]
    raw = row["conditioned_raw"]
    parity_raw = row["parity_raw"]
    require(
        all(row[key] == case[key] == baseline[key] for key in CASE_KEYS)
        and row["recorded_route_status"] == baseline["route_status"]
        and row["recorded_direction"] == baseline["decision"]
        and recorded == baseline.get("raw")
        and row["window"] == baseline["windows"]
        and row["recorded_initial_direction"]
        == (initial(recorded) if original is not None else "unavailable")
        and (
            (
                parity
                and type(parity_raw) is float
                and math.isclose(parity_raw, recorded, abs_tol=1e-6, rel_tol=0)
            )
            or (not parity and parity_raw is None)
        )
        and row["conditioned_status"] == view.status
        and row["conditioned_reason"] == view.reason
        and row["conditioned_pcm_sha256"] == (pcm_sha(view.pcm) if view.pcm is not None else None)
        and row["conditioned_gain_sha256"] == view.mask_sha256
        and (
            (view.status == "ready" and type(raw) is float and math.isfinite(raw) and 0 <= raw <= 1)
            or (view.status == "unavailable" and raw is None)
        )
        and row["conditioned_diagnostic_direction"] == (initial(raw) if raw is not None else "unavailable")
        and row["one_way_ai_rescue"] == rescue(baseline["decision"], row["conditioned_diagnostic_direction"])
        and row["pella_forwards"] == int(parity) + int(view.status == "ready"),
        "Exact original route, transformed view and paired diagnostic required",
    )


def worker(batch, output):
    pairs = verify()[CONDITIONS[batch]]
    stage = TRIAL / f"extract-{batch:02d}"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline owned worker required",
    )
    PILOT.require_model_free()
    output.mkdir(mode=0o700)
    journal = output / "private-observations.jsonl"
    forwards = 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    smoke = parity_cases(pairs)
    with ExitStack() as stack, journal.open("x") as stream:
        score, initial = start(stack, count)
        for case, baseline in pairs:
            before = forwards
            row = build_case(case, baseline, case["case"] in smoke, score, initial)
            row["pella_forwards"] = forwards - before
            require(
                row["pella_forwards"]
                == int(case["case"] in smoke) + int(row["conditioned_status"] == "ready"),
                "Expected one conditioned model forward for each admitted nontrivial transform",
            )
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
        },
    )


def run(batch):
    def launch(argv, **kwargs):
        output = TRIAL / f"extract-{batch:02d}/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "extract", str(output), str(batch)],
            "Exact v211 guarded command required",
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


def directions(rows, key):
    values = Counter(row[key] for row in rows)
    return {name: values[name] for name in DIRECTIONS}


def rank(rows, key):
    positives = [row[key] for row in rows if row["origin"] == "synthetic"]
    negatives = [row[key] for row in rows if row["origin"] == "genuine"]
    wins = ties = losses = 0
    for ai in positives:
        for genuine in negatives:
            wins += ai > genuine
            ties += ai == genuine
            losses += ai < genuine
    total = wins + ties + losses
    return {
        "synthetic_sources": len(positives),
        "genuine_sources": len(negatives),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "auc": (wins + 0.5 * ties) / total if total else None,
    }


def comparison(rows):
    common = [row for row in rows if row["conditioned_status"] == "ready"]
    results = {
        "raw_rank_on_common_assessed": {
            "recorded": rank(common, "recorded_raw"),
            "conditioned": rank(common, "conditioned_raw"),
        },
        "by_origin": {},
    }
    for origin in ORIGINS:
        selected = [row for row in rows if row["origin"] == origin]
        paired = [row for row in common if row["origin"] == origin]
        baseline = directions(selected, "recorded_direction")
        simulated = directions(selected, "one_way_ai_rescue")
        promoted = simulated["ai"] - baseline["ai"]
        movements = Counter()
        for row in paired:
            before, after = row["recorded_raw"], row["conditioned_raw"]
            movements["up" if after > before else "down" if after < before else "tie"] += 1
        results["by_origin"][origin] = {
            "planned": len(selected),
            "original_route_outcomes": dict(Counter(row["recorded_route_status"] for row in selected)),
            "conditioned_assessed": len(paired),
            "conditioned_unavailable_reasons": dict(
                Counter(row["conditioned_reason"] for row in selected if row["conditioned_reason"])
            ),
            "recorded_directions": baseline,
            "conditioned_diagnostic_directions": directions(selected, "conditioned_diagnostic_direction"),
            "diagnostic_initial_direction_transitions": dict(
                Counter(
                    f"{row['recorded_initial_direction']}->{row['conditioned_diagnostic_direction']}"
                    for row in paired
                )
            ),
            "raw_response_movement": {name: movements[name] for name in ("up", "down", "tie")},
            "post_hoc_one_way_ai_rescue_directions": simulated,
            "post_hoc_new_false_ai" if origin == "genuine" else "post_hoc_ai_recoveries": promoted,
            "recorded_to_rescue_transitions": dict(
                Counter(f"{row['recorded_direction']}->{row['one_way_ai_rescue']}" for row in selected)
            ),
        }
    return results


def readout():
    pairs_by_condition = verify()
    from app.detectors.audio import calibration

    bundle = calibration.load()
    results, profiles, journals = {}, [], []
    calls = 0
    for batch, name in enumerate(CONDITIONS):
        pairs = pairs_by_condition[name]
        smoke = parity_cases(pairs)
        stage = TRIAL / f"extract-{batch:02d}"
        journal = stage / "output/private-observations.jsonl"
        rows = prior.records(journal)
        total = sum(row["pella_forwards"] for row in rows)
        receipt = PILOT.speech.qualification_profile(stage, total, {}, Path(__file__).name, "extract", batch)
        require(
            receipt["batch"] == batch
            and receipt["condition"] == name
            and receipt["private_observations_sha256"] == sha(journal)
            and receipt["freeze_sha256"] == sha(PUBLIC),
            "Exact completed v211 condition receipt required",
        )
        for row, (case, baseline) in zip(rows, pairs, strict=True):
            audit_case(
                row,
                case,
                baseline,
                case["case"] in smoke,
                lambda raw: calibrated_direction(raw, calibration, bundle),
            )
        profiles.append(sha(stage / "profile.json"))
        journals.append(sha(journal))
        calls += total
        results[name] = comparison(rows)
    return {
        "schema": "chipmunk.audio-noise-conditioning-result/211",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "profile_sha256": profiles,
        "private_journal_sha256": journals,
        "new_neural_calls": calls,
        "post_hoc_simulation_only": True,
        "holdout_scored": False,
        "defaults_changed": False,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("preflight", "freeze", "readout"):
        commands.add_parser(action)
    for action in ("run", "_worker"):
        cmd = commands.add_parser(action)
        cmd.add_argument("batch", type=int, choices=range(len(CONDITIONS)))
        if action == "_worker":
            cmd.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "_worker":
        worker(args.batch, args.output)
    elif args.action == "run":
        run(args.batch)
    else:
        actions = {"preflight": preflight, "freeze": freeze, "readout": readout}
        print(json.dumps(actions[args.action](), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
