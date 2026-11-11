"""Build frozen role-local paired stereo speech views without model execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
import wave
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import audio_native_stereo_preflight_v274 as stereo
from scripts import audio_short_edit_cases_v172 as sources
from scripts import video_appearance_runtime_v108 as runtime
from scripts.audio_shared_noise_v216 import rms
from scripts.audio_voxforge_tr_transfer_v201 import verify_data_guard
from scripts.video_appearance_corrected_v112 import scoped

ROOT, PANEL = stereo.ROOT, stereo.PANEL
TRIAL = sources.roster.BASE / "audio-native-stereo-pairs-v277"
MEDIA = sources.roster.ARCHIVE / "audio-native-stereo-pairs-v277"
FREEZE = TRIAL / "freeze.json"
DEVELOPMENT_SEAL = TRIAL / "development-admitted.json"
PREVIOUS = PANEL / "audio-native-stereo-preflight-complete-v274.json"
PREVIOUS_SHA = "118c2f4cea21d4481ca5280acb4e1eeb0c7fa17e362b72b8bafde133af5ad968"
SOURCE_FREEZE_SHA = "e911911a33dbc9419ddc7fc12cccd21cc33e13f2c96d415796c9fb8800eb6c3c"
MODULE = "scripts.audio_native_stereo_pairs_v277"
RATE, SAMPLES = stereo.RATE, stereo.SAMPLES
SOURCE_COUNTS = {
    "development": {"genuine": 14, "clone": 5, "expressive": 12, "stock": 6},
    "holdout": {"genuine": 16, "clone": 6, "expressive": 12, "stock": 6},
}
EQUAL_GAIN_DB = -5
FORCED_DB = (5, -10)
PEAK_LIMIT = 32760
SNR_ERROR_DB = 0.1
CODE = (
    "backend/scripts/audio_native_stereo_pairs_v277.py",
    "backend/tests/test_audio_native_stereo_pairs_v277.py",
    "docs/audio-native-stereo-pairs-v277-method.md",
    "backend/scripts/audio_short_edit_cases_v172.py",
    "backend/scripts/audio_short_edit_materialize_v169c.py",
    "backend/scripts/audio_shared_noise_v216.py",
    "backend/scripts/audio_voxforge_tr_transfer_v201.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/video_appearance_runtime_v108.py",
)
require, sha, read, write = sources.require, sources.sha, sources.read, sources.write


def stage(role):
    return TRIAL / role / "bind-00"


def conditions(tapes):
    return ("matched_clean", *(f"{tape}_{mode}" for tape in tapes for mode in ("equal_gain", "forced")))


def previous():
    stereo.holdout.model_free()
    require(sha(PREVIOUS) == PREVIOUS_SHA and sha(sources.PUBLIC) == SOURCE_FREEZE_SHA,
            "The exact v274 and v172 public commitments are required")
    receipt = read(PREVIOUS)
    require(receipt["status"] == "complete" and receipt["plan"] == stereo.plan()
            and receipt["model_calls"] == receipt["foreground_files_opened"] == 0
            and all(sha(ROOT / name) == value for name, value in receipt["code_sha256"].items())
            and all(stereo.tool_hash(name) == value for name, value in receipt["codec_binaries_sha256"].items()),
            "v274 implementation, role plan or codec binary differs")
    return receipt


def role_plan(role):
    old, binding = sources.verify(role)
    rows = old["source_manifest"]["foregrounds"]
    require(Counter(row["lane"] for row in rows) == SOURCE_COUNTS[role], "Frozen foreground denominators differ")
    ordinal, planned = Counter(), []
    for row in rows:
        lane = row["lane"]
        start, quieter = stereo.window_assignment(ordinal[lane])
        ordinal[lane] += 1
        planned.append({"source": row, "tape_start_sample": start, "forced_quieter_channel": quieter})
    return planned, binding


def verify_disjoint(plans):
    keys = ("slot", "group_id", "candidate", "voice", "original_sha256", "prefix_pcm_sha256")
    for key in keys:
        role_sets = [{row["source"][key] for row in plan if key in row["source"]} for plan in plans.values()]
        require(role_sets[0].isdisjoint(role_sets[1]), "Development and holdout source identities overlap")


def identity(receipt):
    return {
        "schema": "chipmunk.audio-native-stereo-pair-freeze/277",
        "preflight_sha256": PREVIOUS_SHA,
        "source_freeze_sha256": SOURCE_FREEZE_SHA,
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "numpy_version": np.__version__,
        "codec_binaries_sha256": receipt["codec_binaries_sha256"],
        "v274_window_plan": receipt["plan"],
        "conditions": {role: list(conditions(tapes)) for role, tapes in receipt["plan"]["roles"].items()},
        "equal_gain_joint_snr_db": EQUAL_GAIN_DB,
        "forced_quieter_and_noisier_snr_db": list(FORCED_DB),
        "max_quantized_snr_error_db": SNR_ERROR_DB,
        "common_absolute_channel_sum_limit": PEAK_LIMIT,
        "model_calls": 0,
    }


def plan_binding(plan, binding):
    payload = json.dumps(plan, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    return {"private_plan_sha256": stereo.digest(payload),
            "v172_private_source_manifest_sha256": binding["private_source_manifest_sha256"],
            "v172_private_plan_sha256": binding["private_plan_sha256"],
            "planned_foregrounds": len(plan), "planned_cases": 5 * len(plan)}


def freeze():
    require(not TRIAL.exists() and not MEDIA.exists(), "Preserve all prior stereo stage attempts")
    receipt = previous()
    plans = {role: role_plan(role) for role in SOURCE_COUNTS}
    verify_disjoint({role: plan for role, (plan, _) in plans.items()})
    frozen = {**identity(receipt), "roles": {role: plan_binding(*plan) for role, plan in plans.items()}}
    TRIAL.mkdir(mode=0o700)
    for role in SOURCE_COUNTS:
        (TRIAL / role).mkdir(mode=0o700)
    write(FREEZE, frozen)
    return frozen


def check_development_seal():
    require(DEVELOPMENT_SEAL.is_file(), "Confirm the completed development preparation before opening holdout audio")
    result, profile, journal = guarded_result("development")
    require(read(DEVELOPMENT_SEAL) == dev_admission(result, profile, journal), "Development admission receipt differs")


def verify(role):
    receipt = previous()
    frozen = read(FREEZE)
    require(all(frozen[key] == value for key, value in identity(receipt).items()), "Exact v277 execution freeze required")
    if role == "holdout":
        check_development_seal()
    plan, binding = role_plan(role)
    require(frozen["roles"][role] == plan_binding(plan, binding), "Exact frozen foreground and window mapping required")
    return plan, frozen, receipt


def read_tapes(role, receipt):
    if role == "development":
        folder = stereo.development.SOURCES
        old = read(stereo.development.PUBLIC)
        require(stereo.development.asset_identity() == old["publisher_assets"], "Development archives differ")
    else:
        folder = stereo.holdout.TAPE_ROOT
        stereo.holdout.publisher_assets()
    tapes = {}
    for environment, filename in receipt["plan"]["roles"][role].items():
        expected = receipt["tapes"][role][environment]
        require(expected["archive"] == filename and expected["samples_per_channel"] == stereo.TAPE_SAMPLES,
                "Exact role-local archive binding required")
        channels = []
        with zipfile.ZipFile(folder / filename) as archive:
            for member in stereo.CHANNELS:
                pcm = stereo.tape_member(archive, member)
                windows = {start: pcm[2 * start:2 * (start + SAMPLES)] for start in stereo.WINDOWS}
                require(expected["members"][member] == {"pcm_sha256": stereo.digest(pcm),
                        "windows_sha256": stereo.digest(b"".join(windows.values()))}, "v274 tape PCM or windows differ")
                channels.append(windows)
        tapes[environment] = {
            start: np.column_stack([np.frombuffer(channel[start], dtype="<i2") for channel in channels])
            for start in stereo.WINDOWS
        }
    return tapes


def snr(signal, background):
    signal_rms, noise_rms = rms(signal), rms(background)
    require(signal_rms > 0 and noise_rms > 0, "A nonzero speech and background RMS is required")
    return float(20 * np.log10(signal_rms / noise_rms))


def mix(pcm, tapes, quieter):
    require(len(pcm) == 2 * SAMPLES and len(tapes) == 2 and quieter in (0, 1), "Exact paired three-second inputs required")
    foreground = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    speech = np.broadcast_to(foreground[:, None], (SAMPLES, 2))
    level = rms(foreground)
    require(level > 0, "Foreground RMS is zero")
    residuals = {"matched_clean": np.zeros_like(speech)}
    for name, recorded in tapes.items():
        noise = np.asarray(recorded, dtype=np.float64)
        require(noise.shape == speech.shape and np.isfinite(noise).all() and all(rms(v) > 0 for v in noise.T),
                "Two aligned nonzero background channels are required")
        residuals[f"{name}_equal_gain"] = noise * (level * 10 ** (-EQUAL_GAIN_DB / 20) / rms(noise))
        targets = tuple(FORCED_DB if quieter == 0 else reversed(FORCED_DB))
        scales = [level * 10 ** (-target / 20) / rms(channel) for target, channel in zip(targets, noise.T, strict=True)]
        residuals[f"{name}_forced"] = noise * scales

    peak = max(float(np.max(np.sum(np.abs(speech) + np.abs(noise), axis=1))) for noise in residuals.values())
    gain = min(1.0, PEAK_LIMIT / peak)
    clean = np.rint(speech * gain).astype(np.int32)
    cases = {}
    for name, noise in residuals.items():
        quantized = np.rint(noise * gain).astype(np.int32)
        merged = clean + quantized
        require(np.max(np.sum(np.abs(merged), axis=1)) <= PEAK_LIMIT + 2,
                "Conservative stereo mix headroom exceeded")
        levels = None
        if name != "matched_clean":
            levels = {"channels_db": [snr(clean[:, index], quantized[:, index]) for index in (0, 1)],
                      "joint_db": snr(clean, quantized)}
            if name.endswith("_equal_gain"):
                errors = [levels["joint_db"] - EQUAL_GAIN_DB]
            else:
                targets = FORCED_DB if quieter == 0 else FORCED_DB[::-1]
                errors = [measured - target for measured, target in zip(levels["channels_db"], targets, strict=True)]
            require(all(abs(error) <= SNR_ERROR_DB for error in errors), "Quantized SNR differs from the frozen treatment")
        body = merged.astype("<i2")
        require(np.array_equal(body.astype(np.int32) - clean, quantized), "Signed PCM residual differs")
        cases[name] = (body, {"common_gain": gain, "snr_db": levels, "residual_pcm_sha256":
                            stereo.digest(quantized.astype("<i2").tobytes())})
    return cases


def planned_cases(plan, role, tapes):
    for index, row in enumerate(plan):
        source = row["source"]
        base = {key: row[key] for key in ("tape_start_sample", "forced_quieter_channel")}
        base.update(source_slot=source["slot"], source_group=source["group_id"], lane=source["lane"])
        reason, mixtures = "foreground_unavailable", {}
        if source["status"] == "qualified":
            try:
                pcm = sources.exact_source(source, role)
            except (OSError, ValueError, wave.Error):
                reason = "foreground_integrity_failure"
            else:
                try:
                    windows = {name: views[row["tape_start_sample"]] for name, views in tapes.items()}
                    mixtures = mix(pcm, windows, row["forced_quieter_channel"])
                except ValueError:
                    reason = "stereo_construction_failure"
        for offset, condition in enumerate(conditions(tapes)):
            case = {**base, "case": f"case-{index:03d}-{offset:02d}", "condition": condition}
            if mixtures:
                payload, diagnostics = mixtures[condition]
                yield {**case, **diagnostics}, payload
            else:
                yield {**case, "status": "unavailable", "reason": reason}, None


def exact_wave(path, expected):
    require(not path.is_symlink(), "The saved stereo WAV must be a direct file")
    with wave.open(str(path), "rb") as wav:
        require((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes(), wav.getcomptype())
                == (2, 2, RATE, SAMPLES, "NONE") and wav.readframes(SAMPLES + 1) == expected.tobytes(),
                "Saved native stereo PCM differs")
    return {"wav_sha256": sha(path), "stereo_pcm_sha256": stereo.digest(expected.tobytes())}


def save_case(destination, case, payload):
    path = destination / f"{case}.wav"
    require(not path.exists(), "Preserve every stereo case output")
    mixed, measured = stereo.fixture_readout(destination, case, (payload[:, 0], payload[:, 1]))
    require(np.max(np.abs(np.frombuffer(mixed, dtype="<i2").astype(np.int32))) <= PEAK_LIMIT + 2,
            "Production downmix lacks the declared clipping headroom")
    return {"status": "prepared", **exact_wave(path, payload), **measured}


def summary(rows):
    counts = Counter(row["status"] for row in rows)
    by_condition = {condition: dict(Counter(row["status"] for row in rows if row["condition"] == condition))
                    for condition in dict.fromkeys(row["condition"] for row in rows)}
    return {"case_outcomes": dict(counts), "by_condition": by_condition}


def worker(role, output):
    plan, frozen, receipt = verify(role)
    folder = stage(role)
    require(output == folder / "output" and (folder / "resources.jsonl").is_file()
            and (folder / "worker.log").is_file() and not (folder / "profile.json").exists(),
            "The exact owned data-only stage is required")
    tapes = read_tapes(role, receipt)
    output.mkdir(mode=0o700)
    destination = MEDIA / role
    MEDIA.mkdir(mode=0o700, exist_ok=True)
    destination.mkdir(mode=0o700)
    journal, rows = output / "private-cases.jsonl", []
    with journal.open("x") as stream:
        for row, payload in planned_cases(plan, role, tapes):
            if payload is not None:
                try:
                    row.update(save_case(destination, row["case"], payload))
                except (OSError, ValueError, wave.Error, subprocess.SubprocessError) as error:
                    row.update(status="failed", reason="stereo_wave_or_production_parity_failure",
                               private_error=str(error))
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    require(len(rows) == frozen["roles"][role]["planned_cases"], "A planned case receipt is missing")
    stereo.holdout.model_free()
    write(output / "result.json", {"schema": "chipmunk.audio-native-stereo-pair-result/277", "role": role,
          "status": "complete", "freeze_sha256": sha(FREEZE), "private_cases_sha256": sha(journal),
          "planned_cases": len(rows), **summary(rows), "new_neural_calls": 0, "holdout_scored": False})


def run(role):
    output = stage(role) / "output"

    def launch(argv, **kwargs):
        require(argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(output), "0"],
                "Only the fixed v277 role data worker may run")
        return subprocess.Popen([*argv[:3], MODULE, "worker", role, str(output)], **kwargs)

    proxy = types.SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(runtime.run, {"TRIAL": TRIAL / role, "verify": lambda: verify(role), "subprocess": proxy,
                         "prerequisites": lambda *_: not (MEDIA / role).exists(), "__file__": __file__})("bind", 0)


def guarded_result(role):
    folder = stage(role)
    profile = folder / "profile.json"
    result_path, journal = folder / "output/result.json", folder / "output/private-cases.jsonl"
    verify_data_guard(read(profile), folder, result_path, sha(Path(__file__)))
    result = read(result_path)
    require(result["schema"] == "chipmunk.audio-native-stereo-pair-result/277" and result["role"] == role
            and result["status"] == "complete" and result["freeze_sha256"] == sha(FREEZE)
            and result["private_cases_sha256"] == sha(journal) and result["new_neural_calls"] == 0
            and result["holdout_scored"] is False, "Exact guarded construction receipt required")
    return result, profile, journal


def readout(role):
    plan, frozen, receipt = verify(role)
    result, profile, journal = guarded_result(role)
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    tapes = read_tapes(role, receipt)
    require(len(rows) == result["planned_cases"] == frozen["roles"][role]["planned_cases"], "Denominator differs")
    for (expected, payload), row in zip(planned_cases(plan, role, tapes), rows, strict=True):
        require(all(row[key] == value for key, value in expected.items()), "Rebuilt case recipe differs")
        if row["status"] == "prepared":
            path = MEDIA / role / f"{row['case']}.wav"
            require(payload is not None and all(row[key] == value for key, value in exact_wave(path, payload).items())
                    and row["channel_pcm_sha256"] == [stereo.digest(payload[:, index].tobytes()) for index in (0, 1)]
                    and row["source_timeline"] == [{"start_sample": 0, "end_sample": SAMPLES, "source_pts": 0}],
                    "Saved PCM, channel commitment or source clock differs")
        else:
            require(row["status"] in {"unavailable", "failed"} and row.get("reason"), "A missing case reason is required")
    require(all(result[key] == value for key, value in summary(rows).items()), "Case accounting differs")
    stereo.holdout.model_free()
    return {**result, "resource_profile_sha256": sha(profile), "raw_case_replay_count": result["case_outcomes"].get("prepared", 0)}


def dev_admission(result, profile, journal):
    require(result["case_outcomes"] == {"prepared": 185} and result["planned_cases"] == 185,
            "All 185 development cases must pass before holdout opens")
    return {"schema": "chipmunk.audio-native-stereo-development-admission/277", "freeze_sha256": sha(FREEZE),
            "private_cases_sha256": sha(journal), "resource_profile_sha256": sha(profile),
            "development_cases_verified": 185, "holdout_scored": False}


def seal_development():
    require(not DEVELOPMENT_SEAL.exists(), "Preserve the original development admission")
    readout("development")
    admission = dev_admission(*guarded_result("development"))
    write(DEVELOPMENT_SEAL, admission)
    return admission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "readout", "seal-development", "worker"))
    parser.add_argument("role", nargs="?", choices=SOURCE_COUNTS)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.action in {"run", "readout", "worker"}) == (args.role is not None), "Use a role for role stages only")
    require((args.action == "worker") == (args.output is not None), "Only the guarded worker takes an output path")
    if args.action == "worker":
        worker(args.role, args.output)
    elif args.action == "run":
        run(args.role)
    else:
        result = readout(args.role) if args.action == "readout" else (freeze if args.action == "freeze" else seal_development)()
        print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
