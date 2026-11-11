"""Construct role-separated paired ordinary and AI speech edits with exact PCM."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import types
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import audio_short_edit_controls as controls
from scripts import audio_short_edit_materialize_v169c as sources
from scripts.video_appearance_corrected_v112 import scoped

roster = sources.roster
TRIAL = roster.BASE / "audio-short-edit-cases-v172"
MEDIA = roster.ARCHIVE / "audio-short-edit-cases-v172"
PUBLIC = roster.ROOT / "docs/evaluation-panels/audio-short-edit-case-freeze-v172.json"
METHOD = roster.ROOT / "docs/audio-short-edit-cases-v172.md"
MODULE = "scripts.audio_short_edit_cases_v172"
SAMPLES, RATE = controls.SAMPLES, controls.SAMPLE_RATE
LEVELS = (("clean", None), ("snr_0db", 0), ("snr_plus5db", 5))
FADES = ((0, "hard"), (20, "fade20"), (50, "fade50"))
require, sha, read, write = sources.require, sources.sha, sources.read, sources.write


def stable(key):
    return hashlib.sha256(f"chipmunk.audio-short-edit-cases/172:{key}".encode()).hexdigest()


def stage_folder(role):
    return TRIAL / role / "bind-00"


def verified_source_receipts():
    sources.verify()
    profile_path = sources.TRIAL / "bind-00/profile.json"
    profile = read(profile_path)
    result_path = sources.OUTPUT / "result.json"
    result = read(result_path)
    require(
        profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(sources.__file__))
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(profile_path.parent / "resources.jsonl")
        and result["freeze_sha256"] == sha(sources.PUBLIC)
        and result["private_attempt_journal_sha256"] == sha(sources.OUTPUT / "attempts.jsonl")
        and result["status"] == "complete"
        and result["source_samples"] == SAMPLES
        and result["sample_rate"] == RATE
        and result["holdout_scored"] is False
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0,
        "Exact completed unscored v169c source preparation required",
    )
    return result, {
        "v169c_public_freeze_sha256": sha(sources.PUBLIC),
        "v169c_private_result_sha256": sha(result_path),
        "v169c_resource_profile_sha256": sha(profile_path),
    }


def verified_manifest(role, receipt):
    path = sources.MEDIA / role / "private-manifest.json"
    manifest = read(path)
    require(
        sha(path) == receipt["roles"][role]["private_manifest_sha256"]
        and manifest["role"] == role
        and manifest["scored"] is False,
        "Exact unscored source role manifest required",
    )
    return manifest


def verify_role_roster(manifest, source_plan):
    role = manifest["role"]
    expected, declared = {}, {}
    for group in source_plan["groups"]:
        if group["role"] == role:
            declared[group["group_id"]] = {row["id"]: row["voice"] for row in group["ordered_candidates"]}
            for index in range(group["required"]):
                expected[f"{group['group_id']}-{index:02d}"] = (group["group_id"], group["lane"])
    found = {row["slot"]: row for row in manifest["foregrounds"]}
    require(
        len(found) == len(manifest["foregrounds"]) and found.keys() == expected.keys(), "Source slots differ"
    )
    for slot, row in found.items():
        require(
            (row["group_id"], row["lane"]) == expected[slot]
            and row["status"] in {"qualified", "unavailable"},
            "Role source group, lane or availability differs",
        )
        if row["status"] == "qualified":
            require(declared[row["group_id"]][row["candidate"]] == row["voice"], "Source voice differs")
    noise = {row["slot"]: row for row in manifest["noise"]}
    expected_noise = {
        f"noise-{index:02d}": row["id"]
        for index, row in enumerate(source_plan["noise_parents"])
        if row["role"] == role
    }
    require(
        len(noise) == len(manifest["noise"]) and noise.keys() == expected_noise.keys(), "Noise roster differs"
    )
    require(
        all(
            row["status"] == "unavailable"
            or (row["status"] == "qualified" and row["source"] == expected_noise[slot])
            for slot, row in noise.items()
        ),
        "Noise source identity or status differs",
    )


def arms(family):
    if family == "genuine":
        return [("untouched", None), *((f"same_source_{name}", ms) for ms, name in FADES)]
    return [
        ("fully_synthetic", None),
        *((f"genuine_to_genuine_{name}", ms) for ms, name in FADES),
        *((f"genuine_to_ai_{name}", ms) for ms, name in FADES),
    ]


def comparisons(manifest):
    foreground = sorted(manifest["foregrounds"], key=lambda row: stable(row["slot"]))
    genuine = [row for row in foreground if row["lane"] == "genuine" and row["status"] == "qualified"]
    noise = sorted(manifest["noise"], key=lambda row: stable(row["slot"]))
    planned = []
    ai_index = 0
    for index, row in enumerate(foreground):
        family = "genuine" if row["lane"] == "genuine" else "ai"
        host, donor = row if family == "genuine" else None, None
        if family == "ai" and genuine:
            host = genuine[ai_index % len(genuine)]
            pool = [
                candidate
                for candidate in genuine
                if candidate["voice"] != host["voice"] and candidate["group_id"] != host["group_id"]
            ]
            if pool:
                donor = pool[(ai_index // len(genuine) + ai_index) % len(pool)]
        ai_index += family == "ai"
        planned.append(
            {
                "comparison_id": f"comparison-{index:03d}",
                "role": manifest["role"],
                "family": family,
                "lane": row["lane"],
                "source_slot": row["slot"],
                "source_group": row["group_id"],
                "genuine_host_slot": host["slot"] if host else None,
                "genuine_host_group": host["group_id"] if host else None,
                "genuine_donor_slot": donor["slot"] if donor else None,
                "genuine_donor_group": donor["group_id"] if donor else None,
                "noise_slot": noise[index % len(noise)]["slot"] if noise else None,
                "planned_cases": len(arms(family)) * len(LEVELS),
            }
        )
    return planned


def identity():
    _, commitments = verified_source_receipts()
    paths = (
        Path(__file__),
        METHOD,
        Path(controls.__file__),
        Path(sources.__file__),
        roster.ROOT / "backend/tests/test_audio_short_edit_cases_v172.py",
        roster.ROOT / "backend/scripts/video_appearance_runtime_v108.py",
    )
    return {
        **commitments,
        "code_sha256": {str(path.relative_to(roster.ROOT)): sha(path) for path in paths},
        "sample_rate": RATE,
        "samples_per_case": SAMPLES,
        "target_samples": list(controls.TARGET_SAMPLES),
        "donor_samples": list(controls.DONOR_SAMPLES),
        "joins_ms": [ms for ms, _ in FADES],
        "noise_snr_db": list(controls.SNR_LEVELS_DB),
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists() and not MEDIA.exists(), "Preserve every case attempt")
    receipt, _ = verified_source_receipts()
    original_plan, _ = sources.source_plan()
    manifests = {role: verified_manifest(role, receipt) for role in roster.ROLES}
    for manifest in manifests.values():
        verify_role_roster(manifest, original_plan)
    frozen = {
        "schema": "chipmunk.audio-short-edit-case-freeze/172",
        **identity(),
        "roles": {},
        "holdout_scored": False,
        "model_calls": 0,
    }
    TRIAL.mkdir(mode=0o700)
    for role, manifest in manifests.items():
        groups = comparisons(manifest)
        folder = TRIAL / role
        folder.mkdir(mode=0o700)
        path = folder / "private-plan.json"
        write(path, {"role": role, "comparisons": groups, "source_manifest": manifest})
        frozen["roles"][role] = {
            "private_plan_sha256": sha(path),
            "private_source_manifest_sha256": receipt["roles"][role]["private_manifest_sha256"],
            "comparisons": dict(Counter(group["family"] for group in groups)),
            "planned_cases": sum(group["planned_cases"] for group in groups),
        }
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)


def verify(role):
    frozen = read(PUBLIC)
    require(
        frozen == read(TRIAL / "freeze.json")
        and all(frozen[key] == value for key, value in identity().items())
        and frozen["holdout_scored"] is False
        and frozen["model_calls"] == 0,
        "Exact case and source freeze required",
    )
    path = TRIAL / role / "private-plan.json"
    require(sha(path) == frozen["roles"][role]["private_plan_sha256"], "Exact private role plan required")
    plan = read(path)
    receipt, _ = verified_source_receipts()
    manifest = verified_manifest(role, receipt)
    require(
        plan == {"role": role, "comparisons": comparisons(manifest), "source_manifest": manifest},
        "Role plan and verified source receipt no longer agree",
    )
    return plan, frozen["roles"][role]


def read_pcm(path):
    with wave.open(str(path), "rb") as stream:
        require(
            (
                stream.getnchannels(),
                stream.getsampwidth(),
                stream.getframerate(),
                stream.getnframes(),
                stream.getcomptype(),
            )
            == (1, 2, RATE, SAMPLES, "NONE"),
            "Exact three-second mono signed16 WAV required",
        )
        payload = stream.readframes(SAMPLES + 1)
    require(len(payload) == 2 * SAMPLES, "WAV PCM truncated")
    return payload


def float_pcm_sha(payload):
    return sources.pcm_digest(np.frombuffer(payload, dtype="<i2").astype("<f4") / np.float32(32768))


def exact_source(row, role):
    path = Path(row["file"])
    require(
        row["status"] == "qualified"
        and path.parent == sources.MEDIA / role
        and not path.is_symlink()
        and sha(path) == row["wav_sha256"],
        "Exact owned role source WAV required",
    )
    pcm = read_pcm(path)
    require(
        float_pcm_sha(pcm) == row["prefix_pcm_sha256"]
        and row["wav_range"]
        == {
            "start_sample": 0,
            "end_sample": SAMPLES,
            "source_start_pts": 0,
            "source_end_pts": SAMPLES,
            "time_base": "1/16000",
        },
        "Source PCM or zero-origin playback clock differs",
    )
    return pcm


def ai_truth(name, fade_ms):
    if name == "fully_synthetic":
        return {
            "speech_origin": "fully_synthetic_source",
            "ai_support_samples": [0, SAMPLES],
            "pure_ai_samples": [0, SAMPLES],
            "blended_ai_samples": [],
        }
    if not name.startswith("genuine_to_ai_"):
        return {
            "speech_origin": "genuine_source_only",
            "ai_support_samples": None,
            "pure_ai_samples": None,
            "blended_ai_samples": [],
        }
    start, end = controls.TARGET_SAMPLES
    fade = controls.FADE_SAMPLES[fade_ms]
    pure = [start + fade - 1, end - fade + 1] if fade else [start, end]
    return {
        "speech_origin": "partly_synthetic_source",
        "ai_support_samples": [start + 1, end - 1] if fade else [start, end],
        "pure_ai_samples": pure,
        "blended_ai_samples": [[start + 1, pure[0]], [pure[1], end - 1]] if fade else [],
    }


def changes(reference, observed):
    a = np.frombuffer(reference, dtype="<i2")
    b = np.frombuffer(observed, dtype="<i2")
    changed = a != b
    endpoints = np.flatnonzero(np.diff(np.r_[False, changed, False]))
    return int(np.count_nonzero(changed)), endpoints.reshape(-1, 2).tolist()


def construct(group, pcm):
    """Return case receipts and bytes for one comparison, without filesystem work."""
    source = pcm.get(group["source_slot"])
    host = pcm.get(group["genuine_host_slot"])
    genuine_donor = pcm.get(group["genuine_donor_slot"])
    noise = pcm.get(group["noise_slot"])
    reference = host if host is not None else source
    edited, failures = {}, {}
    for name, fade in arms(group["family"]):
        if source is None:
            failures[name] = "source_unavailable"
        elif fade is None:
            edited[name] = source
        elif host is None or (group["family"] == "ai" and genuine_donor is None):
            failures[name] = "independent_genuine_host_or_donor_unavailable"
        else:
            donor = (
                None
                if group["family"] == "genuine"
                else source
                if name.startswith("genuine_to_ai_")
                else genuine_donor
            )
            try:
                edited[name] = controls.splice_pcm(host, donor, fade_ms=fade).pcm_s16le
            except controls.UnsupportedControl as error:
                failures[name] = str(error)

    treatment, noise_reason = None, "noise_parent_unavailable" if noise is None else None
    if noise is not None and edited:
        try:
            treatment = controls.mix_paired_noise(
                {"reference": reference, **edited}, reference="reference", noise_pcm_s16le=noise
            )
        except controls.UnsupportedControl as error:
            noise_reason = str(error)
    gain = treatment.common_gain if treatment else 1.0
    output = []
    for name, fade in arms(group["family"]):
        for condition, level in LEVELS:
            row = {
                "arm": name,
                "condition": condition,
                "nominal_snr_db": level,
                "join_ms": fade,
                "label": ai_truth(name, fade),
                "planned_target_samples": list(controls.TARGET_SAMPLES) if fade is not None else None,
                "planned_donor_samples": list(controls.DONOR_SAMPLES) if fade is not None else None,
            }
            reason = failures.get(name)
            if reason is None and level is not None and treatment is None:
                reason = noise_reason or "paired_noise_unavailable"
            if reason:
                output.append(({**row, "status": "unavailable", "reason": reason}, None))
                continue
            payload, base, noise_sha, measured = edited[name], reference, None, None
            if treatment:
                arm, anchor = treatment.variants[name], treatment.variants["reference"]
                payload, base = arm.clean_s16le, anchor.clean_s16le
                if level is not None:
                    payload, base = arm.mixed_s16le[level], anchor.mixed_s16le[level]
                    residual = np.frombuffer(payload, dtype="<i2").astype(np.int32) - np.frombuffer(
                        arm.clean_s16le, dtype="<i2"
                    )
                    require(
                        np.array_equal(residual, np.frombuffer(treatment.noise_s16le[level], dtype="<i2")),
                        "Paired additive noise changed between edit arms",
                    )
                    noise_sha = sources.digest(treatment.noise_s16le[level])
                    measured = arm.measured_snr_db[level]
            row |= {"gain": gain, "shared_noise_pcm_sha256": noise_sha, "measured_snr_db": measured}
            if fade is not None:
                count, intervals = changes(base, payload)
                start, end = controls.TARGET_SAMPLES
                require(all(start <= a < b <= end for a, b in intervals), "Observed edit escaped its target")
                row |= {
                    "observed_changed_samples": count,
                    "observed_changed_intervals": intervals,
                    "matched_host_pcm_sha256": sources.digest(base),
                }
                if count == 0:
                    output.append(
                        ({**row, "status": "unavailable", "reason": "zero_observed_change_after_gain"}, None)
                    )
                    continue
            output.append(({**row, "status": "constructed"}, payload))
    return output


def write_case(path, pcm):
    require(len(pcm) == 2 * SAMPLES and not path.exists(), "One exclusive exact-length case required")
    with path.open("xb") as stream, wave.open(stream, "wb") as wav:
        wav.setparams((1, 2, RATE, SAMPLES, "NONE", "not compressed"))
        wav.writeframes(pcm)
    require(read_pcm(path) == pcm, "Case PCM did not round-trip byte for byte")
    return {
        "file": str(path),
        "wav_sha256": sha(path),
        "pcm_s16le_sha256": sources.digest(pcm),
        "production_f32_pcm_sha256": float_pcm_sha(pcm),
        "samples": SAMPLES,
        "sample_rate": RATE,
    }


def worker(role, output):
    plan, frozen = verify(role)
    stage = stage_folder(role)
    destination = MEDIA / role
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not destination.exists(),
        "Only the exclusive guarded role constructor may run",
    )
    output.mkdir()
    MEDIA.mkdir(mode=0o700, exist_ok=True)
    destination.mkdir(mode=0o700)
    manifest = plan["source_manifest"]
    pcm, unavailable_inputs = {}, {}
    for row in [*manifest["foregrounds"], *manifest["noise"]]:
        if row["status"] != "qualified":
            unavailable_inputs[row["slot"]] = "source_preparation_unavailable"
            continue
        try:
            pcm[row["slot"]] = exact_source(row, role)
        except (OSError, ValueError, EOFError, wave.Error) as error:
            unavailable_inputs[row["slot"]] = str(error)
    input_failures = output / "private-unavailable-inputs.json"
    write(input_failures, unavailable_inputs)
    counts, reasons = Counter(), Counter()
    journal_path = output / "private-cases.jsonl"
    with journal_path.open("x") as journal:
        for group in plan["comparisons"]:
            cases = construct(group, pcm)
            require(len(cases) == group["planned_cases"], "Every planned case must remain in its comparison")
            for index, (receipt, payload) in enumerate(cases):
                case = f"{group['comparison_id']}-{index:02d}"
                if payload is not None:
                    try:
                        receipt.update(write_case(destination / f"{case}.wav", payload))
                    except (OSError, ValueError, wave.Error) as error:
                        receipt |= {
                            "status": "failed",
                            "reason": "case_round_trip_or_write_failed",
                            "private_error": str(error),
                        }
                counts[receipt["status"]] += 1
                if reason := receipt.get("reason"):
                    reasons[reason] += 1
                journal.write(json.dumps({"case": case, **group, **receipt}, allow_nan=False) + "\n")
                journal.flush()
    require(sum(counts.values()) == frozen["planned_cases"], "Every planned case must have a receipt")
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-short-edit-case-result/172",
            "status": "complete",
            "role": role,
            "freeze_sha256": sha(PUBLIC),
            "private_plan_sha256": frozen["private_plan_sha256"],
            "private_case_journal_sha256": sha(journal_path),
            "case_outcomes": counts,
            "private_unavailable_inputs_sha256": sha(input_failures),
            "unavailable_input_count": len(unavailable_inputs),
            "unavailable_or_failed_reasons": reasons,
            "new_neural_calls": 0,
            "holdout_scored": False,
        },
    )


def run(role):
    from scripts import video_appearance_runtime_v108 as runtime

    output = stage_folder(role) / "output"

    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(output), "0"],
            "Only the exact role data worker may run",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker", role, str(output)], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL / role,
            "verify": lambda: verify(role),
            "prerequisites": lambda *_: not (MEDIA / role).exists(),
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("bind", 0)


def readout(role):
    _, frozen = verify(role)
    stage = stage_folder(role)
    profile = read(stage / "profile.json")
    result_path = stage / "output/result.json"
    result = read(result_path)
    journal = stage / "output/private-cases.jsonl"
    require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(stage / "resources.jsonl")
        and result["status"] == "complete"
        and result["role"] == role
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_case_journal_sha256"] == sha(journal)
        and result["private_unavailable_inputs_sha256"]
        == sha(stage / "output/private-unavailable-inputs.json")
        and result["private_plan_sha256"] == frozen["private_plan_sha256"]
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and result["holdout_scored"] is False,
        "Exact completed guarded role result required",
    )
    observed = Counter()
    with journal.open() as stream:
        for line in stream:
            case = json.loads(line)
            observed[case["status"]] += 1
            if case["status"] == "constructed":
                wav = Path(case["file"])
                require(
                    wav.parent == MEDIA / role and sha(wav) == case["wav_sha256"], "Prepared case WAV differs"
                )
    require(
        dict(observed) == result["case_outcomes"] and sum(observed.values()) == frozen["planned_cases"],
        "Role output coverage differs",
    )
    print(
        json.dumps(
            {
                **result,
                "resource_profile_sha256": sha(stage / "profile.json"),
                "worker_duration_s": profile["duration_s"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("role", nargs="?", choices=roster.ROLES)
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.command != "freeze") == (args.role is not None), "An exact role is required outside freeze")
    require(
        (args.command == "worker") == (args.output is not None), "Only the guarded worker takes an output"
    )
    if args.command == "freeze":
        freeze()
    elif args.command == "worker":
        worker(args.role, args.output)
    else:
        (run if args.command == "run" else readout)(args.role)
