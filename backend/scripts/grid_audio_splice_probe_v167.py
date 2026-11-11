"""Measure sensitivity to ordinary speech-copy boundaries on the consumed GRID pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import types
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import median

import numpy as np

from scripts import grid_serving_baseline_v156 as serving
from scripts import grid_source_speech_v147 as speech
from scripts import video_appearance_runtime_v108 as runtime
from scripts.audio_exact_repetition import MIN_REPEAT_SAMPLES, find_repeated_spans
from scripts.prepare_audio_specialists_v15 import ASSETS
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = serving.ROOT, serving.BASE, serving.PANEL
TRIAL = BASE / "grid-audio-splice-probe-v167"
PUBLIC = PANEL / "grid-audio-splice-probe-freeze-v167.json"
SERVING_RESULT = PANEL / "grid-serving-completion-v156.json"
WEIGHTS = ROOT / "models/weights/audio-upgrade"
SCRIPT = "scripts.grid_audio_splice_probe_v167"
ROUTE = "aud.pella_v2"
SR, RAW_ATOL = 16000, 1e-6
ARMS = {"original": None, "hard_copy": 0, "fade_20_ms": 320, "fade_50_ms": 800}
CODE = (
    "backend/scripts/grid_audio_splice_probe_v167.py",
    "backend/tests/test_grid_audio_splice_probe_v167.py",
    "backend/scripts/audio_exact_repetition.py",
    "backend/tests/test_audio_exact_repetition.py",
    "docs/grid-audio-splice-probe-v167-method.md",
    "backend/scripts/grid_source_speech_v147.py",
    "backend/scripts/grid_serving_baseline_v156.py",
    "backend/scripts/grid_transfer_plan_v143.py",
    "backend/scripts/audio_calibration_v3_methods.py",
    "backend/scripts/prepare_audio_specialists_v15.py",
    "backend/scripts/run_audio_specialists_v15.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/app/detectors/audio/calibration.py",
    "backend/app/detectors/audio/calibrations/local-speech-v3.json",
    "backend/app/detectors/audio/content.py",
    "backend/app/detectors/audio/upgraded.py",
    "backend/app/detectors/video/adapters/audio.py",
    "backend/app/detectors/video/audio.py",
    "backend/app/engine/profiles.py",
    "backend/app/models.yaml",
)
read, write, sha, require = serving.read, serving.write, serving.sha, serving.require


def pcm_digest(pcm):
    return hashlib.sha256(speech.pcm_lattice(pcm)).hexdigest()


def render(pcm, support, fade):
    if fade is None:
        return pcm
    copied, _ = speech.delayed_pcm(pcm, support)
    if fade == 0:
        return copied
    start, end = support["output_samples_half_open"]
    require(type(fade) is int and fade >= 2 and 2 * fade <= end - start, "Invalid contained crossfade extent")
    ramp = (1 - np.cos(np.linspace(0, np.pi, fade, dtype=np.float64))) / 2
    for interval, alpha in ((slice(start, start + fade), ramp), (slice(end - fade, end), ramp[::-1])):
        original = pcm[interval].astype(np.float64)
        donor = copied[interval].astype(np.float64)
        samples = np.rint((original + alpha * (donor - original)) * 32768).astype("<i2")
        copied[interval] = samples.astype("<f4") / np.float32(32768)
    speech.pcm_lattice(copied)
    return copied


def seam_jumps(pcm, support):
    start, end = support["output_samples_half_open"]
    require(0 < start < end < len(pcm), "Both source boundaries require an adjacent sample")
    return [int(abs(float(pcm[i]) - float(pcm[i - 1])) * 32768) for i in (start, end)]


def repeated_samples(pcm, support, fade):
    evidence = find_repeated_spans(speech.pcm_lattice(pcm))
    if fade is None:
        return asdict(evidence), None
    lag = support["shift_samples"]
    start, end = support["output_samples_half_open"]
    left, right = start + fade, min(start + lag, end - fade)
    witnessed = any(
        span.lag_samples == lag
        and min(right, span.second[1]) - max(left, span.second[0]) >= MIN_REPEAT_SAMPLES
        for span in evidence.spans
    )
    return asdict(evidence), witnessed


def old_voice(row):
    found = [
        entry
        for entry in row["detector_stage_outputs"]["detect"]["detectors"]
        if entry["detector_id"] == ROUTE
    ]
    require(
        row["status"] == "done"
        and len(found) == 1
        and found[0]["status"] in {"ok", "not_applicable"}
        and all(
            row["sampled_input"][key] is True
            for key in ("audio_pcm_equal", "audio_origin_equal", "audio_samples_equal")
        ),
        "A complete serving case with exact audio and a terminal voice route is required",
    )
    entry = found[0]
    if entry["status"] == "not_applicable":
        return {"status": "not_applicable", "raw": None, "decision": "unavailable", "windows": []}
    screening = entry["details"]["audio_screening"]
    require(screening["scope_supported"] is True, "Expected one fully supported short speech window")
    return {
        "status": "ok",
        "raw": entry["aggregate"]["raw"],
        "decision": screening["decision"],
        "windows": [
            [int(unit["quality"][key]) for key in ("start_sample", "end_sample")]
            for unit in entry["unit_scores"]
        ],
    }


def history():
    """Rebind the exact ordered serving artifacts already validated by the v156 replay."""
    complete = read(SERVING_RESULT)
    require(
        complete["complete_guarded_batches"] == 16
        and complete["overall"]["completed"] == 176
        and complete["serving_freeze_sha256"] == sha(serving.PUBLIC),
        "Exact complete v156 pilot required",
    )
    commitments, output = [], {}
    for ordinal in range(16):
        stage = serving.stage_folder(ordinal)
        result = speech.qualification_profile(stage, None, {}, "grid_serving_baseline_v156.py")
        journal = serving.WORK / f"clip-{ordinal:02d}/cases.jsonl"
        require(
            result["clip_ordinal"] == ordinal
            and result["freeze_sha256"] == complete["serving_freeze_sha256"]
            and result["done_cases"] == 11
            and result["private_cases_sha256"] == sha(journal),
            "Prior serving receipt or case journal differs",
        )
        commitments.append(
            {
                "profile_sha256": sha(stage / "profile.json"),
                "worker_receipt_sha256": sha(stage / "output/result.json"),
                "case_journal_sha256": result["private_cases_sha256"],
            }
        )
        rows = {row["condition"]: row for row in speech.planning.reviewed.lines(journal)}
        require(len(rows) == 11, "All eleven prior conditions are required")
        output[ordinal] = {
            "original": old_voice(rows["original"]),
            "hard_copy": old_voice(rows["audio_delay_320ms"]),
        }
    digest = hashlib.sha256(
        json.dumps(commitments, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        digest == complete["ordered_private_artifact_commitments_sha256"], "Prior receipt collection differs"
    )
    return output


def source_rows():
    plan = speech.planning.load_completed()
    sources = [row for row in plan["sources"] if row["role"] == "pilot"]
    require([row["ordinal"] for row in sources] == list(range(16)), "Exactly sixteen consumed pilot sources")
    return sources


def inputs():
    previous = history()
    rows = []
    for source in source_rows():
        ordinal = source["ordinal"]
        source_input = source["input"]
        archive = speech.ARCHIVE.resolve()
        require(
            all(
                Path(source_input[key]).resolve().is_relative_to(archive) for key in ("wav_file", "pcm_file")
            ),
            "Only the verified prior source bank may supply PCM",
        )
        speech.planning.reviewed.verify_pcm(source_input)
        pcm = np.load(source_input["pcm_file"], allow_pickle=False)
        cases = speech.planning.reviewed.lines(
            serving.baseline.clips.CACHE / f"clip-{ordinal:02d}/cases.jsonl"
        )
        old_pcm = {
            case["condition"]: case["expected"]["pcm_s16_sha256"]
            for case in cases
            if case["condition"] in {"original", "audio_delay_320ms"}
        }
        variants = {name: render(pcm, source["audio_delay"], fade) for name, fade in ARMS.items()}
        digests = {name: pcm_digest(variant) for name, variant in variants.items()}
        require(
            digests["original"] == old_pcm["original"]
            and digests["hard_copy"] == old_pcm["audio_delay_320ms"],
            "Historical PCM mismatch",
        )
        rows.append(
            {
                "ordinal": ordinal,
                "component": source["component_index"],
                "pcm_file": source_input["pcm_file"],
                "pcm_file_sha256": source_input["pcm_file_sha256"],
                "copy": source["audio_delay"],
                "pcm_sha256_by_arm": digests,
                "prior": previous[ordinal],
            }
        )
    return rows


def identity():
    return {
        "schema": "chipmunk.grid-audio-splice-probe-freeze/167",
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "source_freeze_sha256": sha(speech.planning.PUBLIC),
        "prior_serving_result_sha256": sha(SERVING_RESULT),
        "private_inputs_sha256": sha(TRIAL / "private-inputs.json"),
        "local_asset_commitments": ASSETS,
        "runtime_packages": {
            key: importlib.metadata.version(key)
            for key in ("numpy", "scipy", "torch", "transformers", "pyloudnorm", "psutil")
        },
        "sample_rate": SR,
        "crossfade_samples_per_boundary": ARMS,
        "planned_cases": 16 * len(ARMS),
        "score_parity_absolute_tolerance": RAW_ATOL,
        "symmetric_stability_maximum_log_odds_change": math.log(2),
        "new_fitting_permitted": False,
        "scope": "Consumed genuine speech diagnostic only. No false-negative estimate, fresh confirmation or serving change.",
    }


def verify():
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == identity(), "Exact frozen splice probe required")


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every prior attempt")
    rows = inputs()
    require(
        all(sha(WEIGHTS / name) == digest for name, digest in ASSETS.items()), "Pinned asset bytes differ"
    )
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "private-inputs.json", rows)
    lock = identity()
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"status": "frozen", "planned_cases": lock["planned_cases"], "new_neural_calls": 0}))


def checked_history(result, previous):
    require(
        result["status"] == previous["status"]
        and result["decision"] == previous["decision"]
        and result["windows"] == previous["windows"],
        "Prior serving availability, selected windows or direction differs",
    )
    if previous["raw"] is None:
        require(result["raw"] is None, "Unavailable baseline has a score")
        return None
    error = abs(result["raw"] - previous["raw"])
    require(error <= RAW_ATOL, "Prior serving raw score failed parity")
    return error


def score(pcm, model, activity, deadline):
    from app.detectors.audio import calibration
    from scripts.audio_calibration_v3_methods import band_view, score_views

    observed = score_views(pcm, model, activity)
    units = observed["windows"]
    intervals = [[row["start_sample"], row["end_sample"]] for row in units]
    if not units:
        return {
            "status": "not_applicable",
            "raw": None,
            "band": None,
            "windows": [],
            "decision": "unavailable",
            "symmetric": "unavailable",
            "delta": None,
        }
    require(intervals == [[0, len(pcm)]], "Expected exactly one whole-clip speech window")

    class SavedBand:
        def predict(self, chunks):
            require(
                len(chunks) == 1
                and chunks[0].tobytes() == band_view(pcm).tobytes()
                and observed["band_unrounded"] is not None,
                "The baseline band request differs from the actual probe",
            )
            return [observed["band_unrounded"]]

    raw, band = observed["raw"], observed["band"]
    baseline = calibration.evaluate(observed["raw_unrounded"], pcm, [(0, pcm)], SavedBand(), deadline)
    delta = abs(calibration.logit(raw) - calibration.logit(band)) if band is not None else None
    decision = baseline["decision"]
    symmetric = "inconclusive" if decision == "ai" and (delta is None or delta > math.log(2)) else decision
    return {
        "status": "ok",
        "raw": raw,
        "band": band,
        "windows": intervals,
        "decision": decision,
        "symmetric": symmetric,
        "delta": delta,
        "calibrated_probability_ai": baseline["probability_ai"],
        "probe_available": band is not None,
    }


def summary(rows):
    require(
        [(row["ordinal"], row["arm"]) for row in rows]
        == [(ordinal, arm) for ordinal in range(16) for arm in ARMS],
        "All frozen rows are required in their exact order",
    )
    result = {}
    hard_copy = {row["ordinal"]: row for row in rows if row["arm"] == "hard_copy"}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        scores = [row["raw"] for row in subset if row["raw"] is not None]
        parity = [
            row["prior_raw_absolute_error"] for row in subset if row["prior_raw_absolute_error"] is not None
        ]
        paired = [(hard_copy[row["ordinal"]], row) for row in subset]
        deltas = [b["raw"] - a["raw"] for a, b in paired if a["raw"] is not None and b["raw"] is not None]
        result[arm] = {
            "planned": len(subset),
            "baseline_directions": dict(Counter(row["decision"] for row in subset)),
            "symmetric_directions": dict(Counter(row["symmetric"] for row in subset)),
            "contributing_components": len({row["component"] for row in subset}),
            "assessed_raw_median": median(scores) if scores else None,
            "probe_available": sum(row.get("probe_available", False) for row in subset),
            "previous_scored_cases_with_parity": len(parity),
            "max_prior_raw_absolute_error": max(parity) if parity else None,
            "paired_both_scored_vs_hard_copy": len(deltas),
            "median_paired_raw_change_vs_hard_copy": median(deltas) if deltas else None,
            "same_immediate_boundary_jumps_as_original": sum(
                row["boundary_jumps_match_original"] for row in subset
            ),
            "repetition_outcomes": dict(Counter(row["repetition"]["status"] for row in subset)),
            "cases_with_exact_repeat": sum(bool(row["repetition"]["spans"]) for row in subset),
            "cases_with_observed_match_in_copy_support": sum(row["witnessed_copy"] is True for row in subset),
        }
    return result


def worker(output):
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    stage = TRIAL / "authored"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Exact offline guarded worker required",
    )
    verify()
    output.mkdir(mode=0o700, exist_ok=False)
    model = load_voice({"weights": str(WEIGHTS), "assets": ASSETS})
    activity = SpeechActivity(WEIGHTS / "silero_vad.jit")
    forwards = 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    handle = model.register_forward_pre_hook(count)
    journal = output / "private-observations.jsonl"
    rows = []
    try:
        with journal.open("x") as stream:
            for source in read(TRIAL / "private-inputs.json"):
                path = Path(source["pcm_file"])
                require(sha(path) == source["pcm_file_sha256"], "The admitted source PCM changed")
                pcm = np.load(path, allow_pickle=False)
                original_jumps = seam_jumps(pcm, source["copy"])
                for arm, fade in ARMS.items():
                    variant = render(pcm, source["copy"], fade)
                    require(pcm_digest(variant) == source["pcm_sha256_by_arm"][arm], "Variant bytes differ")
                    before = forwards
                    outcome = score(variant, model, activity, Deadline(120))
                    require(
                        forwards - before == 2 * len(outcome["windows"]),
                        "Unexpected raw or band forward count",
                    )
                    prior = source["prior"].get(arm)
                    error = checked_history(outcome, prior) if prior else None
                    jumps = seam_jumps(variant, source["copy"])
                    repetition, witnessed = repeated_samples(variant, source["copy"], fade)
                    row = {
                        "ordinal": source["ordinal"],
                        "component": source["component"],
                        "arm": arm,
                        "boundary_jumps_signed16_units": jumps,
                        "boundary_jumps_match_original": jumps == original_jumps,
                        "repetition": repetition,
                        "witnessed_copy": witnessed,
                        "prior_raw_absolute_error": error,
                        **outcome,
                    }
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
    finally:
        handle.remove()
    write(
        output / "result.json",
        {
            "schema": "chipmunk.grid-audio-splice-probe-result/167",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "planned_cases": len(rows),
            "new_neural_calls": forwards,
            "arms": summary(rows),
            "defaults_changed": False,
        },
    )


def routed_popen(argv, **kwargs):
    output = TRIAL / "authored/output"
    require(
        argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored", str(output)],
        "Exact resource monitor command required",
    )
    return subprocess.Popen([*argv[:3], SCRIPT, "worker", str(output)], **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda _command, _batch: True,
            "__file__": __file__,
        },
    )("authored")


def readout():
    verify()
    folder = TRIAL / "authored"
    journal = folder / "output/private-observations.jsonl"
    raw = read(folder / "output/result.json")
    result = speech.qualification_profile(folder, raw["new_neural_calls"], {}, Path(__file__).name)
    rows = speech.planning.reviewed.lines(journal)
    require(
        result["freeze_sha256"] == sha(PUBLIC)
        and result["private_observations_sha256"] == sha(journal)
        and result["arms"] == summary(rows)
        and result["new_neural_calls"] == sum(2 * len(row["windows"]) for row in rows),
        "Actual complete voice replay journal differs",
    )
    profile = read(folder / "profile.json")
    return {
        **result,
        "private_resource_profile_sha256": sha(folder / "profile.json"),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require((args.command == "worker") == (args.output is not None), "Only the worker takes an output path")
    if args.command == "worker":
        worker(args.output)
    elif args.command == "readout":
        print(json.dumps(readout(), indent=2, allow_nan=False))
    else:
        (freeze if args.command == "freeze" else run)()
