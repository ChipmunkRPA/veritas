"""Fixed consumed-source extraction parity for the inference-only audio runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil

from app.detectors.audio import combined
from scripts import local_probe_limits_v24 as limits
from scripts.audio_tree_sampler_v64 import sample_tree
from scripts.qualify_audio_combined_v97 import NumericalBudget

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-combined-extraction-v98"
SOURCE = BASE / "audio-confidence-mixtures-v82/variants.json"
STUDY = BASE / "audio-combined-confidence-v84"
SERVING = BASE / "audio-combined-serving-v97"
ARCHIVE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify")
PUBLIC = ROOT / "docs/evaluation-panels/audio-combined-extraction-freeze-v98.json"
KINDS = ("genuine", "clone", "expressive", "stock")
SOURCE_SHA = "8411c76325bc4dcde649db08ed05747c5eb917df13d811d1068b7395ee562d24"
CODE = (
    "backend/scripts/qualify_audio_extraction_v98.py",
    "backend/tests/test_audio_extraction_plan_v98.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
    "docs/audio-combined-extraction-v98-protocol.md",
)
sha, require = combined.sha, combined.require


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def selection(rows):
    """One hash-ordered foreground per truth/family; keep all nine conditions."""
    selected = []
    for kind in KINDS:
        options = {r["base_id"] for r in rows if r["role"] == "fusion_fit" and r["kind"] == kind}
        require(bool(options), "Every fixed source family required")
        base = min(
            options, key=lambda value: hashlib.sha256(("serving-parity-v98/" + value).encode()).hexdigest()
        )
        group = [r for r in rows if r["role"] == "fusion_fit" and r["kind"] == kind and r["base_id"] == base]
        require(
            len(group) == len({r["condition"] for r in group}) == 9, "All nine source conditions required"
        )
        selected.extend(sorted(group, key=lambda r: r["condition"]))
    require(
        len(selected) == 36 and len({r["id"] for r in selected}) == 36,
        "Every unique fixed extraction input required",
    )
    return selected


def code():
    old = read(STUDY / "freeze.json")
    prior = read(SERVING / "freeze.json")
    # Original extraction dependencies stay exact until this parity check ends.
    paths = set(old["code_sha256"]) | set(prior["code_sha256"]) | set(CODE)
    require(
        all(sha(ROOT / p) == v for p, v in old["code_sha256"].items()),
        "Original extraction implementation changed",
    )
    require(
        all(sha(ROOT / p) == v for p, v in prior["code_sha256"].items()),
        "Reviewed inference implementation changed",
    )
    return {p: sha(ROOT / p) for p in sorted(paths)}


def prepare():
    with NumericalBudget(BASE / "audio-combined-extraction-v98-preparation-profile") as budget:
        require(sha(SOURCE) == SOURCE_SHA, "Original complete variant plan required")
        rows = selection(read(SOURCE)["variants"])
        require(
            not TRIAL.exists() and not PUBLIC.exists(), "Never attempted extraction qualification required"
        )
        TRIAL.mkdir()
        plan = {"schema": "chipmunk.audio-extraction-plan/98", "sources": rows}
        write(TRIAL / "plan.json", plan)
        frozen = {
            "schema": "chipmunk.audio-combined-extraction-freeze/98",
            "code_sha256": code(),
            "source_plan_sha256": SOURCE_SHA,
            "private_plan_sha256": sha(TRIAL / "plan.json"),
            "serving_freeze_sha256": sha(SERVING / "freeze.json"),
            "private_bundle_sha256": read(SERVING / "freeze.json")["private_inference_bundle_sha256"],
            "planned_inputs": 36,
            "fitting_foregrounds": 4,
            "conditions_per_foreground": 9,
            "selection": "Fixed salted hash order within each genuine/clone/expressive/stock fitting family, all nine conditions, no response filtering or substitutions",
            "tolerances": {
                "raw_probability": 1e-12,
                "combined_probability": 1e-12,
                "feature_bytes": "exact",
                "speech_spans_and_windows": "exact",
            },
            "runtime": {
                "device": "cpu",
                "dtype": "float32",
                "torch_threads": 4,
                "interop_threads": 1,
                "packages": {
                    n: importlib.metadata.version(n)
                    for n in ("torch", "transformers", "numpy", "scipy", "psutil", "pyloudnorm")
                },
            },
            "resources": {
                "initial_ram_bytes": limits.START_RAM,
                "initial_disk_bytes": limits.START_DISK,
                "rss_bytes": limits.RSS_LIMIT,
                "running_reserve_bytes": limits.RESERVE,
                "seconds": limits.SECONDS,
            },
            "new_neural_calls": 0,
            "defaults_changed": False,
            "scope": "Consumed fitting-source model/runtime parity, not a new accuracy test, final-set tuning, full analysis/payment/export qualification or outside review. Skips and failures remain explicit. No learned maps, media or individual records are published.",
        }
        budget.check()
        write(TRIAL / "freeze.json", frozen)
        write(PUBLIC, frozen)
        print(json.dumps({"status": "prepared", "inputs": 36, "new_neural_calls": 0}), flush=True)


def worker():
    import numpy as np

    from app.detectors.audio.decode import decode_audio
    from app.engine.boundary import Deadline
    from scripts import audio_score_fusion_v83 as fusion
    from scripts.score_audio_confirmation_v7 import models, weights

    frozen = read(TRIAL / "freeze.json")
    require(
        frozen == read(PUBLIC) and frozen["code_sha256"] == code(),
        "Exact published extraction implementation required",
    )
    require(
        frozen["runtime"]["packages"]
        == {n: importlib.metadata.version(n) for n in frozen["runtime"]["packages"]},
        "Frozen installed runtime required",
    )
    require(
        sha(SOURCE) == SOURCE_SHA and sha(TRIAL / "plan.json") == frozen["private_plan_sha256"],
        "Exact fixed source plan required",
    )
    require(
        read(TRIAL / "execution-start.json")["remote_freeze_blob_verified"] is True,
        "Remote publication must precede extraction",
    )
    require(
        read(SERVING / "replay-result.json")["status"] == "complete",
        "Completed full numerical parity required",
    )
    body = read(SOURCE)
    rows = selection(body["variants"])
    require(rows == read(TRIAL / "plan.json")["sources"], "Fixed source selection changed")
    original = read(STUDY / "freeze.json")
    require(
        weights() == original["baseline_weights"],
        "Original commercial baseline/VAD checkpoint bytes required",
    )
    expected = {}
    wanted = {r["id"] for r in rows}
    for index in range(12):
        path = ARCHIVE / "audio-confidence-extraction-v84" / f"chunk-{index:02d}-full.jsonl"
        require(
            sha(path) == read(STUDY / f"chunk-{index:02d}-complete.json")["full_sha256"],
            "Original complete response journal changed",
        )
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in wanted:
                    require(row["id"] not in expected, "Duplicate original parity reference")
                    expected[row["id"]] = row
    require(set(expected) == wanted, "All predetermined original references required")
    attempts = []

    def offline(event, args):
        if event == "socket.connect":
            attempts.append(event)
            raise RuntimeError("Offline extraction refused a network connection")

    sys.addaudithook(offline)
    baseline, activity = models()
    encoder_root = Path(read(BASE / "audio-shared-confirmation-v62/private-paths.json")["wavlm_root"])
    encoder = combined.load_encoder(encoder_root)
    calibration = combined.load_calibration(SERVING / "serving-bundle.json", frozen["private_bundle_sha256"])
    runtime = combined.CombinedSpeechRuntime(calibration, encoder)
    trained = read(STUDY / "trained.json")["global"]
    raw_scores, speech_spans, captured = [], [], []
    original_extract = combined.extract

    class VoiceRecorder:
        def predict(self, chunks):
            result = baseline.predict(chunks)
            raw_scores.extend(result)
            return result

    class ActivityRecorder:
        def spans(self, pcm):
            result = activity.spans(pcm)
            speech_spans.extend([list(s) for s in result])
            return result

    def capture(pcm, model):
        result = original_extract(pcm, model)
        captured.append(hashlib.sha256(np.asarray(result, "<f4").tobytes()).hexdigest())
        return result

    combined.extract = capture  # Observe the actual returned vector; do not replace its computation.
    assessed = unassessed = 0
    max_raw = max_p = 0.0
    try:
        with (TRIAL / "responses.jsonl").open("x") as journal:
            for source in rows:
                started = time.monotonic()
                path = Path(body["source_root"]) / source["path"]
                require(sha(path) == source["sha256"], "Original source media changed")
                decoded = decode_audio(path, source_sha256=source["sha256"])
                require(
                    decoded.mapping["pcm_sha256"] == source["pcm_sha256"]
                    and combined.source_supported(decoded.pcm),
                    "Original production-decoded sample clock required",
                )
                reference = expected[source["id"]]
                raw_scores.clear()
                speech_spans.clear()
                captured.clear()
                result = runtime.analyze(decoded.pcm, VoiceRecorder(), ActivityRecorder(), Deadline(300))
                require(speech_spans == reference["speech_spans_s"], "Actual speech gate spans differ")
                target = fusion.predict(reference, trained)
                if reference["status"] == "ok":
                    require(
                        result["status"] == "ok"
                        and len(raw_scores) == 1
                        and captured == [reference["wavlm_sha256"]],
                        "Actual source status, component calls or native representation differ",
                    )
                    raw_error, p_error = (
                        abs(raw_scores[0] - reference["raw"]),
                        abs(result["probability_ai"] - target["probability_ai"]),
                    )
                    require(
                        raw_error <= frozen["tolerances"]["raw_probability"]
                        and p_error <= frozen["tolerances"]["combined_probability"],
                        "Actual component/calibration parity failed",
                    )
                    require(
                        result["decision"] == target["decision"]
                        and result["checked_windows"]
                        == [{"start_sample": 0, "end_sample": 64000, "start_s": 0.0, "end_s": 4.0}],
                        "Tendency or source evidence window differs",
                    )
                    max_raw, max_p = max(max_raw, raw_error), max(max_p, p_error)
                    assessed += 1
                else:
                    require(
                        reference["status"] == "not_applicable"
                        and result["status"] == "not_applicable"
                        and result["probability_ai"] is None
                        and result["decision"] == "inconclusive"
                        and result["checked_windows"] == []
                        and not raw_scores
                        and not captured,
                        "Unassessed source cannot trigger models or an original verdict",
                    )
                    unassessed += 1
                journal.write(
                    json.dumps(
                        {
                            "id": source["id"],
                            "result": result,
                            "raw": raw_scores,
                            "feature_sha256": captured,
                            "runtime_s": time.monotonic() - started,
                        }
                    )
                    + "\n"
                )
                journal.flush()
                print(json.dumps({"completed": assessed + unassessed, "planned": 36}), flush=True)
    finally:
        combined.extract = original_extract
    require(assessed + unassessed == 36 and not attempts, "Every fixed input must complete offline")
    result = {
        "schema": "chipmunk.audio-combined-extraction-parity/98",
        "status": "complete",
        "cases": 36,
        "assessed_reproduced": assessed,
        "unassessed_preserved": unassessed,
        "maximum_raw_probability_error": max_raw,
        "maximum_combined_probability_error": max_p,
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "private_response_journal_sha256": sha(TRIAL / "responses.jsonl"),
        "new_main_baseline_forwards": assessed,
        "new_encoder_forwards": assessed,
        "defaults_changed": False,
        "scope": "Actual CPU extraction on consumed fitting sources; exact native features, source gate and four-second evidence clocks. Not new accuracy, full app/payment/export qualification or outside review.",
    }
    write(TRIAL / "result.json", result)
    print(json.dumps(result), flush=True)


def run():
    folder = TRIAL / "run-profile"
    folder.mkdir(exist_ok=False)
    require(
        not (TRIAL / "responses.jsonl").exists() and not (TRIAL / "result.json").exists(),
        "No automatic retry or overwrite",
    )
    environment = {
        k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE"}
    }
    environment.update(
        PYTHONPATH=str(ROOT / "backend"),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="2",
        VERITAS_DEVICE="cpu",
    )
    started, child, exit_code, guard = time.monotonic(), None, None, None
    samples, errors, transitions = [], [], []
    with (folder / "worker.log").open("x") as log, (folder / "resources.jsonl").open("x") as journal:

        def record(rss=0, initial=False):
            row = limits.observe(folder, started, rss)
            row["violations"] = limits.violations(row, initial=initial)
            row["archive_free_disk_bytes"] = shutil.disk_usage(ARCHIVE).free
            if not Path("/Volumes/SEAGATE").is_mount() or row["archive_free_disk_bytes"] < (
                limits.START_DISK if initial else limits.RESERVE
            ):
                row["violations"].append("archive_volume_or_disk_reserve")
            samples.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            return row["violations"]

        try:
            if record(initial=True):
                guard = "initial_headroom_unavailable"
            else:
                child = subprocess.Popen(
                    [sys.executable, "-B", "-m", "scripts.qualify_audio_extraction_v98", "worker"],
                    env=environment,
                    cwd=ROOT / "backend",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    process = psutil.Process(child.pid)
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    exit_code = child.wait(timeout=0.25)
                    process = None
                    transitions.append(
                        {"classification": "verified_worker_terminal_transition", "exit_code": exit_code}
                    )
                while process is not None and child.poll() is None:
                    try:
                        rss, events, terminal = sample_tree(process, child)
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        exit_code = child.wait(timeout=0.25)
                        transitions.append(
                            {"classification": "verified_worker_terminal_transition", "exit_code": exit_code}
                        )
                        break
                    transitions.extend(events)
                    if record(rss):
                        guard = "resource_guard"
                        break
                    if terminal:
                        break
                    time.sleep(0.25)
        except BaseException as exc:
            errors.append(type(exc).__name__)
            guard = "monitor_failure"
        finally:
            if child:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        exit_code = child.wait(timeout=0.25)
                        transitions.append(
                            {"classification": "verified_worker_exit_before_cleanup", "exit_code": exit_code}
                        )
                try:
                    exit_code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    exit_code = child.wait()
    profile = {
        "schema": "chipmunk.audio-extraction-profile/98",
        "child_started": child is not None,
        "child_exit_code": exit_code,
        "guard": guard,
        "sample_errors": errors,
        "terminal_observations": transitions,
        "duration_s": time.monotonic() - started,
        "maximum_sampled_tree_rss_bytes": max((r["tree_rss_bytes"] for r in samples), default=0),
        "minimum_available_ram_bytes": min((r["ram_available_bytes"] for r in samples), default=None),
        "resource_journal_sha256": sha(folder / "resources.jsonl"),
        "scope": "Unchanged neural900s/6GiB sampled tree RSS/6GiB initial RAM/3GiB initial disk/1.5GiB running reserves. Samples are not true peaks. No retry/overwrite.",
    }
    write(folder / "profile.json", profile)
    print(json.dumps(profile), flush=True)
    require(exit_code == 0 and guard is None and not errors, "Failed or guarded extraction retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker"))
    {"prepare": prepare, "run": run, "worker": worker}[parser.parse_args().command]()
