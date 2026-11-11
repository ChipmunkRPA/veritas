"""Prepare and score the frozen Alania synthetic-only speech panel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import types
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, read_wav, source_ranges
from scripts import audio_alania_roster_v178 as roster
from scripts import audio_matched_edit_diagnostic_v174 as route
from scripts.video_appearance_corrected_v112 import scoped

baseline, runtime = route.baseline, route.runtime
ROOT, BASE, PANEL = route.ROOT, route.BASE, route.PANEL
TRIAL = BASE / "audio-alania-transfer-v179"
PUBLIC = PANEL / "audio-alania-transfer-freeze-v179.json"
MODULE = "scripts.audio_alania_transfer_v179"
MODES = ("reference", "reference+instruction")
NATIVE_RATE = 48000
MIN_SAMPLES = SAMPLE_RATE * 5 // 2
MAX_SAMPLES = 4 * SAMPLE_RATE
MAX_PAYLOAD_BYTES = 2_000_000
MAX_ROW_GROUP_BYTES = 48_000_000
CODE = (
    "backend/scripts/audio_alania_transfer_v179.py",
    "backend/tests/test_audio_alania_transfer_v179.py",
    "docs/audio-alania-transfer-v179-method.md",
)
read, write, sha, require = route.read, route.write, route.sha, route.require


def prepared_dir():
    return TRIAL / "bind-00/output/prepared"


def roster_commitments():
    frozen = read(roster.PUBLIC)
    private = roster.TRIAL / "private-roster.json"
    selected = read(private)
    require(
        frozen == read(roster.TRIAL / "freeze.json")
        and frozen["schema"] == "chipmunk.audio-alania-roster-freeze/178"
        and frozen["publisher_revision"] == roster.REVISION
        and frozen["shard"] == roster.SHARD
        and frozen["shard_sha256"] == roster.SHARD_SHA
        and frozen["shard_bytes"] == roster.SHARD_SIZE
        and frozen["model_calls"] == frozen["decoded_audio_rows"] == 0
        and frozen["private_roster_sha256"] == sha(private)
        and all(sha(ROOT / name) == digest for name, digest in frozen["code_sha256"].items())
        and all(
            sha(roster.SOURCE / name) == digest for name, digest in frozen["publisher_records_sha256"].items()
        )
        and [row["slot"] for row in selected] == [f"positive-{i:02d}" for i in range(roster.PLANNED)]
        and frozen["selected_rows"] == roster.PLANNED
        and len({row["voice_id"] for row in selected})
        == len({row["text_doc_id"] for row in selected})
        == roster.PLANNED
        and len({(row["row_group"], row["row_offset"]) for row in selected}) == roster.PLANNED
        and all(roster.eligible(row) for row in selected)
        and Counter(row["render_mode"] for row in selected) == frozen["selected_render_modes"],
        "Exact independent v178 roster and publisher commitments required",
    )
    return selected, frozen


def context():
    selected, source = roster_commitments()
    serving = read(route.PUBLIC)
    packages = {name: importlib.metadata.version(name) for name in serving["runtime_packages"]}
    require(
        serving == read(route.TRIAL / "freeze.json")
        and serving["schema"] == "chipmunk.audio-matched-edit-diagnostic-freeze/174"
        and serving["local_asset_commitments"] == baseline.ASSETS
        and serving["serving_original_side_probe_only"] is True
        and packages == serving["runtime_packages"]
        and all(sha(ROOT / name) == digest for name, digest in serving["code_sha256"].items()),
        "Exact pinned English serving route, assets and runtime required",
    )
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "The production decoder tools are required")
    return {
        "schema": "chipmunk.audio-alania-transfer-freeze/179",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "roster_freeze_sha256": sha(roster.PUBLIC),
        "private_roster_sha256": source["private_roster_sha256"],
        "voice_route_freeze_sha256": sha(route.PUBLIC),
        "local_asset_commitments": baseline.ASSETS,
        "runtime_packages": {**packages, "pyarrow": importlib.metadata.version("pyarrow")},
        "decoder_tool_sha256": {name: sha(Path(path).resolve()) for name, path in tools.items()},
        "planned_cases": len(selected),
        "planned_by_render_mode": source["selected_render_modes"],
        "input_rate": NATIVE_RATE,
        "production_rate": SAMPLE_RATE,
        "production_sample_count_bounds": [MIN_SAMPLES, MAX_SAMPLES],
        "positive_only_language": "tr",
        "fitting_permitted": False,
        "holdout_access_permitted": False,
    }, selected


def verify():
    lock, selected = context()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact frozen Alania transfer required")
    return selected


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve prior attempts")
    lock, _ = context()
    require(
        all(sha(baseline.WEIGHTS / name) == digest for name, digest in baseline.ASSETS.items()),
        "Pinned model assets changed",
    )
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    print(json.dumps({"status": "frozen", "planned_cases": lock["planned_cases"], "new_neural_calls": 0}))


def selected_payloads(file, selected):
    groups = defaultdict(list)
    for row in selected:
        groups[row["row_group"]].append(row)
    for group, planned in groups.items():
        require(type(group) is int and 0 <= group < file.num_row_groups, "Invalid frozen Parquet row group")
        metadata = file.metadata.row_group(group)
        offsets = [row["row_offset"] for row in planned]
        require(
            len(set(offsets)) == len(offsets)
            and all(type(offset) is int and 0 <= offset < metadata.num_rows for offset in offsets)
            and metadata.total_byte_size <= MAX_ROW_GROUP_BYTES,
            "Frozen row offsets or bounded Parquet group differ",
        )
        index = pa.array(offsets, type=pa.int64())
        rows = file.read_row_group(group, columns=list(roster.COLUMNS), use_threads=False).take(index)
        require(
            rows.to_pylist() == [{name: row[name] for name in roster.COLUMNS} for row in planned],
            "Frozen selected Parquet metadata differs",
        )
        audio = file.read_row_group(group, columns=["audio"], use_threads=False).take(index)
        require(audio.schema.names == ["audio"], "Only the clean audio column is permitted")
        for row, embedded in zip(planned, audio.column("audio").to_pylist(), strict=True):
            yield row, embedded


def source_range(count):
    return {
        "start_sample": 0,
        "end_sample": count,
        "source_start_pts": 0,
        "source_end_pts": count,
        "time_base": f"1/{SAMPLE_RATE}",
    }


def prepare_case(row, embedded, scratch, output):
    result = {"slot": row["slot"], "render_mode": row["render_mode"]}
    payload = embedded.get("bytes") if isinstance(embedded, dict) else None
    if (
        not isinstance(payload, bytes)
        or not payload.startswith(b"fLaC")
        or not 0 < len(payload) <= MAX_PAYLOAD_BYTES
    ):
        return {**result, "status": "unavailable", "reason": "missing_or_unsupported_clean_flac"}
    source_hash = sha256(payload).hexdigest()
    try:
        scratch.write_bytes(payload)
        decoded = decode_audio(scratch, source_sha256=source_hash, max_seconds=5.0)
        count = len(decoded.pcm)
        mapping = decoded.mapping
        actual_duration = count / SAMPLE_RATE
        native_duration = mapping["source_duration_s"]
        require(
            MIN_SAMPLES <= count <= MAX_SAMPLES
            and mapping["source_sample_rate"] == NATIVE_RATE
            and mapping["source_codec"] == "flac"
            and native_duration is not None
            and math.isclose(actual_duration, native_duration, rel_tol=0.0, abs_tol=1 / SAMPLE_RATE)
            and math.isclose(actual_duration, row["duration"], rel_tol=0.0, abs_tol=0.01 + 1 / SAMPLE_RATE)
            and source_ranges(mapping, 0, count) == [source_range(count)],
            "Expected full eligible native extent on the zero-origin production clock",
        )
        pcm_hash = sha256(read_wav(decoded.wav).astype("<f4").tobytes()).hexdigest()
        require(pcm_hash == mapping["pcm_sha256"], "Production PCM and saved WAV differ")
        with output.open("xb") as stream:
            stream.write(decoded.wav)
    except (OSError, ValueError) as error:
        return {
            **result,
            "status": "unavailable",
            "reason": "decode_clock_or_write_failed",
            "source_sha256": source_hash,
            "private_error": str(error),
        }
    return {
        **result,
        "status": "qualified",
        "source_sha256": source_hash,
        "wav_sha256": sha(output),
        "pcm_sha256": pcm_hash,
        "source_range": source_range(count),
        "samples": count,
    }


def exact_output(output, stage):
    folder = TRIAL / stage
    require(
        output == folder / "output"
        and (folder / "resources.jsonl").is_file()
        and (folder / "worker.log").is_file()
        and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "Only the exact offline guarded worker may run",
    )


def outcome_key(row):
    return row["slot"], row["render_mode"]


def prep_summary(rows, selected):
    require(
        [outcome_key(row) for row in rows] == [outcome_key(row) for row in selected]
        and all(row["status"] in {"qualified", "unavailable"} for row in rows),
        "Every frozen case requires one ordered preparation outcome",
    )
    return {
        mode: dict(Counter(row["status"] for row in rows if row["render_mode"] == mode)) for mode in MODES
    }


def prepare_worker(output):
    exact_output(output, "bind-00")
    selected = verify()
    shard = roster.source_identity()
    require(
        shard.parent.resolve(strict=True) == shard.parent, "Pinned archive directory cannot be redirected"
    )
    output.mkdir(mode=0o700, exist_ok=False)
    prepared_dir().mkdir(mode=0o700, exist_ok=False)
    observed = {}
    with TemporaryDirectory(prefix="alania-source-") as tmp, pq.ParquetFile(shard, pre_buffer=False) as file:
        for row, embedded in selected_payloads(file, selected):
            slot = row["slot"]
            observed[slot] = prepare_case(
                row, embedded, Path(tmp) / "source.flac", prepared_dir() / f"{slot}.wav"
            )
    rows = [observed[row["slot"]] for row in selected]
    journal = output / "private-preparation.json"
    write(journal, rows)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-alania-preparation-result/179",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_preparation_sha256": sha(journal),
            "planned_cases": len(rows),
            "by_render_mode": prep_summary(rows, selected),
            "unavailable_reasons": dict(
                Counter(row["reason"] for row in rows if row["status"] == "unavailable")
            ),
            "new_neural_calls": 0,
        },
    )


def read_pcm(row):
    count = row["samples"]
    require(
        row["status"] == "qualified"
        and type(count) is int
        and MIN_SAMPLES <= count <= MAX_SAMPLES
        and row["source_range"] == source_range(count),
        "Expected qualified complete zero-origin production extent",
    )
    media = prepared_dir()
    path = media / f"{row['slot']}.wav"
    require(
        path.parent == media
        and media.resolve(strict=True) == media
        and path.is_file()
        and not path.is_symlink(),
        "Only a direct qualified Alania WAV is permitted",
    )
    data = path.read_bytes()
    require(sha256(data).hexdigest() == row["wav_sha256"], "Prepared WAV bytes changed")
    pcm = read_wav(data)
    require(
        len(pcm) == count and sha256(pcm.astype("<f4").tobytes()).hexdigest() == row["pcm_sha256"],
        "Prepared production PCM differs from its exact receipt",
    )
    return pcm


def verified_preparation(selected):
    stage = TRIAL / "bind-00"
    result_path = stage / "output/result.json"
    profile_path = stage / "profile.json"
    resource_path = stage / "resources.jsonl"
    journal = stage / "output/private-preparation.json"
    result, profile, rows = read(result_path), read(profile_path), read(journal)
    samples = baseline.speech.planning.reviewed.lines(resource_path)
    require(
        profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == profile["resource_limit_observations"] == []
        and profile["command"] == "bind"
        and profile["batch"] == 0
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(resource_path)
        and profile["runtime_sha256"] == sha(Path(__file__))
        and result["schema"] == "chipmunk.audio-alania-preparation-result/179"
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_preparation_sha256"] == sha(journal)
        and result["planned_cases"] == len(selected) == len(rows)
        and result["by_render_mode"] == prep_summary(rows, selected)
        and result["unavailable_reasons"]
        == Counter(row["reason"] for row in rows if row["status"] == "unavailable")
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and samples
        and all(
            not item["violations"]
            and not runtime.violations(item, "bind", initial=i == 0)
            and item["archive_free_disk_bytes"] >= (3 * 1024**3 if i == 0 else 1536 * 1024**2)
            for i, item in enumerate(samples)
        )
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(item["tree_rss_bytes"] for item in samples)
        and samples[-1]["elapsed_s"] <= profile["duration_s"] <= 900,
        "Exact successful guarded data-only receipt required",
    )
    for row in rows:
        if row["status"] == "qualified":
            read_pcm(row)
    return rows, {
        **result,
        "resource_profile_sha256": sha(profile_path),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


def score_voice(pcm, model, activity, deadline):
    from app.detectors.audio import calibration
    from app.detectors.video.adapters.audio import PellaVoiceDetector
    from app.detectors.video.adapters.common import AudioTrack

    require(MIN_SAMPLES <= len(pcm) <= MAX_SAMPLES, "Exact eligible clip extent required")
    lease = types.SimpleNamespace(get=lambda _: types.SimpleNamespace(obj=activity))
    outputs = types.SimpleNamespace(audio=AudioTrack("analysed", pcm, len(pcm)))
    context = types.SimpleNamespace(profile="full", lease=lease, outputs=outputs)
    windows = PellaVoiceDetector()._windows(context)
    if not windows:
        return {
            "route_status": "not_applicable",
            "windows": [],
            "decision": "unavailable",
            "stability_probe": "not_run",
        }
    ((start, chunk),) = windows
    require(start == 0 and len(chunk) == len(pcm), "One complete production speech window required")
    (raw,) = model.predict([chunk])
    require(calibration.valid(raw), "Invalid raw voice response")
    result = calibration.evaluate(float(raw), pcm, windows, model, deadline)
    return {
        "route_status": "ok",
        "windows": [[0, len(pcm)]],
        "decision": result["decision"],
        "raw": float(raw),
        "english_reference_estimate_ai": result["probability_ai"],
        "stability_probe": result["probe_status"],
        "stability_log_odds_change": result["stability_log_odds_change"],
    }


def score_metrics(rows):
    planned = len(rows)
    counts = Counter(row["decision"] for row in rows)
    scored = sum(row["route_status"] == "ok" for row in rows)
    return {
        "planned_positives": planned,
        "scored": scored,
        "directions": {direction: counts[direction] for direction in route.DIRECTIONS},
        "route_statuses": dict(Counter(row["route_status"] for row in rows)),
        "missed_ai": planned - counts["ai"],
        "ai_flag_rate_over_planned": counts["ai"] / planned,
        "missed_ai_rate_over_planned": (planned - counts["ai"]) / planned,
        "scored_coverage_over_planned": scored / planned,
        "inconclusive_rate_over_planned": counts["inconclusive"] / planned,
        "unavailable_rate_over_planned": counts["unavailable"] / planned,
    }


def score_summary(rows, prepared):
    require(
        [outcome_key(row) for row in rows] == [outcome_key(row) for row in prepared],
        "Every frozen case requires one ordered scoring outcome",
    )
    for row, case in zip(rows, prepared, strict=True):
        expected = [[0, case["samples"]]] if row["route_status"] == "ok" else []
        require(
            row["windows"] == expected
            and row["decision"] in route.DIRECTIONS
            and (row["route_status"] == "ok") == (row["decision"] != "unavailable")
            and (row["route_status"] == "preparation_unavailable") == (case["status"] == "unavailable")
            and row["route_status"] in {"ok", "not_applicable", "preparation_unavailable"},
            "Serving coverage differs from the prepared sample extent",
        )
        route.check_forwards(row, row["pella_forwards"])
    return {
        "all_positive_cases": score_metrics(rows),
        "by_render_mode": {
            mode: score_metrics([row for row in rows if row["render_mode"] == mode]) for mode in MODES
        },
    }


def score_worker(output):
    exact_output(output, "authored")
    prepared, prep_receipt = verified_preparation(verify())
    from app.detectors.audio.upgraded import SpeechActivity
    from app.engine.boundary import Deadline
    from scripts.run_audio_specialists_v15 import load_voice

    output.mkdir(mode=0o700, exist_ok=False)
    model = load_voice({"weights": str(baseline.WEIGHTS), "assets": baseline.ASSETS})
    activity = SpeechActivity(baseline.WEIGHTS / "silero_vad.jit")
    forwards = 0

    def count(_module, _args):
        nonlocal forwards
        forwards += 1

    handle = model.register_forward_pre_hook(count)
    journal = output / "private-observations.jsonl"
    rows = []
    try:
        with journal.open("x") as stream:
            for case in prepared:
                before = forwards
                if case["status"] == "qualified":
                    outcome = score_voice(read_pcm(case), model, activity, Deadline(120))
                else:
                    outcome = {
                        "route_status": "preparation_unavailable",
                        "windows": [],
                        "decision": "unavailable",
                        "stability_probe": "not_run",
                    }
                calls = forwards - before
                route.check_forwards(outcome, calls)
                row = {
                    "slot": case["slot"],
                    "render_mode": case["render_mode"],
                    **outcome,
                    "pella_forwards": calls,
                }
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
    finally:
        handle.remove()
    raw = sum(len(row["windows"]) for row in rows)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-alania-transfer-result/179",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "preparation_profile_sha256": prep_receipt["resource_profile_sha256"],
            "private_observations_sha256": sha(journal),
            "language": "tr",
            "scope": "Positive-only transfer of an English-calibrated route to clean, publisher-filtered Turkish speech from one generator.",
            **score_summary(rows, prepared),
            "new_neural_calls": forwards,
            "serving_raw_forwards": raw,
            "serving_stability_forwards": forwards - raw,
            "holdout_scored": False,
            "defaults_changed": False,
        },
    )


def run(command):
    data_only = command == "prepare"
    stage, mode, batch = ("bind-00", "bind", 0) if data_only else ("authored", "authored", None)
    output = TRIAL / stage / "output"
    guarded_worker = "prepare-worker" if data_only else "score-worker"

    def ready(*_):
        if data_only:
            return True
        if not (TRIAL / "bind-00/profile.json").is_file():
            return False
        verified_preparation(verify())
        return True

    def launch(argv, **kwargs):
        require(
            argv[1:]
            == [
                "-B",
                "-m",
                "scripts.video_appearance_native_v108",
                mode,
                str(output),
                *(["0"] if data_only else []),
            ],
            "Exact scoped v108 worker invocation required",
        )
        return subprocess.Popen([*argv[:3], MODULE, guarded_worker, str(output)], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {"TRIAL": TRIAL, "verify": verify, "subprocess": proxy, "prerequisites": ready, "__file__": __file__},
    )(mode, batch)


def readout():
    prepared, prep_receipt = verified_preparation(verify())
    stage = TRIAL / "authored"
    journal = stage / "output/private-observations.jsonl"
    rows = baseline.speech.planning.reviewed.lines(journal)
    summary = score_summary(rows, prepared)
    calls = sum(row["pella_forwards"] for row in rows)
    raw = sum(len(row["windows"]) for row in rows)
    result = baseline.speech.qualification_profile(stage, calls, {}, Path(__file__).name)
    require(
        result["schema"] == "chipmunk.audio-alania-transfer-result/179"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["preparation_profile_sha256"] == prep_receipt["resource_profile_sha256"]
        and result["private_observations_sha256"] == sha(journal)
        and result["serving_raw_forwards"] == raw
        and result["serving_stability_forwards"] == calls - raw
        and all(result[key] == value for key, value in summary.items())
        and result["language"] == "tr"
        and result["holdout_scored"] is False
        and result["defaults_changed"] is False,
        "Exact guarded Turkish transfer receipt required",
    )
    profile = read(stage / "profile.json")
    return {
        **result,
        "resource_profile_sha256": sha(stage / "profile.json"),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "prepare", "prepare-worker", "prepare-readout", "run", "score-worker", "readout"),
    )
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require(
        args.command.endswith("-worker") == (args.output is not None),
        "Only guarded workers take an output path",
    )
    if args.command == "prepare-worker":
        prepare_worker(args.output)
    elif args.command == "score-worker":
        score_worker(args.output)
    elif args.command == "freeze":
        freeze()
    elif args.command in {"prepare", "run"}:
        run(args.command)
    else:
        result = verified_preparation(verify())[1] if args.command == "prepare-readout" else readout()
        print(json.dumps(result, indent=2, allow_nan=False))
