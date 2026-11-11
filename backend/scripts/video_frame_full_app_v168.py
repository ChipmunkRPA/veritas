"""Run two owned full-app arms on one verified genuine HD control."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import types
import uuid
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml

from scripts import grid_serving_baseline_v156 as serving
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

baseline = serving.baseline
ROOT, BASE, PANEL = serving.ROOT, serving.BASE, serving.PANEL
TRIAL = BASE / "video-frame-full-app-v168"
WORK = BASE / "video-frame-full-app-work-v168"
PREPARED = baseline.ARCHIVE / "hd-genuine-controls-v165/prepared-v166-01"
PREPARATION_SHA256 = "89f204be932f136aee8e2c487ce17c0a14725218b675d61ddda8df4bc8487d77"
GROUP, CASE, FILENAME = "01", "02", "sample.mp4"
DETECTOR = "vid.frames.dfdc_b7"
MODELS = ("dfdc.effnet_b7_555", "dfdc.effnet_b7_999")
ARMS = {
    "reference": ("0.3.0", "app.detectors.video.adapters.frames:FrameDFDCB7Detector", 8),
    "batch-one": ("0.3.1-lab.1", "scripts.video_frame_check_batch_one:FrameDFDCB7BatchOneDetector", 1),
}
INVENTORY = baseline.TRIAL / "private-inventory.json"
MODULE = "scripts.video_frame_full_app_v168"
read, write, sha, require = serving.read, serving.write, serving.sha, serving.require


def candidate_manifest(source):
    original = yaml.safe_load(source)
    expected_version, expected_adapter, _ = ARMS["reference"]
    version, adapter, _ = ARMS["batch-one"]
    start = f"  - id: {DETECTOR}\n    adapter: {expected_adapter}\n    version: {expected_version}\n".encode()
    replacement = f"  - id: {DETECTOR}\n    adapter: {adapter}\n    version: {version}\n".encode()
    require(source.count(start) == 1, "Exact original detector stanza required")
    changed = source.replace(start, replacement, 1)
    entries = [row for row in original["detectors"] if row["id"] == DETECTOR]
    require(len(entries) == 1 and tuple(entries[0]["models"]) == MODELS, "Exact model roster required")
    entries[0].update(version=version, adapter=adapter)
    require(yaml.safe_load(changed) == original, "Only one adapter and version may change")
    return changed


def preparation():
    complete = PREPARED / "complete.json"
    require(sha(complete) == PREPARATION_SHA256, "Exact completed v166 preparation required")
    source = read(complete)
    require(source["status"] == "complete" and source["cases_verified"] == 4, "Complete preparation required")
    group = next(row for row in source["groups"] if row["group"] == GROUP)
    index = {row["case"]: row for row in group["cases"]}
    directory = PREPARED / f"group-{GROUP}"
    require(
        set(index) == {"01", "02"} and sha(directory / "source.json") == group["source_record_sha256"],
        "Fixed paired genuine source required",
    )
    receipts = {}
    for case in sorted(index):
        path = directory / f"{case}.json"
        require(sha(path) == index[case]["receipt_sha256"], "Preparation case receipt changed")
        receipt = read(path)
        require(
            receipt["file"] == f"{case}.mp4"
            and receipt["case"] == case
            and receipt["file_sha256"] == index[case]["file_sha256"]
            and receipt["source_record_sha256"] == group["source_record_sha256"],
            "Prepared case binding changed",
        )
        receipts[case] = receipt
    primary, peer = receipts[CASE], receipts["01"]
    require(
        primary["mode"] == "ordinary_h264"
        and peer["mode"] == "reference"
        and primary["clock"] == peer["clock"]
        and primary["pcm_sha256"] == peer["pcm_sha256"],
        "Ordinary recompression must retain the paired input clock and PCM",
    )
    file = baseline.clips.owned_file(directory / primary["file"], PREPARED)
    require(
        file.stat().st_size == primary["bytes"] and sha(file) == primary["file_sha256"],
        "Prepared case bytes changed",
    )
    return file, primary, index[CASE]["receipt_sha256"]


def app_inputs(file, receipt):
    """Independent data-only conversions to the production RGB and mono PCM representations."""
    clock = receipt["clock"]
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file",
        "-copyts",
        "-i",
        str(file),
    ]
    rgb = subprocess.check_output(
        [
            *command,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-vf",
            "format=rgb24",
            "-enc_time_base",
            clock["video_time_base"],
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "pipe:1",
        ],
        timeout=120,
        text=True,
    )
    rows = [line.split(",") for line in rgb.splitlines() if line and not line.startswith("#")]
    require(
        f"#tb 0: {clock['video_time_base']}" in rgb.splitlines()
        and len(rows) == len(clock["picture_pts"]) == 150
        and all(len(row) == 6 for row in rows)
        and [int(row[2]) for row in rows] == clock["picture_pts"]
        and all(
            int(row[0]) == 0
            and int(row[4]) == 1920 * 1080 * 3
            and re.fullmatch(r"[0-9a-f]{64}", row[5].strip())
            for row in rows
        ),
        "Complete independent native RGB clock and pixels required",
    )
    pcm = subprocess.check_output(
        [
            *command,
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000,asetpts=N/SR/TB",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ],
        timeout=120,
    )
    samples = clock["audio_samples"] // 3
    require(len(pcm) == 2 * samples and clock["audio_origin_pts"] == 0, "Exact mono PCM support required")
    as_float = np.frombuffer(pcm, dtype="<i2").astype("<f4") / 32768.0
    return {
        "picture_pts": clock["picture_pts"],
        "video_time_base": clock["video_time_base"],
        "rgb_sha256": [row[5].strip() for row in rows],
        "audio_samples": samples,
        "pcm_sha256": hashlib.sha256(as_float.tobytes()).hexdigest(),
    }


def identity(inputs, receipt_sha):
    old = read(baseline.PUBLIC)
    observed = read(serving.WORK / "clip-00/runtime.json")
    parity_file = PANEL / "video-frame-batch-qualification-complete-v160.json"
    parity = read(parity_file)
    require(
        old["asset_inventory_sha256"] == sha(INVENTORY)
        and old["manifest_sha256"] == sha(baseline.MANIFEST)
        and old["code_sha256"] == baseline.code_identity()
        and old["runtime_packages"] == baseline.package_versions()
        and observed["resolved_device"] == parity["device"] == "mps"
        and observed["profile"] == "full"
        and parity["paired_frames"] == 15
        and parity["parity_within_frozen_tolerances"] is True,
        "Exact existing production application, assets and environment required",
    )
    return {
        "schema": "chipmunk.video-frame-full-app-freeze/168",
        "code_sha256": {
            Path(path).name: sha(path)
            for path in (
                __file__,
                serving.__file__,
                runtime.__file__,
                ROOT / "backend/scripts/video_frame_check_batch_one.py",
                ROOT / "backend/tests/test_video_frame_full_app_v168.py",
                ROOT / "docs/video-frame-full-app-v168-method.md",
            )
        },
        "baseline_freeze_sha256": sha(baseline.PUBLIC),
        "prior_runtime_sha256": sha(serving.WORK / "clip-00/runtime.json"),
        "prior_parity_sha256": sha(parity_file),
        "preparation_sha256": PREPARATION_SHA256,
        "private_case_receipt_sha256": receipt_sha,
        "private_inputs_sha256": sha(inputs),
        "arm_manifest_sha256": {arm: sha(TRIAL / f"{arm}.yaml") for arm in ARMS},
        "arm_adapter_specs": {arm: list(values) for arm, values in ARMS.items()},
        "model_manager_budget_bytes": observed["model_manager_budget_bytes"],
        "device": observed["resolved_device"],
        "resource_limits": {
            "wall_seconds": 900,
            "initial_ram_bytes": 6 * 1024**3,
            "sampled_tree_rss_bytes": 6 * 1024**3,
            "initial_disk_bytes": 3 * 1024**3,
            "ongoing_reserve_bytes": 1536 * 1024**2,
        },
        "scope": "One genuine CRF24 HD control paired across full-app arms. No sensitivity or threshold claim.",
    }


def freeze():
    require(not TRIAL.exists() and not WORK.exists(), "Preserve all prior or incomplete v168 attempts")
    baseline.checked_programs(read(baseline.PUBLIC))
    file, receipt, receipt_sha = preparation()
    inputs = app_inputs(file, receipt)
    source = baseline.MANIFEST.read_bytes()
    candidate = candidate_manifest(source)
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    (TRIAL / "reference.yaml").write_bytes(source)
    (TRIAL / "batch-one.yaml").write_bytes(candidate)
    write(TRIAL / "inputs.json", inputs)
    write(TRIAL / "freeze.json", identity(TRIAL / "inputs.json", receipt_sha))
    print(json.dumps({"status": "frozen", "freeze_sha256": sha(TRIAL / "freeze.json"), "arms": list(ARMS)}))


def verify():
    require(
        (TRIAL / "reference.yaml").read_bytes() == baseline.MANIFEST.read_bytes()
        and (TRIAL / "batch-one.yaml").read_bytes() == candidate_manifest(baseline.MANIFEST.read_bytes()),
        "Private manifest differential changed",
    )
    file, receipt, receipt_sha = preparation()
    frozen = identity(TRIAL / "inputs.json", receipt_sha)
    require(read(TRIAL / "freeze.json") == frozen, "Exact v168 freeze required")
    return frozen, file, receipt


def verify_input(outputs, expected, digest):
    probe, decoded = outputs.get("probe") or {}, outputs.get("decode") or {}
    table, rows = probe.get("frame_table") or {}, decoded.get("frames") or []
    clock_exact = bool(
        table
        and not table["truncated"]
        and not table["interpolated"]
        and Fraction(table["time_base"]) == Fraction(expected["video_time_base"])
        and table["pts"] == expected["picture_pts"]
        and (table["width"], table["height"]) == (1920, 1080)
    )
    planned = ((probe.get("plan") or {}).get("legacy") or {}).get("frames") or []
    rgb_exact = bool(
        clock_exact
        and rows
        and not decoded.get("missing_frames")
        and [row["frame_index"] for row in rows] == planned
        and all(
            row["index"] == i
            and 0 <= row["frame_index"] < len(expected["picture_pts"])
            and row["pts"] == expected["picture_pts"][row["frame_index"]]
            and row["sha256"] == expected["rgb_sha256"][row["frame_index"]]
            for i, row in enumerate(rows)
        )
    )
    audio = decoded.get("audio") or {}
    mapping = audio.get("source_mapping") or {}
    samples = expected["audio_samples"]
    pcm_exact = bool(
        mapping
        and audio["samples"] == mapping["samples"] == samples
        and mapping["source_sha256"] == digest
        and mapping["pcm_sha256"] == expected["pcm_sha256"]
        and mapping["timeline_time_base"] == "1/16000"
        and mapping["timeline"] == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}]
    )
    return {
        "clock_exact": clock_exact,
        "rgb_exact": rgb_exact,
        "pcm_exact": pcm_exact,
        "sampled_frames": len(rows),
        "faces": sum(bool(faces) for faces in (outputs.get("faces") or {}).get("faces", [])),
    }


def frame_series_exact(units, points, outputs):
    from app.core.canonical import canonical_json

    decoded = (outputs.get("decode") or {}).get("frames", [])
    faces = (outputs.get("faces") or {}).get("faces", [])
    if not points or len(decoded) != len(faces):
        return False
    expected = [
        (f"frame:{row['index']}", row["time_s"]) for row, boxes in zip(decoded, faces, strict=True) if boxes
    ]
    return canonical_json(
        [(unit["unit_ref"], unit["quality"]["time_s"]) for unit in units]
    ) == canonical_json(expected) and canonical_json(points) == canonical_json(
        [(time, unit["raw"]) for (_, time), unit in zip(expected, units, strict=True)]
    )


def record_case(file, expected, receipt, registry, manager, settings):
    from sqlalchemy import select

    from app.core.canonical import canonical_json
    from app.engine.boundary import Deadline
    from app.engine.pipelines import run_analysis
    from app.evidence.service import asset_refs, load_packet_set
    from app.storage.db import Analysis, Artefact, DetectorRun, get_session, shutdown
    from app.storage.files import stream_upload, validate_kind
    from app.storage.objects import default_store

    analysis_id = str(uuid.uuid4())
    digest = receipt["file_sha256"]
    try:
        with file.open("rb") as payload:
            _, head, actual_digest, size = stream_upload(
                analysis_id, FILENAME, payload, settings.max_video_bytes
            )
        mime, suffix = validate_kind("video", head, FILENAME, size)
        require(
            actual_digest == digest and size == receipt["bytes"] and (mime, suffix) == ("video/mp4", ".mp4"),
            "The genuine MP4 must enter serving byte-exact",
        )
        with get_session() as db:
            db.add(
                Analysis(
                    id=analysis_id,
                    kind="video",
                    filename=FILENAME,
                    content_type=mime,
                    size_bytes=size,
                    sha256=digest,
                    status="queued",
                )
            )
            db.commit()
        store = default_store()
        outcome = run_analysis(
            analysis_id,
            deadline=Deadline(settings.job_timeout_s),
            profile="full",
            registry=registry,
            models=manager,
            sessions=get_session,
            store=store,
        )
        with get_session() as db:
            analysis = db.get(Analysis, analysis_id)
            require(
                analysis.sha256 == digest and analysis.profile == "full",
                "Expected the actual owned full analysis",
            )
            rows = db.scalars(select(DetectorRun).where(DetectorRun.analysis_id == analysis_id)).all()
            result, evidence = None, None
            if analysis.result_hash:
                published = outcome.outputs.get("publish") or {}
                artefact = (
                    db.get(Artefact, published.get("result_artefact_id"))
                    if published.get("result_artefact_id")
                    else None
                )
                require(
                    artefact is not None
                    and artefact.sha256 == analysis.result_hash == published["result_hash"],
                    "Actual committed result must bind the stored object",
                )
                store.verify(artefact.sha256)
                result = json.loads(store.read_bytes(artefact.sha256))
                require(
                    result["input"]["sha256"] == digest
                    and result["manifest"]["profile"] == "full"
                    and result["manifest"]["models_manifest_sha256"] == sha(settings.models_manifest),
                    "Published input, profile and model-manifest commitments differ",
                )
                evidence = load_packet_set(db, analysis, store)
                for asset in asset_refs(evidence):
                    store.verify(asset)
            target = [row for row in rows if row.detector_id == DETECTOR]
            entries = [
                row
                for row in (outcome.outputs.get("detect") or {}).get("detectors", [])
                if row["detector_id"] == DETECTOR
            ]
            require(len(target) <= 1 and len(entries) <= 1, "One target run per full analysis required")
            target = target[0] if target else None
            entry = entries[0] if entries else None
            linked = bool(
                target
                and entry
                and target.id == entry["run_id"]
                and target.status == entry["status"]
                and target.detector_version == entry["version"]
            )
            raw = (
                db.get(Artefact, target.raw_output_artefact_id)
                if target and target.raw_output_artefact_id
                else None
            )
            if raw:
                store.verify(raw.sha256)
                recorded = json.loads(store.read_bytes(raw.sha256))
                require(
                    raw.role == "raw_output"
                    and raw.analysis_id == analysis_id
                    and entry is not None
                    and all(
                        canonical_json(recorded[key]) == canonical_json(entry[key])
                        for key in ("detector_id", "version", "status", "unit_scores", "aggregate")
                    ),
                    "Stored detector output differs from the actual stage entry",
                )
            signals = [row for row in (result or {}).get("signals", []) if row.get("detector_id") == DETECTOR]
            exhibits = {row["id"]: row for row in (result or {}).get("exhibits", [])}
            target_series = [
                exhibits[key]
                for key in (entry or {}).get("exhibit_ids", [])
                if key in exhibits and exhibits[key].get("kind") == "series"
            ]
            points = [
                point
                for exhibit in target_series
                for series in exhibit["series"]
                for point in series["points"]
            ]
            units = (entry or {}).get("unit_scores", [])
            series_ids = {row["id"] for row in target_series}
            return {
                "pipeline_status": outcome.status,
                "analysis_status": analysis.status,
                "stages": outcome.stages,
                "input": verify_input(outcome.outputs, expected, digest),
                "target_status": target.status if target else None,
                "target_reason": target.reason if target else None,
                "target_version": target.detector_version if target else None,
                "target_device": target.device if target else None,
                "target_dtype": target.dtype if target else None,
                "target_revisions": target.model_revisions if target else {},
                "target_rows_bound": linked,
                "target_raw_verified": raw is not None,
                "target_scored_frames": len(units),
                "target_signal_published": any(
                    row["status"] == "ok" and series_ids.intersection(row["exhibit_ids"]) for row in signals
                ),
                "target_series_published": frame_series_exact(units, points, outcome.outputs),
                "result_hash": analysis.result_hash,
                "packet_set_ready": bool(evidence and evidence.get("status") == "ready"),
                "packet_count": len(evidence.get("packets", [])) if evidence else 0,
                "run_status_counts": dict(Counter(row.status for row in rows)),
                "non_target_runs": sorted(
                    (row.detector_id, row.status) for row in rows if row.detector_id != DETECTOR
                ),
                "fuser_sha256": analysis.fuser_sha256,
                "fusion_mode": analysis.fusion_mode,
                "condition_profile": (result or {}).get("condition_profile"),
                "customer_assessment": baseline.customer_readout(analysis, result),
            }
    finally:
        shutdown()


def worker(arm, output):
    frozen, file, receipt = verify()
    stage = TRIAL / arm / "authored"
    require(
        output == stage / "output"
        and (stage / "resources.jsonl").is_file()
        and (stage / "worker.log").is_file()
        and not (stage / "profile.json").exists(),
        "Owned guard required",
    )
    folder = WORK / arm
    scoped(serving.apfs.work_volume, {"WORK": WORK, "TRIAL": TRIAL})()
    folder.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.mkdir(exist_ok=False)
    (folder / "tmp").mkdir()
    manifest = folder / "models.yaml"
    manifest.write_bytes((TRIAL / f"{arm}.yaml").read_bytes())
    old = read(baseline.PUBLIC)
    baseline.isolate_filesystem(folder, output, baseline.checked_programs(old))

    def environment(device, models, hf, data, work):
        env = serving.controlled_environment(device, models, hf, data, work)
        env["VERITAS_MODELS_MANIFEST"] = str(manifest)
        return env

    context = scoped(serving.initialize, {"controlled_environment": environment})(
        folder, old["configured_device"]
    )
    inventory = read(INVENTORY)
    baseline.install_assets(inventory, folder)
    registry, manager, actual = baseline.runtime_identity(context.confirm(), old, folder)
    version, _, batch = ARMS[arm]
    want = inventory["registry"]
    for row in want["detectors"]:
        if row["id"] == DETECTOR:
            row["version"] = version
    require(
        baseline.registry_snapshot(registry) == want
        and actual["resolved_device"] == frozen["device"]
        and manager.budget_bytes == frozen["model_manager_budget_bytes"]
        and all(actual["installed_required_models"][model] for model in MODELS),
        "Exact full registry and device required",
    )
    require(
        all(
            installed
            is all(
                asset["status"] == "available"
                for asset in inventory["model_files"]
                if asset["model"] == model
            )
            for model, installed in actual["installed_required_models"].items()
        ),
        "Model availability changed",
    )
    from app.engine.boundary import planned_allowance

    allowance, plan = planned_allowance(registry.instance(DETECTOR), None, manager)
    require(
        {use.model_id: use.max_items for use in plan} == dict.fromkeys(MODELS, batch), "Wrong activation plan"
    )
    require(allowance == len(MODELS) * batch * 400 * 1024**2, "Exact declared activation allowance required")
    write(folder / "runtime.json", actual)
    result = record_case(file, read(TRIAL / "inputs.json"), receipt, registry, manager, context.confirm())
    write(folder / "case.json", result)
    write(
        output / "result.json",
        {
            "status": "complete",
            "arm": arm,
            "freeze_sha256": sha(TRIAL / "freeze.json"),
            "manifest_sha256": sha(manifest),
            "planned_activation_bytes": allowance,
            "private_case_sha256": sha(folder / "case.json"),
            "private_runtime_sha256": sha(folder / "runtime.json"),
            "new_neural_calls": None,
        },
    )


def run(arm):
    folder = TRIAL / arm
    verify()
    if arm == "batch-one":
        require((TRIAL / "reference/authored/profile.json").is_file(), "Reference guard must finish first")
    folder.mkdir(exist_ok=True)

    def popen(argv, **kwargs):
        output = folder / "authored/output"
        require(
            argv[3:] == ["scripts.video_appearance_native_v108", "authored", str(output)],
            "Exact worker required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker", arm, str(output)], **kwargs)

    scoped(
        runtime.run,
        {
            "TRIAL": folder,
            "verify": verify,
            "prerequisites": lambda *_: True,
            "subprocess": types.SimpleNamespace(
                Popen=popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
            "__file__": str(Path(__file__)),
        },
    )("authored")


def compare(reference, candidate):
    return {
        "both_full_analyses_published": all(
            row["pipeline_status"] == row["analysis_status"] == "done"
            and row["result_hash"]
            and row["packet_set_ready"]
            for row in (reference, candidate)
        ),
        "both_exact_input_bindings": all(
            all(row["input"][key] for key in ("clock_exact", "rgb_exact", "pcm_exact"))
            for row in (reference, candidate)
        ),
        "reference_budget_skip": reference["target_status"] == "skipped"
        and reference["target_reason"] == "memory budget",
        "candidate_scored_and_published": bool(
            candidate["target_status"] == "ok"
            and candidate["target_scored_frames"]
            and candidate["target_rows_bound"]
            and candidate["target_raw_verified"]
            and candidate["target_device"] == "mps"
            and candidate["target_dtype"] == "float32"
            and set(candidate["target_revisions"]) == set(MODELS)
            and candidate["target_signal_published"]
            and candidate["target_series_published"]
        ),
        "same_non_target_route_statuses": reference["non_target_runs"] == candidate["non_target_runs"],
        "same_fuser": (reference["fusion_mode"], reference["fuser_sha256"])
        == (candidate["fusion_mode"], candidate["fuser_sha256"]),
    }


def readout():
    frozen, _, _ = verify()
    cases, summary = {}, {}
    for arm in ARMS:
        stage, folder = TRIAL / arm / "authored", WORK / arm
        result = baseline.speech.qualification_profile(stage, None, {}, Path(__file__).name)
        require(
            result["arm"] == arm
            and result["freeze_sha256"] == sha(TRIAL / "freeze.json")
            and result["manifest_sha256"] == frozen["arm_manifest_sha256"][arm]
            and result["private_case_sha256"] == sha(folder / "case.json")
            and result["private_runtime_sha256"] == sha(folder / "runtime.json"),
            "Exact monitored arm receipts required",
        )
        cases[arm] = case = read(folder / "case.json")
        profile = read(stage / "profile.json")
        summary[arm] = {
            **{
                key: case[key]
                for key in (
                    "pipeline_status",
                    "analysis_status",
                    "input",
                    "target_status",
                    "target_scored_frames",
                    "target_signal_published",
                    "target_series_published",
                    "packet_count",
                    "packet_set_ready",
                    "run_status_counts",
                )
            },
            "target_budget_skip": case["target_reason"] == "memory budget",
            "customer_direction": case["customer_assessment"]["direction"],
            "planned_activation_bytes": result["planned_activation_bytes"],
            "sampled_tree_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
            "worker_duration_seconds": profile["duration_s"],
            "private_arm_receipt_sha256": sha(stage / "output/result.json"),
            "private_resource_profile_sha256": sha(stage / "profile.json"),
        }
    comparison = compare(cases["reference"], cases["batch-one"])
    return {
        "schema": "chipmunk.video-frame-full-app-readout/168",
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "arms": summary,
        "comparison": comparison,
        "engineering_passed": all(comparison.values()),
        "scope": frozen["scope"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("freeze")
    actions.add_parser("readout")
    actions.add_parser("run").add_argument("arm", choices=ARMS)
    child = actions.add_parser("_worker")
    child.add_argument("arm", choices=ARMS)
    child.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run(args.arm)
    elif args.action == "readout":
        print(json.dumps(readout(), allow_nan=False))
    else:
        worker(args.arm, args.output)


if __name__ == "__main__":
    main()
