"""Run the actual commercial full-profile video pipeline on the fixed pilot."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import types
import uuid
from collections import Counter
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from scripts import grid_native_clips_v149 as clips
from scripts import grid_source_speech_v147 as speech
from scripts import grid_transfer_plan_v143 as planning

ROOT, BASE, PANEL, ARCHIVE = planning.ROOT, planning.BASE, planning.PANEL, planning.ARCHIVE
TRIAL = BASE / "grid-serving-baseline-v152"
CACHE = ARCHIVE / "grid-serving-baseline-v152"
PUBLIC = PANEL / "grid-serving-baseline-freeze-v152.json"
MANIFEST = ROOT / "backend/app/models.yaml"
PROFILE = "full"
FILENAME = "sample.mkv"
DEVICES = ("auto", "cpu", "mps", "cuda")
STAGE_FAMILIES = {"provenance": "provenance", "tracks": "face-tracking", "faces": "face-detection"}
DETECTOR_STAGES = (*STAGE_FAMILIES, "detect", "second_pass")
SETTINGS = {
    "environment": "dev",
    "offline": "true",
    "run_worker": "false",
    "webhooks_enabled": "false",
    "billing_mode": "disabled",
    "licence_policy": "commercial",
    "pipeline": "v2",
    "detector_set": "manifest",
    "default_profile": PROFILE,
    "video_face_review_mode": "appearance",
    "audio_detector_mode": "candidate",
    "audio_enabled": "true",
}
COMPUTE_FIELDS = (
    "device",
    "model_memory_budget_mb",
    "job_timeout_s",
    "video_min_frames",
    "video_max_frames",
    "video_weight_overrides",
    "video_face_track_iou",
    "face_min_size",
    "face_upscaled_weight",
    "dfdc_min_face_px",
    "audio_weight",
    "audio_classifiers",
    "univfd_weight",
    "prior_ai",
    "fuser_versions",
    "h3_influence_bound",
)
RUNTIME_PACKAGES = (
    "numpy",
    "Pillow",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "huggingface-hub",
    "timm",
    "open-clip-torch",
    "scipy",
    "librosa",
    "soundfile",
    "filetype",
    "safetensors",
    "tokenizers",
    "pydantic",
    "pydantic-settings",
    "SQLAlchemy",
    "alembic",
    "c2pa-python",
    "PyYAML",
    "psutil",
)
OWN_CODE = (
    "backend/scripts/grid_serving_baseline_v152.py",
    "backend/tests/test_grid_serving_baseline_v152.py",
    "docs/grid-serving-baseline-v152-method.md",
    "backend/scripts/grid_native_clips_v149.py",
    "backend/scripts/grid_source_speech_v147.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/pyproject.toml",
)
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def code_identity():
    paths = {ROOT / path for path in OWN_CODE}
    paths.update(
        p
        for p in (ROOT / "backend/app").rglob("*")
        if p.is_file()
        and p.suffix
        in {
            ".py",
            ".json",
            ".yaml",
            ".yml",
            ".j2",
            ".css",
            ".npy",
            ".ttf",
            ".woff2",
        }
    )
    paths.update(p for p in (ROOT / "backend/alembic").rglob("*.py") if p.is_file())
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def package_versions():
    versions = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def controlled_environment(device, models, hf_home, data, work):
    """Return a clean process environment using only this protocol's fixed settings."""
    require(device in DEVICES, "Explicit reviewed compute device required")
    env = {
        key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE") if key in os.environ
    }
    env.update(
        {
            "PYTHONPATH": str(ROOT / "backend"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "XDG_CACHE_HOME": str(work / "cache"),
            "TORCH_HOME": str(work / "cache/torch"),
            "MPLCONFIGDIR": str(work / "cache/matplotlib"),
            "TMPDIR": str(work / "tmp"),
            "HF_HOME": str(hf_home),
            "HF_HUB_CACHE": str(hf_home / "hub"),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    env.update({f"VERITAS_{key.upper()}": value for key, value in SETTINGS.items()})
    env.update(
        VERITAS_DEVICE=device,
        VERITAS_MODELS_DIR=str(models),
        VERITAS_HF_HOME=str(hf_home),
        VERITAS_MODELS_MANIFEST=str(MANIFEST),
        VERITAS_DATA_DIR=str(data),
    )
    return env


def fresh_settings(env):
    from pydantic_settings import DotEnvSettingsSource

    require("app.core.config" not in sys.modules, "Fresh settings process required")
    os.environ.clear()
    os.environ.update(env)
    sys.dont_write_bytecode = True
    tempfile.tempdir = None
    with patch.object(DotEnvSettingsSource, "_read_env_files", return_value={}):
        from app.core.config import settings
    require(
        settings.offline is True
        and settings.run_worker is settings.webhooks_enabled is False
        and settings.billing_mode == "disabled"
        and settings.stripe_secret_key is settings.stripe_webhook_secret is None
        and settings.api_key is settings.database_url is settings.fuser_dir is None
        and settings.detector_set == "manifest"
        and settings.pipeline == "v2"
        and settings.licence_policy == "commercial"
        and settings.video_face_review_mode == "appearance"
        and settings.audio_detector_mode == "candidate"
        and settings.audio_enabled is True,
        "Reviewed offline app settings and no inherited billing, API or storage configuration required",
    )
    return settings


def selected_config(settings):
    return {key: getattr(settings, key) for key in (*SETTINGS, *COMPUTE_FIELDS)}


def registry_snapshot(registry):
    require(
        registry.policy.label == "commercial"
        and not registry.policy.allow_detectors
        and not registry.policy.deny_detectors,
        "Exact commercial policy without exceptions required",
    )
    enabled = {kind: registry.enabled(kind, PROFILE) for kind in ("video", "audio")}
    detectors = []
    for entry in registry.manifest.detectors:
        kind = entry.modality.value
        if kind not in enabled:
            continue
        decision = registry.decision(entry.id)
        detectors.append(
            {
                "id": entry.id,
                "version": entry.version,
                "modality": kind,
                "family": entry.family,
                "profiles": entry.profiles,
                "enabled_for_full": entry.id in enabled[kind],
                "registry_reasons": list(decision.reasons),
                "required_models": registry.required_models(entry),
            }
        )
    require(
        {"vid.tracks.yunet_dinov2", "vid.review.face_appearance", "vid.lipsync.landmark_xcorr"}
        <= set(enabled["video"])
        and {"aud.pella_v2", "aud.sound_events"} <= set(enabled["audio"]),
        "Expected current tracking, appearance, timing, voice and sound routes must be enabled",
    )
    return {"policy": registry.policy.model_dump(mode="json"), "detectors": detectors, "enabled": enabled}


def relative_asset(path, root):
    relative = path.relative_to(root)
    require(relative.parts and ".." not in relative.parts, "Owned model-file layout required")
    return relative.as_posix()


def required_model_ids(registry, snapshot):
    used = {
        model for row in snapshot["detectors"] if row["enabled_for_full"] for model in row["required_models"]
    }
    # Presence of a commercial superseder can affect registry enablement even when it is outside the full profile.
    for row in snapshot["detectors"]:
        entry = registry.manifest.detector(row["id"])
        if entry.superseded_by and registry.decision(entry.superseded_by).enabled:
            used.update(registry.required_models(registry.manifest.detector(entry.superseded_by)))
    return sorted(used)


def collect_assets(registry, snapshot, settings):
    from app.engine.fetchers import file_path

    assets = []
    for model_id in required_model_ids(registry, snapshot):
        model = registry.manifest.model(model_id)
        require(
            model.licence.class_ == "commercial", "Every mirrored detector asset must have commercial terms"
        )
        require(model.source.files, "Each required model must declare a file inventory")
        for weight in model.source.files:
            path = file_path(model, weight)
            if model.source.type == "hf":
                slot, root = "hf", settings.hf_home
            elif path.is_relative_to(settings.models_dir):
                slot, root = "models", settings.models_dir
            else:
                slot, root = "data", settings.data_dir
            present = path.is_file()
            digest, size = (sha(path), path.stat().st_size) if present else (None, None)
            require(
                not present or (digest == weight.sha256 and (weight.bytes is None or size == weight.bytes)),
                "Installed commercial model bytes differ from the pinned manifest",
            )
            assets.append(
                {
                    "model": model_id,
                    "licence": model.licence.spdx,
                    "revision": model.source.revision or model.source.ref,
                    "file": weight.path,
                    "slot": slot,
                    "relative": relative_asset(path, root),
                    "source_file": str(path),
                    "status": "available" if present else "missing",
                    "sha256": digest,
                    "bytes": size,
                }
            )
    return assets


def fusion_assets(data):
    root = data / "fusion"
    if not root.is_dir():
        return []
    return [
        {"relative": relative_asset(path, data), "source_file": str(path), "sha256": sha(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".npy", ".npz"}
    ]


def plan_identity():
    bindings = {}
    frozen = planning.frozen_context(clips.TRIAL, clips.PUBLIC, bindings)
    plan = planning.load_completed()
    ordinals = [row["ordinal"] for row in plan["sources"] if row["role"] == "pilot"]
    require(ordinals == frozen["pilot_clip_ordinals"], "Same fixed v149 pilot roster required")
    monitor = planning.reviewed.stage(
        BASE / "video-appearance-native-v108/monitor",
        None,
        "video_appearance_runtime_v108.py",
        None,
        bindings,
        monitor=True,
    )
    require(
        monitor["owned_child_exit_code"] == 0
        and monitor["owned_descendant_sampled"] is True
        and monitor["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py"),
        "Owned descendant monitor qualification required",
    )
    return bindings, ordinals, frozen["binaries_sha256"]


def freeze(device, models_dir, hf_home):
    if not clips.PUBLIC.is_file():
        print(json.dumps({"status": "pending_v149_context_no_freeze"}))
        return
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every freeze and partial attempt")
    bindings, ordinals, binaries = plan_identity()
    data = ROOT / "backend/data"
    settings = fresh_settings(
        controlled_environment(device, models_dir.resolve(), hf_home.resolve(), data, TRIAL)
    )

    from app.engine.registry import get_registry
    from app.policy.thresholds import Thresholds, digest

    registry = get_registry()
    selected = registry_snapshot(registry)
    assets = collect_assets(registry, selected, settings)
    inventory = {
        "schema": "chipmunk.grid-serving-assets/152",
        "registry": selected,
        "model_files": assets,
        "local_fuser_files": fusion_assets(data),
    }
    policy = Thresholds()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "private-inventory.json", inventory)
    body = {
        "schema": "chipmunk.grid-serving-baseline-freeze/152",
        "code_sha256": code_identity(),
        "source_metadata_sha256": bindings,
        "asset_inventory_sha256": sha(TRIAL / "private-inventory.json"),
        "manifest_sha256": sha(MANIFEST),
        "pilot_clip_ordinals": ordinals,
        "cases_per_clip": len(planning.CASES),
        "case_order": list(planning.CASES),
        "configured_device": device,
        "profile": PROFILE,
        "configuration": selected_config(settings),
        "initial_policy": {
            "version": 1,
            "thresholds": policy.model_dump(mode="json"),
            "sha256": digest(policy),
        },
        "binaries_sha256": binaries,
        "runtime_packages": package_versions(),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "commercial_model_files": dict(Counter(a["status"] for a in assets)),
        "enabled_detector_counts": {kind: len(ids) for kind, ids in selected["enabled"].items()},
        "serving_default_changed": False,
        "scope": "Explicit full-profile current video inference with isolated offline app storage and the current appearance and voice modes. No HTTP, paid job, new mouth candidate or ground-truth input.",
    }
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps({"status": "frozen", "pilot_clips": len(ordinals), "freeze_sha256": sha(PUBLIC)}),
        flush=True,
    )


def verify():
    frozen = read(PUBLIC)
    bindings, ordinals, binaries = plan_identity()
    require(
        frozen == read(TRIAL / "freeze.json")
        and frozen["schema"] == "chipmunk.grid-serving-baseline-freeze/152"
        and frozen["code_sha256"] == code_identity()
        and frozen["source_metadata_sha256"] == bindings
        and frozen["asset_inventory_sha256"] == sha(TRIAL / "private-inventory.json")
        and frozen["manifest_sha256"] == sha(MANIFEST)
        and frozen["pilot_clip_ordinals"] == ordinals
        and frozen["binaries_sha256"] == binaries
        and frozen["runtime_packages"] == package_versions()
        and frozen["platform"]
        == {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
        "Unchanged serving code, protocol, runtime and frozen model inventory required",
    )
    return frozen


def isolate_filesystem(folder, output, programs):
    roots = (folder.resolve(), output.resolve())

    def owned(path):
        return isinstance(path, (bytes, str, os.PathLike)) and (
            os.fsdecode(path) == os.devnull
            or any(Path(os.fsdecode(path)).resolve().is_relative_to(root) for root in roots)
        )

    def audit(event, args):
        if event == "open":
            path, _, flags = args
            writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
            if writes and not isinstance(path, int) and not owned(path):
                raise PermissionError("Baseline Python writes must stay in the owned stage")
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate", "os.utime"}:
            # dir_fd operations are used by shutil.rmtree on platforms with symlink-safe directory walking.
            path, *rest = args
            dir_fd = rest[-1] if rest and type(rest[-1]) is int and event != "os.truncate" else -1
            if dir_fd in (-1, None) and not isinstance(path, int) and not owned(path):
                raise PermissionError("Baseline Python writes must stay in the owned stage")
        if event in {"os.rename", "os.link"} and not all(owned(path) for path in args[:2]):
            raise PermissionError("Baseline Python writes must stay in the owned stage")
        if event == "os.symlink" and not owned(Path(os.fsdecode(args[1])).parent):
            raise PermissionError("Baseline asset links must stay in the owned stage")
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "socket.sendto", "os.system"}:
            raise PermissionError("Baseline application networking and service startup are disabled")
        if event == "subprocess.Popen":
            executable = os.fsdecode(args[0])
            found = shutil.which(executable)
            if found is None or Path(found).resolve() not in programs:
                raise PermissionError("Only the exact frozen local decoder binaries may be started")

    sys.addaudithook(audit)


def install_assets(inventory, folder):
    destinations = {}
    for asset in [*inventory["model_files"], *inventory["local_fuser_files"]]:
        target = Path(asset["source_file"])
        slot, relative = asset.get("slot", "data"), Path(asset["relative"])
        destination = folder / slot / relative
        require(
            slot in {"models", "data", "hf"}
            and not relative.is_absolute()
            and relative.parts
            and ".." not in relative.parts
            and destination.parent.resolve().is_relative_to(folder.resolve()),
            "Private asset layout escaped stage",
        )
        if asset.get("status") == "missing":
            require(not target.is_file(), "Frozen missing model inventory changed")
            continue
        target = target.resolve(strict=True)
        require(
            sha(target) == asset["sha256"]
            and ("bytes" not in asset or target.stat().st_size == asset["bytes"]),
            "Frozen local model or fuser asset bytes changed",
        )
        if destination in destinations:
            require(destinations[destination] == target, "Conflicting owned asset layout")
            continue
        require(
            not destination.exists() and not destination.is_symlink(), "Fresh isolated asset path required"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target)
        destinations[destination] = target


def planned_detector_stage(detector):
    if not detector["enabled_for_full"]:
        return None
    for stage, family in STAGE_FAMILIES.items():
        if detector["family"] == family:
            return stage if detector["modality"] == "video" else None
    return "detect"


def detector_inventory(registry, stages, outputs):
    observed = {}
    for stage in DETECTOR_STAGES:
        for run in (outputs.get(stage) or {}).get("detectors", []):
            observed.setdefault(run["detector_id"], []).append({"stage": stage, **run})
    require(
        observed.keys() <= {d["id"] for d in registry["detectors"]},
        "Unexpected detector in the actual production pipeline",
    )
    results = []
    for detector in registry["detectors"]:
        stage = planned_detector_stage(detector)
        runs = observed.get(detector["id"], [])
        require(not runs or all(run["stage"] == stage for run in runs), "Observed detector route differs")
        if runs:
            state, reason = "observed", None
        elif stage is None:
            state, reason = (
                "not_scheduled",
                "; ".join(detector["registry_reasons"]) or "Outside the full video route",
            )
        else:
            state, reason = (
                "unobserved",
                f"Expected in {stage}. Actual stage status: {stages.get(stage, 'unavailable')}.",
            )
        results.append(
            {
                "detector_id": detector["id"],
                "expected_stage": stage,
                "state": state,
                "reason": reason,
                "runs": runs,
            }
        )
    return results


def serving_input(case, outputs):
    probe, decode = outputs.get("probe") or {}, outputs.get("decode") or {}
    table = probe.get("frame_table")
    if not table:
        return {"status": "unavailable", "clock_exact": None, "sampled_rgb_exact": None}
    expected = case["expected"]
    times = [Fraction(p) * Fraction(table["time_base"]) for p in table["pts"]]
    clock_exact = times == [Fraction(t, 1000) for t in expected["picture_pts_ms"]] and not (
        table["truncated"] or table["interpolated"]
    )
    frames = decode.get("frames") or []
    rgb_exact = (
        all(
            0 <= f["frame_index"] < len(times)
            and f["sha256"] == expected["rgb_sha256"][f["frame_index"]]
            and Fraction(f["pts"]) * Fraction(decode["time_base"]) == times[f["frame_index"]]
            for f in frames
        )
        if frames and clock_exact
        else None
    )
    audio = decode.get("audio") or {}
    audio_input = audio.get("source_mapping") or {}
    return {
        "status": "observed" if frames else "unavailable",
        "clock_exact": clock_exact,
        "sampled_rgb_exact": rgb_exact,
        "probe_info": probe.get("info"),
        "probe_frame_table": table,
        "probe_plan": probe.get("plan"),
        "shot_plan": (outputs.get("shots") or {}).get("plan"),
        "shot_ranges": (outputs.get("shots") or {}).get("ranges"),
        "sampled_frames": frames,
        "missing_sampled_frames": decode.get("missing_frames"),
        "tracked_faces": (outputs.get("tracks") or {}).get("tracks"),
        "track_exhibits": [
            e for e in (outputs.get("tracks") or {}).get("exhibits", []) if e["kind"] == "face_track"
        ],
        "sampled_face_counts": [len(faces) for faces in (outputs.get("faces") or {}).get("faces", [])],
        "serving_audio": audio,
        "audio_samples_equal": (audio.get("samples") == expected["audio_samples"] if audio else None),
        "audio_pcm_equal": (
            audio_input["pcm_sha256"] == case["expected_serving_pcm_sha256"]
            if "pcm_sha256" in audio_input
            else None
        ),
        "audio_origin_equal": (
            Fraction(audio_input["timeline"][0]["source_pts"]) * Fraction(audio_input["timeline_time_base"])
            == Fraction(expected["audio_start_seconds"])
            if audio_input.get("timeline")
            else None
        ),
    }


def customer_readout(analysis, result):
    from app.engine.customer_assessment import _video_scope, customer_assessment

    public_input = {**(result or {}), "kind": "video", "status": analysis.status}
    projected = customer_assessment(public_input)
    _, content_assessed = _video_scope(public_input)
    direction = projected["direction"]
    require(
        direction in {"ai", "original", "inconclusive", "unavailable", "record"}
        and (direction not in {"ai", "original"} or (content_assessed and analysis.status == "done")),
        "Only a completed, assessed production conclusion can indicate a direction",
    )
    return {
        "direction": direction,
        "content_assessed": content_assessed,
        "directional_conclusion": direction in {"ai", "original"},
        **{
            key: projected.get(key)
            for key in (
                "rule",
                "headline",
                "confidence_pct",
                "probability_ai_pct",
                "interval_pct",
                "scope",
                "scope_label",
                "scope_note",
                "excluded_channels",
                "finding_assessments",
                "soundtrack",
            )
        },
    }


def analyze_case(case, settings, registry, manager, sessions, store):
    from sqlalchemy import select

    from app.engine.boundary import Deadline
    from app.engine.pipelines import VIDEO_STAGES, run_analysis
    from app.evidence.service import load_packet_set
    from app.storage.db import Analysis, AnalysisStage, Artefact, DetectorRun
    from app.storage.files import stream_upload, validate_kind

    case_file = clips.owned_file(case["file"], clips.CACHE)
    analysis_id = str(uuid.uuid4())
    with case_file.open("rb") as payload:
        _, head, digest, size = stream_upload(analysis_id, FILENAME, payload, settings.max_video_bytes)
    mime, suffix = validate_kind("video", head, FILENAME, size)
    require(
        digest == case["file_sha256"] and mime == "video/x-matroska" and suffix == ".mkv",
        "Matched lossless Matroska input must enter serving unchanged",
    )
    with sessions() as db:
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
    outcome = run_analysis(
        analysis_id,
        deadline=Deadline(settings.job_timeout_s),
        profile=PROFILE,
        registry=registry,
        models=manager,
        sessions=sessions,
        store=store,
    )
    with sessions() as db:
        analysis = db.get(Analysis, analysis_id)
        require(
            analysis.profile == PROFILE and analysis.sha256 == digest, "Actual full serving analysis required"
        )
        require(
            analysis.status != "done" or analysis.result_hash,
            "Completed analysis must have its committed result",
        )
        result = None
        evidence = None
        if analysis.result_hash:
            pub = outcome.outputs.get("publish") or {}
            artefact = (
                db.get(Artefact, pub.get("result_artefact_id")) if pub.get("result_artefact_id") else None
            )
            require(
                artefact is not None and artefact.sha256 == analysis.result_hash == pub.get("result_hash"),
                "Committed production result must bind its canonical stored object",
            )
            store.verify(artefact.sha256)
            stored = store.read_bytes(artefact.sha256)
            result = json.loads(stored)
            evidence = load_packet_set(db, analysis, store)
        stage_rows = {
            stage.stage: stage
            for stage in db.scalars(select(AnalysisStage).where(AnalysisStage.analysis_id == analysis_id))
        }
        stages = [
            {
                "stage": stage.id,
                "eligible": any(p.value == PROFILE for p in stage.profiles),
                "status": outcome.stages[stage.id],
                "persisted_status": stage_rows[stage.id].status if stage.id in stage_rows else None,
                "error": stage_rows[stage.id].error if stage.id in stage_rows else None,
            }
            for stage in VIDEO_STAGES
        ]
        persisted_runs = []
        for row in db.scalars(
            select(DetectorRun)
            .where(DetectorRun.analysis_id == analysis_id)
            .order_by(DetectorRun.started_at, DetectorRun.id)
        ):
            spec = registry.adapter_spec(row.detector_id)
            preferred = (
                "cpu"
                if spec is not None and tuple(spec.requirements.devices) == ("cpu",)
                else manager.default_device
            )
            persisted_runs.append(
                {
                    key: getattr(row, key)
                    for key in (
                        "id",
                        "detector_id",
                        "detector_version",
                        "status",
                        "reason",
                        "device",
                        "dtype",
                        "model_revisions",
                        "duration_ms",
                        "raw_output_artefact_id",
                    )
                }
                | {"preferred_device": preferred}
            )
        footprints = settings.data_dir / "state/model_footprints.json"
        return {
            "analysis_id": analysis_id,
            "filename": FILENAME,
            "input_file_sha256": digest,
            "analysis_status": analysis.status,
            "pipeline_outcome": outcome.status,
            "error": outcome.error,
            "stages": stages,
            "detector_runs": persisted_runs,
            "resident_models_after_case": [dataclasses.asdict(model) for model in manager.snapshot()],
            "model_device_pins_after_case": read(footprints)["device_pins"] if footprints.is_file() else {},
            "detector_stage_outputs": {
                key: outcome.outputs[key] for key in DETECTOR_STAGES if key in outcome.outputs
            },
            "sampled_input": serving_input(case, outcome.outputs),
            "temporal": outcome.outputs.get("temporal"),
            "condition": outcome.outputs.get("condition"),
            "source_gate": outcome.outputs.get("provenance_gate"),
            "result_hash": analysis.result_hash,
            "decision": result.get("decision") if result is not None else None,
            "manifest": result.get("manifest") if result is not None else None,
            "fusion_mode": analysis.fusion_mode,
            "fuser_sha256": analysis.fuser_sha256,
            "policy_thresholds_id": analysis.policy_thresholds_id,
            "effective_profile": analysis.effective_profile,
            "customer_assessment": customer_readout(analysis, result),
            "evidence": evidence,
        }


def runtime_identity(settings, frozen, folder):
    import cv2
    import torch

    from app.core.device import DEVICE
    from app.engine.fetchers import installed
    from app.engine.models import get_model_manager
    from app.engine.pipelines import default_profile
    from app.engine.profiles import resolve_profile
    from app.engine.registry import get_registry
    from app.policy.thresholds import resolve
    from app.storage.db import get_engine, get_session, init_db

    require(
        settings.data_dir == folder / "data"
        and settings.models_dir == folder / "models"
        and settings.hf_home == folder / "hf"
        and settings.objects_dir == folder / "data/objects"
        and settings.sqlalchemy_url == f"sqlite:///{folder / 'data/veritas.db'}"
        and selected_config(settings) == frozen["configuration"],
        "Exclusive app storage and the exact nonsecret configuration required",
    )
    registry = get_registry()
    manager = get_model_manager()
    cpu_quick = resolve_profile("auto", "cpu").value
    require(
        manager.default_device == str(DEVICE).split(":")[0]
        and default_profile().value == PROFILE
        and (settings.device == "auto" or DEVICE.type == settings.device)
        and (DEVICE.type != "mps" or torch.backends.mps.is_available())
        and (DEVICE.type != "cuda" or torch.cuda.is_available()),
        "Exact requested application device and explicit full profile required",
    )
    init_db()
    engine = get_engine()
    require(
        engine.url.get_backend_name() == "sqlite" and Path(engine.url.database) == folder / "data/veritas.db",
        "Only the newly owned private SQLite database may be used",
    )
    policy = resolve(sessions=get_session)
    recorded = policy.as_json()
    require(
        policy.version == frozen["initial_policy"]["version"]
        and policy.sha256 == frozen["initial_policy"]["sha256"]
        and policy.thresholds.model_dump(mode="json") == frozen["initial_policy"]["thresholds"],
        "Fresh isolated database must use the fixed initial app policy",
    )
    return (
        registry,
        manager,
        {
            "configured_device": settings.device,
            "resolved_device": str(DEVICE),
            "model_manager_device": manager.default_device,
            "model_manager_budget_bytes": manager.budget_bytes,
            "profile": PROFILE,
            "device_auto_profile": resolve_profile("auto", str(DEVICE)).value,
            "cpu_auto_profile": cpu_quick,
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "opencv_version": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "policy": recorded,
            "installed_required_models": {
                model: installed(registry.manifest.model(model))
                for model in sorted(
                    {
                        model
                        for detector in registry_snapshot(registry)["detectors"]
                        if detector["enabled_for_full"]
                        for model in detector["required_models"]
                    }
                )
            },
        },
    )


def case_binding(ordinal):
    cases = clips.load_completed(ordinal)
    stage = clips.TRIAL / f"clip-{ordinal:02d}/bind-00"
    folder = clips.CACHE / f"clip-{ordinal:02d}"
    source_bind = folder / "input-binding.json"
    body = source_bind.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    require(
        read(stage / "output/result.json")["private_input_binding_sha256"] == digest,
        "Actual matched v149 input binding changed",
    )
    bound = json.loads(body)
    return bind_expected_pcm(cases, bound["waveforms"]), {
        "result_sha256": sha(stage / "output/result.json"),
        "profile_sha256": sha(stage / "profile.json"),
        "resources_sha256": sha(stage / "resources.jsonl"),
        "case_inventory_sha256": sha(folder / "cases.jsonl"),
        "case_input_binding_sha256": digest,
    }


def bind_expected_pcm(cases, waveforms):
    bound = []
    for case in cases:
        if case["status"] == "ready":
            waveform = waveforms[case["waveform_id"]]
            digest = waveform["waveform"]["pcm_sha256"]
            expected = case["expected"]
            require(
                waveform["status"] == "encoded"
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and waveform["waveform"]["samples"] == expected["audio_samples"]
                and case["audio_output_clock_mapping"].get("pcm_sha256", digest) == digest,
                "Actual complete whole-waveform PCM commitment required",
            )
            case = {**case, "expected_serving_pcm_sha256": digest}
        bound.append(case)
    return bound


def count_results(rows):
    runs = [run for row in rows for run in row.get("detector_runs", [])]
    observed = [
        run for row in rows for detector in row.get("detector_inventory", []) for run in detector["runs"]
    ]
    return {
        "planned_cases": len(rows),
        "input_unavailable_cases": sum(r["status"] == "unavailable_input" for r in rows),
        "attempted_cases": sum(r["status"] != "unavailable_input" for r in rows),
        "done_cases": sum(r["status"] == "done" for r in rows),
        "pipeline_noncomplete_cases": sum(r["status"] not in {"done", "unavailable_input"} for r in rows),
        "detector_run_status_counts": dict(Counter(r["status"] for r in runs)),
        "detector_stage_status_counts": dict(Counter(r["status"] for r in observed)),
        "public_direction_counts": dict(
            Counter(r["customer_assessment"]["direction"] for r in rows if "customer_assessment" in r)
        ),
    }


def checked_programs(frozen):
    result = set()
    for executable, digest in frozen["binaries_sha256"].items():
        program = shutil.which(executable)
        require(program is not None and sha(program) == digest, "Frozen decoder binaries changed")
        result.add(Path(program).resolve())
    return result


def assess(ordinal, output, frozen):
    cases, upstream = case_binding(ordinal)
    inventory = read(TRIAL / "private-inventory.json")
    folder = CACHE / f"clip-{ordinal:02d}"
    require(folder.resolve().is_relative_to(CACHE.resolve()), "Owned pilot clip storage required")
    folder.mkdir(parents=True, mode=0o700, exist_ok=False)
    write(folder / "upstream.json", upstream)
    write(folder / "planned-cases.json", planned_cases(cases))
    programs = checked_programs(frozen)
    isolate_filesystem(folder, output, programs)
    install_assets(inventory, folder)
    (folder / "tmp").mkdir()
    env = controlled_environment(
        frozen["configured_device"], folder / "models", folder / "hf", folder / "data", folder
    )
    settings = fresh_settings(env)
    registry, manager, runtime = runtime_identity(settings, frozen, folder)
    observed_registry = registry_snapshot(registry)
    require(
        observed_registry == inventory["registry"],
        "The isolated registry must match the frozen serving registry",
    )
    require(
        all(
            installed
            is all(
                asset["status"] == "available"
                for asset in inventory["model_files"]
                if asset["model"] == model
            )
            for model, installed in runtime["installed_required_models"].items()
        ),
        "Actual isolated model availability differs from its frozen inventory",
    )
    write(folder / "runtime.json", runtime)

    from app.storage.db import get_session, shutdown
    from app.storage.objects import default_store

    store = default_store()
    rows = []
    try:
        with (folder / "cases.jsonl").open("x") as journal:
            for case in sorted(cases, key=lambda row: row["case_number"]):
                item = {
                    key: case[key]
                    for key in ("asset_index", "case_number", "source_slot", "component_index", "condition")
                }
                if case["status"] == "unavailable":
                    row = {**item, "status": "unavailable_input", "reason": case["reason"]}
                else:
                    print(json.dumps({"case_number": case["case_number"], "status": "starting"}), flush=True)
                    recorded = analyze_case(case, settings, registry, manager, get_session, store)
                    stages = {stage["stage"]: stage["status"] for stage in recorded["stages"]}
                    row = {
                        **item,
                        "status": recorded["pipeline_outcome"],
                        "reason": recorded["error"],
                        **recorded,
                        "detector_inventory": detector_inventory(
                            observed_registry, stages, recorded["detector_stage_outputs"]
                        ),
                    }
                rows.append(row)
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
    finally:
        shutdown()
    result = {
        "schema": "chipmunk.grid-serving-baseline-result/152",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "clip_ordinal": ordinal,
        "profile": PROFILE,
        "configured_device": frozen["configured_device"],
        "resolved_device": runtime["resolved_device"],
        **count_results(rows),
        "private_cases_sha256": sha(folder / "cases.jsonl"),
        "private_planned_cases_sha256": sha(folder / "planned-cases.json"),
        "private_runtime_sha256": sha(folder / "runtime.json"),
        "private_upstream_sha256": sha(folder / "upstream.json"),
        # The production manager can run batched Torch or native OpenCV operations. Do not imply a counted forward total.
        "new_neural_calls": None,
        "defaults_changed": False,
        "candidate_added": False,
        "billing_or_service_started": False,
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def planned_cases(cases):
    return [
        {
            key: case[key]
            for key in (
                "asset_index",
                "case_number",
                "source_slot",
                "component_index",
                "condition",
                "status",
                "reason",
            )
        }
        for case in sorted(cases, key=lambda row: row["case_number"])
    ]


def validate_case_rows(rows, cases, registry):
    order = planned_cases(cases)
    require(len(rows) == len(order) == 11, "Every matched case must have an explicit outcome")
    case_by_number = {case["case_number"]: case for case in cases}
    for row, expected in zip(rows, order, strict=True):
        require(
            all(
                row[key] == expected[key]
                for key in ("asset_index", "case_number", "source_slot", "component_index", "condition")
            ),
            "Private serving case order or identity changed",
        )
        if expected["status"] == "unavailable":
            require(
                row["status"] == "unavailable_input"
                and row["reason"] == expected["reason"]
                and "customer_assessment" not in row,
                "Unavailable constructed input must remain without a verdict",
            )
            continue
        case = case_by_number[row["case_number"]]
        verdict = row["customer_assessment"]
        direction = verdict["direction"]
        require(
            row["status"] == row["pipeline_outcome"]
            and row["status"] in {"done", "failed", "cancelled", "lease_lost"}
            and (row["status"] != "done" or (row["analysis_status"] == "done" and row["result_hash"]))
            and row["filename"] == FILENAME
            and row["input_file_sha256"] == case["file_sha256"]
            and verdict["directional_conclusion"] == (direction in {"ai", "original"})
            and (
                direction not in {"ai", "original"}
                or (row["status"] == "done" and verdict["content_assessed"])
            ),
            "Matched input, actual status and assessed conclusion required",
        )
        stages = {stage["stage"]: stage["status"] for stage in row["stages"]}
        require(
            len(stages) == len(row["stages"])
            and row["detector_inventory"]
            == detector_inventory(registry, stages, row["detector_stage_outputs"]),
            "Serving detector routes and observations do not reconcile",
        )


def load_completed(ordinal):
    """Return the complete private serving observations, in blind numeric case order."""
    frozen = verify()
    require(type(ordinal) is int and ordinal in frozen["pilot_clip_ordinals"], "Fixed pilot ordinal required")
    cases, upstream = case_binding(ordinal)
    stage = TRIAL / f"clip-{ordinal:02d}/authored"
    result = speech.qualification_profile(stage, None, {}, "grid_serving_baseline_v152.py")
    folder = CACHE / f"clip-{ordinal:02d}"
    require(
        result["schema"] == "chipmunk.grid-serving-baseline-result/152"
        and result["clip_ordinal"] == ordinal
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["profile"] == PROFILE
        and result["configured_device"] == frozen["configured_device"]
        and result["defaults_changed"]
        is result["candidate_added"]
        is result["billing_or_service_started"]
        is False
        and result["private_cases_sha256"] == sha(folder / "cases.jsonl")
        and result["private_planned_cases_sha256"] == sha(folder / "planned-cases.json")
        and result["private_runtime_sha256"] == sha(folder / "runtime.json")
        and result["private_upstream_sha256"] == sha(folder / "upstream.json")
        and read(folder / "planned-cases.json") == planned_cases(cases)
        and read(folder / "upstream.json") == upstream
        and read(folder / "runtime.json")["resolved_device"] == result["resolved_device"],
        "Actual completed bounded serving receipt and private journals required",
    )
    rows = planning.reviewed.lines(folder / "cases.jsonl")
    registry = read(TRIAL / "private-inventory.json")["registry"]
    validate_case_rows(rows, cases, registry)
    require(
        all(result[key] == count for key, count in count_results(rows).items()),
        "Serving counts must reconcile with the complete private case roster",
    )
    return rows


def prerequisite(ordinal, ordinals):
    before = ordinals.index(ordinal) - 1
    prior = TRIAL / f"clip-{ordinals[before]:02d}/authored" if before >= 0 else None
    source = clips.TRIAL / f"clip-{ordinal:02d}/bind-00"
    stages = (source,) if prior is None else (prior, source)
    if not all(
        (stage / name).is_file() for stage in stages for name in ("profile.json", "output/result.json")
    ):
        return False
    if prior is not None:
        load_completed(ordinals[before])
    case_binding(ordinal)
    return True


def routed_popen(ordinal, argv, **kwargs):
    output = TRIAL / f"clip-{ordinal:02d}/authored/output"
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"]
        and argv[5:] == [str(output)],
        "Only the exclusive full-profile serving worker is permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_serving_baseline_v152", "assess", str(ordinal), str(output)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    frozen = verify()
    ordinals = frozen["pilot_clip_ordinals"]
    require(type(ordinal) is int and ordinal in ordinals, "One fixed pilot ordinal required")
    folder = TRIAL / f"clip-{ordinal:02d}"
    folder.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kw: routed_popen(ordinal, argv, **kw),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": folder,
            "verify": verify,
            "prerequisites": lambda _command, _batch: prerequisite(ordinal, ordinals),
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--device", choices=DEVICES, default="auto")
    prepare.add_argument("--models-dir", type=Path, default=ROOT / "models")
    prepare.add_argument("--hf-home", type=Path, default=ROOT / "models/hf")
    commands.add_parser("run").add_argument("ordinal", type=int)
    worker = commands.add_parser("assess")
    worker.add_argument("ordinal", type=int)
    worker.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.device, args.models_dir, args.hf_home)
    elif args.command == "run":
        run(args.ordinal)
    else:
        frozen = verify()
        stage = TRIAL / f"clip-{args.ordinal:02d}/authored"
        require(
            args.ordinal in frozen["pilot_clip_ordinals"]
            and args.output.resolve() == (stage / "output").resolve()
            and (stage / "resources.jsonl").is_file()
            and (stage / "worker.log").is_file()
            and not (stage / "profile.json").exists(),
            "Exact exclusive guarded worker output required",
        )
        args.output.mkdir(exist_ok=False)
        assess(args.ordinal, args.output, frozen)


if __name__ == "__main__":
    main()
