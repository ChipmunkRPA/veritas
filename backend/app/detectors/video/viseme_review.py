# SPDX-License-Identifier: MIT
"""Run the bounded source-linked timing trial alongside the regular analysis."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from app.core.config import settings
from app.evidence import video_lip_agreement as agreement
from app.evidence import video_silent_mouth as silent
from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_stereo as stereo

TIME_BUDGET = 45


def use_stereo():
    return (settings.environment == "dev" and settings.video_lip_timing_review and settings.video_lip_articulation_review
            and settings.video_lip_aac48_articulation_review and settings.video_lip_stereo_aac_review)


def supported(table, mapping, shots, stereo_scope=None):
    return (
        table is not None and mapping and shots == [[0, len(table)]]
        and not table.truncated and not table.interpolated and not table.rotation
        and 24 <= len(table) <= 120 and 320 <= table.width <= 1280 and 240 <= table.height <= 720
        and all(Fraction(1, 28) <= (b-a) * table.tb <= Fraction(1, 22)
                for a, b in zip(table.pts, table.pts[1:]))
        and mapping.get("source_codec") in {"aac", "pcm_s16le"}
        and mapping.get("source_sample_rate") in (16_000, 48_000)
        and binding.audio_origin(mapping, stereo_scope) is not None
    )


def sound_levels(pcm, presented):
    """Return the exact peak and sum of squared values of each full 5 ms block."""
    import numpy as np

    full = presented // silent.HOP * silent.HOP
    signed = np.rint(np.asarray(pcm[:full], np.float64) * 32768).clip(-32768, 32767).astype(np.int32)
    blocks = signed.reshape(-1, silent.HOP)
    return np.column_stack((np.abs(blocks).max(axis=1), (blocks.astype(np.int64)**2).sum(axis=1))).tolist()


def supported_shots(table, mapping, shots):
    if (table is None or not mapping or not shots or table.truncated or table.interpolated or table.rotation
            or not 320 <= table.width <= 1280 or not 240 <= table.height <= 720 or mapping.get("audio_stream_indices")):
        return False
    limit = agreement.MAX_MULTISHOT_SAMPLES
    origin = binding.audio_origin(mapping, max_samples=limit)
    if not agreement.silence_carrier(mapping, origin):
        return False
    try:
        agreement.shot_audio_ranges(table.pts, table.time_base, binding.presented_samples(mapping, max_samples=limit), shots)
    except (TypeError, ValueError):
        return False
    return any(agreement.eligible_shot(start, stop) for start, stop in shots)


def activity_eligible(table, mapping, shots, shot_frames):
    return (table is not None and mapping and shots == [[0, len(table)]] and shot_frames == len(table)
            and not (table.truncated or table.interpolated or table.rotation)
            and 320 <= table.width <= 1280 and 240 <= table.height <= 720
            and binding.activity_profile(table.pts, table.time_base, mapping) is not None)


def assess_activity(ctx, frame_store, mapping):
    table = frame_store.table
    geometry = settings.video_lip_timing_face_asset
    runtime = settings.video_lip_timing_runtime
    enhancer = settings.video_speech_activity_gap_enhancer
    vad = settings.weights_dir / "audio-upgrade" / "silero_vad.jit"
    if not geometry or not geometry.is_file() or not enhancer.is_dir() or not vad.is_file():
        return binding.skipped("assets")
    remaining = ctx.deadline.remaining()
    budget = TIME_BUDGET if remaining is None else min(TIME_BUDGET, remaining - 8.)
    if budget < 12.:
        return binding.skipped("runtime")
    request = {"activity": True, "runtime": str(runtime.resolve()) if runtime else None, "geometry": str(geometry.resolve()),
               "enhancer": str(enhancer.resolve()), "vad": str(vad.resolve()), "frames": table.to_json(),
               "video": str(frame_store.path.resolve()), "source_sha256": mapping["source_sha256"]}
    try:
        measured = binding.measured(_observe(request, budget), timestamps=table.pts, time_base=table.time_base,
                                    source={"width": table.width, "height": table.height}, mapping=mapping)
        _verify_anchors(frame_store, measured)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return binding.skipped("runtime")
    return measured


def _verify_anchors(frame_store, measured):
    indices = sorted({i for _, exhibit in binding.exhibits(measured) for i in exhibit["witness"]["picture_frames"].values()})
    if not indices:
        return
    frames = dict(frame_store.iter_many(indices, timeout_s=4.))
    if any(i not in frames or not frame_store.native_pts_verified(i)
           or hashlib.sha256(frames[i].tobytes()).hexdigest() != measured["inputs"]["frames"][i]["rgb_sha256"] for i in indices):
        raise ValueError("Timing frames could not be verified against the original")


def _observe(request, budget):
    env = {"PATH": os.environ.get("PATH", os.defpath), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HUB_DISABLE_TELEMETRY": "1", "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
    run = subprocess.run([sys.executable, "-B", "-m", "app.detectors.video.viseme_worker"],
                         input=json.dumps(request), env=env, text=True, capture_output=True, timeout=budget, check=True)
    return json.loads(run.stdout)


def assess(ctx, frame_store, mapping, audio_input, shots, *, stereo_scope=None, shot_frames=None):
    table = frame_store.table if frame_store is not None else None
    if audio_input is None:
        return binding.skipped("unsupported")
    stereo_requested = bool(mapping and use_stereo() and stereo.carrier(mapping))
    if not stereo_requested:
        stereo_scope = None
    elif stereo_scope is None or binding.audio_origin(mapping, stereo_scope) is None:
        return binding.skipped("stereo_unresolved")
    agreement_models = settings.video_lip_motion_sound_models
    shot_based_silence = bool(settings.video_lip_silent_mouth_review and stereo_scope is None and table is not None
                              and shot_frames == len(table) and supported_shots(table, mapping, shots))
    if not shot_based_silence and not supported(table, mapping, shots, stereo_scope):
        return binding.skipped("unsupported")
    from app.detectors.audio.phone_events import MODEL_BLOBS

    model = settings.video_lip_timing_model_dir
    geometry = settings.video_lip_timing_face_asset
    runtime = settings.video_lip_timing_runtime
    if (not geometry or not geometry.is_file() or (runtime and not runtime.is_dir())
            or (not shot_based_silence and (not model or not all((model / name).is_file() for name in (*MODEL_BLOBS, "pytorch_model.bin"))))):
        return binding.skipped("assets")
    remaining = ctx.deadline.remaining()
    budget = TIME_BUDGET if remaining is None else min(TIME_BUDGET, remaining - 8.)
    if budget < 12.:
        return binding.skipped("runtime")
    source_rate = mapping["source_sample_rate"]
    limit = agreement.MAX_MULTISHOT_SAMPLES if shot_based_silence else 64_000
    agreement_profile = agreement.carrier(
        mapping, origin=binding.audio_origin(mapping, stereo_scope, max_samples=limit), stereo_scope=stereo_scope,
    )
    apply_articulation = not shot_based_silence and settings.video_lip_articulation_review and (
        source_rate == 16_000 or source_rate == 48_000 and settings.video_lip_aac48_articulation_review
    )
    request = {"runtime": str(runtime.resolve()) if runtime else None, "geometry": str(geometry.resolve()),
               "checkpoint": str(model.resolve()) if model else None, "frames": table.to_json(), "video": str(frame_store.path.resolve()),
               "wav": str(ctx.store.path_for(audio_input["wav_sha256"]).resolve()), "samples": mapping["samples"],
               "presented_samples": binding.presented_samples(mapping, max_samples=limit),
               "ranked": settings.video_lip_timing_ranked and source_rate == 16_000,
               "articulation": apply_articulation,
               "shot_ranges": shots if shot_based_silence else None,
               "motion_models": str(agreement_models.resolve())
               if not shot_based_silence and agreement_profile and agreement_models and not mapping.get("audio_stream_indices") else None}
    try:
        raw = _observe(request, budget)
        if settings.video_lip_silent_mouth_review and stereo_scope is None:
            from app.detectors.audio.decode import read_wav

            pcm = read_wav(ctx.store.read_bytes(audio_input["wav_sha256"]))
            raw["sound_levels"] = sound_levels(pcm, request["presented_samples"])
            if shot_based_silence:
                raw[silent.SHOT_KEY] = shots
        measured = binding.measured(raw, timestamps=table.pts, time_base=table.time_base,
                                    source={"width": table.width, "height": table.height}, mapping=mapping, stereo_scope=stereo_scope)
        _verify_anchors(frame_store, measured)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        return binding.skipped("runtime")
    return measured


def audio_assets(pcm, mapping, record):
    from app.detectors.audio.decode import wav_bytes
    from app.evidence.audio_packets import _chart

    limit = binding.sample_limit(record)
    origin = (0 if record["schema"] in binding.ACTIVITY_SCHEMAS else
              binding.audio_origin(mapping, record.get("stereo_scope"), max_samples=limit))
    presented = binding.presented_samples(mapping, max_samples=limit)
    start, end = record["measurement"]["span"]
    if (origin is None or len(pcm) != mapping["samples"] or mapping["pcm_sha256"] != record["pcm_sha256"]
            or presented != record["inputs"]["samples"] or not 0 <= start < end <= presented
            or record["witness"]["audio_source_pts"] != [origin + start, origin + end]):
        raise ValueError("The timing excerpt differs from its frozen soundtrack")
    if silent.KEY in record["witness"] and sound_levels(pcm, presented) != record["inputs"]["sound_levels"]:
        raise ValueError("The near-silent interval differs from its frozen soundtrack levels")
    return {
        binding.CLIP: wav_bytes(pcm[start:end]),
        binding.WAVEFORM: _chart(pcm, [], start=max(0, start - 3200), end=min(presented, end + 3200),
                                 selected=(start, end), scores=False,
                                 title="Highlighted audio interval (source time, peak-scaled amplitude)", source_offset=origin,
                                 time_precision=3, peak_scaled=True),
    }
