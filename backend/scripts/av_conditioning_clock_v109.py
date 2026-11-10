"""Source-clock speech-token routing; no pretrained network is executed.

Temporal indices and nominal convolution geometry are not local AI evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import wave
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
WAVEFORMS = BASE / "av-waveforms-v105/preparation"
PARENT = BASE / "av-sentence-support-v104/preparation"
ASSETS = BASE / "lipsync-v7/assets"
TRIAL = BASE / "av-conditioning-clock-v109"
PANEL = ROOT / "docs/evaluation-panels"
PUBLIC = PANEL / "av-conditioning-clock-freeze-v109.json"
CACHE = Path("/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/av-waveforms-v105")
CODE = (
    "backend/scripts/av_conditioning_clock_v109.py",
    "backend/tests/test_av_conditioning_clock_v109.py",
    "docs/av-conditioning-clock-v109-method.md",
    "backend/scripts/musetalk_benchmark_v7.py",
    "backend/scripts/fetch_mouth_generator_v7.py",
    "backend/scripts/prepare_av_waveforms_v105.py",
    "backend/scripts/av_sentence_support_v104.py",
    "backend/scripts/review_video_native_inserts_v96.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def exact(value):
    require(type(value) in (str, int, Fraction), "Exact rational time required, not float/bool")
    return Fraction(value)


def nominal_hull(tick):
    """Half-open virtual STFT/conv hull; boundary operations are separate."""
    require(type(tick) is int, "One integer encoder index required")
    return [320 * tick - 520, 320 * tick + 520]


def frame_plan(index, source_time, audio_origin, video_origin, samples, allowance):
    require(type(index) is int and index >= 0, "Actual source picture index required")
    require(type(samples) is int and 640 <= samples <= 480000, "Finite supported mono16k extent required")
    source, audio, video, uncertainty = map(exact, (source_time, audio_origin, video_origin, allowance))
    require(uncertainty >= 0, "Nonnegative source-clock uncertainty required")
    relative = source - audio
    point = relative * 16000
    supported = 0 <= math.floor(point) <= math.ceil(point) < samples
    anchor = math.floor(relative * 50)
    ticks = list(range(anchor - 4, anchor + 6))
    prefix = samples // 320
    keep = [0 <= t < prefix for t in ticks]
    hulls = [nominal_hull(t) if valid else None for t, valid in zip(ticks, keep, strict=True)]
    possible = [math.floor((relative - uncertainty) * 50), math.floor((relative + uncertainty) * 50)]
    return {
        "video_frame_index": index,
        "source_time_seconds": str(source),
        "source_audio_origin_seconds": str(audio),
        "source_video_origin_seconds": str(video),
        "observed_mono_samples": samples,
        "native_clock_allowance_seconds": str(uncertainty),
        "audio_relative_seconds": str(relative),
        "sample_position_fraction": str(point),
        "source_sample_status": "supported" if supported else "unavailable",
        "source_unavailable_reason": None
        if supported
        else "Picture/sample bracket outside the observed waveform; no originality verdict.",
        "encoder_anchor": anchor,
        "anchor_time_audio_seconds": str(Fraction(anchor, 50)),
        "anchor_residual_seconds": str(relative - Fraction(anchor, 50)),
        "possible_anchor_range": possible,
        "clock_anchor_ambiguous": possible[0] != possible[1],
        "tick_indices": ticks,
        "retained_tick_prefix": prefix,
        "retained_tick_slots": keep,
        "zero_slots_required": keep.count(False),
        "nominal_convolution_hulls_samples": hulls,
        "nominal_convolution_windows_inside_source": all(keep)
        and all(0 <= h[0] < h[1] <= samples for h in hulls),
        "anchor_difference_from_video_zero": anchor - math.floor((source - video) * 50),
        "anchor_difference_from_25hz_index": anchor - index * 2,
        "hidden_state_dependency": "Possible whole padded-input context from spectral normalization and unmasked encoder attention; not local evidence.",
        "generator_conditioning_qualified": False,
    }


def select_chunk(hidden, plan):
    """Pure array routing, tick-major then layer-major; authored fixtures only here."""
    import numpy as np

    require(
        plan["source_sample_status"] == "supported", "Unavailable source cannot supply invented conditioning"
    )
    require(
        plan["tick_indices"] == list(range(plan["encoder_anchor"] - 4, plan["encoder_anchor"] + 6))
        and plan["retained_tick_slots"]
        == [0 <= t < plan["retained_tick_prefix"] for t in plan["tick_indices"]],
        "Unchanged complete ten-slot clock plan required",
    )
    require(
        isinstance(hidden, np.ndarray)
        and hidden.ndim == 3
        and hidden.shape[1:] == (5, 384)
        and plan["retained_tick_prefix"] <= len(hidden) <= 1500
        and hidden.dtype == np.float32
        and np.isfinite(hidden).all(),
        "Complete finite five-state speech array required",
    )
    output = np.zeros((10, 5, 384), np.float32)
    for slot, (tick, keep) in enumerate(zip(plan["tick_indices"], plan["retained_tick_slots"], strict=True)):
        if keep:
            output[slot] = hidden[tick]
    return output.reshape(50, 384)


def defaults(function):
    return {
        a.arg: ast.literal_eval(v)
        for a, v in zip(
            function.args.args[-len(function.args.defaults) :], function.args.defaults, strict=True
        )
    }


def graph_sources():
    transformer = Path(importlib.metadata.distribution("transformers").locate_file("transformers"))
    torch = Path(importlib.metadata.distribution("torch").locate_file("torch"))
    paths = {
        "transformers/models/whisper/feature_extraction_whisper.py": transformer
        / "models/whisper/feature_extraction_whisper.py",
        "transformers/models/whisper/modeling_whisper.py": transformer / "models/whisper/modeling_whisper.py",
        "torch/functional.py": torch / "functional.py",
    }
    feature = ast.parse(paths["transformers/models/whisper/feature_extraction_whisper.py"].read_text())
    klass = next(
        n for n in feature.body if isinstance(n, ast.ClassDef) and n.name == "WhisperFeatureExtractor"
    )
    init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    observed = defaults(init)
    expected = {
        "feature_size": 80,
        "sampling_rate": 16000,
        "hop_length": 160,
        "chunk_length": 30,
        "n_fft": 400,
        "padding_value": 0.0,
        "dither": 0.0,
        "return_attention_mask": False,
    }
    require(all(observed[k] == v for k, v in expected.items()), "Reviewed feature defaults changed")
    functional = ast.parse(paths["torch/functional.py"].read_text())
    stft = next(n for n in functional.body if isinstance(n, ast.FunctionDef) and n.name == "stft")
    params = defaults(stft)
    require(
        params["center"] is True and params["pad_mode"] == "reflect",
        "Reviewed STFT boundary defaults changed",
    )
    model = ast.parse(paths["transformers/models/whisper/modeling_whisper.py"].read_text())
    encoder = next(n for n in model.body if isinstance(n, ast.ClassDef) and n.name == "WhisperEncoder")
    init = next(n for n in encoder.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    convs = {}
    for n in ast.walk(init):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            for target in n.targets:
                if isinstance(target, ast.Attribute) and target.attr in {"conv1", "conv2"}:
                    convs[target.attr] = {k.arg: ast.literal_eval(k.value) for k in n.value.keywords}
    require(
        convs
        == {
            "conv1": {"kernel_size": 3, "padding": 1},
            "conv2": {"kernel_size": 3, "stride": 2, "padding": 1},
        },
        "Reviewed encoder convolution geometry changed",
    )
    forward = next(n for n in encoder.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    require(
        not any(
            isinstance(n, ast.Name) and n.id == "attention_mask" and isinstance(n.ctx, ast.Load)
            for n in ast.walk(forward)
        ),
        "Reviewed unused encoder mask changed",
    )
    return {
        "sha256": {k: sha(v) for k, v in paths.items()},
        "feature_defaults": expected,
        "stft_center": True,
        "stft_pad_mode": "reflect",
        "convolutions": convs,
        "encoder_attention_mask_used": False,
        "spectral_global_floor_present": True,
        "scope": "Static installed-source inspection, not measured model dependency or numerical output parity.",
    }


def identity():
    from scripts.fetch_mouth_generator_v7 import verify

    manifest = read(PANEL / "mouth-v7-generator-assets.json")
    configs = [
        r
        for r in manifest["assets"]
        if r["path"] in {"speech/config.json", "speech/preprocessor_config.json"}
    ]
    require(
        len(configs) == 2 and all(r["license"] == "apache-2.0" for r in configs),
        "Two permitted publisher speech configs required",
    )
    verified = [verify(ASSETS / r["path"], r) for r in configs]
    pre, cfg = read(ASSETS / "speech/preprocessor_config.json"), read(ASSETS / "speech/config.json")
    require(
        all(
            pre[k] == v
            for k, v in {
                "feature_size": 80,
                "sampling_rate": 16000,
                "hop_length": 160,
                "chunk_length": 30,
                "n_fft": 400,
                "padding_value": 0.0,
                "return_attention_mask": False,
            }.items()
        ),
        "Reviewed preprocessing config changed",
    )
    require(
        all(
            cfg[k] == v
            for k, v in {
                "d_model": 384,
                "encoder_layers": 4,
                "num_mel_bins": 80,
                "max_source_positions": 1500,
            }.items()
        ),
        "Reviewed tiny encoder geometry changed",
    )
    return {
        "schema": "chipmunk.av-conditioning-clock-freeze/109",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in (
                WAVEFORMS / "waveforms.jsonl",
                WAVEFORMS / "result.json",
                WAVEFORMS / "profile.json",
                PARENT / "plan.json",
                PARENT / "result.json",
                PANEL / "mouth-v7-generator-assets.json",
            )
        },
        "speech_config_records": verified,
        "library_source": graph_sources(),
        "packages": {p: importlib.metadata.version(p) for p in ("transformers", "torch", "numpy", "psutil")},
        "sources": 48,
        "new_neural_calls": 0,
        "scope": "Exact native source-time routing and explicit padding/nominal geometry. No pretrained weight load, feature inference, achieved lip synchronization or detector accuracy.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "sources": 48, "new_neural_calls": 0}), flush=True)


def prepare():
    from scripts.review_video_native_inserts_v96 import Budget

    lock = read(TRIAL / "freeze.json")
    require(lock == identity() == read(PUBLIC), "Published unchanged clock-method freeze required")
    output = TRIAL / "preparation"
    output.mkdir(exist_ok=False)
    budget = Budget(output)
    status, error = "failed", None
    try:
        budget.observe(initial=True)
        old, prior = read(WAVEFORMS / "result.json"), read(WAVEFORMS / "profile.json")
        require(
            old["status"] == prior["status"] == "complete"
            and prior["sample_errors"] == 0
            and old["private_journal_sha256"] == sha(WAVEFORMS / "waveforms.jsonl"),
            "Complete unchanged waveform qualification required",
        )
        parents = read(PARENT / "plan.json")["records"]
        require(
            sha(PARENT / "plan.json") == read(PARENT / "result.json")["private_plan_sha256"],
            "Unchanged sentence/role plan required",
        )
        parent = {r["asset_index"]: r for r in parents}
        sources = [json.loads(line) for line in (WAVEFORMS / "waveforms.jsonl").read_text().splitlines()]
        require(len(sources) == len(parent) == 48, "All 48 waveforms and role records required")
        records = []
        with (output / "clock-plan.jsonl").open("x") as journal:
            for row in sources:
                p = parent[row["asset_index"]]
                require(
                    row["source_group"] == p["id"]
                    and row["study_role"] == p["study_role"]
                    and row["source_sha256"] == p["input_sha256"]
                    and row["conditioning_waveform_ready"]
                    and row["generator_conditioning_qualified"] is False,
                    "Exact unchanged source/role waveform required",
                )
                wav = row["mono16k_wav"]
                require(Path(wav["file"]).name == wav["file"], "Safe waveform file required")
                path = CACHE / wav["file"]
                require(sha(path) == wav["sha256"], "Cached waveform bytes changed")
                with wave.open(str(path), "rb") as stream:
                    require(
                        (
                            stream.getframerate(),
                            stream.getnchannels(),
                            stream.getsampwidth(),
                            stream.getnframes(),
                        )
                        == (16000, 1, 2, wav["sample_frames"]),
                        "Exact mono16k source header required",
                    )
                frames = [
                    frame_plan(
                        c["video_frame_index"],
                        c["source_time_seconds"],
                        row["audio_origin_seconds"],
                        row["video_origin_seconds"],
                        wav["sample_frames"],
                        c["native_clock_allowance_seconds"],
                    )
                    for c in row["frame_sample_coordinates"]
                ]
                require(
                    all(
                        f["sample_position_fraction"] == c["sample_position_fraction"]
                        and f["source_sample_status"] == c["status"]
                        for f, c in zip(frames, row["frame_sample_coordinates"], strict=True)
                    ),
                    "Source sample correspondence changed",
                )
                edit = p["clock_support"]["native20_edit"]
                edit_frames = (
                    []
                    if edit is None
                    else [
                        f
                        for f in frames
                        if edit["frames_half_open"][0] <= f["video_frame_index"] < edit["frames_half_open"][1]
                    ]
                )
                require(
                    edit is None or len(edit_frames) == 20, "All native edit centers must remain represented"
                )
                result = {
                    "asset_index": row["asset_index"],
                    "source_group": row["source_group"],
                    "study_role": row["study_role"],
                    "mono_wav_sha256": wav["sha256"],
                    "frames": frames,
                    "native20_edit": edit,
                    "native20_edit_frames_with_full_tick_prefix": sum(
                        all(f["retained_tick_slots"]) for f in edit_frames
                    ),
                    "native20_edit_frames_with_inside_nominal_convolution": sum(
                        f["nominal_convolution_windows_inside_source"] for f in edit_frames
                    ),
                    "generator_conditioning_qualified": False,
                }
                records.append(result)
                journal.write(json.dumps(result, allow_nan=False) + "\n")
                journal.flush()
                budget.observe()
        require(
            not {"torch", "numpy", "transformers"}.intersection(sys.modules),
            "No array/neural library may be imported during source preparation",
        )
        frames = [f for r in records for f in r["frames"]]
        result = {
            "schema": "chipmunk.av-conditioning-clock-result/109",
            "status": "complete",
            "sources": len(records),
            "role_counts": dict(Counter(r["study_role"] for r in records)),
            "frame_coordinates": len(frames),
            "sample_supported_frames": sum(f["source_sample_status"] == "supported" for f in frames),
            "different_from_video_zero_anchor": sum(
                f["anchor_difference_from_video_zero"] != 0 for f in frames
            ),
            "different_from_25hz_index_anchor": sum(
                f["anchor_difference_from_25hz_index"] != 0 for f in frames
            ),
            "maximum_abs_25hz_anchor_error_ticks": max(
                abs(f["anchor_difference_from_25hz_index"]) for f in frames
            ),
            "clock_anchor_ambiguous_frames": sum(f["clock_anchor_ambiguous"] for f in frames),
            "frames_requiring_zero_tick_slots": sum(f["zero_slots_required"] > 0 for f in frames),
            "frames_with_full_nominal_convolution_support": sum(
                f["nominal_convolution_windows_inside_source"] for f in frames
            ),
            "native20_edit_sources": sum(r["native20_edit"] is not None for r in records),
            "native20_edit_frames_with_full_tick_prefix": sum(
                r["native20_edit_frames_with_full_tick_prefix"] for r in records
            ),
            "native20_edit_frames_with_inside_nominal_convolution": sum(
                r["native20_edit_frames_with_inside_nominal_convolution"] for r in records
            ),
            "generator_conditioning_qualified_sources": 0,
            "new_neural_calls": 0,
            "defaults_changed": False,
            "freeze_sha256": sha(TRIAL / "freeze.json"),
            "private_plan_sha256": sha(output / "clock-plan.jsonl"),
            "scope": lock["scope"]
            + " Token indices are temporal routing, not finite local support or proof of synthesis. Actual encoder/generator timing and output remain unqualified.",
        }
        write(output / "result.json", result)
        write(PANEL / "av-conditioning-clock-result-v109.json", result)
        status = "complete"
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        budget.stream.close()
        write(
            output / "profile.json",
            {
                "status": status,
                "error_type": error,
                "duration_s": time.monotonic() - budget.started,
                "maximum_observed_self_rss_bytes": max(
                    (r["combined_rss_bytes"] for r in budget.rows), default=0
                ),
                "minimum_available_ram_bytes": min(
                    (r["available_ram_bytes"] for r in budget.rows), default=None
                ),
                "resource_journal_sha256": sha(output / "resources.jsonl"),
                "new_neural_calls": 0,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "prepare"))
    freeze() if parser.parse_args().command == "freeze" else prepare()
