"""Assemble the fixed pilot's complete native picture and soundtrack cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import types
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import grid_transfer_plan_v143 as planning

ROOT, BASE, PANEL, ARCHIVE = planning.ROOT, planning.BASE, planning.PANEL, planning.ARCHIVE
TRIAL = BASE / "grid-native-clips-v149"
CACHE = ARCHIVE / "grid-native-clips-v149"
PUBLIC = PANEL / "grid-native-clips-freeze-v149.json"
FRAMES = tuple(range(*planning.EDIT))
BRANCHES = ("musetalk", "liveportrait")
MODES = ("source_time", "earlier_320ms")
CONTEXTS = (
    ("grid-portrait-geometry-v144", "grid-portrait-geometry-freeze-v144.json"),
    ("grid-portrait-frames-v146", "grid-portrait-frames-freeze-v146.json"),
    ("grid-source-speech-v147", "grid-source-speech-freeze-v147.json"),
    ("grid-musetalk-frames-v148", "grid-musetalk-frames-freeze-v148.json"),
)
CODE = (
    "backend/scripts/grid_native_clips_v149.py",
    "backend/tests/test_grid_native_clips_v149.py",
    "docs/grid-native-clips-v149-method.md",
    "backend/scripts/grid_transfer_plan_v143.py",
    "backend/scripts/review_grid_native_inputs_v142.py",
    "backend/scripts/native_av_mux_v119.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def fingerprint(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def pilot(ordinal):
    sources = planning.load_completed()["sources"]
    require(type(ordinal) is int and 0 <= ordinal < len(sources), "Original integer source ordinal required")
    source = sources[ordinal]
    require(source["ordinal"] == ordinal and source["role"] == "pilot", "Fixed pilot sources only")
    require([r["id"] for r in source["cases"]] == list(planning.CASES), "All eleven fixed cases required")
    return source


def mux_qualification(bindings):
    folder = BASE / "native-av-mux-v119"
    public = PANEL / "native-av-mux-freeze-v119.json"
    contract = planning.frozen_context(folder, public, bindings)
    stage = folder / "bind-00"
    result = planning.reviewed.stage(stage, sha(public), "native_av_mux_v119.py", 0, bindings)
    wrapper = planning.bind_metadata(PANEL / "native-av-mux-result-v119.json", bindings)
    cases_path = stage / "output/private-cases.json"
    cases = planning.bind_metadata(cases_path, bindings)
    require(
        result["schema"] == "chipmunk.native-av-mux-result/119"
        and result["cases"] == len(cases) == 3
        and result["authored_cases"] == 2
        and result["fitting_source_rewrites"] == 1
        and result["all_picture_clocks_rgb_pcm_audio_origins_exact"] is True
        and result["private_case_results_sha256"] == sha(cases_path)
        and wrapper["measurement"] == result
        and wrapper["profile"] == read(stage / "profile.json")
        and wrapper["result_sha256"] == sha(stage / "output/result.json")
        and wrapper["profile_sha256"] == sha(stage / "profile.json"),
        "Actual published v119 lossless codec qualification required",
    )
    return contract


def identity():
    plan = planning.load_completed()
    bindings = {}
    for path in (
        planning.PUBLIC,
        planning.TRIAL / "bind-00/output/result.json",
        planning.TRIAL / "bind-00/profile.json",
        planning.TRIAL / "bind-00/output/private-plan.json",
    ):
        planning.bind_metadata(path, bindings)
    source = planning.bind_metadata(PANEL / "grid-native-inputs-freeze-v137.json", bindings)
    packages = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "Pillow", "opencv-python-headless", "psutil")
    }
    for name, filename in CONTEXTS:
        frozen = planning.frozen_context(BASE / name, PANEL / filename, bindings)
        for package, version in frozen.get("packages", {}).items():
            if package in packages:
                require(packages[package] == version, "Qualified producer array/image runtime changed")
    mux_qualification(bindings)
    return {
        "schema": "chipmunk.grid-native-clips-freeze/149",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "completed_metadata_sha256": bindings,
        "pilot_clip_ordinals": [row["ordinal"] for row in plan["sources"] if row["role"] == "pilot"],
        "planned_cases_per_clip": 11,
        "source_pictures_per_complete_case": 75,
        "planned_picture_range": list(planning.EDIT),
        "case_order": list(planning.CASES),
        "binaries_sha256": source["binaries_sha256"],
        "packages": packages,
        "scope": "Fixed native FFV1 and PCM construction with exact source clocks, verified masks and lossless decode parity. Achieved motion and detector accuracy remain unassessed.",
    }


def freeze():
    missing = [
        name
        for name, filename in CONTEXTS
        if not all(p.is_file() for p in (BASE / name / "freeze.json", PANEL / filename))
    ]
    if missing:
        print(
            json.dumps({"status": "pending_prerequisite_no_stage_attempt", "missing_contexts": missing}),
            flush=True,
        )
        return
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps(
            {"status": "frozen", "pilot_clips": len(body["pilot_clip_ordinals"]), "new_neural_calls": 0}
        ),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Published unchanged native constructor required",
    )


def producer_modules():
    from scripts import grid_musetalk_frames_v148 as musetalk
    from scripts import grid_portrait_frames_v146 as portrait
    from scripts import grid_portrait_geometry_v144 as geometry
    from scripts import grid_source_speech_v147 as speech

    return geometry, portrait, speech, musetalk


def producer_paths(ordinal, batch):
    geometry, portrait, speech, musetalk = producer_modules()
    clips = [mod.TRIAL / f"clip-{ordinal:02d}/authored" for mod in (geometry, portrait, musetalk)]
    states = speech.TRIAL / f"batch-{batch:02d}/authored"
    return [*clips, states]


def portrait_receipt(source, geometry, portrait, speech):
    ordinal = source["ordinal"]
    folder = portrait.TRIAL / f"clip-{ordinal:02d}/authored"
    output = folder / "output"
    cached = portrait.CACHE / f"clip-{ordinal:02d}/outputs.json"
    result = read(output / "result.json")
    speech.qualification_profile(
        folder, result["new_neural_calls"], {}, caller="grid_portrait_frames_v146.py"
    )
    records = read(cached)
    require(
        result["schema"] == "chipmunk.grid-portrait-frames-result/146"
        and result["clip_ordinal"] == ordinal
        and result["freeze_sha256"] == sha(portrait.PUBLIC)
        and result["private_output_references_sha256"] == sha(cached)
        and records["schema"] == "chipmunk.grid-portrait-frames-private/146"
        and records["clip_ordinal"] == ordinal
        and records["planned"] == source,
        "Actual bound complete full-expression producer receipt required",
    )
    paths = {
        "private-input-binding.json": "private_input_binding_sha256",
        "render-records.jsonl": "private_render_journal_sha256",
        "root-progress.jsonl": "private_root_progress_sha256",
        "operator-progress.jsonl": "private_operator_progress_sha256",
        "repeat-check.json": "private_repeat_check_sha256",
    }
    for name, key in paths.items():
        require(sha(output / name) == result[key], "Completed full-expression journal changed")
    inputs = read(output / "private-input-binding.json")
    require(
        inputs["source"] == source and inputs["geometry"] == geometry,
        "Full-expression source and geometry differ from completed inputs",
    )
    jobs, _ = portrait.jobs_for(source, geometry)
    frames = records["frames"]
    for actual, expected in zip(frames, jobs, strict=True):
        require(
            all(actual[k] == v for k, v in expected.items())
            and actual["rendered"] is (expected["status"] == "available"),
            "Actual full-expression frame routing differs",
        )
    count = sum(r["rendered"] for r in frames)
    repeat = int(frames[0]["rendered"])
    roots = portrait.expected_roots(count + repeat)
    require(
        len(frames) == result["planned_condition_frames"] == 40
        and result["executed_condition_frames"] == result["available_condition_frames"] == count
        and result["skipped_condition_frames"] == 40 - count
        and result["executed_repeat_frames"] == repeat
        and result["renderer_inputs"] == count + repeat
        and result["renderer_repeat_bitwise"] is (True if repeat else None)
        and result["top_level_module_forwards_started"]
        == result["top_level_module_forwards_completed"]
        == roots
        and result["new_neural_calls"] == sum(roots.values())
        and result["new_face_task_inputs"] == result["new_speech_encoder_inputs"] == 0
        and result["defaults_changed"] is False,
        "Actual full-expression execution counts differ",
    )
    return records


def input_binding(ordinal):
    source = pilot(ordinal)
    geo, portrait, speech, musetalk = producer_modules()
    geometry = geo.load_completed(ordinal)
    live = portrait_receipt(source, geometry, portrait, speech)
    states = {name: speech.source_record(ordinal, name) for name in speech.WAVEFORMS}
    muse = musetalk.load_completed(ordinal)
    muse_input = read(musetalk.TRIAL / f"clip-{ordinal:02d}/authored/output/private-input-binding.json")
    require(
        muse_input["source"] == source and muse_input["original_state"] == states["original_pcm"],
        "MuseTalk must bind the same original whole-waveform record",
    )
    paths = producer_paths(ordinal, states["original_pcm"]["batch"])
    bindings = {}
    for stage in paths:
        for item in ("profile.json", "resources.jsonl", "output/result.json"):
            planning.bind_metadata(stage / item, bindings)
    return {
        "source": source,
        "liveportrait_geometry": geometry,
        "liveportrait_frames": live,
        "musetalk_frames": muse,
        "waveforms": states,
        "actual_producer_receipts": bindings,
    }


def case_numbers(source):
    """Stable opaque numeric order, with the mapping retained privately."""
    seed = bytes.fromhex(source["input"]["source_sha256"])
    require(len(seed) == 32, "Exact private source commitment required")
    names = [row["id"] for row in source["cases"]]
    require(names == list(planning.CASES), "All eleven fixed condition labels required")
    order = sorted(
        names, key=lambda name: hashlib.sha256(b"grid-native-clips-v149\0" + seed + name.encode()).digest()
    )
    return {name: i for i, name in enumerate(order)}


def source_clock(row):
    if row["status"] != planning.reviewed.AVAILABLE:
        return None, row["status"]
    table = row["native_video_table"]
    origin = planning.reviewed.video_clock(table, row["pictures"])
    require(
        len(table["pts"]) == 75 and origin == Fraction(row["video_origin_seconds"]),
        "Complete 75-picture source clock and observed video origin required",
    )
    pts = [Fraction(value) * Fraction(table["time_base"]) for value in table["pts"]]
    audio = Fraction(row["audio_origin_seconds"])
    ranges = planning.reviewed.mapping_ranges(row["pcm_mapping"], row["pcm_mapping"]["samples"])
    require(Fraction(ranges[0]["source_start_pts"], 16000) == audio, "Observed source PCM origin differs")
    if any(t < 0 or (t * 1000).denominator != 1 for t in [*pts, audio]):
        return None, "Qualified Matroska mux requires exact nonnegative millisecond source clocks"
    pts_ms = [int(t * 1000) for t in pts]
    picture_steps = sorted(b - a for a, b in zip(pts_ms, pts_ms[1:], strict=False))
    return {
        "pts": pts,
        "pts_ms": pts_ms,
        "constructed_last_picture_duration_ms": picture_steps[len(picture_steps) // 2],
        "audio_origin": audio,
        "shape": (table["height"], table["width"], 3),
    }, None


def owned_file(path, root, expected=None):
    path = Path(path)
    require(
        path.is_absolute()
        and path.resolve().is_relative_to(root.resolve())
        and (expected is None or path.resolve() == expected.resolve()),
        "Private payload path ownership differs",
    )
    return path


def read_png(ref, root, shape, expected=None):
    from PIL import Image

    path = owned_file(ref["path"], root, expected)
    require(sha(path) == ref["sha256"], "Bound PNG file bytes changed")
    with Image.open(path) as image:
        require(
            image.format == "PNG" and image.mode == ("RGB" if len(shape) == 3 else "L"),
            "Plain lossless PNG pixel format required",
        )
        value = np.asarray(image).copy()
    require(
        value.shape == shape and value.dtype == np.uint8 and fingerprint(value) == ref["raw_sha256"],
        "Decoded PNG pixels differ from actual producer commitment",
    )
    return value


def source_pictures(source, clock):
    row = source["input"]
    bank = planning.reviewed.BANK
    folder = bank / f"source-{source['source_slot']:02d}/clip-{source['within_source_ordinal']:02d}"
    result = []
    for index, picture in enumerate(row["pictures"]):
        require(
            picture["frame_index"] == index and picture["pts"] == row["native_video_table"]["pts"][index],
            "Every native source picture must match its observed PTS",
        )
        result.append(
            read_png(
                {
                    "path": picture["file"],
                    "sha256": picture["file_sha256"],
                    "raw_sha256": picture["rgb_sha256"],
                },
                bank,
                clock["shape"],
                folder / f"frame-{index:04d}.png",
            )
        )
    require(len(result) == 75, "All original picture bytes required")
    return result


def read_waveform(source, record, original, speech):
    name, waveform = record["waveform_id"], record["waveform"]
    folder = speech.CACHE / f"batch-{record['batch']:02d}/clip-{source['ordinal']:02d}"
    array = owned_file(waveform["pcm_file"], speech.CACHE, folder / f"{name}.pcm.npy")
    wav = owned_file(waveform["wav_file"], speech.CACHE, folder / f"{name}.wav")
    require(
        sha(array) == waveform["pcm_file_sha256"] and sha(wav) == waveform["wav_file_sha256"],
        "Completed whole-waveform bytes changed",
    )
    pcm = np.load(array, allow_pickle=False)
    raw = speech.pcm_lattice(pcm)
    expected, changes = (
        (original, 0) if name == "original_pcm" else speech.delayed_pcm(original, source["audio_delay"])
    )
    with wave.open(str(wav), "rb") as decoded:
        require(
            (
                decoded.getnchannels(),
                decoded.getsampwidth(),
                decoded.getframerate(),
                decoded.getnframes(),
                decoded.getcomptype(),
            )
            == (1, 2, 16000, len(original), "NONE"),
            "Exact canonical whole-waveform WAV format required",
        )
        require(decoded.readframes(len(original) + 1) == raw, "WAV and complete saved float32 PCM differ")
    require(
        pcm.shape == original.shape
        and pcm.tobytes() == expected.tobytes()
        and fingerprint(pcm) == waveform["pcm_sha256"]
        and changes == waveform["changed_samples"],
        "Completed audio values differ from the exact planned input copy",
    )
    return np.frombuffer(raw, dtype="<i2").copy()


def geometry_rows(bundle, branch):
    rows = (
        bundle["liveportrait_geometry"]["pictures"]
        if branch == "liveportrait"
        else bundle["musetalk_frames"]["geometry"]
    )
    by_index = {r["frame_index"]: r for r in rows}
    require(len(by_index) == len(rows), "Duplicate producer geometry positions")
    return by_index


def render_rows(bundle, branch):
    rows = bundle[f"{branch}_frames"]["frames"]
    require(
        [(r["condition"], r["frame_index"]) for r in rows] == [(mode, i) for mode in MODES for i in FRAMES]
        and all(type(r["rendered"]) is bool for r in rows),
        "All forty explicit producer frame outcomes required",
    )
    return {(r["condition"], r["frame_index"]): r for r in rows}


def case_reason(case, clock_reason, geometry, renders, waveforms):
    if case["status"] == "unavailable":
        require(bool(case["unavailable_reason"]), "An unavailable planned case requires its original reason")
        return case["unavailable_reason"]
    require(case["status"] == "planned", "Exact frozen case availability required")
    if clock_reason is not None:
        return clock_reason
    waveform = waveforms[case["waveform_id"]]
    if waveform["status"] == "unavailable":
        require(bool(waveform["unavailable_reason"]), "An unavailable waveform requires its original reason")
        return waveform["unavailable_reason"]
    require(waveform["status"] == "encoded", "Actual complete source waveform receipt required")
    branch = case["generator_branch"]
    if branch and any(geometry[branch].get(i, {}).get("status") != "geometry_available" for i in FRAMES):
        return "Complete branch source crop and alpha are unavailable"
    if case["planned_neural_provenance"] and any(
        not renders[branch][case["driver"], i]["rendered"] for i in FRAMES
    ):
        return "Complete twenty-picture neural producer output is unavailable"
    return None


def load_geometry(source, originals, branch, row, geo, portrait, muse):
    import cv2

    i, ordinal = row["frame_index"], source["ordinal"]
    if branch == "liveportrait":
        rgb, crop, alpha, matrix, proof = portrait.load_frame(row, source, geo.CACHE / f"clip-{ordinal:02d}")
        require(np.array_equal(rgb, originals[i]), "Full-expression crop source differs from verified bank")
        return {"crop": crop, "alpha": alpha, "matrix": matrix, "proof": proof}
    picture = source["input"]["pictures"][i]
    require(
        row["picture"] == picture and row["geometry"]["mode"] == "landmark-box",
        "Exact bound MuseTalk source geometry required",
    )
    folder = muse.CACHE / f"clip-{ordinal:02d}/geometry-{i:04d}"
    crop = read_png(row["crop"], muse.CACHE, (256, 256, 3), folder / "crop.png")
    path = owned_file(row["alpha"]["path"], muse.CACHE, folder / "alpha.npy")
    require(sha(path) == row["alpha"]["sha256"], "MuseTalk alpha file differs from producer commitment")
    alpha = np.load(path, allow_pickle=False)
    require(
        alpha.shape == originals[i].shape[:2]
        and alpha.dtype == np.float32
        and np.isfinite(alpha).all()
        and ((alpha >= 0) & (alpha <= 1)).all()
        and fingerprint(alpha) == row["alpha_f32_sha256"],
        "Exact native MuseTalk alpha required",
    )
    x0, y0, x1, y1 = row["geometry"]["bbox"]
    h, w = originals[i].shape[:2]
    require(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, "Original crop must fit the native picture")
    sx, sy = 256 / (x1 - x0), 256 / (y1 - y0)
    matrix = np.array([[sx, 0, (0.5 - x0) * sx - 0.5], [0, sy, (0.5 - y0) * sy - 0.5]], np.float64)
    require(
        np.array_equal(matrix, portrait.hex_array(row["source_to_crop_matrix_hex"], (2, 3)))
        and np.array_equal(
            crop, cv2.resize(originals[i][y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_LANCZOS4)
        )
        and fingerprint(crop) == row["crop_rgb_sha256"],
        "Original MuseTalk crop or affine differs",
    )
    return {"crop": crop, "alpha": alpha, "matrix": matrix, "proof": row}


def inverse_warp(crop, matrix, shape):
    import cv2

    require(
        matrix.shape == (2, 3)
        and matrix.dtype == np.float64
        and np.isfinite(matrix).all()
        and abs(np.linalg.det(matrix[:, :2])) > 0,
        "Finite invertible producer affine required",
    )
    return cv2.warpAffine(
        crop, cv2.invertAffineTransform(matrix), (shape[1], shape[0]), flags=cv2.INTER_LINEAR
    )


def ordinary_pixels(original, geometry, branch, brightness):
    import cv2

    crop = geometry["crop"]
    require(
        branch in BRANCHES and crop.dtype == np.uint8 and crop.shape == (256, 256, 3),
        "Original 256-pixel branch crop required",
    )
    if branch == "liveportrait":
        crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
    pixels = inverse_warp(crop, geometry["matrix"], original.shape)
    if brightness:
        pixels = np.rint(pixels * 1.1).clip(0, 255).astype(np.uint8)
    return pixels


def exact_mask(original, edited, mask, alpha=None):
    require(
        original.shape == edited.shape and original.dtype == edited.dtype == np.uint8,
        "Matching native RGB inputs required",
    )
    changed = np.any(original != edited, axis=2)
    require(
        mask.dtype == np.uint8 and np.array_equal(mask, changed.astype(np.uint8) * 255),
        "Producer mask must equal every actually changed native pixel",
    )
    if alpha is not None:
        require(
            alpha.shape == changed.shape and not changed[alpha == 0].any(),
            "Pixels changed outside exact branch alpha support",
        )
    return int(changed.sum())


def neural_picture(source, original, geometry, branch, record, producer, portrait):
    i, mode = record["frame_index"], record["condition"]
    folder = producer.CACHE / f"clip-{source['ordinal']:02d}"
    require(
        record["rendered"] is True
        and record["opacity"] == portrait.opacity(i)
        and record["outside_alpha_changed_pixels"] == 0
        and set(record["files"]) == {"crop", "native", "mask"},
        "Complete bound producer composite and mask required",
    )
    if branch == "liveportrait":
        require(record["source"] == geometry["proof"], "Full-expression rendered source proof differs")
    else:
        frame = source["frames"][i - FRAMES[0]]
        route = frame["source_time_conditioning" if mode == "source_time" else "earlier_conditioning"]
        require(
            record["conditioning_plan"] == route
            and record["proof"]["output_rgb_sha256"] == record["files"]["crop"]["raw_sha256"],
            "MuseTalk render or source-time conditioning proof differs",
        )
    size = 512 if branch == "liveportrait" else 256
    crop = read_png(
        record["files"]["crop"], producer.CACHE, (size, size, 3), folder / f"{mode}-{i:04d}-crop.png"
    )
    edited = read_png(
        record["files"]["native"], producer.CACHE, original.shape, folder / f"{mode}-{i:04d}-native.png"
    )
    mask = read_png(
        record["files"]["mask"], producer.CACHE, original.shape[:2], folder / f"{mode}-{i:04d}-mask.png"
    )
    expected, _ = portrait.composite(
        original,
        inverse_warp(crop, geometry["matrix"], original.shape),
        geometry["alpha"],
        portrait.opacity(i),
    )
    require(
        np.array_equal(edited, expected),
        "Saved native composite differs from its exact source, crop and alpha",
    )
    require(
        exact_mask(original, edited, mask, geometry["alpha"]) == record["changed_native_pixels"],
        "Producer changed-pixel count differs",
    )
    return edited, mask


def case_pictures(case, source, originals, pts, geometry, renders, portrait, muse):
    replacements = {}
    branch = case["generator_branch"]
    for i in case["output_picture_indices"]:
        original = originals[i]
        if case["operation"] == "copy_genuine_earlier_pictures":
            earlier = i - 8
            frame = source["frames"][i - FRAMES[0]]
            require(
                frame["earlier_picture"] == source["input"]["pictures"][earlier]
                and pts[i] - pts[earlier] == Fraction(8, 25),
                "Exact genuine earlier picture/clock required",
            )
            edited = originals[earlier]
            mask = np.any(edited != original, axis=2).astype(np.uint8) * 255
        elif case["planned_neural_provenance"]:
            producer = portrait if branch == "liveportrait" else muse
            edited, mask = neural_picture(
                source,
                original,
                geometry[branch][i],
                branch,
                renders[branch][case["driver"], i],
                producer,
                portrait,
            )
        else:
            require(
                case["operation"] in {"branch_resample", "branch_resample_then_brightness"},
                "Unknown planned picture operation",
            )
            geo = geometry[branch][i]
            native = ordinary_pixels(
                original, geo, branch, case["operation"] == "branch_resample_then_brightness"
            )
            edited, mask = portrait.composite(original, native, geo["alpha"], portrait.opacity(i))
        replacements[i] = (edited, mask)
    require(
        list(replacements) == (list(FRAMES) if case["modified_channels"] == ["video"] else []),
        "Only complete twenty-picture variants may enter construction",
    )
    return replacements


def header_contract(container):
    streams = list(container.streams)
    require(
        "matroska" in container.format.name.split(",")
        and len(streams) == 2
        and [s.type for s in streams] == ["video", "audio"]
        and streams[0].codec_context.name == "ffv1"
        and streams[1].codec_context.name == "pcm_s16le"
        and streams[1].codec_context.sample_rate == 16000
        and streams[1].codec_context.channels == 1,
        "Actual uniform Matroska FFV1 and mono16k PCM streams required",
    )
    for tags in [container.metadata, *(stream.metadata for stream in streams)]:
        for key, value in tags.items():
            require(
                key.lower() in {"encoder", "duration", "language"}
                and not any(
                    label in value.lower() for label in (*planning.CASES[1:], "ground truth", "aigc")
                ),
                "Case/ground-truth labels must stay out of the media container",
            )


def build_case(file, frames, pcm, clock, mux, inspect_header):
    require(
        len(frames) == len(clock["pts_ms"]) == 75, "Complete native frames at every original PTS required"
    )
    expected = mux.encode(file, iter(frames), clock["pts_ms"], clock["shape"], pcm, clock["audio_origin"])
    require(
        expected["rgb_sha256"] == [fingerprint(frame) for frame in frames]
        and expected["picture_pts_ms"] == clock["pts_ms"]
        and expected["constructed_last_picture_duration_ms"] == clock["constructed_last_picture_duration_ms"]
        and Fraction(expected["audio_start_seconds"]) == clock["audio_origin"]
        and expected["audio_samples"] == len(pcm)
        and expected["pcm_s16_sha256"] == fingerprint(pcm),
        "Actual mux input proof differs from planned picture/sample values",
    )
    checked = mux.verify_media(file, expected, clock["shape"])
    require(checked == roundtrip_record(len(pcm)), "Actual lossless decode round trip required")
    inspect_header(file)
    return expected, checked


def roundtrip_record(samples):
    return {
        "pictures": 75,
        "audio_samples": samples,
        "picture_clock_exact": True,
        "rgb_exact": True,
        "pcm_exact": True,
        "audio_origin_exact": True,
    }


def case_counts(cases):
    return {
        "planned_cases": len(cases),
        "ready_cases": sum(c["status"] == "ready" for c in cases),
        "unavailable_cases": sum(c["status"] == "unavailable" for c in cases),
        "verified_output_pictures": sum(c.get("review", {}).get("pictures", 0) for c in cases),
        "verified_output_pcm_samples": sum(c.get("review", {}).get("audio_samples", 0) for c in cases),
        "verified_edit_masks": sum(len(c.get("edit_proofs", [])) for c in cases),
        "zero_effect_neural_cases": sum(
            c["neural_provenance_verified"] and not c["observed_neural_pixel_change"] for c in cases
        ),
    }


def construct(ordinal, output):
    geo, portrait, speech, muse = producer_modules()
    bundle = input_binding(ordinal)
    source = bundle["source"]
    clock, clock_reason = source_clock(source["input"])
    folder = CACHE / f"clip-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "input-binding.json", bundle)
    numbers = case_numbers(source)
    geometry = {branch: geometry_rows(bundle, branch) for branch in BRANCHES}
    renders = {branch: render_rows(bundle, branch) for branch in BRANCHES}
    reasons = {
        case["id"]: case_reason(case, clock_reason, geometry, renders, bundle["waveforms"])
        for case in source["cases"]
    }
    originals, pcm, caches = [], {}, {branch: {} for branch in BRANCHES}
    if any(reason is None for reason in reasons.values()):
        import cv2

        from scripts import native_av_mux_v119 as mux

        av = importlib.import_module("av")  # v119 first adds its pinned isolated runtime.

        for name, digest in read(PUBLIC)["binaries_sha256"].items():
            binary = shutil.which(name)
            require(
                binary is not None and sha(binary) == digest,
                "Previously qualified command decoder binary required",
            )
        require(
            mux.runtime_identity() == read(mux.PUBLIC)["isolated_av_runtime"],
            "Exact qualified isolated mux runtime required",
        )
        cv2.setNumThreads(1)
        originals = source_pictures(source, clock)
        planning.reviewed.verify_pcm(source["input"])
        original = np.load(source["input"]["pcm_file"], allow_pickle=False)
        used_waveforms = {c["waveform_id"] for c in source["cases"] if reasons[c["id"]] is None}
        for name in sorted(used_waveforms):
            pcm[name] = read_waveform(source, bundle["waveforms"][name], original, speech)
        for branch in BRANCHES:
            if any(c["generator_branch"] == branch and reasons[c["id"]] is None for c in source["cases"]):
                caches[branch] = {
                    i: load_geometry(source, originals, branch, geometry[branch][i], geo, portrait, muse)
                    for i in FRAMES
                }

    def inspect_header(path):
        with av.open(str(path), mode="r") as container:
            header_contract(container)

    cases = []
    with (folder / "cases.jsonl").open("x") as journal:
        for case in source["cases"]:
            name, number = case["id"], numbers[case["id"]]
            result = {
                "schema": "chipmunk.grid-native-clip-case/149",
                "asset_index": ordinal,
                "source_slot": source["source_slot"],
                "component_index": source["component_index"],
                "study_role": "pilot",
                "condition": name,
                "case_number": number,
                "status": "unavailable",
                "reason": reasons[name],
                "planned_neural_provenance": case["planned_neural_provenance"],
                "neural_provenance_verified": False,
                "achieved_synchronization_qualified": False,
                "achieved_motion_assessed": False,
                "edit_frames_half_open": case["edit_frames_half_open"],
                "waveform_id": case["waveform_id"],
            }
            if reasons[name] is None:
                replacements = case_pictures(
                    case, source, originals, clock["pts"], caches, renders, portrait, muse
                )
                frames = [replacements[i][0] if i in replacements else rgb for i, rgb in enumerate(originals)]
                file = folder / f"{number:02d}.mkv"
                expected, checked = build_case(
                    file, frames, pcm[case["waveform_id"]], clock, mux, inspect_header
                )
                masks = []
                for i, (edited, mask) in replacements.items():
                    ref = portrait.save_png(folder / f"{number:02d}-{i:04d}.png", mask)
                    masks.append(
                        {
                            "frame_index": i,
                            "pts": source["input"]["native_video_table"]["pts"][i],
                            "mask_file": ref["path"],
                            "mask_sha256": ref["sha256"],
                            "mask_raw_sha256": ref["raw_sha256"],
                            "changed_pixels": exact_mask(originals[i], edited, mask),
                        }
                    )
                wave = bundle["waveforms"][case["waveform_id"]]["waveform"]
                changed = sum(mask["changed_pixels"] for mask in masks)
                bounds = case["edit_frames_half_open"]
                result.update(
                    status="ready",
                    reason=None,
                    file=str(file),
                    file_sha256=sha(file),
                    expected=expected,
                    review=checked,
                    uniform_lossless_header_verified=True,
                    edit_proofs=masks,
                    source_video_time_base=source["input"]["native_video_table"]["time_base"],
                    source_video_pts=source["input"]["native_video_table"]["pts"],
                    audio_output_clock_mapping=wave["output_clock_mapping"],
                    audio_value_mapping=wave["input_output_sample_mapping"],
                    changed_audio_samples=wave["changed_samples"],
                    changed_native_pixels=changed,
                    changed_video_frames=sum(mask["changed_pixels"] > 0 for mask in masks),
                    neural_provenance_verified=case["planned_neural_provenance"],
                    observed_neural_pixel_change=case["planned_neural_provenance"] and changed > 0,
                    source_time_bounds_seconds=None
                    if bounds is None
                    else [str(clock["pts"][i]) for i in bounds],
                    before_during_after=None
                    if bounds is None
                    else {"before": [25, 26], "during": [27, 37, 46], "after": [47, 48]},
                )
            cases.append(result)
            journal.write(json.dumps(result, allow_nan=False) + "\n")
            journal.flush()
            print(
                json.dumps(
                    {"cases_recorded": len(cases), "ready_cases": sum(c["status"] == "ready" for c in cases)}
                ),
                flush=True,
            )
    require(len(cases) == 11, "All eleven planned outcomes must remain recorded")
    result = {
        "schema": "chipmunk.grid-native-clips-result/149",
        "status": "complete",
        "clip_ordinal": ordinal,
        **case_counts(cases),
        "private_case_journal_sha256": sha(folder / "cases.jsonl"),
        "private_input_binding_sha256": sha(folder / "input-binding.json"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Native picture and soundtrack construction plus exact lossless decode verification. Achieved motion, synchronization and detection performance remain unassessed.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def validate_case_inventory(cases, bundle, folder):
    source = bundle["source"]
    clock, clock_reason = source_clock(source["input"])
    geometry = {branch: geometry_rows(bundle, branch) for branch in BRANCHES}
    renders = {branch: render_rows(bundle, branch) for branch in BRANCHES}
    numbers = case_numbers(source)
    require(len(cases) == 11, "All eleven fixed outcomes required")
    for case, spec in zip(cases, source["cases"], strict=True):
        reason = case_reason(spec, clock_reason, geometry, renders, bundle["waveforms"])
        require(
            case["schema"] == "chipmunk.grid-native-clip-case/149"
            and case["condition"] == spec["id"]
            and case["case_number"] == numbers[spec["id"]]
            and case["asset_index"] == source["ordinal"]
            and case["source_slot"] == source["source_slot"]
            and case["component_index"] == source["component_index"]
            and case["study_role"] == "pilot"
            and case["planned_neural_provenance"] is spec["planned_neural_provenance"]
            and case["waveform_id"] == spec["waveform_id"]
            and case["edit_frames_half_open"] == spec["edit_frames_half_open"]
            and case["achieved_synchronization_qualified"] is case["achieved_motion_assessed"] is False
            and case["status"] == ("ready" if reason is None else "unavailable")
            and case["reason"] == reason,
            "Complete fixed case identity and availability required",
        )
        if reason is not None:
            require(
                "file" not in case and case["neural_provenance_verified"] is False,
                "Unavailable cases cannot advertise completed construction",
            )
            continue
        wave = bundle["waveforms"][spec["waveform_id"]]["waveform"]
        table = source["input"]["native_video_table"]
        proof = case["expected"]
        owned_file(case["file"], CACHE, folder / f"{case['case_number']:02d}.mkv")
        require(
            case["uniform_lossless_header_verified"] is True
            and case["review"] == roundtrip_record(wave["samples"])
            and proof["picture_pts_ms"] == clock["pts_ms"]
            and proof["constructed_last_picture_duration_ms"] == clock["constructed_last_picture_duration_ms"]
            and proof["audio_samples"] == wave["samples"]
            and Fraction(proof["audio_start_seconds"]) == clock["audio_origin"]
            and len(proof["rgb_sha256"]) == 75
            and case["source_video_time_base"] == table["time_base"]
            and case["source_video_pts"] == table["pts"]
            and case["audio_output_clock_mapping"] == wave["output_clock_mapping"]
            and case["audio_value_mapping"] == wave["input_output_sample_mapping"]
            and case["changed_audio_samples"] == wave["changed_samples"],
            "Complete actual native clock/pixel/sample proof required",
        )
        hashes = [case["file_sha256"], proof["pcm_s16_sha256"], *proof["rgb_sha256"]]
        require(
            [m["frame_index"] for m in case["edit_proofs"]] == spec["output_picture_indices"],
            "Full planned mask coverage required",
        )
        for mask in case["edit_proofs"]:
            i = mask["frame_index"]
            owned_file(mask["mask_file"], CACHE, folder / f"{case['case_number']:02d}-{i:04d}.png")
            require(
                mask["pts"] == table["pts"][i]
                and type(mask["changed_pixels"]) is int
                and 0 <= mask["changed_pixels"] <= table["height"] * table["width"],
                "Exact bounded native mask metadata required",
            )
            hashes.extend((mask["mask_sha256"], mask["mask_raw_sha256"]))
        changed = sum(m["changed_pixels"] for m in case["edit_proofs"])
        require(
            all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes)
            and case["changed_native_pixels"] == changed
            and case["changed_video_frames"] == sum(m["changed_pixels"] > 0 for m in case["edit_proofs"])
            and case["neural_provenance_verified"] is spec["planned_neural_provenance"]
            and case["observed_neural_pixel_change"] is (spec["planned_neural_provenance"] and changed > 0),
            "Bound payload hashes, changed support and construction provenance required",
        )


def load_completed(ordinal):
    """Read the completed private case inventory. Consumers verify payload bytes inside their worker."""
    verify()
    bundle = input_binding(ordinal)
    stage = TRIAL / f"clip-{ordinal:02d}/bind-00"
    result = planning.reviewed.stage(stage, sha(PUBLIC), "grid_native_clips_v149.py", 0, {})
    folder = CACHE / f"clip-{ordinal:02d}"
    require(
        result["schema"] == "chipmunk.grid-native-clips-result/149"
        and result["clip_ordinal"] == ordinal
        and result["private_case_journal_sha256"] == sha(folder / "cases.jsonl")
        and result["private_input_binding_sha256"] == sha(folder / "input-binding.json")
        and read(folder / "input-binding.json") == bundle,
        "Actual complete native construction metadata required",
    )
    cases = planning.reviewed.lines(folder / "cases.jsonl")
    validate_case_inventory(cases, bundle, folder)
    require(
        all(result[k] == v for k, v in case_counts(cases).items()),
        "All case outcomes must reconcile with the actual stage receipt",
    )
    return cases


def prerequisite(ordinal):
    plan = planning.load_completed()
    pilots = [s["ordinal"] for s in plan["sources"] if s["role"] == "pilot"]
    prior = pilots.index(ordinal) - 1
    _, _, speech, _ = producer_modules()
    batch = next(
        i for i, group in enumerate(speech.batches(plan)) if any(s["ordinal"] == ordinal for s in group)
    )
    stages = producer_paths(ordinal, batch)
    if prior >= 0:
        stages.append(TRIAL / f"clip-{pilots[prior]:02d}/bind-00")
    if not all(
        (folder / name).is_file() for folder in stages for name in ("profile.json", "output/result.json")
    ):
        return False
    if prior >= 0:
        load_completed(pilots[prior])
    input_binding(ordinal)
    return True


def routed_popen(ordinal, argv, **kwargs):
    expected = TRIAL / f"clip-{ordinal:02d}/bind-00/output"
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(expected), "0"],
        "Only the exclusive native construction worker is permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_native_clips_v149", "construct", str(expected), str(ordinal)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    pilot(ordinal)
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
            "prerequisites": lambda _command, _batch: prerequisite(ordinal),
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "construct"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze accepts no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One fixed pilot source ordinal required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {str(i) for i in range(32)},
            "Exact output and source required",
        )
        output, ordinal = Path(args.arguments[0]), int(args.arguments[1])
        require(
            output.resolve() == (TRIAL / f"clip-{ordinal:02d}/bind-00/output").resolve(),
            "Exclusive guarded output required",
        )
        output.mkdir(exist_ok=False)
        construct(ordinal, output)
