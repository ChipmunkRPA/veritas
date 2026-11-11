"""Extract the frozen native mouth and speech features from each constructed pilot clip."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import av_conditioning_clock_v109 as clock
from scripts import grid_transfer_plan_v143 as planning
from scripts import mouth_articulation_core_v122 as articulation

ROOT, BASE, PANEL, ARCHIVE = planning.ROOT, planning.BASE, planning.PANEL, planning.ARCHIVE
TRIAL = BASE / "grid-native-mouth-features-v150"
CACHE = ARCHIVE / "grid-native-mouth-features-v150"
PUBLIC = PANEL / "grid-native-mouth-features-freeze-v150.json"
FRAMES = tuple(range(75))
REPEAT_FRAME = 27
CODE = (
    "backend/scripts/grid_native_mouth_features_v150.py",
    "backend/tests/test_grid_native_mouth_features_v150.py",
    "docs/grid-native-mouth-features-v150-method.md",
    "backend/scripts/grid_transfer_plan_v143.py",
    "backend/scripts/grid_source_speech_v147.py",
    "backend/scripts/grid_native_clips_v149.py",
    "backend/scripts/av_conditioning_clock_v109.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/app/detectors/video/decode.py",
    "backend/app/detectors/audio/decode.py",
)
read, write, sha, require, exact = (
    planning.read,
    planning.write,
    planning.sha,
    planning.require,
    planning.exact,
)


def feature_modules():
    # The historical import chain includes a PyAV path insertion. The feature
    # helpers use the app decoder, so that path is unnecessary in this context.
    saved = sys.path[:]
    try:
        from scripts import native_mouth_features_v123 as features
        from scripts import review_native_mouth_features_v124 as replay
    finally:
        sys.path[:] = saved
    return features, replay


def upstream():
    from scripts import grid_native_clips_v149 as construction
    from scripts import grid_source_speech_v147 as speech

    return construction, speech


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bind(path, bindings):
    bindings[str(path.relative_to(ROOT))] = sha(path)
    return read(path)


def frozen_record(public, trial, bindings):
    record = bind(public, bindings)
    require(record == bind(trial / "freeze.json", bindings), "Published and private frozen contracts differ")
    for name, digest in record["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Frozen dependency source changed")
    return record


def actual_receipt(folder, freeze_sha, bindings):
    result = bind(folder / "output/result.json", bindings)
    profile = bind(folder / "profile.json", bindings)
    resources = folder / "resources.jsonl"
    bindings[str(resources.relative_to(ROOT))] = sha(resources)
    require(
        result["status"] == "complete"
        and result["freeze_sha256"] == freeze_sha
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["new_neural_calls"] == result["new_neural_calls"]
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
        and profile["resource_journal_sha256"] == sha(resources),
        "Actual successful unchanged historical stage required",
    )
    return result, profile


def feature_qualification(bindings):
    """Bind actual v123/v124 receipts and code without opening historical media or vectors."""
    features, replay = feature_modules()
    f = frozen_record(features.PUBLIC, features.TRIAL, bindings)
    a = frozen_record(replay.PUBLIC, replay.TRIAL, bindings)
    fsha, asha = sha(features.PUBLIC), sha(replay.PUBLIC)
    require(
        a["feature_freeze_sha256"] == fsha
        and f["image_device"] == "mps"
        and f["image_dtype"] == "float32"
        and f["image_batch_size"] == 1
        and f["attention"] == "eager"
        and all(importlib.metadata.version(k) == v for k, v in f["packages"].items()),
        "Same qualified singleton image environment required",
    )
    require(
        set(f["model_records"]) == {"dinov2.small", "yunet.2023mar"}
        and all(r["license"]["class_"] == "commercial" for r in f["model_records"].values()),
        "Same commercial image and face records required",
    )
    authored = features.TRIAL / "authored"
    result, profile = actual_receipt(authored, fsha, bindings)
    public = bind(PANEL / "native-mouth-feature-contract-result-v123.json", bindings)
    proof = authored / "output/private-proof.json"
    bindings[str(proof.relative_to(ROOT))] = sha(proof)
    published_measurement = {
        k: v for k, v in public.items() if k not in {"resource_profile", "source_commit"}
    }
    shareable_measurement = {k: v for k, v in result.items() if k != "source_to_constructed_feature_l2"}
    require(
        result["schema"] == "chipmunk.native-mouth-feature-contract/123"
        and result["strict_encoder_tensor_count"] == 223
        and result["face_inputs"] == result["native_image_inputs"] == 5
        and result["new_neural_calls"] == 10
        and result["repeat_face_crop_input_features_bitwise"] is True
        and result["private_proof_sha256"] == sha(proof)
        and published_measurement == shareable_measurement
        and all(profile[k] == v for k, v in public["resource_profile"].items()),
        "Actual strict historical feature and repeat qualification required",
    )
    complete = bind(PANEL / "native-mouth-features-complete-v124.json", bindings)
    require(
        complete["status"] == "complete"
        and complete["planned_cases"] == 288
        and complete["extracted_cases"] == 283
        and complete["audit_new_neural_calls"] == 0
        and complete["feature_freeze_sha256"] == fsha
        and complete["audit_freeze_sha256"] == asha
        and [r["batch"] for r in complete["batches"]] == list(range(8)),
        "All eight published feature replay batches required",
    )
    for i, commitment in enumerate(complete["batches"]):
        folder = features.TRIAL / f"extract-{i:02d}"
        review = replay.TRIAL / f"bind-{i:02d}"
        extraction, _ = actual_receipt(folder, fsha, bindings)
        audit, _ = actual_receipt(review, asha, bindings)
        require(
            extraction["batch"] == audit["batch"] == i
            and audit["new_neural_calls"] == 0
            and audit["extraction_receipt_sha256"] == sha(folder / "output/result.json"),
            "Actual historical feature/replay correspondence required",
        )
        for key, path in (
            ("feature_result_sha256", folder / "output/result.json"),
            ("feature_profile_sha256", folder / "profile.json"),
            ("feature_journal_sha256", folder / "output/feature-cases.jsonl"),
            ("audit_result_sha256", review / "output/result.json"),
            ("audit_profile_sha256", review / "profile.json"),
            ("audit_journal_sha256", review / "output/reviewed-cases.jsonl"),
        ):
            digest = sha(path)
            require(digest == commitment[key], "Published historical batch commitment changed")
            bindings[str(path.relative_to(ROOT))] = digest
        require(
            extraction["feature_case_journal_sha256"] == commitment["feature_journal_sha256"]
            and audit["private_review_journal_sha256"] == commitment["audit_journal_sha256"],
            "Actual historical worker journals differ",
        )


def identity():
    plan = planning.load_completed()
    construction, speech = upstream()
    bindings = {}
    feature_qualification(bindings)
    frozen_record(construction.PUBLIC, construction.TRIAL, bindings)
    frozen_record(speech.PUBLIC, speech.TRIAL, bindings)
    return {
        "schema": "chipmunk.grid-native-mouth-features-freeze/150",
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "upstream_bindings_sha256": bindings,
        "plan_freeze_sha256": sha(planning.PUBLIC),
        "plan_private_sha256": sha(planning.TRIAL / "bind-00/output/private-plan.json"),
        "pilot_clip_ordinals": [r["ordinal"] for r in plan["sources"] if r["role"] == "pilot"],
        "conditions_per_clip": 11,
        "pictures_per_ready_case": 75,
        "fixed_original_repeat_frame": 27,
        "image_device": "mps",
        "image_dtype": "float32",
        "image_batch_size": 1,
        "image_variables": 768,
        "speech_dtype": "float64",
        "speech_variables": 3840,
        "scope": "All native pictures and actual output-clock whole-waveform speech states. Exact clip-local RGB memoization, independent array arithmetic replay and a fixed original repeat. Construction labels/masks/bounds are reserved for subsequent scoring.",
    }


def freeze():
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
        "Unchanged published feature extraction procedure required",
    )


def pilot(ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 32, "Original pilot clip ordinal required")
    source = planning.load_completed()["sources"][ordinal]
    require(
        source["ordinal"] == ordinal and source["role"] == "pilot",
        "Only the original pilot clips may be consumed",
    )
    return source


def case_inputs(case):
    """Project the media/time inputs before numerical processing."""
    return {
        key: case[key]
        for key in (
            "status",
            "file",
            "file_sha256",
            "expected",
            "source_video_time_base",
            "source_video_pts",
            "audio_output_clock_mapping",
            "waveform_id",
        )
    }


def verify_table(table, media, source):
    native = source["input"]["native_video_table"]
    expected = media["expected"]
    require(
        len(table) == len(expected["rgb_sha256"]) == len(native["pts"]) == 75
        and table.time_base == "1/1000"
        and list(table.pts) == expected["picture_pts_ms"]
        and not table.truncated
        and not table.interpolated
        and table.width == native["width"]
        and table.height == native["height"]
        and media["source_video_pts"] == native["pts"]
        and media["source_video_time_base"] == native["time_base"]
        and all(
            exact(t) * table.tb == exact(s) * exact(native["time_base"])
            for t, s in zip(table.pts, native["pts"], strict=True)
        ),
        "Actual complete MKV frame clock differs from native source/constructor proof",
    )


def verified_case(media, source, state, clip_cache, features, speech):
    """Materialize all checked RGB before exposing any picture to feature arithmetic."""
    from app.detectors.audio.decode import decode_audio, source_ranges
    from app.detectors.video.decode import probe_frames

    path = Path(media["file"])
    require(
        path.is_absolute()
        and path.resolve().is_relative_to(clip_cache.resolve())
        and sha(path) == media["file_sha256"],
        "Exact owned constructed clip required",
    )
    table = probe_frames(path, cap=75, timeout_s=60)
    verify_table(table, media, source)
    frames = list(features.verified_frames(media, table, FRAMES))
    require(
        [f.index for f in frames] == list(FRAMES) and all(f.pts == table.pts[f.index] for f in frames),
        "Complete decoded native PTS sequence required",
    )
    wave = state["waveform"]
    sound = decode_audio(path, source_sha256=media["file_sha256"], max_seconds=10)
    ranges = source_ranges(sound.mapping, 0, len(sound.pcm))
    require(len(ranges) == 1, "Contiguous observed case PCM clock required")
    audio_origin = Fraction(ranges[0]["source_start_pts"], 16000)
    expected = media["expected"]
    require(
        state["waveform_id"] == media["waveform_id"] == wave["id"]
        and media["audio_output_clock_mapping"] == wave["output_clock_mapping"]
        and len(sound.pcm) == expected["audio_samples"] == wave["samples"]
        and audio_origin == exact(expected["audio_start_seconds"]) == exact(wave["audio_origin_seconds"])
        and hashlib.sha256(speech.pcm_lattice(sound.pcm)).hexdigest() == expected["pcm_s16_sha256"]
        and fingerprint(sound.pcm) == wave["pcm_sha256"]
        and sha(path) == media["file_sha256"],
        "Actual complete case PCM, origin or immutable clip differs from bound state",
    )
    return table, frames, audio_origin


def speech_contexts(table, audio_origin, samples, allowance):
    video_origin = exact(table.pts[0]) * table.tb
    rows = []
    for i, pts in enumerate(table.pts):
        plan = clock.frame_plan(
            i, str(exact(pts) * table.tb), str(audio_origin), str(video_origin), samples, str(allowance)
        )
        supported = plan["source_sample_status"] == "supported" and all(plan["retained_tick_slots"])
        rows.append(
            {
                "frame_index": i,
                "status": "supported" if supported else "unavailable",
                "plan": plan,
                "reason": None if supported else "Complete observed ten-tick speech context unavailable",
            }
        )
    return rows


def save_array(path, rows, width, dtype):
    values = np.stack(rows) if rows else np.empty((0, width), dtype)
    require(
        values.shape == (len(rows), width) and values.dtype == dtype and np.isfinite(values).all(),
        "Exact finite feature shape/dtype required",
    )
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return {
        "file": str(path),
        "sha256": sha(path),
        "shape": list(values.shape),
        "dtype": np.dtype(dtype).name,
    }


def speech_features(folder, state, contexts, speech, replay):
    name = state["waveform_id"]
    path = Path(state["hidden_file"])
    expected_path = (
        speech.CACHE / f"batch-{state['batch']:02d}/clip-{state['source_ordinal']:02d}/{name}.hidden.npy"
    )
    require(
        path.is_absolute()
        and path.resolve() == expected_path.resolve()
        and path.resolve().is_relative_to(speech.CACHE.resolve())
        and sha(path) == state["hidden_file_sha256"],
        "Exact bound whole-waveform hidden payload required",
    )
    hidden = np.load(path, allow_pickle=False)
    speech.validate_hidden(hidden, state["proof"])
    plans = [row["plan"] for row in contexts if row["status"] == "supported"]
    vectors = []
    for plan in plans:
        vector = articulation.speech_summary(hidden, plan)
        require(
            vector.tobytes() == replay.speech_replay(hidden, plan).tobytes(),
            "Independent speech summary arithmetic differs",
        )
        vectors.append(vector)
    return {
        **save_array(folder / f"speech-{name}.npy", vectors, 3840, np.float64),
        "waveform_id": name,
        "state_file_sha256": state["hidden_file_sha256"],
        "pcm_sha256": state["waveform"]["pcm_sha256"],
        "indices": [p["video_frame_index"] for p in plans],
        "plans": plans,
        "contexts": contexts,
        "scope": "Indexed output-time support. The hidden states may depend on the whole original or delayed waveform.",
    }


def same_observation(first, repeated):
    a, x = first
    b, y = repeated
    same = json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)
    if x is None or y is None:
        return same and x is y is None
    return same and x.dtype == y.dtype and x.shape == y.shape and x.tobytes() == y.tobytes()


def observe_picture(rgb, memo, observe, *, fresh=False):
    """A dictionary key checks shape and exact pixel bytes, even on hash collisions."""
    key = (rgb.shape, rgb.tobytes())
    if not fresh and key in memo:
        return memo[key], True
    observation = observe(rgb)
    if key in memo:
        require(
            same_observation(memo[key], observation),
            "Fixed original picture differs from its identical cached RGB observation",
        )
    memo[key] = observation
    return observation, False


def input_binding(ordinal):
    source = pilot(ordinal)
    construction, speech = upstream()
    cases = construction.load_completed(ordinal)
    stage = construction.TRIAL / f"clip-{ordinal:02d}/bind-00"
    result = read(stage / "output/result.json")
    journal = construction.CACHE / f"clip-{ordinal:02d}/cases.jsonl"
    require(
        len(cases) == 11 and result["private_case_journal_sha256"] == sha(journal),
        "Complete actual constructor journal required",
    )
    names = {case["waveform_id"] for case in cases if case["status"] == "ready"}
    require(
        names.issubset(speech.WAVEFORMS), "Only constructed original/delay whole-waveform identities required"
    )
    states = {name: speech.source_record(ordinal, name) for name in sorted(names)}
    require(
        all(
            state["status"] == "encoded" and state["whole_input_encoded"] is True for state in states.values()
        ),
        "Every ready soundtrack requires its completed whole-waveform states",
    )
    paths = [stage / "output/result.json", stage / "profile.json", stage / "resources.jsonl", journal]
    for batch in sorted({r["batch"] for r in states.values()}):
        s = speech.TRIAL / f"batch-{batch:02d}/authored"
        paths.extend(
            [
                s / "output/result.json",
                s / "profile.json",
                s / "resources.jsonl",
                s / "output/source-states.jsonl",
            ]
        )
    return source, cases, states, {str(p): sha(p) for p in paths}, sha(journal)


def extract(ordinal, output):
    construction, speech = upstream()
    source, cases, states, bindings, journal_sha = input_binding(ordinal)
    features, replay = feature_modules()
    folder = CACHE / f"clip-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(output / "private-input-binding.json", {"clip_ordinal": ordinal, "bindings": bindings})
    memo, audio, records = {}, {}, []
    events, active = Counter(), {}
    encoder = detector = None
    handles = []
    repeat = {
        "frame_index": REPEAT_FRAME,
        "bitwise_equal": None,
        "reason": "Fixed original picture/mouth unavailable",
    }
    hits = assessed = native_pictures = 0

    with (
        (output / "forward-progress.jsonl").open("x") as progress,
        (folder / "feature-cases.jsonl").open("x") as journal,
    ):

        def event(name):
            events[name] += 1
            progress.write(json.dumps({"event": name, **active}, allow_nan=False) + "\n")
            progress.flush()

        def observe(rgb):
            nonlocal encoder, detector
            if encoder is None:
                require(
                    features.model_records() == read(features.PUBLIC)["model_records"],
                    "Exact commercial model bytes required",
                )
                event("encoder_load_begin")
                encoder, detector = features.Encoder(), features.detector()
                require(
                    encoder.tensor_count == 223 and encoder.device == "mps",
                    "Strict qualified MPS float32 encoder required",
                )
                handles.extend(
                    [
                        encoder.model.register_forward_pre_hook(lambda *_: event("image_begin")),
                        encoder.model.register_forward_hook(lambda *_: event("image_end")),
                    ]
                )
                event("encoder_load_end")
            event("face_begin")
            meta, vector = features.detect(rgb, detector, encoder)
            event("face_end")
            if vector is not None:
                replay.pixel_replay(rgb, meta, vector)
            return meta, vector

        try:
            for case in cases:
                record = {
                    "asset_index": ordinal,
                    "source_group": source["source_slot"],
                    "component_index": source["component_index"],
                    "study_role": "pilot",
                    "condition": case["condition"],
                    "input_status": case["status"],
                    "waveform_id": case["waveform_id"],
                    "constructor_case_journal_sha256": journal_sha,
                }
                if case["status"] != "ready":
                    record.update(status="unavailable", reason=case["reason"])
                else:
                    media = case_inputs(case)
                    state = states[media["waveform_id"]]
                    table, frames, audio_origin = verified_case(
                        media, source, state, construction.CACHE / f"clip-{ordinal:02d}", features, speech
                    )
                    native_audio = source["input"]["native_audio_clock"]
                    allowance = exact(native_audio["time_base"]) + Fraction(1, native_audio["sample_rate"])
                    contexts = speech_contexts(table, audio_origin, state["waveform"]["samples"], allowance)
                    name = media["waveform_id"]
                    if name not in audio:
                        audio[name] = speech_features(folder, state, contexts, speech, replay)
                    require(
                        audio[name]["contexts"] == contexts,
                        "Reused speech summary requires identical verified output-time support",
                    )
                    frame_rows, vectors = [], []
                    for frame in frames:
                        active.update(case_number=case["case_number"], frame_index=frame.index, repeat=False)
                        fixed = case["condition"] == "original" and frame.index == REPEAT_FRAME
                        observation, hit = observe_picture(frame.rgb, memo, observe, fresh=fixed)
                        meta, vector = observation
                        hits += hit
                        row = {
                            **meta,
                            "frame_index": frame.index,
                            "pts": frame.pts,
                            "rgb_sha256": fingerprint(frame.rgb),
                            "feature_row": None,
                            "exact_rgb_cache_hit": hit,
                        }
                        if vector is not None:
                            row["feature_row"] = len(vectors)
                            vectors.append(vector)
                        frame_rows.append(row)
                        if fixed and vector is not None:
                            active["repeat"] = True
                            repeated = observe(frame.rgb.copy())
                            equal = same_observation(observation, repeated)
                            repeat = {
                                "frame_index": REPEAT_FRAME,
                                "bitwise_equal": equal,
                                "first": meta,
                                "repeated": repeated[0],
                            }
                            write(output / "fixed-repeat.json", repeat)
                            require(equal, "Available fixed original face/crop/input/vector repeat differs")
                    array = save_array(
                        folder / f"case-{case['case_number']:02d}.npy", vectors, 768, np.float32
                    )
                    array["file_sha256"] = array.pop("sha256")
                    native_pictures += len(frame_rows)
                    assessed += len(vectors)
                    record.update(
                        **array,
                        status="extracted",
                        frames=frame_rows,
                        table=table.to_json(),
                        speech=audio[name],
                        source_clip_sha256=media["file_sha256"],
                        sampling="all_native_pictures",
                        constructed_last_picture_duration_ms=media["expected"][
                            "constructed_last_picture_duration_ms"
                        ],
                        terminal_duration_provenance="Declared by the lossless mux. Native picture PTS were independently replayed.",
                    )
                records.append(record)
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
                print(
                    json.dumps({"cases_recorded": len(records), "native_pictures": native_pictures}),
                    flush=True,
                )
        finally:
            for handle in handles:
                handle.remove()

    repeat_count = int(repeat["bitwise_equal"] is True)
    if not repeat_count:
        write(output / "fixed-repeat.json", repeat)
    image_inputs = 0 if encoder is None else encoder.image_inputs
    require(
        len(records) == 11
        and events["face_begin"] == events["face_end"] == native_pictures - hits + repeat_count
        and events["image_begin"] == events["image_end"] == image_inputs,
        "Observed picture/cache/forward coverage differs",
    )
    result = {
        "schema": "chipmunk.grid-native-mouth-features-result/150",
        "status": "complete",
        "clip_ordinal": ordinal,
        "planned_cases": 11,
        "extracted_cases": sum(r["status"] == "extracted" for r in records),
        "unavailable_cases": sum(r["status"] == "unavailable" for r in records),
        "native_pictures": native_pictures,
        "assessed_mouth_pictures": assessed,
        "exact_rgb_cache_hits": hits,
        "face_inputs": events["face_begin"],
        "image_inputs": image_inputs,
        "new_neural_calls": events["face_begin"] + image_inputs,
        "new_speech_encoder_calls": 0,
        "strict_encoder_tensor_count": 0 if encoder is None else encoder.tensor_count,
        "fixed_original_repeat_bitwise_equal": repeat["bitwise_equal"],
        "speech_supported_positions": sum(len(r.get("speech", {}).get("indices", [])) for r in records),
        "speech_unavailable_positions": sum(
            75 - len(r["speech"]["indices"]) for r in records if r["status"] == "extracted"
        ),
        "private_feature_case_journal_sha256": sha(folder / "feature-cases.jsonl"),
        "private_input_binding_sha256": sha(output / "private-input-binding.json"),
        "private_forward_progress_sha256": sha(output / "forward-progress.jsonl"),
        "private_fixed_repeat_sha256": sha(output / "fixed-repeat.json"),
        "constructor_case_journal_sha256": journal_sha,
        "freeze_sha256": sha(PUBLIC),
        "defaults_changed": False,
        "scope": "Complete native pictures and indexed actual soundtrack features. Whole-waveform speech dependency remains possible. Detector scoring and achieved synchronization are outside this feature stage.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def load_completed(ordinal):
    """Verify actual completion metadata. The scoring worker opens the arrays."""
    verify()
    source, cases, states, bindings, journal_sha = input_binding(ordinal)
    _, speech = upstream()
    stage = TRIAL / f"clip-{ordinal:02d}/authored"
    out, folder = stage / "output", CACHE / f"clip-{ordinal:02d}"
    result = read(out / "result.json")
    speech.qualification_profile(
        stage, result["new_neural_calls"], {}, caller="grid_native_mouth_features_v150.py"
    )
    require(
        result["schema"] == "chipmunk.grid-native-mouth-features-result/150"
        and result["clip_ordinal"] == ordinal
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["constructor_case_journal_sha256"] == journal_sha
        and result["private_feature_case_journal_sha256"] == sha(folder / "feature-cases.jsonl")
        and read(out / "private-input-binding.json") == {"clip_ordinal": ordinal, "bindings": bindings},
        "Actual complete clip feature inventory required",
    )
    for name, key in (
        ("private-input-binding.json", "private_input_binding_sha256"),
        ("forward-progress.jsonl", "private_forward_progress_sha256"),
        ("fixed-repeat.json", "private_fixed_repeat_sha256"),
    ):
        require(sha(out / name) == result[key], "Feature worker journal changed")
    rows = planning.reviewed.lines(folder / "feature-cases.jsonl")
    validate_inventory(rows, cases, source, states, folder, journal_sha)
    repeat = read(out / "fixed-repeat.json")
    original = rows[0]
    require(original["condition"] == "original", "Fixed original repeat requires the original case")
    repeat_expected = original["status"] == "extracted" and original["frames"][REPEAT_FRAME]["status"] == "ok"
    events = Counter(r["event"] for r in planning.reviewed.lines(out / "forward-progress.jsonl"))
    extracted = [r for r in rows if r["status"] == "extracted"]
    require(
        len(extracted) == result["extracted_cases"]
        and len(rows) - len(extracted) == result["unavailable_cases"]
        and result["native_pictures"] == 75 * len(extracted)
        and result["assessed_mouth_pictures"] == sum(r["shape"][0] for r in extracted)
        and result["exact_rgb_cache_hits"]
        == sum(f["exact_rgb_cache_hit"] for r in extracted for f in r["frames"])
        and repeat["frame_index"] == REPEAT_FRAME
        and repeat["bitwise_equal"]
        is result["fixed_original_repeat_bitwise_equal"]
        is (True if repeat_expected else None)
        and events["face_begin"]
        == events["face_end"]
        == result["face_inputs"]
        == result["native_pictures"] - result["exact_rgb_cache_hits"] + int(repeat_expected)
        and events["image_begin"] == events["image_end"] == result["image_inputs"]
        and result["new_neural_calls"] == result["face_inputs"] + result["image_inputs"]
        and result["new_speech_encoder_calls"] == 0,
        "Actual feature/repeat/coverage counts differ",
    )
    return rows


def validate_inventory(rows, cases, source, states, folder, journal_sha):
    from app.detectors.video.decode import FrameTable

    def owned(path, name):
        file = Path(path)
        require(
            file.is_absolute()
            and file.resolve() == (folder / name).resolve()
            and file.resolve().is_relative_to(folder.resolve()),
            "Exact owned feature-array reference required",
        )

    require(len(rows) == len(cases) == 11, "Every fixed case, including unavailable, is required")
    for row, case in zip(rows, cases, strict=True):
        require(
            row["asset_index"] == source["ordinal"]
            and row["source_group"] == source["source_slot"]
            and row["component_index"] == source["component_index"]
            and row["study_role"] == "pilot"
            and row["condition"] == case["condition"]
            and row["input_status"] == case["status"]
            and row["waveform_id"] == case["waveform_id"]
            and row["constructor_case_journal_sha256"] == journal_sha,
            "Feature record lost its exact constructor identity",
        )
        if case["status"] != "ready":
            require(
                row["status"] == "unavailable"
                and row["reason"] == case["reason"]
                and not {"file", "frames", "speech"}.intersection(row),
                "Unavailable case cannot advertise numerical features",
            )
            continue
        require(
            row["status"] == "extracted"
            and row["source_clip_sha256"] == case["file_sha256"]
            and row["constructed_last_picture_duration_ms"]
            == case["expected"]["constructed_last_picture_duration_ms"],
            "Exact verified clip and terminal-duration declaration required",
        )
        table = FrameTable.from_json(row["table"])
        verify_table(table, case_inputs(case), source)
        owned(row["file"], f"case-{case['case_number']:02d}.npy")
        require(
            [f["frame_index"] for f in row["frames"]] == list(FRAMES),
            "All native feature picture rows required",
        )
        used = []
        for frame in row["frames"]:
            i = frame["frame_index"]
            require(
                frame["pts"] == table.pts[i]
                and frame["rgb_sha256"] == case["expected"]["rgb_sha256"][i]
                and frame["status"] in {"ok", "unassessed"}
                and type(frame["exact_rgb_cache_hit"]) is bool,
                "Feature picture lost decoded clock/RGB/coverage binding",
            )
            if frame["status"] == "ok":
                used.append(frame["feature_row"])
            else:
                require(frame["feature_row"] is None, "Unassessed picture cannot invent an array row")
        require(
            used == list(range(len(used))) and row["shape"] == [len(used), 768] and row["dtype"] == "float32",
            "Exact observed visual rows required",
        )
        name, audio = row["waveform_id"], row["speech"]
        state = states[name]
        native = source["input"]["native_audio_clock"]
        contexts = speech_contexts(
            table,
            exact(state["waveform"]["audio_origin_seconds"]),
            state["waveform"]["samples"],
            exact(native["time_base"]) + Fraction(1, native["sample_rate"]),
        )
        plans = [r["plan"] for r in contexts if r["status"] == "supported"]
        owned(audio["file"], f"speech-{name}.npy")
        require(
            audio["waveform_id"] == name
            and audio["state_file_sha256"] == state["hidden_file_sha256"]
            and audio["pcm_sha256"] == state["waveform"]["pcm_sha256"]
            and audio["contexts"] == contexts
            and audio["plans"] == plans
            and audio["indices"] == [p["video_frame_index"] for p in plans]
            and audio["shape"] == [len(plans), 3840]
            and audio["dtype"] == "float64",
            "Exact output-clock speech scope required",
        )


def prerequisite(ordinal):
    construction, _ = upstream()
    stage = construction.TRIAL / f"clip-{ordinal:02d}/bind-00"
    if not all(p.is_file() for p in (stage / "output/result.json", stage / "profile.json")):
        return False
    input_binding(ordinal)
    return True


def routed_popen(argv, ordinal, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"] and len(argv) == 6,
        "Only the fixed feature worker invocation may be routed",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_native_mouth_features_v150", *argv[4:], str(ordinal)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    pilot(ordinal)
    folder = TRIAL / f"clip-{ordinal:02d}"
    folder.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kwargs: routed_popen(argv, ordinal, **kwargs),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": folder,
            "ARCHIVE": ARCHIVE,
            "verify": verify,
            "prerequisites": lambda c, b: prerequisite(ordinal),
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "authored"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One original pilot ordinal required")
        run(int(args.arguments[0]))
    else:
        require(len(args.arguments) == 2, "Exact guarded output and ordinal required")
        output, ordinal = Path(args.arguments[0]), int(args.arguments[1])
        verify()
        pilot(ordinal)
        require(
            output.resolve() == (TRIAL / f"clip-{ordinal:02d}/authored/output").resolve(),
            "Exact exclusive feature output required",
        )
        output.mkdir(exist_ok=False)
        extract(ordinal, output)
