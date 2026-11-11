"""Whole original/delayed speech states for fixed pilot clips; no detector fit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import types
import wave
from pathlib import Path

from scripts import grid_transfer_plan_v143 as planning

ROOT, BASE, PANEL, ARCHIVE = planning.ROOT, planning.BASE, planning.PANEL, planning.ARCHIVE
TRIAL = BASE / "grid-source-speech-v147"
CACHE = ARCHIVE / "grid-source-speech-v147"
PUBLIC = PANEL / "grid-source-speech-freeze-v147.json"
WAVEFORMS = ("original_pcm", "delayed_audio_control")
SHAPE = (1500, 5, 384)
CODE = (
    "backend/scripts/grid_source_speech_v147.py",
    "backend/tests/test_grid_source_speech_v147.py",
    "docs/grid-source-speech-v147-method.md",
    "backend/scripts/grid_transfer_plan_v143.py",
    "backend/scripts/review_grid_native_inputs_v142.py",
    "backend/scripts/av_speech_native_v110.py",
    "backend/scripts/av_source_encoder_v114.py",
    "backend/scripts/av_conditioning_clock_v109.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/audio_tree_sampler_v64.py",
)
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def batches(plan):
    pilot = [s for s in plan["sources"] if s["role"] == "pilot"]
    require(
        len(pilot) == 16 and [s["ordinal"] for s in pilot] == sorted({s["ordinal"] for s in pilot}),
        "All16 fixed ordered pilot clips required; no reserve or quality substitution",
    )
    return [pilot[i : i + 2] for i in range(0, 16, 2)]


def planned_inputs(sources):
    require(len(sources) == 2, "Two fixed pilot clips required")
    rows = []
    for source in sources:
        require(
            source["role"] == "pilot" and [r["id"] for r in source["encoder_inputs"]] == list(WAVEFORMS),
            "Exact original/delay whole-waveform specifications required",
        )
        for request in source["encoder_inputs"]:
            require(
                request["status"] in {"planned", "unavailable"} and request["whole_input_required"] is True,
                "Unavailable scope must remain explicit; only whole-input encoding permitted",
            )
            rows.append(
                {
                    "source_ordinal": source["ordinal"],
                    "waveform_id": request["id"],
                    "status": request["status"],
                    "unavailable_reason": request["unavailable_reason"],
                }
            )
    return rows


def expected_forwards(requests):
    require(
        len(requests) == 4 and [r["waveform_id"] for r in requests] == list(WAVEFORMS) * 2,
        "Both whole waveforms from both fixed clips required",
    )
    return sum(r["status"] == "planned" for r in requests) + (requests[0]["status"] == "planned")


def pcm_lattice(pcm):
    import numpy as np

    require(
        isinstance(pcm, np.ndarray)
        and pcm.ndim == 1
        and pcm.dtype == np.dtype("<f4")
        and 640 <= len(pcm) <= 160000
        and np.isfinite(pcm).all(),
        "Complete finite canonical float32 waveform required",
    )
    scaled = pcm * np.float32(32768)
    require(
        np.equal(scaled, np.rint(scaled)).all() and (scaled >= -32768).all() and (scaled <= 32767).all(),
        "Exact signed16 lattice required; no gain normalization or clipping",
    )
    raw = scaled.astype("<i2").tobytes()
    require(
        (np.frombuffer(raw, dtype="<i2").astype("<f4") / np.float32(32768)).tobytes() == pcm.tobytes(),
        "Canonical signed16 float32 bit pattern required",
    )
    return raw


def delayed_pcm(pcm, support):
    import numpy as np

    pcm_lattice(pcm)
    require(
        support["status"] == "ready"
        and support["shift_samples"] == 5120
        and support["whole_input_reencoding_required"] is True
        and support["preserve_length_origins_and_outside_samples"] is True
        and support["padding_or_wraparound"] is False,
        "Exact fixed source audio-delay specification required",
    )
    a, b = support["output_samples_half_open"]
    c, d = support["source_samples_half_open"]
    require(
        all(type(v) is int for v in (a, b, c, d))
        and 0 <= c < d < b <= len(pcm)
        and a == c + 5120
        and b == d + 5120
        and a < b,
        "Bounded observed earlier-copy intervals required",
    )
    output = pcm.copy()
    output[a:b] = pcm[c:d]
    pcm_lattice(output)
    require(
        output.shape == pcm.shape
        and output.dtype == pcm.dtype
        and output[:a].tobytes() == pcm[:a].tobytes()
        and output[b:].tobytes() == pcm[b:].tobytes()
        and output[a:b].tobytes() == pcm[c:d].tobytes(),
        "Audio delay changed unsupported samples or length",
    )
    return output, int(np.count_nonzero(output != pcm))


def provenance(mapping, waveform_id, support):
    """Output clock positions remain fixed; value provenance may come from earlier samples."""
    n = mapping["samples"]
    source_ranges = planning.reviewed.mapping_ranges(mapping, n)
    require(source_ranges, "Complete original source mapping required")
    pieces = [(0, n, 0, n)]
    if waveform_id == "delayed_audio_control":
        a, b = support["output_samples_half_open"]
        c, d = support["source_samples_half_open"]
        pieces = [(0, a, 0, a), (a, b, c, d), (b, n, b, n)]
    require(waveform_id in WAVEFORMS, "Only original/delay whole waveforms permitted")
    origin = source_ranges[0]["source_start_pts"]
    return [
        {
            "output_samples_half_open": [a, b],
            "input_samples_half_open": [c, d],
            "input_source_pts_half_open": [origin + c, origin + d],
            "time_base": "1/16000",
        }
        for a, b, c, d in pieces
        if a < b
    ]


def save_waveform(folder, waveform_id, pcm, source, changed_samples):
    import numpy as np

    raw = pcm_lattice(pcm)
    array, wav = folder / f"{waveform_id}.pcm.npy", folder / f"{waveform_id}.wav"
    with array.open("xb") as stream:
        np.save(stream, pcm, allow_pickle=False)
    with wav.open("xb") as stream, wave.open(stream, "wb") as output:
        output.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        output.writeframes(raw)
    with wave.open(str(wav), "rb") as saved:
        require(
            (
                saved.getnchannels(),
                saved.getsampwidth(),
                saved.getframerate(),
                saved.getnframes(),
                saved.getcomptype(),
            )
            == (1, 2, 16000, len(pcm), "NONE"),
            "Derived full WAV header differs",
        )
        recovered = saved.readframes(saved.getnframes() + 1)
    roundtrip = np.frombuffer(recovered, dtype="<i2").astype("<f4") / np.float32(32768)
    restored = np.load(array, mmap_mode="r", allow_pickle=False)
    require(
        recovered == raw
        and restored.dtype == pcm.dtype
        and restored.shape == pcm.shape
        and restored.tobytes() == roundtrip.tobytes() == pcm.tobytes(),
        "Derived whole WAV/NPY roundtrip differs",
    )
    mapping = source["input"]["pcm_mapping"]
    return {
        "id": waveform_id,
        "pcm_file": str(array),
        "pcm_file_sha256": sha(array),
        "wav_file": str(wav),
        "wav_file_sha256": sha(wav),
        "samples": len(pcm),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "changed_samples": changed_samples,
        "original_pcm_mapping": copy.deepcopy(mapping),
        "output_clock_mapping": {
            "samples": len(pcm),
            "sample_rate": 16000,
            "channels": 1,
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
            "timeline_time_base": "1/16000",
            "timeline": copy.deepcopy(mapping["timeline"]),
            "meaning": "Output clock retained; modified values are not claimed to be the original soundtrack.",
        },
        "input_output_sample_mapping": provenance(mapping, waveform_id, source["audio_delay"]),
        "audio_origin_seconds": source["input"]["audio_origin_seconds"],
        "whole_input_reencoded": True,
        "synthetic_padding_or_wraparound": False,
    }


def validate_hidden(hidden, proof):
    import numpy as np

    require(
        isinstance(hidden, np.ndarray)
        and hidden.shape == SHAPE
        and hidden.dtype == np.float32
        and np.isfinite(hidden).all()
        and proof["layer_readout_matches"] is True
        and proof["full_hidden_sha256"] == hashlib.sha256(np.ascontiguousarray(hidden).tobytes()).hexdigest(),
        "Full finite1500x5x384 CPUfloat32 native states/proof required",
    )


def audit_chunks(hidden, frames):
    """Independent explicit10tick×5layer copy; never shift old global states."""
    import numpy as np

    require(
        hidden.shape == SHAPE
        and hidden.dtype == np.float32
        and np.isfinite(hidden).all()
        and [f["frame_index"] for f in frames] == list(range(*planning.EDIT)),
        "Exact20 output frames/full native hidden array required",
    )
    records = []
    for frame in frames:
        for key in ("source_time_conditioning", "earlier_conditioning"):
            selected = frame[key]
            entry = {
                "frame_index": frame["frame_index"],
                "view": key,
                "status": "unavailable",
                "unavailable_reason": frame["unavailable_reason"],
            }
            if frame["status"] == "ready":
                require(
                    selected["video_frame_index"] == frame["frame_index"]
                    and selected["source_sample_status"] == "supported"
                    and all(selected["retained_tick_slots"])
                    and len(selected["tick_indices"]) == 10,
                    "Full observed source-time clock routing required",
                )
                expected = np.empty((50, 384), np.float32)
                for slot, tick in enumerate(selected["tick_indices"]):
                    require(
                        0 <= tick < selected["retained_tick_prefix"] <= 1500,
                        "No missing encoder tick may be substituted",
                    )
                    for layer in range(5):
                        expected[slot * 5 + layer] = hidden[tick, layer]
                actual = planning.conditioning.select_chunk(hidden, selected)
                require(
                    actual.dtype == np.float32
                    and actual.shape == (50, 384)
                    and actual.tobytes() == expected.tobytes(),
                    "Explicit tick-major/layer-major copy differs from qualified v109 selector",
                )
                entry.update(
                    status="verified",
                    unavailable_reason=None,
                    chunk_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                    conditioning_plan=copy.deepcopy(selected),
                )
            records.append(entry)
    require(len(records) == 40, "All40 source/earlier routing observations retained")
    return records


def save_hidden(folder, name, hidden, proof):
    import numpy as np

    validate_hidden(hidden, proof)
    file = folder / f"{name}.hidden.npy"
    with file.open("xb") as stream:
        np.save(stream, hidden, allow_pickle=False)
    restored = np.load(file, mmap_mode="r", allow_pickle=False)
    require(
        restored.shape == SHAPE and restored.dtype == np.float32 and restored.tobytes() == hidden.tobytes(),
        "Complete saved hidden states differ",
    )
    return {
        "hidden_file": str(file),
        "hidden_file_sha256": sha(file),
        "hidden_shape": list(SHAPE),
        "hidden_dtype": "float32",
        "proof": copy.deepcopy(proof),
    }


def qualification_profile(folder, expected_calls, bindings, caller=None, command="authored", batch=None):
    result = planning.bind_metadata(folder / "output/result.json", bindings)
    profile = planning.bind_metadata(folder / "profile.json", bindings)
    planning.bind_metadata(folder / "resources.jsonl", bindings)
    require(
        result["status"] == "complete"
        and result["new_neural_calls"] == expected_calls
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == []
        and profile["resource_limit_observations"] == []
        and profile["command"] == command
        and profile["new_neural_calls"] == expected_calls
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
        and profile["resource_journal_sha256"] == sha(folder / "resources.jsonl"),
        "Actual successful qualified encoder resource/receipt required",
    )
    if caller is not None:
        require(
            profile["schema"] == "chipmunk.video-appearance-resource-profile/108"
            and profile["batch"] == batch
            and profile["runtime_sha256"] == sha(ROOT / "backend/scripts" / caller),
            "Exact unchanged scoped v108 monitor/caller required",
        )
    else:
        require(
            profile["schema"] == "chipmunk.av-speech-resource-profile/110",
            "Exact original speech qualification profile required",
        )
    resources = planning.reviewed.lines(folder / "resources.jsonl")
    require(resources, "Actual sampled resource observations required")
    for i, row in enumerate(resources):
        require(
            type(row["elapsed_s"]) in (int, float)
            and math.isfinite(row["elapsed_s"])
            and 0 <= row["elapsed_s"] <= 900
            and (i == 0 or resources[i - 1]["elapsed_s"] <= row["elapsed_s"])
            and all(
                type(row[k]) is int and row[k] >= 0
                for k in (
                    "tree_rss_bytes",
                    "ram_available_bytes",
                    "disk_free_bytes",
                    "archive_free_disk_bytes",
                )
            )
            and not row["violations"]
            and row["tree_rss_bytes"] <= (1.5 if command == "monitor" else 6) * 1024**3
            and row["ram_available_bytes"]
            >= ((3 if command == "monitor" else 6) if i == 0 else 1.5) * 1024**3
            and all(
                row[k] >= (3 if i == 0 else 1.5) * 1024**3
                for k in ("disk_free_bytes", "archive_free_disk_bytes")
            ),
            "Unchanged successful neural/data resource sample bounds required",
        )
    require(
        type(profile["duration_s"]) in (int, float)
        and math.isfinite(profile["duration_s"])
        and resources[-1]["elapsed_s"] <= profile["duration_s"] <= 900
        and profile["minimum_available_ram_bytes"] == min(r["ram_available_bytes"] for r in resources)
        and profile["sampled_tree_rss_lower_bound_bytes"] == max(r["tree_rss_bytes"] for r in resources),
        "Actual encoder resource profile differs from its journal",
    )
    return result


def dependencies():
    plan = planning.load_completed()
    bindings = {}
    for file in (
        planning.PUBLIC,
        planning.TRIAL / "bind-00/output/result.json",
        planning.TRIAL / "bind-00/profile.json",
        planning.TRIAL / "bind-00/output/private-plan.json",
    ):
        planning.bind_metadata(file, bindings)
    batches(plan)
    original = BASE / "av-speech-native-v110"
    native = planning.frozen_context(original, PANEL / "av-speech-native-freeze-v110.json", bindings)
    source = BASE / "av-source-encoder-v114"
    frozen = planning.frozen_context(source, PANEL / "av-source-encoder-freeze-v114.json", bindings)
    require(
        frozen["speech_contract_freeze_sha256"] == sha(original / "freeze.json"),
        "Unchanged qualified v114 native encoder contract required",
    )
    for name, version in native["packages"].items():
        require(
            importlib.metadata.version(name) == version, "Exact qualified speech package versions required"
        )
    for name, digest in native["library_source"]["sha256"].items():
        path = Path(importlib.metadata.distribution(name.split("/", 1)[0]).locate_file(name))
        require(sha(path) == digest, "Exact installed encoder/feature source bytes required")
    manifest_path = PANEL / "mouth-v7-generator-assets.json"
    manifest = planning.bind_metadata(manifest_path, bindings)
    acquired = planning.bind_metadata(BASE / "lipsync-v7/assets/verified-assets.json", bindings)
    speech = [r for r in manifest["assets"] if r["role"] == "speech"]
    acquired_by_path = {r["path"]: r for r in acquired["assets"]}
    expected = [acquired_by_path[r["path"]] for r in speech]
    require(
        len(speech) == 3
        and all(r["license"] == "apache-2.0" for r in speech)
        and native["speech_assets"] == expected
        and acquired["manifest_sha256"] == sha(manifest_path)
        and all(
            r["bytes"] == a["bytes"] and (r["sha256"] is None or r["sha256"] == a["sha256"])
            for r, a in zip(speech, expected, strict=True)
        ),
        "Exact three permitted speech asset manifests required",
    )
    # Two small JSON configurations are pinned by publisher Git blob IDs rather
    # than LFS SHA256. Bind their real metadata bytes; never hash the binary here.
    for declared, actual in zip(speech, expected, strict=True):
        if declared["sha256"] is None:
            file = BASE / "lipsync-v7/assets" / declared["path"]
            require(file.suffix == ".json", "Only small configuration metadata may be read before the guard")
            planning.bind_metadata(file, bindings)
            body = file.read_bytes()
            require(
                len(body) == declared["bytes"]
                and sha(file) == actual["sha256"]
                and hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
                == declared["git_blob"],
                "Publisher Git-blob configuration metadata differs from actual qualified receipt",
            )
    monitor = qualification_profile(original / "monitor", 0, bindings, command="monitor")
    authored = qualification_profile(original / "authored", 3, bindings)
    require(
        monitor["owned_descendant_sampled"] is True
        and monitor["sample_helper_sha256"] == sha(ROOT / "backend/scripts/audio_tree_sampler_v64.py")
        and authored["native_repeat_bitwise"] is True
        and authored["native_layer_readout_matches"] is True
        and authored["legacy_25hz_chunks_equal"] is True
        and authored["hidden_shape"] == list(SHAPE)
        and authored["freeze_sha256"] == sha(original / "freeze.json")
        and frozen["qualified_authored_result_sha256"] == sha(original / "authored/output/result.json")
        and frozen["qualified_monitor_result_sha256"] == sha(original / "monitor/output/result.json"),
        "Actual unchanged native encoder authored/source qualification required",
    )
    for batch in range(8):
        folder = source / f"extract-{batch:02d}"
        receipt = qualification_profile(
            folder, 7, bindings, caller="av_source_encoder_v114.py", command="extract", batch=batch
        )
        journal = folder / "output/source-states.jsonl"
        planning.bind_metadata(journal, bindings)
        require(
            receipt["freeze_sha256"] == sha(source / "freeze.json")
            and receipt["batch"] == batch
            and receipt["sources"] == 6
            and receipt["source_journal_sha256"] == sha(journal),
            "Actual complete retained v114 source execution required",
        )
    return plan, bindings


def identity():
    plan, bindings = dependencies()
    groups = batches(plan)
    return {
        "schema": "chipmunk.grid-source-speech-freeze/147",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "completed_metadata_sha256": bindings,
        "pilot_clips": 16,
        "batches": 8,
        "clips_per_batch": 2,
        "waveforms_per_clip": list(WAVEFORMS),
        "fixed_repeat": "Original waveform of first ordered clip in each batch; never replace unavailable repeat",
        "planned_inputs": [planned_inputs(s) for s in groups],
        "fully_available_batch_encoder_forwards": 5,
        "fully_available_total_encoder_forwards": 40,
        "hidden_shape": list(SHAPE),
        "scope": "Whole original and local-delay waveforms encoded independently with unchanged qualified CPUfloat32 NativeEncoder; source-clock routing, not local evidence or detector accuracy.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","pilot_clips":16,"batches":8,"fully_available_forwards":40}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged whole-input pilot speech-state contract required",
    )


def encode(batch, output):
    import numpy as np

    from scripts.av_speech_native_v110 import NativeEncoder

    plan, _bindings = dependencies()
    sources = batches(plan)[batch]
    requests = planned_inputs(sources)
    cache = CACHE / f"batch-{batch:02d}"
    cache.mkdir(parents=True, exist_ok=False)
    records, observed, encoder, handles = [], {"started": 0, "completed": 0}, None, []
    repeat_record = {
        "source_ordinal": sources[0]["ordinal"],
        "waveform_id": "original_pcm",
        "status": "unavailable",
        "reason": "Fixed first original unavailable; no replacement",
        "bitwise_equal": None,
    }
    with (
        (output / "forward-progress.jsonl").open("x") as progress,
        (output / "source-states.jsonl").open("x") as journal,
    ):

        def event(value):
            progress.write(json.dumps(value, allow_nan=False) + "\n")
            progress.flush()

        def hook(finished):
            def callback(_module, _args, *_rest):
                key = "completed" if finished else "started"
                observed[key] += 1
                event(
                    {
                        "event": "encoder_forward_end" if finished else "encoder_forward_begin",
                        "count": observed[key],
                    }
                )

            return callback

        def forward(pcm, source_ordinal, waveform_id, repeat=False):
            nonlocal encoder
            if encoder is None:
                event({"event": "encoder_load_begin"})
                encoder = (
                    NativeEncoder()
                )  # Its unchanged verify checks actual pretrained bytes inside this guarded worker.
                handles.extend(
                    [
                        encoder.encoder.register_forward_pre_hook(hook(False)),
                        encoder.encoder.register_forward_hook(hook(True)),
                    ]
                )
                event({"event": "encoder_load_end", "native_forward_count": encoder.forward_count})
            event(
                {
                    "event": "encode_begin",
                    "source_ordinal": source_ordinal,
                    "waveform_id": waveform_id,
                    "repeat": repeat,
                    "samples": len(pcm),
                    "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
                }
            )
            hidden, proof = encoder.encode(pcm)
            validate_hidden(hidden, proof)
            event(
                {
                    "event": "encode_end",
                    "source_ordinal": source_ordinal,
                    "waveform_id": waveform_id,
                    "repeat": repeat,
                    "native_forward_count": encoder.forward_count,
                    "proof": proof,
                }
            )
            return hidden, proof

        try:
            for source in sources:
                folder = cache / f"clip-{source['ordinal']:02d}"
                folder.mkdir(exist_ok=False)
                original = None
                for request in [r for r in requests if r["source_ordinal"] == source["ordinal"]]:
                    record = {**request, "role": "pilot", "batch": batch}
                    if request["status"] == "unavailable":
                        records.append(record)
                        journal.write(json.dumps(record, allow_nan=False) + "\n")
                        journal.flush()
                        continue
                    if original is None:
                        planning.reviewed.verify_pcm(source["input"])
                        original = np.load(
                            source["input"]["pcm_file"], mmap_mode="r", allow_pickle=False
                        ).copy()
                        pcm_lattice(original)
                    waveform_id = request["waveform_id"]
                    pcm, changed = (
                        (original.copy(), 0)
                        if waveform_id == "original_pcm"
                        else delayed_pcm(original, source["audio_delay"])
                    )
                    waveform = save_waveform(folder, waveform_id, pcm, source, changed)
                    hidden, proof = forward(pcm, source["ordinal"], waveform_id)
                    record.update(
                        status="encoded",
                        unavailable_reason=None,
                        waveform=waveform,
                        **save_hidden(folder, waveform_id, hidden, proof),
                    )
                    record["routing"] = audit_chunks(hidden, source["frames"])
                    record["whole_input_encoded"] = True
                    if source["ordinal"] == sources[0]["ordinal"] and waveform_id == "original_pcm":
                        repeat, repeated_proof = forward(
                            pcm.copy(), source["ordinal"], waveform_id, repeat=True
                        )
                        repeat_record = {
                            "source_ordinal": source["ordinal"],
                            "waveform_id": waveform_id,
                            "status": "verified",
                            "reason": None,
                            "bitwise_equal": hidden.tobytes() == repeat.tobytes() and proof == repeated_proof,
                            **save_hidden(folder, "fixed-first-original-repeat", repeat, repeated_proof),
                        }
                        write(output / "fixed-repeat.json", repeat_record)
                        require(
                            repeat_record["bitwise_equal"] is True,
                            "Fixed first original whole-input repeat differs; preserve failed attempt",
                        )
                        del repeat
                    records.append(record)
                    journal.write(json.dumps(record, allow_nan=False) + "\n")
                    journal.flush()
                    del hidden, pcm
        finally:
            for handle in handles:
                handle.remove()
    if not (output / "fixed-repeat.json").exists():
        write(output / "fixed-repeat.json", repeat_record)
    expected = expected_forwards(requests)
    actual = encoder.forward_count if encoder is not None else 0
    require(
        len(records) == 4 and actual == observed["started"] == observed["completed"] == expected,
        "Exact observed planned whole-waveform/repeat encoder counts required",
    )
    require(
        repeat_record["status"] == ("verified" if requests[0]["status"] == "planned" else "unavailable"),
        "Fixed repeat must not be replaced",
    )
    result = {
        "schema": "chipmunk.grid-source-speech-result/147",
        "status": "complete",
        "batch": batch,
        "pilot_clips": 2,
        "planned_waveforms": 4,
        "encoded_waveforms": sum(r["status"] == "encoded" for r in records),
        "unavailable_waveforms": sum(r["status"] == "unavailable" for r in records),
        "routing_observations": sum(len(r.get("routing", [])) for r in records),
        "verified_routing_chunks": sum(
            q["status"] == "verified" for r in records for q in r.get("routing", [])
        ),
        "unavailable_routing_contexts": sum(
            q["status"] == "unavailable" for r in records for q in r.get("routing", [])
        ),
        "fixed_repeat_status": repeat_record["status"],
        "fixed_repeat_bitwise_equal": repeat_record["bitwise_equal"],
        "encoder_forward_counts": observed,
        "source_journal_sha256": sha(output / "source-states.jsonl"),
        "forward_progress_sha256": sha(output / "forward-progress.jsonl"),
        "fixed_repeat_sha256": sha(output / "fixed-repeat.json"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": actual,
        "defaults_changed": False,
        "scope": "Original and locally delayed WHOLE waveforms independently encoded; full states/private clock routing only. No finite dependency/local AI evidence, achieved sync, classification or confidence claim.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def validate_record(record, request, source, batch):
    """Metadata-only consumer check; actual state/media bytes belong inside its own guarded worker."""
    require(
        record["source_ordinal"] == request["source_ordinal"] == source["ordinal"]
        and record["waveform_id"] == request["waveform_id"]
        and record["role"] == "pilot"
        and record["batch"] == batch,
        "Completed state source/role/waveform correspondence differs",
    )
    if request["status"] == "unavailable":
        require(
            record["status"] == "unavailable"
            and record["unavailable_reason"] == request["unavailable_reason"]
            and "hidden_file" not in record,
            "Unavailable cannot invent state vectors",
        )
        return
    require(
        record["status"] == "encoded"
        and record["unavailable_reason"] is None
        and record["whole_input_encoded"] is True
        and record["hidden_shape"] == list(SHAPE)
        and record["hidden_dtype"] == "float32"
        and record["proof"]["layer_readout_matches"] is True,
        "Complete actual whole-input state metadata required",
    )
    name = request["waveform_id"]
    folder = CACHE / f"batch-{batch:02d}/clip-{source['ordinal']:02d}"
    waveform = record["waveform"]
    for value in (
        record["hidden_file_sha256"],
        record["proof"]["full_hidden_sha256"],
        record["proof"]["mel_sha256"],
        waveform["pcm_file_sha256"],
        waveform["wav_file_sha256"],
        waveform["pcm_sha256"],
    ):
        require(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "Exact private artifact/state commitments required",
        )
    require(
        all(
            Path(path).resolve().is_relative_to(CACHE.resolve())
            for path in (record["hidden_file"], waveform["pcm_file"], waveform["wav_file"])
        )
        and Path(record["hidden_file"]).resolve() == (folder / f"{name}.hidden.npy").resolve()
        and Path(waveform["pcm_file"]).resolve() == (folder / f"{name}.pcm.npy").resolve()
        and Path(waveform["wav_file"]).resolve() == (folder / f"{name}.wav").resolve()
        and waveform["id"] == name
        and waveform["whole_input_reencoded"] is True
        and waveform["synthetic_padding_or_wraparound"] is False
        and waveform["original_pcm_mapping"] == source["input"]["pcm_mapping"]
        and waveform["samples"] == source["input"]["pcm_mapping"]["samples"]
        and waveform["audio_origin_seconds"] == source["input"]["audio_origin_seconds"]
        and waveform["output_clock_mapping"]
        == {
            "samples": waveform["samples"],
            "sample_rate": 16000,
            "channels": 1,
            "pcm_sha256": waveform["pcm_sha256"],
            "timeline_time_base": "1/16000",
            "timeline": source["input"]["pcm_mapping"]["timeline"],
            "meaning": "Output clock retained; modified values are not claimed to be the original soundtrack.",
        }
        and waveform["input_output_sample_mapping"]
        == provenance(source["input"]["pcm_mapping"], name, source["audio_delay"]),
        "Full waveform/value-provenance metadata differs from fixed source clocks/specification",
    )
    if name == "original_pcm":
        require(
            waveform["changed_samples"] == 0
            and waveform["pcm_sha256"] == source["input"]["pcm_mapping"]["pcm_sha256"],
            "Original waveform content must remain exact",
        )
    else:
        a, b = source["audio_delay"]["output_samples_half_open"]
        require(
            type(waveform["changed_samples"]) is int and 0 <= waveform["changed_samples"] <= b - a,
            "Observed changed-sample count must retain zero-effect controls without invention",
        )
    expected_routes = [
        (f, key) for f in source["frames"] for key in ("source_time_conditioning", "earlier_conditioning")
    ]
    require(
        len(record["routing"]) == len(expected_routes) == 40, "All40 routing contexts must remain recorded"
    )
    for route, (frame, key) in zip(record["routing"], expected_routes, strict=True):
        require(
            route["frame_index"] == frame["frame_index"]
            and route["view"] == key
            and route["status"] == ("verified" if frame["status"] == "ready" else "unavailable"),
            "Exact source/earlier routing order or availability differs",
        )
        if route["status"] == "verified":
            require(
                route["conditioning_plan"] == frame[key]
                and re.fullmatch(r"[0-9a-f]{64}", route["chunk_sha256"]) is not None,
                "Native routing plan/commitment differs",
            )


def validate_progress(events, records, repeat, requests, batch):
    """Exact observation protocol binds each whole PCM and proof to one encoder forward."""
    expected = expected_forwards(requests)
    sequence = []
    for i, (request, record) in enumerate(zip(requests, records, strict=True)):
        if request["status"] == "planned":
            sequence.append((record, False, record["proof"]))
            if i == 0:
                folder = CACHE / f"batch-{batch:02d}/clip-{record['source_ordinal']:02d}"
                require(
                    repeat["status"] == "verified"
                    and repeat["bitwise_equal"] is True
                    and repeat["source_ordinal"] == record["source_ordinal"]
                    and repeat["waveform_id"] == "original_pcm"
                    and repeat["proof"] == record["proof"]
                    and repeat["hidden_shape"] == list(SHAPE)
                    and repeat["hidden_dtype"] == "float32"
                    and repeat["hidden_file_sha256"] == record["hidden_file_sha256"]
                    and Path(repeat["hidden_file"]).resolve().is_relative_to(CACHE.resolve())
                    and Path(repeat["hidden_file"]).resolve()
                    == (folder / "fixed-first-original-repeat.hidden.npy").resolve(),
                    "Actual fixed first-original repeat metadata/proof differs",
                )
                sequence.append((record, True, repeat["proof"]))
    if not sequence:
        require(events == [], "Unavailable whole inputs cannot invent load/forward events")
        return {"started": 0, "completed": 0}
    require(
        len(sequence) == expected
        and len(events) == 2 + 4 * expected
        and events[:2]
        == [{"event": "encoder_load_begin"}, {"event": "encoder_load_end", "native_forward_count": 0}],
        "Exactly one observed unchanged encoder load required",
    )
    for i, (record, is_repeat, proof) in enumerate(sequence, 1):
        start = {
            "event": "encode_begin",
            "source_ordinal": record["source_ordinal"],
            "waveform_id": record["waveform_id"],
            "repeat": is_repeat,
            "samples": record["waveform"]["samples"],
            "pcm_sha256": record["waveform"]["pcm_sha256"],
        }
        finish = {
            "event": "encode_end",
            "source_ordinal": record["source_ordinal"],
            "waveform_id": record["waveform_id"],
            "repeat": is_repeat,
            "native_forward_count": i,
            "proof": proof,
        }
        require(
            events[2 + (i - 1) * 4 : 2 + i * 4]
            == [
                start,
                {"event": "encoder_forward_begin", "count": i},
                {"event": "encoder_forward_end", "count": i},
                finish,
            ],
            "PCM/source/waveform/repeat/proof or monotonic encoder-forward protocol differs",
        )
    return {"started": expected, "completed": expected}


def load_completed(batch):
    require(type(batch) is int and 0 <= batch < 8, "One fixed pilot batch required")
    verify()
    bindings = {}
    folder = TRIAL / f"batch-{batch:02d}/authored"
    receipt = read(folder / "output/result.json")
    result = qualification_profile(
        folder, receipt["new_neural_calls"], bindings, caller="grid_source_speech_v147.py", command="authored"
    )
    journal = folder / "output/source-states.jsonl"
    progress = folder / "output/forward-progress.jsonl"
    repeated = folder / "output/fixed-repeat.json"
    require(
        result["schema"] == "chipmunk.grid-source-speech-result/147"
        and result["batch"] == batch
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["source_journal_sha256"] == sha(journal)
        and result["forward_progress_sha256"] == sha(progress)
        and result["fixed_repeat_sha256"] == sha(repeated),
        "Actual complete bound source-state/repeat/progress journals required",
    )
    records = planning.reviewed.lines(journal)
    planned = read(PUBLIC)["planned_inputs"][batch]
    require(
        len(records) == 4
        and [(r["source_ordinal"], r["waveform_id"]) for r in records]
        == [(r["source_ordinal"], r["waveform_id"]) for r in planned]
        and all(
            r["status"] == ("encoded" if p["status"] == "planned" else "unavailable")
            for r, p in zip(records, planned, strict=True)
        ),
        "Every fixed input, including unavailable, must remain in source lookup",
    )
    plan = planning.load_completed()
    sources = batches(plan)[batch]
    by_source = {source["ordinal"]: source for source in sources}
    for record, request in zip(records, planned, strict=True):
        validate_record(record, request, by_source[request["source_ordinal"]], batch)
    expected = expected_forwards(planned)
    repeat = read(repeated)
    events = planning.reviewed.lines(progress)
    counts = validate_progress(events, records, repeat, planned, batch)
    require(
        result["new_neural_calls"] == expected == counts["started"] == counts["completed"]
        and counts == result["encoder_forward_counts"]
        and repeat["source_ordinal"] == planned[0]["source_ordinal"]
        and repeat["waveform_id"] == "original_pcm"
        and repeat["status"] == ("verified" if planned[0]["status"] == "planned" else "unavailable")
        and (
            repeat["bitwise_equal"] is True
            if planned[0]["status"] == "planned"
            else repeat["bitwise_equal"] is None
        ),
        "Actual observed whole-input counts/fixed original repeat differ",
    )
    require(
        result["planned_waveforms"] == 4
        and result["pilot_clips"] == 2
        and result["defaults_changed"] is False
        and result["fixed_repeat_status"] == repeat["status"]
        and result["fixed_repeat_bitwise_equal"] == repeat["bitwise_equal"]
        and result["encoded_waveforms"] == sum(r["status"] == "encoded" for r in records)
        and result["unavailable_waveforms"] == sum(r["status"] == "unavailable" for r in records)
        and result["routing_observations"] == sum(len(r.get("routing", [])) for r in records)
        and result["verified_routing_chunks"]
        == sum(q["status"] == "verified" for r in records for q in r.get("routing", []))
        and result["unavailable_routing_contexts"]
        == sum(q["status"] == "unavailable" for r in records for q in r.get("routing", [])),
        "Observed source-state/routing totals differ",
    )
    return records


def source_record(source_ordinal, waveform_id="original_pcm"):
    require(
        type(source_ordinal) is int and waveform_id in WAVEFORMS,
        "Exact pilot source/waveform lookup required",
    )
    plan = planning.load_completed()
    for batch, sources in enumerate(batches(plan)):
        if any(s["ordinal"] == source_ordinal for s in sources):
            return next(
                r
                for r in load_completed(batch)
                if r["source_ordinal"] == source_ordinal and r["waveform_id"] == waveform_id
            )
    raise ValueError("Reserve or unknown source must not be encoded")


def prerequisites(batch):
    if batch == 0:
        return True
    folder = TRIAL / f"batch-{batch - 1:02d}/authored"
    if not all(p.is_file() for p in (folder / "output/result.json", folder / "profile.json")):
        return False
    load_completed(batch - 1)
    return True


def routed_popen(batch, argv, **kwargs):
    expected = TRIAL / f"batch-{batch:02d}/authored/output"
    require(
        type(batch) is int
        and 0 <= batch < 8
        and argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "authored"]
        and argv[5:] == [str(expected)],
        "Only exact exclusive two-clip speech stage permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_source_speech_v147", "encode", str(expected), str(batch)], **kwargs
    )


def run(batch):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    require(type(batch) is int and 0 <= batch < 8, "One fixed pilot batch required")
    stage = TRIAL / f"batch-{batch:02d}"
    stage.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kw: routed_popen(batch, argv, **kw),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": stage,
            "verify": verify,
            "prerequisites": lambda _c, _b: prerequisites(batch),
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("authored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "encode"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "No extra freeze arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One fixed pilot batch required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {str(i) for i in range(8)},
            "Exact output/batch required",
        )
        batch = int(args.arguments[1])
        output = Path(args.arguments[0])
        require(
            output.resolve() == (TRIAL / f"batch-{batch:02d}/authored/output").resolve(),
            "Exclusive guarded output required",
        )
        output.mkdir(exist_ok=False)
        encode(batch, output)
