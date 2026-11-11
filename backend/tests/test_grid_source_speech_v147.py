import copy
import hashlib

import numpy as np
import pytest

from scripts import grid_source_speech_v147 as speech


def support():
    return {
        "status": "ready",
        "shift_samples": 5120,
        "whole_input_reencoding_required": True,
        "preserve_length_origins_and_outside_samples": True,
        "padding_or_wraparound": False,
        "output_samples_half_open": [6000, 16000],
        "source_samples_half_open": [880, 10880],
    }


def pcm():
    return (np.arange(32000, dtype=np.int32) % 2048 - 1024).astype("<f4") / np.float32(32768)


def frames():
    result = []
    for index in range(27, 47):
        result.append(
            {
                "frame_index": index,
                "status": "ready",
                "unavailable_reason": None,
                "source_time_conditioning": speech.planning.conditioning.frame_plan(
                    index, f"{index}/25", "0", "0", 32000, "0"
                ),
                "earlier_conditioning": speech.planning.conditioning.frame_plan(
                    index, f"{index - 8}/25", "0", "0", 32000, "0"
                ),
            }
        )
    return result


def source(ordinal=0):
    values = pcm()
    mapped = {
        "timeline_time_base": "1/16000",
        "sample_rate": 16000,
        "channels": 1,
        "sample_format": "s16le",
        "samples": len(values),
        "timeline": [{"start_sample": 0, "end_sample": len(values), "source_pts": 160}],
        "pcm_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    return {
        "ordinal": ordinal,
        "role": "pilot",
        "input": {"pcm_mapping": mapped, "audio_origin_seconds": "1/100"},
        "audio_delay": support(),
        "frames": frames(),
        "encoder_inputs": [
            {"id": name, "status": "planned", "unavailable_reason": None, "whole_input_required": True}
            for name in speech.WAVEFORMS
        ],
    }


def hidden():
    return np.broadcast_to(np.arange(7500, dtype=np.float32).reshape(1500, 5, 1), speech.SHAPE).copy()


def proof(values):
    return {
        "mel_sha256": "a" * 64,
        "full_hidden_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "layer_readout_matches": True,
    }


def test_exact_whole_delay_copies_observed_samples_without_changing_outside_or_length():
    original = pcm()
    delayed, count = speech.delayed_pcm(original, support())
    assert delayed.shape == original.shape and delayed.dtype == np.float32
    assert delayed[6000:16000].tobytes() == original[880:10880].tobytes()
    assert delayed[:6000].tobytes() == original[:6000].tobytes()
    assert delayed[16000:].tobytes() == original[16000:].tobytes()
    assert count == int(np.count_nonzero(delayed != original)) > 0
    assert not np.shares_memory(delayed, original)


def test_zero_effect_copy_is_retained_without_replacement_or_invented_change():
    values = np.zeros(32000, dtype=np.float32)
    delayed, count = speech.delayed_pcm(values, support())
    assert count == 0 and np.array_equal(delayed, values)


@pytest.mark.parametrize("damage", ["wrap", "offset", "bounds", "float", "padding"])
def test_delay_refuses_retiming_relaxation_and_invented_source_samples(damage):
    value = support()
    if damage == "wrap":
        value["source_samples_half_open"][0] = -1
    elif damage == "offset":
        value["source_samples_half_open"][1] += 1
    elif damage == "bounds":
        value["output_samples_half_open"][1] = 40000
    elif damage == "float":
        value["output_samples_half_open"][0] = 6000.0
    else:
        value["padding_or_wraparound"] = True
    with pytest.raises(ValueError):
        speech.delayed_pcm(pcm(), value)


@pytest.mark.parametrize("damage", ["float64", "nan", "overflow", "off_lattice", "negative_zero", "short"])
def test_pcm_is_complete_canonical_signed16_float32_without_clipping(damage):
    value = pcm()
    if damage == "float64":
        value = value.astype(np.float64)
    elif damage == "nan":
        value[10] = np.nan
    elif damage == "overflow":
        value[10] = 1
    elif damage == "off_lattice":
        value[10] = 0.00001
    elif damage == "negative_zero":
        value[10] = -0.0
    else:
        value = value[:639]
    with pytest.raises(ValueError):
        speech.pcm_lattice(value)


def test_whole_waveforms_roundtrip_wav_npy_and_separate_clock_from_value_provenance(tmp_path):
    entry = source()
    original = pcm()
    first = speech.save_waveform(tmp_path, "original_pcm", original, entry, 0)
    changed, n = speech.delayed_pcm(original, support())
    second = speech.save_waveform(tmp_path, "delayed_audio_control", changed, entry, n)
    assert first["samples"] == second["samples"] == 32000
    assert first["pcm_sha256"] != second["pcm_sha256"]
    assert first["output_clock_mapping"]["timeline"] == second["output_clock_mapping"]["timeline"]
    assert second["audio_origin_seconds"] == "1/100"
    assert second["input_output_sample_mapping"] == [
        {
            "output_samples_half_open": [0, 6000],
            "input_samples_half_open": [0, 6000],
            "input_source_pts_half_open": [160, 6160],
            "time_base": "1/16000",
        },
        {
            "output_samples_half_open": [6000, 16000],
            "input_samples_half_open": [880, 10880],
            "input_source_pts_half_open": [1040, 11040],
            "time_base": "1/16000",
        },
        {
            "output_samples_half_open": [16000, 32000],
            "input_samples_half_open": [16000, 32000],
            "input_source_pts_half_open": [16160, 32160],
            "time_base": "1/16000",
        },
    ]
    assert second["whole_input_reencoded"] is True and second["synthetic_padding_or_wraparound"] is False


def test_all40_v109_chunks_use_explicit_tick_then_layer_order():
    values = hidden()
    records = speech.audit_chunks(values, frames())
    assert len(records) == 40 and all(r["status"] == "verified" for r in records)
    assert records[0]["chunk_sha256"] == hashlib.sha256(values[50:60].reshape(50, 384).tobytes()).hexdigest()
    assert records[1]["chunk_sha256"] == hashlib.sha256(values[34:44].reshape(50, 384).tobytes()).hexdigest()


def test_routing_tamper_and_unavailable_context_are_not_hidden_or_zero_substituted(monkeypatch):
    values, plans = hidden(), frames()
    plans[0].update(status="unavailable", unavailable_reason="No complete ticks")
    result = speech.audit_chunks(values, plans)
    assert len(result) == 40 and result[0]["status"] == result[1]["status"] == "unavailable"
    assert "chunk_sha256" not in result[0]
    monkeypatch.setattr(
        speech.planning.conditioning, "select_chunk", lambda *_: np.zeros((50, 384), np.float32)
    )
    with pytest.raises(ValueError, match="tick-major"):
        speech.audit_chunks(values, frames())


def test_hidden_file_readback_retains_full_array_proof(tmp_path):
    values = hidden()
    record = speech.save_hidden(tmp_path, "original_pcm", values, proof(values))
    assert record["hidden_shape"] == [1500, 5, 384]
    assert record["hidden_file_sha256"] == speech.sha(record["hidden_file"])
    wrong = values[:-1]
    with pytest.raises(ValueError):
        speech.validate_hidden(wrong, proof(wrong))


def test_two_clips_per_batch_and_fixed_repeat_never_substitute_unavailable_first():
    plan = {"sources": [{"ordinal": i, "role": "pilot" if i < 16 else "reserve"} for i in range(32)]}
    groups = speech.batches(plan)
    assert len(groups) == 8 and [s["ordinal"] for s in groups[7]] == [14, 15]
    requests = speech.planned_inputs([source(0), source(1)])
    assert speech.expected_forwards(requests) == 5
    for request in requests[:2]:
        request.update(status="unavailable", unavailable_reason="Unavailable fixed source")
    assert speech.expected_forwards(requests) == 2
    with pytest.raises(ValueError):
        speech.batches({"sources": plan["sources"][:-17]})


def protocol_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(speech, "CACHE", tmp_path)
    sources = [source(0), source(1)]
    requests = speech.planned_inputs(sources)
    records = []
    for request in requests:
        ordinal = request["source_ordinal"]
        records.append(
            {
                **request,
                "status": "encoded",
                "role": "pilot",
                "batch": 0,
                "waveform": {"samples": 32000, "pcm_sha256": str(ordinal + 1) * 64},
                "proof": {
                    "mel_sha256": "a" * 64,
                    "full_hidden_sha256": str(ordinal + 3) * 64,
                    "layer_readout_matches": True,
                },
                "hidden_file_sha256": "b" * 64,
                "hidden_shape": list(speech.SHAPE),
                "hidden_dtype": "float32",
            }
        )
    repeat = {
        "source_ordinal": 0,
        "waveform_id": "original_pcm",
        "status": "verified",
        "bitwise_equal": True,
        "hidden_file": str(tmp_path / "batch-00/clip-00/fixed-first-original-repeat.hidden.npy"),
        "hidden_file_sha256": records[0]["hidden_file_sha256"],
        "hidden_shape": list(speech.SHAPE),
        "hidden_dtype": "float32",
        "proof": copy.deepcopy(records[0]["proof"]),
    }
    order = [
        (records[0], False),
        (records[0], True),
        (records[1], False),
        (records[2], False),
        (records[3], False),
    ]
    events = [{"event": "encoder_load_begin"}, {"event": "encoder_load_end", "native_forward_count": 0}]
    for i, (record, is_repeat) in enumerate(order, 1):
        events.extend(
            [
                {
                    "event": "encode_begin",
                    "source_ordinal": record["source_ordinal"],
                    "waveform_id": record["waveform_id"],
                    "repeat": is_repeat,
                    "samples": record["waveform"]["samples"],
                    "pcm_sha256": record["waveform"]["pcm_sha256"],
                },
                {"event": "encoder_forward_begin", "count": i},
                {"event": "encoder_forward_end", "count": i},
                {
                    "event": "encode_end",
                    "source_ordinal": record["source_ordinal"],
                    "waveform_id": record["waveform_id"],
                    "repeat": is_repeat,
                    "native_forward_count": i,
                    "proof": copy.deepcopy(record["proof"]),
                },
            ]
        )
    return requests, records, repeat, events


@pytest.mark.parametrize(
    "damage", [None, "source", "pcm", "proof", "counter", "order", "missing", "repeat_path", "repeat_proof"]
)
def test_exact_load_and_forward_protocol_binds_pcm_identity_and_complete_repeat(
    tmp_path, monkeypatch, damage
):
    requests, records, repeated, events = protocol_fixture(tmp_path, monkeypatch)
    if damage is None:
        assert speech.validate_progress(events, records, repeated, requests, 0) == {
            "started": 5,
            "completed": 5,
        }
        return
    if damage == "source":
        events[2]["source_ordinal"] = 1
    elif damage == "pcm":
        events[2]["pcm_sha256"] = "c" * 64
    elif damage == "proof":
        events[5]["proof"]["mel_sha256"] = "c" * 64
    elif damage == "counter":
        events[3]["count"] = 2
    elif damage == "order":
        events[2:6], events[6:10] = events[6:10], events[2:6]
    elif damage == "missing":
        events.pop()
    elif damage == "repeat_path":
        repeated["hidden_file"] = str(tmp_path / "other-source.hidden.npy")
    else:
        repeated["proof"]["mel_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        speech.validate_progress(events, records, repeated, requests, 0)


def test_fixed_unavailable_repeat_does_not_require_or_invent_encoder_load():
    requests = speech.planned_inputs([source(0), source(1)])
    for request in requests:
        request.update(status="unavailable", unavailable_reason="No source PCM")
    records = copy.deepcopy(requests)
    assert speech.validate_progress([], records, {"status": "unavailable"}, requests, 0) == {
        "started": 0,
        "completed": 0,
    }


def test_unavailable_state_cannot_advertise_hidden_vectors():
    request = {
        "source_ordinal": 0,
        "waveform_id": "original_pcm",
        "status": "unavailable",
        "unavailable_reason": "No PCM",
    }
    record = {**request, "role": "pilot", "batch": 0}
    speech.validate_record(record, request, source(), 0)
    record["hidden_file"] = "/invented/states.npy"
    with pytest.raises(ValueError):
        speech.validate_record(record, request, source(), 0)


def encoded_metadata(tmp_path):
    entry = source()
    request = speech.planned_inputs([entry, source(1)])[0]
    mapping = entry["input"]["pcm_mapping"]
    folder = tmp_path / "batch-00/clip-00"
    record = {
        **request,
        "status": "encoded",
        "unavailable_reason": None,
        "role": "pilot",
        "batch": 0,
        "whole_input_encoded": True,
        "hidden_shape": list(speech.SHAPE),
        "hidden_dtype": "float32",
        "hidden_file": str(folder / "original_pcm.hidden.npy"),
        "hidden_file_sha256": "a" * 64,
        "proof": {"layer_readout_matches": True, "full_hidden_sha256": "b" * 64, "mel_sha256": "c" * 64},
        "waveform": {
            "id": "original_pcm",
            "pcm_file": str(folder / "original_pcm.pcm.npy"),
            "pcm_file_sha256": "d" * 64,
            "wav_file": str(folder / "original_pcm.wav"),
            "wav_file_sha256": "e" * 64,
            "pcm_sha256": mapping["pcm_sha256"],
            "samples": mapping["samples"],
            "changed_samples": 0,
            "whole_input_reencoded": True,
            "synthetic_padding_or_wraparound": False,
            "original_pcm_mapping": copy.deepcopy(mapping),
            "audio_origin_seconds": entry["input"]["audio_origin_seconds"],
            "input_output_sample_mapping": speech.provenance(mapping, "original_pcm", entry["audio_delay"]),
            "output_clock_mapping": {
                "samples": mapping["samples"],
                "sample_rate": 16000,
                "channels": 1,
                "pcm_sha256": mapping["pcm_sha256"],
                "timeline_time_base": "1/16000",
                "timeline": copy.deepcopy(mapping["timeline"]),
                "meaning": "Output clock retained; modified values are not claimed to be the original soundtrack.",
            },
        },
        "routing": [
            {
                "frame_index": f["frame_index"],
                "view": key,
                "status": "verified",
                "chunk_sha256": "f" * 64,
                "conditioning_plan": copy.deepcopy(f[key]),
            }
            for f in entry["frames"]
            for key in ("source_time_conditioning", "earlier_conditioning")
        ],
    }
    return entry, request, record


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "role",
        "batch",
        "other_clip",
        "proof",
        "dtype",
        "clock",
        "values",
        "routing",
        "original_hash",
        "count",
    ],
)
def test_consumer_metadata_binds_actual_owned_source_waveform_clock_and40routes(
    tmp_path, monkeypatch, damage
):
    monkeypatch.setattr(speech, "CACHE", tmp_path)
    entry, request, record = encoded_metadata(tmp_path)
    if damage is None:
        # These files deliberately do not exist: admission checks metadata only.
        speech.validate_record(record, request, entry, 0)
        return
    if damage == "role":
        record["role"] = "reserve"
    elif damage == "batch":
        record["batch"] = 1
    elif damage == "other_clip":
        record["hidden_file"] = str(tmp_path / "batch-00/clip-01/original_pcm.hidden.npy")
    elif damage == "proof":
        record["proof"]["layer_readout_matches"] = False
    elif damage == "dtype":
        record["hidden_dtype"] = "float64"
    elif damage == "clock":
        record["waveform"]["output_clock_mapping"]["timeline"][0]["source_pts"] = 0
    elif damage == "values":
        record["waveform"]["input_output_sample_mapping"][0]["input_samples_half_open"][0] = 1
    elif damage == "routing":
        record["routing"][0], record["routing"][1] = record["routing"][1], record["routing"][0]
    elif damage == "original_hash":
        record["waveform"]["pcm_sha256"] = "f" * 64
        record["waveform"]["output_clock_mapping"]["pcm_sha256"] = "f" * 64
    else:
        record["routing"].pop()
    with pytest.raises(ValueError):
        speech.validate_record(record, request, entry, 0)


def test_source_lookup_never_reads_reserve_or_relabels_delayed_states(monkeypatch):
    plan = {"sources": [{"ordinal": i, "role": "pilot" if i < 16 else "reserve"} for i in range(32)]}
    monkeypatch.setattr(speech.planning, "load_completed", lambda: plan)
    monkeypatch.setattr(
        speech,
        "load_completed",
        lambda batch: [
            {"source_ordinal": i, "waveform_id": name}
            for i in range(batch * 2, batch * 2 + 2)
            for name in speech.WAVEFORMS
        ],
    )
    assert speech.source_record(15)["waveform_id"] == "original_pcm"
    assert speech.source_record(15, "delayed_audio_control")["waveform_id"] == "delayed_audio_control"
    with pytest.raises(ValueError):
        speech.source_record(16)


def test_cache_subdirectory_symlink_cannot_escape_owned_metadata_paths(tmp_path, monkeypatch):
    owned, outside = tmp_path / "owned", tmp_path / "outside"
    owned.mkdir()
    outside.mkdir()
    (owned / "batch-00").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(speech, "CACHE", owned)
    entry, request, record = encoded_metadata(owned)
    with pytest.raises(ValueError, match="provenance"):
        speech.validate_record(record, request, entry, 0)


def test_only_exact_batch_scoped_neural_authored_worker_is_dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(speech.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    argv = [
        "python",
        "-B",
        "-m",
        "scripts.video_appearance_native_v108",
        "authored",
        str(speech.TRIAL / "batch-03/authored/output"),
    ]
    speech.routed_popen(3, argv)
    assert calls[0][3:] == ["scripts.grid_source_speech_v147", "encode", argv[5], "3"]
    with pytest.raises(ValueError):
        speech.routed_popen(2, argv)
    with pytest.raises(ValueError):
        speech.routed_popen(True, argv)
