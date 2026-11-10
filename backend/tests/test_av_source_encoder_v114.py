import hashlib
import json
import wave

import numpy as np
import pytest

from scripts import av_source_encoder_v114 as source
from scripts.av_conditioning_clock_v109 import frame_plan


def wav_record(tmp_path, monkeypatch):
    pcm = np.resize(np.array([-32768, -1, 0, 1, 32767], dtype="<i2"), 1280)
    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 16000, len(pcm), "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    monkeypatch.setattr(source.clock, "CACHE", tmp_path)
    return pcm, {
        "file": path.name,
        "sha256": source.sha(path),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        "sample_frames": len(pcm),
    }


def test_signed_pcm_conversion_keeps_exact_amplitude_and_order(tmp_path, monkeypatch):
    pcm, record = wav_record(tmp_path, monkeypatch)
    actual = source.read_pcm(record)
    assert actual.dtype == np.float32
    assert actual.tobytes() == (pcm.astype(np.float32) / np.float32(32768)).tobytes()
    assert actual.min() == -1 and actual.max() < 1


@pytest.mark.parametrize(
    "key,value",
    [("file", "../test.wav"), ("sha256", "changed"), ("pcm_sha256", "changed"), ("sample_frames", 1279)],
)
def test_changed_waveform_binding_is_rejected(tmp_path, monkeypatch, key, value):
    _, record = wav_record(tmp_path, monkeypatch)
    record[key] = value
    with pytest.raises(ValueError):
        source.read_pcm(record)


def test_native_source_routing_retains_zero_padding_and_time_layer_order():
    hidden = np.arange(1500 * 5 * 384, dtype=np.float32).reshape(1500, 5, 384)
    frames = [frame_plan(i, t, "0", "0", 1280, "0") for i, t in enumerate(("0", "1/25"))]
    actual = source.audit_chunks(hidden, frames)
    assert actual == {"routed_frames": 2, "explicit_zero_tick_slots": 12}


def test_routing_audit_detects_layer_or_tick_permutation(monkeypatch):
    hidden = np.arange(1500 * 5 * 384, dtype=np.float32).reshape(1500, 5, 384)
    frame = frame_plan(0, "1/25", "0", "0", 1280, "0")
    original = source.clock.select_chunk
    monkeypatch.setattr(source.clock, "select_chunk", lambda h, f: original(h, f)[::-1].copy())
    with pytest.raises(ValueError, match="routing differs"):
        source.audit_chunks(hidden, [frame])


def test_out_of_waveform_cannot_become_zero_conditioning():
    frame = frame_plan(0, "1", "0", "0", 1280, "0")
    with pytest.raises(ValueError, match="unavailable"):
        source.audit_chunks(np.zeros((1500, 5, 384), np.float32), [frame])


def test_worker_routing_changes_only_module_and_preserves_input_argv(monkeypatch):
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "extract", "private", "0"]
    calls = []
    monkeypatch.setattr(source.subprocess, "Popen", lambda argv, **kw: calls.append((argv, kw)))
    source.routed_popen(args, cwd="backend")
    assert args[3] == "scripts.video_appearance_native_v108"
    assert calls == [([*args[:3], "scripts.av_source_encoder_v114", *args[4:]], {"cwd": "backend"})]
    with pytest.raises(ValueError):
        source.routed_popen(["python", "-B", "-m", "other", "extract"])


@pytest.mark.parametrize("batch", [-1, 8, True, 1.0])
def test_undeclared_batch_is_rejected_before_model_or_filesystem_work(batch):
    with pytest.raises(ValueError):
        source.run(batch)


def test_role_or_waveform_cross_binding_is_rejected(tmp_path, monkeypatch):
    plans = [
        {
            "asset_index": i,
            "source_group": str(i),
            "study_role": "fusion_fit",
            "mono_wav_sha256": "h",
            "generator_conditioning_qualified": False,
        }
        for i in range(48)
    ]
    waves = [
        {"asset_index": i, "source_group": str(i), "study_role": "fusion_fit", "mono16k_wav": {"sha256": "h"}}
        for i in range(48)
    ]
    plan, wav = tmp_path / "plan", tmp_path / "waves"
    plan.write_text("\n".join(map(json.dumps, plans)))
    wav.write_text("\n".join(map(json.dumps, waves)))
    monkeypatch.setattr(source, "PLAN", plan)
    monkeypatch.setattr(source, "WAVES", wav)
    assert len(source.source_records()) == 48
    waves[2]["study_role"] = "selection"
    wav.write_text("\n".join(map(json.dumps, waves)))
    with pytest.raises(ValueError, match="binding"):
        source.source_records()
