from __future__ import annotations

import copy
import hashlib
import io
import math
import shutil
import subprocess
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_json
from app.detectors.audio import brief_voice_review
from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from app.evidence import audio_brief_voice as evidence
from app.evidence import audio_brief_voice_binding as binding
from app.evidence.audio_packets import _chart


def test_scan_uses_exact_inputs_and_strict_point(monkeypatch):
    pcm = np.arange(binding.SAMPLES, dtype=np.float32)
    above = math.nextafter(binding.POINT, math.inf)
    scores = iter((binding.POINT, above, above, above))
    called = []

    def logit(_, chunk):
        called.append((int(chunk[0]), len(chunk)))
        return next(scores)

    monkeypatch.setattr(brief_voice_review.short, "voice_logit", logit)
    rows = brief_voice_review.scan(pcm, [(0, 30000 / binding.RATE)], None,
                                   SimpleNamespace(check=lambda _: None))
    checks = binding.raw_checks(rows)
    assert len(checks) == 23
    assert [row["start_sample"] for row in checks] == list(binding.GRID)
    assert called == [(0, 12000), (8000, 12000), (16000, 12000), (24000, 12000)]
    assert [row["elevated"] for row in checks[:5]] == [False, True, True, True, None]
    assert checks[3]["speech_samples"] == 6000
    assert binding.selected_indices(checks) == [1, 2, 3]
    assert binding.presentation(checks)["speech_windows_checked"] == 4

    rows[2]["logit_hex"] = binding.POINT.hex()
    isolated = binding.raw_checks(rows)
    assert binding.presentation(isolated)["status"] == "no_flag"


def test_no_speech_and_interrupted_scan_are_distinct():
    pcm = np.zeros(binding.SAMPLES, dtype=np.float32)
    deadline = SimpleNamespace(check=lambda _: None)
    checks = binding.raw_checks(brief_voice_review.scan(pcm, [], None, deadline))
    assert binding.presentation(checks)["status"] == "unassessed"
    assert binding.presentation([], reason=binding.TIMEOUT)["status"] == "skipped"
    with pytest.raises(ValueError, match="Incomplete"):
        binding.raw_checks(checks[:-1])

    for index in (0, 2):
        checks[index].update(speech_samples=6000, elevated=True)
        assert binding.presentation(checks)["status"] == "unassessed"
    checks[1].update(speech_samples=6000, elevated=False)
    assert binding.presentation(checks)["status"] == "no_flag"


@pytest.mark.parametrize(("codec", "container"), [("pcm_s16le", "wav"), ("flac", "flac")])
@pytest.mark.parametrize("source_channels", [1, 2])
def test_frozen_media_match_original_samples_and_reject_changed_windows(monkeypatch, codec, container, source_channels):
    pcm = (np.arange(binding.SAMPLES, dtype=np.int32) % 8192 - 4096).astype(np.float32) / 32768
    source = {"samples": len(pcm), "sample_rate": binding.RATE, "source_sample_rate": binding.RATE,
              "channels": 1, "source_channels": source_channels, "source_codec": codec, "source_container": container,
              "source_bit_depth": 16, "timeline_time_base": "1/16000",
              "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
              "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}]}
    if source_channels == 2:
        source.update(source_time_base="1/16000", source_start_pts=0,
                      **{binding.IDENTICAL_CHANNEL_PCM: hashlib.sha256((pcm * 32768).astype("<i2").tobytes()).hexdigest()})
    monkeypatch.setattr(brief_voice_review.short, "voice_logit", lambda *_: binding.POINT + 1)
    rows = brief_voice_review.scan(pcm, [(0, 14000 / binding.RATE)], None,
                                   SimpleNamespace(check=lambda _: None))
    objects = {}

    def store_bytes(data):
        sha = hashlib.sha256(data).hexdigest()
        objects[sha] = data
        return sha

    def make_asset(data, media_type, **metadata):
        return {"sha256": store_bytes(data), "media_type": media_type, **metadata}

    def make_record():
        raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok", "details": {
            "pcm_sha256": source["pcm_sha256"], binding.KEY: {"schema": binding.SCHEMA, "readings": rows},
        }})
        return {"raw_output_sha256": store_bytes(raw)}

    inputs = dict(store=SimpleNamespace(read_bytes=objects.__getitem__), source=source, pcm=pcm, mapping=source,
                  waveform=_chart)
    record = make_record()
    review = evidence.build(record, **inputs, asset=make_asset)
    assert review["schema"] == (binding.SCHEMA if source_channels == 1 else binding.IDENTICAL_STEREO_SCHEMA)
    assert review["affects_overall_confidence"] is False
    assert [window["id"] for window in review["windows"]] == ["B750-01", "B750-02"]
    for window in review["windows"]:
        start, end = window["start_sample"], window["end_sample"]
        assert end - start == 12000
        assert window["source_timestamp_ranges"][0]["source_end_pts"] == end
        clip = objects[window["artifacts"]["selected_clip"]["sha256"]]
        np.testing.assert_array_equal(read_wav(clip), pcm[start:end])
    evidence.verify(review, record, **inputs, read_asset=objects.__getitem__)
    evidence.verify_source(review, pcm=pcm, mapping=source, waveform=_chart)
    if source_channels == 2:
        with pytest.raises(ValueError, match="channel proof differs"):
            evidence.build(record, **{**inputs, "source": {**source, binding.IDENTICAL_CHANNEL_PCM: "0" * 64}},
                           asset=make_asset)

    changed = copy.deepcopy(review)
    changed["windows"][0]["source_timestamp_ranges"][0]["source_end_pts"] += 1
    with pytest.raises(ValueError, match="exact source interval"):
        binding.validate(changed, source)
    rows[-1]["pcm_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different source samples"):
        evidence.build(make_record(), **inputs, asset=make_asset)
    assert binding.supported_source(source)


def test_identical_channel_proof_keeps_clock_and_format_scope():
    pcm = np.zeros(binding.SAMPLES, dtype=np.float32)
    source = {"samples": binding.SAMPLES, "sample_rate": binding.RATE, "source_sample_rate": binding.RATE,
              "channels": 1, "source_channels": 2, "source_codec": "flac", "source_container": "flac",
              "source_bit_depth": 16, "timeline_time_base": "1/16000", "source_time_base": "1/16000", "source_start_pts": 0,
              "timeline": [{"start_sample": 0, "end_sample": binding.SAMPLES, "source_pts": 0}]}
    decoded = SimpleNamespace(mapping=source, pcm=pcm, stereo_pcm=np.zeros((binding.SAMPLES, 2), dtype=np.int16))
    assert not binding.supported_source(source)
    pin = evidence.identical_stereo_proof(decoded)
    assert pin == hashlib.sha256(bytes(binding.SAMPLES * 2)).hexdigest()
    source[binding.IDENTICAL_CHANNEL_PCM] = pin
    assert binding.supported_source(source)
    review = {**binding.presentation([], reason=binding.TIME_BUDGET, schema=binding.IDENTICAL_STEREO_SCHEMA), "windows": []}
    binding.validate(review, source)
    with pytest.raises(ValueError, match="supported channels"):
        binding.validate({**review, "schema": binding.SCHEMA}, source)
    for changes in ({"source_bit_depth": 24}, {"source_sample_rate": 48000}, {"source_time_base": "1/48000"},
                    {"source_start_pts": 1}, {"samples": binding.SAMPLES + 1}, {"audio_stream_indices": [0, 1]},
                    {"timeline": [{"start_sample": 0, "end_sample": binding.SAMPLES, "source_pts": 1}]}):
        assert not binding.supported_source(source | changes)
        assert evidence.identical_stereo_proof(SimpleNamespace(**{**vars(decoded), "mapping": source | changes})) is None
    decoded.stereo_pcm[17, 0] = 1
    assert evidence.identical_stereo_proof(decoded) is None
    decoded.stereo_pcm[17, 1] = 1
    assert evidence.identical_stereo_proof(decoded) is None  # Equal original channels still differ from analyzed PCM.


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="native dual-mono proof requires FFmpeg")
def test_original_dual_mono_and_equal_average_unequal_channels(tmp_path):
    integer = np.random.default_rng(20261119).integers(-4096, 4096, binding.SAMPLES, dtype=np.int16)
    expected = integer.astype(np.float32) / 32768
    for unequal in (False, True):
        native = np.column_stack((integer, integer))
        if unequal:
            native[::8000, 0] += 1
            native[::8000, 1] -= 1
        encoded = io.BytesIO()
        with wave.open(encoded, "wb") as output:
            output.setnchannels(2)
            output.setsampwidth(2)
            output.setframerate(binding.RATE)
            output.writeframes(native.astype("<i2").tobytes())
        wav = tmp_path / ("unequal.wav" if unequal else "identical.wav")
        wav.write_bytes(encoded.getvalue())
        flac = wav.with_suffix(".flac")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(wav),
                        "-c:a", "flac", "-sample_fmt", "s16", str(flac)], check=True)
        for path in (wav, flac):
            decoded = decode_audio(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   inspect_stereo=True, retain_stereo=True)
            np.testing.assert_array_equal(decoded.pcm, expected)
            np.testing.assert_array_equal(decoded.stereo_pcm, native)
            pin = evidence.identical_stereo_proof(decoded)
            assert (pin is None) is unequal
            if pin:
                decoded.mapping[binding.IDENTICAL_CHANNEL_PCM] = pin
                assert binding.supported_source(decoded.mapping)
            else:
                assert not binding.supported_source(decoded.mapping)


@pytest.mark.parametrize("unsupported", [
    {"source_bit_depth": 24}, {"source_channels": 2}, {"channels": 2},
    {"source_sample_rate": 48000}, {"sample_rate": 48000}, {"samples": binding.SAMPLES - 1},
    {"source_container": "wav"}, {"source_codec": "alac"}, {"audio_stream_indices": [0, 1]},
    {"timeline_time_base": "1/48000"},
    {"timeline": [{"start_sample": 0, "end_sample": binding.SAMPLES, "source_pts": 1}]},
])
def test_flac_keeps_native_source_profile(unsupported):
    source = {"samples": binding.SAMPLES, "sample_rate": binding.RATE, "source_sample_rate": binding.RATE,
              "channels": 1, "source_channels": 1, "source_codec": "flac", "source_container": "flac",
              "source_bit_depth": 16, "timeline_time_base": "1/16000",
              "timeline": [{"start_sample": 0, "end_sample": binding.SAMPLES, "source_pts": 0}]}
    assert binding.supported_source(source)
    assert not binding.supported_source(source | unsupported)


@pytest.mark.parametrize("codec", ["pcm_s16le", "flac", "aac"])
def test_worker_gate_admits_only_qualified_native_codecs(monkeypatch, codec):
    from app.engine import registry
    from app.engine.stages import audio

    monkeypatch.setattr(audio, "settings", SimpleNamespace(environment="dev", audio_brief_voice_review=True,
                                                         audio_detector_mode="candidate"))
    monkeypatch.setattr(audio.provenance_gate, "stop", lambda _: False)
    monkeypatch.setattr(registry, "get_registry", lambda: SimpleNamespace(policy=SimpleNamespace(deny_detectors=set())))
    track = SimpleNamespace(pcm=np.zeros(binding.SAMPLES, dtype=np.float32))
    ctx = SimpleNamespace(profile="full", outputs={"decode": {"long_voice_review_source_supported": True,
        "short_voice_review_source_codec": codec}}, deadline=SimpleNamespace(remaining=lambda: 0))
    actual = audio._brief_voice_review(ctx, track)
    assert actual == (None if codec == "aac" else binding.skipped(binding.TIME_BUDGET))
    ctx.outputs["decode"]["long_voice_review_source_supported"] = False
    assert audio._brief_voice_review(ctx, track) is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="native FLAC parity requires FFmpeg")
def test_exact_flac_decode_readings_and_original_window_replay(tmp_path, monkeypatch):
    rng = np.random.default_rng(172)
    pcm = rng.integers(-4096, 4096, binding.SAMPLES, dtype=np.int16).astype(np.float32) / 32768
    wav = tmp_path / "original.wav"
    wav.write_bytes(wav_bytes(pcm))
    flac = tmp_path / "lossless.flac"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(wav),
                    "-sample_fmt", "s16", "-c:a", "flac", str(flac)], check=True)
    decoded = [decode_audio(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in (wav, flac)]
    np.testing.assert_array_equal(decoded[0].pcm, decoded[1].pcm)
    assert all(binding.supported_source(item.mapping) for item in decoded)
    assert decoded[0].mapping["timeline"] == decoded[1].mapping["timeline"]
    monkeypatch.setattr(brief_voice_review.short, "voice_logit", lambda *_: binding.POINT + 1)
    responses = [brief_voice_review.scan(item.pcm, [(0, 14000 / binding.RATE)], None,
                                       SimpleNamespace(check=lambda _: None)) for item in decoded]
    assert responses[0] == responses[1]
    objects = {}

    def asset(data, media_type, **metadata):
        pin = hashlib.sha256(data).hexdigest()
        objects[pin] = data
        return {"sha256": pin, "media_type": media_type, **metadata}

    reviews = []
    for item, readings in zip(decoded, responses, strict=True):
        raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok", "details": {
            "pcm_sha256": item.mapping["pcm_sha256"], binding.KEY: {"schema": binding.SCHEMA, "readings": readings}}})
        pin = hashlib.sha256(raw).hexdigest()
        objects[pin] = raw
        record = {"raw_output_sha256": pin}
        inputs = {"store": SimpleNamespace(read_bytes=objects.__getitem__), "source": item.mapping,
                  "mapping": item.mapping, "pcm": item.pcm, "waveform": _chart}
        review = evidence.build(record, **inputs, asset=asset)
        evidence.verify(review, record, **inputs, read_asset=objects.__getitem__)
        evidence.verify_source(review, pcm=item.pcm, mapping=item.mapping, waveform=_chart)
        reviews.append(review)
    assert reviews[0] == reviews[1]
    altered = decoded[1].pcm.copy()
    altered[100] += 1 / 32768
    with pytest.raises(ValueError, match="source samples"):
        evidence.verify_source(reviews[1], pcm=altered, mapping=decoded[1].mapping, waveform=_chart)
