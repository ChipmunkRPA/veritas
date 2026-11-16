from contextlib import nullcontext
from copy import deepcopy
from hashlib import sha256
from shutil import which
from subprocess import run
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.audio import short_voice_review
from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from app.detectors.video.adapters.audio import NO_ADMITTED_VOICE
from app.detectors.video.adapters.common import AudioTrack
from app.engine import boundary
from app.engine.boundary import Cancelled, Deadline
from app.engine.detector import DetectorResult, Profile
from app.engine.stages import audio
from app.evidence import audio_packets, audio_short_review
from app.evidence import audio_short_review_binding as binding


def source(pcm):
    return {
        "samples": len(pcm), "sample_rate": 16000, "source_sample_rate": 16000, "channels": 1,
        "source_channels": 1, "source_codec": "pcm_s16le", "pcm_sha256": sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        "timeline_time_base": "1/16000", "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}],
    }


def test_scope_excludes_lossy_stereo_multitrack_and_clock_changes():
    mapping = source(np.zeros(binding.SAMPLES, np.float32))
    assert binding.supported_source(mapping)
    flac = mapping | {"source_codec": "flac", "source_container": "flac", "source_bit_depth": 16}
    assert binding.supported_source(flac)
    assert not binding.supported_source(flac | {"source_bit_depth": 24})
    assert not binding.supported_source(flac | {"source_container": "ogg"})
    assert not binding.supported_source(mapping | {"source_codec": "opus"})
    assert not binding.supported_source(mapping | {"source_channels": 2})
    assert not binding.supported_source(mapping | {"audio_stream_indices": [0, 1]})
    assert not binding.supported_source(mapping | {"timeline": [dict(mapping["timeline"][0], source_pts=1)]})


@pytest.mark.skipif(not which("ffmpeg") or not which("ffprobe"), reason="requires FFmpeg")
@pytest.mark.parametrize("codec", ["wav", "flac"])
def test_short_review_requires_exact_channel_and_analyzed_input_agreement(tmp_path, codec):
    import wave

    values = ((np.arange(binding.SAMPLES) % 256) - 128).astype("<i2")
    stereo = np.column_stack((values, values))

    def decode(name, channels):
        wav = tmp_path / f"{name}.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setparams((2, 2, binding.RATE, binding.SAMPLES, "NONE", "not compressed"))
            stream.writeframes(channels.astype("<i2").tobytes())
        path = wav
        if codec == "flac":
            path = wav.with_suffix(".flac")
            run(["ffmpeg", "-v", "error", "-i", str(wav), str(path)], check=True)
        return decode_audio(path, source_sha256=sha256(path.read_bytes()).hexdigest(),
                            retain_stereo=True, inspect_stereo=True)

    identical = decode("identical", stereo)
    assert not binding.supported_source(identical.mapping)
    proof = audio_short_review.identical_stereo_proof(identical)
    assert proof == sha256(values.tobytes()).hexdigest()
    identical.mapping[binding.IDENTICAL_CHANNEL_PCM] = proof
    assert binding.supported_source(identical.mapping)
    assert binding.source_metadata(identical.mapping)[binding.IDENTICAL_CHANNEL_PCM] == proof
    np.testing.assert_array_equal(identical.pcm, values.astype(np.float32) / 32768)
    changed = decode("minor-difference", stereo + np.array([1, -1]))
    np.testing.assert_array_equal(changed.pcm, identical.pcm)
    assert audio_short_review.identical_stereo_proof(changed) is None
    assert not binding.supported_source(changed.mapping)
    assert audio_short_review.identical_stereo_proof(SimpleNamespace(
        mapping=identical.mapping, stereo_pcm=stereo, pcm=identical.pcm + 1 / 32768)) is None
    assert not binding.supported_source(identical.mapping | {"source_bit_depth": 24})


def test_mp3_uses_its_frozen_point_and_exact_source_clock():
    wav = source(np.zeros(binding.SAMPLES, np.float32))
    mp3 = wav | {"source_codec": "mp3", "source_container": "mp3", "source_bit_rate": 48000,
                 "timeline": [dict(wav["timeline"][0], source_pts=1105)]}
    assert binding.supported_source(mp3)
    assert not binding.supported_source(mp3 | {"source_bit_rate": 64000})
    assert not binding.supported_source(mp3 | {"timeline": wav["timeline"]})
    response = [{"start_sample": 48000, "end_sample": 80000, "logit": 7.0}]
    assert binding.selected_range(response) is not None and binding.selected_range(response, "mp3") is None
    response[0]["logit"] = 7.1
    assert binding.selected_range(response, "mp3") == {"start_sample": 48000, "end_sample": 80000}
    assert binding.source_metadata_matches({"source_codec": "mp3"}, mp3)
    assert not binding.source_metadata_matches({"source_codec": "mp3", "source_bit_rate": 64000}, mp3)


@pytest.mark.skipif(not which("ffmpeg") or not which("ffprobe"), reason="requires FFmpeg")
@pytest.mark.parametrize("carrier", ["wav", "flac"])
def test_resampled_lossless_check_requires_exact_original_extent(tmp_path, carrier):
    import wave

    def decode(native_samples):
        wav = tmp_path / f"{native_samples}.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setparams((1, 2, binding.LOSSLESS_RATE, native_samples, "NONE", "not compressed"))
            stream.writeframes(bytes(2 * native_samples))
        path = wav
        if carrier == "flac":
            path = wav.with_suffix(".flac")
            run(["ffmpeg", "-v", "error", "-i", str(wav), str(path)], check=True)
        return decode_audio(path, source_sha256=sha256(path.read_bytes()).hexdigest()).mapping

    mapping = decode(288000)
    assert binding.supported_source(mapping)
    assert binding.source_metadata(mapping)["source_duration_pts"] == 288000
    assert binding.source_metadata(mapping)[binding.LOSSLESS_NATIVE_CLOCK] is True
    assert binding.source_metadata_matches(mapping, mapping)
    assert not binding.source_metadata_matches(mapping, mapping | {"source_duration_pts": 288001})
    longer = decode(288001)
    assert longer["samples"] == binding.SAMPLES
    assert not binding.supported_source(longer)
    for invalid in ({"source_channels": 2}, {"source_start_pts": 1}, {"source_bit_depth": 24},
                    {binding.LOSSLESS_NATIVE_CLOCK: False}, {"audio_stream_indices": [0, 1]},
                    {"timeline": [dict(mapping["timeline"][0], source_pts=1)]}):
        assert not binding.supported_source(mapping | invalid)
    if carrier == "flac":
        path = tmp_path / "288001.flac"
        raw = path.read_bytes()
        # Change STREAMINFO's total sample count, leaving the audio frames intact.
        fields = int.from_bytes(raw[18:26], "big")
        extent_mask = (1 << 36) - 1
        assert fields & extent_mask == 288001
        raw = raw[:18] + ((fields & ~extent_mask) | 288000).to_bytes(8, "big") + raw[26:]
        path.write_bytes(raw)
        inconsistent = decode_audio(path, source_sha256=sha256(raw).hexdigest()).mapping
        assert inconsistent["samples"] == binding.SAMPLES and inconsistent["source_duration_pts"] == 288000
        assert not binding.supported_source(inconsistent)


def test_aac_keeps_the_exact_presentation_separate_from_decoder_padding():
    presentation = np.zeros(binding.SAMPLES, np.float32)
    padded = np.concatenate((presentation, np.ones(256, np.float32)))
    aac = source(padded) | {
        "source_codec": "aac", "source_container": binding.AAC_CONTAINER, "source_profile": "LC",
        "source_sample_rate": 48000, "source_bit_rate": 65000, "source_time_base": "1/48000",
        "source_start_pts": 0, "source_duration_pts": 288000,
    }
    assert binding.supported_source(aac)
    for key, value in (("source_duration_pts", 288003), ("source_profile", "HE-AAC"), ("source_bit_rate", 96000),
                       ("source_channels", 2), ("audio_stream_indices", [0, 1]), ("source_start_pts", 1024)):
        assert not binding.supported_source(aac | {key: value})
    coverage = binding.window_coverage([32000] * 5, aac)
    assert coverage[-1]["source_timestamp_ranges"][-1]["source_end_pts"] == binding.SAMPLES

    def spans(samples):
        np.testing.assert_array_equal(samples, presentation)
        return [(0.0, 6.0)]

    classifiers = {
        short_voice_review.ACTIVITY_MODEL: SimpleNamespace(obj=SimpleNamespace(spans=spans)),
        short_voice_review.PellaVoiceDetector.MODEL_ID: SimpleNamespace(obj=SimpleNamespace(predict=lambda _: [.9995])),
    }
    ctx = SimpleNamespace(outputs=SimpleNamespace(audio=AudioTrack("analysis", padded, len(padded)), probe={"source_codec": "aac"}),
                          lease=classifiers, deadline=Deadline(None))
    review = short_voice_review.ShortVoiceReview()
    result = review.run(ctx)
    assert result.details["pcm_sha256"] == aac["pcm_sha256"]
    assert result.details["short_voice_review"]["selected"] == {"start_sample": 0, "end_sample": 32000}
    marginal = [{"start_sample": 0, "end_sample": 32000, "logit": 6.9}]
    assert binding.selected_range(marginal) and not binding.selected_range(marginal, "aac")


@pytest.mark.skipif(not which("ffmpeg") or not which("ffprobe"), reason="Requires the local audio decoder")
def test_native_flac_depth_and_exact_decoded_samples(tmp_path):
    source_pcm = np.arange(binding.SAMPLES, dtype=np.int32).astype("<i2").astype(np.float32) / 32768
    input_wav = tmp_path / "source.wav"
    input_wav.write_bytes(wav_bytes(source_pcm))
    for depth, sample_format in ((16, "s16"), (24, "s32")):
        output = tmp_path / f"source-{depth}.flac"
        run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(input_wav), "-c:a", "flac",
             "-sample_fmt", sample_format, "-bits_per_raw_sample", str(depth), str(output)], check=True)
        decoded = decode_audio(output, source_sha256=sha256(output.read_bytes()).hexdigest())
        assert decoded.mapping["source_container"] == "flac" and decoded.mapping["source_bit_depth"] == depth
        assert binding.supported_source(decoded.mapping) == (depth == 16)
        np.testing.assert_array_equal(decoded.pcm, source_pcm)


@pytest.mark.parametrize("optional", [DetectorResult.timeout("slow"), DetectorResult.failed("invalid")])
def test_optional_boundary_failure_does_not_change_completed_primary(monkeypatch, optional):
    pcm = np.zeros(binding.SAMPLES, np.float32)
    original = {
        "detectors": [{"detector_id": short_voice_review.PellaVoiceDetector.spec.id, "status": "ok", "unit_scores": [0.8, 0.7],
                       "details": {"audio_screening": {"decision": "inconclusive", "probability_ai": None}}}],
        "exhibits": ["primary-waveform"],
    }
    calls = []

    def primary(*args, **kwargs):
        calls.append("primary")
        return deepcopy(original)

    def separate(detector, ctx, **kwargs):
        calls.append("optional")
        assert detector.spec.id == binding.DETECTOR_ID and kwargs["timeout_s"] == short_voice_review.BUDGET_S
        return optional

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(audio.common, "run_detectors", primary)
    monkeypatch.setattr(audio.common, "media", lambda ctx: None)
    monkeypatch.setattr(audio, "run_detector", separate)
    ctx = SimpleNamespace(
        profile=Profile.FULL, analysis_id="test", deadline=Deadline(100), check=lambda *_: None,
        outputs={"probe": {"facts": {"container": "wav"}}, "decode": {"short_voice_review_source_supported": True},
                 "condition": {"content": {"duration_s": 6, "speech_spans": [[0, 6]]}}},
        state={"audio": AudioTrack("analysed", pcm, len(pcm)), "run_store": object()},
        exhibits=SimpleNamespace(bind=lambda *_: None), log=SimpleNamespace(bind=lambda **_: None),
    )
    result = audio.detect(ctx)
    assert calls == ["primary", "optional"]
    assert result["detectors"] == original["detectors"] and result["exhibits"] == original["exhibits"]
    assert result["short_voice_review"]["status"] == "skipped"
    assert result["short_voice_review"]["selected"] is None
    ctx.deadline = Deadline(5)
    assert audio.detect(ctx)["short_voice_review"]["status"] == "skipped"
    assert calls == ["primary", "optional", "primary"]
    ctx.deadline = Deadline(100)

    def cancelled(*_):
        raise Cancelled("job cancelled")

    ctx.check = cancelled
    with pytest.raises(Cancelled):
        audio.detect(ctx)


def test_flagged_packet_anchors_exact_slice_without_public_raw_score():
    pcm = np.arange(binding.SAMPLES, dtype=np.int32).astype("<i2").astype(np.float32) / 32768
    mapping = source(pcm)
    p = iter([.9999])
    model = SimpleNamespace(predict=lambda chunks: [next(p)])
    review = short_voice_review.scan(pcm, [(4.5, 6)], model, Deadline(None))
    raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok",
                          "details": {"pcm_sha256": mapping["pcm_sha256"], "short_voice_review": review}})
    pin = sha256(raw).hexdigest()
    assets = {}

    def asset(data, media_type, **meta):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **meta}

    record = {"raw_output_sha256": pin}
    store = SimpleNamespace(read_bytes={pin: raw}.__getitem__)
    result = audio_short_review.build(record, store=store, source=mapping, pcm=pcm, mapping=mapping,
                                      asset=asset, waveform=audio_packets._chart)
    assert result["status"] == "flagged" and result["affects_overall_confidence"] is False
    assert result["interval"] == {"start_sample": 64000, "end_sample": 96000}
    assert result["start_s"] == 4.0 and result["end_s"] == 6.0
    assert result["schema"] == binding.COVERAGE_SCHEMA
    assert [row["speech_samples"] for row in result["window_coverage"]] == [0, 0, 0, 8000, 24000]
    assert [row["status"] for row in result["window_coverage"]] == ["below_speech_minimum"] * 4 + ["checked"]
    assert b"logit" not in canonical_json(result)
    np.testing.assert_array_equal(read_wav(assets[result["artifacts"]["selected_clip"]["sha256"]]), pcm[64000:96000])
    audio_short_review.verify(result, record, store=store, source=mapping, pcm=pcm, mapping=mapping,
                              read_asset=assets.__getitem__, waveform=audio_packets._chart)
    changed = deepcopy(result)
    changed["source_timestamp_ranges"][0]["source_end_pts"] -= 1
    with pytest.raises(ValueError):
        binding.validate(changed, mapping)
    changed = deepcopy(result)
    changed["window_coverage"][0]["source_timestamp_ranges"][0]["source_end_pts"] -= 1
    with pytest.raises(ValueError, match="coverage disagrees"):
        binding.validate(changed, mapping)
    changed = deepcopy(result)
    changed["window_coverage"][0]["speech_samples"] = 120
    with pytest.raises(ValueError, match="differs from its frozen response"):
        audio_short_review.verify(changed, record, store=store, source=mapping, pcm=pcm, mapping=mapping,
                                  read_asset=assets.__getitem__, waveform=audio_packets._chart)
    # Archived packets rebuild their original schema and contain no inferred coverage.
    old = deepcopy(review)
    old["schema"] = binding.SCHEMA
    del old["speech_counts"]
    archived = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok",
                              "details": {"pcm_sha256": mapping["pcm_sha256"], "short_voice_review": old}})
    old_pin = sha256(archived).hexdigest()
    legacy = audio_short_review.build({"raw_output_sha256": old_pin}, store=SimpleNamespace(read_bytes={old_pin: archived}.__getitem__),
                                       source=mapping, pcm=pcm, mapping=mapping, asset=asset, waveform=audio_packets._chart)
    assert "window_coverage" not in legacy and legacy["interval"] == result["interval"]

    nearby = canonical_json({"detector_id": binding.NEARBY_ID, "status": "ok", "details": {
        "pcm_sha256": mapping["pcm_sha256"], "anchor": review["selected"],
        "windows": [{"start_sample": 62720, "end_sample": 94720, "speech_samples": 22720, "logit": 0.0}],
    }})
    nearby_pin = sha256(nearby).hexdigest()
    record["nearby_raw_output_sha256"] = nearby_pin
    store = SimpleNamespace(read_bytes={pin: raw, nearby_pin: nearby}.__getitem__)
    combined = audio_short_review.build(record, store=store, source=mapping, pcm=pcm, mapping=mapping,
                                        asset=asset, waveform=audio_packets._chart)
    assert combined["schema"] == binding.NEARBY_SCHEMA and combined["interval"] == result["interval"]
    assert combined["nearby_window_review"]["status"] == "varied"
    [shifted] = combined["nearby_window_review"]["windows"]
    assert shifted["source_timestamp_ranges"] == [binding.source_range(mapping, 62720, 94720)]
    assert b"logit" not in canonical_json(combined)
    audio_short_review.verify_nearby_source(combined, pcm)
    audio_short_review.verify(combined, record, store=store, source=mapping, pcm=pcm, mapping=mapping,
                              read_asset=assets.__getitem__, waveform=audio_packets._chart)
    changed = deepcopy(combined)
    changed["nearby_window_review"]["windows"][0]["pcm_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="samples differ"):
        audio_short_review.verify_nearby_source(changed, pcm)


def test_nearby_failure_retains_completed_short_review(monkeypatch):
    pcm = np.zeros(binding.SAMPLES, np.float32)
    selected = {"start_sample": 16000, "end_sample": 48000}
    calls = []

    def run_optional(detector, ctx, **kwargs):
        calls.append(detector.spec.id)
        if detector.spec.id == binding.NEARBY_ID:
            assert ctx.outputs.probe["selected"] == selected
            assert kwargs["timeout_s"] == short_voice_review.NEARBY_BUDGET_S
            return DetectorResult.timeout("slow")
        return DetectorResult(status="ok", details={"short_voice_review": {"selected": selected}})

    monkeypatch.setattr(audio, "run_detector", run_optional)
    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(audio.common, "media", lambda _: None)
    track = AudioTrack("analysis", pcm, len(pcm))
    ctx = SimpleNamespace(
        profile=Profile.FULL, analysis_id="test", deadline=Deadline(100), check=lambda *_: None,
        outputs={"probe": {"facts": {"container": "wav"}}, "decode": {"short_voice_review_source_supported": True,
                 "short_voice_review_source_codec": "pcm_s16le"}},
        state={"run_store": SimpleNamespace(run_id=lambda *_: "short-run")},
        exhibits=SimpleNamespace(bind=lambda *_: None), log=SimpleNamespace(bind=lambda **_: None),
    )
    entries = [{"detector_id": short_voice_review.PellaVoiceDetector.spec.id, "status": "ok"}]
    assert audio._short_voice_review(ctx, track, entries) == {"run_id": "short-run"}
    assert calls == [binding.DETECTOR_ID, binding.NEARBY_ID]


def test_insufficient_speech_maps_all_short_windows_without_inference():
    pcm = np.zeros(binding.SAMPLES, np.float32)
    review = short_voice_review.scan(pcm, [(3.5, 4)], None, Deadline(None))
    assert review["status"] == "skipped" and review["reason"] == binding.NO_SPEECH_REASON
    assert review["responses"] == [] and review["speech_counts"] == [0, 0, 8000, 8000, 0]


@pytest.mark.parametrize("status,reason,documentary,enabled,expected", [
    ("not_applicable", NO_ADMITTED_VOICE, False, True, True),
    ("not_applicable", NO_ADMITTED_VOICE, False, False, False),
    ("not_applicable", "no audio track", False, True, False),
    ("failed", NO_ADMITTED_VOICE, False, True, False),
    ("timeout", None, False, True, False),
    ("not_applicable", NO_ADMITTED_VOICE, True, True, False),
])
def test_short_review_only_recovers_an_admission_skip(monkeypatch, status, reason, documentary, enabled, expected):
    calls = []
    ctx = SimpleNamespace(
        profile=Profile.FULL, analysis_id="test", deadline=Deadline(100), check=lambda *_: None,
        outputs={"probe": {"facts": {"container": "wav"}}, "decode": {"short_voice_review_source_supported": True}},
        state={"run_store": SimpleNamespace(run_id=lambda *_: "short-review")},
        exhibits=SimpleNamespace(bind=lambda *_: None), log=SimpleNamespace(bind=lambda **_: None),
    )

    def record(detector, run_ctx, **_kwargs):
        calls.append(detector.spec.id)
        assert run_ctx.outputs.probe["primary_coverage_skip"] is True
        return DetectorResult(status="ok")

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(settings, "environment", "dev")
    monkeypatch.setattr(settings, "audio_short_admission_review", enabled)
    monkeypatch.setattr(audio.provenance_gate, "stop", lambda _: documentary)
    monkeypatch.setattr(audio.common, "media", lambda _: None)
    monkeypatch.setattr(audio, "run_detector", record)
    entry = {"detector_id": short_voice_review.PellaVoiceDetector.spec.id, "status": status, "reason": reason}
    result = audio._short_voice_review(ctx, None, [entry])
    assert calls == ([binding.DETECTOR_ID] if expected else [])
    assert result.get("run_id") == ("short-review" if expected else None)
    if documentary:
        assert result["reason"] == audio.provenance_gate.SKIP_REASON


def test_watchdog_discards_slow_optional_run_without_losing_primary(monkeypatch):
    pcm = np.zeros(binding.SAMPLES, np.float32)
    primary = [{"detector_id": short_voice_review.PellaVoiceDetector.spec.id, "status": "ok", "unit_scores": [0.2, 0.8],
                "details": {"audio_screening": {"decision": "inconclusive", "probability_ai": None}}}]
    release, entered = Event(), Event()
    records = []

    def blocked(self, ctx):
        entered.set()
        release.wait(5)
        return DetectorResult(status="ok")

    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    monkeypatch.setattr(audio.common, "media", lambda ctx: None)
    monkeypatch.setattr(audio.common, "run_detectors", lambda *_args, **_kwargs: {"detectors": primary})
    monkeypatch.setattr(short_voice_review.ShortVoiceReview, "run", blocked)
    monkeypatch.setattr(short_voice_review, "BUDGET_S", 0.05)
    monkeypatch.setattr(boundary, "WATCHDOG_GRACE_S", 0.1)
    models = SimpleNamespace(activation_bytes_for_items=lambda _: 0,
                             lease=lambda *args, **kwargs: nullcontext(SimpleNamespace(device="cpu", dtype="float32")))
    deadline = Deadline(60)
    ctx = SimpleNamespace(
        profile=Profile.FULL, analysis_id="test", deadline=deadline, check=deadline.check,
        outputs={"probe": {"facts": {"container": "wav"}}, "decode": {"short_voice_review_source_supported": True},
                 "condition": {"content": {"duration_s": 6, "speech_spans": [[0, 6]]}}},
        state={"audio": AudioTrack("analysed", pcm, len(pcm)), "run_store": SimpleNamespace(persist=records.append),
               "models": models},
        exhibits=SimpleNamespace(bind=lambda *_: None), log=SimpleNamespace(bind=lambda **_: None),
    )
    try:
        output = audio.detect(ctx)
    finally:
        release.set()
    assert entered.is_set()
    assert output["detectors"] == primary
    assert output["short_voice_review"]["status"] == "skipped"
    assert [record.status for record in records] == ["timeout"]
    deadline.check("primary analysis can continue")
