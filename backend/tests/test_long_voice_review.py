import shutil
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.audio import extended_short_voice_review, long_voice_review, short_voice_review
from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from app.engine import runs
from app.engine.boundary import Deadline
from app.engine.detector import Profile
from app.engine.stages import audio
from app.evidence import audio_focused_edit, audio_focused_edit_binding, audio_long_review, audio_packets
from app.evidence import audio_long_review_binding as binding
from app.evidence import audio_short_review_binding as short
from app.evidence import audio_two_second_binding as two_second
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base, DetectorRun
from app.storage.objects import ObjectStore


def native_source(samples):
    return {"samples": samples, "channels": 1, "sample_rate": binding.RATE, "source_channels": 1,
            "source_sample_rate": binding.RATE, "source_bit_depth": 16, "source_container": "flac", "source_codec": "flac",
            "timeline_time_base": "1/16000", "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": 0}]}


def test_centered_short_edit_binds_the_absolute_clock_and_waveform():
    focused = audio_focused_edit_binding
    count = 160_003
    pcm = np.arange(count).astype("<i2").astype("<f4") / 32768
    source = native_source(count) | {"pcm_sha256": sha256(pcm.tobytes()).hexdigest()}
    start = 32_001
    scope = {"start_sample": start, "end_sample": 128_001}
    response = {"schema": focused.CENTERED_SCHEMA, "scope": scope, "checked_windows": 1,
                "scope_pcm_sha256": sha256(pcm[start:128_001].tobytes()).hexdigest(),
                "readings": [{"score": 10., "range": [800, 4800]}, None, None, None, None, None]}
    raw = canonical_json({"detector_id": focused.DETECTOR_ID, "status": "ok",
                          "details": {"pcm_sha256": source["pcm_sha256"], focused.KEY: response}})
    record = {"schema": focused.CENTERED_SCHEMA, "raw_output_sha256": sha256(raw).hexdigest()}
    store = SimpleNamespace(read_bytes={record["raw_output_sha256"]: raw}.__getitem__)
    assets = {}

    def asset(data, media_type, **metadata):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **metadata}

    shown = audio_focused_edit.build(record, store=store, source=source, pcm=pcm, mapping=source,
                                    asset=asset, waveform=audio_packets._chart)
    assert shown["checked_scope"] == scope
    assert shown["interval"] == {"start_sample": 32_801, "end_sample": 36_801}
    assert shown["affects_overall_confidence"] is False
    np.testing.assert_array_equal(read_wav(assets[shown["artifacts"]["selected_clip"]["sha256"]]), pcm[32_801:36_801])
    audio_focused_edit.verify(shown, record, store=store, source=source, pcm=pcm, mapping=source,
                              read_asset=assets.__getitem__, waveform=audio_packets._chart)
    shifted = shown | {"checked_scope": {"start_sample": 32_000, "end_sample": 128_000}}
    with pytest.raises(ValueError, match="supported source"):
        focused.validate(shifted, source)
    assert not focused.centered_supported(source | {"source_sample_rate": 48_000})


@pytest.mark.parametrize(("samples", "starts"), [
    (95_999, []), (96_000, [0, 48_000]), (143_999, [0, 95_999]), (144_000, [0, 48_000, 96_000]),
    (160_001, [0, 56_000, 112_001]), (224_000, [0, 88_000, 176_000]), (224_001, []),
])
def test_exact_disjoint_geometry(samples, starts):
    assert binding.intervals(samples) == [(start, start + 48_000) for start in starts]
    assert binding.supported_source(native_source(samples)) == bool(starts)


@pytest.mark.parametrize("change", [
    {"source_container": "ogg"}, {"source_codec": "mp3"}, {"source_bit_depth": 24},
    {"source_channels": 2}, {"source_sample_rate": 48_000}, {"audio_stream_indices": [0, 1]},
    {"timeline": [{"start_sample": 0, "end_sample": 144_000, "source_pts": 100}]},
    {"timeline": [{"start_sample": 0, "end_sample": 70_000, "source_pts": 0},
                  {"start_sample": 70_000, "end_sample": 144_000, "source_pts": 71_000}]},
])
def test_declines_other_native_profiles(change):
    assert not binding.supported_source(native_source(144_000) | change)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="Native audio decoder required")
def test_native_pcm_wav_has_the_same_long_window_scope(tmp_path):
    pcm = np.arange(144_001, dtype=np.int32).astype("<i2").astype("<f4") / 32768
    body = wav_bytes(pcm)
    path = tmp_path / "source"
    path.write_bytes(body)
    decoded = decode_audio(path, source_sha256=sha256(body).hexdigest())
    np.testing.assert_array_equal(decoded.pcm, pcm)
    assert binding.supported_source(decoded.mapping)
    assert binding.intervals(len(decoded.pcm)) == [(0, 48_000), (48_000, 96_000), (96_001, 144_001)]
    assert not binding.supported_source(decoded.mapping | {"source_container": "matroska"})
    assert not binding.supported_source(decoded.mapping | {"source_codec": "pcm_s24le"})


@pytest.mark.parametrize(("directions", "status"), [
    (["ai", "unassessed", "ai"], "corroborated"),
    (["ai", "inconclusive", "ai"], "corroborated"),
    (["ai", "original", "ai"], "no_corroboration"),
    (["unassessed", "inconclusive", "ai"], "no_corroboration"),
])
def test_exact_waveform_audio_and_clock_replay(directions, status):
    pcm = np.arange(144_001, dtype=np.int32).astype("<i2").astype("<f4") / 32768
    source = native_source(len(pcm)) | {"pcm_sha256": sha256(pcm.tobytes()).hexdigest()}
    windows = [{"start_sample": a, "end_sample": b, "direction": d, "pcm_sha256": sha256(pcm[a:b].tobytes()).hexdigest()}
               for (a, b), d in zip(binding.intervals(len(pcm)), directions, strict=True)]
    receipt = {"schema": binding.SCHEMA, "status": status, "reason": binding.REASONS[status], "windows": windows}
    raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok",
                          "details": {"pcm_sha256": source["pcm_sha256"], "long_voice_review": receipt}})
    pin = sha256(raw).hexdigest()
    store = SimpleNamespace(read_bytes={pin: raw}.__getitem__)
    record = {"raw_output_sha256": pin}
    assets = {}

    def asset(data, media_type, **metadata):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **metadata}

    review = audio_long_review.build(record, store=store, source=source, pcm=pcm, mapping=source,
                                    asset=asset, waveform=audio_packets._chart)
    assert review["status"] == status and review["affects_overall_confidence"] is False
    sample_coverage = review[binding.COVERAGE]
    assert sample_coverage["samples_by_status"]["outside_schedule"] == len(pcm) - 3 * binding.SPAN
    assert sum(sample_coverage["samples_by_status"].values()) == len(pcm)
    for window in review["windows"]:
        a, b = window["start_sample"], window["end_sample"]
        assert window["source_timestamp_ranges"][0]["source_start_pts"] == a
        assert window["source_timestamp_ranges"][0]["source_end_pts"] == b
        if status == "corroborated" and window["direction"] == "ai":
            clip = window["artifacts"]["selected_clip"]
            np.testing.assert_array_equal(read_wav(assets[clip["sha256"]]), pcm[a:b])
        else:
            assert window["artifacts"] == {}
    audio_long_review.verify(review, record, store=store, source=source, pcm=pcm, mapping=source,
                             read_asset=assets.__getitem__, waveform=audio_packets._chart)
    audio_long_review.verify_source(review, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    corrupted = deepcopy(review)
    corrupted[binding.COVERAGE]["intervals"][0]["source_timestamp_ranges"][0]["source_end_pts"] += 1
    with pytest.raises(ValueError, match="sample coverage differs"):
        audio_long_review.verify_source(corrupted, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    legacy = {key: value for key, value in review.items() if key != binding.COVERAGE}
    audio_long_review.verify(legacy, record, store=store, source=source, pcm=pcm, mapping=source,
                             read_asset=assets.__getitem__, waveform=audio_packets._chart)
    moved = deepcopy(review)
    moved["windows"][0]["source_timestamp_ranges"][0]["source_end_pts"] -= 1
    with pytest.raises(ValueError, match="clock"):
        audio_long_review.verify_source(moved, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    changed_pcm = pcm.copy()
    changed_pcm[0] += 1 / 32768
    with pytest.raises(ValueError, match="PCM differs"):
        audio_long_review.verify_source(review, pcm=changed_pcm, mapping=source, waveform=audio_packets._chart)


def test_auxiliary_run_only_copies_the_detect_stage_pin(tmp_path):
    engine = create_engine("sqlite://")
    store = ObjectStore(tmp_path)
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Analysis(id="input", kind="audio", filename="sample.flac", content_type="audio/flac",
                        size_bytes=1, sha256="a" * 64))
        db.add_all([
            DetectorRun(id=run_id, analysis_id="input", detector_id=detector, detector_version="1", status="ok")
            for run_id, detector in (("primary", "audio-test"), ("attempt-a", binding.DETECTOR_ID))
        ])
        db.flush()
        assert [row.id for row in runs.current(db, "input", store)] == ["primary"]
        assert [row.id for row in runs.current(db, "input", store, include_auxiliary=True)] == ["primary"]
        db.add(DetectorRun(id="attempt-z", analysis_id="input", detector_id=binding.DETECTOR_ID, detector_version="1", status="ok"))
        output = register_artefact(db, store, canonical_json({"detectors": [{"run_id": "primary"}],
                                    "long_voice_review": {"run_id": "attempt-a"}}),
                                    media_type="application/json", role="stage_output", created_by="detect", analysis_id="input")
        db.add(AnalysisStage(analysis_id="input", stage="detect", status="done", output_artefact_id=output.id))
        db.flush()
        assert runs.statuses(db, "input", store=store) == ["ok"]
        assert {row.id for row in runs.current(db, "input", store, include_auxiliary=True)} == {"primary", "attempt-a"}


@pytest.mark.parametrize("schema", [binding.FOCUSED_SCHEMA, binding.RESAMPLED_FOCUSED_SCHEMA])
def test_focused_review_retains_input_readings_and_binds_narrower_playback(schema):
    pcm = np.arange(160_001, dtype=np.int32).astype("<i2").astype("<f4") / 32768
    source = native_source(len(pcm)) | {"pcm_sha256": sha256(pcm.tobytes()).hexdigest()}
    if schema == binding.RESAMPLED_FOCUSED_SCHEMA:
        source.update(source_sample_rate=48_000, source_container="wav", source_codec="pcm_s16le")
    schedule = binding.focused_intervals(len(pcm))
    assert schedule == [(0, 48000), (24000, 72000), (48000, 96000), (56000, 104000),
                        (72000, 120000), (96000, 144000), (112001, 160001)]

    def pin(start, end):
        return sha256(pcm[start:end].tobytes()).hexdigest()

    directions = ("ai", "ai", "inconclusive", "ai", "unassessed", "inconclusive", "inconclusive")
    windows = [{"start_sample": a, "end_sample": b, "direction": direction, "pcm_sha256": pin(a, b)}
               for (a, b), direction in zip(schedule, directions, strict=True)]
    if schema == binding.RESAMPLED_FOCUSED_SCHEMA:
        for window in windows:
            window["speech_samples"] = 12000 if window["direction"] == "unassessed" else 24000
    regions = binding.focused_regions(len(pcm), windows)
    assert regions == [{"start_sample": 24000, "end_sample": 48000, "basis": "overlap", "supporting_windows": [0, 1]},
                       {"start_sample": 56000, "end_sample": 104000, "basis": "sparse", "supporting_windows": [0, 3]}]
    for region in regions:
        region["pcm_sha256"] = pin(region["start_sample"], region["end_sample"])
    receipt = {"schema": schema, "status": "corroborated", "reason": binding.FOCUSED_REASONS["corroborated"],
               "windows": windows, "regions": regions}
    raw = canonical_json({"detector_id": binding.DETECTOR_ID, "status": "ok",
                          "details": {"pcm_sha256": source["pcm_sha256"], "long_voice_review": receipt}})
    digest = sha256(raw).hexdigest()
    store = SimpleNamespace(read_bytes={digest: raw}.__getitem__)
    record = {"raw_output_sha256": digest}
    assets = {}

    def asset(data, media_type, **metadata):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **metadata}

    review = audio_long_review.build(record, store=store, source=source, pcm=pcm, mapping=source,
                                    asset=asset, waveform=audio_packets._chart)
    assert review["affects_overall_confidence"] is False and all(not row["artifacts"] for row in review["windows"])
    marked = review["regions"][0]["artifacts"]["selected_clip"]
    assert marked["samples"] == 24000
    np.testing.assert_array_equal(read_wav(assets[marked["sha256"]]), pcm[24000:48000])
    audio_long_review.verify(review, record, store=store, source=source, pcm=pcm, mapping=source,
                             read_asset=assets.__getitem__, waveform=audio_packets._chart)
    audio_long_review.verify_source(review, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    if schema == binding.RESAMPLED_FOCUSED_SCHEMA:
        assert "decoded 16 kHz mono audio" in review["limitation"]
        assert [window["speech_samples"] for window in review["windows"]] == [24000, 24000, 24000, 24000, 12000, 24000, 24000]
        wrong_count = deepcopy(review)
        wrong_count["windows"][0]["speech_samples"] = 23999
        with pytest.raises(ValueError, match="disagrees with long speech-window admission"):
            binding.validate(wrong_count, source)
        missing_count = deepcopy(review)
        del missing_count["windows"][0]["speech_samples"]
        with pytest.raises(ValueError, match="Incomplete measured speech coverage"):
            binding.validate(missing_count, source)
        for altered in ({"source_sample_rate": 44100}, {"source_channels": 2},
                        {"timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 1}]}):
            with pytest.raises(ValueError, match="source scope"):
                binding.validate(review, source | altered)
    changed = deepcopy(review)
    changed["regions"][0]["supporting_windows"] = [0, 2]
    with pytest.raises(ValueError, match="region or source clock"):
        binding.validate(changed, source)
    changed = deepcopy(review)
    changed["regions"][0]["pcm_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="PCM differs"):
        audio_long_review.verify_source(changed, pcm=pcm, mapping=source, waveform=audio_packets._chart)


def test_optional_review_respects_the_serving_model_and_provenance_gate(monkeypatch):
    monkeypatch.setattr(settings, "audio_detector_mode", "candidate")
    ctx = SimpleNamespace(profile=Profile.FULL, deadline=Deadline(5), state={},
                          outputs={"decode": {"long_voice_review_source_supported": True}})
    primary = [{"detector_id": long_voice_review.PellaVoiceDetector.spec.id, "status": "not_applicable"}]
    assert audio._long_voice_review(ctx, None, [])["reason"] == binding.SKIPPED_UNAVAILABLE
    assert audio._long_voice_review(ctx, None, primary)["reason"] == binding.SKIPPED_BUDGET
    monkeypatch.setattr(audio.provenance_gate, "stop", lambda _: True)
    assert audio._long_voice_review(ctx, None, primary) is None


@pytest.mark.parametrize(("schema", "with_confirmation", "marked"), [
    (two_second.SCHEMA, False, True),
    (two_second.CONFIRMED_SCHEMA, False, False),
    (two_second.CONFIRMED_SCHEMA, True, True),
])
def test_two_second_scan_deduplicates_source_ranges_and_replays_confirmation(schema, with_confirmation, marked):
    pcm = np.arange(160_001, dtype=np.int32).astype("<i2").astype("<f4") / 32768
    source = native_source(len(pcm)) | {"pcm_sha256": sha256(pcm.tobytes()).hexdigest()}
    starts = two_second.excerpt_starts(len(pcm))
    assert starts == [0, 32_000, 64_000, 64_001]
    assert not two_second.excerpt_starts(two_second.MINIMUM - 1)
    assert not two_second.excerpt_starts(two_second.MAXIMUM + 1)
    readings = []
    for start, chosen in zip(starts, (4, 2, 0, None), strict=True):
        nearby = {0: 3, 32_000: 3}.get(start) if with_confirmation else None
        rows = [] if chosen is None else [
            {"start_sample": a, "end_sample": b, "logit": short.REVIEW_POINT - 1 + 3 * int(index == chosen) + 2 * int(index == nearby)}
            for index, (a, b) in enumerate(short.GRID)
        ]
        readings.append({"start_sample": start, "pcm_sha256": sha256(pcm[start:start+short.SAMPLES].tobytes()).hexdigest(),
                         "speech_samples": [short.RATE if rows else 0] * len(short.GRID), "responses": rows})
    raw = canonical_json({"detector_id": two_second.DETECTOR_ID, "status": "ok", "details": {
        "pcm_sha256": source["pcm_sha256"], two_second.KEY: {"schema": schema, "excerpts": readings}}})
    pin = sha256(raw).hexdigest()
    store = SimpleNamespace(read_bytes={pin: raw}.__getitem__)
    record = long_voice_review.skipped(binding.SKIPPED_BUDGET) | {two_second.KEY: {"raw_output_sha256": pin}}
    assets = {}

    def asset(data, media_type, **metadata):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **metadata}

    review = audio_long_review.build(record, store=store, source=source, pcm=pcm, mapping=source, asset=asset, waveform=audio_packets._chart)
    scan = review[two_second.KEY]
    assert scan["status"] == ("flagged" if marked else "no_flag") and scan["affects_overall_confidence"] is False
    assert all("responses" not in excerpt and ("elevated_windows" in excerpt) == (schema == two_second.CONFIRMED_SCHEMA)
               for excerpt in scan["excerpts"])
    windows = binding.all_media_windows(review)
    if marked:
        assert windows[0]["supporting_excerpts"] == [0, 1, 2]
    expected = [(64_000, 96_000)] if marked else []
    if with_confirmation:
        expected.append((48_000, 80_000))
        assert windows[1]["supporting_excerpts"] == [0]
    assert [(window["start_sample"], window["end_sample"]) for window in windows] == expected
    for window, (a, b) in zip(windows, expected, strict=True):
        np.testing.assert_array_equal(read_wav(assets[window["artifacts"]["selected_clip"]["sha256"]]), pcm[a:b])
    audio_long_review.verify(review, record, store=store, source=source, pcm=pcm, mapping=source,
                             read_asset=assets.__getitem__, waveform=audio_packets._chart)
    audio_long_review.verify_source(review, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    spoofed = deepcopy(review)
    spoofed[two_second.KEY]["excerpts"][0]["pcm_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="excerpt differs from the original samples"):
        audio_long_review.verify_source(spoofed, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    if marked:
        spoofed = deepcopy(review)
        row = spoofed[two_second.KEY]["regions"][0]
        changed = row["confirming_window"] if with_confirmation else row
        changed["source_timestamp_ranges"][0]["source_start_pts"] += 1
        with pytest.raises(ValueError, match="exhibit differs"):
            binding.validate(spoofed, source)
    if with_confirmation:
        spoofed = deepcopy(review)
        spoofed[two_second.KEY]["excerpts"][0]["elevated_windows"] = [4, 4]
        with pytest.raises(ValueError, match="admitted elevated windows"):
            binding.validate(spoofed, source)
        forged = deepcopy(review)
        forged[two_second.KEY]["regions"][0]["confirming_window"]["pcm_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="PCM differs"):
            audio_long_review.verify_source(forged, pcm=pcm, mapping=source, waveform=audio_packets._chart)


def test_two_second_reciprocal_windows_share_one_exhibit():
    excerpts = [{"start_sample": start, "selected_window": selected, "elevated_windows": [selected]}
                for start, selected in ((0, 2), (32_000, 1))]
    first, repeat = two_second.regions(excerpts, schema=two_second.CONFIRMED_SCHEMA)
    assert "same_pair_as" not in first and repeat["same_pair_as"] == first["start_sample"]
    assert first["confirming_window"]["start_sample"] == repeat["start_sample"]
    assert repeat["confirming_window"]["start_sample"] == first["start_sample"]


def test_two_second_scan_reuses_only_identical_source_windows():
    pcm = np.linspace(0, .5, 160_001, dtype="<f4")
    spans = [(0, 6)]

    class Classifier:
        def __init__(self):
            self.starts = []

        def predict(self, chunks):
            window, = chunks
            first = float(window[0])
            self.starts.append(first)
            return [.2 + first]

    cached, independent = Classifier(), Classifier()
    rows = extended_short_voice_review.scan(pcm, SimpleNamespace(spans=lambda _: spans), cached, Deadline(None))
    for row in rows:
        start = row["start_sample"]
        expected = short_voice_review.scan(pcm[start:start + short.SAMPLES], spans, independent, Deadline(None))
        assert row["responses"] == expected["responses"] and row["speech_samples"] == expected["speech_counts"]
    assert set(cached.starts) == set(independent.starts)
    assert len(cached.starts) == len(set(cached.starts)) < len(independent.starts)


def test_two_second_aac_review_ends_at_presented_samples_and_binds_native_ticks():
    presented = 160_001
    pcm = np.linspace(-.5, .5, presented + 127, dtype="<f4")
    source = native_source(len(pcm)) | {
        "source_container": short.AAC_CONTAINER, "source_codec": "aac", "source_profile": "LC",
        "source_sample_rate": 48_000, "source_time_base": "1/48000", "source_bit_rate": 96_000,
        "source_start_pts": 0, "source_duration_pts": 3 * presented, short.LOSSLESS_NATIVE_CLOCK: True,
        "pcm_sha256": sha256(pcm.tobytes()).hexdigest(),
    }
    assert two_second.presented_samples(source) == presented
    assert two_second.presented_samples(source | {short.LOSSLESS_NATIVE_CLOCK: False}) is None
    assert two_second.presented_samples(source | {"source_duration_pts": 3 * presented + 1}) is None
    assert two_second.presented_samples(source | {"source_start_pts": 480}) is None
    assert two_second.presented_samples(source | {"source_bit_rate": 160_000}) is None

    readings = []
    for start in two_second.excerpt_starts(presented):
        excerpt = pcm[start:start + short.SAMPLES]
        scores = [{"start_sample": a, "end_sample": b, "logit": short.AAC_REVIEW_POINT + (1 if i > 2 else -1)}
                  for i, (a, b) in enumerate(short.GRID)]
        readings.append({"start_sample": start, "pcm_sha256": sha256(excerpt.tobytes()).hexdigest(),
                         "speech_samples": [short.RATE] * len(short.GRID), "responses": scores})
    raw = canonical_json({"detector_id": two_second.DETECTOR_ID, "status": "ok", "details": {
        "pcm_sha256": source["pcm_sha256"], two_second.KEY: {"schema": two_second.PRESENTED_SCHEMA, "excerpts": readings}}})
    pin = sha256(raw).hexdigest()
    store, assets = SimpleNamespace(read_bytes={pin: raw}.__getitem__), {}

    def asset(data, media_type, **metadata):
        digest = sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": media_type, **metadata}

    record = {"raw_output_sha256": pin}
    review = audio_long_review.build_presented(record, store=store, source=source, pcm=pcm, mapping=source,
                                             asset=asset, waveform=audio_packets._chart)
    assert review["status"] == "flagged"
    assert review["excerpts"][-1]["start_sample"] + short.SAMPLES == presented
    assert all(window["end_sample"] <= presented for window in two_second.media_windows(review))
    audio_long_review.verify(review, record, store=store, source=source, pcm=pcm, mapping=source,
                             read_asset=assets.__getitem__, waveform=audio_packets._chart)
    audio_long_review.verify_presented_source(review, pcm=pcm, mapping=source, waveform=audio_packets._chart)
    review["regions"][0]["confirming_window"]["native_source_range"]["end_pts"] += 1
    with pytest.raises(ValueError, match="differs from the native source clock"):
        two_second.validate(review, source)
