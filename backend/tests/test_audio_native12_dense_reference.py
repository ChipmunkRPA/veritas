"""Native twelve-second opt-in, historical parity and source-bound excerpts."""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import subprocess
import wave
from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_image_reference_access import reference_http as reference_http

from app.api import v2_media_reference
from app.core.canonical import canonical_hash
from app.core.config import Settings, settings
from app.evidence import (
    audio_clean_speech_dense,
    audio_clean_speech_long,
    audio_clean_speech_long_dense,
    audio_reference,
)
from app.storage.db import Analysis, Artefact, BillingOrderRow, Job

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


def authored_pair():
    rng = np.random.default_rng(172)
    shape = butter(2, (260, 3600), fs=16000, btype="bandpass", output="sos")
    first, donor, background = [sosfiltfilt(shape, rng.standard_normal(192000)) * scale
                                for scale in (2000, 1800, 1600)]
    latter = first.copy()
    latter[21760:37120] = .60 * first[21760:37120] + .4 * donor[21760:37120]
    return tuple(np.rint(v).astype("<i2") for v in (first, latter + .7 * background))


def save_wav(path, pcm, *, rate=16000, channels=1):
    with wave.open(str(path), "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", ""))
        output.writeframes(pcm.astype("<i2").tobytes())
    return path


def test_long_dense_reproduces_the_frozen_authored_fallback():
    first, second = authored_pair()
    assert audio_clean_speech_long.locate(first, second) == ("no_corroborated_local_change", None, None)
    reason, measured, profile, method = audio_clean_speech_long_dense.locate(first, second)
    assert reason == "local_change" and profile is None and method == "dense_fallback"
    assert measured["samples"] == [21760, 37120]
    assert measured["supporting_window_starts"] == [0, 19200]
    assert measured["relative_drop"] == .39118701


def test_dense_fallback_preserves_the_existing_interval_without_retry(monkeypatch):
    pcm = np.zeros(192000, dtype="<i2")
    retained = {"samples": [32000, 43520], "relative_drop": .4, "supporting_window_starts": [0, 19200]}
    monkeypatch.setattr(audio_clean_speech_long, "locate", lambda *_: ("local_change", retained, None))
    monkeypatch.setattr(audio_clean_speech_dense, "locate_with_match_support", lambda *_: pytest.fail("Retained ranges stay first"))
    assert audio_clean_speech_long_dense.locate(pcm, pcm) == ("local_change", retained, None, "retained")


@pytest.mark.parametrize("reason", ["reference_match_not_supported", "consistent_frequency_balance_change"])
def test_long_dense_preserves_unassessed_and_broad_processing_guards(monkeypatch, reason):
    pcm = np.zeros(192000, dtype="<i2")
    profile = {"bands_hz": [[300, 850]]} if reason == "consistent_frequency_balance_change" else None
    monkeypatch.setattr(audio_clean_speech_long, "locate", lambda *_: (reason, None, profile))
    monkeypatch.setattr(audio_clean_speech_dense, "locate_with_match_support", lambda *_: pytest.fail("Admission stays fixed"))
    assert audio_clean_speech_long_dense.locate(pcm, pcm) == (reason, None, profile, None)


@pytest.mark.parametrize("mode", ["corroborated", "one_window", "different_ranges"])
def test_dense_windows_need_neighbor_agreement_and_keep_absolute_clocks(monkeypatch, mode):
    pcm = np.zeros(192000, dtype="<i2")
    baseline = "no_corroborated_local_change"
    monkeypatch.setattr(audio_clean_speech_long, "locate", lambda *_: (baseline, None, None))
    calls = []

    def measure(first, second):
        assert first.shape == second.shape == (96000,)
        index = len(calls)
        calls.append(index)
        if index == 0:
            return "local_change", {"samples": [32000, 47360], "relative_drop": .7}, None, {}
        if index == 1 and mode != "one_window":
            start = 12800 if mode == "corroborated" else 20800
            return "local_change", {"samples": [start, start + 15360], "relative_drop": .8}, None, {}
        return "no_local_change", None, None, {}

    monkeypatch.setattr(audio_clean_speech_dense, "locate_with_match_support", measure)
    reason, result, _, method = audio_clean_speech_long_dense.locate(pcm, pcm)
    assert calls == list(range(6))
    if mode == "corroborated":
        assert reason == "local_change" and method == "dense_fallback"
        assert result["samples"] == [32000, 47360] and result["supporting_window_starts"] == [0, 19200]
    else:
        assert (reason, result, method) == (baseline, None, None)


@pytest.mark.parametrize("samples", [np.zeros(96000, dtype="<i2"), np.zeros(192001, dtype="<i2"),
                                     np.zeros(192000, dtype=np.float32), np.zeros((192000, 2), dtype="<i2")])
def test_core_dense_lane_requires_exact_native_pair_shape(samples):
    with pytest.raises(ValueError, match="twelve-second native mono PCM16"):
        audio_clean_speech_long_dense.locate(samples, samples)


@pytest.mark.parametrize("extension", ["wav", "flac"])
def test_new_native_schema_keeps_source_clips_and_default_export_parity(tmp_path, extension):
    first, second = authored_pair()
    earlier, submitted = (tmp_path / f"{side}.{extension}" for side in ("earlier", "submitted"))
    for path, pcm in ((earlier, first), (submitted, second)):
        sf.write(path, pcm, 16000, subtype="PCM_16")
    baseline, baseline_assets = audio_reference.compare(earlier, submitted, include_clean_speech=True)
    disabled, disabled_assets = audio_reference.compare(earlier, submitted, include_clean_speech=True,
                                                       use_dense_long_clean_speech=False)
    assert baseline == disabled and baseline_assets == disabled_assets
    assert baseline["schema"] == audio_reference.SCHEMA_WITH_LONG_CLEAN_SPEECH
    assert baseline["clean_speech"]["reason"] == "no_corroborated_local_change"
    record, assets = audio_reference.compare(earlier, submitted, include_clean_speech=True, use_dense_long_clean_speech=True)
    assert record["schema"] == audio_reference.SCHEMA_WITH_DENSE_LONG_CLEAN_SPEECH
    speech = record["clean_speech"]
    assert speech["basis"] == "dense_long_clean_reference_v1" and speech["range_method"] == "dense_fallback"
    assert speech["sample_count"] == 192000 and speech["affects_overall_confidence"] is False
    assert record["affects_overall_confidence"] is False and record["audio"] == baseline["audio"]
    interval = speech["range"]
    assert interval["samples"] == [21760, 37120] and interval["supporting_window_starts"] == [0, 19200]
    measured = {key: value for key, value in interval.items() if key not in {"earlier", "submitted", "support_sha256"}}
    assert interval["support_sha256"] == canonical_hash(measured)
    for side, pcm in (("earlier", first), ("submitted", second)):
        excerpt = interval[side]
        assert excerpt["samples"] == excerpt["source_pts"] == [21760, 37120]
        assert (excerpt["start_s"], excerpt["end_s"]) == (1.36, 2.32)
        raw = pcm[21760:37120].tobytes()
        assert excerpt["pcm_sha256"] == hashlib.sha256(raw).hexdigest()
        asset = excerpt["audio"]
        with wave.open(io.BytesIO(assets[asset["file"]]), "rb") as reader:
            assert reader.getparams()[:4] == (1, 2, 16000, 15360) and reader.readframes(15360) == raw


def test_retained_native_interval_and_no_interval_use_distinct_metadata(tmp_path):
    first, _ = authored_pair()
    edited = first.copy()
    edited[145920:157440] = first[64000:75520]
    earlier = save_wav(tmp_path / "earlier.wav", first)
    for name, pcm in (("retained", edited), ("unchanged", first), ("unrelated", first[::-1].copy())):
        path = save_wav(tmp_path / f"{name}.wav", pcm)
        old, _ = audio_reference.compare(earlier, path, include_clean_speech=True)
        record, _ = audio_reference.compare(earlier, path, include_clean_speech=True, use_dense_long_clean_speech=True)
        speech = record["clean_speech"]
        assert record["schema"] == audio_reference.SCHEMA_WITH_DENSE_LONG_CLEAN_SPEECH
        if name == "retained":
            assert speech["range_method"] == "retained" and speech["range"] == old["clean_speech"]["range"]
        else:
            assert "range_method" not in speech and "range" not in speech
            assert speech["reason"] == old["clean_speech"]["reason"]
            assert speech["status"] == ("unassessed" if name == "unrelated" else "no_local_difference")


@pytest.mark.parametrize("profile", ["six_seconds", "stereo", "rate_48000", "aac"])
def test_new_option_leaves_other_format_lanes_unchanged(tmp_path, monkeypatch, profile):
    first, second = authored_pair()
    earlier = save_wav(tmp_path / "earlier.wav", first[:96000] if profile == "six_seconds" else first)
    submitted = tmp_path / "submitted.wav"
    if profile == "six_seconds":
        save_wav(submitted, second[:96000])
    elif profile == "stereo":
        save_wav(submitted, np.column_stack((second, second)), channels=2)
    elif profile == "rate_48000":
        save_wav(submitted, np.clip(np.rint(resample_poly(second.astype(float), 3, 1)), -32768, 32767), rate=48000)
    else:
        wav = save_wav(submitted, second)
        submitted = tmp_path / "submitted.m4a"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav), "-ar", "48000", "-ac", "1",
                        "-c:a", "aac", "-b:a", "96k", str(submitted)], check=True, timeout=15)
    monkeypatch.setattr(audio_clean_speech_long_dense, "locate", lambda *_: pytest.fail("Other format lanes stay unchanged"))
    options = {"include_clean_speech": True, "include_clean_speech_aac": profile == "aac"}
    old, old_assets = audio_reference.compare(earlier, submitted, **options)
    result, assets = audio_reference.compare(earlier, submitted, use_dense_long_clean_speech=True, **options)
    assert result == old and assets == old_assets
    assert result["schema"] != audio_reference.SCHEMA_WITH_DENSE_LONG_CLEAN_SPEECH


def test_nonzero_native_origin_declines_without_dense_retry(monkeypatch):
    pcm = np.zeros(192000, dtype=np.float32)
    tracks = tuple((("measured", pcm, {"samples": 192000, "source_pts": origin}), True, True, None, True)
                   for origin in (0, 80))
    monkeypatch.setattr(audio_clean_speech_long_dense, "locate", lambda *_: pytest.fail("A shared native clock is required"))
    schema, result = audio_reference._clean_speech(tracks, {}, allow_long=True, use_dense=False, use_eq_guard=False,
                                                  include_match_support=True, use_dense_long=True)
    assert schema == audio_reference.SCHEMA_WITH_LONG_CLEAN_SPEECH
    assert result["status"] == "unassessed" and result["reason"] == "common_clock_required"
    assert "basis" not in result and "range_method" not in result


def test_new_feature_stays_default_off_and_requires_authenticated_clean_copy(reference_http, tmp_path, monkeypatch):
    assert Settings.model_fields["audio_reference_dense_long_clean_speech"].default is False
    fixture = reference_http
    first, second = authored_pair()
    earlier = save_wav(tmp_path / "earlier.wav", first)
    submitted = save_wav(tmp_path / "submitted.wav", second)
    original = fixture.store.put_file(submitted)

    @contextmanager
    def sessions():
        with Session(fixture.engine) as db:
            yield db

    monkeypatch.setattr(v2_media_reference, "get_session", sessions)
    monkeypatch.setattr(v2_media_reference, "default_store", lambda: fixture.store)
    for flag in ("audio_reference_comparison_enabled", "audio_reference_clean_speech", "audio_reference_long_clean_speech"):
        monkeypatch.setattr(settings, flag, True)
    for flag in ("audio_reference_local_timing", "audio_reference_aligned_speech", "audio_reference_matching_waveform",
                 "audio_reference_clean_speech_aac", "audio_reference_dense_long_clean_speech"):
        monkeypatch.setattr(settings, flag, False)
    with sessions() as db:
        row = db.get(Analysis, "other-kind")
        row.kind, row.filename, row.content_type = "audio", "submitted.wav", "audio/wav"
        row.sha256, row.size_bytes = original.sha256, original.size_bytes
        retained_result = row.result_hash
        db.commit()
        retained_counts = tuple(len(db.scalars(select(model.id)).all()) for model in (Analysis, Artefact, BillingOrderRow, Job))
    retained_objects = {path.relative_to(fixture.store.root): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in fixture.store.root.rglob("*") if path.is_file()}
    route = "/api/v2/analyses/other-kind/audio-reference-comparison"
    files = {"earlier": ("earlier.wav", earlier.read_bytes(), "audio/wav")}
    old_availability = fixture.client.get(route, headers=fixture.viewer).json()
    assert "dense_long_clean_speech_available" not in old_availability
    monkeypatch.setattr(settings, "audio_reference_dense_long_clean_speech", True)
    assert fixture.client.get(route, headers=fixture.viewer).json() == old_availability | {"dense_long_clean_speech_available": True}
    assert fixture.client.post(route, files=files, data={"earlier_has_clean_speech": "true"}).status_code == 401
    without_confirmation = fixture.client.post(route, headers=fixture.viewer, files=files)
    assert without_confirmation.status_code == 200 and "clean_speech" not in without_confirmation.json()["comparison"]
    result = fixture.client.post(route, headers=fixture.viewer, files=files, data={"earlier_has_clean_speech": "true"})
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    response = result.json()
    assert response["comparison"]["schema"] == audio_reference.SCHEMA_WITH_DENSE_LONG_CLEAN_SPEECH
    assert response["comparison"]["clean_speech"]["range_method"] == "dense_fallback"
    assert response["comparison"]["clean_speech"]["range"]["samples"] == [21760, 37120]
    assert "pcm_s16le" not in result.text and "model" not in response["comparison"]["clean_speech"]
    for value in response["assets"].values():
        assert base64.b64decode(value, validate=True)
    with sessions() as db:
        assert db.get(Analysis, "other-kind").result_hash == retained_result
        assert tuple(len(db.scalars(select(model.id)).all()) for model in (Analysis, Artefact, BillingOrderRow, Job)) == retained_counts
    assert {path.relative_to(fixture.store.root): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in fixture.store.root.rglob("*") if path.is_file()} == retained_objects
