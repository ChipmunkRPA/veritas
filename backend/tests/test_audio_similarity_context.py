"""Test paired similarity evidence against real WAV and encoded soundtrack inputs."""
from __future__ import annotations

import hashlib
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_audio_evidence import audio_fixture  # noqa: F401
from test_video_nested_evidence import _check_public_zip

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.audio.decode import decode_audio, read_wav, source_ranges, wav_bytes
from app.detectors.video.decode import probe_frames
from app.evidence import (
    audio_exact_repetition,
    audio_packets,
    audio_similarity,
    audio_similarity_binding,
    service,
    video_packets,
)
from app.reports import engine as reports_engine
from app.reports import multimedia_evidence
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base
from app.storage.objects import ObjectStore

RATE = 16000
COUNT = 3 * RATE
FIRST, SECOND, LENGTH = 5120, 28160, 6400


def _similar_pcm():
    rng = np.random.default_rng(81211)
    pcm = np.rint(rng.normal(0, 4500, COUNT)).astype("<i2")
    change = np.rint(rng.normal(0, 380, LENGTH)).astype("<i2")
    pcm[SECOND:SECOND + LENGTH] = pcm[FIRST:FIRST + LENGTH] + change
    return pcm


def _normalize(samples):
    return samples.astype("<f4") / np.float32(32768)


def _source(samples, *, timeline=None):
    return {"sha256": "a" * 64, "pcm_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
            "samples": len(samples), "sample_rate": RATE, "timeline_time_base": "1/16000",
            "timeline": timeline or [{"start_sample": 0, "end_sample": len(samples), "source_pts": 16000}]}


def _assets():
    assets = {}

    def save(data, media_type, **meta):
        sha = hashlib.sha256(data).hexdigest()
        assets[sha] = data
        return {"sha256": sha, "media_type": media_type, **meta, "url": f"/api/v2/assets/{sha}.wav"}

    return assets, save


@pytest.mark.parametrize("kind", ["audio", "video"])
def test_published_similarity_freezes_two_distinct_exact_wavs_and_source_clocks(tmp_path, monkeypatch, kind):
    original = _similar_pcm()
    assert not audio_exact_repetition.find_repeated_spans(original.tobytes()).spans
    source = tmp_path / "sample.wav"
    source.write_bytes(wav_bytes(_normalize(original)))
    if kind == "video":
        encoded = tmp_path / "offset.mkv"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=size=96x64:rate=25:d=4",
                        "-itsoffset", "0.4", "-i", str(source), "-map", "0:v", "-map", "1:a", "-c:v", "ffv1",
                        "-c:a", "libopus", "-b:a", "64k", "-output_ts_offset", "5", str(encoded)], check=True, timeout=20)
        source = encoded
    store = ObjectStore(tmp_path / "objects")
    source_record = store.put_file(source)
    decoded = decode_audio(source, source_sha256=source_record.sha256)
    original_origin = decoded.mapping["timeline"][0]["source_pts"]
    assert original_origin > 5 * RATE if kind == "video" else original_origin == 0
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    snapshot = specimen(kind, "standard")
    with Session(engine) as db:
        analysis = Analysis(id=snapshot.analysis.id, kind=kind, filename=source.name,
                            content_type="audio/wav" if kind == "audio" else "video/x-matroska",
                            size_bytes=source_record.size_bytes, sha256=source_record.sha256, status="done", revision=1)
        db.add(analysis)
        db.flush()
        wav = register_artefact(db, store, decoded.wav, media_type="audio/wav", role="working_copy",
                                created_by="audio-input/1", analysis_id=analysis.id)
        mapping = register_artefact(db, store, canonical_json(decoded.mapping), media_type="application/json",
                                    role="stage_output", created_by="audio-input/1", analysis_id=analysis.id)
        binding = {"version": "audio-input/1", "pcm_sha256": decoded.mapping["pcm_sha256"], "sample_rate": RATE,
                   "samples": len(decoded.pcm), "wav_sha256": wav.sha256, "mapping_sha256": mapping.sha256}
        result = deepcopy(snapshot.result)
        result["input"]["sha256"] = source_record.sha256
        sampling = result["manifest"]["sampling"] = {}
        result.update(signals=[], findings=[], exhibits=[])
        if kind == "audio":
            sampling["audio_evidence"] = {"input": binding, "checks": []}
        else:
            table = probe_frames(source).to_json()
            anchors = []
            for stage, data in (("probe", {"frame_table": table, "info": {"has_audio": True}}),
                                ("render", {"frame_anchor_version": video_packets.ANCHOR_VERSION, "frame_anchors": anchors})):
                record = register_artefact(db, store, canonical_json(data), media_type="application/json", role="stage_output",
                                           created_by=stage, analysis_id=analysis.id)
                db.add(AnalysisStage(analysis_id=analysis.id, stage=stage, status="done", output_artefact_id=record.id))
            sampling.update(video_audio_evidence={"input": binding, "voice_checks": [], "source_audio_present": True},
                            frame_anchor_version=video_packets.ANCHOR_VERSION, frame_table_sha256=canonical_hash(table),
                            frame_anchor_sha256=canonical_hash(anchors), timeline_origin_s=5.0)
        analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
        db.flush()
        document = service.persist_packet_set(db, analysis, store, result=result)
        assert service.load_packet_set(db, analysis, store) == document
        content = document if kind == "audio" else document["audio"]
        context = content["similarity_context"]
        assert context["status"] == "measured" and context["affects_overall_confidence"] is False
        assert content["repetition_context"]["pairs"] == []
        [pair] = context["pairs"]
        assert pair["id"] == "RS01" and pair["lag_samples"] == SECOND - FIRST
        assert pair["sample_count"] >= LENGTH - 640
        assert pair["artifacts"]["first_clip"]["sha256"] != pair["artifacts"]["second_clip"]["sha256"]
        for name, expected in (("first", FIRST), ("second", SECOND)):
            part = pair[name]
            start, stop = part["start_sample"], part["end_sample"]
            assert min(stop, expected + LENGTH) - max(start, expected) >= LENGTH - 640
            assert part["source_timestamp_ranges"] == source_ranges(decoded.mapping, start, stop)
            assert part["source_timestamp_ranges"][0]["source_start_pts"] == original_origin + start
            assert part["pcm_s16le_sha256"] == hashlib.sha256(np.rint(decoded.pcm[start:stop] * 32768).astype("<i2").tobytes()).hexdigest()
            asset = pair["artifacts"][f"{name}_clip"]
            np.testing.assert_array_equal(read_wav(store.read_bytes(asset["sha256"])), decoded.pcm[start:stop])
        audio_similarity.verify(context, pcm=decoded.pcm, mapping=decoded.mapping, source=content["source"], read_asset=store.read_bytes)
        with pytest.raises(ValueError, match="playback clip"):
            audio_similarity.verify(context, pcm=decoded.pcm, mapping=decoded.mapping, source=content["source"], read_asset=lambda _: b"bad")

        snapshot = snapshot.model_copy(update={
            "result": result, "evidence": document,
            "media": snapshot.media.model_copy(update={"sha256": source_record.sha256, "size_bytes": source_record.size_bytes,
                                                       "filename": analysis.filename, "content_type": analysis.content_type}),
            "report": snapshot.report.model_copy(update={"template_set": reports_engine.TEMPLATE_SET}),
        })
        monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
        rendered = reports_engine.render_pdf(snapshot)
        text = " ".join(reports_engine.pdf_text(rendered.pdf).split())
        compact = "".join(text.split())
        assert "Similar repeated audio" in text and "RS01" in text and "similar sound" in text
        assert ("/audio/similarity_context" if kind == "video" else "/similarity_context") in text
        asset_refs = service.asset_refs(document)
        for role in ("first", "second"):
            span = pair[role]["source_timestamp_ranges"][0]
            start, end = span["source_start_pts"], span["source_end_pts"]
            assert f"Source PTS [{start}, {end})" in text
            clip = pair["artifacts"][f"{role}_clip"]
            assert clip["sha256"] in asset_refs and clip["sha256"] in compact and pair[role]["pcm_s16le_sha256"] in compact
            if kind == "video":
                assert f"Video playback {start / RATE - 5:.3f}–{end / RATE - 5:.3f} s" in text
        redacted = snapshot.model_copy(update={"report": snapshot.report.model_copy(update={"redacted": True})})
        html = reports_engine.render_html(redacted, embed_fonts=False)
        assert "Similar repeated audio" in html and all(clip["url"] not in html for clip in pair["artifacts"].values())
        assert "data:audio/" not in html and "<audio" not in html
        _check_public_zip(SimpleNamespace(db=db, store=store, analysis=analysis, frozen=snapshot, document=document),
                          tmp_path, monkeypatch, rendered.pdf)


@pytest.mark.parametrize("mode,status,reason", [
    ("short", "unavailable", "duration_out_of_scope"),
    ("long", "unavailable", "duration_out_of_scope"),
    ("gap", "unavailable", "source_clock_discontinuous"),
    ("quiet", "unavailable", "insufficient_active_bands"),
    ("periodic", "ambiguous", "span_budget"),
    ("unrelated", "measured", None),
])
def test_bounded_search_reports_ineligible_inputs_and_ambiguous_pairs(mode, status, reason):
    count = {"short": 39999, "long": 64001}.get(mode, COUNT)
    raw = np.random.default_rng(124).integers(-10000, 10000, size=count, dtype="<i2")
    timeline = None
    if mode == "quiet":
        raw.fill(0)
    elif mode == "periodic":
        x = np.arange(count) * (2 * np.pi / RATE)
        raw = np.rint(3000 * np.sin(x * 500) + 3200 * np.sin(x * 1800) + 2600 * np.sin(x * 3200)).astype("<i2")
    elif mode == "gap":
        timeline = [{"start_sample": 0, "end_sample": 16000, "source_pts": 0},
                    {"start_sample": 16000, "end_sample": COUNT, "source_pts": 32000}]
    pcm = _normalize(raw)
    source = _source(pcm, timeline=timeline)
    context = audio_similarity.build(pcm, {**source, "source_sha256": source["sha256"]}, source,
                                     lambda *args, **kwargs: pytest.fail("No pair expected"))
    assert (context["status"], context["reason"], context["pairs"]) == (status, reason, [])
    assert (context["searched_interval"] is None) == (mode in {"short", "long", "gap"})


def test_similarity_binding_rejects_retiming_support_and_conflicting_assets():
    pcm = _normalize(_similar_pcm())
    source = _source(pcm)
    mapping = {**source, "source_sha256": source["sha256"]}
    assets, save = _assets()
    context = audio_similarity.build(pcm, mapping, source, save)
    retimed = deepcopy(context)
    retimed["pairs"][0]["second"]["source_timestamp_ranges"][0]["source_start_pts"] += 1
    with pytest.raises(ValueError, match="source timestamps"):
        audio_similarity_binding.validate(retimed, source)
    wrong_support = deepcopy(context)
    wrong_support["pairs"][0]["support_windows"] += 1
    with pytest.raises(ValueError, match="ranges or support"):
        audio_similarity_binding.validate(wrong_support, source)
    wrong_clip = deepcopy(context)
    wrong_clip["pairs"][0]["artifacts"]["second_clip"] = wrong_clip["pairs"][0]["artifacts"]["first_clip"]
    with pytest.raises(ValueError, match="contradict"):
        audio_similarity_binding.validate(wrong_clip, source)
    changed_pcm = pcm.copy()
    changed_pcm[0] += np.float32(1 / 32768)
    with pytest.raises(ValueError, match="samples differ"):
        audio_similarity.verify(context, pcm=changed_pcm, mapping=mapping, source=source, read_asset=assets.__getitem__)


def test_legacy_audio_packet_bytes_are_preserved_without_similarity(audio_fixture, monkeypatch):  # noqa: F811
    db, store, analysis, result, _, _ = audio_fixture
    build_content = audio_packets.build_content

    def historical_content(*args, **kwargs):
        content = build_content(*args, **kwargs)
        del content["similarity_context"]
        return content

    with monkeypatch.context() as scoped:
        scoped.setattr(audio_packets, "build_content", historical_content)
        historical = audio_packets.persist_packet_set(db, analysis, store, result=result)
    encoded = canonical_json(historical)
    digest = hashlib.sha256(encoded).hexdigest()
    assert "similarity_context" not in historical and store.read_bytes(digest) == encoded
    monkeypatch.setattr(audio_packets, "build_packet_set", lambda *a, **k: pytest.fail("Historical set must not be rebuilt"))
    assert audio_packets.persist_packet_set(db, analysis, store, result=result) == historical
    assert store.read_bytes(digest) == encoded
