from __future__ import annotations

import hashlib
import io
import json
import subprocess
import uuid
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.video.adapters import swap as swap_adapter
from app.detectors.video.decode import DecodedFrame, probe_frames
from app.detectors.video.frame_store import FrameStore
from app.detectors.video.frames import probe
from app.detectors.video.swap import Window
from app.evidence.image_packets import PacketIntegrityError
from app.evidence.video_packets import (
    ANCHOR_VERSION,
    build_packet_set,
    load_packet_set,
    packet_hash,
    persist_packet_set,
    validate_packet_set,
)
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Artefact, Base
from app.storage.objects import ObjectStore


@pytest.fixture(scope="module")
def vfr_source(tmp_path_factory):
    path = tmp_path_factory.mktemp("video-source")
    concat = []
    for i, duration in enumerate([0.04, 0.12, 0.08, 0.20, 0.04]):
        yy, xx = np.mgrid[:64, :96]
        rgb = np.stack([(xx + 30*i) % 256, (yy * 3 + 20*i) % 256, (xx + yy + 10*i) % 256], axis=2).astype(np.uint8)
        Image.fromarray(rgb).save(path / f"frame{i}.png")
        concat += [f"file 'frame{i}.png'", f"duration {duration}"]
    concat.append("file 'frame4.png'")
    (path / "frames.txt").write_text("\n".join(concat))
    video = path / "offset-vfr.mkv"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(path / "frames.txt"),
                    "-fps_mode", "vfr", "-c:v", "ffv1", "-output_ts_offset", "5", str(video)], check=True, timeout=30)
    table = probe_frames(video)
    assert table.seconds(0) == pytest.approx(5.0)
    assert len(set(np.diff(table.pts))) > 1
    return video, table


@pytest.fixture
def video_fixture(tmp_path, vfr_source):
    video, table = vfr_source
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    store = ObjectStore(tmp_path / "objects")
    with Session(engine) as db:
        original = store.put_file(video)
        analysis = Analysis(id="video-test", kind="video", filename="video.mkv", content_type="video/x-matroska",
                            size_bytes=original.size_bytes, sha256=original.sha256, status="done", revision=1,
                            result_schema_version=2)
        db.add(analysis)
        db.flush()
        frames = FrameStore(store.path_for(original.sha256), table)
        pixels = frames.get_many(range(5))
        anchors = frames.native_anchors()
        probing = {"frame_table": table.to_json(), "info": probe(video)}
        rendering = {"frame_anchor_version": ANCHOR_VERSION, "frame_anchors": anchors}
        for name, data in [("probe", probing), ("render", rendering)]:
            art = register_artefact(db, store, canonical_json(data), media_type="application/json", role="stage_output",
                                    created_by=name, analysis_id=analysis.id)
            db.add(AnalysisStage(analysis_id=analysis.id, stage=name, status="done", output_artefact_id=art.id))
        signal = {"id": "vid.swap.private:track:0", "display_id": "S01", "detector_id": "vid.swap.private", "status": "ok",
                  "family": "learned-face", "exhibit_ids": ["curve"], "name": "PRIVATE MODEL"}
        curve = {"id": "curve", "kind": "series", "media_ref": "frame", "x_unit": "s", "source_signal_ids": [signal["id"]],
                 "series": [{"name": "p", "track_id": "track:0", "points": [[table.seconds(i), p] for i, p in enumerate([.2, .6, .9, .7, .1])]}]}
        tracks = {"id": "track", "kind": "face_track", "track": {"track_id": "track:0"},
                  "frames": [{"frame_index": i, "pts": table.pts[i], "bbox": {"x": 10, "y": 8, "w": 32, "h": 24}} for i in range(5)]}
        result = {"input": {"sha256": original.sha256}, "manifest": {"sampling": {
            "frame_anchor_version": ANCHOR_VERSION, "frame_anchor_sha256": canonical_hash(anchors),
            "frame_table_sha256": canonical_hash(table.to_json()), "timeline_origin_s": probing["info"]["timeline_origin_s"],
            "video_audio_evidence": {"version":"video-audio-inputs/1", "decode_status":"absent", "source_audio_present":False,
                                     "input":None, "voice_checks":[], "av_checks":[]}}},
                  "exhibits": [curve, tracks], "signals": [signal],
                  "findings": [{"id": "F1", "evidence_ids": [signal["id"]]}]}
        analysis.result_hash = canonical_hash(result)
        store.put_bytes(canonical_json(result))
        db.flush()
        yield db, store, analysis, result, table, pixels


def test_vfr_nonzero_origin_and_exact_frame_crops(video_fixture):
    db, store, analysis, result, table, pixels = video_fixture
    before = deepcopy(result)
    document = build_packet_set(db, analysis, store, result)
    assert result == before and analysis.result_hash == canonical_hash(before)
    assert document["sha256"] == packet_hash(document)
    assert document["coverage"]["scored_frame_count"] == 5
    assert len(document["packets"]) == 1
    packet = document["packets"][0]
    assert packet["finding_ids"] == ["F1"] and packet["evidence_type"] == "statistical"
    assert [f["anchor"]["frame_index"] for f in packet["frames"]] == [1, 2, 3]
    for frame in packet["frames"]:
        anchor = frame["anchor"]
        i = anchor["frame_index"]
        assert anchor["pts"] == table.pts[i] and anchor["time_base"] == table.time_base
        assert anchor["stream_time_s"] == pytest.approx(table.seconds(i))
        assert anchor["playback_time_s"] == pytest.approx(table.seconds(i) - 5.0)
        assert anchor["decoded_rgb_sha256"] == hashlib.sha256(pixels[i].tobytes()).hexdigest()
        assets = packet["artifacts"]
        original = Image.open(io.BytesIO(store.read_bytes(assets[frame["artifact_refs"]["original"]]["sha256"])))
        crop = Image.open(io.BytesIO(store.read_bytes(assets[frame["artifact_refs"]["crop"]]["sha256"])))
        np.testing.assert_array_equal(np.asarray(original), pixels[i])
        np.testing.assert_array_equal(np.asarray(crop), pixels[i][8:32, 10:42])
        assert frame["region_meaning"] == "detected_face_context"
    assert "PRIVATE MODEL" not in json.dumps(document)
    assert "not necessarily consecutive" in " ".join(document["limitations"])
    assert document["version"] == "video-evidence-4"
    assert packet["clip"]["status"] == "ready"
    assert packet["clip"]["source_pts"] == list(table.pts)
    assert packet["clip"]["sampled_frame_indices"] == [1, 2, 3]
    assert document["coverage"]["channels"]["audio"]["status"] == "absent"
    assert document["coverage"]["channels"]["av"]["status"] == "absent"
    assert document["audio"]["status"] == document["av"]["status"] == "absent"


@pytest.mark.parametrize("change", ["failed", "negative", "milliseconds", "attribution"])
def test_no_fabricated_frame_candidates(video_fixture, change):
    db, store, analysis, result, _, _ = video_fixture
    if change == "failed":
        result["signals"][0]["status"] = "failed"
    elif change == "negative":
        result["exhibits"][0]["series"][0]["points"] = [[t, .1] for t, p in result["exhibits"][0]["series"][0]["points"]]
    elif change == "milliseconds":
        result["exhibits"][0]["series"][0]["points"] = [[t*1000, p] for t, p in result["exhibits"][0]["series"][0]["points"]]
    else:
        result["exhibits"][0]["kind"] = "heatmap"
        result["exhibits"][0]["semantics"] = "attribution"
    analysis.result_hash = canonical_hash(result)
    document = build_packet_set(db, analysis, store, result)
    assert document["packets"] == []
    assert document["coverage"]["scored_frame_count"] == (5 if change == "negative" else 0)


def test_missing_and_changed_frame_commitments_fail_closed(video_fixture):
    db, store, analysis, result, _, _ = video_fixture
    result["manifest"]["sampling"]["frame_anchor_sha256"] = "f"*64
    analysis.result_hash = canonical_hash(result)
    with pytest.raises(PacketIntegrityError, match="published commitments"):
        build_packet_set(db, analysis, store, result)
    result["manifest"]["sampling"].pop("frame_anchor_version")
    analysis.result_hash = canonical_hash(result)
    with pytest.raises(PacketIntegrityError, match="reanalyse"):
        build_packet_set(db, analysis, store, result)


def test_redecoded_pixels_must_match_inference_pixels(video_fixture, monkeypatch):
    from app.evidence import video_packets
    db, store, analysis, result, _, pixels = video_fixture

    def corrupt(path, table, indices):
        for i in indices:
            changed = pixels[i].copy()
            changed[0, 0, 0] ^= 1
            yield DecodedFrame(i, table.pts[i], changed)

    monkeypatch.setattr(video_packets, "decode_frames", corrupt)
    with pytest.raises(PacketIntegrityError, match="pixels differ"):
        build_packet_set(db, analysis, store, result)


def test_clip_failure_is_explicit_without_discarding_exact_source_frames(video_fixture, monkeypatch):
    from app.evidence import video_packets
    from app.evidence.video_clips import ClipUnavailable
    db, store, analysis, result, _, _ = video_fixture
    def unavailable(*args, **kwargs):
        raise ClipUnavailable('The source clock could not be preserved')
    monkeypatch.setattr(video_packets, 'build_clip', unavailable)
    document = build_packet_set(db, analysis, store, result)
    packet = document['packets'][0]
    assert packet['clip'] == {'status': 'unavailable', 'reason': 'The source clock could not be preserved'}
    assert len(packet['frames']) == 3 and 'context_clip' not in packet['artifacts']


def test_successful_voice_responses_do_not_hide_missing_video_audio_packets(video_fixture, monkeypatch):
    from app.evidence import video_packets
    monkeypatch.setattr(video_packets, 'VERSION', 'video-evidence-2')
    monkeypatch.setattr(video_packets, 'CREATOR', 'video-evidence-packets/2')
    db, store, analysis, result, _, _ = video_fixture
    stage = db.get(AnalysisStage, (analysis.id, 'probe'))
    artifact = db.get(Artefact, stage.output_artefact_id)
    probe = json.loads(store.read_bytes(artifact.sha256))
    probe['info']['has_audio'] = True
    updated = register_artefact(db, store, canonical_json(probe), media_type='application/json', role='stage_output',
                                created_by='probe', analysis_id=analysis.id)
    stage.output_artefact_id = updated.id
    result['signals'] += [{'id': 'aud.historical_voice:global', 'family': 'voice', 'status': 'ok', 'exhibit_ids': []},
                          {'id': 'vid.historical_sync:global', 'family': 'av-sync', 'status': 'failed', 'exhibit_ids': []}]
    analysis.result_hash = canonical_hash(result)
    document = build_packet_set(db, analysis, store, result)
    channels = document['coverage']['channels']
    assert channels['audio']['status'] == 'evidence_unavailable'
    assert channels['audio']['successful_check_count'] == 1
    assert channels['audio']['source_mapping_verified'] is False
    assert channels['av_sync']['status'] == 'not_assessed'
    assert channels['av_sync']['successful_check_count'] == 0
    assert 'historical_voice' not in json.dumps(document) and 'historical_sync' not in json.dumps(document)


def test_v3_older_result_missing_audio_commitments_is_explicitly_unavailable(video_fixture):
    db,store,analysis,result,_,_=video_fixture
    result['manifest']['sampling'].pop('video_audio_evidence')
    analysis.result_hash=canonical_hash(result)
    document=build_packet_set(db,analysis,store,result)
    assert document['audio']['status']==document['av']['status']=='unavailable'
    assert document['audio']['coverage'] is None and document['av']['coverage'] is None
    assert 'Reanalyse' in document['audio']['reason']


def test_old_immutable_packets_remain_readable_without_regeneration(video_fixture, monkeypatch):
    from app.evidence import video_packets
    db, store, analysis, result, _, _ = video_fixture
    document = build_packet_set(db, analysis, store, result)
    document['id'] = str(uuid.uuid5(video_packets._NS, f'{analysis.id}:{analysis.result_hash}:video-evidence-1'))
    document['version'] = 'video-evidence-1'
    document.pop('audio'); document.pop('av')
    document['coverage'].pop('channels')
    for packet in document['packets']:
        packet.pop('clip')
        packet['artifacts'].pop('context_clip')
        packet['derivation']['version'] = 'video-evidence-1'
    document['sha256'] = packet_hash(document)
    data = canonical_json(document)
    row = register_artefact(db, store, data, media_type='application/json', role='exhibit_payload',
                           created_by='video-evidence-packets/1', analysis_id=analysis.id, artefact_id=document['id'])
    def forbid(*args, **kwargs): raise AssertionError('Read tried to regenerate old evidence')
    monkeypatch.setattr(video_packets, 'build_clip', forbid)
    assert load_packet_set(db, analysis, store, set_id=document['id']) == document
    assert store.read_bytes(row.sha256) == data


def test_video_read_is_pure_and_persist_is_idempotent(video_fixture):
    db, store, analysis, result, _, _ = video_fixture
    count = len(db.scalars(select(Artefact)).all())
    assert load_packet_set(db, analysis, store) is None
    assert len(db.scalars(select(Artefact)).all()) == count
    document = persist_packet_set(db, analysis, store, result=result)
    count = len(db.scalars(select(Artefact)).all())
    assert persist_packet_set(db, analysis, store, result=result) == document
    assert load_packet_set(db, analysis, store, set_id=document["id"]) == document
    assert len(db.scalars(select(Artefact)).all()) == count
    assert load_packet_set(db, analysis, store, set_id="wrong-set") is None
    tampered = deepcopy(document)
    tampered["packets"][0]["frames"][0]["anchor"]["playback_time_s"] += 1
    tampered["sha256"] = packet_hash(tampered)
    with pytest.raises(PacketIntegrityError, match="playback time"):
        validate_packet_set(tampered, analysis_id=analysis.id, revision=analysis.revision,
                            original_result_hash=analysis.result_hash, source_sha256=analysis.sha256)


def test_native_anchors_survive_eviction_and_detect_redecode_change(vfr_source):
    video, table = vfr_source
    store = FrameStore(video, table, max_bytes=1)
    first = store.get(0)
    assert store.bytes == 0
    assert store.native_anchors()[0]["decoded_rgb_sha256"] == hashlib.sha256(first.tobytes()).hexdigest()
    changed = first.copy()
    changed[0, 0, 0] ^= 1
    store._decoder = lambda *a, **k: [DecodedFrame(0, table.pts[0], changed)]
    with pytest.raises(ValueError, match="repeated frame changed"):
        store.get(0)


def _add_window(result, table):
    result["signals"][0]["details"] = {"video_window_binding_version": "source-frame-window/1", "evidence_windows": [
        {"frame_indices": [0, 1, 2, 3, 4], "start_pts": table.pts[0], "end_pts": table.pts[4], "time_base": table.time_base,
         "measure": "temporal_feature_variation", "value": .0123,
         "curve": [{"frame_index": i + 1, "support_frame_indices": [i, i+1, i+2], "value": i / 10} for i in range(3)]}]}


def test_window_observations_bind_all_inputs_without_claiming_frame_scores(video_fixture):
    db, store, analysis, result, table, _ = video_fixture
    _add_window(result, table)
    analysis.result_hash = canonical_hash(result)
    document = build_packet_set(db, analysis, store, result)
    packet = document["packets"][0]
    assert packet["claim_scope"] == "sampled_window" and packet["evidence_type"] == "anomaly"
    assert packet["window"]["frame_indices"] == [0, 1, 2, 3, 4]
    assert [f["anchor"]["frame_index"] for f in packet["frames"]] == [0, 2, 4]
    assert all(f["score"] is None and not f["is_peak"] for f in packet["frames"])
    assert packet["window"]["start_playback_s"] == 0
    assert packet["observed_curve"][0]["support_frame_indices"] == [0, 1, 2]
    assert "No individual-window decision threshold" in " ".join(packet["limitations"])


def test_window_with_uncommitted_or_wrong_frame_support_is_rejected(video_fixture):
    db, store, analysis, result, table, _ = video_fixture
    _add_window(result, table)
    result["signals"][0]["details"]["evidence_windows"][0]["start_pts"] += 1
    analysis.result_hash = canonical_hash(result)
    with pytest.raises(PacketIntegrityError, match="committed input frames"):
        build_packet_set(db, analysis, store, result)


def test_track_attribution_uses_legacy_ms_not_stream_ticks(monkeypatch):
    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    captured = []
    window = Window("track:0", 0, 153600, 5.12, ((10, 10),)*5, 32, False)
    monkeypatch.setattr(swap_adapter.gend, "align_transform", lambda points: (np.array([[1, 0, 0], [0, 1, 0]]), 32))
    monkeypatch.setattr(swap_adapter.gend, "warp", lambda *args: rgb[:32, :32])
    monkeypatch.setattr(swap_adapter.gend, "to_batch", lambda x: x)
    monkeypatch.setattr(swap_adapter.gend, "gradcam", lambda *args: np.ones((4, 4), dtype=np.float32))
    ctx = SimpleNamespace(device="cpu", deadline=None,
                          exhibits=SimpleNamespace(heatmap=lambda values, **kw: captured.append(kw) or "map"))
    inputs = SimpleNamespace(store=SimpleNamespace(get_many=lambda indices: {0: rgb}))
    scored = [({"track_id": "track:0"}, [window], [1.0], False, False, SimpleNamespace(top_mean=1.0))]
    assert swap_adapter.GenDTrackDetector()._gradcam(ctx, None, inputs, scored) == ["map"]
    assert captured[0]["frame_pts"] == 5120 and captured[0]["semantics"] == "attribution"
