from __future__ import annotations

import copy
import hashlib
import io
import subprocess
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_video_nested_evidence import _check_public_zip

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.video import picture_hold
from app.detectors.video.decode import DecodedFrame, FrameTable, probe_frames
from app.evidence import picture_hold_binding as binding
from app.evidence import service, video_packets
from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base
from app.storage.objects import ObjectStore


def _source(path, *, held=False):
    y, x = np.mgrid[:48, :80]
    frames = [np.stack(((x + 13*i) % 256, (y * 5 + i) % 256, (x + y + i) % 256), axis=-1).astype(np.uint8)
              for i in range(40)]
    if held:
        frames[10:26] = [frames[10]] * 16
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "80x48",
                    "-framerate", "20", "-i", "pipe:0", "-c:v", "libx264rgb", "-crf", "0", "-output_ts_offset", "5", str(path)],
                   input=b"".join(frame.tobytes() for frame in frames), check=True, timeout=30)
    return frames


def test_encoded_picture_hold_is_reported_exported_and_verified(tmp_path, monkeypatch):
    path = tmp_path / "exact-hold.mkv"
    frames = _source(path, held=True)
    table = probe_frames(path)
    store = ObjectStore(tmp_path / "objects")
    original = store.put_file(path)
    scan = picture_hold.inspect(store.path_for(original.sha256), table, origin=5.)
    match = scan["match"]
    assert scan["coverage"]["frames_scanned"] == 40
    assert (match["start_frame_index"], match["end_frame_index"], match["frame_count"]) == (10, 25, 16)
    assert (match["start_playback_s"], match["end_playback_s"], match["span_s"]) == (.5, 1.25, .75)
    assert match["decoded_rgb_sha256"] == hashlib.sha256(frames[10]).hexdigest()

    database = create_engine("sqlite://")
    Base.metadata.create_all(database)
    with Session(database) as db:
        analysis = Analysis(id="video-picture-hold", kind="video", filename=path.name, content_type="video/x-matroska",
                            size_bytes=original.size_bytes, sha256=original.sha256, status="done", revision=1,
                            result_schema_version=2)
        db.add(analysis)
        db.flush()
        rendered = {"frame_anchor_version": video_packets.ANCHOR_VERSION, "frame_anchors": [], "picture_hold": scan}
        for name, data in [("probe", {"frame_table": table.to_json(), "info": {"has_audio": False}}), ("render", rendered)]:
            art = register_artefact(db, store, canonical_json(data), media_type="application/json", role="stage_output",
                                    created_by=name, analysis_id=analysis.id)
            db.add(AnalysisStage(analysis_id=analysis.id, stage=name, status="done", output_artefact_id=art.id))
        frozen = specimen("video", "standard")
        result = copy.deepcopy(frozen.result)
        result["input"]["sha256"] = original.sha256
        result["signals"] = result["exhibits"] = result["findings"] = []
        result["manifest"]["sampling"].update({
            "frame_anchor_version": video_packets.ANCHOR_VERSION, "frame_anchor_sha256": canonical_hash([]),
            "frame_table_sha256": canonical_hash(table.to_json()), "timeline_origin_s": 5., "picture_hold_sha256": canonical_hash(scan),
            "video_audio_evidence": {"version": "video-audio-inputs/1", "decode_status": "absent", "source_audio_present": False,
                                     "input": None, "voice_checks": [], "av_checks": []},
        })
        analysis.result_hash = canonical_hash(result)
        store.put_bytes(canonical_json(result))
        db.flush()
        document = service.persist_packet_set(db, analysis, store, result=result)
        assert service.load_packet_set(db, analysis, store) == document
        assert document["coverage"]["scored_frame_indices"] == []
        [packet] = document["packets"]
        assert packet["id"] == binding.PACKET_ID
        assert packet["picture_hold"] == match
        assert [f["anchor"]["frame_index"] for f in packet["frames"]] == [10, 25]
        asset = service.asset_refs(document)[packet["artifacts"]["held_picture"]["sha256"]]
        with Image.open(io.BytesIO(store.read_bytes(asset["sha256"]))) as saved:
            np.testing.assert_array_equal(np.asarray(saved), frames[10])
        changed = copy.deepcopy(document)
        changed["picture_hold"]["last_pts"] += 1
        changed["picture_hold"]["last_stream_time_s"] += .001
        with pytest.raises(ValueError, match="recorded native inspection"):
            binding.validate(changed)
        changed = copy.deepcopy(document)
        changed["packets"][0]["picture_hold"]["pts"][1] = match["start_pts"]
        with pytest.raises(ValueError, match="discontinuous timestamp"):
            binding.validate(changed)
        frozen = frozen.model_copy(update={"result": result, "evidence": document,
            "analysis": frozen.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
            "media": frozen.media.model_copy(update={"sha256": original.sha256, "size_bytes": original.size_bytes,
                                                     "filename": analysis.filename, "content_type": analysis.content_type})})
        monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
        pdf = engine.render_pdf(frozen)
        with pymupdf.open(stream=pdf.pdf, filetype="pdf") as rendered_pdf:
            text = " ".join(" ".join(page.get_text() for page in rendered_pdf).split())
            assert "Repeated picture across consecutive frames" in text and "VPH01" in text
            assert "16 consecutive decoded frames" in text and "0.750 seconds apart" in text
        _check_public_zip(SimpleNamespace(db=db, store=store, analysis=analysis, frozen=frozen, document=document),
                          tmp_path, monkeypatch, pdf.pdf)


def test_moving_encoded_clip_returns_checked_no_match(tmp_path):
    path = tmp_path / "moving.mkv"
    _source(path)
    record = picture_hold.inspect(path, probe_frames(path), origin=5.)
    assert record["coverage"]["status"] == "no_match"
    assert record["coverage"]["frames_scanned"] == 40
    assert record["match"] is None


@pytest.mark.parametrize("problem", ["unverified", "interpolated", "duplicate", "gap", "partial"])
def test_clock_or_decode_break_prevents_joining_held_frames(monkeypatch, tmp_path, problem):
    pts = list(range(0, 700, 50))
    if problem == "duplicate":
        pts[7] = pts[6]
    if problem == "gap":
        pts[7:] = [p + 500 for p in pts[7:]]
    table = FrameTable("1/1000", tuple(pts), (0,), "I"*14, 8, 8, 8, 8, interpolated=(7,) if problem == "interpolated" else ())
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)

    def decode(path, table, wanted, **kwargs):
        for i in wanted:
            if problem == "partial" and i == 11:
                return
            yield DecodedFrame(i, pts[i], rgb, pts_verified=not ((problem == "unverified" or i in table.interpolated) and i == 7))

    monkeypatch.setattr(picture_hold, "decode_frames", decode)
    record = picture_hold.inspect(tmp_path / "unused.mkv", table, origin=None)
    assert record["coverage"]["status"] == "unavailable" and record["match"] is None
    assert record["coverage"]["reason"] == (binding.INCOMPLETE if problem == "partial" else binding.TOO_SHORT)
