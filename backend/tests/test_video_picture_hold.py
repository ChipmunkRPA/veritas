from __future__ import annotations

import copy
import hashlib
import io
import subprocess
from fractions import Fraction
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
from app.detectors.video.decode import DecodedFrame, FrameTable, decode_frames, probe_frames
from app.evidence import picture_hold_binding as binding
from app.evidence import picture_sequence, service, video_packets
from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base
from app.storage.objects import ObjectStore
from scripts.video_lossy_sequence_transfer import deliver, read_source
from scripts.video_motion_sequence_360p_transfer import authored


def _source(path, *, held=False, late=False, repeated=False):
    y, x = np.mgrid[:48, :80]
    frames = [np.stack(((x + 13*i) % 256, (y * 5 + i) % 256, (x + y + i) % 256), axis=-1).astype(np.uint8)
              for i in range(190 if late else 40)]
    if held:
        start = 145 if late else 10
        frames[start:start + 16] = [frames[start]] * 16
    if repeated:
        frames[145:165] = frames[24:44]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "80x48",
                    "-framerate", "20", "-i", "pipe:0", "-c:v", "libx264rgb", "-crf", "0", "-output_ts_offset", "5", str(path)],
                   input=b"".join(frame.tobytes() for frame in frames), check=True, timeout=30)
    return frames


@pytest.fixture(scope="module")
def motion_raster_360p():
    x, y = np.meshgrid(np.arange(40), np.arange(30))
    chroma = bytes([128]) * (640 * 360 // 2)
    frames = []
    for i in range(225):
        grid = ((7 * x + 3 * y + 9 * i) % 205 + 20).astype("u1")
        frames.append(grid.repeat(12, axis=0).repeat(16, axis=1).tobytes() + chroma)
    return b"".join(frames)


@pytest.mark.parametrize("case", ["opening_hold", "later_hold", "moving_sequence", "lossy_sequence", "lossy_sequence_360p"])
def test_encoded_picture_repeat_is_reported_exported_and_verified(tmp_path, monkeypatch, case, request):
    high_res = case == "lossy_sequence_360p"
    lossy = case.startswith("lossy_sequence")
    repeated, late = lossy or case == "moving_sequence", case != "opening_hold"
    path = tmp_path / "exact-hold.mkv"
    if lossy:
        indices = [21, 43, 147, 169] if high_res else [20, 41, 130, 151]
        if high_res:
            authored(request.getfixturevalue("motion_raster_360p"), path, 21, 147, 23, "copy", "h264")
        else:
            deliver(read_source(None), path, 20, 130, 22, "copy", "h264")
    else:
        frames = _source(path, held=not repeated, late=late, repeated=repeated)
    table = probe_frames(path)
    if lossy:
        frames = {frame.index: frame.rgb for frame in decode_frames(path, table, indices)}
    store = ObjectStore(tmp_path / "objects")
    original = store.put_file(path)
    origin = 0. if lossy else 5.
    encoding = ("h264", "yuv420p") if lossy else None
    scan = picture_hold.inspect(store.path_for(original.sha256), table, origin=origin, sequence_review=repeated,
                                motion_encoding=encoding)
    assert scan["coverage"]["frames_scanned"] == len(table)
    if lossy:
        coverage = scan["coverage"]
        assert coverage["sequence_review"]["status"] == "no_match"
        motion = coverage["motion_sequence_review"]
        assert motion["status"] == "matched" and motion["frames_scanned"] == (225 if high_res else 180)
        assert motion["parameters"]["frame_rate"] == (25 if high_res else 20)
        match = motion["pair"]
        assert [row[2] for row in picture_sequence.endpoints(match)] == indices
        assert not np.array_equal(frames[indices[0]], frames[indices[2]])
    elif repeated:
        assert scan["match"] is None and scan["coverage"]["scan_scope"] == picture_sequence.SCAN_SCOPE
        match = scan["coverage"]["sequence_review"]["pair"]
        indices = [24, 43, 145, 164]
        assert match["frame_count"] == match["unique_pictures"] == 20
        assert match["span_s"] == .95 and [r[2] for r in picture_sequence.endpoints(match)] == indices
    else:
        match = scan["match"]
        start, end = (145, 160) if late else (10, 25)
        indices = [start, end]
        assert scan["coverage"].get("scan_scope") == (binding.EXTENDED_SCOPE if late else None)
        assert (match["start_frame_index"], match["end_frame_index"], match["frame_count"]) == (start, end, 16)
        assert (match["start_playback_s"], match["end_playback_s"], match["span_s"]) == (start / 20, end / 20, .75)
        assert match["decoded_rgb_sha256"] == hashlib.sha256(frames[start]).hexdigest()

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
            "frame_table_sha256": canonical_hash(table.to_json()), "timeline_origin_s": origin, "picture_hold_sha256": canonical_hash(scan),
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
        expected_packet = picture_sequence.MOTION_PACKET_ID if lossy else picture_sequence.PACKET_ID if repeated else binding.PACKET_ID
        assert packet["id"] == expected_packet
        assert packet["picture_sequence" if repeated else "picture_hold"] == match
        assert [f["anchor"]["frame_index"] for f in packet["frames"]] == indices
        if lossy:
            assert packet["affects_overall_confidence"] is False
            assert len({f["artifact_refs"]["original"] for f in packet["frames"]}) == 4
        for frame in packet["frames"]:
            ref = packet["artifacts"][frame["artifact_refs"]["original"]]
            asset = service.asset_refs(document)[ref["sha256"]]
            with Image.open(io.BytesIO(store.read_bytes(asset["sha256"]))) as saved:
                np.testing.assert_array_equal(np.asarray(saved), frames[frame["anchor"]["frame_index"]])
        changed = copy.deepcopy(document)
        changed["picture_hold"]["last_pts"] += 1
        changed["picture_hold"]["last_stream_time_s"] += .001
        with pytest.raises(ValueError, match="recorded native inspection"):
            binding.validate(changed)
        changed = copy.deepcopy(document)
        if repeated:
            changed["packets"][0]["frames"][2]["anchor"]["frame_index"] += 1
            with pytest.raises(ValueError, match="endpoint differs"):
                binding.validate(changed)
        else:
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
            if lossy:
                assert "Similar repeated moving sequences" in text and "VPR02" in text
                if high_res:
                    assert "first 225 native 640 × 360 frames" in text
                    assert "Repeated ranges 0.840-1.720 s and 5.880-6.760 s" in text
                else:
                    assert "Repeated ranges 1.000-2.050 s and 6.500-7.550 s" in text
            elif repeated:
                assert "Repeated moving sequences" in text and "VPR01" in text
                assert "Repeated ranges 6.200-7.150 s and 12.250-13.200 s" in text
            else:
                assert "Repeated picture across consecutive frames" in text and "VPH01" in text
                assert "16 consecutive decoded frames" in text and "0.750 seconds apart" in text
        _check_public_zip(SimpleNamespace(db=db, store=store, analysis=analysis, frozen=frozen, document=document),
                          tmp_path, monkeypatch, pdf.pdf, source_video=path if late else None)


@pytest.mark.parametrize(("kind", "status", "reason"), [
    ("copy", "matched", None),
    ("hold", "no_match", None),
    ("spacing", "unavailable", "source_clock"),
])
def test_motion_sequence_checks_verified_spacing_and_requires_movement(tmp_path, kind, status, reason):
    path = tmp_path / "vp9.mkv"
    deliver(read_source(None), path, 20, 130, 22, kind, "vp9")
    table = probe_frames(path)
    output = picture_hold.inspect(path, table, origin=0., sequence_review=True, motion_encoding=("vp9", "yuv420p"))
    review = output["coverage"]["motion_sequence_review"]
    assert (review["status"], review["reason"]) == (status, reason)
    assert (review["pair"] is not None) == (status == "matched")


def test_motion_sequence_360p_abstains_when_frame_spacing_varies(tmp_path, motion_raster_360p):
    path = tmp_path / "variable-clock.webm"
    authored(motion_raster_360p, path, 21, 147, 23, "spacing", "vp9")
    table = probe_frames(path)
    output = picture_hold.inspect(path, table, origin=0., sequence_review=True, motion_encoding=("vp9", "yuv420p"))
    review = output["coverage"]["motion_sequence_review"]
    assert (review["status"], review["reason"], review["frames_scanned"], review["pair"]) == ("unavailable", "source_clock", 0, None)
    assert review["parameters"]["frame_rate"] == 25


def test_repeated_sequence_requires_distinct_pixels_and_equal_verified_spacing():
    pts = list(range(0, 2000, 50))
    moving = [str(i) for i in range(20)] * 2
    measured = picture_sequence.scan(list(zip(pts, moving, strict=True)), Fraction(1, 1000))
    assert measured["pair"]["frame_count"] == 20
    assert measured["pair"]["first"]["end_frame_index"] == 19
    assert measured["pair"]["second"]["start_frame_index"] == 20
    pts[30] += 10
    result = picture_sequence.scan(list(zip(pts, moving, strict=True)), Fraction(1, 1000))
    assert result["status"] == "no_match"
    pts[30] = None
    assert picture_sequence.scan(list(zip(pts, moving, strict=True)), Fraction(1, 1000))["status"] == "no_match"
    assert picture_sequence.scan([(i * 50, "static") for i in range(40)], Fraction(1, 1000))["status"] == "no_match"


def test_completed_opening_is_preserved_when_extended_decoder_stops(monkeypatch, tmp_path):
    pts = tuple(range(0, 15000, 50))
    table = FrameTable("1/1000", pts, (0,), "I" * len(pts), 8, 8, 8, 8)
    arrays = [np.full((8, 8, 3), i % 256, dtype=np.uint8) for i in range(len(pts))]
    arrays[20:40] = [arrays[20]] * 20
    arrays[150:179] = [arrays[150]] * 29

    def decode(path, table, wanted, **kwargs):
        for i in wanted:
            if i == 190:
                return
            yield DecodedFrame(i, pts[i], arrays[i])

    monkeypatch.setattr(picture_hold, "decode_frames", decode)
    record = picture_hold.inspect(tmp_path / "unused.mkv", table, origin=None)
    assert record["coverage"] == binding.coverage("matched", binding.MATCH, table.time_base, limit=120, scanned=120,
                                                   last_pts=pts[119])
    assert record["match"]["start_frame_index"] == 20
    assert record["match"]["end_frame_index"] == 39


def test_extended_budget_is_bounded_by_duration_geometry_and_frame_count():
    pts = tuple(range(5000, 65000, 40))
    table = FrameTable("1/1000", pts, (0,), "I" * len(pts), 640, 360, 640, 360)
    assert binding.extended_frame_limit(pts, table.tb, 640, 360) == 751
    assert binding.extended_frame_limit(pts, table.tb, 1920, 1080) == binding.EXTENDED_MAX_RGB_BYTES // (1920 * 1080 * 3)
    assert binding.extended_frame_limit(pts, table.tb, 7680, 4320) == 0
    assert binding.extended_frame_limit(tuple(range(0, 2000)), table.tb, 640, 360) == binding.EXTENDED_MAX_FRAMES


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
