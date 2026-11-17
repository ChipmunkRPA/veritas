from __future__ import annotations

import copy
import hashlib
import io
import itertools
import json
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
from app.detectors.video import mouth_local_hold, picture_hold
from app.detectors.video.decode import DecodedFrame, FrameTable, decode_frames, probe_frames
from app.evidence import mouth_local_hold_binding, picture_sequence, service, video_packets
from app.evidence import picture_hold_binding as binding
from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base
from app.storage.objects import ObjectStore
from scripts.video_face_720p30_transfer import authored as encode_hd_hold
from scripts.video_fractional_retiming_panel import altered as fractional_delivery
from scripts.video_lossy_sequence_transfer import deliver, read_source
from scripts.video_motion_late_transfer import encode as late_authored
from scripts.video_motion_sequence_360p_transfer import authored
from scripts.video_mouth_local_hold_panel import encode as encode_mouth_hold
from scripts.video_picture_speed_panel import _delivery as speed_delivery
from scripts.video_picture_speed_panel import _encode as encode_speed
from scripts.video_reflected_sequence_panel import delivery as reflected_delivery


def _source(path, *, held=False, late=False, repeated=False, reverse=False):
    y, x = np.mgrid[:48, :80]
    frames = [np.stack(((x + 13*i) % 256, (y * 5 + i) % 256, (x + y + i) % 256), axis=-1).astype(np.uint8)
              for i in range(190 if late else 40)]
    if held:
        start = 145 if late else 10
        frames[start:start + 16] = [frames[start]] * 16
    if repeated:
        donor = frames[24:44]
        frames[145:165] = donor[::-1] if reverse else donor
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "80x48",
                    "-framerate", "20", "-i", "pipe:0", "-c:v", "libx264rgb", "-crf", "0", "-output_ts_offset", "5", str(path)],
                   input=b"".join(frame.tobytes() for frame in frames), check=True, timeout=30)
    return frames


@pytest.fixture(scope="module")
def motion_raster_360p():
    return _motion_raster(225)


def _motion_raster(count):
    x, y = np.meshgrid(np.arange(40), np.arange(30))
    chroma = bytes([128]) * (640 * 360 // 2)
    frames = []
    for i in range(count):
        drift = i // 11 if count > 225 else 0
        grid = (((7 + drift) * x + 3 * y + 9 * i) % 205 + 20).astype("u1")
        frames.append(grid.repeat(12, axis=0).repeat(16, axis=1).tobytes() + chroma)
    return b"".join(frames)


def _mouth_source(path, monkeypatch, *, hd):
    width, height, count = (1280, 720, 240) if hd else (360, 288, 50)
    start, end = (99, 111) if hd else (16, 26)
    cx, cy, span = (640, 400, 64) if hd else (180, 159, 44)
    rx, ry = (58, 42) if hd else (40, 34)
    texture = np.indices((ry * 2, rx * 2)).sum(axis=0) % 5 * 28 + 26

    def pictures():
        for i in range(count):
            rgb = np.full((height, width, 3), 72 + (i % 4) * 4, dtype=np.uint8)
            core = texture + (70 if start <= i < end else (i % 6) * 7)
            rgb[cy-ry:cy+ry, cx-rx:cx+rx] = core[..., None]
            yield rgb

    if hd:
        encode_hd_hold(path, pictures=pictures(), crf=0)
    else:
        encode_mouth_hold(path, list(pictures()), 0)
    monkeypatch.setattr(mouth_local_hold, "local_detector_path", lambda: path)
    monkeypatch.setattr(mouth_local_hold, "load_yunet", lambda _: None)
    monkeypatch.setattr(mouth_local_hold, "source_geometry", lambda *_: (mouth_local_hold.Geometry(cx, cy, span), 1))


@pytest.mark.parametrize("case", ["opening_hold", "later_hold", "moving_sequence", "reverse_sequence", "lossy_reverse", "lossy_sequence", "lossy_sequence_360p",
                                   "lossy_sequence_360p_late", "retimed_sequence", "fractional_retimed_sequence", "reflected_sequence", "reflected_hd",
                                   "mouth_hold", "mouth_hold_hd"])
def test_encoded_picture_repeat_is_reported_exported_and_verified(tmp_path, monkeypatch, case, request):
    mouth = case in {"mouth_hold", "mouth_hold_hd"}
    hd = case == "mouth_hold_hd"
    high_res = case.startswith("lossy_sequence_360p")
    extension = case == "lossy_sequence_360p_late"
    lossy_reverse = case == "lossy_reverse"
    fractional = case == "fractional_retimed_sequence"
    retimed = case == "retimed_sequence" or fractional
    reflected_hd = case == "reflected_hd"
    reflected = case == "reflected_sequence" or reflected_hd
    lossy = case.startswith("lossy_sequence") or lossy_reverse or retimed or reflected
    reverse = case == "reverse_sequence"
    repeated, late = mouth or lossy or reverse or case == "moving_sequence", case != "opening_hold"
    path = tmp_path / ("hd-picture.webm" if reflected_hd else "native-picture.mp4" if mouth or retimed or reflected else "exact-hold.mkv")
    if mouth:
        _mouth_source(path, monkeypatch, hd=hd)
    elif lossy:
        if retimed:
            source = np.frombuffer(_motion_raster(226), dtype=np.uint8).reshape(226, -1)
            if fractional:
                frames, expected_pair = fractional_delivery(source, 26, 144, "later_fast_4_3")
                indices = [index for a, b, _ in expected_pair for index in (a, b)]
            else:
                frames, _ = speed_delivery(source, 26, 144, "later_fast")
                indices = [26, 66, 144, 164]
            encode_speed(path, frames)
        elif reflected:
            source = np.frombuffer(_motion_raster(226), dtype=np.uint8).reshape(226, -1)
            if reflected_hd:
                frames, _ = reflected_delivery(source, 26, 143, 22, "later_mirrored")
                master = tmp_path / "source.mp4"
                encode_speed(master, frames)
                subprocess.run([
                    "ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(master), "-an",
                    "-vf", "scale=1280:720,settb=1/120000,setpts=N*4004", "-r", "30000/1001", "-fps_mode", "cfr",
                    "-c:v", "libvpx-vp9", "-crf", "18", "-b:v", "0", "-deadline", "good", "-cpu-used", "6",
                    "-threads", "2", str(path),
                ], check=True, timeout=60)
                indices = [26, 47, 143, 164]
            else:
                frames, _ = reflected_delivery(source, 26, 144, 23, "later_mirrored")
                encode_speed(path, frames)
                indices = [26, 48, 144, 166]
        elif extension:
            indices = late_authored(_motion_raster(450), path, 0, "late_cross", "h264")
        elif high_res:
            indices = [21, 43, 147, 169]
            authored(request.getfixturevalue("motion_raster_360p"), path, 21, 147, 23, "copy", "h264")
        else:
            indices = [20, 41, 130, 151]
            deliver(read_source(None), path, 20, 130, 22, "reverse" if lossy_reverse else "copy", "h264")
    else:
        frames = _source(path, held=not repeated, late=late, repeated=repeated, reverse=reverse)
    table = probe_frames(path)
    if lossy:
        frames = {frame.index: frame.rgb for frame in decode_frames(path, table, indices)}
    store = ObjectStore(tmp_path / "objects")
    original = store.put_file(path)
    origin = 0. if lossy or mouth else 5.
    encoding = (("vp9" if reflected_hd else "h264"), "yuv420p") if lossy or mouth else None
    scan = picture_hold.inspect(store.path_for(original.sha256), table, origin=origin, sequence_review=repeated,
                                motion_encoding=encoding, mouth_review=mouth, retimed_review=retimed,
                                fractional_retimed_review=fractional,
                                reflected_review=reflected)
    assert scan["coverage"]["frames_scanned"] == len(table)
    if mouth:
        review = scan["coverage"]["mouth_local_hold"]
        assert review["status"] == "matched" and review["affects_overall_confidence"] is False
        assert review["frames_scanned"] == (120 if hd else 50)
        if hd:
            assert review["schema"] == mouth_local_hold_binding.HD_SCHEMA and review["assessed_window_starts"] == [0, 30, 60]
        match = review["match"]
        assert match["frames"] == ([99, 111] if hd else [16, 26])
        indices = [98, 105, 111] if hd else [15, 21, 26]
        assert match["anchors"] == indices
        prefix = [frame.rgb for frame in decode_frames(path, table, range(review["frames_scanned"]))]
        mouth_local_hold.verify_measurements(prefix, json.loads(canonical_json(match)), hd=hd)
        altered_match = {**match, "minimum_flank_peak": match["minimum_flank_peak"] + .1}
        with pytest.raises(ValueError, match="do not reproduce the held mouth-region measurements"):
            mouth_local_hold.verify_measurements(prefix, altered_match, hd=hd)
        frames = {i: prefix[i] for i in indices}
        del prefix
    elif lossy:
        coverage = scan["coverage"]
        assert coverage["sequence_review"]["status"] == "no_match"
        motion_key = (picture_sequence.MOTION_REFLECTED_KEY if reflected else picture_sequence.MOTION_RETIMED_KEY if retimed else picture_sequence.MOTION_REVERSE_KEY
                      if lossy_reverse else "motion_sequence_review")
        motion = coverage[motion_key]
        assert motion["status"] == "matched" and motion["frames_scanned"] == (450 if extension else 225 if high_res or retimed or reflected else 180)
        assert motion["parameters"]["frame_rate"] == ("30000/1001" if reflected_hd else 25 if high_res or retimed or reflected else 20)
        expected_schema = (picture_sequence.MOTION_REFLECTED_HD_SCHEMA if reflected_hd
                           else picture_sequence.MOTION_REFLECTED_SCHEMA if reflected
                           else picture_sequence.MOTION_RETIMED_FRACTIONAL_SCHEMA if fractional
                           else picture_sequence.MOTION_RETIMED_SCHEMA if retimed else picture_sequence.MOTION_REVERSE_SCHEMA if lossy_reverse else
                           picture_sequence.MOTION_EXTENDED_SCHEMA if extension else picture_sequence.MOTION_SCHEMA)
        assert motion["schema"] == expected_schema
        match = motion["pair"]
        assert [row[2] for row in picture_sequence.endpoints(match)] == indices
        assert not np.array_equal(frames[indices[0]], frames[indices[-1] if lossy_reverse else indices[2]])
    elif repeated:
        assert scan["match"] is None and scan["coverage"]["scan_scope"] == picture_sequence.SCAN_SCOPE
        key = picture_sequence.REVERSE_KEY if reverse else "sequence_review"
        other_key = "sequence_review" if reverse else picture_sequence.REVERSE_KEY
        assert scan["coverage"][other_key]["status"] == "no_match"
        match = scan["coverage"][key]["pair"]
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
        content_type = "video/webm" if reflected_hd else "video/mp4" if mouth else "video/x-matroska"
        analysis = Analysis(id="video-picture-hold", kind="video", filename=path.name, content_type=content_type,
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
        expected_packet = (mouth_local_hold_binding.PACKET_ID if mouth else picture_sequence.MOTION_REVERSE_PACKET_ID if lossy_reverse
                           else picture_sequence.MOTION_REFLECTED_PACKET_ID if reflected
                           else picture_sequence.MOTION_RETIMED_PACKET_ID if retimed
                           else picture_sequence.MOTION_PACKET_ID if lossy
                           else picture_sequence.REVERSE_PACKET_ID if reverse
                           else picture_sequence.PACKET_ID if repeated else binding.PACKET_ID)
        [packet] = [candidate for candidate in document["packets"] if candidate["id"] == expected_packet] if mouth else document["packets"]
        assert packet["id"] == expected_packet
        assert canonical_hash(packet["mouth_local_hold" if mouth else "picture_sequence" if repeated else "picture_hold"]) == canonical_hash(match)
        assert [f["anchor"]["frame_index"] for f in packet["frames"]] == indices
        if reverse:
            assert packet["affects_overall_confidence"] is False and match["order"] == "reverse"
            assert [f["artifact_refs"]["original"] for f in packet["frames"]] == [
                "start_picture", "end_picture", "end_picture", "start_picture"]
            altered = copy.deepcopy(document)
            del altered["packets"][0]["picture_sequence"]["order"]
            del altered["picture_hold"][picture_sequence.REVERSE_KEY]["pair"]["order"]
            with pytest.raises(ValueError, match="wrong matching method"):
                binding.validate(altered)
        if mouth:
            assert not packet["affects_overall_confidence"]
            assert [f["interval_role"] for f in packet["frames"]] == ["before", "during", "after"]
            modified = copy.deepcopy(document)
            next(p for p in modified["packets"] if p["id"] == expected_packet)["window"]["end_playback_s"] += .01
            with pytest.raises(ValueError, match="differs from the committed clock"):
                binding.validate(modified)
            x, y, w, h = match["box"]
            for frame in packet["frames"]:
                crop = packet["artifacts"][frame["artifact_refs"]["crop"]]
                with Image.open(io.BytesIO(store.read_bytes(crop["sha256"]))) as saved:
                    np.testing.assert_array_equal(np.asarray(saved), frames[frame["anchor"]["frame_index"]][y:y+h, x:x+w])
        if lossy:
            assert packet["affects_overall_confidence"] is False
            assert len({f["artifact_refs"]["original"] for f in packet["frames"]}) == 4
            if lossy_reverse:
                assert match["order"] == "reverse"
                assert [f["artifact_refs"]["original"] for f in packet["frames"]] == [
                    "first_start_picture", "first_end_picture", "second_start_picture", "second_end_picture"]
            if retimed:
                assert match["frame_steps"] == ({"first": 4, "second": 3} if fractional else {"first": 2, "second": 1})
                altered = copy.deepcopy(document)
                altered["picture_hold"][picture_sequence.MOTION_RETIMED_KEY]["pair"]["frame_steps"]["first"] = 1
                altered["packets"][0]["picture_sequence"]["frame_steps"]["first"] = 1
                with pytest.raises(ValueError, match="unsupported playback ratio" if fractional else "two-to-one frame stride"):
                    binding.validate(altered)
            if reflected:
                assert match["spatial_transform"] == picture_sequence.HORIZONTAL_REFLECTION
                altered = copy.deepcopy(document)
                del altered["picture_hold"][picture_sequence.MOTION_REFLECTED_KEY]["pair"]["spatial_transform"]
                del altered["packets"][0]["picture_sequence"]["spatial_transform"]
                with pytest.raises(ValueError, match="wrong spatial transform"):
                    binding.validate(altered)
            if extension or retimed or reflected:
                interrupted = list(table.pts)
                interrupted[400 if extension else 27] += 12 if reflected_hd else 1
                with pytest.raises(ValueError, match="complete verified native clock"):
                    picture_sequence.validate_clock(motion, interrupted, (), tb=table.tb)
            if reflected_hd:
                assert [m["value"] for m in packet["measurements"][1:]] == [.7, .701]
        for frame in packet["frames"]:
            ref = packet["artifacts"][frame["artifact_refs"]["original"]]
            asset = service.asset_refs(document)[ref["sha256"]]
            with Image.open(io.BytesIO(store.read_bytes(asset["sha256"]))) as saved:
                np.testing.assert_array_equal(np.asarray(saved), frames[frame["anchor"]["frame_index"]])
        changed = copy.deepcopy(document)
        changed["picture_hold"]["last_pts"] += 1
        changed["picture_hold"]["last_stream_time_s"] += float(table.tb)
        with pytest.raises(ValueError, match="recorded native inspection"):
            binding.validate(changed)
        changed = copy.deepcopy(document)
        if mouth or repeated:
            next(p for p in changed["packets"] if p["id"] == expected_packet)["frames"][2]["anchor"]["frame_index"] += 1
            with pytest.raises(ValueError, match="Mouth hold image differs" if mouth else "endpoint differs"):
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
            if mouth:
                assert "Mouth-region picture hold" in text and "VPH02" in text
                assert ("3.300 to 3.700 s" if hd else "0.640 to 1.040 s") in text and "Exact measured mouth-region crop" in text
                if hd:
                    assert "Mouth geometry was measured in 3 of 3 overlapping windows" in text
            elif lossy_reverse:
                assert "Similar moving pictures in reverse order" in text and "VPR04" in text
                assert "Reversed ranges 1.000-2.050 s and 6.500-7.550 s" in text
                assert "transitions support the reversed motion" in text
            elif retimed:
                assert "Footage repeated at two playback speeds" in text and "VPR05" in text
                ranges = "1.040-3.920 s and 5.760-7.920 s" if fractional else "1.040-2.640 s and 5.760-6.560 s"
                assert f"Ranges at two playback speeds {ranges}" in text
                assert ("4:3 speed ratio" if fractional else "twice the speed") in text
            elif reflected:
                assert "Similar footage flipped left to right" in text and "VPR06" in text
                location = "0.868-1.568 s and 4.771-5.472 s" if reflected_hd else "1.040-1.920 s and 5.760-6.640 s"
                assert f"Mirrored ranges {location}" in text
                assert "matching picture changes support the comparison" in text
            elif lossy:
                assert "Similar repeated moving sequences" in text and "VPR02" in text
                if extension:
                    assert "first 450 native 640 × 360 frames" in text
                    assert "Repeated ranges 1.680-2.560 s and 11.520-12.400 s" in text
                elif high_res:
                    assert "first 225 native 640 × 360 frames" in text
                    assert "Repeated ranges 0.840-1.720 s and 5.880-6.760 s" in text
                else:
                    assert "Repeated ranges 1.000-2.050 s and 6.500-7.550 s" in text
            elif reverse:
                assert "Reversed moving sequences" in text and "VPR03" in text
                assert "Reversed ranges 6.200-7.150 s and 12.250-13.200 s" in text
                assert "plays the same 20 decoded pictures backward" in text
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


def test_larger_motion_matrix_keeps_opening_record_if_pair_ends_inside_it(monkeypatch, tmp_path):
    ticks = tuple(i * 40 for i in range(450))
    table = FrameTable("1/1000", ticks, (0,), "I" * len(ticks), 640, 360, 640, 360)
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)

    def decode(path, table, indices, **kwargs):
        for i in indices:
            yield DecodedFrame(i, table.pts[i], rgb)

    opening = {"schema": picture_sequence.MOTION_SCHEMA, "status": "no_match", "pair": None}
    longer = {"schema": picture_sequence.MOTION_EXTENDED_SCHEMA, "status": "matched",
              "pair": {"second": {"end_frame_index": 200}}}

    def scan(*args, source_profile, **kwargs):
        return longer if source_profile[1] == 450 else opening

    monkeypatch.setattr(picture_hold, "decode_frames", decode)
    monkeypatch.setattr(picture_hold.picture_motion, "descriptor", lambda _: None)
    monkeypatch.setattr(picture_hold.picture_motion, "scan", scan)
    result = picture_hold.inspect(tmp_path / "source.mp4", table, origin=0., sequence_review=True,
                                  motion_encoding=("h264", "yuv420p"))
    assert result["coverage"]["motion_sequence_review"] == opening


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


def test_reverse_sequence_preserves_the_spacing_between_corresponding_frames():
    tb = Fraction(1, 1000)
    first_steps = [40, 60] * 6
    first_pts = list(itertools.accumulate(first_steps, initial=0))
    second_pts = list(itertools.accumulate(reversed(first_steps), initial=900))
    digests = [str(i) for i in range(len(first_pts))]
    pictures = digests + digests[::-1]

    matched = picture_sequence.scan(list(zip(first_pts + second_pts, pictures, strict=True)), tb, reverse=True)
    assert matched["pair"]["frame_count"] == 13
    assert matched["pair"]["span_s"] == .6
    same_order_pts = list(itertools.accumulate(first_steps, initial=900))
    failed = picture_sequence.scan(list(zip(first_pts + same_order_pts, pictures, strict=True)), tb, reverse=True)
    assert failed["status"] == "no_match"
    second_pts[6] = None
    unverified = picture_sequence.scan(list(zip(first_pts + second_pts, pictures, strict=True)), tb, reverse=True)
    assert unverified["status"] != "matched"


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
