from __future__ import annotations

import hashlib
import io
import subprocess
from copy import deepcopy
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.video import eye_local_hold, picture_hold
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence import eye_local_hold_binding as binding
from app.evidence import service, video_packets
from app.reports import public_verify, verify_video_source
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, AnalysisStage, Base
from app.storage.objects import ObjectStore


def _author_picture_sequence(path):
    rng = np.random.default_rng(46271)
    texture = rng.integers(36, 220, (37, 52), dtype=np.uint8)
    rgb = np.empty((binding.FRAME_COUNT, 360, 480, 3), dtype=np.uint8)
    for index, frame in enumerate(rgb):
        light = 70 + (0, 2, 4, 6, 4, 2)[index % 6]
        gray = np.full((360, 480), light, dtype=np.uint8)
        gray[99:205, 118:245] = rng.integers(28, 220, (106, 127), dtype=np.uint8)
        gray[106:215, 260:356] = rng.integers(28, 220, (109, 96), dtype=np.uint8)
        if 16 <= index < 32:
            gray[132:169, 154:206] = texture
        frame[:] = gray[..., None]
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", "480x360", "-framerate", "30", "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "1",
        "-pix_fmt", "yuv420p", "-video_track_timescale", "30000", "-crf", "0", str(path),
    ], input=rgb.tobytes(), check=True, capture_output=True, timeout=20)


def _package_evidence(db, original, table, store, filename, scan):
    analysis = Analysis(id="eye-hold-source", kind="video", filename=filename, content_type="video/mp4",
                        size_bytes=original.size_bytes, sha256=original.sha256, status="done", revision=1,
                        result_schema_version=2)
    db.add(analysis)
    db.flush()
    rendered = {"frame_anchor_version": video_packets.ANCHOR_VERSION, "frame_anchors": [], "picture_hold": scan}
    for name, data in (("probe", {"frame_table": table.to_json(), "info": {"has_audio": False}}), ("render", rendered)):
        art = register_artefact(db, store, canonical_json(data), media_type="application/json", role="stage_output",
                                created_by=name, analysis_id=analysis.id)
        db.add(AnalysisStage(analysis_id=analysis.id, stage=name, status="done", output_artefact_id=art.id))
    result = deepcopy(specimen("video", "standard").result)
    result["input"]["sha256"] = original.sha256
    result["signals"] = result["exhibits"] = result["findings"] = []
    result["manifest"]["sampling"].update({
        "frame_anchor_version": video_packets.ANCHOR_VERSION, "frame_anchor_sha256": canonical_hash([]),
        "frame_table_sha256": canonical_hash(table.to_json()), "timeline_origin_s": 0.,
        "picture_hold_sha256": canonical_hash(scan),
        "video_audio_evidence": {"version": "video-audio-inputs/1", "decode_status": "absent", "source_audio_present": False,
                                 "input": None, "voice_checks": [], "av_checks": []},
    })
    analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
    db.flush()
    return service.persist_packet_set(db, analysis, store, result=result)


def test_original_eye_evidence_replays_pixels_clock_and_observation(tmp_path, monkeypatch):
    path = tmp_path / "eye-region.mp4"
    _author_picture_sequence(path)
    table = probe_frames(path)
    decoded = list(decode_frames(path, table, None))
    hashes = [hashlib.sha256(frame.rgb).hexdigest() for frame in decoded]
    monkeypatch.setattr(eye_local_hold, "_landmarks", lambda *_: (([(180., 150.), (300., 150.)], 120.), 1))
    status, match = eye_local_hold.inspect([frame.rgb for frame in decoded], None)
    assert status == "matched"

    def unavailable(*_):
        pytest.fail("Original-file replay should recompute eye texture measurements without running landmark inference")

    monkeypatch.setattr(eye_local_hold, "_landmarks", unavailable)
    scan = picture_hold.inspect(path, table, origin=0., sequence_review=False)
    scan["coverage"][binding.KEY] = binding.record(status, pts=table.pts, hashes=hashes, match=match)
    store = ObjectStore(tmp_path / "objects")
    original = store.put_file(path)
    database = create_engine("sqlite://")
    Base.metadata.create_all(database)
    with Session(database) as db:
        doc = _package_evidence(db, original, table, store, path.name, scan)
    bundle = {"schema": public_verify.SCHEMA, "analysis": {"kind": "video"}, "source_commitment": {"sha256": original.sha256}}

    def replay(document, replacements=None):
        with ZipFile(io.BytesIO(), "w") as archive:
            archive.writestr("evidence.json", canonical_json(document))
            for digest, asset in service.asset_refs(document).items():
                if asset["media_type"] == "image/png":
                    contents = replacements[digest] if replacements and digest in replacements else store.read_bytes(digest)
                    archive.writestr(f"assets/{digest}.png", contents)
            return verify_video_source.check(archive, bundle, path)

    def rebind_coverage(document):
        document["source"]["picture_hold_sha256"] = canonical_hash({"coverage": document["picture_hold"], "match": scan["match"]})

    assert "Recomputed the held eye-region pixel measurements." in replay(doc)
    changed = deepcopy(doc)
    recorded = changed["picture_hold"][binding.KEY]
    recorded["frame_hashes"][0] = "f" * 64
    recorded["input_sha256"] = canonical_hash(recorded["frame_hashes"])
    rebind_coverage(changed)
    with pytest.raises(ValueError, match="Eye-region input differs from the decoded source pixels"):
        replay(changed)

    changed = deepcopy(doc)
    packet = next(entry for entry in changed["packets"] if entry["id"] == binding.PACKET_ID)
    recorded = changed["picture_hold"][binding.KEY]
    recorded["match"]["midpoint_texture_change"] += .001
    packet[binding.KEY]["midpoint_texture_change"] = recorded["match"]["midpoint_texture_change"]
    rebind_coverage(changed)
    with pytest.raises(ValueError, match="Eye-region measurements differ from the submitted video"):
        replay(changed)

    crop = packet["artifacts"]["during_crop"]
    with io.BytesIO() as output:
        Image.new("RGB", (crop["width"], crop["height"]), color=(0, 0, 0)).save(output, format="PNG")
        with pytest.raises(ValueError, match=r"VPH03\.during crop differs from pixels"):
            replay(doc, {crop["sha256"]: output.getvalue()})

    measured_no_match = deepcopy(doc)
    measured_no_match["picture_hold"][binding.KEY] = binding.record("no_match", pts=table.pts, hashes=hashes)
    measured_no_match["packets"] = [entry for entry in measured_no_match["packets"] if entry["id"] != binding.PACKET_ID]
    rebind_coverage(measured_no_match)
    assert "Verified the eye-region input clock and pictures." in replay(measured_no_match)
    recorded = measured_no_match["picture_hold"][binding.KEY]
    recorded["pts"] = [point + 1 for point in recorded["pts"]]
    rebind_coverage(measured_no_match)
    with pytest.raises(ValueError, match="Eye-region timestamps differ from the verified source clock"):
        replay(measured_no_match)
