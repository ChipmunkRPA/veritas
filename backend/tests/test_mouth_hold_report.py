from __future__ import annotations

import hashlib
import io
from copy import deepcopy

import pytest
from PIL import Image

from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen


@pytest.mark.parametrize("with_sound", [True, False])
def test_mouth_hold_pdf_names_the_measured_region_and_distinguishes_clocks(monkeypatch, with_sound):
    snapshot = specimen("video", "standard")
    assets = {}

    def png(image, **metadata):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        assets[digest] = data
        return {"sha256": digest, "media_type": "image/png", **metadata}

    packet = {
        "id": "VPH02", "title": "Mouth-region picture hold",
        "observation": "A textured mouth region stayed nearly unchanged from 0.520 to 0.840 s.",
        "interpretation": "This observation carries zero weight in the AI estimate.",
        "limitations": [], "finding_ids": [], "signal_refs": [], "affects_overall_confidence": False,
        "mouth_local_hold": {"frames": [13, 21], "anchors": [12, 17, 21], "box": [176, 192, 36, 24]},
        "window": {"frame_indices": list(range(12, 22)), "start_pts": 6656, "end_pts": 10752,
                   "time_base": "1/12800", "start_stream_s": .52, "end_stream_s": .84},
        "clip": {"status": "unavailable", "reason": "Use the original video to play the measured range."},
        "frames": [], "artifacts": {},
    }
    for role, number in zip(("before", "during", "after"), (12, 17, 21), strict=True):
        original = Image.new("RGB", (360, 288), (number, 45, 122))
        original_ref, crop_ref = f"{role}_original", f"{role}_crop"
        packet["artifacts"][original_ref] = png(original)
        packet["artifacts"][crop_ref] = png(original.crop((176, 192, 212, 216)))
        packet["frames"].append({
            "id": f"VPH02.{role}", "score": None, "region_meaning": "measured_mouth_region",
            "interval_role": role, "artifact_refs": {"original": original_ref, "context": original_ref, "crop": crop_ref},
            "context_region": {"x": 176, "y": 192, "w": 36, "h": 24},
            "anchor": {"frame_index": number, "pts": 512 * number, "time_base": "1/12800",
                       "stream_time_s": number / 25, "playback_time_s": number / 25,
                       "decoded_rgb_sha256": hashlib.sha256(original.tobytes()).hexdigest()},
        })
    if with_sound:
        packet["mouth_hold_sound"] = {"schema": "chipmunk.video-mouth-hold-sound/1", "samples": [4320, 9440],
                                     "source_pts": [8320, 13440], "time_base": "1/16000"}
        packet["artifacts"]["mouth_hold_waveform"] = png(Image.new("RGB", (960, 280)), time_axis="source_pts")
        packet["artifacts"]["mouth_hold_audio"] = {"sha256": "c" * 64, "media_type": "audio/wav"}
    else:
        packet["limitations"].append("The soundtrack was not measured alongside these pictures.")
    document = {
        "schema": "chipmunk.video-evidence/1", "id": "synthetic-mouth-pdf", "version": "video-evidence-2",
        "sha256": "a" * 64, "original_result_hash": "b" * 64, "packets": [packet],
        "source": {"sha256": snapshot.media.sha256, "width": 360, "height": 288, "frame_count": 50},
        "picture_hold": {"status": "no_match", "frames_scanned": 50, "continuity_breaks": 0, "last_stream_time_s": 1.96,
                         "mouth_local_hold": {"status": "matched", "frames_scanned": 50, "packet_id": "VPH02"}},
    }
    frozen = deepcopy(document)
    monkeypatch.setattr(multimedia_evidence, "verified_set", lambda _: document)
    monkeypatch.setattr(multimedia_evidence, "verify_assets", lambda *_args, **_kwargs: assets)

    text = " ".join(engine.pdf_text(engine.render_pdf(snapshot).pdf).split())

    assert document == frozen
    assert "Measured source frames [13, 21) and source PTS [6656, 10752) at time base 1/12800" in text
    for role, number in (("Before", 12), ("During", 17), ("After", 21)):
        assert f"{role} mouth-region hold - source frame {number}" in text
    assert text.count("Exact measured mouth-region crop") == 3
    assert text.count("Measured mouth region: x 176, y 192, width 36, height 24 pixels") == 3
    assert "Detected face context" not in text
    assert "flagged range" not in text
    assert "Context clip unavailable" not in text
    assert "This window used 10 recorded input frames" not in text
    assert ("Soundtrack source PTS [8320, 13440) at time base 1/16000" in text) is with_sound
    assert ("decoded samples [4320, 9440)" in text) is with_sound
    assert ("Mono soundtrack during these measured pictures (source time)" in text) is with_sound
    assert ("The soundtrack was not measured alongside these pictures." in text) is not with_sound
