from __future__ import annotations

import hashlib
import io

from PIL import Image

from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen


def test_eye_hold_pdf_preserves_scope_clock_and_all_three_exact_crops(monkeypatch):
    snapshot = specimen("video", "standard")
    assets = {}
    packet = {
        "id": "VPH03", "title": "Eye-region picture hold",
        "observation": "One eye region stayed stable while nearby picture detail changed.",
        "interpretation": "The crop shows an observable stability pattern.",
        "limitations": [], "finding_ids": [], "signal_refs": [],
        "eye_local_hold": {"frames": [16, 32], "anchors": [15, 24, 32], "box": [120, 96, 44, 28]},
        "window": {"frame_indices": list(range(16, 32)), "start_pts": 16, "end_pts": 32,
                   "time_base": "1/30", "start_stream_s": 16 / 30, "end_stream_s": 32 / 30},
        "frames": [], "artifacts": {},
    }
    for index, (role, number) in enumerate(zip(("before", "during", "after"), (15, 24, 32), strict=True), 1):
        refs = {}
        original = Image.new("RGB", (480, 360), (15 + index, 45, 122))
        for name, img in (("original", original), ("crop", original.crop((120, 96, 164, 124)))):
            data = io.BytesIO()
            img.save(data, format="PNG")
            digest = hashlib.sha256(data.getvalue()).hexdigest()
            assets[digest] = data.getvalue()
            ref = f"{role}_{name}"
            refs[name] = ref
            packet["artifacts"][ref] = {"sha256": digest, "media_type": "image/png"}
        packet["frames"].append({
            "id": f"VPH03.{index}", "score": None, "region_meaning": "measured_eye_region",
            "interval_role": role, "artifact_refs": refs,
            "context_region": {"x": 120, "y": 96, "w": 44, "h": 28},
            "anchor": {"frame_index": number, "pts": number, "time_base": "1/30", "stream_time_s": number / 30,
                       "playback_time_s": number / 30, "decoded_rgb_sha256": hashlib.sha256(original.tobytes()).hexdigest()},
        })
    document = {
        "schema": "chipmunk.video-evidence/1", "id": "synthetic-eye-pdf", "version": "video-evidence-2",
        "sha256": "a" * 64, "original_result_hash": "b" * 64, "packets": [packet],
        "source": {"sha256": snapshot.media.sha256, "width": 480, "height": 360, "frame_count": 60},
        "picture_hold": {"status": "no_match", "frames_scanned": 60, "continuity_breaks": 0, "last_stream_time_s": 59 / 30,
                         "eye_local_hold": {"status": "matched", "frames_scanned": 60, "packet_id": "VPH03"}},
    }
    monkeypatch.setattr(multimedia_evidence, "verified_set", lambda _: document)
    monkeypatch.setattr(multimedia_evidence, "verify_assets", lambda *_args, **_kwargs: assets)
    rendered = engine.render_pdf(snapshot)
    pdf_text = " ".join(engine.pdf_text(rendered.pdf).split())
    assert "Checked 60 verified source frames" in pdf_text
    assert "480 × 360 H.264 MP4 at 30 fps" in pdf_text
    assert "This check carries zero AI confidence weight" in pdf_text
    assert "Eye-region hold 0.533-1.067 s" in pdf_text
    assert "Measured source frames [16, 32) and source PTS [16, 32) at time base 1/30" in pdf_text
    assert "/picture_hold/eye_local_hold" in pdf_text.replace(" ", "")
    assert pdf_text.count("Exact measured eye-region crop") == 3
    for name, frame in (("Before", 15), ("During", 24), ("After", 32)):
        assert f"{name} eye-region hold - source frame {frame}" in pdf_text
