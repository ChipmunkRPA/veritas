import base64
import hashlib
import json
import shutil
import subprocess
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_image_reference_access import reference_http as reference_http

from app.api import v2_video_reference
from app.core.config import settings
from app.detectors.audio.decode import read_wav, wav_bytes
from app.detectors.video import decode
from app.evidence import video_reference
from app.storage.db import Analysis, Artefact, BillingOrderRow, Job, utcnow
from scripts.compare_video_references import run

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


def encode(folder, name, frames, samples=None, *, rate=25, origin=0):
    path = folder / (name + ".mov")
    height, width = frames.shape[1:3]
    args = ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(rate), "-i", "pipe:0"]
    if samples is not None:
        audio = folder / (name + ".wav")
        audio.write_bytes(wav_bytes(samples))
        args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "pcm_s16le"]
    args += ["-c:v", "png", "-pix_fmt", "rgb24", "-output_ts_offset", str(origin), str(path)]
    subprocess.run(args, input=frames.tobytes(), capture_output=True, check=True, timeout=20)
    return path


@pytest.fixture
def aligned_pair(tmp_path):
    rng = np.random.default_rng(314)
    frames = rng.integers(0, 256, (50, 48, 64, 3), dtype=np.uint8)
    pcm = rng.integers(-5000, 5000, 32000, dtype=np.int16).astype(np.float32) / 32768
    edited = frames.copy()
    edited[12:20, 7:19, 10:32, 1] ^= 255
    changed_audio = pcm.copy()
    changed_audio[3840:6720] += 500 / 32768
    earlier = encode(tmp_path, "earlier", frames, pcm)
    later = encode(tmp_path, "submitted", edited, changed_audio, origin=3)
    return earlier, later, frames, pcm


def test_clock_matched_video_and_sound_changes_have_exact_native_evidence(aligned_pair):
    earlier, later, _, pcm = aligned_pair
    result, assets = video_reference.compare(earlier, later)
    assert result["status"] == "measured" and result["affects_overall_confidence"] is False
    assert result["picture"]["matching_frames"] == 42
    assert result["picture"]["changed_frames"] == 8
    span, = result["picture"]["ranges"]
    assert (span["start_frame"], span["end_frame_exclusive"]) == (12, 20)
    assert span["strongest_frame"]["changed_pixels"] == 264
    assert span["strongest_frame"]["difference_bounds"] == {"x": 10, "y": 7, "w": 22, "h": 12}
    assert {label: item["frame"] for label, item in span["previews"].items()} == {"before": 11, "during": 12, "after": 20}
    for side, row in span["source_ranges"].items():
        scale = Fraction(result[side]["time_base"])
        assert (row["start_pts"] - result[side]["first_pts"]) * scale == Fraction(12, 25)
        assert (row["end_exclusive_pts"] - result[side]["first_pts"]) * scale == Fraction(20, 25)
    assert result["audio"]["status"] == "measured"
    sound, = result["audio"]["ranges"]
    assert (sound["start_sample"], sound["end_sample"], sound["changed_samples"]) == (3840, 6720, 2880)
    assert sound["source_ranges"]["submitted"]["start_pts"] - sound["source_ranges"]["earlier"]["start_pts"] == 48000
    np.testing.assert_array_equal(read_wav(assets[sound["preview"]["assets"]["earlier"]["file"]]), pcm[3840:6720])


def test_clock_geometry_and_audio_mismatch_stay_unassessed(tmp_path, aligned_pair):
    earlier, _, frames, pcm = aligned_pair
    different_clocks = encode(tmp_path, "alternate-clock", frames, rate=24)
    different_geometry = encode(tmp_path, "alternate-size", frames[:, :, :32])
    no_audio = encode(tmp_path, "no-audio", frames)
    different_audio = encode(tmp_path, "alternate-audio", frames, pcm[:-160])
    for path, status in ((different_clocks, "picture_timelines_differ"), (different_geometry, "picture_geometry_differs")):
        result, assets = video_reference.compare(earlier, path)
        assert result["status"] == status and "picture" not in result and not assets
    result, assets = video_reference.compare(earlier, no_audio)
    assert result["audio"] == {"status": "audio_unassessed", "earlier": "measured", "submitted": "no_stream"}
    assert result["picture"]["matching_frames"] == 50 and not assets
    result, _ = video_reference.compare(earlier, different_audio)
    assert result["audio"]["status"] == "audio_timelines_differ"
    result, _ = video_reference.compare(no_audio, no_audio)
    assert result["audio"]["status"] == "no_stream"


def test_overlarge_picture_is_refused_before_per_frame_probe(monkeypatch, aligned_pair):
    earlier, later, _, _ = aligned_pair
    monkeypatch.setattr(decode, "_stream_info", lambda _: {"width": 3840, "height": 2160, "time_base": "1/1000"})
    monkeypatch.setattr(decode.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("The full frame probe must not start"))
    with pytest.raises(decode.DecodeError, match="921,600 pixels"):
        video_reference.compare(earlier, later)


def test_offline_replay_checks_exact_bytes(tmp_path, aligned_pair):
    earlier, later, _, _ = aligned_pair
    target = tmp_path / "review"
    assert run(earlier, later, target) == run(earlier, later, target, verify=True)
    wav = next(target.glob("*.wav"))
    wav.write_bytes(wav.read_bytes() + b"other data")
    with pytest.raises(ValueError, match="differs from the re-decoded sources"):
        run(earlier, later, target, verify=True)
    with pytest.raises(ValueError, match="readable video files"):
        video_reference.compare(earlier, Path(__file__))


def test_authenticated_video_comparison_exports_exact_evidence_without_new_analysis(reference_http, aligned_pair, monkeypatch, tmp_path):
    f = reference_http
    earlier, later, _, _ = aligned_pair
    stored = f.store.put_file(later)
    monkeypatch.setattr(settings, "video_reference_comparison_enabled", True)

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_video_reference, "get_session", sessions)
    monkeypatch.setattr(v2_video_reference, "default_store", lambda: f.store)
    with sessions() as db:
        analysis = db.get(Analysis, "other-kind")
        analysis.filename, analysis.content_type = "submitted.mov", "video/quicktime"
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        db.commit()

    def state():
        with sessions() as db:
            rows = [db.scalars(select(model.id)).all() for model in (Analysis, Artefact, BillingOrderRow, Job)]
        return rows, sorted(path for path in f.store.root.rglob("*") if path.is_file())

    before = state()
    url = "/api/v2/analyses/other-kind/video-reference-comparison"
    files = {"earlier": ("reference.mov", earlier.read_bytes(), "video/quicktime")}
    assert f.client.get(url).status_code == 401
    assert f.client.post(url, files=files).status_code == 401
    assert f.client.post(url, files=files, headers=f.outsider).status_code == 404
    assert f.client.get(url, headers=f.viewer).json() == {"available": True}
    f.client.cookies.set("veritas_session", f.session)
    assert f.client.post(url, files=files).status_code == 403
    f.client.cookies.clear()
    response = f.client.post(url, headers=f.viewer, files=files, data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.json()
    record, assets = video_reference.compare(earlier, later)
    assert payload["comparison"] == record
    assert {name: base64.b64decode(body) for name, body in payload["assets"].items()} == assets
    assert hashlib.sha256(earlier.read_bytes()).hexdigest() == record["earlier"]["sha256"]
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as document:
        assert len(document) == 3
        text = " ".join(page.get_text() for page in document)
        assert "0.480 to 0.800 s" in text and "0.240 to 0.420 s" in text
        assert "Before" in text and "During" in text and "After" in text
        assert "Waveforms cover" in text
    exported = tmp_path / "comparison.json"
    exported.write_text(json.dumps({key: payload[key] for key in ("comparison", "assets")}))
    assert run(earlier, later, exported, verify=True) == record
    assert state() == before
    with sessions() as db:
        db.get(Analysis, "other-kind").withdrawn_at = utcnow()
        db.commit()
    assert f.client.get(url, headers=f.viewer).json() == {"available": False}
    assert f.client.post(url, headers=f.viewer, files=files).status_code == 404
