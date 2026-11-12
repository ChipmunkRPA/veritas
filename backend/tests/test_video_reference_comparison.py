import base64
import hashlib
import io
import json
import shutil
import subprocess
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_image_reference_access import reference_http as reference_http

from app.api import v2_media_reference
from app.core.canonical import canonical_json
from app.core.config import settings
from app.detectors.audio.decode import read_wav, wav_bytes
from app.detectors.video import decode
from app.evidence import video_reference, video_reference_local
from app.storage.db import Analysis, Artefact, BillingOrderRow, Job, utcnow
from scripts.compare_video_references import run
from scripts.verify_native_video_interval import verify_comparison

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


def encode_lossy_comparison(path, pictures, *, lossless, crf=28):
    height, width = pictures.shape[1:3]
    output = (["-c:v", "ffv1", "-pix_fmt", "bgr0"] if lossless
              else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)])
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-threads", "1", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", "25", "-i", "pipe:0",
                    *output, "-threads", "1", str(path)], input=pictures.tobytes(), capture_output=True, check=True, timeout=20)
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


@pytest.mark.parametrize("lossless", [True, False], ids=["earlier-ffv1", "earlier-h264"])
def test_bounded_lossy_pair_keeps_exact_results_and_replays_local_native_crops(tmp_path, reference_http, monkeypatch, lossless):
    yy, xx = np.indices((288, 360))
    shade = 42 + xx * .27 + yy * .21 + 18 * np.cos(xx / 13) * np.sin(yy / 15)
    base = np.uint8(np.stack((shade, shade + 7, shade - 5), axis=-1))
    earlier_frames = np.stack([np.roll(base, i % 4, axis=1) for i in range(68)])
    later_frames = earlier_frames.copy()
    later_frames[24:44, 176:220, 110:174] = (224, 223, 219)
    extension, mime = ("mkv", "video/x-matroska") if lossless else ("mp4", "video/mp4")
    earlier = encode_lossy_comparison(tmp_path / f"earlier.{extension}", earlier_frames, lossless=lossless, crf=23)
    later = encode_lossy_comparison(tmp_path / f"submitted.{extension}", later_frames, lossless=False)
    neutral = encode_lossy_comparison(tmp_path / f"reencoded.{extension}", earlier_frames, lossless=False)
    wrong = encode_lossy_comparison(tmp_path / f"different.{extension}", 255 - earlier_frames, lossless=lossless, crf=23)

    record, assets = video_reference.compare(earlier, later)
    local = record["picture"]["local_comparison"]
    assert record["schema"] == video_reference.SCHEMA_WITH_LOCAL
    assert record["affects_overall_confidence"] is False and local["status"] == "local_difference"
    assert local["scope"] == (video_reference_local.SCOPE if lossless else video_reference_local.H264_SCOPE)
    span = local["range"]
    assert record["picture"]["matching_frames"] == 0
    assert max(span["start_s"], .96) < min(span["end_s"], 1.76)
    x, y, w, h = (span["tile"][key] for key in ("x", "y", "w", "h"))
    assert 110 <= x < 174 and 176 <= y < 220
    assert list(span["previews"]) == ["before", "during", "after"]
    assert span["previews"]["before"]["frame"] == span["start_frame"] - 1
    assert span["previews"]["after"]["frame"] == span["end_frame_exclusive"]
    for side, path in (("earlier", earlier), ("submitted", later)):
        table = decode.probe_frames(path)
        needed = [row["frame"] for row in span["previews"].values()]
        decoded = {f.index: f for f in decode.decode_frames(path, table, needed)}
        for sample in span["previews"].values():
            frame = decoded[sample["frame"]]
            entry = sample["crop_assets"][side]
            assert frame.pts_verified and frame.pts == sample[f"{side}_pts"]
            assert entry["sha256"] == hashlib.sha256(assets[entry["file"]]).hexdigest()
            with Image.open(io.BytesIO(assets[entry["file"]])) as png:
                np.testing.assert_array_equal(np.asarray(png), frame.rgb[y:y+h, x:x+w])

    f = reference_http
    stored = f.store.put_file(later)
    monkeypatch.setattr(settings, "video_reference_comparison_enabled", True)

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_media_reference, "get_session", sessions)
    monkeypatch.setattr(v2_media_reference, "default_store", lambda: f.store)
    with sessions() as db:
        analysis = db.get(Analysis, "other-kind")
        analysis.filename, analysis.content_type = later.name, mime
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        db.commit()
    response = f.client.post("/api/v2/analyses/other-kind/video-reference-comparison", headers=f.viewer,
                             data={"include_pdf": "true"},
                             files={"earlier": (earlier.name, earlier.read_bytes(), mime)})
    assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store"
    package = response.json()
    assert package["comparison"] == record
    assert {name: base64.b64decode(data) for name, data in package["assets"].items()} == assets

    with pymupdf.open(stream=base64.b64decode(package["pdf"]), filetype="pdf") as pdf:
        assert "Local picture comparison" in pdf[0].get_text()
        assert f"{span['start_s']:.3f} to {span['end_s']:.3f} s" in pdf[0].get_text()
        assert "Local picture difference" in pdf[1].get_text()
        assert all(role in pdf[1].get_text() for role in ("Before", "During", "After"))
        assert any(image[2:4] == (w, h) for image in pdf[1].get_images())

    export = tmp_path / "pair.json"
    payload = {"comparison": record, "assets": {key: base64.b64encode(value).decode() for key, value in assets.items()}}
    export.write_text(json.dumps(payload))
    assert run(earlier, later, export, verify=True) == record
    assert verify_comparison(earlier, later, payload)["native_crops_verified"] == 6
    if not lossless:
        altered = deepcopy(payload)
        previews = altered["comparison"]["picture"]["local_comparison"]["range"]["previews"]
        previews["during"]["crop_assets"]["submitted"] = previews["before"]["crop_assets"]["submitted"]
        with pytest.raises(ValueError, match="differs from the committed native picture"):
            verify_comparison(earlier, later, altered)
        previews["during"]["time_s"] += .04
        with pytest.raises(ValueError, match="Preview time differs"):
            verify_comparison(earlier, later, altered)
    local["range"]["tile"]["x"] += 1
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="differs from the re-decoded sources"):
        run(earlier, later, export, verify=True)

    v1, v1_assets = video_reference.compare(earlier, later, include_local=False)
    assert v1["schema"] == video_reference.SCHEMA and "local_comparison" not in v1["picture"]
    assert v1["picture"] == {key: value for key, value in record["picture"].items() if key != "local_comparison"}
    historical = tmp_path / "historical"
    historical.mkdir()
    for name, data in {"comparison.json": canonical_json(v1) + b"\n", **v1_assets}.items():
        (historical / name).write_bytes(data)
    assert run(earlier, later, historical, verify=True) == v1
    assert video_reference.compare(earlier, neutral)[0]["picture"]["local_comparison"]["status"] == "no_local_range"
    assert video_reference.compare(wrong, later)[0]["picture"]["local_comparison"]["status"] == "insufficient_shared_picture_content"


@pytest.mark.parametrize(("width", "height", "step"), [(360, 288, 400001), (480, 360, 333333)])
def test_earlier_h264_requires_exact_grid_dimensions_and_cadence(monkeypatch, width, height, step):
    table = decode.FrameTable("1/10000000", (0, step, 2 * step), (0,), "IPP", width, height, width, height)
    clock = video_reference._elapsed(table)
    paths = Path("earlier"), Path("submitted")
    monkeypatch.setattr(video_reference_local, "_carrier", lambda _: [("h264", "yuv420p")])
    assert video_reference_local.eligible(paths, (table, table), clock) is None
    monkeypatch.setattr(video_reference_local, "_carrier", lambda path: [("ffv1", "bgr0")] if path == paths[0]
                        else [("h264", "yuv420p")])
    assert video_reference_local.eligible(paths, (table, table), clock)[1] == video_reference_local.SCOPE


def test_authenticated_video_comparison_exports_exact_evidence_without_new_analysis(reference_http, aligned_pair, monkeypatch, tmp_path):
    f = reference_http
    earlier, later, _, _ = aligned_pair
    stored = f.store.put_file(later)
    monkeypatch.setattr(settings, "video_reference_comparison_enabled", True)

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_media_reference, "get_session", sessions)
    monkeypatch.setattr(v2_media_reference, "default_store", lambda: f.store)
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
