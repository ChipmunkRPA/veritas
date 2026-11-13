import base64
import hashlib
import io
import json
import shutil
import subprocess
import wave
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
from app.evidence import video_reference, video_reference_local, video_reference_timing
from app.reports.media_reference_pdf import render as render_comparison
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


def encode_lossy_comparison(path, pictures, *, lossless, crf=28, rate=25):
    height, width = pictures.shape[1:3]
    output = (["-c:v", "ffv1", "-pix_fmt", "bgr0"] if lossless
              else ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf)])
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-threads", "1", "-f", "rawvideo",
                    "-pixel_format", "rgb24", "-video_size", f"{width}x{height}", "-framerate", str(rate), "-i", "pipe:0",
                    *output, "-threads", "1", str(path)], input=pictures.tobytes(), capture_output=True, check=True, timeout=20)
    return path


def add_soundtracks(source, folder, name, tracks, *, subtitle_first=False):
    output = folder / (name + ".mov")
    inputs, streams = [], ["-map", "0:v:0", "-map", "0:a:0"]
    for index, pcm in enumerate(tracks, 1):
        path = folder / f"{name}-{index}.wav"
        path.write_bytes(wav_bytes(pcm))
        inputs.extend(["-i", str(path)])
        streams.extend(["-map", f"{index}:a:0"])
    if subtitle_first:
        captions = folder / "captions.srt"
        captions.write_text("1\n00:00:00,000 --> 00:00:01,000\nExample\n")
        inputs.extend(["-i", str(captions)])
        streams[0:0] = ["-map", f"{len(tracks)+1}:s:0"]
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), *inputs, *streams,
                    "-c:v", "copy", "-c:a", "pcm_s16le", "-c:s", "mov_text", str(output)],
                   capture_output=True, check=True, timeout=20)
    return output


def test_earlier_video_identifies_exact_move_in_one_stereo_channel(tmp_path):
    generator = np.random.default_rng(766)
    frames = generator.integers(0, 256, (70, 48, 64, 3), dtype=np.uint8)
    video = encode(tmp_path, "silent", frames)
    first = generator.integers(-6000, 6000, (44000, 2), dtype=np.int16)
    analyzed = first.copy()
    start, stop, offset = 15031, 22017, 3201
    analyzed[start:stop, 1] = first[start - offset:stop - offset, 1]

    def mux(name, pcm):
        audio = tmp_path / f"{name}.wav"
        with wave.open(str(audio), "wb") as output:
            output.setparams((2, 2, 16000, 0, "NONE", ""))
            output.writeframes(pcm.astype("<i2").tobytes())
        clip = tmp_path / f"{name}.mov"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-i", str(audio),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", str(clip)],
                       capture_output=True, check=True, timeout=20)
        return clip

    before, after = mux("stereo-original", first), mux("stereo-edited", analyzed)
    previous, _ = video_reference.compare(before, after)
    assert "channel_timing" not in previous
    result, assets = video_reference.compare(before, after, include_channel_timing=True)
    assert result["schema"] == video_reference.SCHEMA_WITH_CHANNEL_TIMING
    assert result["picture"]["changed_frames"] == 0
    stereo = result["channel_timing"]
    assert stereo["stream_indices"] == {"earlier": 1, "submitted": 1}
    assert stereo["affects_overall_confidence"] is False
    [right] = stereo["ranges"]
    assert right["channel"] == "right" and right["range"]["shift_samples"] == offset
    span = right["range"]
    assert span["submitted_samples"] == span["changed_samples"] == [start, stop]
    assert span["earlier"]["source_pts"] == [start - offset, stop - offset]
    actual = read_wav(assets[span["submitted"]["audio"]["file"]])
    np.testing.assert_array_equal(actual, analyzed[start:stop, 1] / 32768)
    with pymupdf.open(stream=render_comparison(result, assets), filetype="pdf") as document:
        assert "Right channel timing" in document[-1].get_text()
        assert "Analyzed clip" in document[-1].get_text()
        assert "Earlier stream 1" in document[-1].get_text() and "Analyzed stream 1" in document[-1].get_text()
        assert "verified picture timeline" in document[-1].get_text()
    exported = tmp_path / "stereo-export.json"
    exported.write_text(json.dumps({"comparison": result, "assets": {
        name: base64.b64encode(raw).decode() for name, raw in assets.items()}}))
    assert run(before, after, exported, verify=True) == result


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


def test_trimmed_picture_comparison_retains_both_clocks_and_export(tmp_path, aligned_pair, reference_http, monkeypatch):
    earlier, _, frames, _ = aligned_pair
    edited = frames[4:44].copy()
    edited[10:18, 7:19, 10:32, 1] ^= 255
    submitted = encode(tmp_path, "trimmed", edited, origin=3)
    assert video_reference.compare(earlier, submitted)[0]["status"] == "picture_timelines_differ"

    f = reference_http
    stored = f.store.put_file(submitted)
    monkeypatch.setattr(settings, "video_reference_comparison_enabled", True)
    monkeypatch.setattr(v2_media_reference, "default_store", lambda: f.store)

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_media_reference, "get_session", sessions)
    with sessions() as db:
        analysis = db.get(Analysis, "other-kind")
        analysis.filename, analysis.content_type = "trimmed.mov", "video/quicktime"
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        db.commit()
    response = f.client.post("/api/v2/analyses/other-kind/video-reference-comparison", headers=f.viewer,
                             files={"earlier": (earlier.name, earlier.read_bytes(), "video/quicktime")},
                             data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    payload = response.json()
    record = payload["comparison"]
    assert record["schema"] == video_reference.SCHEMA_WITH_TRIM and record["affects_overall_confidence"] is False
    assert record["trim_alignment"] == {"earlier_frames": [4, 44], "submitted_frames": [0, 40],
                                        "unique_matching_frames": 32, "audio_checked": False}
    assert "audio" not in record
    [span] = record["picture"]["ranges"]
    assert (span["start_s"], span["end_s"]) == (0.4, 0.72)
    assert (span["earlier_start_s"], span["earlier_end_s"]) == (0.56, 0.88)
    assert {role: (v["earlier_frame"], v["frame"]) for role, v in span["previews"].items()} == {
        "after": (22, 18), "before": (13, 9), "during": (14, 10),
    }
    for side, index, pixels in (("earlier", 14, frames[14]), ("submitted", 10, edited[10])):
        view = span["previews"]["during"]
        assert view[f"{side}_rgb_sha256"] == hashlib.sha256(pixels).hexdigest()
        assert view["assets"][side]["file"] == f"{side}-frame-{index:03d}.png"
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as document:
        text = " ".join(" ".join(page.get_text().split()) for page in document)
        assert "earlier frames [4, 44)" in text
        assert "Analyzed 0.400 to 0.720 s / Earlier 0.560 to 0.880 s" in text
        assert "The soundtrack was outside this trimmed picture comparison." in text
    exported = tmp_path / "trimmed-evidence.json"
    exported.write_text(json.dumps({key: payload[key] for key in ("comparison", "assets")}))
    assert run(earlier, submitted, exported, verify=True) == record
    record["trim_alignment"]["earlier_frames"][0] += 1
    exported.write_text(json.dumps({key: payload[key] for key in ("comparison", "assets")}))
    with pytest.raises(ValueError, match="differs from the re-decoded sources"):
        run(earlier, submitted, exported, verify=True)
    different_clock = encode(tmp_path, "trim-retimed", edited, rate=24)
    assert video_reference.compare(earlier, different_clock, include_trimmed=True)[0]["status"] == "picture_timelines_differ"


@pytest.mark.parametrize("rate", [24, 25])
def test_reencoded_trim_api_keeps_native_crops_and_two_clocks(tmp_path, reference_http, monkeypatch, rate):
    yy, xx = np.indices((480, 720))
    shade = 65 + xx * .12 + yy * .12 + 10 * np.cos(xx / 12)
    base = np.uint8(np.stack((shade, shade + 5, shade + 9), axis=-1))
    rng = np.random.default_rng(515)
    frames = np.repeat(base[None], 52, axis=0)
    for i, (x, y) in enumerate(rng.integers(100, 240, (len(frames), 2))):
        frames[i, y:y + 38, x:x + 45] = (214, 198, 151)
    altered = frames[5:-4].copy()
    altered[13:29, 157:271, 361:527] = (222, 217, 203)
    earlier = encode_lossy_comparison(tmp_path / "earlier.mp4", frames, lossless=False, crf=22, rate=rate)
    submitted = encode_lossy_comparison(tmp_path / "trimmed.mp4", altered, lossless=False, crf=28, rate=rate)
    assert video_reference.compare(earlier, submitted, include_trimmed=True)[0]["status"] == "picture_timelines_differ"

    f = reference_http
    stored = f.store.put_file(submitted)
    monkeypatch.setattr(settings, "video_reference_comparison_enabled", True)
    monkeypatch.setattr(settings, "video_reference_local_area", True)
    monkeypatch.setattr(v2_media_reference, "default_store", lambda: f.store)

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_media_reference, "get_session", sessions)
    with sessions() as db:
        analysis = db.get(Analysis, "other-kind")
        analysis.filename, analysis.content_type = submitted.name, "video/mp4"
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        db.commit()
    response = f.client.post("/api/v2/analyses/other-kind/video-reference-comparison", headers=f.viewer,
                             files={"earlier": (earlier.name, earlier.read_bytes(), "video/mp4")},
                             data={"include_pdf": "true"})
    assert response.status_code == 200 and response.headers["Cache-Control"] == "no-store", response.text
    payload = response.json()
    record = payload["comparison"]
    assert record["schema"] == video_reference.SCHEMA_WITH_TRIM_REENCODE
    assert record["affects_overall_confidence"] is False and "audio" not in record
    assert record["trim_alignment"]["earlier_frames"] == [5, 48]
    assert not record["picture"]["ranges"]
    span = record["picture"]["local_comparison"]["range"]
    for anchor in span["previews"].values():
        assert anchor["earlier_frame"] == anchor["frame"] + 5
    assert span["earlier_start_s"] - span["start_s"] == pytest.approx(5 / rate)
    assert span["earlier_end_s"] - span["end_s"] == pytest.approx(5 / rate)
    with pymupdf.open(stream=base64.b64decode(payload["pdf"]), filetype="pdf") as pdf:
        assert "Matching motion near both ends" in pdf[0].get_text()
        assert all(role in pdf[1].get_text() for role in ("Before", "During", "After"))
        assert f"Earlier {span['earlier_start_s']:.3f} to {span['earlier_end_s']:.3f} s" in pdf[1].get_text()
    export = {key: payload[key] for key in ("comparison", "assets")}
    path = tmp_path / "trimmed-reencoded.json"
    path.write_text(json.dumps(export))
    assert run(earlier, submitted, path, verify=True) == record
    assert verify_comparison(earlier, submitted, export)["native_crops_verified"] == 6
    span["previews"]["during"]["earlier_time_s"] = 0.
    with pytest.raises(ValueError, match="Preview time differs"):
        verify_comparison(earlier, submitted, export)


def test_second_soundtrack_keeps_exact_stream_and_source_sample_coordinates(tmp_path, aligned_pair, reference_http, monkeypatch):
    source, _, _, pcm = aligned_pair
    second = np.roll(pcm, 173)
    modified = second.copy()
    modified[12901:13401] += 501 / 32768
    earlier = add_soundtracks(source, tmp_path, "two-earlier", [second])
    later = add_soundtracks(source, tmp_path, "three-later", [modified, -pcm], subtitle_first=True)
    record, assets = video_reference.compare(earlier, later)
    track = record["additional_soundtrack"]
    assert record["schema"] == video_reference.SCHEMA_WITH_ADDITIONAL_SOUNDTRACK
    assert record["audio"]["changed_samples"] == 0 and record["picture"]["changed_frames"] == 0
    assert track["stream_indices"] == {"earlier": 2, "submitted": 3}
    assert track["inventory"]["submitted"] == {"stream_indices": [2, 3, 4], "unassessed_stream_indices": [4]}
    assert track["changed_samples"] == 500 and record["affects_overall_confidence"] is False
    span, = track["ranges"]
    assert span["id"] == "T2-A01"
    assert (span["start_s"], span["end_s"]) == (.8, .84)
    for side, data in (("earlier", second), ("submitted", modified)):
        assert span["source_ranges"][side] == {"start_pts": 12800, "end_exclusive_pts": 13440}
        wave = assets[span["preview"]["assets"][side]["file"]]
        np.testing.assert_array_equal(read_wav(wave), data[12800:13440])
    with pymupdf.open(stream=render_comparison(record, assets), filetype="pdf") as pdf:
        assert "Second soundtrack comparison" in pdf[-2].get_text()
        assert "500 decoded mono samples" in pdf[-2].get_text()
        assert "Difference waveform" in pdf[-1].get_text()
        assert "Source ticks [12800, 13440)" in pdf[-1].get_text()
    assert run(earlier, later, tmp_path / "replay") == record
    assert run(earlier, later, tmp_path / "replay", verify=True) == record

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
        analysis.filename, analysis.content_type = later.name, "video/quicktime"
        analysis.sha256, analysis.size_bytes = stored.sha256, stored.size_bytes
        db.commit()
    response = f.client.post("/api/v2/analyses/other-kind/video-reference-comparison", headers=f.viewer,
                             files={"earlier": (earlier.name, earlier.read_bytes(), "video/quicktime")},
                             data={"include_pdf": "true"})
    assert response.status_code == 200, response.text
    package = response.json()
    assert package["comparison"] == record
    assert {name: base64.b64decode(body) for name, body in package["assets"].items()} == assets
    with pymupdf.open(stream=base64.b64decode(package["pdf"]), filetype="pdf") as pdf:
        assert "Difference waveform" in pdf[-1].get_text()

    legacy, legacy_assets = video_reference.compare(earlier, later, include_additional_soundtrack=False)
    historical = tmp_path / "older.json"
    historical.write_text(json.dumps({"comparison": legacy, "assets": {k: base64.b64encode(v).decode() for k, v in legacy_assets.items()}}))
    assert run(earlier, later, historical, verify=True) == legacy
    without_pair, _ = video_reference.compare(source, later)
    assert without_pair["additional_soundtrack"]["status"] == "missing_counterpart"
    unequal = add_soundtracks(source, tmp_path, "unequal", [second[:-200]])
    misaligned, _ = video_reference.compare(earlier, unequal)
    assert misaligned["additional_soundtrack"]["status"] == "audio_timelines_differ"
    assert not misaligned["additional_soundtrack"].get("ranges")
    overlong = add_soundtracks(source, tmp_path, "overlong", [np.tile(second, 7)])
    unassessed, _ = video_reference.compare(earlier, overlong)
    assert unassessed["additional_soundtrack"]["status"] == "comparison_limit"


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


def test_picture_identical_soundtrack_shift_has_signed_exact_evidence(tmp_path):
    rng = np.random.default_rng(231)
    frames = rng.integers(0, 256, (75, 48, 64, 3), dtype=np.uint8)
    pcm = rng.integers(-6000, 6000, 48000, dtype=np.int16).astype(np.float32) / 32768
    modified = pcm.copy()
    modified[16000:26240] = pcm[12800:23040]
    earlier = encode(tmp_path, "timing-original", frames, pcm)
    later = encode(tmp_path, "timing-local-later", frames, modified)
    record, assets = video_reference.compare(earlier, later)
    assert record["schema"] == video_reference.SCHEMA_WITH_SOUNDTRACK
    timing = record["soundtrack_alignment"]
    assert timing["status"] == "local_shift" and timing["offset_ms"] == 200
    assert not timing["affects_overall_confidence"] and timing["picture_pair_sha256"] == record["picture"]["frame_pairs_sha256"]
    span = timing["range"]
    assert span["submitted"]["samples"] == [16000, 26240]
    assert span["earlier"]["samples"] == [12800, 23040]
    np.testing.assert_array_equal(read_wav(assets[span["submitted"]["audio"]["file"]]), pcm[12800:23040])
    np.testing.assert_array_equal(read_wav(assets[span["earlier"]["audio"]["file"]]), pcm[12800:23040])
    with Image.open(io.BytesIO(assets[span["picture"]["asset"]["file"]])) as picture:
        np.testing.assert_array_equal(np.asarray(picture), frames[span["picture"]["frame_index"]])
    target = tmp_path / "timing-review"
    assert run(earlier, later, target) == run(earlier, later, target, verify=True) == record
    global_shift = encode(tmp_path, "timing-global", frames, np.r_[np.zeros(3200, dtype=np.float32), pcm[:-3200]])
    global_record, _ = video_reference.compare(earlier, global_shift)
    assert global_record["soundtrack_alignment"]["status"] == "no_local_shift"


@pytest.mark.parametrize(("lossless", "local_area"), [(True, False), (False, False), (False, True)],
                         ids=["earlier-ffv1", "earlier-h264", "earlier-h264-local-area"])
def test_bounded_lossy_pair_keeps_exact_results_and_replays_local_native_crops(
    tmp_path, reference_http, monkeypatch, lossless, local_area
):
    yy, xx = np.indices((480, 720) if local_area else (288, 360))
    shade = 42 + xx * (.12 if local_area else .27) + yy * (.13 if local_area else .21) + 18 * np.cos(xx / 13) * np.sin(yy / 15)
    base = np.uint8(np.stack((shade, shade + 7, shade - 5), axis=-1))
    earlier_frames = np.stack([np.roll(base, i % 4, axis=1) for i in range(68)])
    later_frames = earlier_frames.copy()
    y0, y1, x0, x1 = (176, 288, 110, 290) if local_area else (176, 220, 110, 174)
    later_frames[24:44, y0:y1, x0:x1] = (224, 223, 219)
    extension, mime = ("mkv", "video/x-matroska") if lossless else ("mp4", "video/mp4")
    rate = 24 if local_area else 25
    earlier = encode_lossy_comparison(tmp_path / f"earlier.{extension}", earlier_frames, lossless=lossless, crf=23, rate=rate)
    later = encode_lossy_comparison(tmp_path / f"submitted.{extension}", later_frames, lossless=False, rate=rate)
    neutral = encode_lossy_comparison(tmp_path / f"reencoded.{extension}", earlier_frames, lossless=False, rate=rate)
    wrong = encode_lossy_comparison(tmp_path / f"different.{extension}", 255 - earlier_frames, lossless=lossless, crf=23, rate=rate)

    record, assets = video_reference.compare(earlier, later, include_local_area=local_area)
    local = record["picture"]["local_comparison"]
    assert record["schema"] == video_reference.SCHEMA_WITH_LOCAL
    assert record["affects_overall_confidence"] is False and local["status"] == "local_difference"
    if local_area:
        assert local["basis"] == "paired_h264_local_area/1"
        assert "overall brightness and contrast" in local["scope"]
        assert "local_comparison" not in video_reference.compare(earlier, later)[0]["picture"]
    else:
        assert local["scope"] == (video_reference_local.SCOPE if lossless else video_reference_local.H264_SCOPE)
    span = local["range"]
    assert record["picture"]["matching_frames"] == 0
    assert max(span["start_s"], .96) < min(span["end_s"], 1.76)
    x, y, w, h = (span["tile"][key] for key in ("x", "y", "w", "h"))
    if local_area:
        assert x <= x0 < x1 <= x + w and y <= y0 < y1 <= y + h
    else:
        assert x0 <= x < x1 and y0 <= y < y1
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
    monkeypatch.setattr(settings, "video_reference_local_area", local_area)

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
    assert video_reference.compare(earlier, neutral, include_local_area=local_area)[0]["picture"]["local_comparison"]["status"] == "no_local_range"
    assert video_reference.compare(wrong, later, include_local_area=local_area)[0]["picture"]["local_comparison"]["status"] == "insufficient_shared_picture_content"


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
    malformed = f.client.post(url, headers=f.viewer, files=files, data={"timing_event_s": "0.2400"})
    assert malformed.status_code == 422
    optional = f.client.post(url, headers=f.viewer, files=files, data={"timing_event_s": "0.240", "include_pdf": "true"})
    assert optional.status_code == 200
    timing = optional.json()["comparison"]["timing"]
    assert timing["status"] == "unassessed" and timing["reason"] == "profile"
    assert timing["requested_earlier_time_s"] == "6/25" and timing["affects_overall_confidence"] is False
    assert optional.json()["comparison"]["schema"] == video_reference.SCHEMA_WITH_MARKED_SOUND_FALLBACK
    assert optional.json()["assets"] == payload["assets"]
    with pymupdf.open(stream=base64.b64decode(optional.json()["pdf"]), filetype="pdf") as document:
        assert len(document) == 4
        timing_text = " ".join(document[-1].get_text().split())
        assert "Marked mouth and speech timing" in timing_text
        assert "continuous sound" in timing_text
    optional_json = tmp_path / "comparison-marked-unassessed.json"
    optional_json.write_text(json.dumps({key: optional.json()[key] for key in ("comparison", "assets")}))
    assert run(earlier, later, optional_json, verify=True) == optional.json()["comparison"]
    assert state() == before
    with sessions() as db:
        db.get(Analysis, "other-kind").withdrawn_at = utcnow()
        db.commit()
    assert f.client.get(url, headers=f.viewer).json() == {"available": False}
    assert f.client.post(url, headers=f.viewer, files=files).status_code == 404


def test_marked_timing_capability_requires_exact_optional_task_bytes(monkeypatch, tmp_path):
    model = tmp_path / "local.task"
    model.write_bytes(b"different task")
    monkeypatch.setattr(video_reference_timing.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(video_reference_timing.importlib.metadata, "version", lambda _: video_reference_timing.RUNTIME_VERSION)
    assert not video_reference_timing.available(model)
    monkeypatch.setattr(video_reference_timing, "MODEL_SHA256", hashlib.sha256(model.read_bytes()).hexdigest())
    assert video_reference_timing.available(model)


def test_short_marked_timing_recovers_a_shift_only_when_existing_match_lacks_support(monkeypatch):
    anchor = 28
    start = anchor * 640
    moved = start + 3840
    generator = np.random.default_rng(1356)
    earlier = generator.integers(-2000, 2000, 47648, dtype=np.int16)
    submitted = earlier.copy()
    submitted[start:start + 7000] = generator.integers(-2000, 2000, 7000, dtype=np.int16)
    submitted[moved:moved + 960] = earlier[start:start + 960]
    visual = {"source_frame": anchor, "submitted_frame": anchor, "offset_pictures": 0}
    monkeypatch.setattr(video_reference_timing, "visual_shift", lambda *_: visual)

    assert video_reference_timing.compare(None, None, earlier, submitted, anchor)["status"] == "unassessed"
    result = video_reference_timing.compare(None, None, earlier, submitted, anchor, short_fallback=True)
    assert result["status"] == "timing_difference"
    assert result["picture_minus_audio_shift_s"] == "-6/25"
    assert result["waveform_basis"] == video_reference_timing.SHORT_WAVEFORM_BASIS
    assert result["sound"]["source_samples"] == [start, start + 960]
    assert result["sound"]["submitted_samples"] == [moved, moved + 960]

    previous = video_reference_timing.compare(None, None, earlier, earlier, anchor)
    assert previous["status"] == "no_large_timing_difference"
    assert video_reference_timing.compare(None, None, earlier, earlier, anchor, short_fallback=True) == previous


def test_marked_local_sound_requires_exact_excerpt_and_unchanged_flanks():
    generator = np.random.default_rng(1356)
    earlier = generator.integers(-2000, 2000, 43200, dtype=np.int16)
    anchor, offset = 14400, 3200
    start, stop = 12800, 22400
    submitted = earlier.copy()
    submitted[start:stop] = earlier[start - offset:stop - offset]
    sound, short, changed = video_reference_timing.local_sound_match(earlier, submitted, anchor)
    assert changed == [start, stop]
    assert not short
    assert sound["source_samples"] == [15040, 17600]
    assert sound["submitted_samples"] == [18240, 20800]
    assert sound["offset_samples"] == offset
    assert sound["source_pcm_sha256"] == sound["submitted_pcm_sha256"]

    altered = submitted.copy()
    altered[sound["submitted_samples"][0]:sound["submitted_samples"][1]] += 1
    assert video_reference_timing.local_sound_match(earlier, altered, anchor) is None
    shifted = np.r_[np.zeros(offset, dtype=np.int16), earlier[:-offset]]
    assert video_reference_timing.local_sound_match(earlier, shifted, anchor) is None
    assert video_reference_timing.local_sound_match(earlier, earlier, anchor) is None
