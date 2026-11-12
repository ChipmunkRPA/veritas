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

from app.api import auth, v2_media_reference
from app.core.canonical import canonical_hash
from app.core.config import settings
from app.detectors.audio.decode import read_wav, wav_bytes
from app.evidence import audio_reference
from app.storage.db import Analysis, Artefact, BillingOrderRow, Job, utcnow
from scripts.compare_audio_references import run

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


def save(folder, name, samples):
    path = folder / f"{name}.wav"
    path.write_bytes(wav_bytes(samples))
    return path


@pytest.fixture
def aligned_audio_pair(tmp_path):
    rng = np.random.default_rng(418)
    earlier = rng.integers(-4000, 4000, 32031, dtype=np.int16)
    earlier[4001], earlier[4090] = -32768, 32767
    analyzed = earlier.copy()
    analyzed[3841:6721] += 800
    analyzed[4001], analyzed[4090] = 32767, -32768
    analyzed[14401] += 1
    analyzed[-1] -= 1
    before, after = (source.astype(np.float32) / 32768 for source in (earlier, analyzed))
    return save(tmp_path, "earlier", before), save(tmp_path, "analyzed", after), before, after


def test_audio_reference_measures_exact_sample_counts_and_delta_waveform(aligned_audio_pair):
    earlier, analyzed, before, after = aligned_audio_pair
    record, assets = audio_reference.compare(earlier, analyzed)
    assert record["status"] == "measured"
    assert record["affects_overall_confidence"] is False
    audio = record["audio"]
    assert audio["sample_rate"] == 16000 and audio["bin_samples"] == 320
    assert audio["changed_samples"] == 2882
    independent_bins = [
        int(np.count_nonzero(before[start:start + 320] != after[start:start + 320]))
        for start in range(0, len(before), 320)
    ]
    assert audio["bin_counts_sha256"] == canonical_hash(independent_bins)
    spans = audio["ranges"]
    assert [(s["start_sample"], s["end_sample"], s["changed_samples"]) for s in spans] == [
        (3840, 7040, 2880), (14400, 14720, 1), (32000, 32031, 1),
    ]
    for span in spans:
        start, end = span["start_sample"], span["end_sample"]
        assert span["start_s"] == start / 16000 and span["end_s"] == end / 16000
        assert span["source_ranges"] == {
            side: {"start_pts": start, "end_exclusive_pts": end} for side in ("earlier", "submitted")
        }
        delta = after[start:end].astype(np.float64) - before[start:end]
        assert span["difference"]["peak"] == max(abs(delta))
        assert span["difference"]["rms"] == pytest.approx(np.linalg.norm(delta) / np.sqrt(len(delta)))
        preview = span["preview"]
        assert (preview["start_sample"], preview["end_sample"]) == (start, end)
        for side, original in (("earlier", before), ("submitted", after)):
            info = preview["assets"][side]
            assert hashlib.sha256(assets[info["file"]]).hexdigest() == info["sha256"]
            np.testing.assert_array_equal(read_wav(assets[info["file"]]), original[start:end])
    residuals = [span["preview"]["difference_waveform"] for span in spans]
    assert max(residuals[0]["high"]) == round(65535 / 32768, 8)
    assert min(residuals[0]["low"]) == -round(65535 / 32768, 8)
    assert max(residuals[1]["high"]) == round(1 / 32768, 8)
    assert min(residuals[2]["low"]) == -round(1 / 32768, 8)


def test_lossless_container_change_matches_and_trim_is_unassessed(tmp_path, aligned_audio_pair):
    earlier, _, pcm, _ = aligned_audio_pair
    flac = tmp_path / "same.flac"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(earlier), str(flac)],
                   capture_output=True, check=True, timeout=20)
    matched, assets = audio_reference.compare(earlier, flac)
    assert matched["status"] == "measured" and not assets
    assert matched["earlier"]["sha256"] != matched["submitted"]["sha256"]
    assert matched["audio"]["earlier"]["pcm_sha256"] == matched["audio"]["submitted"]["pcm_sha256"]
    assert matched["audio"]["changed_samples"] == 0 and not matched["audio"]["ranges"]

    trimmed = save(tmp_path, "trimmed", pcm[:-1])
    result, assets = audio_reference.compare(earlier, trimmed)
    assert result["status"] == result["audio"]["status"] == "audio_timelines_differ"
    assert result["audio"]["earlier"]["samples"] == len(pcm)
    assert result["audio"]["submitted"]["samples"] == len(pcm) - 1
    assert "ranges" not in result["audio"] and not assets


def test_matching_nonzero_audio_starts_preserve_both_source_clocks(tmp_path, aligned_audio_pair):
    earlier, analyzed, _, _ = aligned_audio_pair
    shifted = []
    for source in (earlier, analyzed):
        path = tmp_path / f"{source.stem}-offset.m4a"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source), "-c:a", "alac",
             "-output_ts_offset", "1.25", str(path)],
            capture_output=True, check=True, timeout=20,
        )
        shifted.append(path)
    unmatched, assets = audio_reference.compare(earlier, shifted[0])
    assert unmatched["status"] == "audio_timelines_differ" and not assets
    assert unmatched["audio"]["earlier"]["pcm_sha256"] == unmatched["audio"]["submitted"]["pcm_sha256"]
    assert unmatched["audio"]["earlier"]["source_pts"] == 0
    assert unmatched["audio"]["submitted"]["source_pts"] == 20000

    matched, _ = audio_reference.compare(*shifted)
    assert matched["status"] == "measured"
    first = matched["audio"]["ranges"][0]
    assert (first["start_s"], first["end_s"]) == (0.24, 0.44)
    assert first["source_ranges"] == {
        side: {"start_pts": 23840, "end_exclusive_pts": 27040} for side in ("earlier", "submitted")
    }


def test_duration_limit_and_upload_type_fail_without_comparison(tmp_path, aligned_audio_pair, monkeypatch):
    earlier, _, _, _ = aligned_audio_pair
    boundary = audio_reference.MAX_SECONDS * audio_reference.AUDIO_RATE
    permitted = save(tmp_path, "duration-boundary", np.zeros(boundary, dtype=np.float32))
    result, assets = audio_reference.compare(permitted, permitted)
    assert result["status"] == "measured" and result["audio"]["changed_samples"] == 0 and not assets

    over = save(tmp_path, "over-duration", np.zeros(boundary + 1, dtype=np.float32))
    result, assets = audio_reference.compare(earlier, over)
    assert result["status"] == "comparison_limit"
    assert result["audio"] == {"status": "audio_unassessed", "earlier": "measured", "submitted": "comparison_limit"}
    assert not assets
    with pytest.raises(ValueError, match="readable audio files"):
        audio_reference.compare(earlier, Path(__file__))
    monkeypatch.setattr(audio_reference, "MAX_BYTES", earlier.stat().st_size - 1)
    with pytest.raises(ValueError, match="at most 48 MiB"):
        audio_reference.compare(earlier, permitted)


def test_offline_replay_rejects_changed_audio_media_or_manifest(tmp_path, aligned_audio_pair):
    earlier, analyzed, _, _ = aligned_audio_pair
    target = tmp_path / "review"
    assert run(earlier, analyzed, target) == run(earlier, analyzed, target, verify=True)
    wav = next(target.glob("*.wav"))
    wav.write_bytes(wav.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="differs from the re-decoded sources"):
        run(earlier, analyzed, target, verify=True)

    record, assets = audio_reference.compare(earlier, analyzed)
    downloaded = tmp_path / "export.json"
    wrapper = {"comparison": record, "assets": {name: base64.b64encode(body).decode() for name, body in assets.items()}}
    downloaded.write_text(json.dumps(wrapper))
    assert run(earlier, analyzed, downloaded, verify=True) == record
    record["audio"]["ranges"][0]["difference"]["peak"] = 0.5
    downloaded.write_text(json.dumps(wrapper))
    with pytest.raises(ValueError, match="exported JSON or media differs"):
        run(earlier, analyzed, downloaded, verify=True)


@pytest.fixture
def adjusted_audio_pair(tmp_path):
    original = np.random.default_rng(2309).integers(-8192, 8193, 35591, dtype=np.int16)
    adjusted = np.rint(original * 0.625).astype(np.int16)
    altered = adjusted.copy()
    altered[9602:12521] += 502
    paths = [save(tmp_path, name, samples.astype(np.float32) / 32768)
             for name, samples in (("volume-original", original), ("volume-adjusted", adjusted),
                                   ("volume-altered", altered))]
    return paths, original, adjusted, altered


def test_volume_adjustment_proves_exact_changed_samples_and_preserves_raw(tmp_path, adjusted_audio_pair):
    (earlier, adjusted, altered), original, _, edited_pcm = adjusted_audio_pair
    clean, _ = audio_reference.compare(earlier, adjusted)
    assert clean["schema"] == audio_reference.SCHEMA_WITH_LEVEL_ADJUSTMENT
    assert clean["audio"]["changed_samples"] > 35000
    assert clean["level_adjusted"]["changed_samples"] == 0 and clean["level_adjusted"]["ranges"] == []

    record, assets = audio_reference.compare(earlier, altered)
    raw, raw_assets = audio_reference.compare(earlier, altered, include_level_adjustment=False)
    assert record["audio"] == raw["audio"]
    assert all(assets[name] == content for name, content in raw_assets.items())
    assert record["schema"] == audio_reference.SCHEMA_WITH_LEVEL_ADJUSTMENT
    assert record["affects_overall_confidence"] is False
    measured = record["level_adjusted"]
    assert measured["gain_db"] == pytest.approx(-4.082, abs=0.001)

    def bounds(start, end):
        admissible = [tuple(sorted((Fraction(2 * int(y) - 1, 2 * int(x)), Fraction(2 * int(y) + 1, 2 * int(x)))))
                      for x, y in zip(original[start:end], edited_pcm[start:end], strict=True) if x != 0]
        return max(interval[0] for interval in admissible), min(interval[1] for interval in admissible)

    support_ranges = ((0, 8000), (len(original) - 8000, len(original)))
    support = measured["support"]
    assert len(support) == 2
    for (start, end), anchor in zip(support_ranges, support, strict=True):
        assert (anchor["start_sample"], anchor["end_sample"]) == (start, end)
        assert tuple(Fraction(anchor["gain_bounds"][key]) for key in ("lower", "upper")) == bounds(start, end)
        assert anchor["pcm_sha256"] == {
            side: hashlib.sha256(samples[start:end].astype("<i2").tobytes()).hexdigest()
            for side, samples in (("earlier", original), ("submitted", edited_pcm))
        }
    first, last = (bounds(*interval) for interval in support_ranges)
    lower, upper = max(first[0], last[0]), min(first[1], last[1])
    assert measured["gain_bounds"] == {"lower": str(lower), "upper": str(upper)}

    def confirmed(x, y):
        predicted = sorted((int(x) * lower, int(x) * upper))
        return int(y) - Fraction(1, 2) > predicted[1] or int(y) + Fraction(1, 2) < predicted[0]

    mask = [confirmed(x, y) for x, y in zip(original, edited_pcm, strict=True)]
    assert measured["bin_counts_sha256"] == canonical_hash(
        [sum(mask[i:i + 320]) for i in range(0, len(mask), 320)])
    assert measured["changed_samples"] == sum(mask) == 2919
    [span] = measured["ranges"]
    assert (span["id"], span["start_sample"], span["end_sample"]) == ("V01", 9600, 12800)
    assert span["source_ranges"] == {side: {"start_pts": 9600, "end_exclusive_pts": 12800}
                                     for side in ("earlier", "submitted")}
    for side, samples in (("earlier", original), ("submitted", edited_pcm)):
        asset = span["preview"]["assets"][side]
        np.testing.assert_array_equal(read_wav(assets[asset["file"]]) * 32768, samples[9600:12800])
    residual = [int(y) - round(int(x) * (lower + upper) / 2)
                for x, y in zip(original[9600:12800], edited_pcm[9600:12800], strict=True)]
    assert span["difference"]["peak"] == max(map(abs, residual)) / 32768

    downloaded = tmp_path / "adjusted-export.json"
    downloaded.write_text(json.dumps({"comparison": record,
                                     "assets": {name: base64.b64encode(body).decode() for name, body in assets.items()}}))
    assert run(earlier, altered, downloaded, verify=True) == record
    tampered = json.loads(downloaded.read_bytes())
    tampered["comparison"]["level_adjusted"]["changed_samples"] += 1
    downloaded.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="exported JSON or media differs"):
        run(earlier, altered, downloaded, verify=True)
    downloaded.write_text(json.dumps({"comparison": raw,
                                     "assets": {name: base64.b64encode(body).decode() for name, body in raw_assets.items()}}))
    assert run(earlier, altered, downloaded, verify=True) == raw


def test_volume_adjustment_abstains_without_matching_support_or_native_samples(tmp_path, adjusted_audio_pair):
    (earlier, adjusted, _), _, gain_only, _ = adjusted_audio_pair
    clipped, changed_anchor = gain_only.copy(), gain_only.copy()
    clipped[9600] = 32767
    changed_anchor[3122] += 400
    incompatible = [save(tmp_path, name, samples.astype(np.float32) / 32768)
                    for name, samples in (("clipped", clipped), ("changed-anchor", changed_anchor))]
    stereo = tmp_path / "gain-stereo.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(adjusted), "-ac", "2", str(stereo)],
                   capture_output=True, check=True, timeout=20)
    for variant in (*incompatible, stereo):
        record, _ = audio_reference.compare(earlier, variant)
        assert record["schema"] == audio_reference.SCHEMA and record["status"] == "measured"
        assert "level_adjusted" not in record and record["audio"]["changed_samples"] > 0


def test_authenticated_audio_comparison_pdf_json_and_temporary_storage(
        reference_http, aligned_audio_pair, monkeypatch, tmp_path):
    f = reference_http
    earlier, analyzed, _, _ = aligned_audio_pair
    stored = f.store.put_file(analyzed)
    monkeypatch.setattr(settings, "audio_reference_comparison_enabled", True)

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_media_reference, "get_session", sessions)
    monkeypatch.setattr(v2_media_reference, "default_store", lambda: f.store)
    with sessions() as db:
        row = db.get(Analysis, "other-kind")
        row.kind, row.filename, row.content_type = "audio", "analyzed.wav", "audio/wav"
        row.sha256, row.size_bytes = stored.sha256, stored.size_bytes
        db.commit()

    def state():
        with sessions() as db:
            return (
                db.execute(select(Analysis.id, Analysis.result_hash).order_by(Analysis.id)).all(),
                [db.scalars(select(model.id).order_by(model.id)).all() for model in (Artefact, BillingOrderRow, Job)],
                {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in f.store.root.rglob("*") if path.is_file()},
            )

    initial = state()
    endpoint = "/api/v2/analyses/other-kind/audio-reference-comparison"
    files = {"earlier": ("earlier.wav", earlier.read_bytes(), "audio/wav")}
    assert f.client.get(endpoint).status_code == 401
    assert f.client.post(endpoint, files=files).status_code == 401
    assert f.client.get(endpoint, headers=f.outsider).status_code == 404
    assert f.client.post(endpoint, headers=f.outsider, files=files).status_code == 404
    assert f.client.get(endpoint, headers=f.viewer).json() == {"available": True}
    wrong_kind = endpoint.replace("other-kind", "image-a")
    assert f.client.get(wrong_kind, headers=f.viewer).json() == {"available": False}
    with monkeypatch.context() as config:
        config.setattr(settings, "audio_reference_comparison_enabled", False)
        assert f.client.get(endpoint, headers=f.viewer).json() == {"available": False}
        assert f.client.post(endpoint, headers=f.viewer, files=files).status_code == 404

    f.client.cookies.set(auth.SESSION_COOKIE, f.session)
    assert f.client.post(endpoint, files=files).status_code == 403
    response = f.client.post(endpoint, headers={auth.CSRF_HEADER: f.csrf}, files=files, data={"include_pdf": "true"})
    f.client.cookies.clear()
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    record, assets = audio_reference.compare(earlier, analyzed)
    assert payload["comparison"] == record
    assert {name: base64.b64decode(body, validate=True) for name, body in payload["assets"].items()} == assets
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as report:
        assert len(report) == 4
        assert report.metadata["title"] == "Earlier audio comparison"
        assert report.metadata["author"] == "Chipmunk Verify"
        assert all(record[side]["sha256"] in report[0].get_text() for side in ("earlier", "submitted"))
        assert "2,882 of 32,031 mono samples differ across 3 ranges." in report[0].get_text()
        for index, span in enumerate(record["audio"]["ranges"], 1):
            evidence = report[index].get_text()
            assert f"{span['start_s']:.3f} to {span['end_s']:.3f} s" in evidence
            assert f"[{span['start_sample']}, {span['end_sample']})" in evidence
            assert "Difference waveform" in evidence
            assert len(report[index].get_drawings()) >= 8
        assert "peak 1.999969" in report[1].get_text()
        assert "peak 0.000031" in report[2].get_text()

    exported = tmp_path / "comparison-download.json"
    exported.write_text(json.dumps({key: payload[key] for key in ("comparison", "assets")}))
    assert run(earlier, analyzed, exported, verify=True) == record
    assert state() == initial
    with sessions() as db:
        db.get(Analysis, "other-kind").withdrawn_at = utcnow()
        db.commit()
    assert f.client.get(endpoint, headers=f.viewer).json() == {"available": False}
    assert f.client.post(endpoint, headers=f.viewer, files=files).status_code == 404
