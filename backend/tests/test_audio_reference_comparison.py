import base64
import hashlib
import json
import shutil
import subprocess
import wave
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
from app.evidence import video_reference_soundtrack as reference_timing
from app.reports.media_reference_pdf import render as render_comparison
from app.storage.db import Analysis, Artefact, BillingOrderRow, Job, utcnow
from scripts.compare_audio_references import run
from scripts.verify_audio_clean_speech_export import verify_export as verify_clean_speech
from scripts.verify_audio_local_timing_export import verify_export as verify_timing
from scripts.verify_stereo_reference import verify as verify_stereo

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


def save(folder, name, samples):
    path = folder / f"{name}.wav"
    path.write_bytes(wav_bytes(samples))
    return path


def save_stereo(folder, name, samples):
    path = folder / f"{name}.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, 16000, 0, "NONE", ""))
        output.writeframes(samples.astype("<i2", copy=False).tobytes())
    return path


def test_stereo_comparison_preserves_changes_hidden_by_downmix_and_exact_tail(tmp_path):
    rng = np.random.default_rng(451)
    before = rng.integers(-4000, 4001, (20123, 2), dtype=np.int16) * 2
    after = before.copy()
    after[12801:13511] += [160, -160]
    after[-13:] += [-120, 120]
    earlier = save_stereo(tmp_path, "earlier-stereo", before)
    analyzed = save_stereo(tmp_path, "analyzed-stereo", after)
    record, assets = audio_reference.compare(earlier, analyzed)
    assert record["schema"] == audio_reference.SCHEMA_WITH_STEREO
    assert record["status"] == "measured" and record["audio"]["changed_samples"] == 0
    result = record["stereo_comparison"]
    assert result["status"] == "measured" and result["affects_overall_confidence"] is False
    for side, samples in (("earlier", before), ("submitted", after)):
        assert result["sources"][side]["stereo_pcm_sha256"] == hashlib.sha256(samples.astype("<i2").tobytes()).hexdigest()
    for index, label in enumerate(("left", "right")):
        channel = result["channels"][label]
        assert channel["changed_samples"] == 723
        for idx, (span, bounds) in enumerate(zip(channel["ranges"], ((12800, 13760), (19840, 20123)), strict=True), 1):
            start, end = bounds
            assert span["id"] == f"{label[0].upper()}{idx:02d}"
            assert (span["start_sample"], span["end_sample"]) == bounds
            assert (span["first_differing_sample"], span["last_differing_sample"]) == ((12801, 13510), (20110, 20122))[idx - 1]
            assert "difference_waveform" in span["preview"]
            for side, samples in (("earlier", before), ("submitted", after)):
                assert span["source_ranges"][side] == {"start_pts": start, "end_exclusive_pts": end}
                wav = assets[span["preview"]["assets"][side]["file"]]
                np.testing.assert_array_equal(read_wav(wav), samples[start:end, index] / 32768)
    with pymupdf.open(stream=render_comparison(record, assets), filetype="pdf") as document:
        report = " ".join(page.get_text() for page in document)
        assert all(text in report for text in ("Left and right channel comparison", "L01", "R02", "Difference waveform",
                                               "Source ticks [19840, 20123)"))
    output = tmp_path / "stereo-replay"
    assert run(earlier, analyzed, output) == run(earlier, analyzed, output, verify=True) == record
    browser_export = tmp_path / "browser-export.json"
    browser_export.write_text(json.dumps({"comparison": record, "assets": {
        name: base64.b64encode(raw).decode() for name, raw in assets.items()}}))
    assert verify_stereo(earlier, analyzed, browser_export) == {
        channel: {"differing_samples": 723, "ranges": 2, "previews_verified": 4} for channel in ("left", "right")}
    tampered = json.loads(browser_export.read_bytes())
    tampered["comparison"]["stereo_comparison"]["channels"]["left"]["ranges"][0]["last_differing_sample"] += 1
    browser_export.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="Measured source range differs"):
        verify_stereo(earlier, analyzed, browser_export)
    legacy, _ = audio_reference.compare(earlier, analyzed, include_stereo=False)
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"comparison": legacy, "assets": {}}))
    assert run(earlier, analyzed, historical, verify=True) == legacy

    shortened = save_stereo(tmp_path, "shortened-stereo", after[:-23])
    mismatched, _ = audio_reference.compare(earlier, shortened)
    assert mismatched["stereo_comparison"]["status"] == "audio_timelines_differ"
    mono = save(tmp_path, "mono", before.mean(axis=1) / 32768)
    mixed, _ = audio_reference.compare(earlier, mono)
    assert mixed["stereo_comparison"]["status"] == "stereo_pair_required"


def test_local_timing_evidence_replays_offsets_excerpts_and_pdf(tmp_path):
    rng = np.random.default_rng(231)
    original = rng.integers(-6000, 6000, 48000, dtype=np.int16).astype(np.float32) / 32768
    shifted = original.copy()
    shifted[16000:26240] = original[12800:23040]
    earlier = save(tmp_path, "timing-earlier", original)
    analyzed = save(tmp_path, "timing-analyzed", shifted)
    record, assets = audio_reference.compare(earlier, analyzed, include_local_timing=True)
    assert record["schema"] == audio_reference.SCHEMA_WITH_LOCAL_TIMING
    timing = record["local_timing"]
    assert timing["status"] == "local_shift" and timing["offset_ms"] == 200
    assert timing["affects_overall_confidence"] is False
    span = timing["range"]
    assert span["earlier"]["samples"] == span["earlier"]["source_pts"] == [12800, 23040]
    assert span["submitted"]["samples"] == span["submitted"]["source_pts"] == [16000, 26240]
    for side in ("earlier", "submitted"):
        np.testing.assert_array_equal(read_wav(assets[span[side]["audio"]["file"]]), original[12800:23040])
    with pymupdf.open(stream=render_comparison(record, assets), filetype="pdf") as report:
        timing_page = report[-1].get_text()
        assert "Local audio timing" in timing_page
        assert "0.800 to 1.440 s" in timing_page and "1.000 to 1.640 s" in timing_page

    exported = tmp_path / "comparison.json"
    package = {"comparison": record, "assets": {name: base64.b64encode(raw).decode() for name, raw in assets.items()}}
    exported.write_text(json.dumps(package))
    assert run(earlier, analyzed, exported, verify=True) == record
    assert verify_timing(earlier, analyzed, exported) == {
        "offset_ms": 200, "earlier_samples": [12800, 23040], "submitted_samples": [16000, 26240], "excerpts_verified": 2,
    }
    package["comparison"]["local_timing"]["range"]["submitted"]["samples"][0] += 1
    exported.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="differs from the re-decoded sources"):
        run(earlier, analyzed, exported, verify=True)
    with pytest.raises(ValueError, match="Independent signed offset or range differs|Exported source clock differs"):
        verify_timing(earlier, analyzed, exported)
    package["comparison"]["local_timing"]["range"]["submitted"]["samples"][0] -= 1
    package["comparison"]["local_timing"]["range"]["submitted"]["waveform"]["low"][0] = 7
    exported.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="Source PCM or waveform differs"):
        verify_timing(earlier, analyzed, exported)

    globally_shifted = save(tmp_path, "timing-globally-shifted", np.r_[np.zeros(3200), original[:-3200]])
    silence = save(tmp_path, "timing-silence", np.zeros_like(original))
    for alternative, outcome in ((earlier, "no_local_shift"), (globally_shifted, "no_local_shift"), (silence, "unassessed")):
        compared, _ = audio_reference.compare(earlier, alternative, include_local_timing=True)
        assert compared["local_timing"]["status"] == outcome
        assert "range" not in compared["local_timing"]
    wrapped = tmp_path / "timing-ogg-flac.ogg"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(analyzed), "-c:a", "flac", "-f", "ogg", str(wrapped)],
                   capture_output=True, check=True, timeout=20)
    unsupported, _ = audio_reference.compare(earlier, wrapped, include_local_timing=True)
    assert unsupported["local_timing"]["status"] == "unassessed"
    assert "range" not in unsupported["local_timing"]
    assert "local_timing" not in audio_reference.compare(earlier, analyzed)[0]


def test_aac_timing_uses_presentation_duration_and_replays_source_excerpts(tmp_path, monkeypatch):
    original = np.random.default_rng(34).integers(-9000, 9000, 48000).astype(np.float32) / 32768
    shifted = original.copy()
    shifted[16000:26240] = original[12800:23040]
    earlier = save(tmp_path, "before", original)
    wav = save(tmp_path, "shift", shifted)
    aac = tmp_path / "submitted.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "aac", "-b:a", "48k", str(aac)],
                   check=True, timeout=20, capture_output=True)
    inactive, _ = audio_reference.compare(earlier, aac, include_local_timing=True)
    assert inactive["local_timing"]["status"] == "unassessed"

    result, assets = audio_reference.compare(earlier, aac, include_local_timing=True, include_aac_timing=True)
    timing = result["local_timing"]
    assert result["schema"] == audio_reference.SCHEMA_WITH_AAC_TIMING
    assert result["audio"] == inactive["audio"]
    assert timing["status"] == "local_shift" and timing["offset_ms"] == 200
    assert timing["affects_overall_confidence"] is False
    assert timing["aac_presentation"] == {
        "side": "submitted", "sample_rate": 16000, "time_base": "1/16000",
        "presentation_start_pts": 0, "presentation_end_exclusive_pts": 48000,
        "decoded_samples": 48128, "excluded_trailing_samples": 128,
    }
    native = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(aac), "-map", "0:a:0",
                                      "-c:a", "pcm_s16le", "-f", "s16le", "-"], timeout=20)
    span = timing["range"]["submitted"]
    start, end = span["samples"]
    assert span["source_pts"] == [start, end] == [16000, 26240]
    assert hashlib.sha256(native[2*start:2*end]).hexdigest() == span["pcm_sha256"]
    assert read_wav(assets[span["audio"]["file"]]).tobytes() == (
        np.frombuffer(native[2*start:2*end], dtype="<i2").astype(np.float32) / 32768
    ).tobytes()
    with pymupdf.open(stream=render_comparison(result, assets), filetype="pdf") as report:
        assert "128 trailing decoded samples excluded" in report[-1].get_text()
        assert "1.000 to 1.640 s" in report[-1].get_text()

    output = tmp_path / "aac-comparison.json"
    package = {"comparison": result, "assets": {name: base64.b64encode(raw).decode() for name, raw in assets.items()}}
    output.write_text(json.dumps(package))
    assert run(earlier, aac, output, verify=True) == result
    assert verify_timing(earlier, aac, output)["excerpts_verified"] == 2
    timing["aac_presentation"]["excluded_trailing_samples"] += 1
    output.write_text(json.dumps(package))
    with pytest.raises(ValueError, match="differs from the re-decoded sources"):
        run(earlier, aac, output, verify=True)
    with pytest.raises(ValueError, match="AAC presentation proof differs"):
        verify_timing(earlier, aac, output)
    reverse, _ = audio_reference.compare(aac, earlier, include_local_timing=True, include_aac_timing=True)
    assert reverse["local_timing"]["aac_presentation"]["side"] == "earlier"

    short = save(tmp_path, "mismatched-extent", original[:-1])
    mismatch, _ = audio_reference.compare(short, aac, include_local_timing=True, include_aac_timing=True)
    assert mismatch["local_timing"]["status"] == "unassessed"
    assert "aac_presentation" not in mismatch["local_timing"]

    monkeypatch.setattr(audio_reference, "aac_native_extent", lambda _: None)
    unavailable, _ = audio_reference.compare(earlier, aac, include_local_timing=True, include_aac_timing=True)
    assert unavailable == inactive


def test_local_channel_timing_replays_opposite_stereo_moves(tmp_path):
    generator = np.random.default_rng(665)
    first = generator.integers(-10000, 10000, (52800, 2), dtype=np.int16)
    last = first.copy()
    start, stop = 14417, 21619
    last[start:stop, 0] = first[start - 3200:stop - 3200, 0]
    last[start:stop, 1] = first[start + 2727:stop + 2727, 1]
    earlier = save_stereo(tmp_path, "original", first)
    analyzed = save_stereo(tmp_path, "edited", last)
    result, assets = audio_reference.compare(earlier, analyzed, include_local_timing=True, include_channel_timing=True)
    assert result["schema"] == audio_reference.SCHEMA_WITH_CHANNEL_TIMING
    timing = result["channel_timing"]
    assert timing["status"] == "local_shift" and timing["affects_overall_confidence"] is False
    for index, (channel, shift) in enumerate(zip(timing["ranges"], (3200, -2727), strict=True)):
        span = channel["range"]
        assert channel["channel"] == ("left", "right")[index]
        assert channel["offset_ms"] == shift / 16
        assert span["changed_samples"] == span["submitted_samples"] == [start, stop]
        assert span["earlier_samples"] == [start - shift, stop - shift]
        for key, signal in (("earlier", first), ("submitted", last)):
            a, b = span[key]["samples"]
            assert span[key]["source_pts"] == [a, b]
            np.testing.assert_array_equal(read_wav(assets[span[key]["audio"]["file"]]), signal[a:b, index] / 32768)
    with pymupdf.open(stream=render_comparison(result, assets), filetype="pdf") as document:
        assert "Left channel timing" in document[-2].get_text()
        assert "Right channel timing" in document[-1].get_text()
        assert "0.901 to 1.351 s" in document[-1].get_text()
    exported = tmp_path / "channel-evidence.json"
    exported.write_text(json.dumps({"comparison": result, "assets": {
        name: base64.b64encode(raw).decode() for name, raw in assets.items()}}))
    assert run(earlier, analyzed, exported, verify=True) == result
    verified = verify_stereo(earlier, analyzed, exported)
    assert verified["left"]["differing_samples"] == int(np.count_nonzero(first[:, 0] != last[:, 0]))
    assert verified["right"]["local_timing"] == {"signed_shift_samples": -2727, "excerpts_verified": 2}
    unchanged, _ = audio_reference.compare(earlier, earlier, include_local_timing=True, include_channel_timing=True)
    assert unchanged["channel_timing"]["status"] == "no_local_shift"
    assert unchanged["channel_timing"]["ranges"] == []


@pytest.mark.parametrize("span", [
    {"earlier_samples": [-4, 4476], "submitted_samples": [3200, 7680], "shift_samples": 3204},
    {"earlier_samples": [42000, 46480], "submitted_samples": [45200, 49680], "shift_samples": 3200},
])
def test_local_timing_rejects_a_match_outside_either_sample_clock(tmp_path, monkeypatch, span):
    samples = np.ones(48000, dtype=np.float32) * .1
    earlier = save(tmp_path, "timing-clock-earlier", samples)
    submitted = save(tmp_path, "timing-clock-submitted", samples)
    monkeypatch.setattr(reference_timing, "measure", lambda *_: (span, 12))
    record, assets = audio_reference.compare(earlier, submitted, include_local_timing=True)
    assert record["local_timing"]["status"] == "unassessed"
    assert "range" not in record["local_timing"] and not assets


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
    result, assets = audio_reference.compare(earlier, trimmed, include_trim_alignment=False)
    assert result["status"] == result["audio"]["status"] == "audio_timelines_differ"
    assert result["audio"]["earlier"]["samples"] == len(pcm)
    assert result["audio"]["submitted"]["samples"] == len(pcm) - 1
    assert "ranges" not in result["audio"] and not assets


@pytest.mark.parametrize("reversed_pair", [False, True])
def test_audio_reference_compares_overlap_after_a_small_trim(tmp_path, reversed_pair):
    rng = np.random.default_rng(394)
    original = rng.integers(-5000, 5000, 52000, dtype=np.int16)
    trimmed = original[331:-133].copy()
    trimmed[22002:22802] += 500
    before, after = (pcm.astype(np.float32) / 32768 for pcm in (original, trimmed))
    earlier, submitted = save(tmp_path, "original", before), save(tmp_path, "trimmed", after)
    if reversed_pair:
        earlier, submitted, before, after = submitted, earlier, after, before
    record, assets = audio_reference.compare(earlier, submitted)
    audio = record["audio"]
    alignment = audio["trim_alignment"]
    assert record["schema"] == audio_reference.SCHEMA_WITH_TRIM_ALIGNMENT
    assert record["affects_overall_confidence"] is False
    assert alignment["earlier_minus_submitted_samples"] == (-331 if reversed_pair else 331)
    assert alignment["compared_samples"] == len(trimmed)
    for side, start in zip(("earlier", "submitted"), (0, 331) if reversed_pair else (331, 0), strict=True):
        assert alignment["overlap"][side] == {
            "start_sample": start, "end_sample": start + len(trimmed),
            "start_pts": start, "end_exclusive_pts": start + len(trimmed),
        }
    for support in alignment["support"]:
        for side, native in (("earlier", before), ("submitted", after)):
            a, b = (support[side][key] for key in ("start_sample", "end_sample"))
            assert hashlib.sha256((native[a:b] * 32768).astype("<i2").tobytes()).hexdigest() == support["pcm_sha256"]
    assert audio["changed_samples"] == 800
    span, = audio["ranges"]
    assert span["start_s"] == span["file_ranges"]["submitted"]["start_sample"] / 16000
    for side, native in (("earlier", before), ("submitted", after)):
        local = span["file_ranges"][side]
        samples = read_wav(assets[span["preview"]["assets"][side]["file"]])
        np.testing.assert_array_equal(samples, native[local["start_sample"]:local["end_sample"]])
    with pymupdf.open(stream=render_comparison(record, assets), filetype="pdf") as report:
        assert "Shared audio after sample alignment" in report[0].get_text()
        assert "800" in report[0].get_text()
        assert "Waveform covers" in report[1].get_text()
        assert "Difference waveform" in report[1].get_text()


def test_audio_reference_trim_abstains_on_ambiguous_or_inconsistent_support(tmp_path):
    rng = np.random.default_rng(418)
    pcm = rng.integers(-5000, 5000, 52000).astype(np.float32) / 32768
    periodic = np.tile(pcm[:100], 520)
    for name, left, right in (
        ("periodic", periodic, periodic[331:]),
        ("internal-removal", pcm, np.r_[pcm[:25000], pcm[25331:]]),
        ("large-trim", pcm, pcm[4100:]),
    ):
        record, assets = audio_reference.compare(save(tmp_path, name + "-1", left), save(tmp_path, name + "-2", right))
        assert record["schema"] == audio_reference.SCHEMA
        assert record["status"] == "audio_timelines_differ" and not assets


def test_audio_reference_trim_replays_new_and_historical_exports(tmp_path, aligned_audio_pair):
    earlier, _, pcm, _ = aligned_audio_pair
    submitted = save(tmp_path, "trimmed-copy", pcm[20:])
    for name, use_alignment in (("current", True), ("historical", False)):
        record, assets = audio_reference.compare(earlier, submitted, include_trim_alignment=use_alignment)
        assert record["status"] == ("measured" if use_alignment else "audio_timelines_differ")
        assert not assets
        exported = tmp_path / f"{name}.json"
        exported.write_text(json.dumps({"comparison": record, "assets": {}}))
        assert run(earlier, submitted, exported, verify=True) == record


@pytest.mark.parametrize("gain", [0.625, 1.25])
def test_audio_reference_combines_trim_and_volume_with_both_clocks(tmp_path, gain):
    original = np.random.default_rng(985).integers(-8000, 8000, 72000, dtype=np.int16)
    trimmed = np.rint(original[331:-133] * gain).astype(np.int16)
    earlier = save(tmp_path, "earlier", original.astype(np.float32) / 32768)
    for edited in (False, True):
        native = trimmed.copy()
        if edited:
            native[32000:32640] += 800
        submitted = save(tmp_path, f"submitted-{edited}", native.astype(np.float32) / 32768)
        previous, _ = audio_reference.compare(earlier, submitted, include_trim_level_adjustment=False)
        assert previous["status"] == "audio_timelines_differ"
        record, assets = audio_reference.compare(earlier, submitted)
        audio, adjusted = record["audio"], record["level_adjusted"]
        assert record["schema"] == audio_reference.SCHEMA_WITH_TRIM_AND_LEVEL
        assert audio["trim_alignment"]["earlier_minus_submitted_samples"] == 331
        assert adjusted["changed_samples"] == (640 if edited else 0)
        assert record["affects_overall_confidence"] is False
        assert all(Fraction(window["gain_bounds"]["lower"]) <= gain <= Fraction(window["gain_bounds"]["upper"])
                   for window in adjusted["support"])
        if edited:
            span, = adjusted["ranges"]
            assert span["start_sample"] == 32000
            assert span["file_ranges"]["earlier"]["start_sample"] == 32331
            assert span["file_ranges"]["submitted"]["start_sample"] == 32000
            np.testing.assert_array_equal(read_wav(assets[span["preview"]["assets"]["earlier"]["file"]]) * 32768,
                                          original[32331:32971])
            with pymupdf.open(stream=render_comparison(record, assets), filetype="pdf") as report:
                assert "Shared audio after sample alignment" in report[0].get_text()
                assert "two supporting sections" in report[1].get_text()
                assert "Volume-adjusted difference waveform" in report[2].get_text()
        folder = tmp_path / f"receipt-{edited}"
        assert run(earlier, submitted, folder) == record
        assert run(earlier, submitted, folder, verify=True) == record


def test_audio_reference_trim_and_volume_abstains_on_different_gain_or_clock(tmp_path):
    original = np.random.default_rng(437).integers(-8000, 8000, 72000, dtype=np.int16)
    earlier = save(tmp_path, "earlier", original.astype(np.float32) / 32768)
    partial = original[331:-133]
    for name, samples in (
        ("internal-trim", np.r_[original[:36000], original[36464:]] * 0.8),
        ("varying-gain", np.r_[partial[:36000] * 0.8, partial[36000:] * 0.9]),
        ("periodic", np.resize(partial[:111], len(partial)) * 0.8),
    ):
        submitted = save(tmp_path, name, np.rint(samples).astype(np.float32) / 32768)
        record, assets = audio_reference.compare(earlier, submitted)
        assert record["status"] == "audio_timelines_differ" and not assets


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
        assert record["status"] == "measured"
        if variant == stereo:
            assert record["schema"] == audio_reference.SCHEMA_WITH_STEREO
            assert record["stereo_comparison"]["status"] == "stereo_pair_required"
        else:
            assert record["schema"] == audio_reference.SCHEMA
        assert "level_adjusted" not in record and record["audio"]["changed_samples"] > 0


def test_authenticated_audio_comparison_pdf_json_and_temporary_storage(
        reference_http, aligned_audio_pair, monkeypatch, tmp_path):
    f = reference_http
    earlier, analyzed, _, _ = aligned_audio_pair
    stored = f.store.put_file(analyzed)
    monkeypatch.setattr(settings, "audio_reference_comparison_enabled", True)
    monkeypatch.setattr(settings, "audio_reference_local_timing", False)
    monkeypatch.setattr(settings, "audio_reference_clean_speech", False)

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

    with monkeypatch.context() as config:
        config.setattr(settings, "audio_reference_local_timing", True)
        optional = f.client.post(endpoint, headers=f.viewer, files=files, data={"include_pdf": "true"})
        assert optional.status_code == 200, optional.text
        assert optional.json()["comparison"]["local_timing"]["status"] == "no_local_shift"
        with pymupdf.open(stream=base64.b64decode(optional.json()["pdf"], validate=True), filetype="pdf") as report:
            assert "Local audio timing" in report[-1].get_text()

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
    trimmed_samples = aligned_audio_pair[3][13:-11].copy()
    trimmed_samples[11001:11201] += 1 / 32768
    trimmed = save(tmp_path, "earlier-with-small-trim", trimmed_samples)
    files_after_trim = {"earlier": ("earlier.wav", trimmed.read_bytes(), "audio/wav")}
    response = f.client.post(endpoint, headers=f.viewer, files=files_after_trim, data={"include_pdf": "true"})
    assert response.status_code == 200
    trimmed_payload = response.json()
    trimmed_record = trimmed_payload["comparison"]
    assert trimmed_record["schema"] == audio_reference.SCHEMA_WITH_TRIM_ALIGNMENT
    assert trimmed_record["audio"]["trim_alignment"]["earlier_minus_submitted_samples"] == -13
    assert trimmed_record["audio"]["changed_samples"] == 200
    with pymupdf.open(stream=base64.b64decode(trimmed_payload["pdf"], validate=True), filetype="pdf") as report:
        assert "Shared audio after sample alignment" in report[0].get_text()
        assert len(report[1].get_drawings()) >= 8
    browser_json = json.dumps({key: trimmed_payload[key] for key in ("comparison", "assets")})
    browser_json = browser_json.replace("0.0,", "0,").replace("0.0]", "0]")
    exported.write_text(browser_json)
    assert run(trimmed, analyzed, exported, verify=True) == trimmed_record
    exported.write_text(browser_json.replace('"affects_overall_confidence": false', '"affects_overall_confidence": 0'))
    with pytest.raises(ValueError, match="differs"):
        run(trimmed, analyzed, exported, verify=True)
    assert state() == initial
    with sessions() as db:
        db.get(Analysis, "other-kind").withdrawn_at = utcnow()
        db.commit()
    assert f.client.get(endpoint, headers=f.viewer).json() == {"available": False}
    assert f.client.post(endpoint, headers=f.viewer, files=files).status_code == 404


@pytest.fixture
def clean_speech_pair(tmp_path):
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(87142)
    shape = butter(2, (260, 3600), fs=16000, btype="bandpass", output="sos")
    sources = [sosfiltfilt(shape, rng.standard_normal(96000)) * scale for scale in (1800, 1700, 2400)]
    first, donor, background = sources
    altered = first.copy()
    altered[32000:43520] = donor[32000:43520]
    return {
        name: save(tmp_path, name, (np.rint(pcm) / 32768).astype(np.float32))
        for name, pcm in (("earlier", first), ("altered", altered + background), ("control", first + background),
                          ("other", donor), ("shorter", first[:-1]))
    }


def test_clean_speech_lane_localizes_and_falls_back_with_source_clips(tmp_path, clean_speech_pair):
    first = clean_speech_pair["earlier"]
    output = tmp_path / "clean-speech.json"
    for kind, reason in (("altered", "local_change"), ("control", "no_local_change"),
                         ("shorter", "six_second_common_clock_required"), ("other", "reference_match_not_supported")):
        second = clean_speech_pair[kind]
        record, assets = audio_reference.compare(first, second, include_clean_speech=True, include_local_timing=True,
                                                 allow_long_clean_speech=False)
        standard, standard_assets = audio_reference.compare(first, second, include_local_timing=True)
        assert record["schema"] == audio_reference.SCHEMA_WITH_CLEAN_SPEECH
        result = record["clean_speech"]
        assert result["reason"] == reason
        assert record["affects_overall_confidence"] is result["affects_overall_confidence"] is False
        assert {k: v for k, v in record.items() if k not in {"schema", "clean_speech"}} == {
            k: v for k, v in standard.items() if k != "schema"
        }
        assert all(assets[name] == blob for name, blob in standard_assets.items())
        payload = {"comparison": record, "assets": {name: base64.b64encode(blob).decode() for name, blob in assets.items()}}
        output.write_text(json.dumps(payload))
        assert run(first, second, output, verify=True) == record
        if kind == "altered":
            span = result["range"]
            assert span["samples"] == [32000, 43520]
            assert verify_clean_speech(first, second, output)["excerpts_verified"] == 2
            span["earlier"]["source_pts"][0] += 1
            output.write_text(json.dumps(payload))
            with pytest.raises(ValueError, match="source clock differs"):
                verify_clean_speech(first, second, output)
        else:
            assert "range" not in result and assets == standard_assets
            if kind != "shorter":
                assert verify_clean_speech(first, second, output)["excerpts_verified"] == 0

    converted = tmp_path / "earlier.flac"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(first), str(converted)], check=True)
    record, _ = audio_reference.compare(converted, clean_speech_pair["altered"], include_clean_speech=True)
    assert record["clean_speech"]["reason"] == "local_change"
    lossy = tmp_path / "earlier.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(first), str(lossy)], check=True)
    record, _ = audio_reference.compare(lossy, clean_speech_pair["altered"], include_clean_speech=True)
    assert record["clean_speech"]["reason"] == "native_format_required"


@pytest.mark.parametrize("dense", [False, True])
def test_clean_speech_lane_api_is_opt_in_and_renders_source_ranges(reference_http, monkeypatch, tmp_path, clean_speech_pair, dense):
    f = reference_http

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_media_reference, "get_session", sessions)
    monkeypatch.setattr(v2_media_reference, "default_store", lambda: f.store)
    monkeypatch.setattr(settings, "audio_reference_comparison_enabled", True)
    monkeypatch.setattr(settings, "audio_reference_clean_speech", True)
    monkeypatch.setattr(settings, "audio_reference_long_clean_speech", False)
    monkeypatch.setattr(settings, "audio_reference_dense_clean_speech", dense)
    first, second = (clean_speech_pair[name] for name in ("earlier", "altered"))
    stored = f.store.put_file(second)
    with sessions() as db:
        row = db.get(Analysis, "other-kind")
        row.kind, row.filename, row.content_type = "audio", second.name, "audio/wav"
        row.sha256, row.size_bytes = stored.sha256, stored.size_bytes
        original_result = row.result_hash
        db.commit()
    endpoint = "/api/v2/analyses/other-kind/audio-reference-comparison"
    assert f.client.get(endpoint, headers=f.viewer).json() == {"available": True, "clean_speech_available": True}
    files = {"earlier": (first.name, first.read_bytes(), "audio/wav")}
    assert f.client.post(endpoint, files=files, data={"earlier_has_clean_speech": "true"}).status_code == 401
    standard = f.client.post(endpoint, headers=f.viewer, files=files).json()
    assert "clean_speech" not in standard["comparison"]
    response = f.client.post(endpoint, headers=f.viewer, files=files,
                             data={"include_pdf": "true", "earlier_has_clean_speech": "true"})
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    payload = response.json()
    speech = payload["comparison"]["clean_speech"]
    span = speech["range"]
    assert payload["comparison"]["schema"] == (
        audio_reference.SCHEMA_WITH_DENSE_CLEAN_SPEECH if dense else audio_reference.SCHEMA_WITH_CLEAN_SPEECH
    )
    assert span["samples"][1] - span["samples"][0] == (15360 if dense else 11520)
    assert ("basis" in speech) is dense
    with pymupdf.open(stream=base64.b64decode(payload["pdf"], validate=True), filetype="pdf") as report:
        page = report[1]
        copy = " ".join(page.get_text().split())
        assert "Local speech comparison" in copy
        assert "Ordinary edits" in copy and "zero weight" in copy
        start, end = (sample / 16000 for sample in span["samples"])
        assert f"{start:.3f} to {end:.3f} s" in copy
        assert ("Matching sound gain by band" in copy) is dense
        assert f"[{span['samples'][0]}, {span['samples'][1]})" in copy
        waveform_paths = [path for path in page.get_drawings()
                          if path.get("color") == pytest.approx((.18, .39, .31))]
        for side, center in (("earlier", 379), ("submitted", 565)):
            lines = [item for path in waveform_paths for item in path["items"]
                     if item[0] == "l" and abs(item[1].y - center) <= 45]
            assert len({round(item[1].x, 2) for item in lines}) == len(span[side]["waveform"]["t"])
    export = tmp_path / "local-speech-comparison.json"
    export.write_text(json.dumps({key: payload[key] for key in ("comparison", "assets")}))
    assert verify_clean_speech(first, second, export)["excerpts_verified"] == 2
    assert run(first, second, export, verify=True) == payload["comparison"]
    with sessions() as db:
        assert db.get(Analysis, "other-kind").result_hash == original_result


def test_clean_speech_twelve_seconds_distinguishes_late_change_and_broad_frequency_processing(tmp_path):
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(172)
    shape = butter(2, (260, 3600), fs=16000, btype="bandpass", output="sos")
    earlier, donor, background = [sosfiltfilt(shape, rng.standard_normal(192000)) * scale for scale in (2000, 1800, 1600)]
    replaced = earlier.copy()
    replaced[145920:157440] = donor[145920:157440]
    lowpass = sosfiltfilt(butter(2, 1900, fs=16000, btype="lowpass", output="sos"), earlier)
    path = save(tmp_path, "earlier-twelve", (np.rint(earlier) / 32768).astype(np.float32))
    output = tmp_path / "twelve-comparison.json"
    for name, waveform, expected in (
        ("late", replaced + background, "localized_difference"),
        ("broad", lowpass + background, "broad_processing_difference"),
        ("mismatched", donor + background, "unassessed"),
    ):
        other = save(tmp_path, name, (np.rint(waveform) / 32768).astype(np.float32))
        record, assets = audio_reference.compare(path, other, include_clean_speech=True, use_dense_clean_speech=True)
        assert record["schema"] == audio_reference.SCHEMA_WITH_LONG_CLEAN_SPEECH
        check = record["clean_speech"]
        assert check["status"] == expected and check["sample_count"] == 192000
        assert check["affects_overall_confidence"] is record["affects_overall_confidence"] is False
        output.write_text(json.dumps({"comparison": record,
                                      "assets": {key: base64.b64encode(raw).decode() for key, raw in assets.items()}}))
        assert run(path, other, output, verify=True) == record
        measured = verify_clean_speech(path, other, output)
        if expected == "localized_difference":
            assert measured["samples"] == [145920, 157440]
            assert measured["excerpts_verified"] == 2
            assert len(check["range"]["supporting_window_starts"]) == 2
        else:
            assert "range" not in check
            assert measured["excerpts_verified"] == 0
            if expected == "broad_processing_difference":
                assert check["frequency_balance"]["section_start_samples"] == [0, 64000, 128000]
                with pymupdf.open(stream=render_comparison(record, assets), filetype="pdf") as report:
                    assert "Frequency balance across the recording" in report[1].get_text()
                    assert "Selected interval" not in report[1].get_text()
