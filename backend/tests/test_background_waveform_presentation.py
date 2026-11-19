"""Presentation-only checks for frozen original and processed waveform plates.

The source proof and detector binding are exercised in test_background_voice_review.
Here the verified boundary is supplied with authored chart assets to isolate layout.
"""
from __future__ import annotations

import copy
import hashlib
import io

import numpy as np
import pymupdf
import pytest
from PIL import Image

from app.core.canonical import canonical_hash
from app.evidence.audio_packets import _chart
from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen


def _document(version=2, status="flagged"):
    native = np.sin(np.arange(48_000) * .04).astype(np.float32) * .6
    processed = native * .25
    images = [_chart(values, [], start=0, end=48_000, selected=(0, 48_000), scores=False)
              for values in (native, processed)]
    assets = {hashlib.sha256(body).hexdigest(): body for body in images}
    waveforms = [{"sha256": hashlib.sha256(body).hexdigest(), "media_type": "image/png", "width": 960, "height": 280}
                 for body in images]
    interval = {"start_sample": 0, "end_sample": 48_000}
    review = {
        "schema": f"chipmunk.background-reduced-voice-review/{version}",
        "title": "Speech review with reduced background sound", "status": status,
        "reason": {"flagged": "The speech response leaned AI-generated after background sound was reduced.",
                   "unassessed": "The processed copy did not contain enough detected speech for this review.",
                   "no_flag": "The speech response did not lean clearly toward AI generation after background sound was reduced.",
                   "skipped": "The optional background-reduced speech review could not complete."}[status],
        "limitation": "This review carries no weight in the overall assessment.", "affects_overall_confidence": False,
        "interval": interval, "source_timestamp_ranges": [{**interval, "source_start_pts": 0, "source_end_pts": 48_000, "time_base": "1/16000"}],
        "artifacts": {},
    }
    if status == "flagged":
        review["artifacts"] = {"waveform": waveforms[0],
                               "selected_clip": {"sha256": "d" * 64, "media_type": "audio/wav"},
                               "processed_clip": {"sha256": "e" * 64, "media_type": "audio/wav"}}
        if version == 2:
            review["artifacts"]["processed_waveform"] = waveforms[1]
    document = {
        "schema": "chipmunk.audio-evidence/1", "version": "audio-evidence-3", "status": "ready", "id": "authored-view",
        "sha256": "a" * 64, "source": {"sha256": "b" * 64, "samples": 48_000, "sample_rate": 16_000, "channels": 1,
                                      "source_channels": 1, "source_sample_rate": 16_000},
        "coverage": {"decoded_samples": 48_000, "scored_samples": 0, "unscored_samples": 48_000, "windows": [],
                     "decode_limit_reached": False},
        "packets": [], "omitted_windows": 0, "limitations": [], "background_reduced_voice_review": review,
    }
    return document, assets


def _snapshot(monkeypatch, *, version=2, status="flagged", profile="standard", redacted=False):
    document, assets = _document(version, status)
    snapshot = specimen("audio", profile)
    document["original_result_hash"] = canonical_hash(snapshot.result)
    snapshot = snapshot.model_copy(update={"evidence": document,
        "report": snapshot.report.model_copy(update={"redacted": redacted})})
    monkeypatch.setattr(multimedia_evidence, "verified_set", lambda _: copy.deepcopy(document))
    monkeypatch.setattr(multimedia_evidence, "verify_assets", lambda *args, **kwargs: assets)
    return snapshot, document, assets


@pytest.mark.parametrize("version", [1, 2])
def test_waveform_view_preserves_legacy_plate_and_source_labels(monkeypatch, version):
    snapshot, original, _ = _snapshot(monkeypatch, version=version)
    before = copy.deepcopy(original)
    view = multimedia_evidence.view(snapshot)
    review = view["background_reduced_voice_review"]
    assert review["waveform"]["uri"].startswith("data:image/png;base64,")
    assert ("processed_waveform" in review) is (version == 2)
    if version == 2:
        assert review["processed_waveform"]["uri"].startswith("data:image/png;base64,")
        assert "source 0.000-3.000 s" in review["waveform"]["label"]
        assert "source 0.000-3.000 s" in review["processed_waveform"]["label"]
        assert "same digital amplitude scale" in review["processed_waveform"]["label"]
    else:
        assert review["waveform"]["label"] == "Original waveform across the flagged three-second interval"
    assert original == before


@pytest.mark.parametrize(("profile", "redacted"), [("newsroom", False), ("standard", True)])
def test_withheld_reports_never_embed_either_waveform(monkeypatch, profile, redacted):
    snapshot, _, _ = _snapshot(monkeypatch, profile=profile, redacted=redacted)
    view = multimedia_evidence.view(snapshot)
    review = view["background_reduced_voice_review"]
    assert view["withheld"] is True
    assert review["waveform"]["uri"] is None and review["processed_waveform"]["uri"] is None
    html = engine.render_html(snapshot, embed_fonts=False)
    assert "data:image/png;base64," not in html
    assert "Both waveforms cover the same" not in html
    assert "cannot identify which speaker drove the result" in html


@pytest.mark.parametrize("status", ["no_flag", "unassessed", "skipped"])
def test_unassessed_or_clear_reviews_do_not_display_a_waveform_pair(monkeypatch, status):
    snapshot, document, _ = _snapshot(monkeypatch, status=status)
    view = multimedia_evidence.view(snapshot)
    assert "waveform" not in view["background_reduced_voice_review"]
    assert "processed_waveform" not in view["background_reduced_voice_review"]
    html = engine.render_html(snapshot, embed_fonts=False)
    assert document["background_reduced_voice_review"]["reason"] in html
    assert "Both waveforms cover the same" not in html


def test_standard_pdf_keeps_both_actual_waveform_pixels_and_clock_on_one_page(monkeypatch, tmp_path):
    snapshot, document, assets = _snapshot(monkeypatch)
    rendered = engine.render_pdf(snapshot)
    (tmp_path / "paired-waveforms.pdf").write_bytes(rendered.pdf)

    def pixels(data):
        with Image.open(io.BytesIO(data)) as image:
            rgb = image.convert("RGB")
            return rgb.size, hashlib.sha256(rgb.tobytes()).hexdigest()

    wanted = {pixels(body) for body in assets.values()}
    with pymupdf.open(stream=rendered.pdf, filetype="pdf") as pdf:
        witnessed = []
        for page in pdf:
            text = " ".join(page.get_text().split())
            found = {pixels(pdf.extract_image(image["xref"])["image"])
                     for image in page.get_image_info(xrefs=True) if image["xref"]}
            if wanted <= found:
                witnessed.append(page.number)
                assert "source 0.000-3.000 s" in text
                assert "fixed digital amplitude scale" in text and "same digital amplitude scale" in text
                assert "Processing can change the sound of overlapping voices." in text
                page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(tmp_path / "paired-waveforms.png")
        assert len(witnessed) == 1
    assert document["background_reduced_voice_review"]["affects_overall_confidence"] is False


@pytest.mark.parametrize("version", [1, 2])
def test_flagged_background_review_has_no_false_empty_state(monkeypatch, version):
    snapshot, document, _ = _snapshot(monkeypatch, version=version)
    assert document["packets"] == []
    pdf = engine.render_pdf(snapshot).pdf
    text = " ".join(engine.pdf_text(pdf).split())
    assert "Speech review with reduced background sound" in text
    assert "The speech response leaned AI-generated after background sound was reduced." in text
    assert "No usable local candidates were recorded in this evidence set." not in text
