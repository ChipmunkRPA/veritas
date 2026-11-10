from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import pytest
from PIL import Image
from test_audio_evidence import audio_fixture as audio_fixture
from test_document_packets import packet_factory as packet_factory
from test_video_evidence_packets import vfr_source as vfr_source
from test_video_evidence_packets import video_fixture as video_fixture

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import audio_packets, document_packets, service, video_packets
from app.reports import engine, multimedia_evidence
from app.reports.public_view import public_manifest
from app.reports.snapshot import dump, load, snapshot_hash
from app.reports.specimens import specimen
from app.storage.objects import ObjectIntegrityError, ObjectNotFound


@pytest.fixture(params=["video", "audio", "document"])
def evidence_case(request):
    kind = request.param
    if kind == "video":
        db, store, analysis, result, _, _ = request.getfixturevalue("video_fixture")
        document = video_packets.build_packet_set(db, analysis, store, result)
    elif kind == "audio":
        db, store, analysis, result, _, _ = request.getfixturevalue("audio_fixture")
        document = audio_packets.build_packet_set(db, analysis, store, result)
    else:
        db, store, analysis, result, _, _ = request.getfixturevalue("packet_factory")(".pdf")
        document = document_packets.build_packet_set(db, analysis, store, result)
    # The report scaffolding is a labelled QA specimen; the packet/assets and
    # source bindings come from the real FFmpeg/PDF fixture builders above.
    snapshot = specimen(kind, "standard")
    report_result = copy.deepcopy(snapshot.result)
    report_result["input"]["sha256"] = analysis.sha256
    report_result["manifest"]["sampling"] = copy.deepcopy(result["manifest"]["sampling"])
    if kind == "document":
        report_result["signals"] = copy.deepcopy(result["signals"])
    frozen = copy.deepcopy(document)
    frozen["analysis_id"] = snapshot.analysis.id
    frozen["revision"] = snapshot.analysis.revision
    frozen["original_result_hash"] = canonical_hash(report_result)
    frozen["sha256"] = service.packet_hash(frozen)
    snapshot = snapshot.model_copy(update={"result": report_result, "evidence": frozen,
        "media": snapshot.media.model_copy(update={"sha256": analysis.sha256})})
    return snapshot, store, db, analysis, result, document


@pytest.mark.parametrize("kind", ["image", "video", "audio", "document"])
def test_historical_snapshot_without_evidence_retains_exact_bytes(kind):
    old = specimen(kind, "standard")
    before = canonical_json(old.model_dump(mode="python", exclude={"evidence"}))
    assert b'"evidence":' not in before
    assert dump(load(before)) == before
    assert dump(old.model_copy(update={"evidence": None})) == before


def test_capture_is_readonly_and_render_never_queries_live_analysis(evidence_case, monkeypatch):
    snapshot, store, db, analysis, result, document = evidence_case
    frozen = multimedia_evidence.capture(db, analysis, result=result, store=store)
    assert frozen == document and frozen is not document
    document["packets"][0]["observation"] = "later mutable observation"
    assert frozen["packets"][0]["observation"] != document["packets"][0]["observation"]
    monkeypatch.setattr(service, "load_packet_set", lambda *a, **k: pytest.fail("render queried mutable state"))
    loaded = load(dump(snapshot))
    before = snapshot_hash(loaded)
    viewed = multimedia_evidence.view(loaded, store=store)
    assert viewed["sha256"] == snapshot.evidence["sha256"]
    assert snapshot_hash(loaded) == before


def test_manifest_commits_every_json_asset_including_overview_and_wav(evidence_case):
    snapshot, *_ = evidence_case
    manifest = public_manifest(snapshot, "test-engine", engine.TEMPLATE_SET)["evidence"]
    doc = snapshot.evidence
    assert manifest["id"] == doc["id"] and manifest["sha256"] == doc["sha256"]
    assert manifest["source_sha256"] == doc["source"]["sha256"]
    assert manifest["original_result_hash"] == canonical_hash(snapshot.result)
    expected = [{"packet_id": None, "role": key, "sha256": a["sha256"], "media_type": a["media_type"]}
                for key, a in sorted(doc.get("overview", {}).items())]
    expected += [{"packet_id": p["id"], "role": key, "sha256": a["sha256"], "media_type": a["media_type"]}
                 for p in doc["packets"] for key, a in sorted(p["artifacts"].items())]
    if doc.get("acoustic_context"):
        expected += [{"packet_id": None, "role": f"acoustic_{key}", "sha256": a["sha256"], "media_type": a["media_type"]}
                     for key, a in sorted(doc["acoustic_context"]["overview"].items())]
        expected += [{"packet_id": p["id"], "role": key, "sha256": a["sha256"], "media_type": a["media_type"]}
                     for p in doc["acoustic_context"]["events"] for key, a in sorted(p["artifacts"].items())]
    assert manifest["artifacts"] == expected
    assert all("url" not in asset and "text" not in asset for asset in manifest["artifacts"])


def test_every_plate_caption_carries_its_json_packet_and_source_location(evidence_case):
    snapshot, store, *_ = evidence_case
    view = multimedia_evidence.view(snapshot, store=store)
    for row in view["packets"]:
        assert all(image["label"].startswith(f"{row['id']} - {row['locator']} - ") for image in row["images"])
        for frame in row["frames"]:
            anchor = frame["anchor"]
            prefix = (f"{frame['id']} - source {anchor['stream_time_s']:.6f} s; "
                      f"frame {anchor['frame_index']}, PTS {anchor['pts']} at {anchor['time_base']} - ")
            assert all(image["label"].startswith(prefix) for image in frame["images"])
    for image in view["overview"]:
        assert "overview - " in image["label"]


@pytest.mark.parametrize("profile,redacted", [("summary", False), ("standard", False), ("forensic", False),
                                               ("newsroom", False), ("standard", True)])
def test_every_profile_renders_frozen_observations_and_withholds_source(evidence_case, monkeypatch, profile, redacted):
    snapshot, store, *_ = evidence_case
    snapshot = snapshot.model_copy(update={"report": snapshot.report.model_copy(update={"profile": profile, "redacted": redacted})})
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    html = engine.render_html(snapshot, embed_fonts=False)
    assert snapshot.evidence["sha256"] in html
    assert snapshot.evidence["packets"][0]["observation"] in html
    assert "raw_output_sha256" not in html and "source_map_sha256" not in html
    assert "aud.gustking_xlsr" not in html
    assert "Evidence locator" in html and "/packets/0" in html
    withheld = profile == "newsroom" or redacted
    if withheld:
        assert "Source media, page images and verbatim excerpts are withheld" in html
        assert '<img src="data:image/png;base64,' not in html
    else:
        assert '<img src="data:image/png;base64,' in html
    if snapshot.analysis.kind == "document":
        excerpt = snapshot.evidence["packets"][0]["focus"]["text"]
        assert (excerpt in html) is not withheld
        assert "Actual scored input" in html
    elif snapshot.analysis.kind == "video":
        assert "not continuous playback" in html or withheld
        assert "Decoded RGB SHA-256" in html and "not a pixel-edit mask" in html
    else:
        assert "source timestamps" in html.lower() and "PDF is static" in html


@pytest.mark.parametrize("change", ["packet", "source", "modality", "result_binding"])
def test_untrusted_frozen_changes_fail_closed(evidence_case, change):
    snapshot, store, *_ = evidence_case
    evidence = copy.deepcopy(snapshot.evidence)
    if change == "packet":
        evidence["packets"][0]["observation"] = "altered without a new set hash"
    elif change == "source":
        snapshot = snapshot.model_copy(update={"media": snapshot.media.model_copy(update={"sha256": "0" * 64})})
    elif change == "modality":
        snapshot = snapshot.model_copy(update={"analysis": snapshot.analysis.model_copy(update={"kind": "image"})})
    else:
        result = copy.deepcopy(snapshot.result)
        result["manifest"]["sampling"] = {}
        snapshot = snapshot.model_copy(update={"result": result})
        evidence["original_result_hash"] = canonical_hash(result)
        evidence["sha256"] = service.packet_hash(evidence)
    with pytest.raises(ObjectIntegrityError):
        multimedia_evidence.view(snapshot.model_copy(update={"evidence": evidence}), store=store)


def test_missing_corrupt_or_relabelled_assets_are_not_substituted(evidence_case):
    snapshot, store, *_ = evidence_case
    evidence = copy.deepcopy(snapshot.evidence)
    asset = next(a for _, _, a in multimedia_evidence.asset_items(evidence) if a["media_type"] == "image/png")
    asset["width"] += 1
    evidence["sha256"] = service.packet_hash(evidence)
    with pytest.raises(ObjectIntegrityError):
        multimedia_evidence.view(snapshot.model_copy(update={"evidence": evidence}), store=store)
    path = store.path_for(asset["sha256"])
    path.unlink()
    with pytest.raises(ObjectNotFound):
        multimedia_evidence.view(snapshot, store=store)
    path.write_bytes(b"corrupt bytes")
    with pytest.raises(ObjectIntegrityError):
        multimedia_evidence.view(snapshot, store=store)


def test_video_source_frame_pixels_are_verified_beyond_asset_hash(video_fixture):
    db, store, analysis, result, *_ = video_fixture
    evidence = video_packets.build_packet_set(db, analysis, store, result)
    frame = evidence["packets"][0]["frames"][0]
    asset = evidence["packets"][0]["artifacts"][frame["artifact_refs"]["original"]]
    output = io.BytesIO()
    Image.new("RGB", (asset["width"], asset["height"]), "purple").save(output, "PNG")
    asset["sha256"] = store.put_bytes(output.getvalue()).sha256
    evidence["sha256"] = service.packet_hash(evidence)
    with pytest.raises(ObjectIntegrityError, match="pixels"):
        multimedia_evidence.verify_assets(evidence, store)


def test_audio_clip_samples_are_verified_beyond_asset_hash(audio_fixture):
    db, store, analysis, result, *_ = audio_fixture
    evidence = audio_packets.build_packet_set(db, analysis, store, result)
    clip = evidence["packets"][0]["artifacts"]["selected_clip"]
    import numpy as np

    from app.detectors.audio.decode import wav_bytes

    clip["sha256"] = store.put_bytes(wav_bytes(np.zeros(clip["samples"], np.float32))).sha256
    evidence["sha256"] = service.packet_hash(evidence)
    with pytest.raises(ObjectIntegrityError, match="sample interval"):
        multimedia_evidence.verify_assets(evidence, store)


def test_source_excerpt_does_not_undergo_house_style_or_html_execution(packet_factory, monkeypatch):
    db, store, analysis, result, decoded, _ = packet_factory()
    # Rebind a new source extraction, with a quoted percentage/model name/HTML
    # string that must remain source text rather than derived presentation prose.
    text = 'Source Mesorch claims 83% <script>alert("example")</script>.'
    from app.detectors.text.extract import text_hash

    decoded["text"] = text
    decoded["source_map"] = [{"kind": "text", "start": 0, "end": len(text)}]
    binding = result["manifest"]["sampling"]["document_source"]
    binding.update(decode_sha256=store.put_bytes(canonical_json(decoded)).sha256,
                   extracted_text_sha256=text_hash(text), source_map_sha256=canonical_hash(decoded["source_map"]))
    # Build a minimal valid frozen text packet directly to isolate report escaping.
    doc = {"schema": document_packets.SCHEMA, "status": "ready", "id": "quoted-source-test", "version": "document-evidence-2",
           "analysis_id": analysis.id, "revision": analysis.revision, "original_result_hash": canonical_hash(result),
           "source": {"sha256": analysis.sha256, "chars": len(text), "offset_unit": "unicode_codepoint",
                      **{k: binding[k] for k in ("decode_sha256", "extracted_text_sha256", "source_map_sha256")}},
           "coverage": {}, "packets": [{"id": "T01", "title": "Text context", "evidence_type": "statistical", "observation": "Recorded response.",
            "interpretation": "Context matters.", "finding_ids": [], "signal_refs": [], "measurements": [], "limitations": [], "artifacts": {},
            "focus": {"start": 0, "end": len(text), "text": text}, "context": {"start": 0, "end": len(text), "text": text},
            "anchors": [{"kind": "text", "start": 0, "end": len(text), "selected_start": 0, "selected_end": len(text)}],
            "supporting_windows": [{"text": text, "input_sha256": text_hash(text), "spans": [{"start": 0, "end": len(text)}],
                                   "score": .9, "signal_ref": {"display_id": "S01"}}]}], "limitations": []}
    snapshot = specimen("document", "summary")
    merged = copy.deepcopy(snapshot.result)
    merged["input"]["sha256"] = analysis.sha256
    merged["manifest"]["sampling"] = result["manifest"]["sampling"]
    doc.update(analysis_id=snapshot.analysis.id, revision=snapshot.analysis.revision, original_result_hash=canonical_hash(merged))
    doc["sha256"] = service.packet_hash(doc)
    snapshot = snapshot.model_copy(update={"result": merged, "evidence": doc,
                                           "media": snapshot.media.model_copy(update={"sha256": analysis.sha256})})
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    html = engine.render_html(snapshot, embed_fonts=False)
    assert "Source Mesorch claims 83%" in html
    assert "&lt;script&gt;" in html and '<script>alert("example")</script>' not in html


def test_missing_optional_evidence_is_explicit_and_readonly(monkeypatch):
    monkeypatch.setattr(service, "load_packet_set", lambda *a, **k: pytest.fail("read created or queried evidence"))
    assert multimedia_evidence.view(specimen("video", "summary"))["available"] is False
    assert multimedia_evidence.view(specimen("image", "summary")) is None
    assert multimedia_evidence.capture(None, SimpleNamespace(kind="image"), result={}) is None
