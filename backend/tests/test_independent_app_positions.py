from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app.bench.independent_app_positions import _positions, packet_positions

FIXTURE = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/signed-evidence-example"


def test_reads_signed_public_image_positions_with_a_separate_key_and_original():
    report = packet_positions(FIXTURE / "evidence.zip", FIXTURE / "authored-gradient.png", FIXTURE / "example-public.pem")
    assert report["kind"] == "image"
    assert report["coordinates"] == {"basis": "display-oriented bounding boxes", "unit": "pixels", "extent": [160, 120]}
    assert report["packets"] == [{"id": "E01", "boxes": [[40, 30, 80, 70]]}]


def test_original_and_separately_pinned_key_are_required(tmp_path):
    changed = tmp_path / "changed.png"
    data = (FIXTURE / "authored-gradient.png").read_bytes()
    changed.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="original differs"):
        packet_positions(FIXTURE / "evidence.zip", changed, FIXTURE / "example-public.pem")
    key = tmp_path / "unrelated.pem"
    key.write_bytes(Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    with pytest.raises(ValueError, match="did not verify"):
        packet_positions(FIXTURE / "evidence.zip", FIXTURE / "authored-gradient.png", key)


def test_witness_frames_do_not_become_a_contiguous_video_claim():
    source = {"frame_count": 125, "time_base": "1/30000", "frame_table_sha256": "1" * 64}
    witnesses = [{"anchor": {"frame_index": index}} for index in (1, 14, 5, 5)]
    evidence = {"source": source, "packets": [
        {"id": "sampled", "frames": witnesses, "clip": {"start_frame_index": 0, "end_frame_index_exclusive": 125}},
        {"id": "range", "frames": witnesses, "face_return": {"start_frame_index": 5, "end_frame_index": 14}},
        {"id": "transition", "face_single_boundary": {"boundary": 44}},
    ]}
    coordinates, rows = _positions("video", evidence)
    assert coordinates["extent"] == [125]
    assert rows == [{"id": "sampled", "witness_frames": [1, 5, 14]},
                    {"id": "range", "witness_frames": [1, 5, 14], "face_change_span": [5, 14]},
                    {"id": "transition", "face_transition_index": 44}]
    evidence["packets"][1]["face_return"]["end_frame_index"] = 126
    with pytest.raises(ValueError, match="outside its source extent"):
        _positions("video", evidence)


def test_selected_audio_windows_and_passage_spans_use_their_recorded_clocks():
    pcm_sha256, text_sha256 = "a" * 64, "b" * 64
    audio = {"source": {"samples": 96000, "sample_rate": 16000, "pcm_sha256": pcm_sha256}, "packets": [
        {"id": "A01", "interval": {"start_sample": 8000, "end_sample": 48000}, "anchor": {"pcm_sha256": pcm_sha256},
         "context_interval": {"start_sample": 0, "end_sample": 64000}},
        {"id": "other-clock", "interval": {"start_sample": 0, "end_sample": 5}, "anchor": {"pcm_sha256": "0" * 64}},
    ]}
    document = {"source": {"chars": 200, "offset_unit": "unicode_codepoint", "extracted_text_sha256": text_sha256},
                "packets": [{"id": "T01", "anchors": [
                    {"start": 0, "end": 50, "selected_start": 5, "selected_end": 14, "excerpt": "private"},
                    {"start": 50, "end": 120, "selected_start": 77, "selected_end": 120, "excerpt": "private"},
                ]}]}
    audio_coords, audio_rows = _positions("audio", audio)
    text_coords, text_rows = _positions("document", document)
    assert audio_coords["pcm_sha256"] == pcm_sha256 and audio_coords["sample_rate"] == 16000
    assert audio_rows == [{"id": "A01", "windows": [[8000, 48000]]}]
    assert text_coords["extracted_text_sha256"] == text_sha256
    assert text_rows == [{"id": "T01", "selected_spans": [[5, 14], [77, 120]]}]
