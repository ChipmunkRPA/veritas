from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.image import faces
from app.detectors.video import decode
from scripts import native_mouth_features_v123 as extraction


def face(corners=((40, 60), (60, 60))):
    return faces.Face(20, 10, 70, 80, 0.92, [(35, 25), (65, 25), (50, 40), *corners])


def test_native_mouth_crop_preserves_actual_pixels_and_clips_at_image_edges():
    rgb = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crop, bounds = extraction.mouth_crop(rgb, face())
    assert bounds == [32, 48, 68, 72]
    assert np.array_equal(crop, rgb[48:72, 32:68])
    crop, bounds = extraction.mouth_crop(rgb, face(((0, 1), (20, 1))))
    assert bounds == [0, 0, 28, 13]
    assert np.array_equal(crop, rgb[:13, :28])
    assert crop.flags.c_contiguous


@pytest.mark.parametrize("problem", ["missing", "nonfinite", "small", "outside"])
def test_insufficient_mouth_support_is_unassessed(problem):
    candidate = face()
    if problem == "missing":
        candidate.landmarks = []
    elif problem == "nonfinite":
        candidate.landmarks[3] = (float("nan"), 60)
    elif problem == "small":
        candidate.landmarks[4] = (41, 60)
    else:
        candidate.landmarks[3:] = [(300, 300), (320, 300)]
    with pytest.raises(extraction.UnsupportedMouth):
        extraction.mouth_crop(np.zeros((100, 100, 3), np.uint8), candidate)


@pytest.mark.parametrize("detected", [[], [face(), face()], [face(((1, 1), (2, 1)))]])
def test_unsupported_detection_never_calls_image_encoder(monkeypatch, detected):
    monkeypatch.setattr(faces, "detect_faces", lambda *a, **k: detected)

    class ForbiddenEncoder:
        def encode(self, crop):
            raise AssertionError("Unavailable mouth reached image encoder")

    metadata, value = extraction.detect(np.zeros((100, 100, 3), np.uint8), object(), ForbiddenEncoder())
    assert metadata["status"] == "unassessed" and value is None
    assert "score" not in metadata


def test_valid_crop_records_actual_input_feature_coordinates_without_an_ai_score(monkeypatch):
    monkeypatch.setattr(faces, "detect_faces", lambda *a, **k: [face()])
    vector = np.arange(768, dtype=np.float32)

    class Encoder:
        def encode(self, crop):
            assert crop.shape == (24, 36, 3)
            return vector, "a" * 64

    metadata, value = extraction.detect(np.zeros((100, 100, 3), np.uint8), object(), Encoder())
    assert metadata["region_bbox"] == [32, 48, 68, 72]
    assert metadata["face"]["box"] == [20, 10, 70, 80]
    assert metadata["face"]["score_hex"] == (0.92).hex()
    assert metadata["feature_float32_sha256"] == extraction.clips.inputs.speech.speech.fingerprint(vector)
    assert metadata["model_input_float32_sha256"] == "a" * 64
    assert np.array_equal(value, vector) and "ai_probability" not in metadata


def test_verified_frames_reject_changed_pixels_or_incomplete_decode(tmp_path, monkeypatch):
    clip = tmp_path / "test.mkv"
    clip.write_bytes(b"authored test fixture, not media")
    rgb = np.zeros((4, 4, 3), np.uint8)
    sample = SimpleNamespace(index=0, pts=0, rgb=rgb)
    case = {
        "status": "ready",
        "file": str(clip),
        "file_sha256": extraction.sha(clip),
        "expected": {"rgb_sha256": [extraction.clips.inputs.speech.speech.fingerprint(rgb)]},
    }
    monkeypatch.setattr(decode, "decode_frames", lambda *a, **k: iter([sample]))
    assert list(extraction.verified_frames(case, object(), [0])) == [sample]
    sample.rgb = np.ones_like(rgb)
    with pytest.raises(ValueError, match="RGB"):
        list(extraction.verified_frames(case, object(), [0]))
    monkeypatch.setattr(decode, "decode_frames", lambda *a, **k: iter([]))
    with pytest.raises(ValueError, match="Complete"):
        list(extraction.verified_frames(case, object(), [0]))
    clip.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Exact"):
        list(extraction.verified_frames(case, object(), [0]))


def test_out_of_scope_batch_is_rejected_before_model_access(monkeypatch):
    monkeypatch.setattr(extraction, "Encoder", lambda: pytest.fail("Model accessed before batch validation"))
    for batch in (-1, 8, True, 1.5):
        with pytest.raises(ValueError):
            extraction.extract(batch, None)


def test_other_stages_cannot_be_dispatched():
    with pytest.raises(ValueError):
        extraction.run("train")
    with pytest.raises(ValueError):
        extraction.routed_popen(["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind"])
