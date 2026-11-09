"""Editable or incompletely validated records must not decide origin in fallback paths."""
import pytest
from PIL import Image, PngImagePlugin

from app.core.evidence import Signal
from app.detectors.common.fusion import fuse
from app.detectors.image import metadata


def read_signals(tmp_path, monkeypatch, *, text=None, exif=None, credential=None):
    path = tmp_path / "ordinary.png"
    info = PngImagePlugin.PngInfo()
    for key, value in (text or {}).items():
        info.add_text(key, value)
    Image.new("RGB", (16, 16), (60, 90, 120)).save(path, pnginfo=info)
    monkeypatch.setattr(metadata, "read_c2pa", lambda _: credential or {"present": False})
    if exif is not None:
        monkeypatch.setattr(metadata, "read_exif", lambda _: exif)
    with Image.open(path) as image:
        return metadata.metadata_signals(path, image)[0]


@pytest.mark.parametrize("text,expected", [
    ({"Software": "OpenAI"}, "generator_metadata"),
    ({"parameters": "Steps: 20, Sampler: Euler, CFG: 7, Seed: 42"}, "generator_metadata"),
    ({"Comment": "Compare this photograph with an AI generated version"}, "generator_mention_freetext"),
])
def test_real_png_with_editable_ai_fields_has_no_authorship_probability(tmp_path, monkeypatch, text, expected):
    signals = read_signals(tmp_path, monkeypatch, text=text)
    assert expected in {s.id for s in signals}
    assert all(s.ai_probability is None and s.weight == 0 and not s.decisive for s in signals)
    verdict = fuse(signals)
    assert verdict.band == "uncertain" and verdict.confidence == "low"
    assert verdict.signals_total == 0


@pytest.mark.parametrize("state", ["trusted", "valid", "invalid", "unknown"])
@pytest.mark.parametrize("declares_ai", [True, False])
def test_parser_state_is_not_a_substitute_for_full_trust_binding_and_ai_gate(tmp_path, monkeypatch, state, declares_ai):
    credential = {"present": True, "validation_state": state, "claim_generator": "Example signing tool",
                  "digital_source_types": ["trainedAlgorithmicMedia"] if declares_ai else ["digitalCapture"],
                  "actions": ["c2pa.created"], "raw": {}}
    signals = read_signals(tmp_path, monkeypatch, credential=credential)
    record = next(s for s in signals if s.category == "provenance")
    assert record.ai_probability is None and record.weight == 0 and not record.decisive
    assert fuse(signals).band == "uncertain"


def test_tool_name_without_ai_action_is_described_as_metadata_not_a_verified_declaration(tmp_path, monkeypatch):
    signals = read_signals(tmp_path, monkeypatch, credential={"present": True, "validation_state": "trusted",
        "claim_generator": "OpenAI", "digital_source_types": [], "actions": [], "raw": {}})
    record = next(s for s in signals if s.category == "provenance")
    assert "metadata" in record.name
    assert "Tool names alone" in record.reasoning
    assert not record.decisive and record.ai_probability is None


def test_camera_fields_copied_to_a_file_do_not_prove_originality(tmp_path, monkeypatch):
    signals = read_signals(tmp_path, monkeypatch, exif={"Make": "Example", "Model": "Camera",
        "ExposureTime": "1/100", "ISOSpeedRatings": "100"})
    camera = next(s for s in signals if s.id == "camera_exif")
    assert camera.ai_probability is None and camera.weight == 0
    assert "do not establish original capture" in camera.reasoning
    assert fuse(signals).band == "uncertain"


def signal(category, p, *, decisive=False):
    return Signal(id="example", name="Example check", category=category, observed=str(p), reference="Test fixture",
        ai_probability=p, weight=100 if category in {"metadata", "provenance"} else 1,
        direction=Signal.direction_from_prob(p), strength="strong", decisive=decisive,
        reasoning="An independently supplied signal can contain an unsupported decisive flag.")


@pytest.mark.parametrize("category", ["metadata", "provenance"])
@pytest.mark.parametrize("p", [0.01, 0.99])
def test_fuser_defends_against_old_or_forged_decisive_records(category, p):
    content = signal("model", 0.2)
    baseline = fuse([content])
    verdict = fuse([content, signal(category, p, decisive=True)])
    assert verdict.probability_ai == baseline.probability_ai
    assert verdict.signals_total == baseline.signals_total == 1
    assert verdict.confidence == baseline.confidence
    assert not any("Decisive:" in reason for reason in verdict.reasoning)
    assert any("advisory" in caveat for caveat in verdict.caveats)


def test_statistical_decisive_flag_cannot_invent_a_provenance_override():
    verdict = fuse([signal("model", 0.8, decisive=True)])
    assert verdict.probability_ai == pytest.approx(0.8)
    assert verdict.confidence == "low"
    assert "uncalibrated screening score" in verdict.summary
    assert any("not a calibrated probability" in caveat for caveat in verdict.caveats)
    assert not any("Decisive:" in reason for reason in verdict.reasoning)
