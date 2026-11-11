import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.detectors.image import sharp_boundary
from app.detectors.image.adapters.forensic import SharpBoundaryDetector
from app.engine.detector import Modality, Profile, StageOutputs
from app.engine.exhibits import ExhibitWriter
from app.engine.manifest import load_manifest
from app.engine.registry import DetectorRegistry, parse_policy
from app.storage.objects import ObjectStore


@pytest.fixture
def images():
    row, col = np.indices((512, 768))
    original = np.stack((33 + col // 32, 45 + row // 24, 68 + (row + col) // 80), axis=-1).astype(np.uint8)
    pasted = original.copy()
    pasted[192:320, 240:432] = [182, 155, 137]
    return original, pasted


def run(pixels, writer, **override):
    height, width = pixels.shape[:2]
    probe = {"format": "PNG", "mode": "RGB", "width": width, "height": height, "n_frames": 1} | override
    context = SimpleNamespace(
        outputs=StageOutputs(probe=probe),
        exhibits=writer,
        media=SimpleNamespace(native_pixels=lambda: pixels),
    )
    return SharpBoundaryDetector().run(context)


def test_registry_enables_only_an_explicit_model_free_commercial_check(monkeypatch):
    detector_id = sharp_boundary.DETECTOR_ID
    manifest = load_manifest()
    registry = DetectorRegistry(manifest, parse_policy("commercial"))
    monkeypatch.setattr(settings, "image_sharp_boundary_enabled", False)
    assert not registry.decision(detector_id).enabled
    monkeypatch.setattr(settings, "image_sharp_boundary_enabled", True)
    assert detector_id in registry.enabled(Modality.IMAGE, Profile.QUICK)
    assert isinstance(registry.instance(detector_id), SharpBoundaryDetector)
    card = manifest.detector(detector_id)
    assert not card.models and card.legacy_weight == 0
    denied = DetectorRegistry(
        manifest, parse_policy({"preset": "commercial", "deny_detectors": [detector_id]})
    )
    assert not denied.decision(detector_id).enabled


def test_actual_exhibits_record_only_native_sampled_sides(tmp_path, images):
    original, pasted = images
    store = ObjectStore(tmp_path / "objects")
    writer = ExhibitWriter("sharp-boundary", store, created_by=sharp_boundary.DETECTOR_ID)
    no_hit = run(original, writer)
    assert no_hit.status == "ok" and no_hit.aggregate is None and not writer.exhibits

    observed = run(pasted, writer)
    assert observed.status == "ok" and observed.aggregate.raw_kind == "score"
    assert observed.aggregate.raw > sharp_boundary.REVIEW_POINT
    details = observed.details["sharp_boundary"]
    assert details["schema"] == "sharp-boundary/1" and set(details["sides"]) == {
        "left",
        "top",
        "right",
        "bottom",
    }
    assert all(side == pytest.approx(3.0, abs=1e-5) for side in details["sides"].values())
    heat, region = writer.exhibits
    assert heat.semantics == "residual" and not heat.calibrated
    assert heat.value_range == (0.0, 3.0)
    assert heat.coordinate_space.width == 768 and heat.coordinate_space.height == 512
    [region] = region.regions
    assert region.region.bbox.model_dump() == {"x": 240.0, "y": 192.0, "w": 192.0, "h": 128.0}
    assert region.probability is None
    artefact = next(a for a in writer.artefacts if a.id == heat.payload_artefact_id)
    with Image.open(io.BytesIO(store.read_bytes(artefact.sha256))) as pixels:
        map_pixels = np.asarray(pixels)
    expected = np.zeros(original.shape[:2], dtype=np.uint8)
    expected[192:320, (240, 432)] = 255
    expected[(192, 320), 240:432] = 255
    np.testing.assert_array_equal(map_pixels, expected)


@pytest.mark.parametrize(
    "probe",
    [
        {"format": "JPEG"},
        {"mode": "RGBA"},
        {"width": 769},
        {"height": 511},
        {"n_frames": 2},
        {"width": None},
    ],
)
def test_unsupported_inputs_emit_no_score_or_map(images, probe):
    result = run(images[1], None, **probe)
    assert result.status == "not_applicable" and result.aggregate is None and not result.exhibit_ids
