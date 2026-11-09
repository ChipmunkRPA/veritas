"""Serving safety, source-coordinate parity and preservation of upstream mathematics."""
import ast
import concurrent.futures
import hashlib
import inspect
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from app.detectors.image import regional
from scripts import relayformer_native_v18 as reference

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "backend/app/detectors/image/regional_graph"
UPSTREAM = ROOT / "models/research/image-v18-relayformer-v2/upstream/models"


@pytest.mark.parametrize("size", [(64, 65), (515, 389), (389, 515), (1401, 773), (3, 2001)])
def test_source_geometry_matches_frozen_reference(size):
    image = Image.fromarray(np.random.default_rng(81).integers(0, 256, (*size[::-1], 3), dtype=np.uint8))
    a, ga = regional.prepare(image)
    b, gb = reference.prepare(image)
    assert torch.equal(a, b) and ga == gb
    canvas = torch.arange(1024**2, dtype=torch.float32).reshape(1, 1, 1024, 1024) / (1024**2)
    assert np.array_equal(regional.source_map(canvas, ga), reference.source_map(canvas, gb))


def test_non_upright_and_invalid_responses_rejected():
    image = Image.new("RGB", (64, 80))
    image.getexif()[274] = 6
    with pytest.raises(ValueError, match="upright"):
        regional.prepare(image)
    with pytest.raises(ValueError, match="RGB"):
        regional.prepare(image.convert("L"))
    _, geometry = regional.prepare(Image.new("RGB", (64, 80)))
    for output in (torch.zeros(1, 1, 64, 80), torch.full((1, 1, 1024, 1024), float("nan")),
                   torch.ones(1, 1, 1024, 1024, dtype=torch.float64), torch.full((1, 1, 1024, 1024), 2.)):
        with pytest.raises(ValueError):
            regional.source_map(output, geometry)


def functions(path):
    tree = ast.parse(path.read_text())
    return {f"{cls.name}.{fn.name}": ast.dump(fn, include_attributes=False)
            for cls in tree.body if isinstance(cls, ast.ClassDef)
            for fn in cls.body if isinstance(fn, ast.FunctionDef)}


@pytest.mark.parametrize("name", ["RelayFormer.py", "GLoRA_vit.py"])
def test_vendored_math_unchanged(name):
    if not UPSTREAM.is_dir():
        pytest.skip("Optional pinned upstream checkout absent; public production code remains usable")
    original, serving = functions(UPSTREAM / name), functions(GRAPH / name)
    allowed = {"RelayFormer.__init__", "RelayFormer._load_vit_checkpoint"}
    assert set(original) == set(serving)
    for key in original.keys() - allowed:
        assert original[key] == serving[key], key
    assert "torch.load" not in (GRAPH / name).read_text()


def test_complete_digest_checked_before_deserialization(tmp_path, monkeypatch):
    from safetensors import torch as safe

    checkpoint = tmp_path / "changed.safetensors"
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(regional, "CHECKPOINT_BYTES", checkpoint.stat().st_size)
    monkeypatch.setattr(safe, "load_file", lambda *a, **k: pytest.fail("Parsed unverified weights"))
    with pytest.raises(ValueError, match="digest"):
        regional.load_model(checkpoint)
    with pytest.raises(ValueError, match="Complete pinned"):
        regional.load_model(tmp_path / "missing")


class AuthoredGraph(torch.nn.Module):
    """A geometry/concurrency fixture, not a replacement model or accuracy test."""
    def __init__(self, fail=False):
        super().__init__()
        self.vit = SimpleNamespace(patch_embed=torch.nn.Identity())
        self.fail, self.active, self.maximum_active = fail, 0, 0

    def forward(self, image, origin_shape, clip_len):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            fh, fw = origin_shape[0].tolist()
            patches = torch.stack([image[0, :, r * 496:r * 496 + 528, c * 496:c * 496 + 528]
                                   for r in range((2 * fh + 1023) // 1024)
                                   for c in range((2 * fw + 1023) // 1024)])
            self.vit.patch_embed(patches)
            time.sleep(.005)
            if self.fail:
                raise RuntimeError("Authored failure")
            assert clip_len.tolist() == [1]
            return torch.full((1, 1, 1024, 1024), .95)
        finally:
            self.active -= 1


def test_reentrant_no_process_global_patches_or_thread_changes():
    import timm

    before = (tuple(sys.path), socket.socket.connect, timm.create_model, torch.get_num_threads(),
              torch.are_deterministic_algorithms_enabled())
    graph = AuthoredGraph()
    model = regional.RegionalModel(graph)
    images = [Image.new("RGB", size, "gray") for size in ((65, 80), (90, 64))]
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(model.predict, images))
    assert graph.maximum_active == 1
    assert len(graph.vit.patch_embed._forward_pre_hooks) == 0
    for image, row in zip(images, rows, strict=True):
        assert row.raw_mask.shape == image.size[::-1]
        assert row.geometry["source_rgb_sha256"] == hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        assert row.candidate_eligibility["eligible"] is True
    assert before == (tuple(sys.path), socket.socket.connect, timm.create_model, torch.get_num_threads(),
                      torch.are_deterministic_algorithms_enabled())


def test_failed_inference_removes_hook():
    graph = AuthoredGraph(fail=True)
    with pytest.raises(RuntimeError, match="Authored failure"):
        regional.RegionalModel(graph).predict(Image.new("RGB", (64, 64)))
    assert len(graph.vit.patch_embed._forward_pre_hooks) == 0


@pytest.mark.parametrize("gate,threshold", [(True, .90), (False, .90), (True, .70)])
def test_retained_source_masks_match_frozen_evaluation(gate, threshold):
    from app.detectors.image.regional_support import readout
    from scripts.image_native_comparison_readout_v45 import source_mask

    pixels = np.random.default_rng(123).integers(10, 245, (273, 411, 3), dtype=np.uint8)
    raw = np.zeros((117, 151), np.float32)
    raw[10:50, 20:50] = .96
    raw[70:100, 100:135] = .98
    raw[70:100, 10:35] = .97  # only the two ranked regions are retained
    measured = readout(raw, gate, threshold, b"not-jpeg", pixels)
    expected = source_mask(raw, gate, threshold, b"not-jpeg", pixels, np.zeros(pixels.shape[:2], bool))
    assert np.array_equal(measured.source_mask(411, 273), expected)


def test_shared_pixels_do_not_refilter_small_intersections():
    from app.detectors.image.regional_support import readout, shared_pixels

    pixels = np.random.default_rng(44).integers(10, 245, (160, 200, 3), dtype=np.uint8)
    a, b = np.zeros((160, 200), np.float32), np.zeros((160, 200), np.float32)
    a[20:100, 20:70] = .99
    b[99:130, 69:140] = .99
    first, second = (readout(raw, True, .90, b"png", pixels) for raw in (a, b))
    shared = shared_pixels(first, second)
    assert shared.sum() == 1
    assert np.array_equal(shared, first.selected_grid_mask & second.selected_grid_mask)


def test_resource_observer_math_unchanged():
    from scripts import image_native_confirmation_runtime_v46 as prior
    from scripts import native_regional_serving_runtime_v47 as serving

    def body(fn):
        source = inspect.getsource(fn)
        source = source[source.index("    if output.exists():"):]
        source = source.replace("fixed-monitor-profile/46", "fixed-monitor-profile/47").replace(
            "One explicit confirmation phase", "One explicit serving parity phase")
        return ast.dump(ast.parse("def observer():\n" + source))

    assert body(prior.run) == body(serving.run)
