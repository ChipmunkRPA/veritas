from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.evidence import image_reference


def _scene():
    rng = np.random.default_rng(741)
    structure = cv2.resize(rng.integers(65, 175, (18, 26, 3), dtype=np.uint8), (384, 256), interpolation=cv2.INTER_CUBIC)
    texture = rng.integers(-12, 13, structure.shape, dtype=np.int16)
    return np.clip(structure.astype(np.int16) + texture, 0, 255).astype(np.uint8)


def test_actual_cli_writes_exact_reference_and_submitted_crops(tmp_path):
    reference, submitted = tmp_path / "reference.png", tmp_path / "submitted.jpg"
    source = _scene()
    changed = source.copy()
    changed[70:142, 142:222] = (210, 52, 119)
    Image.fromarray(source).save(reference)
    Image.fromarray(changed).save(submitted, quality=75, subsampling=2)
    evidence = tmp_path / "evidence"
    command = [sys.executable, "-m", "app.cli", "compare-image", str(reference), str(submitted), "--evidence-dir", str(evidence)]
    call = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert call.returncode == 0, call.stderr
    result = json.loads(call.stdout)
    assert result == json.loads((evidence / "comparison.json").read_text())
    assert result["status"] == "comparable" and result["jpeg_delivery_modeled"]
    assert result["reference"]["sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert result["submitted"]["sha256"] == hashlib.sha256(submitted.read_bytes()).hexdigest()
    with Image.open(evidence / result["mask"]["file"]) as mask:
        assert mask.size == (384, 256) and np.asarray(mask)[80:130, 155:210].mean() > 250
    region, = result["regions"]
    x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
    for role, path in (("reference_crop", reference), ("submitted_crop", submitted)):
        artifact = region[role]
        crop_path = evidence / artifact["file"]
        assert artifact["sha256"] == hashlib.sha256(crop_path.read_bytes()).hexdigest()
        with Image.open(path) as full, Image.open(crop_path) as crop:
            expected = ImageOps.exif_transpose(full).convert("RGB").crop((x, y, x + w, y + h))
            assert crop.tobytes() == expected.tobytes()
    repeated = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=20)
    assert repeated.returncode == 2 and "already exists" in repeated.stderr
    assert result == json.loads((evidence / "comparison.json").read_text())


def test_original_jpeg_orientation_matches_and_shift_is_unassessed(tmp_path):
    reference, submitted, shifted = (tmp_path / name for name in ("reference.png", "submitted.jpg", "shifted.png"))
    source = _scene()
    Image.fromarray(source).save(reference)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(source).transpose(Image.Transpose.ROTATE_90).save(submitted, quality=75, subsampling=2, exif=exif)
    result = image_reference.inspect(reference, submitted)
    assert result["status"] == "comparable" and result["regions"] == []
    assert result["jpeg_delivery_modeled"] is True
    Image.fromarray(np.roll(source, 3, axis=1)).save(shifted)
    assert image_reference.inspect(reference, shifted)["status"] == "unaligned"
