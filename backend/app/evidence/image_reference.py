"""Local reference comparison and source-coordinate evidence for two images."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from app.detectors.image import reference_alignment, reference_compare, reference_resize

MAX_INPUT_BYTES = 60 * 1024 * 1024
RESIZE_SCHEMA = "chipmunk.image-reference-comparison/3"
DIFFERENCE_MARK = (255, 173, 46, 144)


def _bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError("An image exceeds the 60 MiB input limit")
    return data


def _png(path: Path, pixels: np.ndarray) -> dict:
    Image.fromarray(pixels).save(path, format="PNG")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"file": path.name, "sha256": digest, "width": pixels.shape[1], "height": pixels.shape[0]}


def _write_evidence(directory, comparison, reference, submitted, mask):
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f"Evidence directory already exists: {directory}")
    with tempfile.TemporaryDirectory(prefix=".image-comparison-", dir=directory.parent) as workspace:
        staged = Path(workspace) / "evidence"
        staged.mkdir()
        if mask is not None and mask.any():
            comparison["mask"] = _png(staged / "difference-mask.png", mask.astype(np.uint8) * 255)
            for i, region in enumerate(comparison["regions"], 1):
                for label, pixels, bounds in (("reference", reference, region.get("reference_region", region)),
                                              ("submitted", submitted, region)):
                    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
                    region[f"{label}_crop"] = _png(staged / f"{label}-{i:02d}.png", pixels[y:y+h, x:x+w])
                x, y, w, h = (region[key] for key in ("x", "y", "w", "h"))
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                overlay[mask[y:y+h, x:x+w]] = DIFFERENCE_MARK
                region["submitted_overlay"] = _png(staged / f"submitted-overlay-{i:02d}.png", overlay)
        (staged / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Evidence directory already exists: {directory}")
        staged.rename(directory)


def inspect(reference_path: Path, submitted_path: Path, *, evidence_dir: Path | None = None,
            allow_integer_alignment: bool = False, allow_proportional_resize: bool = False) -> dict:
    """Use an earlier reference supplied by the reviewer. This measures pixel differences."""
    before, after = _bytes(reference_path), _bytes(submitted_path)
    with Image.open(io.BytesIO(before)) as original, Image.open(io.BytesIO(after)) as questioned:
        if max(image.width * image.height for image in (original, questioned)) > reference_compare.MAX_PIXELS:
            raise ValueError("Reference comparison supports images up to 16 megapixels")
        with ImageOps.exif_transpose(original) as oriented_before, ImageOps.exif_transpose(questioned) as oriented_after:
            reference = np.array(oriented_before.convert("RGB"))
            submitted = np.array(oriented_after.convert("RGB"))
        resized = False
        if (allow_integer_alignment or allow_proportional_resize) and not np.array_equal(reference, submitted):
            result, mask = reference_alignment.compare(reference, submitted, questioned)
            if (allow_proportional_resize and reference.shape != submitted.shape
                    and result["status"] == "integer_alignment_unresolved"):
                result, mask = reference_resize.compare(reference, submitted, questioned)
                resized = True
        else:
            result, mask = reference_compare.compare_across_deliveries(reference, submitted, questioned)
    aligned = "alignment" in result
    output = {"schema": RESIZE_SCHEMA if resized else f"chipmunk.image-reference-comparison/{2 if aligned else 1}",
              "coordinate_space": "exif-oriented-submitted" if aligned or resized else "exif-oriented-original",
              "reference": {"sha256": hashlib.sha256(before).hexdigest(), "width": reference.shape[1], "height": reference.shape[0]},
              "submitted": {"sha256": hashlib.sha256(after).hexdigest(), "width": submitted.shape[1], "height": submitted.shape[0]},
              **result, "interpretation": "Regions show differences that survived the comparison settings. Ordinary processing can produce differences. The earlier file's authenticity and the cause of changes require separate evidence."}
    if evidence_dir is not None:
        _write_evidence(evidence_dir, output, reference, submitted, mask)
    return output
