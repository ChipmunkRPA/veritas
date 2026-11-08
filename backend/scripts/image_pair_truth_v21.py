"""Exact source-relative AI-splice truth and non-generative retouch controls."""

import hashlib

import numpy as np
from PIL import Image, ImageFilter


def require(ok, message):
    if not ok:
        raise ValueError(message)


def qualify(original, edited, alpha):
    require(original.dtype == edited.dtype == np.uint8 and original.shape == edited.shape and original.ndim == 3 and original.shape[2] == 3, "Matching native uint8 RGB source/edit required")
    require(alpha.dtype == np.uint8 and alpha.shape == original.shape[:2], "Matching native uint8 compositing mask required")
    changed = np.any(original != edited, axis=2)
    support = alpha > 0
    require(bool(support.any()) and bool(changed.any()), "Nonempty intended and observed edit required")
    require(not bool((changed & ~support).any()), "Changed pixels outside declared compositing support")
    # Both controls use the same source and published selection, without a generator.
    tone = np.clip(np.rint((original.astype(np.float64) - 127.5) * 1.12 + 127.5 + np.asarray([3, -2, 1])), 0, 255).astype(np.uint8)
    image, mask = Image.fromarray(original), Image.fromarray(alpha)
    controls = {
        "tone": np.asarray(Image.composite(Image.fromarray(tone), image, mask)),
        "blur": np.asarray(Image.composite(image.filter(ImageFilter.GaussianBlur(1.25)), image, mask)),
    }
    proof = {
        "changed_pixels": int(changed.sum()), "support_pixels": int(support.sum()), "outside_support_changes": 0,
        "truth_mask_sha256": hashlib.sha256(changed.tobytes()).hexdigest(),
        "ordinary_controls": {kind: {"rgb_sha256": hashlib.sha256(value.tobytes()).hexdigest(), "changed_pixels": int(np.any(value != original, axis=2).sum()), "ai_generation_used": False} for kind, value in controls.items()},
        "scope": "Observed changed pixels in the publisher-labeled SD2 splice; not independent pixel-authorship or raw-camera authentication.",
    }
    return changed, proof
