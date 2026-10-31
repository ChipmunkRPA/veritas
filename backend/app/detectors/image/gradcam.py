"""Grad-CAM for HuggingFace ViT classifiers: which image regions drove the 'fake' decision.

We hook the last encoder block, reshape the patch tokens into a grid, and weight them by the gradient of the
positive-class logit.
- `vit_gradcam_map` returns the normalised patch-grid map. v2 adapters store it as an attribution heatmap
  ("where a classifier looked", never localisation).
- `vit_gradcam` renders the v1 overlay artifact.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

from app.detectors.image.classifiers import HFImageClassifier


def vit_gradcam_map(clf: HFImageClassifier, rgb: np.ndarray) -> np.ndarray | None:
    """Patch-grid Grad-CAM in [0, 1] (side × side), or None when the model is not a ViT or the map is empty."""
    model = clf.model
    if not hasattr(model, "vit"):
        return None
    # The classifier reads only the CLS token of the last block's output, so patch tokens there get zero gradient (v0.3
    # hooked that output and never produced a map). Hook the last block's pre-attention norm instead: its patch tokens
    # reach CLS through the final attention. transformers 5 renamed `vit.encoder.layer` to `vit.layers`; under 5.x the old
    # attribute raised inside v1's classifier try-block and dropped the face classifier's signal whenever p ≥ 0.5.
    blocks = model.vit.layers if hasattr(model.vit, "layers") else model.vit.encoder.layer
    layer = blocks[-1].layernorm_before
    acts: dict = {}
    grads: dict = {}

    def fwd_hook(_m, _i, out):
        o = out[0] if isinstance(out, tuple) else out
        acts["a"] = o
        o.register_hook(lambda g: grads.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        inputs = clf.processor(images=[Image.fromarray(rgb)], return_tensors="pt").to(clf.device)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(**inputs).logits
            logits[0, clf.pos_idx].backward()
        a = acts["a"][0, 1:, :]  # drop CLS
        g = grads["g"][0, 1:, :]
        n = a.shape[0]
        side = int(round(n**0.5))
        if side * side != n:
            return None
        w = g.mean(dim=0, keepdim=True)
        cam = torch.relu((a * w).sum(dim=-1)).reshape(side, side).detach().float().cpu().numpy()
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)  # the leased model is shared: do not keep gradient buffers alive
    if cam.max() <= 0:
        return None
    return cam / cam.max()


def vit_gradcam(clf: HFImageClassifier, rgb: np.ndarray) -> np.ndarray | None:
    cam = vit_gradcam_map(clf, rgb)
    if cam is None:
        return None
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = (0.55 * rgb + 0.45 * heat).astype(np.uint8)
    return overlay
